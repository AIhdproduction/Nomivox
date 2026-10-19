"""Post-process a retained vote cache without repeating segmentation."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np

from e57cls.files import atomic_json, new_output
from e57cls.geometry.fill import extract_planes
from e57cls.vote.geometry_rules import apply_geometry_rules
from e57cls.io.store import load_scan, reduced_store
from e57cls.output.e57 import write_e57
from e57cls.output.las import write_cloud
from e57cls.output.reports import legend_rows, write_legends, write_report
from e57cls.pipeline.config import DEFAULTS, resolve_config
from e57cls.pipeline.run import geometry_scan_context
from e57cls.slam.cameras import build_camera_list
from e57cls.slam.stations import cluster_stations
from e57cls.vote.engine import VoteStore


# Only parameters applied after observations and vote weights have been stored.
# Keep this deliberately central: later quality packages can append new
# post-processing settings here without opening the segmentation cache to edits.
RECLASSIFY_OVERRIDE_KEYS = frozenset({
    'min_observations', 'min_winning_share', 'stairs_priority', 'stairs_min_confidence', 'stairs_priority_min_share',
    'stairs_priority_over', 'stairs_geometry_check', 'stairs_plane_tolerance',
    'stairs_planar_fraction', 'stairs_min_rise', 'stairs_min_voxels', 'stairs_min_levels',
    'stairs_level_tolerance', 'stairs_step_rise_min', 'stairs_step_rise_max',
    'stairs_step_tread_min', 'stairs_step_tread_max', 'stairs_floor_ring_radius',
    'stairs_floor_level_tolerance', 'surface_tie_break', 'surface_tie_share',
    'surface_normal_radius', 'component_cleanup', 'component_min_voxels', 'component_min_cells',
    'cable_component_min_voxels', 'cable_component_min_span', 'electrical_component_min_voxels',
    'electrical_component_min_span', 'vegetation_base_to_ground', 'vegetation_ground_cell_size',
    'vegetation_ground_band', 'building_enclosure', 'building_boundary_cell_size',
    'building_boundary_min_height', 'building_boundary_max_cells', 'geometry_fill',
    'geometry_fill_classes', 'geometry_fill_min_area', 'geometry_fill_max_gap',
    'geometry_fill_tolerance', 'geometry_fill_max_planes', 'geometry_rules', 'geometry_cell_size',
    'geometry_tile_size', 'geometry_halo', 'geometry_knn', 'geometry_knn_radius',
    'geometry_rules_outdoor_vegetation', 'geometry_rules_ground_floor', 'terrain_cell_size',
    'terrain_max_gap', 'terrain_max_slope', 'terrain_ground_tolerance', 'terrain_fill',
    'terrain_cover_min_height', 'terrain_cover_max_height', 'tree_crown_radius', 'tree_crown_max_height',
    'tree_crown_grow_radius', 'class_transfer_radius', 'class_transfer_range_factor',
    'class_transfer_radius_max', 'output_density', 'format', 'extra_no_reflection',
    'cache_cleanup', 'vote_families',
})
RECLASSIFY_POST_VOTE_PREFIXES = ('stairs_', 'surface_tie_', 'component_', 'cable_', 'electrical_',
                                 'vegetation_', 'building_', 'geometry_fill', 'geometry_', 'terrain_', 'tree_',
                                 'class_transfer_')
INPUT_MODES = frozenset({'static', 'slam'})
REQUESTED_INPUT_MODES = INPUT_MODES | {'auto'}


def allowed_override_keys():
    """Include future post-vote tuning parameters without opening input stages."""
    return RECLASSIFY_OVERRIDE_KEYS | frozenset(key for key in DEFAULTS
        if key.startswith(RECLASSIFY_POST_VOTE_PREFIXES))


class _StoredMirrorStates:
    """Replay persisted per-record reflection decisions without SAM or cameras."""

    def __init__(self, directory: Path):
        self.paths = {int(path.stem.rsplit('_', 1)[-1]): path for path in Path(directory).glob('states_*.npy')}

    def output(self, store, start, stop, valid, world, codes):
        path = self.paths.get(store.meta['scan_index'])
        if path is None:
            return codes, np.zeros(len(codes), np.uint8)
        states = np.load(path, mmap_mode='r', allow_pickle=False)
        try:
            values = np.asarray(states[start:stop][valid], np.uint8)
        finally:
            states._mmap.close()
        codes[values == 2] = 68
        return codes, values


def _load_overrides(path: Path | None) -> dict:
    if path is None:
        return {}
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f'Override-Datei kann nicht gelesen werden: {error}') from error
    if not isinstance(value, dict):
        raise ValueError('Die Override-Datei muss ein JSON-Objekt enthalten.')
    forbidden = sorted(set(value)-allowed_override_keys())
    if forbidden:
        raise ValueError('Nicht erlaubte Nachklassifizierungsparameter: '+', '.join(forbidden))
    return value


def _load_json_object(path: Path, label: str) -> dict:
    """Load a required JSON object and report corrupt run artifacts clearly."""
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f'{label} kann nicht gelesen werden: {error}') from error
    if not isinstance(value, dict):
        raise ValueError(f'{label} muss ein JSON-Objekt enthalten.')
    return value


def _input_mode_from_report(report: dict, label: str) -> tuple[dict | None, str | None]:
    """Return the detected input mode from a run artifact, if it is usable."""
    input_mode = report.get('input_mode')
    if not isinstance(input_mode, dict):
        return None, f'{label} enthält keine gültige Aufnahmeart.'
    mode = input_mode.get('mode')
    if not isinstance(mode, str) or mode not in INPUT_MODES:
        return None, f'{label} enthält eine ungültige Aufnahmeart: {mode!r}.'
    return input_mode, None


def _resolved_input_mode(source_run: Path, manifest: dict) -> dict:
    """Read the mode actually used by the source run, including late rule d updates."""
    requested = manifest.get('input_mode')
    if not isinstance(requested, str) or requested not in REQUESTED_INPUT_MODES:
        raise ValueError(f'Die konfigurierte Aufnahmeart in run_config.json ist ungültig: {requested!r}.')

    issues = []
    for filename, label in (("report.json", "report.json"), ("inspect_report.json", "inspect_report.json")):
        try:
            report = _load_json_object(source_run/filename, label)
        except ValueError as error:
            issues.append(str(error))
            continue
        input_mode, issue = _input_mode_from_report(report, label)
        if input_mode is not None:
            return input_mode
        issues.append(issue)

    if requested != 'auto':
        return {'mode': requested, 'requested': requested}
    detail = ' '.join(issue for issue in issues if issue)
    raise ValueError(f'Die tatsächlich verwendete Aufnahmeart kann nicht ermittelt werden. {detail}')


def _copy_sqlite(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as old, sqlite3.connect(target) as new:
        old.backup(new)


def _postprocess(votes, config, by_id, scanner_heights, work: Path, active_codes, scan_context):
    """Apply precisely the decision/output stages from the regular pipeline."""
    codes = list(by_id.values())
    priority = {'code': by_id['stairs'], 'over': [by_id[key] for key in config['stairs_priority_over'] if key in by_id],
                'confidence': config['stairs_min_confidence'], 'min_share': config['stairs_priority_min_share']} if config['stairs_priority'] and 'stairs' in by_id else None
    families = [[by_id[name] for name in family if name in by_id] for family in config['vote_families']]
    votes.decide(config['min_observations'], config['min_winning_share'], codes, priority=priority, vote_families=families)
    result = {'surface_ties': None, 'building_enclosure': None, 'stairs_geometry': None,
              'component_cleanup': None, 'vegetation_base_to_ground': None, 'geometry_rules': None, 'geometry_fill': None}
    origin = np.asarray(config['voxel_origin'])
    if config['surface_tie_break']:
        result['surface_ties'] = votes.resolve_surface_ties(by_id, origin, config['voxel_size'], config['surface_tie_share'],
            config['surface_normal_radius'], config['min_observations'], scanner_heights)
    if config['building_enclosure'] and {'ground', 'floor'}.issubset(by_id):
        result['building_enclosure'] = votes.split_ground_floor_by_building(by_id['ground'], by_id['floor'],
            [by_id[key] for key in ('building', 'wall') if key in by_id], config['voxel_size'],
            config['building_boundary_cell_size'], config['building_boundary_min_height'], config['building_boundary_max_cells'])
    terrain = None
    if config['geometry_rules']:
        terrain = work/'terrain.npz'
        result['geometry_rules'] = apply_geometry_rules(votes, by_id, origin, config['voxel_size'], config, terrain,
                                                         active_codes=active_codes, scan_context=scan_context)
    if config['stairs_geometry_check'] and 'stairs' in by_id:
        result['stairs_geometry'] = votes.reject_planar_stairs(by_id['stairs'], origin, config['voxel_size'],
            config['stairs_plane_tolerance'], config['stairs_planar_fraction'], config['stairs_min_rise'], config['seed'], codes,
            minimum_voxels=config['stairs_min_voxels'], minimum_levels=config['stairs_min_levels'],
            level_tolerance=config['stairs_level_tolerance'], rise_minimum=config['stairs_step_rise_min'],
            rise_maximum=config['stairs_step_rise_max'], tread_minimum=config['stairs_step_tread_min'],
            tread_maximum=config['stairs_step_tread_max'], floor_code=by_id.get('floor'),
            floor_ring_radius=config['stairs_floor_ring_radius'], floor_level_tolerance=config['stairs_floor_level_tolerance'],
            underside_minimum_voxels=config['stairs_underside_min_voxels'],
            underside_cluster_size=config['stairs_underside_cluster_size'],
            underside_maximum_gap=config['stairs_underside_max_gap'])
    if config['component_cleanup']:
        rules = []
        if 'cable' in by_id:
            rules.append((by_id['cable'], config['cable_component_min_voxels'], config['cable_component_min_span']))
        if 'electrical' in by_id:
            rules.append((by_id['electrical'], config['electrical_component_min_voxels'], config['electrical_component_min_span']))
        result['component_cleanup'] = votes.remove_small_components(
            config['voxel_size'], config['component_min_voxels'], rules,
            component_min_cells=config['component_min_cells'], feature_cell_size=config['geometry_cell_size'],
            sparse_maximum_voxels_per_cell=config['component_sparse_max_voxels_per_cell'],
            outdoor_codes=[by_id[name] for name in ('building', 'ground', 'low_vegetation', 'tree', 'road', 'sidewalk') if name in by_id])
    if config['vegetation_base_to_ground'] and 'low_vegetation' in by_id and 'ground' in by_id:
        result['vegetation_base_to_ground'] = votes.vegetation_bases_to_ground(by_id['low_vegetation'], by_id['ground'],
            config['voxel_size'], config['vegetation_ground_cell_size'], config['vegetation_ground_band'])
    planes = None
    if config['geometry_fill']:
        planes = work/'planes.npz'
        result['geometry_fill'] = extract_planes(votes, by_id, origin, config['voxel_size'], config, planes)
    return result, planes, terrain


def reclassify(source_run: Path, output: Path, overrides_path: Path | None = None) -> dict:
    source_run = Path(source_run).resolve(strict=True)
    manifest_path = source_run/'run_config.json'
    if not manifest_path.is_file():
        raise ValueError('Der Ausgangslauf enthält keine run_config.json.')
    manifest = _load_json_object(manifest_path, 'run_config.json')
    overrides = _load_overrides(overrides_path)
    if manifest.get('cache_cleanup') != 'keep_all' and not manifest.get('vote_source_identity'):
        raise ValueError('Nachklassifizieren braucht einen Lauf mit Cache-Modus „alles behalten“ (keep_all).')
    cache = Path(manifest['cache'])
    identity = manifest.get('vote_source_identity', manifest.get('identity'))
    vote_source = cache/'votes'/identity/'votes.sqlite'
    vote_directory = vote_source.parent
    store_root = cache/'scan_store'/manifest['input_sha256']
    if not vote_source.is_file() or not store_root.is_dir():
        raise ValueError('Der benötigte Cache fehlt. Den Lauf mit Cache-Modus „alles behalten“ starten.')
    config = resolve_config({**{key: value for key, value in manifest.items() if key in DEFAULTS or key in ('source', 'cache', 'output', 'preset', 'model')},
                             **overrides, 'output': str(output), 'cache': str(cache)})
    if config['processing_mode'] != 'classify_and_filter':
        raise ValueError('Nachklassifizieren ist nur für semantisch klassifizierte Läufe verfügbar.')
    input_mode = _resolved_input_mode(source_run, manifest)
    target = new_output(Path(output), Path(config['source']))
    started = time.monotonic()
    copied_votes = target/'reclassify_cache'/'votes.sqlite'
    _copy_sqlite(vote_source, copied_votes)
    station_tracking = input_mode['mode'] == 'slam'
    votes = VoteStore(copied_votes, station_tracking=station_tracking)
    try:
        stores = [load_scan(Path(item['path']), manifest['input_sha256'], verify_hashes=False)
                  for item in json.loads((source_run/'store_report.json').read_text(encoding='utf-8'))['scans']]
        output_stores = [reduced_store(store, config['point_reduction'], config['seed']) for store in stores]
        preset = manifest['resolved_preset']
        rows = legend_rows(preset)
        all_classes = [item for item in preset['classes'] if item.get('enabled', True)]
        mapping = preset.get('code_map', {})
        by_id = {item['id']: int(mapping.get(str(item['code']), item['code'])) for item in all_classes}
        config = {**config, 'voxel_origin': manifest['voxel_origin']}
        source_report = _load_json_object(source_run/'report.json', 'report.json')
        scan_scenes = {int(item['scan']): item.get('scene') for item in source_report.get('scans', [])
                       if isinstance(item, dict) and isinstance(item.get('scan'), int)}
        if input_mode['mode'] == 'static':
            representatives_by_scan = {store.meta['scan_index']: vote_directory/f"scan_{store.meta['scan_index']:04d}"/'representatives.npy'
                                       for store in stores}
            existing = {scan: path for scan, path in representatives_by_scan.items() if path.is_file()}
            votes.rebuild_scan_observations(existing)
        scan_context = geometry_scan_context(preset, config, scan_scenes)
        stages, planes, terrain = _postprocess(votes, config, by_id,
            [store.meta['metadata']['pose']['translation']['z'] for store in stores], copied_votes.parent,
            {row['code'] for row in rows}, scan_context)
        if station_tracking:
            votes.finalize_station_ids()
        inspect = _load_json_object(source_run/'inspect_report.json', 'inspect_report.json')
        writer = write_e57 if config['format'] == 'e57' else write_cloud
        mirror = _StoredMirrorStates(vote_directory/'mirrors') if manifest.get('mirror_analysis') else None
        transfer_station_positions = None
        if station_tracking:
            cameras = build_camera_list(inspect)
            stations, _, _ = cluster_stations(cameras, config['slam_station_spacing'])
            transfer_station_positions = {station.id: np.asarray([cameras[index].pose.translation
                for index in station.camera_indices], np.float64) for station in stations}
        terrain_outdoor_scans = ({store.meta['scan_index'] for store in stores}
                                 if input_mode['mode'] == 'slam' and any(context['outdoor'] for context in scan_context.values())
                                 else {scan for scan, context in scan_context.items() if context['outdoor']})
        name = f"{Path(config['source']).stem}_classified.{config['format']}"
        path = target/name
        args = (path, Path(config['source']), output_stores, votes, np.asarray(config['voxel_origin']), config['voxel_size'],
                [row['code'] for row in rows]) if writer is write_e57 else (
                path, output_stores, votes, np.asarray(config['voxel_origin']), config['voxel_size'],
                inspect['metadata'].get('coordinateMetadata', ''), [row['code'] for row in rows])
        result = writer(*args, mirror=mirror.output if mirror else None, no_reflection=config['extra_no_reflection'],
                        output_density=config['output_density'], class_transfer_radius=config['class_transfer_radius'],
                        class_transfer_range_factor=config['class_transfer_range_factor'],
                        class_transfer_radius_max=config['class_transfer_radius_max'],
                        transfer_station_positions=transfer_station_positions,
                        geometry_planes=planes if config['geometry_fill'] else None,
                        geometry_fill_tolerance=config['geometry_fill_tolerance'], terrain=terrain,
                        terrain_ground_tolerance=config['terrain_ground_tolerance'],
                        terrain_fill=config['terrain_fill'] and any(row['code'] == 2 for row in rows),
                        station_ids=station_tracking,
                        terrain_outdoor_scans=terrain_outdoor_scans)
        result.update(point_reduction=config['point_reduction'], output_density=config['output_density'], format=config['format'])
        write_legends(target, rows)
        shutil.copy2(source_run/'scan_index.json', target/'scan_index.json')
        shutil.copy2(source_run/'inspect_report.json', target/'inspect_report.json')
        shutil.copy2(source_run/'store_report.json', target/'store_report.json')
        new_manifest = {**config, 'identity': manifest['identity'], 'vote_source_identity': identity,
                        'reclassified_from': str(source_run), 'reclassify_overrides': overrides}
        atomic_json(target/'run_config.json', new_manifest)
        report = {'schema_version': 1, 'input': manifest.get('source', config['source']), 'input_sha256': manifest['input_sha256'],
                  'input_mode': input_mode, 'output': result,
                  'scans': source_report.get('scans', []),
                  'warnings': [],
                  'reclassified_from': str(source_run), 'reclassify_overrides': overrides, **stages,
                  'total_seconds': time.monotonic()-started, 'created_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
        write_report(target, report, rows)
        return report
    finally:
        votes.close()
