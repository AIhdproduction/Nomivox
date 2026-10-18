"""Streaming post-run quality measurements and compact review artefacts."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import laspy
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import minimum_filter
from scipy.spatial import cKDTree

from e57cls.files import atomic_json, atomic_text


DISTANCE_BANDS = ((0., 5.), (5., 10.), (10., 15.), (15., 20.), (20., 30.), (30., None))
UNCLASSIFIED = 1
GROUND = 2
BUILDING_CODES = (6, 100)
TREE = 5
STAIRS = 105
DTM_DENSE_CELL_LIMIT = 16_000_000
NEIGHBOUR_BATCH_SIZE = 10_000
MAX_ANALYSIS_POINTS = 5_000_000


def _laz_options(path: Path) -> dict:
    if path.suffix.lower() != '.laz':
        return {}
    return {'laz_backend': getattr(laspy.LazBackend, 'LazrsParallel', laspy.LazBackend.Lazrs)}


def _cloud(run: Path) -> Path:
    clouds = sorted([*run.glob('*.laz'), *run.glob('*.las')])
    clouds = [path for path in clouds if '_no_reflection' not in path.stem]
    if len(clouds) != 1:
        raise ValueError('Der Laufordner muss genau eine primäre LAS- oder LAZ-Datei enthalten.')
    return clouds[0]


def _scene_data(run: Path):
    report_path, index_path, scan_index_path = run/'report.json', run/'inspect_report.json', run/'scan_index.json'
    if not report_path.is_file() or not index_path.is_file() or not scan_index_path.is_file():
        raise ValueError('Im Laufordner fehlen report.json, inspect_report.json oder scan_index.json.')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    inspection = json.loads(index_path.read_text(encoding='utf-8'))
    scan_index = json.loads(scan_index_path.read_text(encoding='utf-8'))
    scenes = {int(item['scan']): item.get('scene', 'unknown') for item in report.get('scans', []) if 'scan' in item}
    positions = {}
    for item in inspection.get('scans', []):
        try:
            translation = item['metadata']['pose']['translation']
            positions[int(item['index'])] = np.array([translation[key] for key in 'xyz'], np.float64)
        except (KeyError, TypeError, ValueError):
            continue
    return scenes, {scan: position for scan, position in positions.items() if str(scan) in scan_index}


def _apply_scan_overrides(scenes, indoor_scans, outdoor_scans):
    """Apply explicit scan-group overrides without relying on scan names."""
    indoor = set() if indoor_scans is None else {int(scan) for scan in indoor_scans}
    outdoor = set() if outdoor_scans is None else {int(scan) for scan in outdoor_scans}
    overlap = indoor & outdoor
    if overlap:
        values = ', '.join(map(str, sorted(overlap)))
        raise ValueError(f'Scans cannot be both indoor and outdoor: {values}.')
    resolved = {int(scan): str(scene).lower() for scan, scene in scenes.items()}
    resolved.update({scan: 'indoor' for scan in indoor})
    resolved.update({scan: 'outdoor' for scan in outdoor})
    return resolved


def _smoothed_dtm(cells: np.ndarray, cell_minimums: np.ndarray) -> np.ndarray:
    """Return the exact 5x5-cell minimum DTM while keeping sparse runs bounded."""
    lower, upper = cells.min(axis=0), cells.max(axis=0)
    shape = upper-lower+1
    cell_count = int(shape[0])*int(shape[1])
    if cell_count <= DTM_DENSE_CELL_LIMIT:
        grid = np.full(tuple(shape), np.inf)
        local = cells-lower
        grid[local[:, 0], local[:, 1]] = cell_minimums
        return minimum_filter(grid, size=5)[local[:, 0], local[:, 1]]

    # A sparse equivalent of a 5x5 minimum filter.  There are at most 25
    # occupied cells in each Chebyshev-radius-two neighbourhood.
    tree = cKDTree(cells)
    result = np.empty(len(cells), np.float64)
    count = min(25, len(cells))
    for start in range(0, len(cells), NEIGHBOUR_BATCH_SIZE):
        stop = min(start+NEIGHBOUR_BATCH_SIZE, len(cells))
        _, neighbours = tree.query(cells[start:stop], k=count, p=np.inf,
                                   distance_upper_bound=2., workers=-1)
        neighbours = np.asarray(neighbours).reshape(stop-start, count)
        values = np.full(neighbours.shape, np.inf)
        valid = neighbours < len(cells)
        values[valid] = cell_minimums[neighbours[valid]]
        result[start:stop] = values.min(axis=1)
    return result


def _vegetation_mask(xyz: np.ndarray, outside: np.ndarray, hag: np.ndarray) -> np.ndarray:
    """Find spatially scattered, elevated points with bounded neighbour queries."""
    result = np.zeros(len(xyz), bool)
    candidates = np.flatnonzero(outside & (hag > 2.5) & (hag < 25.))
    if not len(candidates):
        return result
    tree = cKDTree(xyz)
    counts = np.asarray(tree.query_ball_point(xyz[candidates], .6, workers=-1,
                                               return_length=True))
    eligible = candidates[counts >= 10]
    for start in range(0, len(eligible), NEIGHBOUR_BATCH_SIZE):
        selected = eligible[start:start+NEIGHBOUR_BATCH_SIZE]
        neighbourhoods = tree.query_ball_point(xyz[selected], .6, workers=-1)
        for point, neighbours in zip(selected, neighbourhoods):
            values = xyz[neighbours]
            centered = values-values.mean(axis=0)
            sigma = np.sqrt(np.clip(np.linalg.eigvalsh(centered.T@centered/len(values)), 0., None))[::-1]
            if sigma[0] > 0 and sigma[2]/sigma[0] > .3:
                result[point] = True
    return result


def _class_mix(codes: np.ndarray, selected: np.ndarray) -> dict:
    counts = Counter(map(int, codes[selected]))
    total = int(sum(counts.values()))
    return {'sampled_points': total, 'class_counts': dict(sorted(counts.items())),
            'class_shares': {code: _percent(value, total) for code, value in sorted(counts.items())},
            'tree_share': _percent(counts[TREE], total),
            'building_share': _percent(sum(counts[code] for code in BUILDING_CODES), total)}


def _floor_groups(scenes, positions):
    """Group indoor scanners by elevation without relying on scan names or GUIDs."""
    indoor = sorted((position[2], scan) for scan, position in positions.items() if scenes.get(scan) == 'indoor')
    groups = []
    for height, scan in indoor:
        if not groups or height-groups[-1]['scanner_height_m'] > 1.0:
            groups.append({'scanner_height_m': float(height), 'scans': [scan]})
        else:
            groups[-1]['scans'].append(scan)
            groups[-1]['scanner_height_m'] = float(np.mean([positions[value][2] for value in groups[-1]['scans']]))
    return groups


def _band_label(low, high):
    return f'{low:g}-{high:g}m' if high is not None else f'{low:g}m_plus'


def _percent(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _overview_png(path: Path, points: np.ndarray, codes: np.ndarray, title: str, colors: dict[int, str]):
    if not len(points):
        return
    low, high = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    span = np.maximum(high-low, .001)
    width, height = 1400, max(500, min(1400, round(1400*span[1]/span[0])))
    xy = np.rint((points[:, :2]-low)/span*np.array((width-1, height-1))).astype(np.int32)
    xy[:, 1] = height-1-xy[:, 1]
    image = Image.new('RGB', (width, height+54), 'white')
    pixels = image.load()
    palette = {code: tuple(int(value[index:index+2], 16) for index in (1, 3, 5)) for code, value in colors.items()}
    for (x, y), code in zip(xy, codes):
        pixels[int(x), int(y)] = palette.get(int(code), (120, 120, 120))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, height, width, height+54), fill='white')
    draw.text((12, height+10), title, fill='black')
    draw.text((12, height+30), 'Magenta: nicht klassifiziert', fill='#cc00cc')
    image.save(path)


def analyze_run(run_directory: Path, step: int = 40, *, indoor_scans=None, outdoor_scans=None) -> dict:
    """Create quality.json, German PNG plans and a small localized HTML index."""
    if step < 1:
        raise ValueError('--step muss mindestens 1 sein.')
    run = Path(run_directory).resolve(strict=True)
    cloud = _cloud(run)
    scenes, positions = _scene_data(run)
    scenes = _apply_scan_overrides(scenes, indoor_scans, outdoor_scans)
    legend = json.loads((run/'legend.json').read_text(encoding='utf-8')) if (run/'legend.json').is_file() else []
    colors = {int(row['code']): row.get('color', '#787878') for row in legend}
    colors[UNCLASSIFIED] = '#cc00cc'
    quality = run/'quality'
    quality.mkdir(exist_ok=True)
    distance = {scene: {kind: { _band_label(low, high): [0, 0] for low, high in DISTANCE_BANDS}
                        for kind in ('all_points', 'ground_points')}
                for scene in ('outdoor', 'indoor', 'unknown')}
    samples_xyz, samples_codes, samples_scan = [], [], []
    class_counts = Counter()
    scanned = sampled = 0
    with laspy.open(cloud, **_laz_options(cloud)) as reader:
        # Distance and class metrics use every requested sample.  The map and
        # Neighbourhood diagnostics use a bounded, evenly distributed subset.
        # Five million points retain sufficient local density for the 0.6-m
        # scatter test without retaining a whole multi-gigabyte run in RAM.
        analysis_stride = max(1, int(np.ceil(reader.header.point_count / step / MAX_ANALYSIS_POINTS)))
        for block in reader.chunk_iterator(1_000_000):
            count = len(block)
            ids = np.arange(scanned, scanned+count, dtype=np.int64)
            take = ids % step == 0
            scanned += count
            if not take.any():
                continue
            xyz = np.column_stack((block.x[take], block.y[take], block.z[take])).astype(np.float64, copy=False)
            codes = np.asarray(block.classification[take], np.uint8)
            scan = np.asarray(block.point_source_id[take], np.int64)
            sampled += len(xyz)
            class_counts.update(map(int, codes))
            analysis_take = ((ids[take] // step) % analysis_stride) == 0
            if analysis_take.any():
                samples_xyz.append(xyz[analysis_take]); samples_codes.append(codes[analysis_take]); samples_scan.append(scan[analysis_take])
            for scan_id in np.unique(scan):
                indices = scan == scan_id
                position = positions.get(int(scan_id))
                if position is None:
                    continue
                scene = scenes.get(int(scan_id), 'unknown')
                if scene not in distance:
                    scene = 'unknown'
                values = xyz[indices]
                labels = codes[indices]
                ranges = np.linalg.norm(values-position, axis=1)
                ground_like = (labels == GROUND) | (values[:, 2] < position[2]-1.)
                for low, high in DISTANCE_BANDS:
                    in_band = ranges >= low if high is None else (ranges >= low) & (ranges < high)
                    for kind, mask in (('all_points', in_band), ('ground_points', in_band & ground_like)):
                        total = int(mask.sum())
                        distance[scene][kind][_band_label(low, high)][0] += total
                        distance[scene][kind][_band_label(low, high)][1] += int(((labels == UNCLASSIFIED) & mask).sum())
    if not samples_xyz:
        raise ValueError('Die Ausgabedatei enthält keine Punkte.')
    xyz = np.concatenate(samples_xyz); codes = np.concatenate(samples_codes); scan_ids = np.concatenate(samples_scan)
    # The footprint is deliberately class-independent: every non-outdoor
    # scan contributes its observed 1-m cells, including mixed and unsure scans.
    cell = np.floor(xyz[:, :2]).astype(np.int64)
    keys, inverse = np.unique(cell, axis=0, return_inverse=True)
    cell_minimums = np.full(len(keys), np.inf)
    np.minimum.at(cell_minimums, inverse, xyz[:, 2])
    hag = xyz[:, 2]-_smoothed_dtm(keys, cell_minimums)[inverse]
    outdoor_scans_mask = np.array([scenes.get(int(scan), 'unknown') == 'outdoor' for scan in scan_ids])
    indoor_cells = np.zeros(len(keys), bool)
    indoor_cells[np.unique(inverse[~outdoor_scans_mask])] = True
    vegetation_like = _vegetation_mask(xyz, ~indoor_cells[inverse], hag)
    vegetation_groups = {
        'all_scans': _class_mix(codes, vegetation_like),
        'outdoor_scans': _class_mix(codes, vegetation_like & outdoor_scans_mask),
        'indoor_scans': _class_mix(codes, vegetation_like & ~outdoor_scans_mask),
    }
    groups = _floor_groups(scenes, positions)
    group_results = []
    for number, group in enumerate(groups, 1):
        selected = np.isin(scan_ids, group['scans'])
        values, labels = xyz[selected], codes[selected]
        if len(values):
            base = float(np.quantile(values[:, 2], .05))
            floor_band = (values[:, 2] >= base) & (values[:, 2] <= base+.30)
            group_results.append({'group': number, 'scanner_height_m': group['scanner_height_m'], 'scans': group['scans'],
                                  'floor_band_points': int(floor_band.sum()),
                                  'ground_code_2_share': _percent(int((labels[floor_band] == GROUND).sum()), int(floor_band.sum())),
                                  'stairs_share': _percent(int((labels == STAIRS).sum()), len(labels))})
            _overview_png(quality/f'floor_group_{number}.png', values[values[:, 2] <= group['scanner_height_m']+.3],
                          labels[values[:, 2] <= group['scanner_height_m']+.3], f'Innen, Geschossgruppe {number}', colors)
    converted = {scene: {kind: {band: {'sampled_points': values[0], 'unclassified_points': values[1],
                                        'unclassified_share': _percent(values[1], values[0])}
                                  for band, values in kinds.items()} for kind, kinds in by_kind.items()}
                 for scene, by_kind in distance.items()}
    _overview_png(quality/'outdoor_overview.png', xyz[outdoor_scans_mask], codes[outdoor_scans_mask], 'Aussenansicht', colors)
    vegetation_all = vegetation_groups['all_scans']
    result = {'schema_version': 1, 'source_cloud': cloud.name, 'sampling_step': step, 'sampled_points': sampled,
              'analysis_sampled_points': len(xyz),
              'distance_unclassified': converted,
              'vegetation_like_outside_buildings': vegetation_all | {'by_scan_group': vegetation_groups,
                  'footprint_indoor_scan_ids': sorted({int(scan) for scan in scan_ids[~outdoor_scans_mask]}),
                  'footprint_outdoor_scan_ids': sorted({int(scan) for scan in scan_ids[outdoor_scans_mask]})},
              'indoor_floor_groups': group_results, 'class_counts': dict(sorted(class_counts.items()))}
    atomic_json(quality/'quality.json', result)
    rows = ''.join(f'<li>Geschossgruppe {item["group"]}: Klasse 2 im Bodenband: '
                   f'{(item["ground_code_2_share"] or 0)*100:.2f} %, Treppen: {(item["stairs_share"] or 0)*100:.2f} %</li>' for item in group_results)
    atomic_text(quality/'quality.html', '<!doctype html><meta charset="utf-8"><title>Qualitätsbericht</title>'
                '<h1>Qualitätsbericht</h1><p>Die Kennzahlen basieren auf einer gleichmässigen Unterabtastung der Ausgabepunktwolke.</p>'
                '<ul><li><a href="quality.json">Maschinenlesbarer Bericht</a></li><li><a href="outdoor_overview.png">Aussenansicht</a></li>'
                f'{rows}</ul>')
    return result
