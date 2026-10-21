import io
import json
from collections import Counter
from pathlib import Path
import threading
import time
import numpy as np
from PIL import Image
import pye57
import pytest
from pye57 import libe57 as lib
import laspy

from e57cls.files import file_hash
from e57cls.geometry.cameras import Camera, plan_views
from e57cls.io.inspector import inspect_file
from e57cls.pipeline.config import resolve_config
from e57cls.pipeline import run as pipeline_run
from e57cls.pipeline.run import execute, InputNotClassifiable
from e57cls.pipeline.control import Cancelled, Control
from e57cls.mirror import pipeline as mirror_pipeline
from e57cls.output.reports import write_report


def small_project(path, image_problem=None, scan_count=1, unique_images=False):
    writer = pye57.E57(str(path), 'w')
    # A wall straight ahead, with modest angular separation between points.
    y, z = np.meshgrid(np.linspace(-1, 1, 11), np.linspace(-1, 1, 11))
    points = {'cartesianX': np.full(y.size, 3.), 'cartesianY': y.ravel(), 'cartesianZ': z.ravel()}
    for index in range(scan_count):
        writer.write_scan_raw(points, name=f'synthetic-{index + 1}')
    if image_problem == 'no_images':
        writer.close()
        return
    imf = writer.image_file
    images = writer.root['images2D']
    for index in range(scan_count):
        node = lib.StructureNode(imf)
        images.append(node)
        node.set('guid', lib.StringNode(imf, f'synthetic-image-{index + 1}'))
        guid = 'unknown-scan' if image_problem == 'unassociated' else writer.data3d[index]['guid'].value()
        node.set('associatedData3DGuid', lib.StringNode(imf, guid))
        pose = lib.StructureNode(imf)
        node.set('pose', pose)
        for group, values in [('rotation', dict(w=1., x=0., y=0., z=0.)), ('translation', dict(x=0., y=0., z=0.))]:
            child = lib.StructureNode(imf)
            pose.set(group, child)
            for key, value in values.items():
                if image_problem == 'invalid_pose' and key == 'w':
                    value = 2.
                child.set(key, lib.FloatNode(imf, value))
        rep = lib.StructureNode(imf)
        node.set('visualReferenceRepresentation' if image_problem == 'reference_only' else 'sphericalRepresentation', rep)
        for key, value in dict(imageWidth=256, imageHeight=128).items():
            rep.set(key, lib.IntegerNode(imf, value))
        for key in ('pixelWidth', 'pixelHeight'):
            if image_problem == 'missing_parameters' and key == 'pixelHeight':
                continue
            rep.set(key, lib.FloatNode(imf, np.pi/128))
        stream = io.BytesIO()
        color = (130 + index, 140, 150) if unique_images else (130, 140, 150)
        Image.new('RGB', (256, 128), color).save(stream, format='PNG')
        data = np.frombuffer(stream.getvalue(), np.uint8)
        blob = lib.BlobNode(imf, len(data))
        rep.set('pngImage', blob)
        blob.write(data, 0, len(data))
    writer.close()


class Backend:
    calls = 0
    reductions = []
    def predict_tile(self, image, classes):
        self.calls += 1
        return np.full((len(classes), image.height, image.width), .9, np.float32)


class RecordingBackend(Backend):
    def __init__(self):
        self.calls = 0
        self.reductions = []
        self.prompt_groups = []
        self.sizes = []

    def predict_tile(self, image, classes):
        self.prompt_groups.append(tuple(item.id for item in classes))
        self.sizes.append((image.width, image.height))
        return super().predict_tile(image, classes)


@pytest.mark.parametrize('problem', ['no_images', 'invalid_pose', 'reference_only', 'missing_parameters'])
def test_unusable_input_stops_before_point_store_and_sam(tmp_path, monkeypatch, problem):
    source = tmp_path/'invalid.e57'
    small_project(source, problem)
    before = file_hash(source)
    config = dict(source=str(source), output=str(tmp_path/'out'), cache=str(tmp_path/'cache'))
    def forbidden(*args, **kwargs):
        pytest.fail('Rejected input must not build the point store, hash the model or load SAM')
    monkeypatch.setattr('e57cls.pipeline.run.build_store', forbidden)
    monkeypatch.setattr('e57cls.pipeline.run.file_hash', forbidden)
    with pytest.raises(InputNotClassifiable, match='vor der Punktverarbeitung abgebrochen'):
        execute(config, forbidden, listen=False)
    output = Path(config['output'])
    report = json.loads((output/'inspect_report.json').read_text(encoding='utf-8'))
    assert not report['scans'][0]['classification_eligible_metadata']
    events = [json.loads(line) for line in (output/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert events[-1]['code'] == 'input_rejected'
    assert ('keine eingebetteten Kamerabilder' if problem == 'no_images' else 'enthält Bilder') in events[-1]['detail']
    assert [e['phase'] for e in events if e['event'] == 'phase_started'] == ['inspection']
    assert not (Path(config['cache'])/'scan_store').exists()
    assert not list(output.glob('*.las'))
    assert file_hash(source) == before


@pytest.mark.parametrize('state,expected',[(4,100),(2,68)])
def test_reflection_export_and_uncertain_mask_semantics(tmp_path,monkeypatch,state,expected):
    source=tmp_path/'input.e57'
    small_project(source)
    preset=json.loads(Path('presets/default.json').read_text(encoding='utf-8'))
    preset['classes']=[c for c in preset['classes'] if c['id']=='wall']
    path=tmp_path/'preset.json'; path.write_text(json.dumps(preset),encoding='utf-8')
    model=tmp_path/'model.pt'; model.write_bytes(b'fixture')
    config=resolve_config(dict(source=str(source),cache=str(tmp_path/'cache'),output=str(tmp_path/'output'),
                               preset=str(path),model=str(model),scene_mode='indoor',profile='fast',
                               view_size=128,boundary_margin=0,depth_width=256,mirror_analysis=True,
                               component_cleanup=False,point_reduction=0,format='las',pipeline_parallel=False))
    class Analysis:
        registry=type('Registry',(),{'planes':[]})()
        report={'images':[],'scans':{}}
        def states(self,world,*args):
            return np.full(len(world),state,np.uint8)
        def membership(self,*args):
            raise AssertionError('An uncertain mask must not block normal semantic classification')
        def output(self,store,start,stop,valid,world,codes):
            if state==2: codes[:]=68
            return codes,self.states(world)
        def close(self): pass
    monkeypatch.setattr('e57cls.pipeline.run.build_mirrors',lambda *args:Analysis())
    report=execute(config,lambda:Backend(),listen=False)
    assert report['output']['class_counts']=={str(expected):121}
    assert report['output']['mirror_counts'][state]==121


def test_full_pipeline_and_voting_only_rerun(tmp_path):
    source = tmp_path/'small.e57'
    small_project(source)
    before = file_hash(source)
    preset = json.loads(Path('presets/default.json').read_text(encoding='utf-8'))
    preset['classes'] = [c for c in preset['classes'] if c['id'] == 'wall']
    preset['scene_resolved_pairs'] = []
    preset_path = tmp_path/'preset.json'
    preset_path.write_text(json.dumps(preset), encoding='utf-8')
    model = tmp_path/'model.pt'
    model.write_bytes(b'synthetic backend')
    config = resolve_config(dict(source=str(source), cache=str(tmp_path/'c'), output=str(tmp_path/'out'),
                                 preset=str(preset_path), model=str(model), profile='fast', scene_mode='indoor',
                                 view_size=128, boundary_margin=0, depth_width=256, mirror_analysis=False,
                                 component_cleanup=False, point_reduction=0, format='las',
                                 cache_cleanup='keep_masks'))
    backend = Backend()
    report = execute(config, lambda: backend, listen=False)
    assert report['output']['record_count'] == 121
    assert report['output']['class_counts']['100'] == 121
    assert report['sam_calls'] > 0
    config['output'] = str(tmp_path/'out2')
    config['min_observations'] = 2
    def forbidden():
        raise AssertionError('Voting changes must not load SAM')
    report2 = execute(config, forbidden, listen=False)
    assert report2['sam_calls'] == 0
    assert report2['output']['class_counts'] == {'1': 121}
    assert file_hash(source) == before

    # Interrupted work keeps only atomically published mask entries. Resume equals baseline.
    config['min_observations'] = 1
    config['cache'] = str(tmp_path/'cancel_cache')
    config['output'] = str(tmp_path/'cancel_out')
    class Interrupted(Backend):
        def predict_tile(self,image,classes):
            if self.calls == 2:
                raise Cancelled()
            return super().predict_tile(image,classes)
    assert execute(config,Interrupted,listen=False) == {'cancelled':True}
    assert (Path(config['cache']) / 'scan_store' / file_hash(source)).is_dir()
    assert list((Path(config['cache']) / 'seg').glob('*.npz'))
    resumed = execute(config,Backend,listen=False)
    assert resumed['output']['class_counts'] == report['output']['class_counts']
    assert resumed['sam_calls'] < report['sam_calls']


def test_high_profile_pipeline_runs_with_fake_backend(tmp_path):
    source = tmp_path/'high-profile.e57'
    small_project(source)
    preset = json.loads(Path('presets/default.json').read_text(encoding='utf-8'))
    preset['classes'] = [item for item in preset['classes'] if item['id'] == 'wall']
    preset['scene_resolved_pairs'] = []
    preset_path = tmp_path/'preset.json'; preset_path.write_text(json.dumps(preset), encoding='utf-8')
    model = tmp_path/'model.pt'; model.write_bytes(b'synthetic backend')
    config = resolve_config(dict(source=str(source), cache=str(tmp_path/'cache'), output=str(tmp_path/'out'),
                                 preset=str(preset_path), model=str(model), profile='high', scene_mode='indoor',
                                 view_size=128, boundary_margin=0, depth_width=256, mirror_analysis=True,
                                 component_cleanup=False, point_reduction=0, format='las', cache_cleanup='keep_masks'))
    backend = RecordingBackend()
    report = execute(config, lambda: backend, listen=False)
    assert backend.calls == 25
    assert report['sam_calls'] == 25
    assert report['output']['class_counts']['100'] == 121


def test_shared_sam_batches_class_and_reflection_prompts_once_per_view(tmp_path, monkeypatch):
    """Shared mode intentionally creates reflection masks at class-view resolution."""
    source = tmp_path/'shared.e57'
    small_project(source)
    preset = json.loads(Path('presets/default.json').read_text(encoding='utf-8'))
    preset['classes'] = [item for item in preset['classes'] if item['id'] == 'wall']
    preset['scene_resolved_pairs'] = []
    preset_path = tmp_path/'preset.json'; preset_path.write_text(json.dumps(preset), encoding='utf-8')
    model = tmp_path/'model.pt'; model.write_bytes(b'synthetic backend')
    common = dict(source=str(source), cache=str(tmp_path/'cache'), preset=str(preset_path), model=str(model),
                  profile='fast', scene_mode='both', view_size=128, mirror_view_size=336,
                  image_scale=.9, mirror_image_scale=.75, boundary_margin=0, depth_width=256,
                  mirror_analysis=True, component_cleanup=False, point_reduction=0, format='las',
                  cache_cleanup='keep_masks')
    detect_calls = []
    monkeypatch.setattr('e57cls.pipeline.run.detect', lambda *args: (detect_calls.append(args[1].meta['scan_index'])
                                                                     or {'scene': 'indoor', 'metrics': {}}))
    backend = RecordingBackend()
    report = execute(resolve_config({**common, 'output': str(tmp_path/'out')}), lambda: backend, listen=False)
    mirror_ids = [f'{kind}_{index}' for kind in ('mirror_mask', 'reflective_mask', 'glass_mask')
                  for index, _ in enumerate(next(item for item in preset['auxiliary'] if item['id'] == kind)['prompts'])]
    assert backend.calls == 9  # synthetic panorama + fast profile: nine semantic views, one tile each
    assert all(group == tuple(['wall', *mirror_ids]) for group in backend.prompt_groups)
    assert all(size == (80, 80) for size in backend.sizes)
    assert report['sam_calls'] == 9
    assert detect_calls == [1]
    # Mirror fitting and semantic voting consume only the masks written above.
    second = execute(resolve_config({**common, 'output': str(tmp_path/'out_again')}),
                     lambda: (_ for _ in ()).throw(AssertionError('shared masks must be cache hits')), listen=False)
    assert second['sam_calls'] == 0


def test_scene_resume_and_name_hint_resolution(tmp_path, monkeypatch):
    source = tmp_path/'scene.e57'; small_project(source)
    preset = json.loads(Path('presets/default.json').read_text(encoding='utf-8'))
    preset['classes'] = [item for item in preset['classes'] if item['id'] == 'wall']
    preset['scene_resolved_pairs'] = []
    preset_path = tmp_path/'preset.json'; preset_path.write_text(json.dumps(preset), encoding='utf-8')
    model = tmp_path/'model.pt'; model.write_bytes(b'synthetic backend')
    common = dict(source=str(source), cache=str(tmp_path/'cache'), preset=str(preset_path), model=str(model),
                  profile='fast', scene_mode='both', view_size=128, boundary_margin=0, depth_width=256,
                  mirror_analysis=False, component_cleanup=False, point_reduction=0, format='las',
                  cache_cleanup='keep_all')
    detect_calls = []
    monkeypatch.setattr(pipeline_run, 'detect', lambda *args: (detect_calls.append(args[1].meta['scan_index'])
                                                               or {'scene': 'indoor', 'metrics': {}}))
    first = execute(resolve_config({**common, 'output': str(tmp_path/'first')}), lambda: Backend(), listen=False)
    assert first['scans'][0]['scene'] == 'indoor'
    resumed = execute(resolve_config({**common, 'output': str(tmp_path/'resumed')}),
                      lambda: (_ for _ in ()).throw(AssertionError('completed scan must resume')), listen=False)
    assert resumed['scans'][0]['resumed']
    assert resumed['scans'][0]['scene'] == 'indoor'
    assert detect_calls == [1]

    original_load_scan = pipeline_run.load_scan
    def load_with_scan_name(*args, **kwargs):
        store = original_load_scan(*args, **kwargs)
        store.meta['metadata']['rlms:scanposGuid'] = '/P/SCANS/01_Aussen 001'
        return store
    monkeypatch.setattr(pipeline_run, 'load_scan', load_with_scan_name)
    named = execute(resolve_config({**common, 'cache': str(tmp_path/'named-cache'),
                                    'output': str(tmp_path/'named')}), lambda: Backend(), listen=False)
    assert named['scans'][0]['scene'] == 'outdoor'
    assert named['scans'][0]['scene_resolution']['source'] == 'name'
    assert {'scene_resolution', 'scene_consistency', 'scene_audit'} <= named.keys()
    assert (tmp_path/'named'/'report.json').is_file()
    assert (tmp_path/'named'/'report.html').is_file()
    assert any(warning['code'] == 'scene_name_conflict' for warning in named['warnings'])


def test_html_report_tolerates_scan_without_scene(tmp_path):
    report = {'output': {'class_counts': {'1': 1}, 'record_count': 1, 'unclassified_share': 1.,
                         'readback': {'max_coordinate_error_m': [0., 0., 0.]}, 'path': str(tmp_path/'cloud.las')},
              'scans': [{'scan': 1}], 'warnings': [], 'total_seconds': 0.}
    write_report(tmp_path, report, [{'code': 1, 'name_de': 'Unklassifiziert', 'color': '#000000'}])
    assert (tmp_path/'report.html').is_file()


def test_filter_only_uses_mirror_views_without_sam_and_writes_fixed_states(tmp_path, monkeypatch):
    source = tmp_path/'filter-only.e57'; small_project(source)
    preset = json.loads(Path('presets/default.json').read_text(encoding='utf-8'))
    preset['classes'] = [item for item in preset['classes'] if item['id'] == 'wall']
    preset['scene_resolved_pairs'] = []
    preset_path = tmp_path/'preset.json'; preset_path.write_text(json.dumps(preset), encoding='utf-8')
    model = tmp_path/'model.pt'; model.write_bytes(b'synthetic backend')
    config = resolve_config(dict(source=str(source), cache=str(tmp_path/'cache'), output=str(tmp_path/'out'),
                                 preset=str(preset_path), model=str(model), processing_mode='filter_only',
                                 profile='fast', scene_mode='indoor', view_size=128, mirror_view_size=336,
                                 image_scale=.9, mirror_image_scale=.5, boundary_margin=0, depth_width=256,
                                 component_cleanup=False, point_reduction=0, format='las', cache_cleanup='keep_all'))
    planned_sizes, scales = [], []
    original_plan_views = mirror_pipeline.plan_views
    original_segment_view = mirror_pipeline.segment_view
    original_build = pipeline_run.build_mirrors

    def record_plan_views(camera, profile, size, fov):
        planned_sizes.append(size)
        return original_plan_views(camera, profile, size, fov)

    def record_segment_view(*args, **kwargs):
        scales.append(args[8] if len(args) > 8 else kwargs['scale'])
        return original_segment_view(*args, **kwargs)

    def build_with_fixed_states(*args, **kwargs):
        analysis = original_build(*args, **kwargs)

        class FixedStates:
            report = analysis.report
            registry = analysis.registry

            @staticmethod
            def output(store, start, stop, valid, world, codes):
                states = np.zeros(len(world), np.uint8)
                states[np.arange(start, stop)[valid] % 3 == 0] = 4
                return codes, states

            @staticmethod
            def close():
                analysis.close()

        return FixedStates()

    monkeypatch.setattr(mirror_pipeline, 'plan_views', record_plan_views)
    monkeypatch.setattr(mirror_pipeline, 'segment_view', record_segment_view)
    monkeypatch.setattr(pipeline_run, 'build_mirrors', build_with_fixed_states)
    backend = RecordingBackend()
    report = execute(config, lambda: backend, listen=False)
    events = [json.loads(line) for line in (Path(config['output'])/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    output_path = next(Path(config['output']).glob('*.las'))
    output = laspy.read(output_path)
    expected_states = np.zeros(121, np.uint8)
    expected_states[::3] = 4

    assert backend.calls == len(scales) > 0
    assert planned_sizes and set(planned_sizes) == {config['mirror_view_size']}
    assert set(scales) == {config['mirror_image_scale']}
    assert all(all(prompt.startswith(('mirror_mask_', 'reflective_mask_', 'glass_mask_')) for prompt in group)
               for group in backend.prompt_groups)
    assert 'sam' not in [event['phase'] for event in events if event['event'] == 'phase_started']
    np.testing.assert_array_equal(np.asarray(output.mirror_state), expected_states)
    np.testing.assert_array_equal(np.asarray(output.classification), np.ones(121, np.uint8))
    assert report['output']['mirror_counts'][4] == int(expected_states.astype(bool).sum())

    reference_config = resolve_config({**config, 'output': str(tmp_path/'reference')})
    reference = execute(reference_config, lambda: pytest.fail('all fixed-score masks must come from cache'), listen=False)
    reference_path = next(Path(reference_config['output']).glob('*.las'))
    assert reference['sam_calls'] == 0
    assert output_path.read_bytes() == reference_path.read_bytes()


def test_classify_without_mirror_analysis_never_sends_reflection_prompts(tmp_path):
    source = tmp_path/'no-mirror-prompts.e57'; small_project(source)
    preset = json.loads(Path('presets/default.json').read_text(encoding='utf-8'))
    preset['classes'] = [item for item in preset['classes'] if item['id'] == 'wall']
    preset['scene_resolved_pairs'] = []
    preset_path = tmp_path/'preset.json'; preset_path.write_text(json.dumps(preset), encoding='utf-8')
    model = tmp_path/'model.pt'; model.write_bytes(b'synthetic backend')
    config = resolve_config(dict(source=str(source), cache=str(tmp_path/'cache'), output=str(tmp_path/'out'),
                                 preset=str(preset_path), model=str(model), processing_mode='classify_and_filter',
                                 profile='fast', scene_mode='indoor', view_size=128, boundary_margin=0,
                                 depth_width=256, mirror_analysis=False, component_cleanup=False,
                                 point_reduction=0, format='las', cache_cleanup='keep_all'))
    backend = RecordingBackend()
    execute(config, lambda: backend, listen=False)

    assert backend.calls > 0
    assert backend.prompt_groups and set(backend.prompt_groups) == {('wall',)}


def test_shared_sam_makes_no_backend_calls_after_preparation(tmp_path, monkeypatch):
    source = tmp_path/'sam-phase-boundary.e57'; small_project(source)
    preset = json.loads(Path('presets/default.json').read_text(encoding='utf-8'))
    preset['classes'] = [item for item in preset['classes'] if item['id'] == 'wall']
    preset['scene_resolved_pairs'] = []
    preset_path = tmp_path/'preset.json'; preset_path.write_text(json.dumps(preset), encoding='utf-8')
    model = tmp_path/'model.pt'; model.write_bytes(b'synthetic backend')
    config = resolve_config(dict(source=str(source), cache=str(tmp_path/'cache'), output=str(tmp_path/'out'),
                                 preset=str(preset_path), model=str(model), processing_mode='classify_and_filter',
                                 profile='fast', scene_mode='indoor', view_size=128, boundary_margin=0,
                                 depth_width=256, mirror_analysis=True, component_cleanup=False,
                                 point_reduction=0, format='las', cache_cleanup='keep_all'))
    class SamBoundaryControl(Control):
        def event(self, kind, **payload):
            if kind == 'phase_finished' and payload.get('phase') == 'sam':
                backend.sam_finished = True
            return super().event(kind, **payload)

    class RejectAfterSam(RecordingBackend):
        sam_finished = False

        def __init__(self):
            super().__init__()
            self.calls_after_sam = 0

        def predict_tile(self, image, classes):
            if self.sam_finished:
                self.calls_after_sam += 1
            return super().predict_tile(image, classes)

    backend = RejectAfterSam()
    monkeypatch.setattr(pipeline_run, 'Control', SamBoundaryControl)
    report = execute(config, lambda: backend, listen=False)

    assert backend.calls > 0
    assert backend.sam_finished
    assert backend.calls_after_sam == 0
    assert report['sam_calls'] == backend.calls


def test_shared_sam_resume_reuses_prepared_views_after_cancellation(tmp_path, monkeypatch):
    source = tmp_path/'resume-sam.e57'; small_project(source, scan_count=2)
    preset = json.loads(Path('presets/default.json').read_text(encoding='utf-8'))
    preset['classes'] = [item for item in preset['classes'] if item['id'] == 'wall']
    preset['scene_resolved_pairs'] = []
    preset_path = tmp_path/'preset.json'; preset_path.write_text(json.dumps(preset), encoding='utf-8')
    model = tmp_path/'model.pt'; model.write_bytes(b'synthetic backend')
    common = dict(source=str(source), preset=str(preset_path), model=str(model), processing_mode='classify_and_filter',
                  profile='fast', scene_mode='indoor', view_size=128, boundary_margin=0, depth_width=256,
                  mirror_analysis=True, component_cleanup=False, point_reduction=0, format='las', cache_cleanup='keep_all')
    fixture = inspect_file(source)
    fixture_image = fixture['images'][0]
    fixture_kind = next(kind for kind in fixture_image['representations'] if kind != 'visualReferenceRepresentation')
    fixture_camera = Camera(fixture_kind, fixture_image['metadata'][fixture_kind])
    views_per_fixture_image = len(plan_views(fixture_camera, common['profile'], common['view_size'],
                                              resolve_config(common)['view_fov']) or [None])
    uninterrupted_config = resolve_config({**common, 'cache': str(tmp_path/'baseline-cache'), 'output': str(tmp_path/'baseline')})
    context = {'phase': None, 'scan': None, 'image': None, 'view': None}
    image_scans, view_numbers = {}, Counter()
    original_prepare = pipeline_run.prepare_shared_sam
    original_assets = pipeline_run.image_assets
    original_segment = pipeline_run.segment_view

    class TraceControl(Control):
        def event(self, kind, **payload):
            if kind == 'phase_started':
                context['phase'] = payload['phase']
            return super().event(kind, **payload)

    class PhaseRecordingBackend(RecordingBackend):
        def __init__(self):
            super().__init__()
            self.call_records = []

        def predict_tile(self, image, classes):
            self.call_records.append({**context, 'prompts': tuple(item.id for item in classes)})
            return super().predict_tile(image, classes)

    def trace_prepare(reader, stores, inspection, *args, **kwargs):
        image_scans.update({image: store.meta['scan_index'] for store in stores
                            for image in inspection['scans'][store.meta['scan_index']-1]['image_indices']})
        return original_prepare(reader, stores, inspection, *args, **kwargs)

    def trace_assets(reader, item, cache):
        context.update(scan=image_scans[item['index']], image=item['index'], view=None)
        return original_assets(reader, item, cache)

    def trace_segment(*args, **kwargs):
        key = context['phase'], context['scan'], context['image']
        view_numbers[key] += 1
        context['view'] = view_numbers[key]
        return original_segment(*args, **kwargs)

    monkeypatch.setattr(pipeline_run, 'Control', TraceControl)
    monkeypatch.setattr(pipeline_run, 'prepare_shared_sam', trace_prepare)
    monkeypatch.setattr(pipeline_run, 'image_assets', trace_assets)
    monkeypatch.setattr(pipeline_run, 'segment_view', trace_segment)

    uninterrupted_backend = PhaseRecordingBackend()
    uninterrupted = execute(uninterrupted_config, lambda: uninterrupted_backend, listen=False)
    calls_by_scan = Counter(record['scan'] for record in uninterrupted_backend.call_records)
    # ``small_project`` writes one byte-identical spherical image with the same
    # camera data for each scan. With one tile per fixture view, the first scan
    # computes every planned view; the cache identity excludes the scan number,
    # so scan 2 reuses all of them.
    expected_calls_by_scan = {1: views_per_fixture_image, 2: 0}
    assert expected_calls_by_scan[1] > 0
    assert all(calls_by_scan[scan] == expected for scan, expected in expected_calls_by_scan.items())
    assert all(max(size) <= uninterrupted_config['tile_size'] for size in uninterrupted_backend.sizes)
    assert all(record['phase'] == 'sam' for record in uninterrupted_backend.call_records)
    assert all(record['image'] == record['scan'] for record in uninterrupted_backend.call_records)
    assert all(record['prompts'][0] == 'wall' and len(record['prompts']) > 1
               for record in uninterrupted_backend.call_records)

    config = resolve_config({**common, 'cache': str(tmp_path/'resume-cache'), 'output': str(tmp_path/'resumed')})
    calls_before_cancel = max(1, expected_calls_by_scan[1] // 2)

    class CancelDuringPreparedViews(RecordingBackend):
        def predict_tile(self, image, classes):
            if self.calls >= calls_before_cancel:
                raise Cancelled()
            return super().predict_tile(image, classes)

    assert execute(config, CancelDuringPreparedViews, listen=False) == {'cancelled': True}
    resumed_backend = RecordingBackend()
    resumed = execute(config, lambda: resumed_backend, listen=False)
    baseline_las = laspy.read(next(Path(uninterrupted_config['output']).glob('*.las')))
    resumed_las = laspy.read(next(Path(config['output']).glob('*.las')))

    assert resumed_backend.calls == expected_calls_by_scan[1]-calls_before_cancel
    assert resumed['output']['class_counts'] == uninterrupted['output']['class_counts']
    assert resumed['output']['mirror_counts'] == uninterrupted['output']['mirror_counts']
    np.testing.assert_array_equal(resumed_las.points.array, baseline_las.points.array)


def test_parallel_pipeline_matches_serial_runs_sam_in_producer_and_overlaps_cpu(tmp_path, monkeypatch):
    source = tmp_path/'parallel.e57'; small_project(source, scan_count=2, unique_images=True)
    preset = json.loads(Path('presets/default.json').read_text(encoding='utf-8'))
    preset['classes'] = [item for item in preset['classes'] if item['id'] == 'wall']
    preset['scene_resolved_pairs'] = []
    preset_path = tmp_path/'preset.json'; preset_path.write_text(json.dumps(preset), encoding='utf-8')
    model = tmp_path/'model.pt'; model.write_bytes(b'synthetic backend')
    common = dict(source=str(source), preset=str(preset_path), model=str(model), profile='fast', scene_mode='indoor',
                  view_size=128, boundary_margin=0, depth_width=256, mirror_analysis=True,
                  component_cleanup=False, point_reduction=0, format='las', cache_cleanup='keep_all')
    original_classify = pipeline_run.classify_scan

    def delayed_classify(*args, **kwargs):
        time.sleep(.10)
        return original_classify(*args, **kwargs)

    monkeypatch.setattr(pipeline_run, 'classify_scan', delayed_classify)

    class SlowBackend(RecordingBackend):
        def __init__(self):
            super().__init__()
            self.thread_ids = []

        def predict_tile(self, image, classes):
            self.thread_ids.append(threading.get_ident())
            time.sleep(.025)
            return super().predict_tile(image, classes)

    serial_config = resolve_config({**common, 'cache': str(tmp_path/'serial-cache'), 'output': str(tmp_path/'serial'),
                                    'pipeline_parallel': False})
    serial_started = time.monotonic()
    serial = execute(serial_config, lambda: SlowBackend(), listen=False)
    serial_seconds = time.monotonic()-serial_started

    parallel_config = resolve_config({**common, 'cache': str(tmp_path/'parallel-cache'), 'output': str(tmp_path/'parallel'),
                                      'pipeline_parallel': True, 'pipeline_prefetch_scans': 1})
    backend = SlowBackend()
    parallel_started = time.monotonic()
    parallel = execute(parallel_config, lambda: backend, listen=False)
    parallel_seconds = time.monotonic()-parallel_started

    serial_las = laspy.read(next(Path(serial_config['output']).glob('*.las')))
    parallel_las = laspy.read(next(Path(parallel_config['output']).glob('*.las')))
    np.testing.assert_array_equal(parallel_las.points.array, serial_las.points.array)
    assert parallel['output']['class_counts'] == serial['output']['class_counts']
    assert parallel['output']['mirror_counts'] == serial['output']['mirror_counts']
    assert parallel['reflection_analysis']['scans'] == serial['reflection_analysis']['scans']
    assert backend.thread_ids and set(backend.thread_ids) != {threading.get_ident()}
    assert parallel_seconds < serial_seconds


@pytest.mark.parametrize('strict', [False, True])
def test_parallel_producer_failure_is_forwarded_without_a_live_thread(tmp_path, strict):
    source = tmp_path/'producer-error.e57'; small_project(source, scan_count=2, unique_images=True)
    preset = json.loads(Path('presets/default.json').read_text(encoding='utf-8'))
    preset['classes'] = [item for item in preset['classes'] if item['id'] == 'wall']
    preset['scene_resolved_pairs'] = []
    preset_path = tmp_path/'preset.json'; preset_path.write_text(json.dumps(preset), encoding='utf-8')
    model = tmp_path/'model.pt'; model.write_bytes(b'synthetic backend')
    config = resolve_config(dict(source=str(source), cache=str(tmp_path/'cache'), output=str(tmp_path/'out'),
                                 preset=str(preset_path), model=str(model), profile='fast', scene_mode='indoor',
                                 view_size=128, boundary_margin=0, depth_width=256, mirror_analysis=True,
                                 component_cleanup=False, point_reduction=0, format='las', cache_cleanup='keep_all',
                                 pipeline_parallel=True, strict=strict))

    class FailingBackend(RecordingBackend):
        def predict_tile(self, image, classes):
            if self.calls >= 9:
                raise RuntimeError('producer failure')
            return super().predict_tile(image, classes)

    if strict:
        with pytest.raises(RuntimeError, match='producer failure'):
            execute(config, lambda: FailingBackend(), listen=False)
    else:
        report = execute(config, lambda: FailingBackend(), listen=False)
        assert report['skipped_scans'] == [2]
        assert any(item['code'] == 'scan_failed' and item['scan'] == 2 for item in report['warnings'])
    assert not any(thread.name == 'sam-producer' and thread.is_alive() for thread in threading.enumerate())


def test_default_reduction_writes_laz_density(tmp_path):
    source = tmp_path/'default.e57'; small_project(source)
    preset = json.loads(Path('presets/default.json').read_text(encoding='utf-8'))
    preset['classes'] = [item for item in preset['classes'] if item['id'] == 'wall']
    preset_path = tmp_path/'preset.json'; preset_path.write_text(json.dumps(preset), encoding='utf-8')
    model = tmp_path/'model.pt'; model.write_bytes(b'synthetic backend')
    config = resolve_config(dict(source=str(source), cache=str(tmp_path/'cache'), output=str(tmp_path/'out'),
                                 preset=str(preset_path), model=str(model), profile='fast', scene_mode='indoor',
                                 view_size=128, boundary_margin=0, depth_width=256, mirror_analysis=False,
                                 component_cleanup=False))
    report = execute(config, lambda: Backend(), listen=False)
    assert report['output']['record_count'] == round(.6*121)
    assert report['output']['format'] == 'laz'


@pytest.mark.parametrize('mode,keep_masks', [('all', False), ('keep_masks', True)])
def test_successful_cache_cleanup_is_run_scoped(tmp_path, mode, keep_masks):
    source = tmp_path / 'cleanup.e57'
    small_project(source)
    preset = json.loads(Path('presets/default.json').read_text(encoding='utf-8'))
    preset['classes'] = [item for item in preset['classes'] if item['id'] == 'wall']
    preset['scene_resolved_pairs'] = []
    preset_path = tmp_path / 'preset.json'; preset_path.write_text(json.dumps(preset), encoding='utf-8')
    model = tmp_path / 'model.pt'; model.write_bytes(b'synthetic backend')
    cache = tmp_path / 'cache'
    for name in ('venv', 'pip'):
        path = cache / name; path.mkdir(parents=True); (path / 'keep.txt').write_text('keep', encoding='utf-8')
    (cache / 'launcher.json').write_text('{}', encoding='utf-8')
    foreign = cache / 'scan_store' / 'foreign-input'; foreign.mkdir(parents=True); (foreign / 'keep.txt').write_text('keep', encoding='utf-8')
    config = resolve_config(dict(source=str(source), cache=str(cache), output=str(tmp_path / 'out'),
                                 preset=str(preset_path), model=str(model), profile='fast', scene_mode='indoor',
                                 view_size=128, boundary_margin=0, depth_width=256, mirror_analysis=False,
                                 component_cleanup=False, point_reduction=0, format='las', cache_cleanup=mode))
    report = execute(config, lambda: Backend(), listen=False)
    identity = json.loads((tmp_path / 'out' / 'run_config.json').read_text(encoding='utf-8'))['identity']
    assert (tmp_path / 'out' / 'report.json').is_file()
    assert list((tmp_path / 'out').glob('*.las'))
    assert report['cache_cleanup']['mode'] == mode
    assert report['cache_cleanup']['freed_bytes'] > 0
    assert not (cache / 'scan_store' / report['input_sha256']).exists()
    assert not (cache / 'votes' / identity).exists()
    assert not (cache / 'geometry' / report['input_sha256']).exists()
    assert bool(list((cache / 'seg').glob('*.npz'))) is keep_masks
    assert foreign.is_dir()
    assert all((cache / name).exists() for name in ('venv', 'pip', 'launcher.json'))


def test_abort_and_error_keep_cache_for_resume(tmp_path, monkeypatch):
    source = tmp_path / 'abort.e57'; small_project(source)
    preset = json.loads(Path('presets/default.json').read_text(encoding='utf-8'))
    preset['classes'] = [item for item in preset['classes'] if item['id'] == 'wall']
    preset_path = tmp_path / 'preset.json'; preset_path.write_text(json.dumps(preset), encoding='utf-8')
    model = tmp_path / 'model.pt'; model.write_bytes(b'synthetic backend')
    cache = tmp_path / 'cache'
    config = resolve_config(dict(source=str(source), cache=str(cache), output=str(tmp_path / 'out'),
                                 preset=str(preset_path), model=str(model), profile='fast', scene_mode='indoor',
                                 view_size=128, boundary_margin=0, depth_width=256, mirror_analysis=False,
                                 component_cleanup=False, point_reduction=0, format='las', cache_cleanup='all'))
    monkeypatch.setattr('e57cls.pipeline.run.write_cloud', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('output failure')))
    with pytest.raises(RuntimeError, match='output failure'):
        execute(config, lambda: Backend(), listen=False)
    assert (cache / 'scan_store' / file_hash(source)).is_dir()
    assert list((cache / 'votes').glob('*'))
    assert list((cache / 'seg').glob('*.npz'))


def test_committed_votes_discard_large_scan_intermediates(tmp_path):
    source = tmp_path / 'intermediates.e57'; small_project(source)
    preset = json.loads(Path('presets/default.json').read_text(encoding='utf-8'))
    preset['classes'] = [item for item in preset['classes'] if item['id'] == 'wall']
    preset_path = tmp_path / 'preset.json'; preset_path.write_text(json.dumps(preset), encoding='utf-8')
    model = tmp_path / 'model.pt'; model.write_bytes(b'synthetic backend')
    cache = tmp_path / 'cache'
    config = resolve_config(dict(source=str(source), cache=str(cache), output=str(tmp_path / 'out'),
                                 preset=str(preset_path), model=str(model), profile='fast', scene_mode='indoor',
                                 view_size=128, boundary_margin=0, depth_width=256, mirror_analysis=True,
                                 component_cleanup=False, point_reduction=0, format='las', cache_cleanup='keep_all'))
    execute(config, lambda: Backend(), listen=False)
    identity = json.loads((tmp_path / 'out' / 'run_config.json').read_text(encoding='utf-8'))['identity']
    work = cache / 'votes' / identity / 'scan_0001'
    assert not (work / 'weights.npy').exists()
    assert not (work / 'mirror_rep.npy').exists()
