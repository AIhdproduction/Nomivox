import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pye57
import pytest
from pye57 import libe57 as lib

from e57cls.files import file_hash
from e57cls.pipeline.config import resolve_config
from e57cls.pipeline.run import execute, InputNotClassifiable
from e57cls.pipeline.control import Cancelled


def small_project(path, image_problem=None):
    writer = pye57.E57(str(path), 'w')
    # A wall straight ahead, with modest angular separation between points.
    y, z = np.meshgrid(np.linspace(-1, 1, 11), np.linspace(-1, 1, 11))
    writer.write_scan_raw({'cartesianX': np.full(y.size, 3.), 'cartesianY': y.ravel(),
                          'cartesianZ': z.ravel()}, name='synthetic')
    if image_problem == 'no_images':
        writer.close()
        return
    imf = writer.image_file
    images = writer.root['images2D']
    node = lib.StructureNode(imf)
    images.append(node)
    node.set('guid', lib.StringNode(imf, 'synthetic-image'))
    guid = 'unknown-scan' if image_problem == 'unassociated' else writer.data3d[0]['guid'].value()
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
    Image.new('RGB', (256, 128), (130, 140, 150)).save(stream, format='PNG')
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


@pytest.mark.parametrize('problem', ['no_images', 'unassociated', 'invalid_pose', 'reference_only', 'missing_parameters'])
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
                               view_size=128,boundary_margin=0,depth_width=256,mirror_analysis=True))
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
                                 view_size=128, boundary_margin=0, depth_width=256, mirror_analysis=False))
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
    resumed = execute(config,Backend,listen=False)
    assert resumed['output']['class_counts'] == report['output']['class_counts']
    assert resumed['sam_calls'] < report['sam_calls']
