import json
from pathlib import Path

import laspy
import pytest

from e57cls.files import file_hash
from e57cls.pipeline.config import resolve_config
from e57cls.pipeline.reclassify import reclassify
from e57cls.pipeline.run import execute
from test_pipeline import Backend, small_project


def _classified_run(tmp_path):
    source = tmp_path / 'source.e57'
    small_project(source)
    preset = json.loads(Path('presets/default.json').read_text(encoding='utf-8'))
    preset['classes'] = [item for item in preset['classes'] if item['id'] == 'wall']
    preset['scene_resolved_pairs'] = []
    preset_path = tmp_path / 'preset.json'
    preset_path.write_text(json.dumps(preset), encoding='utf-8')
    model = tmp_path / 'model.pt'
    model.write_bytes(b'synthetic backend')
    config = resolve_config(dict(source=str(source), cache=str(tmp_path / 'cache'), output=str(tmp_path / 'original'),
                                 preset=str(preset_path), model=str(model), profile='fast', scene_mode='indoor',
                                 view_size=128, boundary_margin=0, depth_width=256, mirror_analysis=False,
                                 component_cleanup=False, geometry_fill=False, point_reduction=0, format='las',
                                 cache_cleanup='keep_all'))
    execute(config, Backend, listen=False)
    return config


def test_reclassify_reproduces_output_and_keeps_original_votes(tmp_path):
    config = _classified_run(tmp_path)
    manifest = json.loads((Path(config['output']) / 'run_config.json').read_text(encoding='utf-8'))
    votes = Path(config['cache']) / 'votes' / manifest['identity'] / 'votes.sqlite'
    before = file_hash(votes)

    report = reclassify(Path(config['output']), tmp_path / 'again')

    original = laspy.read(next(Path(config['output']).glob('*.las')))
    rerun = laspy.read(next((tmp_path / 'again').glob('*.las')))
    assert report['output']['class_counts']
    assert original.points.array.tobytes() == rerun.points.array.tobytes()
    assert file_hash(votes) == before


def test_reclassify_override_applies_and_segmenting_override_is_rejected(tmp_path):
    config = _classified_run(tmp_path)
    allowed = tmp_path / 'allowed.json'
    allowed.write_text(json.dumps({'min_observations': 2}), encoding='utf-8')
    report = reclassify(Path(config['output']), tmp_path / 'changed', allowed)
    assert report['output']['class_counts'] == {'1': 121}
    denied = tmp_path / 'denied.json'
    denied.write_text(json.dumps({'voxel_size': .1}), encoding='utf-8')
    with pytest.raises(ValueError, match='Nicht erlaubte'):
        reclassify(Path(config['output']), tmp_path / 'denied', denied)
