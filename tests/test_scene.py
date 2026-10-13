import json
from pathlib import Path
import numpy as np
from e57cls.scene.presets import validate_preset
from e57cls.scene.detect import decide
from e57cls.geometry.planes import fit_plane
from e57cls.pipeline.config import DEFAULTS


def test_scene_thresholds_keep_ambiguity():
    base = dict(sky_fraction=.2, open_sky_fraction=.2, ceiling_fraction=.01, overhead_plane=False, upper_coverage=1.)
    assert decide(base, DEFAULTS) == 'outdoor'
    assert decide({**base, 'sky_fraction':0, 'open_sky_fraction':0, 'ceiling_fraction':.5}, DEFAULTS) == 'indoor'
    assert decide({**base, 'ceiling_fraction':.2}, DEFAULTS) == 'mixed'
    assert decide({**base, 'sky_fraction':.05}, DEFAULTS) == 'unsure'
    assert decide({**base, 'upper_coverage':.2}, DEFAULTS) == 'unsure'


def test_indoor_supported_ceiling_and_sky_through_window():
    base = dict(sky_fraction=.04,open_sky_fraction=.01,ceiling_fraction=.21,
                upper_return_fraction=.94,upper_coverage=.9,overhead_plane=False)
    assert decide(base,DEFAULTS) == 'indoor'
    assert decide({**base,'sky_fraction':.25,'open_sky_fraction':.03},DEFAULTS) == 'indoor'
    assert decide({**base,'ceiling_fraction':0,'sky_fraction':0},DEFAULTS) == 'unsure'
    from e57cls.pipeline.run import select_classes
    preset=json.loads(Path('presets/default.json').read_text(encoding='utf-8'))
    assert all(c['scene']!='outdoor' for c in select_classes(preset,'unsure'))
    clutter=next(c for c in select_classes(preset,'indoor') if c['code']==76)
    assert {'suitcase','clothes','toys'} <= set(clutter['prompts'])


def test_preset_validation_and_code_map():
    p = json.loads(Path('presets/default.json').read_text(encoding='utf-8'))
    assert validate_preset(p) == ([], ['reserved_las_code:sanitary_fixtures:63'])
    p['classes'][1]['prompts'].append(p['classes'][0]['prompts'][0])
    assert any(e.startswith('prompt_conflict') for e in validate_preset(p)[0])
    p['code_map'] = {'1': 0}
    assert 'special_codes_cannot_be_remapped' in validate_preset(p)[0]


def test_added_classes_are_scene_scoped_and_have_english_prompts():
    from e57cls.pipeline.run import select_classes
    p=json.loads(Path('presets/default.json').read_text(encoding='utf-8'))
    indoor={c['code']:c for c in select_classes(p,'indoor')}
    outdoor={c['code']:c for c in select_classes(p,'outdoor')}
    assert 63 in indoor and 63 not in outdoor
    assert {'toilet','bathtub','washbasin'} <= set(indoor[63]['prompts'])
    assert 75 in outdoor and 75 not in indoor and 'manhole cover' in outdoor[75]['prompts']
    assert 95 in outdoor and 95 not in indoor and 'tree trunk' in outdoor[95]['prompts']
    assert 'tree trunk' not in outdoor[5]['prompts']


def test_plane_fit_large_origin_and_outliers():
    rng = np.random.default_rng(9)
    points = np.column_stack((rng.uniform(-2,2,300), rng.uniform(-2,2,300), np.full(300,3.)))
    points[:40,2] = 10
    points += [2700000, 1200000, 400]
    plane = fit_plane(points, horizontal=True)
    assert plane['inlier_fraction'] > .8
    assert plane['residual_p95'] < 1e-6
