import json
from pathlib import Path
import numpy as np
from e57cls.scene.presets import validate_preset
from e57cls.scene.detect import decide, horizontal_openness
from e57cls.scene.resolve import name_hint, resolve_neighbours, resolve_scan_scene
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
    assert 105 in outdoor and 105 in indoor and outdoor[105]['scene'] == 'both'
    assert 'tree trunk' not in outdoor[5]['prompts']


def test_plane_fit_large_origin_and_outliers():
    rng = np.random.default_rng(9)
    points = np.column_stack((rng.uniform(-2,2,300), rng.uniform(-2,2,300), np.full(300,3.)))
    points[:40,2] = 10
    points += [2700000, 1200000, 400]
    plane = fit_plane(points, horizontal=True)
    assert plane['inlier_fraction'] > .8
    assert plane['residual_p95'] < 1e-6


def test_scene_name_hints_and_resolution():
    cases = {'01_Aussen 001': 'outdoor', '03_UG1 040': 'indoor', '1OG': 'indoor',
             '2.OG': 'indoor', 'EG_12': 'indoor', 'Innenhof 3': 'outdoor',
             'Außenbereich': 'outdoor', 'Weg 5': None, 'Steg': None, '251004_095257': None}
    for value, expected in cases.items():
        hint = name_hint({'name': value}, DEFAULTS)
        assert (hint or {}).get('scene') == expected
    assert name_hint({'rlms:scanposGuid': '/Aussenanlage/SCANS/EG 01'}, DEFAULTS)['scene'] == 'indoor'
    assert name_hint({'name': 'Aussen_EG'}, DEFAULTS)['ambiguous']
    assert name_hint({'name': 'Aussen'}, {**DEFAULTS, 'scene_name_hints': False}) is None
    detected = {'scene': 'indoor', 'metrics': {}}
    resolved = resolve_scan_scene(detected, {'rlms:scanposGuid': '/P/01_Aussen'}, DEFAULTS)
    assert resolved['scene'] == 'outdoor' and resolved['source'] == 'name' and resolved['conflict']
    assert resolve_scan_scene({'scene': 'covered_outdoor', 'metrics': {}}, {}, DEFAULTS)['class_mode'] == 'outdoor'
    assert resolve_scan_scene({'scene': 'unsure', 'metrics': {}}, {}, DEFAULTS)['class_mode'] == 'both'


def test_scene_openness_and_neighbours():
    config = {**DEFAULTS, 'scene_openness_sectors': 36}
    empty = horizontal_openness(np.empty((0, 3)), config)
    assert empty['horizontal_openness'] == 1
    angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
    closed = np.vstack([np.column_stack((3*np.cos(angles), 3*np.sin(angles),
                                         np.full(len(angles), 3*np.tan(np.radians(elevation)))))
                        for elevation in (2.5, 7.5, 12.5, 17.5, 22.5)])
    assert horizontal_openness(closed, config)['horizontal_openness'] < .1
    records, warnings = resolve_neighbours(
        [{'scan': 2, 'scene': 'unsure', 'source': 'pending'}, {'scan': 1, 'scene': 'outdoor', 'source': 'data'}],
        {1: [0, 0, 0], 2: [1, 0, 0]}, DEFAULTS)
    assert records[0]['scene'] == 'outdoor' and warnings[0]['code'] == 'scene_neighbour_resolved'


def _room_returns(missing=(), windows=()):
    """Create first-return samples for a 6 x 6 x 3 m room around the scanner."""
    azimuth = np.linspace(0, 2*np.pi, 720, endpoint=False)
    elevation = np.radians(np.linspace(2.5, 22.5, 5))
    rays = []
    for angle in azimuth:
        direction = np.array([np.cos(angle), np.sin(angle)])
        for angle_up in elevation:
            horizontal = np.cos(angle_up)
            candidates = []
            for axis in range(2):
                if direction[axis]:
                    bound = 3 if direction[axis] > 0 else -3
                    distance = bound / (horizontal * direction[axis])
                    if distance > 0:
                        point = np.array([distance*horizontal*direction[0], distance*horizontal*direction[1],
                                          distance*np.sin(angle_up)])
                        if abs(point[1-axis]) <= 3.000001:
                            side = ('east', 'west')[int(direction[0] < 0)] if axis == 0 else ('north', 'south')[int(direction[1] < 0)]
                            candidates.append((distance, side, point))
            distance, side, point = min(candidates, key=lambda item: item[0])
            if side in missing:
                point *= 25 / max(abs(point[0]), abs(point[1]))
            elif side in windows and -.6 <= point[2] <= .9:
                point *= 30 / max(abs(point[0]), abs(point[1]))
            rays.append(point)
    return np.asarray(rays)


def test_horizontal_openness_distinguishes_roof_opening_from_windows():
    config = {**DEFAULTS, 'scene_openness_sectors': 72}
    grid = np.linspace(-3, 3, 80)
    roof = np.array([[x, y, 3.] for x in grid for y in grid])
    ground = np.array([[x, y, -1.5] for x in grid for y in grid])
    assert horizontal_openness(np.vstack((roof, ground)), config)['horizontal_openness'] >= .8
    missing_wall = horizontal_openness(_room_returns(missing={'north'}), config)
    assert .2 <= missing_wall['horizontal_openness'] <= .5
    window_room = horizontal_openness(_room_returns(windows={'north'}), config)
    corner_windows = horizontal_openness(_room_returns(windows={'north', 'east'}), config)
    assert window_room['horizontal_openness'] < .1
    assert corner_windows['horizontal_openness'] < .1


def test_openness_decisions_keep_legacy_metrics_unchanged():
    base = dict(sky_fraction=.2, open_sky_fraction=.2, ceiling_fraction=.4, overhead_plane=False,
                upper_coverage=.2, upper_return_fraction=.9)
    assert decide({**base, 'horizontal_openness': .6}, DEFAULTS) == 'covered_outdoor'
    mixed = {**base, 'ceiling_fraction': .2, 'horizontal_openness': .1}
    assert decide(mixed, DEFAULTS) == 'indoor'
    unsure = {**base, 'sky_fraction': .05, 'ceiling_fraction': 0, 'horizontal_openness': .1}
    assert decide(unsure, DEFAULTS) == 'indoor'
    assert decide({k: v for k, v in base.items() if k != 'upper_return_fraction'}, DEFAULTS) == 'unsure'
