import json

from e57cls.tools.scene_eval import evaluate, main


def test_scene_eval_reports_name_hint_accuracy_and_grid(tmp_path, capsys):
    report = {'scans': [
        {'scan': 1, 'scene': 'outdoor', 'scene_resolution': {'data_scene': 'outdoor', 'source': 'data',
         'metrics': {'sky_fraction': .3, 'ceiling_fraction': 0.,
         'upper_coverage': .8, 'overhead_plane': False, 'open_sky_fraction': .2, 'horizontal_openness': .6}}},
        {'scan': 2, 'scene': 'indoor', 'scene_resolution': {'data_scene': 'indoor', 'source': 'data',
         'metrics': {'sky_fraction': 0., 'ceiling_fraction': .7,
         'upper_coverage': .9, 'overhead_plane': True, 'open_sky_fraction': 0., 'horizontal_openness': .1}}},
        {'scan': 9, 'scene': 'outdoor', 'scene_detection': {'scene': 'covered_outdoor', 'source': 'openness',
         'metrics': {'sky_fraction': .3, 'ceiling_fraction': .2,
         'upper_coverage': .8, 'overhead_plane': True, 'open_sky_fraction': .2, 'horizontal_openness': .6}}}],
        'scene_audit': {'totals': {'outdoor': {'wrong_scene_share': 0.}, 'indoor': {'wrong_scene_share': 0.}}}}
    inspect = {'scans': [{'index': 9, 'metadata': {'rlms:scanposGuid': '01_Aussen 009'}},
                         {'index': 1, 'metadata': {'rlms:scanposGuid': '01_Aussen 001'}},
                         {'index': 2, 'metadata': {'rlms:scanposGuid': '03_UG1 040'}}]}
    (tmp_path/'report.json').write_text(json.dumps(report), encoding='utf-8')
    (tmp_path/'inspect_report.json').write_text(json.dumps(inspect), encoding='utf-8')
    result = evaluate(tmp_path, grid=True)
    assert result['final_accuracy'] == 1.
    assert result['data_accuracy'] == 1.
    assert result['data_count'] == 3
    assert result['confusion'][-1]['data'] == 'covered_outdoor'
    assert result['confusion'][-1]['data_normalized'] == 'outdoor'
    assert result['grid']
    main([str(tmp_path), '--grid'])
    assert 'final_accuracy' in capsys.readouterr().out
