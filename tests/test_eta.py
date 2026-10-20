import json

from e57cls.gui.eta import DEFAULT_PROFILE, RunEstimator, load_profile, save_profile


def estimator(**kwargs):
    return RunEstimator(applies={stage: True for stage in DEFAULT_PROFILE}, **kwargs)


def test_initial_estimate_uses_counts_and_defaults():
    value = estimator(unit_counts={'mirror_masks': 10, 'mirror_evidence': 12, 'segmentation': 10})
    value.phase_started('mirror', 0)
    total, maximum, stage, stage_maximum = value.remaining(0)
    assert (total, maximum, stage, stage_maximum) == (440.0, 1800.0, 50.0, 150.0)


def test_recalibration_blends_average_and_converges_maximum():
    value = estimator(unit_counts={'mirror_masks': 0, 'mirror_evidence': 0, 'segmentation': 10})
    value.phase_started('segmentation', 0)
    for scan, timestamp in enumerate((0, 100, 200, 300), 1):
        value.progress('segmentation', timestamp, scan=scan)
    average, maximum = value.per_unit('segmentation')
    assert average == 57.5
    assert maximum == 100.0


def test_current_stage_accounts_for_partial_unit_time():
    value = RunEstimator(applies={'segmentation': True}, unit_counts={'segmentation': 10})
    value.phase_started('segmentation', 0)
    value.progress('segmentation', 0, scan=1)
    total, maximum, stage, stage_maximum = value.remaining(100)
    assert (stage, stage_maximum) == (135.0, 405.0)
    assert (total, maximum) == (255.0, 645.0)


def test_view_progress_uses_variable_view_counts_as_work_units():
    value = RunEstimator(applies={'segmentation': True}, unit_counts={'segmentation': 25})
    value.phase_started('segmentation', 0)
    value.progress('segmentation', 0, scan=1, image=1, view=1, view_count=17)
    _, _, stage, _ = value.remaining(0)
    assert stage == 25*DEFAULT_PROFILE['segmentation']['avg_seconds']
    value.progress('segmentation', 10, scan=1, image=1, view=2, view_count=17)
    assert value.measurements['segmentation'] == [10.0]
    assert value.current_unit == (1, 2)
    value.progress('segmentation', 20, scan=1, image=2, view=1, view_count=8)
    assert value.current_unit == (2, 1)


def test_mirror_mask_view_progress_keeps_the_planned_view_count():
    value = RunEstimator(applies={'mirror_masks': True}, unit_counts={'mirror_masks': 25})
    value.phase_started('mirror', 0)
    value.progress('mirror_masks', 10, image=1, image_count=2, view=1, view_count=17)

    assert value.unit_counts['mirror_masks'] == 25


def test_mirror_stage_transition_uses_last_mask_progress_for_evidence_start():
    value = RunEstimator(applies={'mirror_masks': True, 'mirror_evidence': True},
                         unit_counts={'mirror_masks': 2, 'mirror_evidence': 2})
    value.phase_started('mirror', 0)
    value.progress('mirror_masks', 10, image=1)
    value.progress('mirror_masks', 20, image=2)
    value.progress('mirror_masks', 30, image=2)
    value.progress('mirror_evidence', 50, scan=1, scan_count=2)
    assert value.measurements['mirror_masks'] == [20.0, 10.0]
    assert value.measurements['mirror_evidence'] == [20.0]


def test_resumed_segmentation_uses_remaining_eligible_scan_indices():
    value = RunEstimator(applies={'segmentation': True}, unit_counts={'segmentation': 10})
    value.set_unit_indices('segmentation',range(1,11))
    value.phase_started('segmentation', 0)
    value.progress('segmentation', 0, scan=6)
    _, _, stage, stage_maximum = value.remaining(0)
    assert (stage, stage_maximum) == (75.0, 225.0)


def test_skipped_stages_for_mirror_off_and_filter_only():
    value = RunEstimator(applies={'mirror_masks': False, 'mirror_evidence': False, 'segmentation': False})
    assert value.remaining(0) == (60.0, 120.0, 60.0, 120.0)


def test_profile_save_and_load_round_trip(tmp_path):
    path = tmp_path/'timing_profile.json'
    value = RunEstimator(applies={'segmentation': True, 'slam_scene': True, 'slam_segmentation': True}, unit_counts={'segmentation': 3})
    value.measurements['segmentation'] = [12.0, 18.0, 24.0]
    value.measurements['slam_scene'] = [5.0, 7.0]
    value.measurements['slam_segmentation'] = [8.0, 10.0]
    save_profile(path, value)
    profile = load_profile(path)
    assert profile['segmentation']['avg_seconds'] == 18.0
    assert profile['segmentation']['max_seconds'] == 24.0
    assert profile['segmentation']['units'] == 3
    assert profile['segmentation']['unit'] == 'view'
    assert profile['slam_scene']['avg_seconds'] == 6.0
    assert profile['slam_segmentation']['avg_seconds'] == 9.0
    assert profile['mirror_masks'] == DEFAULT_PROFILE['mirror_masks']


def test_legacy_view_profile_is_ignored_for_view_count_eta(tmp_path):
    path = tmp_path/'timing_profile.json'
    path.write_text(json.dumps({
        'segmentation': {'avg_seconds': 256.6, 'max_seconds': 361.8, 'units': 156},
    }), encoding='utf-8')

    value = RunEstimator(applies={'segmentation': True}, unit_counts={'segmentation': 2652},
                         profile=load_profile(path))
    value.phase_started('segmentation', 0)
    total, _, _, _ = value.remaining(0)

    assert total < 20*60*60


def test_legacy_profile_keeps_unchanged_units_and_uses_realistic_parallel_eta(tmp_path):
    path = tmp_path/'timing_profile.json'
    path.write_text(json.dumps({
        'sam': {'avg_seconds': 275.0, 'max_seconds': 472.0, 'units': 0},
        'mirror_masks': {'avg_seconds': 74.1, 'max_seconds': 105.0, 'units': 156},
        'mirror_evidence': {'avg_seconds': 6.82, 'max_seconds': 78.06, 'units': 157},
        'segmentation': {'avg_seconds': 256.6, 'max_seconds': 361.8, 'units': 156},
        'slam_scene': {'avg_seconds': 0.54, 'max_seconds': 0.98, 'units': 71},
        'slam_segmentation': {'avg_seconds': 682.5, 'max_seconds': 682.5, 'units': 1},
    }), encoding='utf-8')

    profile = load_profile(path)

    assert profile['mirror_evidence'] == {
        'avg_seconds': 6.82, 'max_seconds': 78.06, 'units': 157, 'unit': 'scan',
    }
    assert profile['slam_scene'] == {
        'avg_seconds': 0.54, 'max_seconds': 0.98, 'units': 71, 'unit': 'image',
    }
    for stage in ('sam', 'mirror_masks', 'segmentation', 'slam_segmentation'):
        assert profile[stage] == DEFAULT_PROFILE[stage]

    value = RunEstimator(
        applies={'mirror_evidence': True, 'segmentation': True},
        unit_counts={'mirror_evidence': 157, 'segmentation': 2652},
        profile=profile,
        parallel=True,
    )
    value.phase_started('segmentation', 0)
    total, _, _, _ = value.remaining(0)

    assert 8*60*60 < total < 16*60*60


def test_implausible_view_profile_is_ignored(tmp_path):
    path = tmp_path/'timing_profile.json'
    path.write_text(json.dumps({
        'segmentation': {'avg_seconds': 151.0, 'max_seconds': 200.0, 'units': 10, 'unit': 'view'},
        'mirror_evidence': {'avg_seconds': 6.82, 'max_seconds': 78.06, 'units': 157, 'unit': 'image'},
    }), encoding='utf-8')

    profile = load_profile(path)
    assert profile['segmentation'] == DEFAULT_PROFILE['segmentation']
    assert profile['mirror_evidence'] == DEFAULT_PROFILE['mirror_evidence']


def test_slam_segmentation_progress_records_each_view():
    value = RunEstimator(applies={'slam_segmentation': True}, unit_counts={'slam_segmentation': 3})
    value.phase_started('slam_segmentation', 0)
    value.progress('slam_segmentation', 10, station=1, image=4, view=1)
    value.progress('slam_segmentation', 40, station=1, image=4, view=2)
    value.progress('slam_segmentation', 75, station=2, image=4, view=1)
    value.phase_finished('slam_segmentation', 120)

    assert value.measurements['slam_segmentation'] == [40.0, 35.0, 45.0]


def test_corrupt_profile_is_ignored(tmp_path):
    path = tmp_path/'timing_profile.json'
    path.write_text('{broken', encoding='utf-8')
    assert load_profile(path) == DEFAULT_PROFILE


def test_slam_eta_waits_for_two_views_and_uses_a_learned_profile():
    value = RunEstimator(applies={'slam_scene': True, 'slam_segmentation': True})
    value.set_slam_plan(cameras=71, views_total=1207, stations=19)
    value.phase_started('slam_scene', 0)
    value.progress('slam_scene', 10, image=1, image_count=71)
    assert value.remaining(10) is None
    value.progress('slam_scene', 20, image=2, image_count=71)
    first = value.remaining(20)
    value.progress('slam_scene', 30, image=3, image_count=71)
    second = value.remaining(30)
    assert first is not None and second is not None
    assert second[0] <= first[0]

    learned = RunEstimator(applies={'slam_scene': True, 'slam_segmentation': True}, profile={
        'slam_scene': {'avg_seconds': 5., 'max_seconds': 7., 'units': 71},
        'slam_segmentation': {'avg_seconds': 6., 'max_seconds': 8., 'units': 1207},
    })
    learned.set_slam_plan(cameras=71, views_total=1207, stations=19)
    assert learned.remaining(0) is not None
