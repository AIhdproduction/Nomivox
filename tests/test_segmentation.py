import numpy as np
import pytest

from e57cls.seg.backend import PromptClass
from e57cls.seg.cache import MaskCache, MissingMaskCache, SCORE_FLOOR, segment_view
from e57cls.mirror.pipeline import agreed_probability
from e57cls.pipeline.config import resolve_config
from e57cls.pipeline.run import evidence_map
from e57cls.scene.detect import decide
from e57cls.scene.presets import validate_preset


class Backend:
    calls = 0
    def predict_tile(self, image, classes):
        self.calls += 1
        return np.stack([np.full((image.height, image.width), .7+i*.1, np.float32) for i in range(len(classes))])


def test_cache_prompt_granularity_raw_confidence_and_vote_independence(tmp_path):
    cache, backend = MaskCache(tmp_path), Backend()
    image = np.zeros((20, 30, 3), np.uint8)
    valid = np.ones((20, 30), bool)
    valid[:, :2] = False
    classes = [PromptClass("wall", ("wall",)), PromptClass("floor", ("floor",))]
    args = (image, valid, classes, lambda: backend, cache, "imagehash", {"yaw": 0}, "modelhash")
    scores, stats = segment_view(*args, scale=1., tile_size=16)
    assert stats["misses"] == 2 and backend.calls > 1
    assert all(s.dtype == np.float16 for s in scores)
    assert not scores[0][:, :2].any()
    before = backend.calls
    again, stats = segment_view(*args, scale=1., tile_size=16)
    assert backend.calls == before and stats["hits"] == 2
    # Vote thresholds are external, so rerunning any number of decisions needs no SAM call.
    assert (again[0] > .6).sum() > (again[0] > .9).sum()
    classes[1] = PromptClass("floor", ("wood floor",))
    _, stats = segment_view(*args, scale=1., tile_size=16)
    assert stats == {"hits": 1, "misses": 1}


def test_cache_only_segmentation_never_recomputes_missing_masks(tmp_path):
    image = np.zeros((10, 10, 3), np.uint8)
    valid = np.ones((10, 10), bool)
    cache = MaskCache(tmp_path)
    with pytest.raises(MissingMaskCache):
        segment_view(image, valid, [PromptClass('wall', ('wall',))],
                     lambda: pytest.fail('cache-only access must not create a backend'), cache,
                     'imagehash', {'yaw': 0}, 'modelhash', scale=1., tile_size=16, cache_only=True)


def test_score_floor_is_neutral_for_decisions_and_validation(tmp_path):
    rng = np.random.default_rng(20261017)
    scores = [rng.random((37, 31), dtype=np.float32) for _ in range(3)]
    floored = [np.where(score < SCORE_FLOOR, 0, score) for score in scores]
    classes = [
        {'id': 'floor', 'min_mask_conf': .2},
        {'id': 'stairs', 'min_mask_conf': .35},
        {'id': 'wall', 'min_mask_conf': None},
    ]
    config = {'stairs_priority': True, 'stairs_priority_over': ['floor'],
              'stairs_min_confidence': .2, 'mask_confidence': .2, 'boundary_margin': 1}
    before = evidence_map(scores, classes, config)
    after = evidence_map(floored, classes, config)
    np.testing.assert_array_equal(before[0], after[0])
    accepted = before[0] >= 0
    np.testing.assert_array_equal(before[1][accepted], after[1][accepted])
    np.testing.assert_array_equal(agreed_probability(np.stack(scores), 2) >= .2,
                                  agreed_probability(np.stack(floored), 2) >= .2)
    metrics = {'sky_fraction': .22, 'ceiling_fraction': .1, 'open_sky_fraction': .2,
               'upper_return_fraction': .1, 'upper_coverage': .9, 'overhead_plane': False}
    scene_config = {'scene_ceiling_supported': .2, 'scene_upper_returns_indoor': .7,
                    'scene_ceiling_indoor': .3, 'scene_open_sky_indoor_max': .05,
                    'scene_sky_outdoor': .2, 'scene_open_sky_outdoor': .1,
                    'scene_ceiling_outdoor_max': .15}
    assert decide(metrics, scene_config) == decide(dict(metrics), scene_config)

    source = tmp_path / 'input.e57'
    source.write_bytes(b'fixture')
    with pytest.raises(ValueError, match='at least'):
        resolve_config({'source': str(source), 'mask_confidence': SCORE_FLOOR - .01})
    preset = {'schema_version': 1, 'unclassified_code': 1, 'code_map': {'1': 1, '68': 68},
              'scene_resolved_pairs': [],
              'special': [{'id': 'unclassified', 'code': 1, 'color': '#000000'},
                          {'id': 'reflection', 'code': 68, 'color': '#000000'}],
              'classes': [{'id': 'wall', 'code': 100, 'color': '#ffffff', 'scene': 'both',
                           'prompts': ['wall'], 'min_mask_conf': SCORE_FLOOR - .01}]}
    assert any(error.startswith('invalid_threshold:wall') for error in validate_preset(preset)[0])
