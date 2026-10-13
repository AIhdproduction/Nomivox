import numpy as np

from e57cls.seg.backend import PromptClass
from e57cls.seg.cache import MaskCache, segment_view


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
