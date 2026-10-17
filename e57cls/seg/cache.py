"""Per-class float16 probability caches; vote thresholds never enter identity."""

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from e57cls.files import atomic_json, file_hash
from e57cls.seg.backend import BACKEND_VERSION, SAM_REVISION

SCORE_FLOOR = .2


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def class_identity(image_hash, view, model_hash, cls, scale, tile_size, overlap):
    return {"schema": 2, "score_floor": SCORE_FLOOR, "image_sha256": image_hash, "view": view,
            "model_sha256": model_hash, "sam_revision": SAM_REVISION, "backend": BACKEND_VERSION,
            "prompts": list(cls.prompts), "instance_threshold": cls.instance_threshold,
            "scale": scale, "tile_size": tile_size, "overlap": overlap}


def tile_boxes(width, height, size, overlap):
    if size < 1 or not 0 <= overlap < 1:
        raise ValueError("Invalid tile size or overlap")
    stride = max(1, round(size*(1-overlap)))
    def starts(length):
        last = max(0, length-size)
        return sorted(set([*range(0, last+1, stride), last]))
    for y in starts(height):
        for x in starts(width):
            yield x, y, min(x+size, width), min(y+size, height)


class MaskCache:
    def __init__(self, root: Path):
        self.root = root
        self.used_paths = set()
        root.mkdir(parents=True, exist_ok=True)

    def load(self, identity):
        key = identity_hash(identity)
        try:
            manifest = json.loads((self.root/f"{key}.json").read_text())
            path = self.root/manifest["file"]
            if not path.resolve().is_relative_to(self.root.resolve()):
                return None
            if manifest["identity"] != identity or file_hash(path) != manifest["sha256"]:
                return None
            with np.load(path, allow_pickle=False) as archive:
                score = archive["confidence"]
            if score.dtype != np.float16 or score.ndim != 2 or not np.isfinite(score).all() or np.any((score < 0)|(score > 1)):
                return None
            self.used_paths.update((self.root/f"{key}.json", path))
            return score
        except (OSError, KeyError, ValueError):
            return None

    def save(self, identity, scores, effective_tile):
        key = identity_hash(identity)
        name = f"{key}_{uuid4().hex}.npz"
        path = self.root/name
        stored = np.where(scores < SCORE_FLOOR, 0, scores).astype(np.float16)
        with path.open("wb") as stream:
            np.savez_compressed(stream, confidence=stored)
        manifest = self.root/f"{key}.json"
        atomic_json(manifest, {"identity": identity, "file": name,
                                            "sha256": file_hash(path), "effective_tile_size": effective_tile})
        self.used_paths.update((manifest, path))


def segment_view(image, valid, classes, backend_factory, cache: MaskCache, image_hash,
                 view, model_hash, scale=.75, tile_size=1008, overlap=.2, checkpoint=None):
    if not 0 < scale <= 1:
        raise ValueError("Image scale must be in (0,1]")
    identities = [class_identity(image_hash, view, model_hash, cls, scale, tile_size, overlap) for cls in classes]
    scores = [cache.load(identity) for identity in identities]
    missing = [i for i, score in enumerate(scores) if score is None]
    if missing:
        backend = backend_factory()
        height, width = image.shape[:2]
        size = (max(1, round(width*scale)), max(1, round(height*scale)))
        scaled = Image.fromarray(image).resize(size, Image.Resampling.LANCZOS)
        valid_scaled = np.array(Image.fromarray(valid).resize(size, Image.Resampling.NEAREST), bool)
        effective = tile_size
        while True:
            merged = np.zeros((len(missing), size[1], size[0]), np.float32)
            try:
                for box in tile_boxes(*size, effective, overlap):
                    if checkpoint:
                        checkpoint()
                    if not valid_scaled[box[1]:box[3], box[0]:box[2]].any():
                        continue
                    tile = scaled.crop(box)
                    probabilities = backend.predict_tile(tile, [classes[i] for i in missing])
                    expected = (len(missing), box[3]-box[1], box[2]-box[0])
                    if probabilities.shape != expected or not np.isfinite(probabilities).all():
                        raise ValueError("SAM returned invalid probability maps")
                    np.maximum(merged[:, box[1]:box[3], box[0]:box[2]], probabilities,
                               out=merged[:, box[1]:box[3], box[0]:box[2]])
                break
            except Exception as error:
                import torch
                if not isinstance(error, torch.cuda.OutOfMemoryError) or effective <= 336:
                    raise
                del merged
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                old, effective = effective, max(336, effective//2)
                backend.reductions.append({"kind": "tile_size", "before": old, "after": effective})
                import logging
                logging.getLogger(__name__).warning("CUDA OOM: reducing tile size from %d to %d", old, effective)
        for offset, index in enumerate(missing):
            merged[offset, ~valid_scaled] = 0
            cache.save(identities[index], merged[offset], effective)
            scores[index] = np.where(merged[offset] < SCORE_FLOOR, 0, merged[offset]).astype(np.float16)
        scaled.close()
    return scores, {"hits": len(classes)-len(missing), "misses": len(missing)}
