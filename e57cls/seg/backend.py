"""SAM 3 raw grounding adapter, adapted from the inspected msam3 backend.

Provenance: THIRD_PARTY_NOTICES.md and docs/reuse_map.md.
"""

from dataclasses import dataclass
import gc
import logging
from pathlib import Path

import numpy as np

SAM_REVISION = "0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc"
BACKEND_VERSION = "e57cls-grounding-1"
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptClass:
    id: str
    prompts: tuple[str, ...]
    instance_threshold: float = .3


class SAM3Backend:
    def __init__(self, checkpoint: Path, device: str = "cuda:0", batch: int | str = "auto",
                 memory_fraction: float = .85):
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        if not checkpoint.is_file():
            raise FileNotFoundError(f"SAM 3 checkpoint missing: {checkpoint}")
        if not torch.cuda.is_available() or not device.startswith("cuda"):
            raise RuntimeError("SAM 3 requires a CUDA GPU")
        if not 0 < memory_fraction <= 1 or (batch != 'auto' and (not isinstance(batch, int) or batch < 1)):
            raise ValueError("Invalid GPU memory fraction or prompt batch")
        self.torch = torch
        self.device = torch.device(device)
        torch.cuda.set_per_process_memory_fraction(memory_fraction, self.device)
        self.auto_batch = batch == 'auto'
        self.batch = 1 if self.auto_batch else batch
        self.max_auto_batch = 64
        self.memory_fraction = memory_fraction
        self.calls = 0
        self.reductions = []
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = build_sam3_image_model(checkpoint_path=str(checkpoint), device="cpu",
                                           load_from_HF=False, enable_segmentation=True,
                                           enable_inst_interactivity=False, compile=False).to(self.device).eval()
        if self.model.supervise_joint_box_scores:
            raise RuntimeError("Unexpected joint SAM scores; refusing double presence gating")
        self.processor = Sam3Processor(self.model, resolution=1008, device=str(self.device))
        self._text_key = None
        self._text_features = None

    def _probabilities(self, output, shape, threshold):
        import torch.nn.functional as F
        torch = self.torch
        semantic = output["semantic_seg"]
        masks = output["pred_masks"]
        logits = output["pred_logits"]
        presence = output["presence_logit_dec"].sigmoid().float().reshape(-1)
        if semantic.shape[:2] != (1, 1) or masks.ndim != 4 or presence.numel() != 1:
            raise RuntimeError("Unexpected SAM 3 output shapes")
        result = F.interpolate(semantic.float(), shape, mode="bilinear", align_corners=False)[0, 0].sigmoid()
        scores = logits[0, :, 0].sigmoid().float()
        for i in torch.nonzero(scores >= threshold).flatten():
            mask = F.interpolate(masks[:, i:i+1].float(), shape, mode="bilinear", align_corners=False)[0, 0].sigmoid()
            result = torch.maximum(result, mask*scores[i])
        return result*presence[0]

    def predict_tile(self, image, classes: list[PromptClass]):
        torch = self.torch
        while True:
            try:
                return self._predict(image, classes)
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                if self.batch == 1:
                    raise
                old = self.batch
                self.batch = max(1, self.batch//2)
                self.reductions.append({"kind": "prompt_batch", "before": old, "after": self.batch})
                LOG.warning("CUDA OOM: reducing prompt batch from %d to %d", old, self.batch)

    def _grounding(self, state, start, end):
        from sam3.model.data_misc import FindStage
        torch = self.torch
        find = FindStage(img_ids=torch.zeros(end-start, dtype=torch.long, device=self.device),
                         text_ids=torch.arange(start, end, device=self.device),
                         input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
                         input_points=None, input_points_mask=None)
        self.calls += 1
        outputs = self.model.forward_grounding(backbone_out=state["backbone_out"], find_input=find,
                                               geometric_prompt=self.model._get_dummy_prompt(num_prompts=end-start), find_target=None)
        required = ("semantic_seg", "pred_masks", "pred_logits", "presence_logit_dec")
        if any(key not in outputs or outputs[key].shape[0] != end-start for key in required):
            raise RuntimeError("Unexpected SAM 3 batched grounding contract")
        return outputs

    def _choose_auto_batch(self, state, prompt_count):
        """Calibrate once on the actual tile, retaining the configured VRAM reserve."""
        gc.collect()
        self.torch.cuda.empty_cache()
        low, high, best = 1, min(prompt_count, self.max_auto_batch), 1
        while low <= high:
            candidate = (low+high)//2
            try:
                outputs = self._grounding(state, 0, candidate)
                del outputs
                self.torch.cuda.synchronize(self.device)
                self.torch.cuda.empty_cache()
                best, low = candidate, candidate+1
            except self.torch.cuda.OutOfMemoryError:
                gc.collect()
                self.torch.cuda.empty_cache()
                high = candidate-1
        self.batch = best
        self.auto_batch = False
        self.reductions.append({"kind": "auto_prompt_batch", "selected": best,
                                "memory_fraction": self.memory_fraction})
        LOG.info("Auto-selected prompt batch %d with %.0f%% GPU memory budget", best,
                 self.memory_fraction*100)

    def _predict(self, image, classes):
        torch = self.torch
        required = ("semantic_seg", "pred_masks", "pred_logits", "presence_logit_dec")
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=self.dtype):
            state = self.processor.set_image(image)
            result = np.zeros((len(classes), image.height, image.width), np.float32)
            prompts = tuple(p for cls in classes for p in cls.prompts)
            indices = [i for i, cls in enumerate(classes) for _ in cls.prompts]
            if prompts != self._text_key:
                self._text_features = self.model.backbone.forward_text(list(prompts), device=str(self.device))
                self._text_key = prompts
            state["backbone_out"].update(self._text_features)
            if self.auto_batch:
                self._choose_auto_batch(state, len(prompts))
            for start in range(0, len(prompts), self.batch):
                end = min(len(prompts), start+self.batch)
                outputs = self._grounding(state, start, end)
                for offset, index in enumerate(indices[start:end]):
                    single = {k: outputs[k][offset:offset+1] for k in required}
                    probability = self._probabilities(single, (image.height, image.width), classes[index].instance_threshold)
                    np.maximum(result[index], probability.cpu().numpy(), out=result[index])
                    del single, probability
                del outputs
            return result
