"""Offline-only Hugging Face DINOv3 CLS and patch feature extraction."""
from pathlib import Path

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from .image_decode import decode_image


def normalize(x, axis=-1):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


class Encoder:
    def __init__(self, cfg):
        path = Path(cfg.path)
        if not path.is_dir():
            raise FileNotFoundError(f"Local model directory not found: {path}")
        self.path = path.resolve()
        self.processor = AutoImageProcessor.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(path), local_files_only=True).eval()
        self.device = "cuda" if cfg.device == "auto" and torch.cuda.is_available() else ("cpu" if cfg.device == "auto" else cfg.device)
        if cfg.inference_dtype not in ("float32", "float16"):
            raise ValueError("model.inference_dtype must be float32 or float16")
        if cfg.inference_dtype == "float16" and not str(self.device).startswith("cuda"):
            raise ValueError("FP16 model inference requires CUDA; use float32 on CPU")
        self.dtype = torch.float16 if cfg.inference_dtype == "float16" else torch.float32
        self.model.to(device=self.device, dtype=self.dtype)
        self.registers = int(getattr(self.model.config, "num_register_tokens", 0))
        self.patch_size = int(getattr(self.model.config, "patch_size", 16))
        self.hidden_size = int(self.model.config.hidden_size)
        self.batch_size = int(cfg.batch_size)
        if self.batch_size < 1:
            raise ValueError("model.batch_size must be positive")

    def signature(self):
        config_file = self.path / "config.json"
        import hashlib
        import json
        signature = {
            "model_path": str(self.path),
            "model_config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
            "processor": self.processor.to_dict(),
            "register_tokens": self.registers,
            "patch_size": self.patch_size,
            "hidden_size": self.hidden_size,
        }
        return json.loads(json.dumps(signature, sort_keys=True))

    @torch.inference_mode()
    def encode(self, image_bytes):
        images = [decode_image(data) for data in image_bytes]
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device=self.device, dtype=self.dtype)
        tokens = self.model(pixel_values=pixel_values).last_hidden_state.float().cpu().numpy()
        expected = (pixel_values.shape[-2] // self.patch_size) * (pixel_values.shape[-1] // self.patch_size)
        patches = tokens[:, 1 + self.registers:, :]
        if patches.shape[1] != expected:
            raise ValueError(f"Expected {expected} patches, got {patches.shape[1]}; check model and processor")
        return normalize(tokens[:, 0, :]), normalize(patches)
