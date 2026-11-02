"""soniqo/Sidon-ONNX (MIT) wrapped as drop-in stand-ins for the TorchScript modules of our engine."""

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import transformers

ROOT = Path(os.environ.get("VR_EVAL_DATA", Path.home() / ".cache" / "make-normal-voice-evals")) / "sidon-onnx"


class _Fe:
    def __init__(self, path):
        self.s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def __call__(self, feats):
        y = self.s.run(None, {"input_features": feats.numpy().astype(np.float32)})[0]
        return {"last_hidden_state": torch.from_numpy(y)}


class _Dec:
    def __init__(self, path):
        self.s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def __call__(self, f):  # engine passes (1, 1024, T)
        y = self.s.run(None, {"features": f.transpose(1, 2).contiguous().numpy().astype(np.float32)})[0]
        return torch.from_numpy(y)


def load(variant: str):
    pre = transformers.SeamlessM4TFeatureExtractor(feature_size=80, num_mel_bins=80, padding_value=1.0,
                                                   sampling_rate=16000, stride=2)
    return _Fe(ROOT / variant / "sidon-predictor.onnx"), _Dec(ROOT / variant / "sidon-vocoder.onnx"), pre
