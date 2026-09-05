"""ONNX policy wrapper. I/O names match duck-control / infer_policy (obs → actions)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


class OnnxPolicy:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.sess = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        ishape = self.sess.get_inputs()[0].shape
        oshape = self.sess.get_outputs()[0].shape
        self.obs_dim = int(ishape[-1]) if ishape[-1] else 61
        self.act_dim = int(oshape[-1]) if oshape[-1] else 14

    def infer(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        y = self.sess.run([self.output_name], {self.input_name: x})[0]
        return np.asarray(y, dtype=np.float32).reshape(-1)
