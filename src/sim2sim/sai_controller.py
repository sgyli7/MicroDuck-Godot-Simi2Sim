"""Sai release control contract with the independently validated flat actor.

Only the flat ONNX changes. Observation construction, previous-action history,
height IK, heading control, stair selection and all motor targets use alpha.3.
"""
import hashlib
import json
from pathlib import Path

import onnxruntime as ort
from sai_agent.godot_controller import GodotController


class MotionController(GodotController):
    def __init__(self, root, stair_profile=None):
        super().__init__(root, stair_profile=stair_profile)
        directory = Path(__file__).with_name("assets") / "sai"
        manifest = json.loads((directory / "flat-motion-v1.json").read_text())
        policy_path = directory / "flat-motion-v1.onnx"
        if hashlib.sha256(policy_path.read_bytes()).hexdigest() != manifest["onnx_sha256"]:
            raise ValueError("Sai flat motion policy hash mismatch")
        if manifest["observation_size"] != 82 or manifest["action_size"] != 16:
            raise ValueError("Sai flat motion policy contract mismatch")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.policy = ort.InferenceSession(str(policy_path), options, providers=["CPUExecutionProvider"])
        self.flat_policy_id = manifest["id"]

    def command(self, state):
        result = super().command(state)
        result["flat_policy_id"] = self.flat_policy_id
        return result
