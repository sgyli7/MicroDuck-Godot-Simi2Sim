"""Sai release control contract with the independently validated flat actor.

The flat ONNX and free-driving terrain gate are project-owned. Observation
construction, history, IK, heading and motor targets use alpha.3. Explicit
stair courses preserve its published selection and control settings.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
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
        self.flat_policy_sha256 = manifest["onnx_sha256"]

    def command(self, state):
        stair_actor = self.stair_policy
        relevant = self._step_in_wheel_path(state)
        request = state["command"]
        crouch_blocked = (float(request[2]) > .5 and float(request[0]) > .015
                          and np.ptp(state["terrain_heights"]) > .008
                          and self._step_in_wheel_path(state, honor_course=False))
        if crouch_blocked:
            # Stay low at a real edge instead of repeatedly trying a stair gait
            # with insufficient clearance. Turning/reversing away remain live.
            state = dict(state, command=[0., request[1], request[2]])
        # alpha.3 has no selection hook. Gate its optional actor for this one
        # sequential request, without altering raw observations or target math.
        if not relevant:
            self.stair_policy = None
        try:
            result = super().command(state)
        finally:
            self.stair_policy = stair_actor
        result["flat_policy_id"] = self.flat_policy_id
        result["flat_policy_sha256"] = self.flat_policy_sha256
        result["driving_profile"] = "sai-driving-20260913"
        result["terrain_step_relevant"] = relevant
        if crouch_blocked:
            result["stage"] = "crouch_blocked"
        return result

    def _step_in_wheel_path(self, state, honor_course=True):
        if honor_course and (state.get("stair_course") or self.experimental_profile):
            return True
        path = state.get("terrain_path_heights")
        if path is None:
            return True  # Original wire clients retain their release behavior.
        path = np.asarray(path, dtype=float)
        if path.shape != (15,) or not np.isfinite(path).all():
            raise ValueError("Invalid wheel-path terrain scan")
        crouch = float(state["command"][2]) > .5
        # Crouch prioritizes rolling over small joints in the floor. Real
        # 20+ mm steps still activate before the front wheels reach the edge.
        near = path.reshape(5, 3)[:4 if crouch else 5]
        return bool(np.ptp(near) > (.008 if crouch else .004))
