"""Sai release control contract with the independently validated flat actor.

The flat ONNX, terrain gate and trained suspension parameters are project-owned. Observation
construction, history and heading use alpha.3. Ascending keeps its step targets;
validated descents use contact-following support.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sai_agent.godot_controller import GodotController
from sim2sim.sai_terrain import step_in_path, descending_in_path
from sim2sim.sai_suspension import Suspension


class MotionController(GodotController):
    def __init__(self, root, stair_profile=None, suspension_profile=None):
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
        profile = suspension_profile if suspension_profile is not None else os.environ.get("SIM2SIM_SAI_SUSPENSION_PROFILE", str(directory / "suspension-v2.json"))
        self.impedance = None
        self.roll_descent = False
        self.suspension = None
        self.suspension_id = None
        if profile != "off":
            suspension = json.loads(Path(profile).read_text())
            if suspension.get("schema_version") not in (1, 2):
                raise ValueError("Unsupported Sai suspension profile")
            if suspension.get("flat_policy_sha256", manifest["onnx_sha256"]) != manifest["onnx_sha256"]:
                raise ValueError("Suspension profile does not match the flat actor")
            if suspension["schema_version"] == 2:
                from sim2sim.sai_compliance import StanceImpedance
                self.impedance = StanceImpedance(self, suspension["parameters"], suspension.get("turn_support_blend", .25))
                self.roll_descent = suspension.get("descent_control") == "contact_following"
                self.suspension = Suspension(suspension["geometry_parameters"])
            else:
                self.suspension = Suspension(suspension["parameters"])
            self.suspension_id = suspension["id"]
        self.flat_policy_id = manifest["id"]
        self.flat_policy_sha256 = manifest["onnx_sha256"]

    def command(self, state):
        stair_actor = self.stair_policy
        relevant = self._step_in_wheel_path(state)
        request = state["command"]
        descending = self.roll_descent and descending_in_path(state) and float(request[0]) > .015
        if descending:
            state = dict(state, command=[min(.16, float(request[0])), request[1], request[2]])
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
        if descending:
            result["stage"] = "descending"
        if crouch_blocked:
            result["stage"] = "crouch_blocked"
        if self.suspension is not None:
            result = self.suspension.apply(result, state)
            result["suspension_profile"] = self.suspension_id
        if self.impedance is not None and len(state.get("wheel_ground_heights", [])) == 4:
            result = self.impedance.apply(result, state)
        return result

    def _step_in_wheel_path(self, state, honor_course=True):
        if self.roll_descent and descending_in_path(state):
            return False
        if honor_course and (state.get("stair_course") or self.experimental_profile):
            return True
        return step_in_path(state, crouch=float(state["command"][2]) > .5)
