"""Regression checks for task semantics, real telemetry and deployable actors."""
import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from sim2sim.research.tasks import TASKS,command
from sim2sim.research.models import Policy,export_policy,parity
from sim2sim.research.world import World
from sim2sim.research.rewards import Objective,EXTRA_DIM
from sim2sim.research.evaluate import summarize

def fake_world(name):
    f=dict(z=.115,xy=np.zeros(2),rot=np.eye(3),yaw=0.,up=1.,tilt=0.,
           vel=np.zeros(3),gyro=np.zeros(3),contact=np.ones(2),foot_pos=np.zeros((2,3)),
           foot_vel=np.zeros((2,3)),mouth_pos=np.array([.08,0,.17]),mouth_down=.5,
           head_contact=False,head_up=1.,supported=True,ball_pos=np.array([.09,.042,.035]),
           ball_vel=np.zeros(3),kick_contacts=[])
    w=SimpleNamespace(task=TASKS[name],features=f,t=.02,home=np.zeros(14),last=np.zeros(14),old_last=np.zeros(14),heading=np.array([1.,0.]))
    w.state=SimpleNamespace(q=np.zeros(14),qd=np.zeros(14),base_pos=np.array([0,0,.115]))
    w.executed_command=command(w.task,0)
    return w

class TaskSemantics(unittest.TestCase):
    def test_phase_commands_and_posture_are_not_velocity(self):
        c=command(TASKS["roller_crouch"],1.25)
        np.testing.assert_allclose(c[:2],[0,1],atol=1e-6)
        w=fake_world("sitstand");obj=Objective(w)
        first=obj.compute()[0];w.features["vel"][:]=[1.,0,0]
        self.assertEqual(obj.compute()[0],first)

    def test_inverted_is_not_upright(self):
        w=fake_world("standing");r=Objective(w);upright=r.compute()[2]["upright"]
        w.features["up"]=-1.;w.features["tilt"]=180.
        reward,terminated,terms=r.compute()
        self.assertGreater(upright,terms["upright"]+1.9)
        self.assertTrue(terminated)

    def test_kick_requires_real_correct_foot_contact(self):
        w=fake_world("kick_left");r=Objective(w)
        w.features["foot_vel"][0]=[2,0,1]
        terms=r.compute()[2]
        self.assertEqual(terms["touch"],0);self.assertEqual(terms["ball_forward"],0)
        w.features["kick_contacts"]=["ankle_left"];w.features["ball_vel"][0]=.6
        terms=r.compute()[2]
        self.assertGreater(terms["touch"],0);self.assertGreater(terms["ball_forward"],0)
        self.assertEqual(r.extra().shape,(EXTRA_DIM,))

    def test_rocking_does_not_count_as_full_roll(self):
        rows=[]
        for k in range(250):
            rows.append(dict(time=.02*(k+1),z=.115,xy=np.zeros(2),up=-1. if k==30 else 1.,
                 tilt=180. if k==30 else 0.,yaw=0.,vel=np.zeros(3),gyro=np.array([0,2. if k%20<10 else -2.,0]),
                 contact=np.ones(2),cmd=np.zeros(13),actions=np.zeros(14),q=np.zeros(14),mouth_z=.1,mouth_down=0.,
                 head_contact=k==20,head_up=-1.,supported=True,lateral_z=0.,ball_pos=np.zeros(3),ball_vel=np.zeros(3),correct_kick=False,wrong_kick=False))
        result=summarize(TASKS["roulade"],rows,np.array([1,0]))
        self.assertTrue(result["ordered_roll_events"])
        self.assertLess(result["supported_forward_rotation"],.5)
        self.assertFalse(result["success"])

    def test_roll_reward_stops_at_one_revolution(self):
        w=fake_world("roulade");r=Objective(w);r.net=7.;r.frontier=2*np.pi;r.pivot=True
        w.features["gyro"][1]=4.
        self.assertEqual(r.compute()[2]["forward_progress"],0)

class RealTelemetry(unittest.TestCase):
    def test_mouth_is_actual_site_and_ball_is_free(self):
        import mujoco
        w=World(TASKS["kick_left"],"mujoco")
        try:
            w.reset(11,randomize=False)
            self.assertEqual(w.mj.model.nu,14)
            self.assertIn("ball",w.features["bodies"])
            np.testing.assert_allclose(w.features["mouth_pos"],w.mj.data.site_xpos[w.site_id],atol=1e-12)
            bid=w.meta["ball"];jid=w.mj.model.body_jntadr[bid]
            self.assertEqual(w.mj.model.jnt_type[jid],mujoco.mjtJoint.mjJNT_FREE)
        finally:w.close()

    def test_godot_reports_distinct_ball_and_ground_contact(self):
        w=World(TASKS["kick_left"])
        try:
            w.reset(11,randomize=False)
            for _ in range(10):w.step(np.zeros(14))
            ball=w.features["bodies"]["ball"]
            self.assertTrue(ball["ground_contact"])
            self.assertTrue(all(c["ground"] for c in ball["contact_events"]))
            self.assertEqual(w.features["kick_contacts"],[])
            saved=w.state.extra["raw"].pop("body_states")
            with self.assertRaises(RuntimeError):w.measure()
            w.state.extra["raw"]["body_states"]=saved
        finally:w.close()

class Deployment(unittest.TestCase):
    def test_nonzero_adaptation_exports_without_relaxed_tolerance(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            for variant in ("plain","anchor","residual"):
                torch.manual_seed(7);p=Policy(TASKS["kick_left"].source,variant)
                with torch.no_grad():
                    for v in p.delta.net.parameters():v.add_(torch.randn_like(v)*1e-5)
                export=export_policy(p,Path(tmp)/(variant+".onnx"))
                result=parity(p,export,n=1000)
                self.assertTrue(result["passed"],(variant,result))

if __name__=="__main__":unittest.main()
