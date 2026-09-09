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
    def test_interactive_command_tapes_are_reproducible_bounded_and_ramped(self):
        from sim2sim.research.schedules import random_schedule,scheduled_command
        for skill in ("walking","roller"):
            task=TASKS[skill];a=random_schedule(task,62000);b=random_schedule(task,62000)
            np.testing.assert_array_equal([x[0] for x in a],[x[0] for x in b])
            np.testing.assert_array_equal([x[1] for x in a],[x[1] for x in b])
            values=np.array([scheduled_command(a,t) for t in np.arange(0,task.seconds,.02)])
            self.assertEqual(values.shape[1],13)
            self.assertLessEqual(values[:,0].max(),.6 if skill=="roller" else .4)
            self.assertGreaterEqual(values[:,0].min(),-.3)
            self.assertLessEqual(np.abs(values[:,2]).max(),1.)
            np.testing.assert_array_equal(values[:,3:],0.)
            if skill=="roller":np.testing.assert_array_equal(values[:,1],0.)
            for i,(t,target) in enumerate(a[1:],1):
                np.testing.assert_allclose(scheduled_command(a,t)[:3],a[i-1][1],atol=1e-6)
                np.testing.assert_allclose(scheduled_command(a,t+.1)[:3],target,atol=1e-6)

    def test_tracking_variance_changes_tolerance_without_changing_task_command(self):
        w=fake_world("walking");w.executed_command[2]=.4
        broad=Objective(w).compute()[2]["yaw"]
        narrow=Objective(w,params={"yaw_variance":.04}).compute()[2]["yaw"]
        self.assertLess(narrow,broad)
        with self.assertRaises(ValueError):Objective(w,params={"yaw_variance":0})

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

    def test_excess_rotation_penalty_is_bounded(self):
        w=fake_world("roulade");r=Objective(w,{"over_rotation":5})
        r.net=100.;r.frontier=2*np.pi
        self.assertEqual(r.compute()[2]["over_rotation"],-5.)

    def test_fall_then_recovery_is_not_a_clean_kick(self):
        rows=[]
        for k in range(250):
            fallen=80<k<130
            rows.append(dict(time=.02*(k+1),z=.03 if fallen else .115,xy=np.zeros(2),up=-1. if fallen else 1.,
                 tilt=180. if fallen else 0.,yaw=0.,vel=np.zeros(3),gyro=np.zeros(3),
                 contact=np.zeros(2) if fallen else np.ones(2),cmd=np.zeros(13),actions=np.zeros(14),q=np.zeros(14),mouth_z=.1,mouth_down=0.,
                 head_contact=False,head_up=1.,supported=True,lateral_z=0.,ball_pos=np.array([k*.004,0,.035]),ball_vel=np.array([.2,0,0]),correct_kick=k==1,wrong_kick=False))
        result=summarize(TASKS["kick_left"],rows,np.array([1,0]))
        self.assertTrue(result["final_standing"])
        self.assertTrue(result["correct_foot_contact"])
        self.assertFalse(result["success"])

    def test_slow_cumulative_idle_drift_is_not_a_stop(self):
        rows=[]
        for k in range(500):
            rows.append(dict(time=.02*(k+1),z=.115,xy=np.array([k*.0006,0]),up=1.,tilt=0.,
                yaw=k*.001,vel=np.array([.03,0,0]),gyro=np.array([0,0,.05]),contact=np.ones(2),
                cmd=np.zeros(13),actions=np.zeros(14),head_contact=False))
        result=summarize(TASKS["walking"],rows,np.array([1,0]))
        self.assertLess(result["vel_err_1s"],.1)
        self.assertLess(result["yaw_err_1s"],.15)
        self.assertGreater(result["idle_displacement"],.25)
        self.assertGreater(result["idle_yaw_drift_deg"],20)
        self.assertFalse(result["success"])

    def test_roll_must_finish_facing_its_original_direction(self):
        rows=[]
        for k in range(250):
            rows.append(dict(time=.02*(k+1),z=.115,xy=np.zeros(2),up=-1. if k==30 else 1.,
                tilt=180. if k==30 else 0.,yaw=0.,vel=np.zeros(3),gyro=np.array([0,2*np.pi/5,0]),
                contact=np.ones(2),cmd=np.zeros(13),actions=np.zeros(14),head_contact=k==20,
                head_up=-1.,supported=True,lateral_z=0.))
        self.assertTrue(summarize(TASKS["roulade"],rows,np.array([1,0]))["success"])
        rows[-1]["yaw"]=np.pi
        result=summarize(TASKS["roulade"],rows,np.array([1,0]))
        self.assertTrue(result["final_standing"])
        self.assertTrue(result["single_revolution"])
        self.assertFalse(result["success"])

class RealTelemetry(unittest.TestCase):
    def test_body_transfer_velocity_matches_com_jacobian(self):
        import mujoco
        w=World(TASKS["roulade"],"mujoco")
        try:
            w.reset(72000,randomize=False);m,d=w.mj.model,w.mj.data
            d.qvel[:]=np.random.default_rng(72000).normal(0,2,m.nv)
            mujoco.mj_forward(m,d)
            reports=w.mj.body_poses_mujoco()
            for b in reports:
                bid=w.meta[b["name"]];jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
                mujoco.mj_jacBodyCom(m,d,jp,jr,bid)
                np.testing.assert_allclose(b["linvel"],jp@d.qvel,atol=1e-10)
                np.testing.assert_allclose(b["angvel"],jr@d.qvel,atol=1e-10)
            w.state=w.mj._state();f=w.measure(reset=True)
            jp=np.zeros((3,m.nv));jr=np.zeros_like(jp)
            mujoco.mj_jacBodyCom(m,d,jp,jr,w.mj.base_body_id)
            np.testing.assert_allclose(f["vel"],jp@d.qvel,atol=1e-10)
        finally:w.close()

    def test_interactive_tapes_match_across_backends_and_survive_handoff(self):
        worlds=[World(TASKS["walking"],b) for b in ("mujoco","godot")]
        try:
            for w in worlds:w.reset(62001,"random_seq");w.enter_from_standing()
            for _ in range(35):
                np.testing.assert_array_equal(worlds[0].obs()[48:],worlds[1].obs()[48:])
                for w in worlds:w.step(np.zeros(14))
            for w in worlds:
                w.reset(62002,"walk_025")
                np.testing.assert_array_equal(w.obs()[48:],command(w.task,0,"walk_025"))
        finally:
            for w in worlds:w.close()

    def test_source_play_wheel_friction_is_explicit_and_reference_only(self):
        import mujoco
        a=World(TASKS["roller"],"mujoco");b=World(TASKS["roller"],"mujoco",reference_profile="source_play")
        try:
            for j in range(a.mj.model.njnt):
                name=mujoco.mj_id2name(a.mj.model,mujoco.mjtObj.mjOBJ_JOINT,j) or ""
                if name.startswith("passive_"):
                    adr=a.mj.model.jnt_dofadr[j]
                    self.assertEqual(a.mj.model.dof_frictionloss[adr],0.)
                    self.assertEqual(b.mj.model.dof_frictionloss[adr],.003)
            self.assertNotEqual(a.physics,b.physics)
        finally:a.close();b.close()

    def test_roll_start_preserves_physical_pose_and_action_history(self):
        from sim2sim.research.models import NativeAnchor
        source=World(TASKS["roulade"],"mujoco");target=World(TASKS["roulade"])
        try:
            obs=source.reset(40000);policy=NativeAnchor(TASKS["roulade"].source)
            obj=Objective(source)
            for _ in range(45):
                obs=source.step(policy(obs[None])[0]);obj.compute()
            target.reset(7)
            target.reset_from_roll_state(source.mj.data.qpos,source.mj.data.qvel,
                source.last,source.heading,[obj.net,obj.frontier,obj.pivot,obj.inverted])
            np.testing.assert_allclose(target.state.q,source.state.q,atol=2e-5)
            np.testing.assert_allclose(target.state.base_pos,source.state.base_pos,atol=2e-6)
            np.testing.assert_array_equal(target.obs()[34:48],source.last)
            for name,body in source.features["bodies"].items():
                np.testing.assert_allclose(target.features["bodies"][name]["linvel"],body["linvel"],atol=2e-6)
            self.assertEqual(Objective(target).net,obj.net)
            target.step(policy(target.obs()[None])[0])
            self.assertTrue(np.isfinite(target.state.q).all())
            target.reset(8)
            self.assertEqual(Objective(target).net,0.)
            self.assertIsNone(target.roll_start)
        finally:source.close();target.close()

    def test_real_handoff_preserves_history_and_counted_time(self):
        w=World(TASKS["kick_left"])
        try:
            w.reset(23);obs=w.enter_from_standing()
            self.assertEqual(w.t,0.)
            self.assertGreater(np.linalg.norm(w.last),.01)
            np.testing.assert_array_equal(obs[34:48],w.last)
            self.assertGreater(w.features["z"],.09)
            self.assertIsNotNone(w.pending_ball)
            from sim2sim.research.models import NativeAnchor
            w.step(NativeAnchor(TASKS["kick_left"].source)(obs[None])[0])
            self.assertAlmostEqual(w.t,.02)
            self.assertIsNone(w.pending_ball)
            self.assertLess(np.linalg.norm(w.features["ball_pos"][:2]-w.features["xy"]),.2)
        finally:w.close()

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
    def test_adapted_native_anchor_can_be_adapted_again(self):
        import onnx
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as td:
            source=TASKS["walking"].source
            first=Policy(source,"plain");first.task_name="walking"
            intermediate=export_policy(first,Path(td)/"first.onnx")
            second=Policy(intermediate,"residual",template=source);second.task_name="walking"
            with torch.no_grad():second.delta.net[-1].bias.add_(.01)
            final=export_policy(second,Path(td)/"second.onnx")
            props=onnx.load(final).metadata_props
            self.assertEqual(len({p.key for p in props}),len(props))
            self.assertTrue(parity(second,final,n=1000)["passed"])

    def test_command_adapter_is_a_single_onnx_with_exact_parity(self):
        from sim2sim.research.conditioning import adapt,parity as conditioning_parity
        with tempfile.TemporaryDirectory() as td:
            matrix=np.array([[2,0,.5],[0,1,0],[0,0,1.5]],np.float32)
            source=TASKS["walking"].source;dest=adapt(source,Path(td)/"policy.onnx",matrix)
            self.assertTrue(conditioning_parity(source,dest,matrix,n=1000)["passed"])

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
