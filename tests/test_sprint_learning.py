import unittest
import numpy as np
from test_research import fake_world
from sim2sim.research.rewards import Objective
from sim2sim.research.sprint_tasks import commands
from sim2sim.research.world import World
from sim2sim.motion_control import MotionControl
from types import SimpleNamespace


class SprintLearning(unittest.TestCase):
    def test_ordinary_feedback_override_preserves_sprint_and_legacy_defaults(self):
        state=SimpleNamespace(base_pos=np.zeros(3),base_quat_wxyz=np.array([1.,0,0,0]))
        cmd=np.zeros(13,np.float32);cmd[0]=.3
        controls=[MotionControl({'walk_path_gain':4.,'walk_ordinary_path_gain':6.}),MotionControl({'walk_path_gain':4.})]
        for c in controls:c.command(cmd,state,'walking')
        state.base_pos[1]=.02
        self.assertAlmostEqual(float(controls[0].command(cmd,state,'walking')[1]),-.12,places=6)
        self.assertAlmostEqual(float(controls[0].command(cmd,state,'sprint')[1]),-.08,places=6)
        np.testing.assert_array_equal(controls[1].command(cmd,state,'walking'),controls[1].command(cmd,state,'sprint'))

    def test_entry_fingerprint_detects_models_and_control_changes(self):
        import json,tempfile
        from pathlib import Path
        from sim2sim.research.sprint_entry import SprintEntryBank
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);actor=root/'actor.onnx';actor.write_bytes(b'original')
            bank=root/'bank.json';config=dict(version='sprint_entry_v1',walking=str(actor),sprint=str(actor),twist_limits={})
            bank.write_text(json.dumps(config));before=SprintEntryBank.fingerprint(bank)
            actor.write_bytes(b'candidate');changed=SprintEntryBank.fingerprint(bank)
            self.assertNotEqual(before['walking'],changed['walking']);self.assertEqual(before['config'],changed['config'])
            config['twist_limits']['vmax_ang']=.8;bank.write_text(json.dumps(config))
            self.assertNotEqual(changed['config'],SprintEntryBank.fingerprint(bank)['config'])

    def test_restart_can_capture_current_heading_without_resetting_actor_history(self):
        control=MotionControl({'walk_heading_gain':6.,'walk_reanchor_on_start':True})
        state=SimpleNamespace(base_pos=np.zeros(3),base_quat_wxyz=np.array([1.,0,0,0]))
        idle=np.zeros(13,np.float32);forward=idle.copy();forward[0]=.3
        control.command(forward,state,'walking')
        state.base_quat_wxyz=np.array([np.cos(.2),0,0,np.sin(.2)])
        self.assertLess(float(control.command(idle,state,'walking')[2]),0.)
        restarted=control.command(forward,state,'walking')
        self.assertAlmostEqual(float(restarted[2]),0.)
        self.assertAlmostEqual(control.target_yaw,.4)

    def test_feedback_reward_tracks_corrective_lateral_request(self):
        scores={}
        for version in ['sprint_v1','sprint_v2']:
            scores[version]=[]
            for speed in [0.,-.1]:
                world=fake_world('walking');world.executed_command[1]=-.1
                world.features['vel'][1]=speed
                objective=Objective(world,walking_objective=version)
                objective.smooth[1]=speed
                scores[version].append(objective.compute()[0])
        self.assertGreater(scores['sprint_v1'][0],scores['sprint_v1'][1])
        self.assertLess(scores['sprint_v2'][0],scores['sprint_v2'][1])

    def test_feedback_is_sampled_once_per_physical_state(self):
        world=object.__new__(World)
        world.motion=MotionControl({'walk_path_gain':2.})
        world._command_stamp=None;world.t=0.
        world.state=SimpleNamespace(base_pos=np.array([0.,0.,.115]),base_quat_wxyz=np.array([1.,0,0,0]))
        world.requested_command=lambda: np.array([.35,0.,0.,*([0.]*10)],np.float32)
        first=world.command();np.testing.assert_array_equal(first,world.command())
        world.state.base_pos[1]=.04;world.t=.02
        corrected=world.command();self.assertAlmostEqual(float(corrected[1]),-.08,places=6)
        np.testing.assert_array_equal(corrected,world.command())
        world.requested_command=lambda: np.array([.35,0.,.8,*([0.]*10)],np.float32)
        world.t=.04;turn=world.command()
        self.assertEqual(float(turn[1]),0.);self.assertFalse(world.motion.walk_path_started)

    def test_target_speed_has_progress_gradient_and_overrun_cost(self):
        rewards=[]
        for speed in [0.,.1,.2,.3,.4,.5]:
            world=fake_world('walking');world.executed_command[0]=.4
            world.features['vel'][0]=speed
            objective=Objective(world,walking_objective='sprint_v1')
            objective.smooth[0]=speed
            rewards.append(objective.compute()[0])
        self.assertTrue(all(a<b for a,b in zip(rewards[:4],rewards[1:5])))
        self.assertLess(rewards[5],rewards[4])

    def test_fall_is_terminal_and_heading_is_critic_state(self):
        world=fake_world('walking');objective=Objective(world,walking_objective='sprint_v1')
        world.features['yaw']=.2
        np.testing.assert_allclose(objective.extra()[8:10],[np.sin(-.2),np.cos(-.2)],atol=1e-7)
        world.features['tilt']=61.
        self.assertTrue(objective.compute()[1])
        self.assertFalse(Objective(world).compute()[1])

    def test_curriculum_has_idle_acceleration_both_turns_and_release(self):
        for direction,turn in [('straight',0.),('left',.8),('right',-.8)]:
            tape=commands('sprint_040_'+direction)
            self.assertEqual(tape.shape,(500,13))
            np.testing.assert_allclose(tape[80,:3],[.4,0,turn],atol=1e-6)
            np.testing.assert_array_equal(tape[0],np.zeros(13))
            np.testing.assert_array_equal(tape[-1],np.zeros(13))
        self.assertAlmostEqual(float(commands('sprint_040_release')[300,0]),.3)
