"""PlayBrain / hold-to-move mapping (no Godot, no ONNX)."""

from __future__ import annotations

import unittest

import numpy as np

from sim2sim.play_input import (
    PlayBrain,
    TwistLimits,
    clamp_time_scale,
    held_twist,
    keys_to_held,
    keys_to_taps,
    relaunch_argv,
    wall_dt,
)


class TestHeldTwist(unittest.TestCase):
    def test_empty_is_idle(self) -> None:
        self.assertEqual(held_twist(set()), (0.0, 0.0, 0.0))

    def test_fwd_back_and_space(self) -> None:
        lim = TwistLimits()
        self.assertEqual(held_twist({"fwd"}, lim)[0], lim.vmax_x)
        self.assertEqual(held_twist({"back"}, lim)[0], lim.vmin_x)
        self.assertEqual(held_twist({"fwd", "idle"}, lim), (0.0, 0.0, 0.0))

    def test_yaw_and_strafe(self) -> None:
        lim = TwistLimits()
        self.assertEqual(held_twist({"left"}, lim)[2], lim.vmax_ang)
        self.assertEqual(held_twist({"right"}, lim)[2], -lim.vmax_ang)
        self.assertEqual(held_twist({"strafe_l"}, lim)[1], lim.vmax_y)
        self.assertEqual(held_twist({"strafe_r"}, lim)[1], lim.vmin_y)

    def test_arrow_aliases(self) -> None:
        self.assertEqual(keys_to_held({"W", "UP"}), {"fwd"})
        self.assertEqual(keys_to_held({"A", "LEFT"}), {"left"})
        self.assertIn("pick", keys_to_taps({"G", "KEY_1"}))
        self.assertEqual(keys_to_taps({"ESCAPE"}), ["quit"])
        self.assertEqual(keys_to_taps({"KEY_6"}), ["switch_robot"])


class TestTimeScale(unittest.TestCase):
    def test_default_is_realtime(self) -> None:
        self.assertAlmostEqual(wall_dt(0.02, 1.0), 0.02)

    def test_faster_and_slower(self) -> None:
        self.assertAlmostEqual(wall_dt(0.02, 2.0), 0.01)
        self.assertAlmostEqual(wall_dt(0.02, 0.25), 0.08)

    def test_clamp(self) -> None:
        self.assertAlmostEqual(clamp_time_scale(0.01), 0.25)
        self.assertAlmostEqual(clamp_time_scale(9.0), 3.0)
        self.assertAlmostEqual(wall_dt(0.02, 99.0), 0.02 / 3.0)


class TestPlayBrain(unittest.TestCase):
    def setUp(self) -> None:
        self.b = PlayBrain()

    def test_hold_fwd_switches_to_walking(self) -> None:
        out = self.b.tick({"fwd"}, [], 0.02)
        self.assertEqual(out.policy, "walking")
        np.testing.assert_allclose(out.command[0:3], [0.3, 0.0, 0.0], atol=1e-6)

    def test_release_returns_standing(self) -> None:
        self.b.tick({"fwd"}, [], 0.02)
        out = self.b.tick(set(), [], 0.02)
        self.assertEqual(out.policy, "standing")
        np.testing.assert_allclose(out.command, np.zeros(13))

    def test_sit_toggle_and_blocks_kick(self) -> None:
        out = self.b.tick(set(), ["sit"], 0.02)
        self.assertEqual(out.policy, "sitstand")
        self.assertAlmostEqual(float(out.command[0]), 1.0)
        self.b.tick(set(), ["kick_left"], 0.02)
        self.assertEqual(self.b.policy, "sitstand")
        out = self.b.tick(set(), ["sit"], 0.02)
        self.assertFalse(self.b.sit)
        self.assertAlmostEqual(float(out.command[0]), 0.0)

    def test_pick_cycle_returns(self) -> None:
        self.b.tick(set(), ["pick"], 0.02)
        self.assertEqual(self.b.policy, "ground_pick")
        # 4s period; 4.1s must finish
        for _ in range(205):
            self.b.tick(set(), [], 0.02)
        self.assertEqual(self.b.policy, "standing")

    def test_roulade_timer(self) -> None:
        self.b.tick(set(), ["roulade"], 0.02)
        self.assertEqual(self.b.policy, "roulade")
        np.testing.assert_allclose(self.b.command_13(), np.zeros(13))
        for _ in range(101):
            self.b.tick(set(), [], 0.02)
        self.assertEqual(self.b.policy, "standing")

    def test_idle_tap_clears_walk(self) -> None:
        self.b.tick({"fwd"}, [], 0.02)
        out = self.b.tick({"fwd"}, ["idle"], 0.02)
        self.assertEqual(out.policy, "standing")
        np.testing.assert_allclose(out.command[0:3], [0.0, 0.0, 0.0])

    def test_reset_and_quit(self) -> None:
        self.b.tick({"fwd"}, [], 0.02)
        out = self.b.tick(set(), ["reset"], 0.02)
        self.assertTrue(out.reset)
        self.assertEqual(out.policy, "standing")
        out = self.b.tick(set(), ["quit"], 0.02)
        self.assertTrue(out.quit)

    def test_walking_only_stays_on_walking(self) -> None:
        b = PlayBrain(has_standing=False, has_sitstand=False, has_pick=False)
        out = b.tick(set(), [], 0.02)
        self.assertEqual(out.policy, "walking")

    def test_switch_robot_tap(self) -> None:
        out = self.b.tick(set(), ["switch_robot"], 0.02)
        self.assertTrue(out.switch_robot)
        self.assertEqual(out.policy, "standing")

    def test_roller_limits_block_strafe(self) -> None:
        lim = TwistLimits(vmax_x=0.6, vmin_x=-0.5, vmax_y=0.0, vmin_y=0.0, vmax_ang=1.0)
        b = PlayBrain(has_sitstand=False, has_pick=False, has_kick_left=False, has_kick_right=False, has_roulade=False, lim=lim)
        out = b.tick({"fwd"}, [], 0.02)
        np.testing.assert_allclose(out.command[0:3], [0.6, 0.0, 0.0], atol=1e-6)
        out = b.tick({"strafe_l"}, [], 0.02)
        np.testing.assert_allclose(out.command[0:3], [0.0, 0.0, 0.0], atol=1e-6)
        b.tick(set(), ["kick_left"], 0.02)
        self.assertNotEqual(b.policy, "kick_left")


class TestRelaunchArgv(unittest.TestCase):
    def test_toggle_roller_keeps_local_ppo(self) -> None:
        walk = ["/bin/sim2sim-play", "--local-ppo"]
        roller = relaunch_argv(walk, want_roller=True, executable="/py")
        self.assertEqual(roller, ["/py", "-u", "/bin/sim2sim-play", "--local-ppo", "--roller"])
        child = ["/bin/sim2sim-play", "--local-ppo", "--roller"]
        back = relaunch_argv(child, want_roller=False, executable="/py")
        self.assertEqual(back, ["/py", "-u", "/bin/sim2sim-play", "--local-ppo"])


if __name__ == "__main__":
    unittest.main()
