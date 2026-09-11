import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

from sim2sim.research.budget import ActiveBudget, remaining, run_supervised, legacy_deadline, require_supervision
from unittest.mock import patch


class ActiveBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.budget = ActiveBudget(self.path)
        self.budget.initialize(seconds=100.)

    def pulse(self, actor, now, **kwargs):
        return self.budget.heartbeat(actor, now=now, boot='boot1', **kwargs)

    def test_parallel_work_is_interval_union(self):
        self.pulse('agent', 10)
        self.pulse('training', 15)
        self.pulse('agent', 30)
        self.pulse('training', 40, stop=True)
        self.assertEqual(self.budget.status()['used_seconds'], 30)

    def test_disconnected_gap_is_not_charged(self):
        self.pulse('agent', 0)
        self.pulse('agent', 20)
        self.pulse('agent', 1000)
        self.pulse('agent', 1010)
        self.assertEqual(self.budget.status()['used_seconds'], 30)
        self.assertEqual(self.budget.status()['unconfirmed_gaps'], 1)

    def test_background_work_counts_during_agent_disconnect(self):
        self.pulse('agent', 0)
        self.pulse('job', 0)
        self.pulse('agent', 10)
        for t in (30, 60, 90, 120, 150, 180, 200):
            self.pulse('job', t)
        self.pulse('agent', 200)
        self.assertEqual(self.budget.status()['used_seconds'], 200)
        self.assertEqual(self.budget.status()['remaining_seconds'], 0)
        self.assertEqual(self.budget.status()['unconfirmed_gaps'],0)

    def test_reboot_monotonic_origins_do_not_overlap(self):
        self.pulse('agent', 10)
        self.pulse('agent', 20)
        self.budget.heartbeat('agent', now=2, boot='boot2')
        self.budget.heartbeat('agent', now=8, boot='boot2')
        self.assertEqual(self.budget.status()['used_seconds'], 16)

    def test_stopped_actor_does_not_bridge_pause(self):
        self.pulse('agent', 10)
        self.pulse('agent', 20, stop=True)
        self.pulse('agent', 30)
        self.pulse('agent', 40)
        self.assertEqual(self.budget.status()['used_seconds'], 20)

    def test_reinitialize_refused_and_reserve_applies(self):
        with self.assertRaises(FileExistsError):
            self.budget.initialize()
        self.assertEqual(remaining(self.path, reserve=90), 10)

    def test_legacy_deadline_still_supported(self):
        self.budget.path.unlink()
        (self.path/'session.json').write_text(json.dumps({'deadline_unix':time.time()+100}))
        self.assertAlmostEqual(remaining(self.path, reserve=90), 10, delta=1)

    def test_supervised_timeout_is_recorded_and_job_stops(self):
        code = run_supervised(self.path, [sys.executable, '-c', 'import time;time.sleep(60)'],
                              timeout=.1, label='test')
        self.assertEqual(code, 124)
        data = json.loads(self.budget.path.read_text())
        self.assertEqual(data['events'][-1]['reason'], 'budget_or_timeout')
        self.assertEqual(data['actors'], {})
        self.assertGreater(self.budget.status()['used_seconds'], .1)

    def test_active_session_rejects_legacy_wall_clock_scheduler(self):
        with self.assertRaises(RuntimeError):legacy_deadline(self.path)
        with patch.dict('os.environ',{},clear=True):
            with self.assertRaises(RuntimeError):require_supervision(self.path)

    def test_supervision_sets_matching_session_and_budget(self):
        target=self.path/'child.json'
        code='import os,json,pathlib;pathlib.Path('+repr(str(target))+').write_text(json.dumps({k:os.environ.get(k) for k in ["SIM2SIM_ACTIVE_BUDGET_DIR","SIM2SIM_RESEARCH_DIR"]}))'
        self.assertEqual(run_supervised(self.path,[sys.executable,'-c',code],timeout=10),0)
        self.assertEqual(set(json.loads(target.read_text()).values()),{str(self.path.resolve())})


if __name__ == '__main__':
    unittest.main()
