"""scripts/vi-bg-watch.py: phone pings and status file for a paid run.

Fake status and permission JSON are served by `cat`; sends go to outbox.jsonl (--dry-run) or to a
stubbed post_ntfy. No real network, no real push, and the real ntfy topic file is never read.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "vi-bg-watch.py"
spec = importlib.util.spec_from_file_location("vi_bg_watch", SCRIPT)
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)

DAY, NIGHT, MORNING = "2026-11-26T14:00:00", "2026-11-26T23:30:00", "2026-11-27T08:05:00"


def status(**kw):
    s = dict(run_id="r1", status="running", exit_code=None, process_alive=True, phase="train",
             spend=dict(total=1.234), cap=5.0, cap_left=3.766,
             train=dict(step=10, total=100, eta_seconds=3600))
    s.update(kw)
    return s


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.platter = self.d / "platter"
        self.platter.mkdir()
        self.set_status(status())
        self.set_perms([])

    def set_status(self, s):
        (self.d / "st.json").write_text(json.dumps(s))

    def set_perms(self, p):
        (self.d / "perm.json").write_text(json.dumps(p))

    def add_set(self, n):
        (self.platter / f"step-{n}").mkdir(exist_ok=True)
        (self.platter / f"step-{n}" / "clips.toml").write_text("x")

    def run_pass(self, now=DAY, dry=True, run_id="r1"):
        argv = ["--state-dir", str(self.d / "s"), "--run-id", run_id, "--platter", str(self.platter),
                "--status-cmd", f"cat {self.d / 'st.json'}",
                "--agents-cmd", f"cat {self.d / 'perm.json'}",
                "--ntfy-topic-file", str(self.d / "no-topic"), "--now", now]
        if dry:
            argv.append("--dry-run")
        self.assertEqual(W.main(argv), 0)

    def outbox(self):
        return W.read_lines(self.d / "s" / "outbox.jsonl")

    def pending(self):
        return W.read_lines(self.d / "s" / "pending.jsonl")


class Events(Base):
    def test_needs_approval(self):
        self.set_perms([{"id": "abc"}])
        self.run_pass()
        (m,) = self.outbox()
        self.assertEqual(m["title"], "VI needs-approval")
        self.assertIn("run r1", m["message"])
        self.assertIn("$1.23", m["message"])

    def test_listening_set_ready_per_step(self):
        self.add_set(3)
        self.add_set(7)
        self.run_pass()
        self.assertEqual([m["message"].split(", ")[1] for m in self.outbox()],
                         ["step 3", "step 7"])

    def test_run_failed_status(self):
        self.set_status(status(status="failed", exit_code=5))
        self.run_pass()
        (m,) = self.outbox()
        self.assertEqual(m["title"], "VI run-failed")
        self.assertIn("exit 5", m["message"])

    def test_run_failed_process_gone(self):
        self.set_status(status(process_alive=False))
        self.run_pass()
        self.assertEqual(self.outbox()[0]["title"], "VI run-failed")

    def test_run_finished(self):
        self.set_status(status(status="done", exit_code=0))
        self.run_pass()
        (m,) = self.outbox()
        self.assertEqual(m["title"], "VI run-finished")

    def test_dedupe_on_second_pass(self):
        self.set_perms([{"id": "abc"}])
        self.add_set(1)
        self.run_pass()
        self.run_pass()
        self.assertEqual(len(self.outbox()), 2)
        self.add_set(2)
        self.run_pass()
        self.assertEqual(len(self.outbox()), 3)

    def test_no_topic_or_corpus_text_in_messages(self):
        self.set_status(status(status="failed", exit_code=1, message="secret corpus line"))
        self.run_pass()
        self.assertNotIn("secret", json.dumps(self.outbox()))


class QuietHours(Base):
    def test_hold_at_2330_then_one_digest_at_0805(self):
        self.set_status(status(status="failed", exit_code=2))
        self.set_perms([{"id": "abc"}])
        self.add_set(1)
        self.run_pass(NIGHT)
        self.assertEqual(self.outbox(), [])
        self.assertEqual(len(self.pending()), 3)
        self.run_pass("2026-11-27T02:00:00")  # still night: no duplicates held
        self.assertEqual(len(self.pending()), 3)
        self.run_pass(MORNING)
        (m,) = self.outbox()
        self.assertEqual(m["title"], "VI digest")
        self.assertEqual(self.pending(), [])
        self.run_pass("2026-11-27T08:10:00")
        self.assertEqual(len(self.outbox()), 1)

    def test_boundaries(self):
        self.assertTrue(W.in_quiet_hours(W.datetime(2026, 1, 1, 22, 0)))
        self.assertTrue(W.in_quiet_hours(W.datetime(2026, 1, 1, 7, 59)))
        self.assertFalse(W.in_quiet_hours(W.datetime(2026, 1, 1, 8, 0)))
        self.assertFalse(W.in_quiet_hours(W.datetime(2026, 1, 1, 21, 59)))


class IdleAndStatus(Base):
    def test_idle_without_run_id_does_nothing(self):
        with mock.patch.object(W, "post_ntfy") as p, mock.patch.object(W, "run_json") as r:
            self.run_pass(run_id="", dry=False)
        p.assert_not_called()
        r.assert_not_called()
        self.assertFalse((self.d / "s").exists())

    def test_no_status_is_idle(self):
        (self.d / "st.json").write_text("not json")
        self.run_pass()
        self.assertFalse((self.d / "s" / "status.json").exists())

    def test_status_file_written(self):
        self.set_perms([{"id": "abc"}])
        self.run_pass()
        card = json.loads((self.d / "s" / "status.json").read_text())
        self.assertEqual((card["state"], card["spend"], card["cap"], card["eta_seconds"]),
                         ("running", 1.234, 5.0, 3600))
        self.assertEqual(card["waiting_on_you"], ["approve request abc"])
        self.assertEqual(card["last_pass"], DAY)

    def test_run_id_file_activates(self):
        (self.d / "s").mkdir()
        (self.d / "s" / "run-id").write_text("r1\n")
        self.add_set(1)
        self.run_pass(run_id="")
        self.assertEqual(len(self.outbox()), 1)


class HttpFailure(Base):
    def test_failed_send_is_retried(self):
        self.add_set(1)
        with mock.patch.object(W, "post_ntfy", return_value=False) as p:
            self.run_pass(dry=False)
            self.run_pass(dry=False)
        self.assertEqual(p.call_count, 2)  # unsent, so tried again
        with mock.patch.object(W, "post_ntfy", return_value=True) as p:
            self.run_pass(dry=False)
            self.run_pass(dry=False)
        self.assertEqual(p.call_count, 1)  # sent once, then deduped

    def test_post_never_raises(self):
        with mock.patch.object(W.urllib.request, "urlopen", side_effect=OSError("down")):
            (self.d / "topic").write_text("t\n")
            self.assertFalse(W.post_ntfy(str(self.d / "topic"), "t", "m", "default"))
        self.assertFalse(W.post_ntfy(str(self.d / "missing"), "t", "m", "default"))

    def test_failed_digest_stays_pending(self):
        self.add_set(1)
        self.run_pass(NIGHT)
        with mock.patch.object(W, "post_ntfy", return_value=False):
            self.run_pass(MORNING, dry=False)
        self.assertEqual(len(self.pending()), 1)


if __name__ == "__main__":
    unittest.main()
