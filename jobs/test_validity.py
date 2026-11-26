"""r32 WP3: the rental validity state machine (jobs/paidrun/validity.py) with a fake clock, and
release()'s destroy order for an INVALID rental with a fake `vi`. No sleeps, no Vast, no GPU."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

JOBS = Path(__file__).resolve().parent
sys.path.insert(0, str(JOBS))

from paidrun import validity as V  # noqa: E402
from paidrun.common import EXIT, Stop  # noqa: E402
from paidrun.release import ReleaseMixin  # noqa: E402


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


def machine(clock, **kw):
    kw.setdefault("wall_cap_s", 10 * 3600)
    return V.Validity(started=clock(), clock=clock, attempt=1, **kw)


class StateMachine(unittest.TestCase):
    def setUp(self):
        self.c = FakeClock()

    def test_stall_then_new_progress_is_valid_again(self):
        v = machine(self.c)
        v.enter("tokenise", stall_s=120, tries=2, recover_s=600)
        v.evidence("progress", 10)  # baseline
        self.c.advance(119)
        self.assertEqual(v.poll(), V.VALID)
        self.c.advance(1)
        self.assertEqual(v.poll(), V.SUSPECT)
        self.assertTrue(v.try_recover())
        self.c.advance(30)
        self.assertEqual(v.evidence("progress", 11), V.VALID)
        self.assertEqual(v.poll(), V.VALID)
        # the stall window restarts from the new progress
        self.c.advance(119)
        self.assertEqual(v.poll(), V.VALID)

    def test_recovery_tries_exhausted_is_invalid_stalled(self):
        v = machine(self.c)
        v.enter("venv", stall_s=60, tries=2, recover_s=3600)
        self.c.advance(60)
        self.assertEqual(v.poll(), V.SUSPECT)
        self.assertTrue(v.try_recover())
        self.assertTrue(v.try_recover())
        self.assertFalse(v.try_recover())
        self.assertEqual((v.state, v.reason), (V.INVALID, V.REASON_STALLED))
        self.assertIn("2 recovery tries", v.detail)

    def test_recovery_time_exhausted_is_invalid_stalled(self):
        v = machine(self.c)
        v.enter("base-model-fetch", stall_s=120, tries=5, recover_s=300)
        self.c.advance(120)
        self.assertEqual(v.poll(), V.SUSPECT)
        self.c.advance(299)
        self.assertEqual(v.poll(), V.SUSPECT)
        self.c.advance(1)
        self.assertEqual((v.poll(), v.reason), (V.INVALID, V.REASON_STALLED))

    def test_recovery_time_accumulates_across_suspect_spells_in_a_stage(self):
        v = machine(self.c)
        v.enter("upload", stall_s=60, tries=5, recover_s=100)
        v.evidence("progress", 1)
        self.c.advance(60)
        self.assertEqual(v.poll(), V.SUSPECT)
        self.c.advance(80)
        self.assertEqual(v.evidence("progress", 2), V.VALID)  # 80 s of recovery used
        self.c.advance(60)
        self.assertEqual(v.poll(), V.SUSPECT)
        self.c.advance(20)
        self.assertEqual((v.poll(), v.reason), (V.INVALID, V.REASON_STALLED))

    def test_stage_deadline_is_invalid_stalled(self):
        v = machine(self.c)
        v.enter("cuda-load", deadline_s=300)
        self.c.advance(299)
        self.assertEqual(v.poll(), V.VALID)
        self.c.advance(1)
        self.assertEqual((v.poll(), v.reason), (V.INVALID, V.REASON_STALLED))

    def test_cap_reached_mid_recovery_is_invalid_cap(self):
        v = machine(self.c, wall_cap_s=1000)
        v.enter("tokenise", stall_s=120, tries=3, recover_s=3600)
        self.c.advance(900)
        self.assertEqual(v.poll(), V.SUSPECT)
        self.assertTrue(v.try_recover())
        self.c.advance(100)
        self.assertEqual((v.poll(), v.reason), (V.INVALID, V.REASON_CAP))
        # final: later progress or a new stage never brings it back, the cap is not moved
        self.assertEqual(v.evidence("progress", 99), V.INVALID)
        self.assertEqual(v.enter("train"), V.INVALID)
        self.assertEqual(v.cap_at, 1000.0 + 1000)
        self.assertEqual(v.reason, V.REASON_CAP)

    def test_cap_is_the_smaller_of_wall_and_dollars(self):
        # $2 at $0.40/hr + $0.10/hr storage = 4 h, under the 8 h wall cap
        v = machine(self.c, wall_cap_s=8 * 3600, dollar_cap=2.0, dph=0.40, storage_dph=0.10)
        self.assertAlmostEqual(v.cap_s, 4 * 3600)
        self.c.advance(4 * 3600 - 1)
        self.assertEqual(v.poll(), V.VALID)
        self.c.advance(1)
        self.assertEqual(v.poll(), V.INVALID)
        self.assertEqual(v.reason, V.REASON_CAP)
        # the wall cap wins when it is smaller
        self.assertAlmostEqual(V.effective_cap_seconds(3600, 100.0, 0.5), 3600)
        self.assertAlmostEqual(V.effective_cap_seconds(None, 1.0, 0.5), 7200)
        self.assertIsNone(V.effective_cap_seconds(None, None, 0.5))

    def test_cap_uses_the_full_billed_rate_of_a_listed_017_offer(self):
        # `vi cloud offers` prints the full billed rate: an offer Vast lists at $0.17/hr whose
        # fields (dph_base 0.17 + storage 1.022 $/GB/month x 100 GB / 730 h) rebuild to $0.31
        # is printed as $0.310/hr. The cap math reads that figure (crates/cloud, billed_hourly).
        from paidrun.signals import parse_offers
        line = ("  9001  RTX 3090      24 GB  $0.310/hr  reliability 0.990  "
                "net down   900 / up   500 Mbps  16.0 cores  128 GB ram  disk 1000 MB/s  "
                "Texas, US  machine 7 host 8")
        (o,) = parse_offers(line)
        self.assertEqual(o["dph"], 0.31)
        # $3.10 at $0.31/hr is exactly 10 h (it would be about 18.2 h at the listed $0.17)
        self.assertAlmostEqual(V.effective_cap_seconds(None, 3.10, o["dph"]), 10 * 3600)
        self.assertEqual(round(o["dph"] * 8, 2), 2.48)

    def test_cap_wins_over_a_stage_still_making_progress(self):
        v = machine(self.c, wall_cap_s=600)
        v.enter("train", stall_s=600)
        for k in range(1, 10):
            self.c.advance(60)
            v.evidence("progress", k)
            v.poll()
        self.c.advance(60)
        self.assertEqual((v.poll(), v.reason), (V.INVALID, V.REASON_CAP))

    def test_liveness_only_evidence_never_recovers(self):
        v = machine(self.c)
        v.enter("train", stall_s=600, tries=3, recover_s=1800)
        v.evidence("progress", 100)
        self.c.advance(600)
        self.assertEqual(v.poll(), V.SUSPECT)
        for _ in range(29):
            self.c.advance(60)
            for kind in V.LIVENESS_KINDS:
                self.assertEqual(v.evidence(kind), V.SUSPECT, kind)
            self.assertEqual(v.poll(), V.SUSPECT)
        self.c.advance(60)
        self.assertEqual((v.poll(), v.reason), (V.INVALID, V.REASON_STALLED))

    def test_stale_required_liveness_is_suspect_and_its_return_does_not_recover(self):
        v = machine(self.c)
        v.enter("debug", required={"return_push_ok": 300}, recover_s=900)
        v.evidence("return_push_ok")
        self.c.advance(300)
        self.assertEqual(v.poll(), V.VALID)
        self.c.advance(1)
        self.assertEqual(v.poll(), V.SUSPECT)
        v.evidence("return_push_ok")
        self.assertEqual(v.poll(), V.SUSPECT, "liveness is back, but only progress ends SUSPECT")
        v.evidence("progress", 1)
        v.evidence("progress", 2)
        self.assertEqual(v.poll(), V.VALID)

    def test_a_counter_that_does_not_increase_never_recovers(self):
        v = machine(self.c)
        v.enter("tokenise", stall_s=120, tries=1, recover_s=600)
        v.evidence("progress", 50)
        self.c.advance(120)
        self.assertEqual(v.poll(), V.SUSPECT)
        for value in (50, 49, 50.0, "50", None, "not a number"):
            self.c.advance(10)
            self.assertEqual(v.evidence("progress", value), V.SUSPECT, value)
        self.assertEqual(v.poll(), V.SUSPECT)
        self.assertEqual(v.evidence("progress", 51), V.VALID)

    def test_first_reading_of_a_counter_is_a_baseline_not_progress(self):
        v = machine(self.c)
        v.enter("upload", stall_s=60, recover_s=600)
        self.c.advance(60)
        self.assertEqual(v.poll(), V.SUSPECT)
        self.assertEqual(v.evidence("progress", 5, counter="bytes-acked"), V.SUSPECT)
        self.assertEqual(v.evidence("progress", 6, counter="bytes-acked"), V.VALID)

    def test_invalid_is_final_and_keeps_the_first_reason(self):
        v = machine(self.c)
        v.invalidate(V.REASON_HOST_DROP, "two Vast reads said exited")
        v.invalidate(V.REASON_OUR_FAILURE, "later")
        self.c.advance(11 * 3600)
        self.assertEqual((v.poll(), v.reason), (V.INVALID, V.REASON_HOST_DROP))
        with self.assertRaises(ValueError):
            v.invalidate("bored")
        with self.assertRaises(ValueError):
            v.evidence("mtime")

    def test_new_stage_clears_suspect_but_keeps_the_cap(self):
        v = machine(self.c, wall_cap_s=5000)
        v.enter("venv", stall_s=60, recover_s=600)
        self.c.advance(60)
        self.assertEqual(v.poll(), V.SUSPECT)
        v.enter("tokenise")
        self.assertEqual(v.poll(), V.VALID)
        self.c.advance(5000)
        self.assertEqual((v.poll(), v.reason), (V.INVALID, V.REASON_CAP))

    def test_injectable_clock_is_the_default_for_every_call(self):
        v = machine(self.c, wall_cap_s=100)
        self.c.advance(100)
        self.assertEqual(v.poll(), V.INVALID)
        self.assertEqual(v.invalid_at, 1100.0)

    def test_invalidated_stop_codes(self):
        v = machine(self.c, wall_cap_s=1)
        v.enter("train")
        self.c.advance(1)
        v.poll()
        e = v.stop()
        self.assertIsInstance(e, Stop)
        self.assertEqual((e.code, e.reason, e.stage), (EXIT["budget"], V.REASON_CAP, "train"))
        self.assertIn("INVALID (cap)", e.message)
        self.assertEqual(V.Invalidated(V.REASON_STALLED, "x").code, EXIT["other"])

    def test_stage_table_rows_follow_c2(self):
        t = V.STAGE_TABLE
        self.assertEqual((t["create-running"].deadline_s, t["boot-ssh"].deadline_s), (600, 900))
        self.assertEqual((t["host-qual"].deadline_s, t["host-qual"].stall_s), (240, 60))
        self.assertEqual((t["tokenise"].deadline_s, t["tokenise"].stall_s), (450, 120))
        self.assertEqual(V.train_stall_seconds(30), 600)
        self.assertEqual(V.train_stall_seconds(200), 1000)
        self.assertEqual(V.checkpoint_push_deadline_seconds(7e9, 10e6), 1050)
        self.assertEqual(V.checkpoint_push_deadline_seconds(1e9, 10e6), 600)
        for name in list(t) + list(V.REASONS):
            self.assertRegex(name, r"^[a-z]+(-[a-z]+)*$")


class Rental2Rules(unittest.TestCase):
    def test_rental2_eligibility_table(self):
        # C.6 + D3 (WP5): host-drop and host-unqualified allow rental 2; stalled, our-failure
        # and cap never
        want = {V.REASON_HOST_DROP: True, V.REASON_HOST_UNQUALIFIED: True,
                V.REASON_STALLED: False, V.REASON_OUR_FAILURE: False, V.REASON_CAP: False,
                None: False, "anything-else": False}
        self.assertEqual(set(V.REASONS), set(want) - {None, "anything-else"})
        for reason, allowed in want.items():
            with self.subTest(reason=reason):
                self.assertIs(V.rental2_allowed(reason), allowed)

    def test_host_unqualified_follows_its_hook(self):
        orig = V.host_unqualified_allows_rental2
        try:
            V.host_unqualified_allows_rental2 = lambda: False
            self.assertFalse(V.rental2_allowed(V.REASON_HOST_UNQUALIFIED))
        finally:
            V.host_unqualified_allows_rental2 = orig
        self.assertTrue(V.host_unqualified_allows_rental2())
        self.assertTrue(V.rental2_allowed(V.REASON_HOST_UNQUALIFIED))

    def test_outcome_reason_classes(self):
        self.assertEqual(V.reason_for_outcome("host-drop"), V.REASON_HOST_DROP)
        self.assertEqual(V.reason_for_outcome("worker-gate"), V.REASON_HOST_UNQUALIFIED)
        self.assertEqual(V.reason_for_outcome("host-unqualified"), V.REASON_HOST_UNQUALIFIED)
        self.assertEqual(V.reason_for_outcome("train-failed"), V.REASON_OUR_FAILURE)
        for good in ("done", "debug-done", "orphan", "interrupted", "train-vanished"):
            self.assertIsNone(V.reason_for_outcome(good), good)


class FakeRelease(ReleaseMixin):
    """Just enough of PaidRun for release(): every outside effect is logged in order."""

    def __init__(self, tmp, validity):
        self.log = []
        self.a = SimpleNamespace(debug_pass=False, run_id="t1")
        self.dir = Path(tmp)
        self.s = dict(attempts=[], spent_total=0.0, hours_total=0.0)
        self.validity = validity
        self.worker_maybe_up = True
        self.in_teardown = False
        self.user_stopped = False

    def save(self):
        self.log.append("save")

    def fetch_diagnostics(self, at, outcome):
        self.log.append("ssh-diagnostics")

    def vi_cmd(self, *args, stream=False):
        self.log.append("DELETE" if args[:2] == ("cloud", "down") else " ".join(args))
        return 0, "destroyed instance 101 ($0.42 spent)\n"

    def vast_evidence(self, at, why):
        self.log.append("vast-evidence")

    def validity_evidence(self, kind):
        self.log.append(kind)

    def instance_id(self):
        return "101"

    def credit(self):
        return None

    def close_attempt(self, at, outcome, spent, spent_estimated=False):
        at.update(ended=1, outcome=outcome, spent=spent)
        self.log.append("close")


class DestroyOrder(unittest.TestCase):
    def setUp(self):
        self.c = FakeClock()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.at = dict(n=1, instance="101", started=self.c(), dph=0.4)

    def release(self, v, outcome="aborted"):
        r = FakeRelease(self.tmp.name, v)
        r.s["attempts"].append(self.at)
        r.release(self.at, outcome)
        return r

    def test_recovery_exhausted_destroys_before_any_diagnostics(self):
        v = machine(self.c)
        v.enter("tokenise", stall_s=120, tries=1, recover_s=3600)
        self.c.advance(120)
        v.try_recover()
        self.assertFalse(v.try_recover())
        self.assertEqual(v.reason, V.REASON_STALLED)
        r = self.release(v)
        self.assertEqual(r.log[0], "DELETE", r.log)
        self.assertNotIn("ssh-diagnostics", r.log)
        self.assertLess(r.log.index("DELETE"), r.log.index("vast-evidence"))
        self.assertEqual(self.at["invalid"]["reason"], V.REASON_STALLED)
        self.assertEqual(r.s["validity"]["reason"], V.REASON_STALLED)
        self.assertIsNone(r.validity)
        self.assertFalse(r.worker_maybe_up)

    def test_cap_destroys_first_too(self):
        v = machine(self.c, wall_cap_s=60)
        self.c.advance(60)
        v.poll()
        r = self.release(v)
        self.assertEqual(r.log[0], "DELETE", r.log)
        self.assertEqual(self.at["invalid"]["reason"], V.REASON_CAP)

    def test_failed_destroy_is_exit_9_and_records_no_reason(self):
        v = machine(self.c, wall_cap_s=60)
        self.c.advance(60)
        v.poll()
        r = FakeRelease(self.tmp.name, v)
        r.vi_cmd = lambda *a, stream=False: (r.log.append("DELETE") or (1, "error"))
        with unittest.mock.patch("paidrun.release.nap", lambda s: None):
            with self.assertRaises(Stop) as e:
                r.release(self.at, "aborted")
        self.assertEqual(e.exception.code, EXIT["release"])
        self.assertEqual(r.log.count("DELETE"), 5)
        self.assertTrue(r.worker_maybe_up)
        self.assertNotIn("invalid", self.at)

    def test_ordinary_failure_keeps_its_bounded_diagnostics_before_the_delete(self):
        # unchanged from r32 WP1: a failure the machine did not reach by itself keeps the
        # <=25 s ssh diagnostics first; the reason class is the outcome's
        r = self.release(machine(self.c), "preflight-failed")
        self.assertLess(r.log.index("ssh-diagnostics"), r.log.index("DELETE"))
        self.assertEqual(self.at["invalid"]["reason"], V.REASON_OUR_FAILURE)

    def test_host_drop_keeps_the_existing_order_and_its_reason(self):
        v = machine(self.c)
        v.invalidate(V.REASON_HOST_DROP, "HOST-DROP (status exited)")
        r = self.release(v, "host-drop")
        self.assertLess(r.log.index("ssh-diagnostics"), r.log.index("DELETE"))
        self.assertEqual(self.at["invalid"]["reason"], V.REASON_HOST_DROP)

    def test_good_end_and_user_stop_record_no_reason(self):
        r = self.release(machine(self.c), "done")
        self.assertNotIn("invalid", self.at)
        self.assertEqual(r.s["validity"]["state"], V.VALID)
        self.at.pop("ended", None)
        r = FakeRelease(self.tmp.name, machine(self.c))
        r.user_stopped = True
        r.release(self.at, "aborted")
        self.assertNotIn("invalid", self.at)


import unittest.mock  # noqa: E402

if __name__ == "__main__":
    unittest.main()
