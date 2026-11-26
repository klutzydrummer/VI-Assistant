"""r32 WP5: the host-qualification gate, rental-2 rules by reason, and host exclusion.

No rental and no network: the paid run drives the fake `vi` from test_paid_run (its `cloud ssh`
answers the host-qual download per attempt's "hostqual": fast, slow, nocurl or fail), and the
shell tests run jobs/push-probe.sh against a local "worker" with a fake curl, as
test_push_probe does.
"""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import test_debug_pass as tdp
import test_paid_run as tp
import test_push_probe as tpp
from paidrun import config, limits, validity

JOBS = Path(__file__).resolve().parent
PROBE = JOBS / "push-probe.sh"


def kinds(calls, names=("up", "hostqual", "probe", "stage", "down")):
    return [c[0] for c in calls if c[0] in names]


class Thresholds(unittest.TestCase):
    def test_g_is_a_named_constant_of_5_mb_s_over_30_s(self):
        self.assertEqual(limits.HOST_QUAL_MIN_DOWNLOAD_MB_S, 5)
        self.assertEqual(limits.HOST_QUAL_DOWNLOAD_SECONDS, 30)
        self.assertEqual(limits.HOST_QUAL_MIN_PUSH_MB_S, 1)
        # the gate's own call is bounded: the window plus a margin, never open-ended
        self.assertGreater(limits.HOST_QUAL_TIMEOUT_SECONDS, limits.HOST_QUAL_DOWNLOAD_SECONDS)
        self.assertLessEqual(limits.HOST_QUAL_TIMEOUT_SECONDS, 300)

    def test_rental2_by_reason(self):
        self.assertTrue(validity.rental2_allowed(validity.REASON_HOST_UNQUALIFIED))
        self.assertTrue(validity.rental2_allowed(validity.REASON_HOST_DROP))
        for r in (validity.REASON_STALLED, validity.REASON_OUR_FAILURE, validity.REASON_CAP, None):
            self.assertFalse(validity.rental2_allowed(r), r)

    def test_debug_max_rentals_is_two(self):
        self.assertEqual(config.DEBUG_MAX_RENTALS, 2)


class FullRun(tp.Base):
    def test_fast_probe_goes_on_to_the_upload(self):
        self.scenario([{}])
        self.assertEqual(self.run_orch(), 0, self.out)
        k = kinds(self.calls())
        self.assertEqual(k[:3], ["up", "hostqual", "stage"], k)
        self.assertEqual(self.state()["failed_hosts"], [])
        self.assertEqual(self.state()["attempts"][0]["host_qual"]["download"], "pass")
        # the gate got G through the environment; the preflight got the push half
        self.assertIn(["hostqual", "fast", "5"], self.calls())
        self.assertIn("host gate", self.out)

    def test_slow_probe_is_invalid_destroyed_and_the_next_search_excludes_it(self):
        self.scenario([dict(hostqual="slow"), {}],
                      offer_list=tp.machine_offers((1, 0.20, 10, 1), (2, 0.25, 10, 2),
                                                   (3, 0.30, 11, 1), (4, 0.35, 12, 3)))
        self.assertEqual(self.run_orch(), 0, self.out)
        # offer 2 shares machine 10 and offer 3 shares host 1: both are skipped
        self.assertEqual([u["offer"] for u in self.ups()], ["1", "4"])
        k = kinds(self.calls())
        self.assertEqual(k[:4], ["up", "hostqual", "down", "up"], k)
        # nothing was uploaded on the slow host: the first stage ran on rental 2
        self.assertLess(k.index("up", 1), k.index("stage"))
        down1 = self.downs()[0][1]
        self.assertIn("--force", down1)
        self.assertIn("--verify", down1)
        s = self.state()
        self.assertEqual((s["failed_machines"], s["failed_hosts"]), ([10], [1]))
        offer_calls = [c[1] for c in self.calls() if c[0] == "offers"]
        self.assertIn("--exclude-host", offer_calls[-1])
        self.assertIn("--exclude-machine", offer_calls[-1])
        self.assertIn("host-unqualified", self.out)

    def test_a_probe_that_cannot_run_is_unqualified(self):
        for mode in ("fail", "nocurl"):
            with self.subTest(mode=mode):
                self.setUp()
                self.scenario([dict(hostqual=mode), dict(hostqual=mode)],
                              offer_list=tp.offers((1, 0.20, "US", 800)))
                rc = self.run_orch()
                self.assertNotEqual(rc, 0, self.out)
                self.assertEqual(self.stages(), [], "uploaded to an unqualified host")
                self.assertIsNone(self.worker())
                self.assertIn("host-unqualified", self.out)

    def test_upload_never_comes_before_the_gate_passes(self):
        self.scenario([dict(hostqual="slow")] * 3,
                      offer_list=tp.machine_offers((1, 0.20, 10, 1), (2, 0.25, 11, 2),
                                                   (3, 0.30, 12, 3)))
        self.run_orch(max_attempts=3)
        self.assertEqual(self.stages(), [], self.out)
        self.assertEqual(len(self.ups()), 3)
        self.assertIsNone(self.worker())

    def test_slow_push_half_is_unqualified_and_the_next_offer_is_tried(self):
        self.scenario([dict(preflight="pushslow"), {}],
                      offer_list=tp.machine_offers((1, 0.20, 10, 1), (2, 0.25, 11, 2)))
        self.assertEqual(self.run_orch(), 0, self.out)
        self.assertEqual([u["offer"] for u in self.ups()], ["1", "2"])
        self.assertIn(["preflight_env", "1", "a"], self.calls())
        s = self.state()
        self.assertEqual((s["failed_machines"], s["failed_hosts"]), ([10], [1]))
        self.assertIn("host-unqualified", self.out)

    def test_host_drop_excludes_the_host_and_machine(self):
        self.scenario([dict(preflight="drop", vast_gone=True), {}],
                      offer_list=tp.machine_offers((1, 0.20, 10, 1), (2, 0.25, 11, 2)))
        self.run_orch()
        s = self.state()
        self.assertIn(10, s["failed_machines"])
        self.assertIn(1, s["failed_hosts"])

    def test_cap_bounds_the_retries(self):
        # each failed host costs $0.60 of a $1 cap: the second is never rented
        self.scenario([dict(hostqual="slow", spent=0.6, hours=0.1)] * 3,
                      offer_list=tp.machine_offers((1, 0.20, 10, 1), (2, 0.25, 11, 2),
                                                   (3, 0.30, 12, 3)))
        rc = self.run_orch(cap=1, max_hours=4)
        self.assertNotEqual(rc, 0, self.out)
        self.assertLessEqual(len(self.ups()), 1, self.out)
        self.assertEqual(self.stages(), [])


class DebugPass(tdp.DebugBase):
    def scen2(self, first, second=None, **extra):
        self.scenario([dict(spent=0.3, **first), dict(spent=0.3, **(second or {}))],
                      corpus=str(self.tmp / "corpus"),
                      offers=tp.machine_offers((1, 0.20, 10, 1), (2, 0.25, 11, 1),
                                               (3, 0.30, 12, 2)), **extra)

    def scenario(self, attempts, offer_list=None, credit=20.0, offers=None, **extra):
        super().scenario(attempts, offer_list=offers or offer_list, credit=credit, **extra)

    def test_fast_probe_runs_the_stages(self):
        self.scen()
        self.assertEqual(self.run_debug(), 0, self.out)
        self.assertEqual(kinds(self.calls()), ["up", "hostqual", "probe", "stage", "down"])
        self.assertIn("host qualified", self.details()[0] if 0 in self.details() else self.out)

    def test_slow_probe_rents_once_more_with_the_host_and_machine_excluded(self):
        self.scen2(dict(hostqual="slow"))
        self.assertEqual(self.run_debug(), 0, self.out)
        self.assertEqual([u["offer"] for u in self.ups()], ["1", "3"])  # offer 2 shares host 1
        k = kinds(self.calls())
        self.assertEqual(k, ["up", "hostqual", "down", "up", "hostqual", "probe", "stage", "down"], k)
        r1 = self.summary()["rentals"][0]
        self.assertTrue(r1["status"].startswith("HOST-UNQUALIFIED") or
                        "unqualified" in r1["status"].lower(), r1)
        s = self.state()
        self.assertEqual((s["failed_machines"], s["failed_hosts"]), ([10], [1]))
        offer_calls = [c[1] for c in self.calls() if c[0] == "offers"]
        self.assertIn(["--exclude-machine", "10", "--exclude-host", "1"], offer_calls[1:])

    def test_probe_failure_is_unqualified(self):
        self.scen2(dict(hostqual="fail"), dict(hostqual="fail"))
        self.assertEqual(self.run_debug(), 16, self.out)
        self.assertEqual(self.stages(), [], "a worker stage ran on an unqualified host")
        self.assertNotIn("probe", kinds(self.calls()))
        self.assertIsNone(self.worker())

    def test_never_more_than_debug_max_rentals(self):
        self.scenario([dict(spent=0.3, hostqual="slow")] * 4, corpus=str(self.tmp / "corpus"),
                      offers=tp.machine_offers((1, 0.20, 10, 1), (2, 0.25, 11, 2),
                                               (3, 0.30, 12, 3), (4, 0.35, 13, 4)))
        self.assertEqual(self.run_debug(), 16, self.out)
        self.assertEqual(len(self.ups()), config.DEBUG_MAX_RENTALS, self.out)
        self.assertEqual(len(self.downs()), config.DEBUG_MAX_RENTALS, self.out)
        self.assertIsNone(self.worker())

    def test_host_drop_still_rents_once_more_and_excludes_the_host(self):
        self.scen2(dict(debug="drop"))
        self.assertEqual(self.run_debug(), 0, self.out)
        self.assertEqual(len(self.ups()), 2, self.out)
        s = self.state()
        self.assertEqual((s["failed_machines"], s["failed_hosts"]), ([10], [1]))
        self.assertEqual([u["offer"] for u in self.ups()], ["1", "3"])

    def test_stalled_never_rents_again(self):
        self.env["PAID_RUN_TIMEOUT_SCALE"] = "0.00001"
        self.scen2(dict(debug="hang"))
        self.assertEqual(self.run_debug(), 16, self.out)
        self.assertEqual(len(self.ups()), 1, self.out)

    def test_our_failure_never_rents_again(self):
        self.scen2({}, debug_fail=[4])
        self.assertEqual(self.run_debug(), 16, self.out)
        self.assertEqual(len(self.ups()), 1, self.out)

    def test_cap_never_rents_again(self):
        self.scenario([dict(spent=0.95, hostqual="slow"), dict(spent=0.1)],
                      corpus=str(self.tmp / "corpus"))
        self.assertEqual(self.run_debug(cap=1, max_hours=1), 16, self.out)
        self.assertEqual(len(self.ups()), 1, self.out)
        self.assertIn("no second rental", self.summary()["note"])


FAKE_VI = '#!/bin/bash\n# fake `vi cloud ssh SCRIPT`: runs SCRIPT locally\nshift 2\nexec bash -c "$1"\n'


class HostQualShell(unittest.TestCase):
    """`push-probe.sh host-qual`: the download half, run from home through `$VI_BIN cloud ssh`."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        (bin_dir / "curl").write_text(tpp.FAKE_CURL)
        (bin_dir / "curl").chmod(0o755)
        vi = root / "vi"
        vi.write_text(FAKE_VI)
        vi.chmod(0o755)
        self.env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "VI_BIN": str(vi),
                    "FAKE_CURL_LOG": str(root / "curl.log"),
                    "VI_PUSH_PROBE_URLS": "http://fake.invalid/one",
                    "VI_HOST_QUAL_DOWNLOAD_SECONDS": "6", "VI_HOST_QUAL_MIN_DOWNLOAD_MB_S": "5"}

    def tearDown(self):
        self.tmp.cleanup()

    def run_gate(self, **env):
        return subprocess.run(["bash", str(PROBE), "host-qual"], env={**self.env, **env},
                              capture_output=True, text=True, timeout=60)

    def test_fast_download_passes(self):
        r = self.run_gate(FAKE_CURL="ok")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertRegex(r.stdout, r"VI-HOST-QUAL download pass mb_s=26\.21 min=5 ")

    def test_under_g_fails(self):
        r = self.run_gate(FAKE_CURL="ok", VI_HOST_QUAL_MIN_DOWNLOAD_MB_S="30")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("VI-HOST-QUAL download fail mb_s=26.21 min=30", r.stdout)

    def test_slow_or_failed_curl_fails(self):
        for mode in ("slow", "fail"):
            with self.subTest(mode=mode):
                r = self.run_gate(FAKE_CURL=mode)
                self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
                self.assertIn("VI-HOST-QUAL download fail", r.stdout)

    def test_a_probe_that_cannot_run_fails(self):
        r = self.run_gate(VI_BIN="/nonexistent/vi")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("VI-HOST-QUAL download fail mb_s=null", r.stdout)
        self.assertIn("detail=ssh_or_timeout_exit_", r.stdout)

    def test_a_hung_ssh_is_killed_and_fails(self):
        r = self.run_gate(FAKE_CURL="hang", VI_HOST_QUAL_DOWNLOAD_SECONDS="5")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("VI-HOST-QUAL download fail", r.stdout)

    def test_unknown_command_is_usage(self):
        r = subprocess.run(["bash", str(PROBE), "nope"], env=self.env, capture_output=True,
                           text=True, timeout=30)
        self.assertEqual(r.returncode, 2)


@unittest.skipUnless(shutil.which("rsync"), "rsync is not installed")
class PushGateShell(tpp.PushProbeTest):
    """push_probe with VI_HOST_QUAL_GATE=1: leg a under the minimum ends the caller with 75."""

    def gate(self, **env):
        return subprocess.run(
            ["bash", "-c", tpp.RUN, "probe", self.work, self.out, self.port, PROBE],
            env={**self.env, "VI_HOST_QUAL_GATE": "1", "VI_PUSH_PROBE_LEGS": "a", **env},
            capture_output=True, text=True, timeout=120)

    def test_gate_passes_and_runs_only_leg_a(self):
        r = self.gate(VI_HOST_QUAL_MIN_PUSH_MB_S="0.001")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("after-probe rc=0", r.stdout)
        self.assertIn("VI-HOST-QUAL push pass", r.stderr)
        self.assertNotIn("leg b 4-streams:", r.stderr)
        self.assert_cleaned()

    def test_gate_fail_exits_75_before_anything_else(self):
        r = self.gate(VI_HOST_QUAL_MIN_PUSH_MB_S="1000000")
        self.assertEqual(r.returncode, 75, r.stdout + r.stderr)
        self.assertNotIn("after-probe", r.stdout)
        self.assertIn("VI-HOST-QUAL push fail", r.stderr)
        self.assertFalse((self.out / ".push-probe").exists())

    def test_no_receiver_in_gate_mode_is_unqualified(self):
        subprocess.run([tpp.RETURN, "stop", self.state], capture_output=True)
        r = self.gate(VI_HOST_QUAL_MIN_PUSH_MB_S="0.001", VI_PUSH_PROBE_LEG_SECONDS="15")
        self.assertEqual(r.returncode, 75, r.stdout + r.stderr)
        self.assertIn("VI-HOST-QUAL push fail", r.stderr)


# PushGateShell inherits PushProbeTest's tests too; run them once, in test_push_probe.
for _name in [n for n in dir(tpp.PushProbeTest) if n.startswith("test_")]:
    setattr(PushGateShell, _name, None)


if __name__ == "__main__":
    unittest.main()
