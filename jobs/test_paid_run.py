"""Tests for jobs/paid_run.py with a stub `vi` (no Vast, no GPU, no network).

The stub (`STUB` below) plays the `vi` CLI: offers, up, down, status and the two dataset stages.
A JSON scenario says what each rental does. It keeps a log of every call, its own credit
account and a fake clock file the orchestrator reads, so hours and spend are exact and fast.
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

JOBS = Path(__file__).resolve().parent
SCRIPT = JOBS / "paid_run.py"

STUB = textwrap.dedent('''\
    #!/usr/bin/env python3
    import json, os, sys, time
    from pathlib import Path
    D = Path(os.environ["STUB_DIR"])
    sc = json.loads((D / "scenario.json").read_text())
    stp = D / "state.json"
    st = json.loads(stp.read_text()) if stp.exists() else {"credit": sc["credit"], "worker": None, "ups": 0}
    def save(): stp.write_text(json.dumps(st))
    def log(*a): (D / "calls.log").open("a").write(json.dumps(a) + "\\n")
    def tick(h):
        f = Path(os.environ["PAID_RUN_CLOCK_FILE"])
        f.write_text(str(float(f.read_text() or 0) + h * 3600) if f.exists() else str(h * 3600))
    def att():
        a = sc["attempts"]; i = min(max(st["ups"], 1) - 1, len(a) - 1); return a[i]
    args = sys.argv[1:]
    cfg = os.environ.get("VI_CLOUD_CONFIG")
    cfgtext = Path(cfg).read_text() if cfg else ""
    if args[:2] == ["cloud", "offers"]:
        log("offers")
        for o in sc["offers"]:
            print(f"{o['id']:>10}  {o.get('gpu','RTX 3090'):<12} 24 GB  ${o['dph']:.3f}/hr  reliability 0.990  "
                  f"net down   900 / up {o.get('up',800):>5} Mbps   8.0 cores  32 GB ram  disk 500 MB/s  {o.get('loc','US')}")
    elif args[:2] == ["cloud", "up"]:
        opt = {args[i]: args[i + 1] for i in range(2, len(args) - 1) if args[i].startswith("--")}
        mh = [l for l in cfgtext.splitlines() if l.startswith("max_hours")]
        log("up", {"offer": opt.get("--offer"), "cap": opt.get("--cap"), "until": opt.get("--until"),
                   "max_hours": mh[0].split("=")[1].strip() if mh else None, "yes": "--yes" in args,
                   "countries": [l for l in cfgtext.splitlines() if l.startswith("countries")]})
        if st["worker"]:
            print("worker already up"); sys.exit(0)
        st["ups"] += 1
        a = att()
        if a.get("up") == "fail":
            print("error: boot failure: machine given back", file=sys.stderr); save(); sys.exit(1)
        st["worker"] = {"id": 100 + st["ups"], "offer": opt.get("--offer"), "credit": st["credit"]}
        tick(a.get("hours", 0))
        save(); print(f"worker {st['worker']['id']} ready: ssh -p 1 root@h  ($0.2/hr)")
    elif args[:2] == ["cloud", "down"]:
        log("down", args[2:])
        if sc.get("down_fail") == "always" or (sc.get("down_fail") == "plain" and "--force" not in args):
            print("error: refused", file=sys.stderr); sys.exit(1)
        if not st["worker"]:
            print("no worker to release"); sys.exit(0)
        w = st["worker"]; spent = att().get("spent", 0.0)
        st["credit"] = round(st["credit"] - spent, 4); st["worker"] = None; save()
        print(f"destroyed instance {w['id']}")
        print(f"credit ${w['credit']:.2f} before, ${st['credit']:.2f} now (${spent:.2f} spent)")
    elif args[:2] == ["cloud", "status"]:
        if st["worker"]:
            print(f"worker {st['worker']['id']} running: RTX 3090 at $0.200/hr, up 0h01m, about $0.00 so far")
        else:
            print("no worker rented")
        print(f"credit ${st['credit']:.2f}")
    elif args[0] == "dataset" and args[1] == "run":
        stage = args[3]
        log("stage", stage, bool(st["worker"]))
        a = att()
        if not st["worker"]:
            print("error: no worker", file=sys.stderr); sys.exit(1)
        def hang():
            (D / f"{stage}.pid").write_text(str(os.getpid())); time.sleep(600)
        if stage == "omnivoice_preflight":
            r = a.get("preflight", "pass")
            if r == "hang": hang()
            if r == "push":
                print("worker-preflight: worker push (MB/s) is 3.10, under the required 4.5", file=sys.stderr)
                print("on-worker.sh: the worker preflight failed; nothing was uploaded", file=sys.stderr); sys.exit(1)
            if r == "vram":
                print("worker-preflight: VRAM (GiB) is 15.6, under the required 23.5", file=sys.stderr); sys.exit(1)
            print("preflight: ok")
        else:
            r = a.get("train", "pass")
            if r == "hang": hang()
            if r == "fail":
                print("train failed: exit 1", file=sys.stderr); sys.exit(1)
            run = Path(os.environ["VI_CHECKPOINT_ROOT"]) / "omnivoice_train" / "run1"
            for s in (1000, 2500, 4500, 5000):
                (run / f"checkpoint-{s}").mkdir(parents=True, exist_ok=True)
            if r != "noverify": (run / "VERIFIED").write_text("{}")
            if r == "marker":
                m = Path(os.environ["VI_STATE_DIR"]) / "unverified"; m.mkdir(parents=True, exist_ok=True)
                (m / "x").write_text(str(run))
            if r == "extra": (run / "checkpoint-3000").mkdir()
            if r != "nolisten":
                for s in (1000, 2500, 5000):
                    p = Path(os.environ["VI_PLATTER_DIR"]) / f"step-{s}"; p.mkdir(parents=True, exist_ok=True)
                    (p / "wav").mkdir(exist_ok=True)
                    rows = []
                    for n in range(2):
                        lab = f"styleA__line{n}__s1__step{s}"
                        if not (r == "nowav" and s == 2500): (p / "wav" / f"{lab}.wav").write_bytes(b"RIFF")
                        rows.append({"label": lab, "text": f'say "this" {n}', "file": f"/worker/{lab}.wav"})
                    (p / "listening.jsonl").write_text("".join(json.dumps(x) + "\\n" for x in rows))
                    (p / "README.txt").write_text("human listening decides")
            print("train: done")
    else:
        print("stub: unknown", args, file=sys.stderr); sys.exit(2)
''')


def offers(*specs):
    return [dict(id=i, dph=d, loc=loc, up=up) for i, d, loc, up in specs]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="paid-run-test."))
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(self.tmp)]))
        self.stub = self.tmp / "vi"
        self.stub.write_text(STUB)
        self.stub.chmod(0o755)
        self.sd = self.tmp / "stub"
        self.sd.mkdir()
        self.env = dict(
            os.environ, VI_BIN=str(self.stub), STUB_DIR=str(self.sd),
            PAID_RUN_CLOCK_FILE=str(self.tmp / "clock"),
            VI_PAID_RUN_STATE_DIR=str(self.tmp / "paid"), VI_STATE_DIR=str(self.tmp / "vistate"),
            VI_CHECKPOINT_ROOT=str(self.tmp / "ckpt"), VI_PLATTER_DIR=str(self.tmp / "platter"))
        self.procs = []
        self.addCleanup(self.reap)

    def reap(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
                p.wait()
        for pidf in self.sd.glob("*.pid"):
            try:
                os.kill(int(pidf.read_text()), signal.SIGKILL)
            except (OSError, ValueError):
                pass

    def scenario(self, attempts, offer_list=None, credit=20.0, **extra):
        offer_list = offer_list or offers((1, 0.20, "US", 800), (2, 0.25, "DE", 900),
                                           (3, 0.30, "CZ", 700), (4, 0.35, "FR", 600))
        (self.sd / "scenario.json").write_text(json.dumps(
            dict(credit=credit, offers=offer_list, attempts=attempts, **extra)))

    def args(self, **kw):
        a = ["--run-id", kw.pop("run_id", "t1"), "--cap", str(kw.pop("cap", 5)),
             "--max-hours", str(kw.pop("max_hours", 8)), "--repo", str(self.tmp),
             "--checkpoint-root", str(self.tmp / "ckpt"), "--platter", str(self.tmp / "platter")]
        for k, v in kw.items():
            a += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
        return a

    def start(self, *a):
        p = subprocess.Popen([sys.executable, "-I", str(SCRIPT), *a], env=self.env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.procs.append(p)
        return p

    def run_orch(self, **kw):
        p = self.start(*self.args(**kw))
        out, _ = p.communicate(timeout=60)
        self.out = out
        return p.returncode

    def calls(self):
        f = self.sd / "calls.log"
        return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []

    def ups(self):
        return [c[1] for c in self.calls() if c[0] == "up"]

    def downs(self):
        return [c for c in self.calls() if c[0] == "down"]

    def stages(self):
        return [c[1] for c in self.calls() if c[0] == "stage"]

    def worker(self):
        return json.loads((self.sd / "state.json").read_text())["worker"]

    def state(self, run_id="t1"):
        return json.loads((self.tmp / "paid" / f"{run_id}.json").read_text())

    def wait_for(self, path, timeout=20):
        end = time.time() + timeout
        while time.time() < end:
            if Path(path).exists():
                return
            time.sleep(0.05)
        self.fail(f"{path} never appeared\n")


class Flow(Base):
    def test_plan_mode_calls_nothing(self):
        self.scenario([{}])
        self.assertEqual(self.run_orch(plan=True), 0)
        self.assertIn("cap (all tries) : $5.00", self.out)
        self.assertIn(str(self.stub), self.out)
        self.assertEqual(self.calls(), [])
        self.assertFalse((self.tmp / "paid").exists())

    def test_bad_args(self):
        self.scenario([{}])
        self.assertEqual(self.run_orch(cap=-1), 2)
        self.assertEqual(self.calls(), [])

    def test_1_happy_path(self):
        self.scenario([dict(spent=0.80, hours=2.0)])
        self.assertEqual(self.run_orch(), 0, self.out)
        self.assertEqual(self.stages(), ["omnivoice_preflight", "omnivoice_train"])
        u = self.ups()
        self.assertEqual(len(u), 1)
        self.assertEqual(u[0]["offer"], "1")  # the cheapest
        self.assertTrue(u[0]["yes"])
        self.assertEqual(u[0]["cap"], "5.00")
        self.assertEqual(float(u[0]["max_hours"]), 8.0)
        self.assertEqual(len(self.downs()), 1)
        self.assertIsNone(self.worker())
        self.assertIn("total actual spend: $0.80 of $5.00", self.out)
        self.assertIn("#1 offer 1", self.out)
        self.assertIn(str(self.tmp / "platter"), self.out)
        self.assertEqual(self.state()["status"], "done")
        for step in (1000, 2500, 5000):  # each set is made listable in the hub (clips.toml)
            toml = (self.tmp / "platter" / f"step-{step}" / "clips.toml").read_text()
            self.assertIn(f"step {step}", toml)
            self.assertIn(f'path = "wav/styleA__line1__s1__step{step}.wav"', toml)
            self.assertNotIn("blind = true", toml)
            if sys.version_info >= (3, 11):
                import tomllib
                self.assertEqual(len(tomllib.loads(toml)["clip"]), 2)

    def test_2_push_gate_fails_once_then_passes(self):
        self.scenario([dict(preflight="push", spent=0.30, hours=1.0), dict(spent=0.90, hours=2.0)])
        self.assertEqual(self.run_orch(), 0, self.out)
        u = self.ups()
        self.assertEqual([x["offer"] for x in u], ["1", "2"])  # failed offer excluded
        self.assertEqual(u[1]["cap"], "4.70")
        self.assertEqual(float(u[1]["max_hours"]), 7.0)
        self.assertEqual(len(self.downs()), 2)
        self.assertEqual(self.stages(), ["omnivoice_preflight", "omnivoice_preflight",
                                         "omnivoice_train"])
        s = self.state()
        self.assertEqual(s["failed_offers"], [1])
        self.assertAlmostEqual(s["spent_total"], 1.20)
        self.assertIn("total actual spend: $1.20", self.out)

    def test_2b_second_offer_same_machine_skipped(self):
        self.scenario([dict(preflight="push", spent=0.1), dict(spent=0.1)],
                      offer_list=offers((1, 0.20, "US", 800)))
        # After attempt 1 the list shows a relisted id 5 for the same machine plus a different one.
        code = self.start_with_relist()
        self.assertEqual(code, 0, self.out)
        self.assertEqual([x["offer"] for x in self.ups()], ["1", "6"])

    def start_with_relist(self):
        # swap the offer list once the first down has happened
        p = self.start(*self.args())
        end = time.time() + 30
        while time.time() < end and not any(c[0] == "down" for c in self.calls()):
            time.sleep(0.02)
        sc = json.loads((self.sd / "scenario.json").read_text())
        sc["offers"] = offers((5, 0.19, "US", 800), (6, 0.40, "SE", 500))
        (self.sd / "scenario.json").write_text(json.dumps(sc))
        self.out, _ = p.communicate(timeout=60)
        return p.returncode

    def test_3_push_gate_fails_every_attempt(self):
        self.scenario([dict(preflight="push", spent=0.2, hours=0.5)])
        self.assertEqual(self.run_orch(max_attempts=3), 5, self.out)
        self.assertEqual([x["offer"] for x in self.ups()], ["1", "2", "3"])
        self.assertEqual(len(self.downs()), 3)
        self.assertEqual(self.stages().count("omnivoice_train"), 0)
        self.assertIsNone(self.worker())
        self.assertAlmostEqual(self.state()["spent_total"], 0.6)

    def test_4_cap_carried_across_attempts_and_refused_when_too_small(self):
        # attempt 1 burns $1.80 of $2.00; attempt 2 would need 8h*$0.25=$2.00 > $0.20 left
        self.scenario([dict(preflight="push", spent=1.80, hours=0.1)])
        self.assertEqual(self.run_orch(cap=2, max_hours=8), 10, self.out)
        self.assertEqual(len(self.ups()), 1)
        self.assertIn("no offer fits what is left", self.out)
        self.assertIsNone(self.worker())

    def test_4b_remaining_cap_is_passed_to_second_up(self):
        self.scenario([dict(preflight="push", spent=1.00, hours=0.5), dict(spent=0.5, hours=1.0)])
        self.assertEqual(self.run_orch(cap=3, max_hours=4), 0, self.out)
        self.assertEqual(self.ups()[1]["cap"], "2.00")

    def test_5_hours_carried_and_enforced(self):
        # attempt 1 uses 7.8 of 8 h: 0.2 h left, under --min-hours 0.5, so no second rental
        self.scenario([dict(preflight="push", spent=0.5, hours=7.8)])
        self.assertEqual(self.run_orch(cap=10, max_hours=8), 10, self.out)
        self.assertEqual(len(self.ups()), 1)
        self.assertIn("not renting again", self.out)

    def test_5b_remaining_hours_in_second_config(self):
        self.scenario([dict(preflight="push", spent=0.5, hours=3.0), dict(spent=0.5, hours=1.0)])
        self.assertEqual(self.run_orch(cap=10, max_hours=8), 0, self.out)
        self.assertAlmostEqual(float(self.ups()[1]["max_hours"]), 5.0, places=1)

    def test_countries_written_to_attempt_config(self):
        self.scenario([dict(spent=0.1)])
        self.assertEqual(self.run_orch(countries="de, cz"), 0, self.out)
        self.assertIn('["DE", "CZ"]', self.ups()[0]["countries"][0])
        self.assertTrue((self.tmp / "paid" / "t1" / "attempt-1.toml").exists())

    def test_6_other_failures_destroy_stop_and_exit_codes(self):
        cases = [
            ("other preflight gate", dict(preflight="vram", spent=0.1), 4, 1),
            ("up error / boot failure", dict(up="fail"), 3, 1),
            ("train failure", dict(train="fail", spent=0.1), 6, 1),
            ("checkpoints never verified", dict(train="noverify", spent=0.1), 7, 1),
            ("unverified marker left", dict(train="marker", spent=0.1), 7, 1),
            ("wrong checkpoint set", dict(train="extra", spent=0.1), 7, 1),
            ("listening sets missing", dict(train="nolisten", spent=0.1), 7, 1),
            ("listening set audio missing", dict(train="nowav", spent=0.1), 7, 1),
        ]
        for name, att, code, nups in cases:
            with self.subTest(name):
                for f in ("calls.log", "state.json"):
                    (self.sd / f).unlink(missing_ok=True)
                subprocess.run(["rm", "-rf", str(self.tmp / "paid"), str(self.tmp / "ckpt"),
                                str(self.tmp / "platter"), str(self.tmp / "vistate")])
                self.scenario([att])
                self.assertEqual(self.run_orch(), code, self.out)
                self.assertEqual(len(self.ups()), nups)  # never a second rental
                self.assertGreaterEqual(len(self.downs()), 1)
                self.assertIsNone(self.worker(), "worker must be gone")
                self.assertEqual(self.state()["exit_code"], code)

    def test_release_failure_is_loud_and_forced_first(self):
        self.scenario([dict(train="fail", spent=0.1)], down_fail="plain")
        self.assertEqual(self.run_orch(), 6, self.out)  # --force worked
        self.assertEqual([c[1] for c in self.downs()], [[], ["--force"]])
        self.assertIsNone(self.worker())

    def test_release_total_failure_exit_9(self):
        self.scenario([dict(train="fail", spent=0.1)], down_fail="always")
        self.assertEqual(self.run_orch(), 9, self.out)
        self.assertIn("COULD NOT RELEASE", self.out)
        self.assertIsNotNone(self.worker())


class ViBinResolution(unittest.TestCase):
    def setUp(self):
        import paid_run
        self.pr = paid_run
        self.tmp = Path(tempfile.mkdtemp(prefix="vi-bin-test."))
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(self.tmp)]))
        self.old_repo = paid_run.REPO
        paid_run.REPO = self.tmp
        self.addCleanup(lambda: setattr(paid_run, "REPO", self.old_repo))
        self.saved = os.environ.pop("VI_BIN", None)
        self.addCleanup(self.restore_env)

    def restore_env(self):
        os.environ.pop("VI_BIN", None)
        if self.saved is not None:
            os.environ["VI_BIN"] = self.saved

    def release(self, mode=0o755):
        f = self.tmp / "target" / "release" / "vi"
        f.parent.mkdir(parents=True)
        f.write_text("#!/bin/sh\n")
        f.chmod(mode)
        return f

    def test_env_wins_over_release_binary(self):
        self.release()
        os.environ["VI_BIN"] = "/some/stub/vi"
        self.assertEqual(self.pr.resolve_vi_bin(), "/some/stub/vi")

    def test_release_binary_is_default_when_present(self):
        f = self.release()
        self.assertEqual(self.pr.resolve_vi_bin(), str(f))

    def test_falls_back_to_path_vi_when_absent(self):
        self.assertEqual(self.pr.resolve_vi_bin(), "vi")

    def test_falls_back_when_release_not_executable(self):
        self.release(0o644)
        self.assertEqual(self.pr.resolve_vi_bin(), "vi")


class KillAndSignals(Base):
    def test_7a_sigkill_in_preflight_then_resume_destroys_and_carries_limits(self):
        self.scenario([dict(preflight="hang", spent=1.0, hours=2.0),
                       dict(spent=0.5, hours=1.0)])
        p = self.start(*self.args(cap=6, max_hours=8))
        self.wait_for(self.sd / "omnivoice_preflight.pid")
        p.kill()
        p.wait()
        os.kill(int((self.sd / "omnivoice_preflight.pid").read_text()), signal.SIGKILL)
        self.assertIsNotNone(self.worker(), "the worker is left up by the kill")
        self.assertEqual(self.state()["status"], "running")
        self.assertEqual(self.run_orch(cap=6, max_hours=8), 0, self.out)
        self.assertEqual(len(self.ups()), 2)  # the first rental was never reused or doubled
        self.assertIn("left open", self.out)
        # destroyed BEFORE the second `up`
        names = [c[0] for c in self.calls() if c[0] in ("up", "down")]
        self.assertEqual(names[:3], ["up", "down", "up"])
        up2 = self.ups()[1]
        self.assertEqual(up2["cap"], "5.00")  # 6 - 1.00 spent
        self.assertAlmostEqual(float(up2["max_hours"]), 6.0, places=1)  # 8 - 2.0 h
        s = self.state()
        self.assertEqual(s["attempts"][0]["outcome"], "interrupted")
        self.assertAlmostEqual(s["spent_total"], 1.5)

    def test_7b_sigkill_in_train_then_resume_adopts_the_worker(self):
        self.scenario([dict(train="hang", spent=1.2, hours=1.0)])
        p = self.start(*self.args())
        self.wait_for(self.sd / "omnivoice_train.pid")
        p.kill()
        p.wait()
        os.kill(int((self.sd / "omnivoice_train.pid").read_text()), signal.SIGKILL)
        (self.sd / "omnivoice_train.pid").unlink()
        sc = json.loads((self.sd / "scenario.json").read_text())
        sc["attempts"][0]["train"] = "pass"  # the reattached train finishes
        (self.sd / "scenario.json").write_text(json.dumps(sc))
        self.assertEqual(self.run_orch(), 0, self.out)
        self.assertEqual(len(self.ups()), 1, "no second rental")
        self.assertIn("adopting it", self.out)
        self.assertEqual(self.stages(), ["omnivoice_preflight", "omnivoice_train", "omnivoice_train"])
        self.assertIsNone(self.worker())
        self.assertAlmostEqual(self.state()["spent_total"], 1.2)

    def test_7c_resume_with_different_limits_refused(self):
        self.scenario([dict(spent=0.1)])
        self.assertEqual(self.run_orch(cap=5), 0)
        # finished runs just report; but a changed cap on an unfinished one is refused
        s = self.state()
        s["status"] = "running"
        (self.tmp / "paid" / "t1.json").write_text(json.dumps(s))
        self.assertEqual(self.run_orch(cap=9), 2, self.out)
        self.assertEqual(len(self.ups()), 1)

    def _signal_in(self, sig, stage):
        self.scenario([dict(**{"preflight" if stage == "pre" else "train": "hang"}, spent=0.4,
                            hours=0.5)])
        p = self.start(*self.args())
        self.wait_for(self.sd / ("omnivoice_preflight.pid" if stage == "pre" else "omnivoice_train.pid"))
        self.assertIsNotNone(self.worker())
        p.send_signal(sig)
        out, _ = p.communicate(timeout=60)
        self.out = out
        return p.returncode

    def test_8_sigterm_tears_down_in_preflight_and_train(self):
        for stage in ("pre", "train"):
            for sig in (signal.SIGTERM, signal.SIGINT):
                with self.subTest(stage=stage, sig=sig.name):
                    for f in ("calls.log", "state.json"):
                        (self.sd / f).unlink(missing_ok=True)
                    for f in self.sd.glob("*.pid"):
                        f.unlink()
                    subprocess.run(["rm", "-rf", str(self.tmp / "paid")])
                    self.assertEqual(self._signal_in(sig, stage), 8, self.out)
                    self.assertIsNone(self.worker(), "worker released after the signal")
                    self.assertEqual(len(self.downs()), 1)
                    self.assertEqual(self.state()["status"], "interrupted")
                    self.assertAlmostEqual(self.state()["spent_total"], 0.4)

    def test_8b_interrupted_run_can_be_resumed(self):
        self.assertEqual(self._signal_in(signal.SIGTERM, "pre"), 8)
        sc = json.loads((self.sd / "scenario.json").read_text())
        sc["attempts"] = [dict(spent=0.4, hours=0.5), dict(spent=0.5, hours=1.0)]
        (self.sd / "scenario.json").write_text(json.dumps(sc))
        (self.sd / "omnivoice_preflight.pid").unlink()
        self.assertEqual(self.run_orch(), 0, self.out)
        self.assertEqual(len(self.ups()), 2)


if __name__ == "__main__":
    unittest.main()
