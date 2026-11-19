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
        # "offers_after_down" is the relisted market: it applies once a worker has been destroyed.
        lst = sc["offers_after_down"] if st.get("downs") and "offers_after_down" in sc else sc["offers"]
        for o in lst:
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
        st["credit"] = round(st["credit"] - spent, 4); st["worker"] = None; st["downs"] = st.get("downs", 0) + 1; save()
        print(f"destroyed instance {w['id']}")
        print(f"credit ${w['credit']:.2f} before, ${st['credit']:.2f} now (${spent:.2f} spent)")
    elif args[:2] == ["cloud", "status"]:
        log("status")
        if st["worker"]:
            print(f"worker {st['worker']['id']} running: RTX 3090 at $0.200/hr, up 0h01m, about $0.00 so far")
        else:
            print("no worker rented")
        print(f"credit ${st['credit']:.2f}")
    elif args[:2] == ["cloud", "ssh"]:
        log("ssh")
        mode = sc.get("diag", "ok")
        if mode == "hang":
            (D / "diag.pid").write_text(str(os.getpid())); print("partial nvidia-smi line", flush=True); time.sleep(600)
        if mode == "fail":
            print("ssh: connection refused", file=sys.stderr); sys.exit(255)
        if mode == "big":
            sys.stdout.write("x" * 700000); sys.exit(0)
        print("== nvidia-smi (raw)\\n| NVIDIA-SMI 550   CUDA Version: N/A |\\n== pip freeze\\ntorch==2.8.0")
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
                print("worker-preflight: worker push (MB/s) is 1.80, under the required 2.5", file=sys.stderr)
                print("on-worker.sh: the worker preflight failed; nothing was uploaded", file=sys.stderr); sys.exit(1)
            gates = dict(
                vram="VRAM (GiB) is 15.6, under the required 23.5",
                disk="free disk (GiB) is 60.1, under the required 100",
                ram="RAM (GiB) is 15.6, under the required 28",
                cuda="CUDA version is 12.4, under the required 12.8",
                cores="effective CPU cores is 2.0, under the required 4",
            )
            if r in gates:
                print("worker-preflight: " + gates[r], file=sys.stderr)
                print("on-worker.sh: the worker preflight failed; nothing was uploaded", file=sys.stderr); sys.exit(1)
            if r == "upload":
                print("on-worker.sh: worker upload ETA is 41 min, over the 30 min limit, so nothing was sent", file=sys.stderr); sys.exit(1)
            if r == "other":
                print("on-worker.sh: input upload failed", file=sys.stderr); sys.exit(1)
            if r == "ok-then-fail":
                print("worker-preflight: ok (VRAM 23.9 GiB, push 9 MB/s)", file=sys.stderr)
                print("on-worker.sh: input upload failed", file=sys.stderr); sys.exit(1)
            print("preflight: ok")
        else:
            r = a.get("train", "pass")
            if r == "hang": hang()
            if r == "fail":
                print("train failed: exit 1", file=sys.stderr); sys.exit(1)
            if r == "lines":
                # a fake trainer: print the scenario's lines, then (optionally) wait for the test
                # to release a checkpoint ack, then stay up until killed
                for ln in a.get("lines", []): print(ln, flush=True)
                (D / "train.ready").write_text("1")
                if a.get("ack_wait"):
                    for _ in range(600):
                        if (D / "ack.go").exists(): break
                        time.sleep(0.1)
                    print("on-worker.sh: checkpoint 1000 verified home and acked", flush=True)
                hang()
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


def load_paid_run():
    import importlib.util
    spec = importlib.util.spec_from_file_location("paid_run_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def install_fake_realpath(home, rc=0, text="real-path check: OK (3 batches, forward skipped)"):
    """A fake home CPU venv + base snapshot under `home`: its python prints `text` and exits `rc`,
    appending its argv to home/realpath-calls.log. The real subprocess path in paid_run runs it."""
    pr = load_paid_run()
    py = pr.realpath_venv(home) / "bin" / "python"
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text(f"#!/bin/sh\necho \"$@\" >> {home}/realpath-calls.log\necho '{text}'\nexit {rc}\n")
    py.chmod(0o755)
    snap = (Path(home) / ".cache" / "huggingface" / "hub" / "models--k2-fsa--OmniVoice" / "snapshots"
            / pr.base_model_revision() / "audio_tokenizer")
    snap.mkdir(parents=True, exist_ok=True)
    return py



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
        # hours left = limit minus real elapsed time; the diagnostics pull adds a fraction of a second
        self.assertAlmostEqual(float(u[1]["max_hours"]), 7.0, delta=0.01)
        self.assertEqual(len(self.downs()), 2)
        self.assertEqual(self.stages(), ["omnivoice_preflight", "omnivoice_preflight",
                                         "omnivoice_train"])
        s = self.state()
        self.assertEqual(s["failed_offers"], [1])
        self.assertAlmostEqual(s["spent_total"], 1.20)
        self.assertIn("total actual spend: $1.20", self.out)

    def test_2c_every_cheap_gate_retries_with_the_next_offer(self):
        for gate in ("vram", "disk", "ram", "cuda", "cores", "upload"):
            with self.subTest(gate):
                for f in ("calls.log", "state.json"):
                    (self.sd / f).unlink(missing_ok=True)
                subprocess.run(["rm", "-rf", str(self.tmp / "paid"), str(self.tmp / "ckpt"),
                                str(self.tmp / "platter"), str(self.tmp / "vistate")])
                self.scenario([dict(preflight=gate, spent=0.1, hours=0.5),
                               dict(spent=0.2, hours=1.0)])
                self.assertEqual(self.run_orch(), 0, self.out)
                self.assertEqual([x["offer"] for x in self.ups()], ["1", "2"])
                self.assertEqual(len(self.downs()), 2)
                self.assertEqual(self.state()["failed_offers"], [1])
                self.assertEqual(self.stages().count("omnivoice_train"), 1)

    def test_2d_gate_failures_never_rent_the_same_offer_twice_and_stop_at_max_attempts(self):
        self.scenario([dict(preflight="disk", spent=0.1, hours=0.2)])
        self.assertEqual(self.run_orch(max_attempts=2), 5, self.out)
        self.assertEqual([x["offer"] for x in self.ups()], ["1", "2"])
        self.assertIsNone(self.worker())
        self.assertEqual(self.stages().count("omnivoice_train"), 0)

    def test_2e_the_ok_line_alone_is_not_a_gate_failure(self):
        paid_run = load_paid_run()
        self.assertIsNone(paid_run.retryable_gate("worker-preflight: ok (VRAM 23.9 GiB)"))
        self.assertEqual(paid_run.retryable_gate("worker-preflight: RAM (GiB) is 1, under"),
                         "worker-preflight")
        self.assertEqual(paid_run.retryable_gate("worker-preflight: worker push (MB/s) is 1"),
                         "push-speed")

    def test_2b_second_offer_same_machine_skipped(self):
        self.scenario([dict(preflight="push", spent=0.1), dict(spent=0.1)],
                      offer_list=offers((1, 0.20, "US", 800)))
        # After attempt 1 the list shows a relisted id 5 for the same machine plus a different one.
        # The stub switches lists on its own state (the first destroyed worker), so no timing matters.
        sc = json.loads((self.sd / "scenario.json").read_text())
        sc["offers_after_down"] = offers((5, 0.19, "US", 800), (6, 0.40, "SE", 500))
        (self.sd / "scenario.json").write_text(json.dumps(sc))
        self.assertEqual(self.run_orch(), 0, self.out)
        self.assertEqual([x["offer"] for x in self.ups()], ["1", "6"])

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
            ("preflight failure that is no gate", dict(preflight="other", spent=0.1), 4, 1),
            ("preflight passed then failed", dict(preflight="ok-then-fail", spent=0.1), 4, 1),
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


def load_paid_run():
    import importlib.util
    spec = importlib.util.spec_from_file_location("paid_run_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DiskGate(Base):
    """The run config's disk_gb must clear the worker's free-disk gate before anything is rented."""

    def cfg(self, disk, name="c.toml"):
        p = self.tmp / name
        p.write_text(f"disk_gb = {disk}\nmax_hours = 8\n[search]\nmax_dph = 0.31\n")
        return str(p)

    def test_gate_default_is_read_from_the_worker_script(self):
        paid_run = load_paid_run()
        self.assertEqual(paid_run.gate_default_gb(), 100.0)
        self.assertEqual(paid_run.GATE_FALLBACK_GB, paid_run.gate_default_gb())

    def test_100_fails_and_calls_nothing(self):
        self.scenario([{}])
        self.assertEqual(self.run_orch(config=self.cfg(100)), 11, self.out)
        self.assertIn("FAIL", self.out)
        self.assertIn("Nothing was rented", self.out)
        self.assertEqual(self.calls(), [])
        self.assertFalse((self.tmp / "paid").exists())

    def test_125_passes(self):
        self.scenario([dict(spent=0.5, hours=1.0)])
        self.assertEqual(self.run_orch(config=self.cfg(125)), 0, self.out)
        self.assertIn("disk_gb 125 - overhead 5 vs gate 100 GiB free: OK", self.out)

    def test_boundary(self):
        self.scenario([{}])
        self.assertEqual(self.run_orch(config=self.cfg(104), plan=True), 11, self.out)
        self.assertEqual(self.run_orch(config=self.cfg(105), plan=True), 0, self.out)
        self.assertEqual(self.calls(), [])

    def test_env_override_of_the_gate(self):
        self.scenario([{}])
        self.env["VI_MIN_DISK_GB"] = "50"
        self.assertEqual(self.run_orch(config=self.cfg(100), plan=True), 0, self.out)
        self.assertIn("gate 50 GiB", self.out)
        self.env["VI_MIN_DISK_GB"] = "121"
        self.assertEqual(self.run_orch(config=self.cfg(125), plan=True), 11, self.out)

    def test_plan_shows_the_line_and_fails_nonzero(self):
        self.scenario([{}])
        self.assertEqual(self.run_orch(config=self.cfg(100), plan=True), 11, self.out)
        self.assertIn("worker disk     : disk_gb 100 - overhead 5 vs gate 100 GiB free: FAIL",
                      self.out)
        self.assertNotIn("--plan: nothing was called", self.out)
        self.assertEqual(self.calls(), [])

    def test_missing_disk_gb_uses_the_vast_default_and_fails(self):
        p = self.tmp / "n.toml"
        p.write_text("max_hours = 8\n")
        self.scenario([{}])
        self.assertEqual(self.run_orch(config=str(p), plan=True), 11, self.out)

    def test_checked_in_configs_pass(self):
        paid_run = load_paid_run()
        ok, disk, gate, line = paid_run.disk_check(paid_run.DEFAULT_CONFIG.read_text(), {})
        self.assertTrue(ok, line)
        self.assertEqual(disk, 125)
        for toml in (JOBS / "paid-run").glob("*.toml"):
            self.assertTrue(paid_run.disk_check(toml.read_text(), {})[0], toml)


class HomeChecks(Base):
    """Home's disk gate and the cache note run in --plan / before `vi cloud up`."""

    def stages_file(self, floor, reserve=27, name="stages.toml"):
        p = self.tmp / name
        # "epoch" has no retention term, so a small scratch disk can pass the small floors.
        body = (f'checkpoint_format = "epoch"\nmin_home_gb = {floor}\n'
                f"home_gate_reserve_gb = {reserve}\n")
        p.write_text(f"[stage.omnivoice_preflight.settings]\n{body}"
                     f"[stage.omnivoice_train.settings]\n{body}")
        return str(p)

    def test_short_home_disk_rents_nothing_and_exits_11(self):
        self.scenario([dict(spent=0.1, hours=1.0)])
        self.env["HOME"] = str(self.tmp)
        self.assertEqual(self.run_orch(stages=self.stages_file(100000000)), 11, self.out)
        self.assertIn("home disk", self.out)
        self.assertIn("FAIL", self.out)
        self.assertIn("Nothing was rented", self.out)
        self.assertEqual(self.calls(), [])
        self.assertFalse((self.tmp / "paid").exists())

    def test_short_home_disk_also_fails_plan(self):
        self.scenario([{}])
        self.env["HOME"] = str(self.tmp)
        self.assertEqual(self.run_orch(stages=self.stages_file(100000000), plan=True), 11, self.out)
        self.assertEqual(self.calls(), [])

    def test_enough_home_disk_passes_and_plan_shows_the_lines(self):
        self.scenario([{}])
        self.env["HOME"] = str(self.tmp)
        self.assertEqual(self.run_orch(stages=self.stages_file(1, 0), plan=True), 0, self.out)
        self.assertIn("home disk       : omnivoice_preflight:", self.out)
        self.assertIn("floor 1 + reserve 0 GiB: OK", self.out)
        self.assertIn("pinned revision) not verified (needs network)", self.out)

    def test_bundle_cache_empty_and_warm_notes(self):
        self.scenario([{}])
        self.env["HOME"] = str(self.tmp)
        self.env["VI_BUNDLE_DIR"] = str(self.tmp / "bundles")
        st = self.stages_file(1, 0)
        self.assertEqual(self.run_orch(stages=st, plan=True), 0, self.out)
        self.assertIn("bundle cache    : not checked", self.out)
        self.assertNotIn("bundle files", self.out)
        (self.tmp / "bundles").mkdir()
        (self.tmp / "bundles" / "abc.tar.zst").write_bytes(b"x" * 2048)
        self.assertEqual(self.run_orch(stages=st, plan=True), 0, self.out)
        self.assertIn("bundle files    : 1 bundle(s)", self.out)
        self.assertNotIn("warm if", self.out)

    def folder_fixture(self):
        corpus = self.tmp / "corpus"
        out = corpus / "stages" / "omnivoice_data" / "k1" / "out"
        (out / "sub dir").mkdir(parents=True)
        (out / "a.txt").write_text("hello")
        (out / "sub dir" / "B b.bin").write_bytes(b"\0" * 5000)
        (out / "Z").write_text("z")
        os.symlink("a.txt", out / "link")
        (corpus / "stages" / "omnivoice_data" / "current").write_text("k1\n")
        st = self.tmp / "ck.toml"
        st.write_text(f'corpus = "{corpus}"\n[stage.omnivoice_preflight]\n'
                      'inputs = ["omnivoice_data"]\n[stage.omnivoice_train]\n'
                      'inputs = ["omnivoice_data"]\n')
        return out, st

    def test_python_bundle_key_equals_the_shell_key(self):
        out, _ = self.folder_fixture()
        shell = subprocess.run(
            ["bash", "-c", 'cd "$1" && { pwd -P && find . -printf \'%P %s %T@\\n\' | LC_ALL=C sort; }'
             " | sha256sum | cut -c1-16", "_", str(out)], capture_output=True, text=True)
        self.assertEqual(shell.returncode, 0, shell.stderr)
        self.assertEqual(load_paid_run().bundle_key(out), shell.stdout.strip())

    def test_bundle_cache_line_not_built_then_built_and_current(self):
        paid_run = load_paid_run()
        out, st = self.folder_fixture()
        env = {"VI_BUNDLE_DIR": str(self.tmp / "b"), "VI_PACK_MIN_KB": "1"}
        self.assertIn("not built, will pack before renting",
                      paid_run.bundle_cache_status(str(st), str(self.tmp), str(self.tmp), env))
        key = paid_run.bundle_key(out)
        (self.tmp / "b").mkdir()
        (self.tmp / "b" / f"{key}.tar.zst").write_bytes(b"x")
        self.assertEqual(f"bundle cache    : built and current ({key})",
                         paid_run.bundle_cache_status(str(st), str(self.tmp), str(self.tmp), env))
        (out / "a.txt").write_text("changed size")  # a changed listing is a different key
        self.assertIn("not built", paid_run.bundle_cache_status(str(st), str(self.tmp),
                                                               str(self.tmp), env))

    def test_small_inputs_are_not_packed(self):
        _, st = self.folder_fixture()
        env = {"VI_BUNDLE_DIR": str(self.tmp / "b"), "VI_PACK_MIN_KB": "100000"}
        self.assertIn("no input is large enough", load_paid_run().bundle_cache_status(
            str(st), str(self.tmp), str(self.tmp), env))

    def test_plan_shows_disk_billing_bound(self):
        self.scenario([{}])
        self.env["HOME"] = str(self.tmp)
        cfg = self.tmp / "c.toml"
        cfg.write_text("disk_gb = 125\nmax_hours = 8\n[search]\nmax_dph = 0.31\n")
        self.assertEqual(self.run_orch(stages=self.stages_file(1, 0), plan=True, config=str(cfg)),
                         0, self.out)
        self.assertIn("worst case for 125 GB over 8 h is about $1.37 at an assumed $1.00/GB/month",
                      self.out)
        self.assertIn("estimate", self.out)

    def test_disk_cost_formula(self):
        f = load_paid_run().disk_cost_line
        self.assertIn("$1.37", f(125, 8))
        self.assertIn("$0.34", f(125, 2))

    def test_checked_in_stage_file_settings_are_read(self):
        paid_run = load_paid_run()
        ok, lines = paid_run.home_check("characters/tali/omnivoice-ft.toml", str(JOBS.parent),
                                        home=str(self.tmp), env={})
        self.assertTrue(any("floor 50 + reserve 27 GiB" in ln for ln in lines), lines)

    def test_base_revision_is_read_from_the_stage_script(self):
        self.assertRegex(load_paid_run().base_model_revision(), r"^[0-9a-f]{40}$")


class StatusAndStop(Base):
    def setUp(self):
        super().setUp()
        self.env["PAID_RUN_PROGRESS_SECONDS"] = "0"

    PUSH_LINES = ["  25%|██▍       | 1200/5000 [10:00<30:00,  2.00it/s]"]

    def cli(self, *a):
        r = subprocess.run([sys.executable, "-I", str(SCRIPT), *a], env=self.env, text=True,
                           capture_output=True, timeout=60)
        self.out = r.stdout + r.stderr
        return r.returncode

    def status(self, run_id="t1", *extra):
        rc = self.cli("status", "--run-id", run_id, "--json", *extra)
        self.assertEqual(rc, 0, self.out)
        return json.loads(self.out)

    def wait_step(self, step, run_id="t1", timeout=20):
        end = time.time() + timeout
        while time.time() < end:
            try:
                if (self.state(run_id).get("progress") or {}).get("step") == step:
                    return
            except (OSError, ValueError):
                pass
            time.sleep(0.02)
        self.fail(f"progress never reached step {step}")

    def start_train(self, lines, **extra):
        self.scenario([dict(train="lines", lines=lines, spent=0.5, hours=1.0, **extra)])
        p = self.start(*self.args())
        self.wait_for(self.sd / "train.ready")
        return p

    def read_until(self, p, text, timeout=30):
        end = time.time() + timeout
        seen = []
        while time.time() < end:
            ln = p.stdout.readline()
            if not ln:
                break
            seen.append(ln)
            if text in ln:
                return "".join(seen)
        self.fail(f"never saw {text!r} in:\n" + "".join(seen))

    def finish(self, p):
        out, _ = p.communicate(timeout=60)
        self.out = out
        return p.returncode

    # ----- status -----
    def test_status_running_shows_everything_and_makes_no_vi_call(self):
        lines = [f"  2%|▏ | {n}/5000 [00:01<1:00:00, 1.0it/s]" for n in (100, 200, 300)]
        p = self.start_train(lines)
        self.wait_step(300)
        before = len(self.calls())
        self.assertEqual(self.cli("status", "--run-id", "t1"), 0, self.out)
        for want in ("run t1: running", "phase    : train; attempt 1/3", "offer 1 at $0.200/hr",
                     "(estimate)", "step 300/5000", "ETA about", "last stage output"):
            self.assertIn(want, self.out)
        st = self.status()
        self.assertEqual(len(self.calls()), before, "status without --credit called vi")
        self.assertTrue(st["process_alive"])
        self.assertEqual(st["phase"], "train")
        self.assertEqual((st["train"]["step"], st["train"]["total"]), (300, 5000))
        self.assertTrue(st["spend"]["total_is_estimate"])
        self.assertAlmostEqual(st["cap_left"], 5 - st["spend"]["total"])
        self.assertTrue(any("300/5000" in x for x in st["stage_log_tail"]))
        self.status("t1", "--credit")
        self.assertEqual(len(self.calls()), before + 1, "--credit makes exactly one vi call")
        self.assertEqual(self.calls()[-1], ["status"])
        self.assertEqual(self.cli("stop", "--run-id", "t1", "--now"), 0, self.out)
        self.assertEqual(self.finish(p), 8)

    def test_status_json_shape_and_credit(self):
        p = self.start_train(self.PUSH_LINES)
        self.wait_step(1200)
        st = self.status("t1", "--credit")
        self.assertEqual(set(st), {
            "run_id", "status", "exit_code", "message", "phase", "attempt", "offer",
            "price_per_hour", "spend", "hours_used", "max_hours", "hours_left", "cap",
            "cap_left", "train", "process_alive", "stop_requested", "stage_log_tail", "credit"})
        self.assertEqual(set(st["spend"]), {"closed_attempts", "open_attempt_estimate", "total",
                                            "total_is_estimate"})
        self.assertEqual(set(st["train"]), {"step", "total", "updated_at", "eta_seconds",
                                            "eta_is_estimate"})
        self.assertEqual(st["credit"], 20.0)
        self.assertEqual(st["attempt"], {"n": 1, "max": 3})
        self.cli("stop", "--run-id", "t1", "--now")
        self.finish(p)

    def test_status_done_failed_interrupted_and_unknown(self):
        self.scenario([dict(spent=0.8, hours=2.0)])
        self.assertEqual(self.run_orch(), 0, self.out)
        n = len(self.calls())
        st = self.status()
        self.assertEqual((st["status"], st["exit_code"], st["phase"]), ("done", 0, None))
        self.assertFalse(st["process_alive"])
        self.assertAlmostEqual(st["spend"]["total"], 0.8)
        self.assertFalse(st["spend"]["total_is_estimate"])
        self.assertEqual(len(self.calls()), n)
        self.scenario([dict(train="fail", spent=0.1)])
        self.assertEqual(self.run_orch(run_id="f1"), 6)
        st = self.status("f1")
        self.assertEqual((st["status"], st["exit_code"]), ("failed", 6))
        self.assertIn("train stage failed", st["message"])
        self.scenario([dict(preflight="hang", spent=0.4, hours=0.5)])
        p = self.start(*self.args(run_id="i1"))
        self.wait_for(self.sd / "omnivoice_preflight.pid")
        p.send_signal(signal.SIGTERM)
        self.assertEqual(self.finish(p), 8)
        self.assertEqual(self.status("i1")["status"], "interrupted")
        before = {f.name: f.read_bytes() for f in (self.tmp / "paid").glob("*.json")}
        self.assertEqual(self.cli("status", "--run-id", "nope"), 2)
        self.assertIn("no run with id", self.out)
        self.assertEqual(self.cli("stop", "--run-id", "nope"), 2)
        self.assertEqual(before, {f.name: f.read_bytes() for f in (self.tmp / "paid").glob("*.json")})
        self.assertFalse(list((self.tmp / "paid").glob("nope*")))

    def test_progress_parsing_and_eta_needs_three_samples(self):
        import paid_run as pr
        self.assertEqual(pr.parse_step("  3%|▎  | 150/5000 [02:10<1:10:00,  1.15it/s]"), (150, 5000))
        self.assertEqual(pr.parse_step("Step 150/5000 loss 1.2"), (150, 5000))
        self.assertEqual(pr.parse_step("step: 7 of 5000"), (7, 5000))
        self.assertEqual(pr.parse_step("Epoch 1/4: 50%|█| 100/2500 [00:01<00:01, 9it/s]"),
                         (100, 2500))
        self.assertIsNone(pr.parse_step("loading model 3 of 4 shards"))
        self.assertIsNone(pr.parse_step("on-worker.sh: poll failed"))
        self.assertIsNone(pr.eta_seconds([[0, 0], [10, 100]], 100, 5000))
        self.assertIsNone(pr.eta_seconds([[0, 100], [5, 100], [10, 100]], 100, 5000))
        self.assertAlmostEqual(pr.eta_seconds([[0, 0], [10, 100], [20, 200]], 200, 5000), 480.0)
        # end to end: two samples give no ETA, a third gives one
        p = self.start_train(["Step 100/5000", "noise", "step 200/5000"])
        self.wait_step(200)
        t = self.status()["train"]
        self.assertEqual((t["step"], t["total"], t["eta_seconds"]), (200, 5000, None))
        self.cli("stop", "--run-id", "t1", "--now")
        self.finish(p)

    def test_progress_write_is_throttled(self):
        self.env["PAID_RUN_PROGRESS_SECONDS"] = "3600"
        p = self.start_train(["Step 100/5000", "Step 200/5000", "Step 300/5000"])
        self.wait_for(self.tmp / "paid" / "t1" / "stage-tail.log")
        end = time.time() + 20
        while time.time() < end and "Step 300/5000" not in p.stdout.readline():
            pass
        self.assertEqual(self.state()["progress"]["step"], 100)  # first write only
        self.cli("stop", "--run-id", "t1", "--now")
        self.finish(p)

    # ----- stop -----
    def test_stop_during_preflight_is_immediate(self):
        self.scenario([dict(preflight="hang", spent=0.4, hours=0.5)])
        p = self.start(*self.args())
        self.wait_for(self.sd / "omnivoice_preflight.pid")
        self.assertEqual(self.cli("stop", "--run-id", "t1"), 0, self.out)
        self.assertTrue((self.tmp / "paid" / "t1.stop").exists())
        self.assertEqual(self.finish(p), 8, self.out)
        self.assertIn("released at once", self.out)
        self.assertIsNone(self.worker())
        self.assertEqual(len(self.downs()), 1)
        self.assertEqual(len(self.ups()), 1, "stop never rents")
        s = self.state()
        self.assertEqual((s["status"], s["exit_code"]), ("interrupted", 8))
        self.assertAlmostEqual(s["spent_total"], 0.4)
        self.assertFalse((self.tmp / "paid" / "t1.stop").exists())

    def test_stop_during_train_waits_for_the_checkpoint_push(self):
        p = self.start_train(self.PUSH_LINES, ack_wait=True)
        self.wait_step(1200)
        self.assertEqual(self.cli("stop", "--run-id", "t1"), 0, self.out)
        seen = self.read_until(p, "waiting for the checkpoint push (checkpoint 1000)")
        self.assertEqual(len(self.downs()), 0, "torn down before the push finished")
        self.assertIsNotNone(self.worker())
        (self.sd / "ack.go").write_text("1")
        self.assertEqual(self.finish(p), 8, seen + self.out)
        self.assertIn("checkpoint 1000 verified home and acked", self.out)
        self.assertIn("checkpoint 1000 reached home", self.out)
        self.assertIsNone(self.worker())
        self.assertEqual(len(self.downs()), 1)
        self.assertEqual(self.state()["status"], "interrupted")
        self.assertIn("total actual spend", self.out)

    def test_stop_wait_times_out_and_tears_down_anyway(self):
        self.env["PAID_RUN_STOP_WAIT_SECONDS"] = "1"
        p = self.start_train(self.PUSH_LINES, ack_wait=True)
        self.wait_step(1200)
        self.assertEqual(self.cli("stop", "--run-id", "t1"), 0, self.out)
        self.assertEqual(self.finish(p), 8, self.out)
        self.assertIn("gave up waiting for checkpoint 1000", self.out)
        self.assertIsNone(self.worker())
        self.assertEqual(len(self.downs()), 1)

    def test_stop_with_no_push_pending_or_unreadable_step_is_immediate(self):
        p = self.start_train(["Step 300/5000"])  # newest checkpoint would be 0: nothing pending
        self.wait_step(300)
        self.cli("stop", "--run-id", "t1")
        self.assertEqual(self.finish(p), 8, self.out)
        self.assertIn("no checkpoint push was pending", self.out)
        self.assertIsNone(self.worker())
        for f in ("calls.log", "state.json", "train.ready"):
            (self.sd / f).unlink(missing_ok=True)
        subprocess.run(["rm", "-rf", str(self.tmp / "paid")])
        p = self.start_train(["no step lines here"])
        self.cli("stop", "--run-id", "t1")
        self.assertEqual(self.finish(p), 8, self.out)
        self.assertIn("could not be read", self.out)
        self.assertIsNone(self.worker())

    def test_stop_now_signals_the_recorded_process(self):
        p = self.start_train(self.PUSH_LINES, ack_wait=True)
        self.wait_step(1200)
        pid = self.state()["pid"]
        self.assertEqual(pid, p.pid)
        self.assertEqual(self.cli("stop", "--run-id", "t1", "--now"), 0, self.out)
        self.assertEqual(self.finish(p), 8, self.out)  # no waiting for the ack
        self.assertNotIn("waiting for the checkpoint push", self.out)
        self.assertIsNone(self.worker())
        self.assertEqual(len(self.downs()), 1)

    def test_stop_now_never_signals_a_pid_that_is_not_this_run(self):
        self.scenario([dict(spent=0.1)])
        self.assertEqual(self.run_orch(), 0)
        bystander = subprocess.Popen(["sleep", "60"])
        self.addCleanup(bystander.kill)
        s = self.state()
        s.update(status="running", pid=bystander.pid, pid_start=1)
        (self.tmp / "paid" / "t1.json").write_text(json.dumps(s))
        self.assertEqual(self.cli("stop", "--run-id", "t1", "--now"), 2, self.out)
        self.assertIn("nothing changed", self.out.lower())
        self.assertIsNone(bystander.poll(), "an unrelated process was signalled")

    def test_stop_on_a_finished_run_refuses_and_changes_nothing(self):
        self.scenario([dict(spent=0.1)])
        self.assertEqual(self.run_orch(), 0)
        before = (self.tmp / "paid" / "t1.json").read_bytes()
        self.assertEqual(self.cli("stop", "--run-id", "t1"), 2)
        self.assertIn("is not running", self.out)
        self.assertEqual(self.cli("stop", "--run-id", "t1", "--now"), 2)
        self.assertEqual((self.tmp / "paid" / "t1.json").read_bytes(), before)
        self.assertFalse((self.tmp / "paid" / "t1.stop").exists())

    def test_stale_stop_file_is_not_honoured_by_a_new_run(self):
        (self.tmp / "paid").mkdir()
        (self.tmp / "paid" / "t1.stop").write_text("stop\n")
        self.scenario([dict(spent=0.1)])
        self.assertEqual(self.run_orch(), 0, self.out)
        self.assertEqual(self.stages(), ["omnivoice_preflight", "omnivoice_train"])
        self.assertFalse((self.tmp / "paid" / "t1.stop").exists())


class ViBinResolution(unittest.TestCase):
    def setUp(self):
        paid_run = load_paid_run()
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


class IdsAndMarkers(Base):
    """The whole-bundle id check (blocks renting) and the read-only stale-marker plan line."""

    def stages_with_data(self, train_ids, dev_ids):
        st = self.tmp / "stages.toml"
        st.write_text('corpus = "corpus"\n[stage.omnivoice_preflight]\ninputs = ["omnivoice_data"]\n'
                      '[stage.omnivoice_preflight.settings]\ncheckpoint_format = "epoch"\n'
                      'min_home_gb = 1\nhome_gate_reserve_gb = 0\n')
        base = self.tmp / "corpus" / "stages" / "omnivoice_data"
        out = base / "abc123" / "out"
        out.mkdir(parents=True)
        (base / "current").write_text("abc123")
        rows = lambda ids: "".join(json.dumps({"id": i}) + "\n" for i in ids)  # noqa: E731
        (out / "train_manifest.jsonl").write_text(rows(train_ids))
        (out / "dev_manifest.jsonl").write_text(rows(dev_ids))
        (out / "style_manifest.jsonl").write_text(rows(dev_ids))
        (out / "replay_texts.jsonl").write_text('{"text": "x"}\n')
        return str(st)

    def test_dotted_ids_block_renting_and_plan(self):
        self.scenario([{}])
        self.env["HOME"] = str(self.tmp)
        st = self.stages_with_data(["le2.s1_m", "le2.s2_m"], ["le3.s9_m"])
        self.assertEqual(self.run_orch(stages=st, plan=True), 12, self.out)
        self.assertIn("clip ids        : FAIL", self.out)
        self.assertIn("2 of 2 train id(s)", self.out)
        self.assertIn("'le2.s1_m'", self.out)
        self.assertEqual(self.run_orch(stages=st), 12, self.out)
        self.assertIn("Nothing was rented", self.out)
        self.assertEqual(self.calls(), [])
        self.assertFalse((self.tmp / "paid").exists())

    def test_clean_ids_pass(self):
        self.scenario([dict(spent=0.1, hours=1.0)])
        self.env["HOME"] = str(self.tmp)
        st = self.stages_with_data(["le2_s1_m"], ["le3_s9_m"])
        install_fake_realpath(self.tmp)
        self.assertEqual(self.run_orch(stages=st, plan=True), 0, self.out)
        self.assertIn("clip ids        : OK", self.out)

    def test_plan_lists_stale_markers_and_deletes_nothing(self):
        self.scenario([{}])
        self.env["HOME"] = str(self.tmp)
        empty, full = self.tmp / "ck-empty", self.tmp / "ck-full"
        empty.mkdir()
        full.mkdir()
        (full / "model.safetensors").write_bytes(b"x")
        mdir = self.tmp / "vistate" / "unverified"
        mdir.mkdir(parents=True)
        (mdir / "a").write_text(str(empty))
        (mdir / "b").write_text(str(full))
        (mdir / "c").write_text(str(self.tmp / "ck-gone"))
        st = HomeChecks.stages_file(self, 1, 0)
        self.assertEqual(self.run_orch(stages=st, plan=True), 0, self.out)
        self.assertIn("stale markers   : 2 unverified marker(s)", self.out)
        self.assertIn(f"checkpoints.py abandon {empty} ", self.out)
        self.assertIn("ck-gone", self.out)
        self.assertNotIn(f"abandon {full} ", self.out)
        self.assertEqual(sorted(p.name for p in mdir.iterdir()), ["a", "b", "c"])
        self.assertTrue(empty.is_dir())


class RealPathCheck(Base):
    """The home real-path check in the plan: runs, caches by code hash, blocks with exit 13."""

    def setUp(self):
        super().setUp()
        self.pr = load_paid_run()
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.env["HOME"] = str(self.home)
        self.st = IdsAndMarkers.stages_with_data(self, ["le2_s1_m"], ["le3_s9_m"])
        self.calls_log = self.home / "realpath-calls.log"

    def run_check(self, **kw):
        return self.pr.realpath_check(self.st, str(self.tmp), home=self.home, **kw)

    def n_runs(self):
        return len(self.calls_log.read_text().splitlines()) if self.calls_log.exists() else 0

    def test_pass_runs_once_then_cache_hit(self):
        install_fake_realpath(self.home)
        ok, lines = self.run_check(mem_mb=20000)
        self.assertTrue(ok, lines)
        self.assertIn("OK (3 batches", lines[0])
        self.assertEqual(self.n_runs(), 1)
        cmd = self.calls_log.read_text()
        self.assertNotIn("--forward", cmd)
        ok, lines = self.run_check(mem_mb=20000)
        self.assertTrue(ok)
        self.assertIn("OK (cached", lines[0])
        self.assertEqual(self.n_runs(), 1)  # not run again

    def test_data_folder_id_is_part_of_the_cache_key(self):
        other = self.tmp / "other-data" / "zzz" / "out"
        other.mkdir(parents=True)
        h1 = self.pr.realpath_code_hash(self.tmp / "a" / "x" / "out")
        h2 = self.pr.realpath_code_hash(other)
        self.assertNotEqual(h1, h2)  # the data folder id is part of the key

    def test_failure_blocks_plan_and_run_with_exit_13_and_is_not_cached(self):
        install_fake_realpath(self.home, rc=1, text="AttributeError: no text_tokenizer")
        self.scenario([{}])
        self.assertEqual(self.run_orch(stages=self.st, plan=True), 13, self.out)
        self.assertIn("real-path check : FAIL (exit 1)", self.out)
        self.assertIn("AttributeError: no text_tokenizer", self.out)
        self.assertEqual(self.run_orch(stages=self.st), 13, self.out)
        self.assertIn("Nothing was rented", self.out)
        self.assertEqual(self.calls(), [])
        self.assertFalse((self.tmp / "paid").exists())
        self.assertEqual(self.n_runs(), 2)  # a failure is re-run, never cached

    def test_missing_venv_blocks_and_prints_the_build_command(self):
        self.scenario([{}])
        self.assertEqual(self.run_orch(stages=self.st, plan=True), 13, self.out)
        self.assertIn("real-path check : NOT RUN (venv missing", self.out)
        self.assertIn("uv venv -p 3.11", self.out)
        self.assertFalse((self.home / ".cache" / "vi" / "home-cpu-venv").exists())  # never builds it

    def test_passing_check_lets_plan_through(self):
        install_fake_realpath(self.home)
        self.scenario([dict(spent=0.1, hours=1.0)])
        self.assertEqual(self.run_orch(stages=self.st, plan=True), 0, self.out)
        self.assertIn("real-path check : OK", self.out)

    def test_low_ram_refuses_instead_of_running(self):
        install_fake_realpath(self.home)
        ok, lines = self.run_check(mem_mb=1000)
        self.assertFalse(ok)
        self.assertIn("NOT RUN (only 1000 MB RAM", lines[0])
        self.assertEqual(self.n_runs(), 0)

    def test_forward_flag_needs_12gb_else_stops_at_the_batch_and_says_so(self):
        install_fake_realpath(self.home)
        ok, lines = self.run_check(forward=True, mem_mb=8000)
        self.assertTrue(ok, lines)
        self.assertIn("forward step skipped: 8000 MB RAM available", "\n".join(lines))
        self.assertNotIn("--forward", self.calls_log.read_text())
        ok, lines = self.run_check(forward=True, mem_mb=20000)  # batch-only cache does not satisfy it
        self.assertTrue(ok, lines)
        self.assertEqual(self.n_runs(), 2)
        self.assertIn("--forward", self.calls_log.read_text().splitlines()[1])
        ok, lines = self.run_check(forward=True, mem_mb=20000)
        self.assertIn("cached, with forward", lines[0])
        self.assertEqual(self.n_runs(), 2)

    def test_runner_is_injectable_and_gets_nice_19(self):
        install_fake_realpath(self.home)
        seen = []
        ok, _ = self.run_check(mem_mb=20000, runner=lambda c, t, e: seen.append((c, t)) or (0, "ok"))
        self.assertTrue(ok)
        self.assertEqual(seen[0][0][:3], ["nice", "-n", "19"])
        self.assertIn("-I", seen[0][0])

    def test_no_data_folder_is_not_checked(self):
        st = HomeChecks.stages_file(self, 1, 0)
        ok, lines = self.pr.realpath_check(st, str(self.tmp), home=self.home)
        self.assertTrue(ok)
        self.assertIn("not checked", lines[0])

    def test_home_cpu_pins_match_the_worker_pins_except_torch(self):
        def pins(name):
            return {ln.strip() for ln in (JOBS / name).read_text().splitlines()
                    if ln.strip() and not ln.startswith("#") and not ln.startswith("--")}
        worker = pins("requirements-omnivoice-worker.txt")
        home = pins("requirements-omnivoice-home-cpu.txt")
        swap = {"torch==2.8.0": "torch==2.8.0+cpu", "torchaudio==2.8.0": "torchaudio==2.8.0+cpu"}
        self.assertEqual({swap.get(p, p) for p in worker}, home)
        self.assertIn("--extra-index-url https://download.pytorch.org/whl/cpu",
                      (JOBS / "requirements-omnivoice-home-cpu.txt").read_text())

    @staticmethod
    def _pin_map(text):
        """name -> version for every requirement line; index/option lines and comments dropped,
        a local +cpu tag stripped. A line that is not name==version fails the caller's assert."""
        out, loose = {}, []
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or ln.startswith("-"):
                continue
            name, sep, ver = ln.partition("==")
            if not sep or any(c in ver for c in "<>=!~,"):
                loose.append(ln)
                continue
            out[name.strip().lower()] = ver.split("+")[0].strip()
        return out, loose

    def test_omnivoice_pins_match(self):
        worker_text = (JOBS / "requirements-omnivoice-worker.txt").read_text()
        home_text = (JOBS / "requirements-omnivoice-home-cpu.txt").read_text()
        worker, w_loose = self._pin_map(worker_text)
        home, h_loose = self._pin_map(home_text)
        self.assertEqual(w_loose, [])
        self.assertEqual(h_loose, [])
        self.assertGreater(len(worker), 50)
        self.assertEqual(worker, home)
        # A mutated copy (one pin differs, or one is loosened) must fail the same comparison.
        mutated = home_text.replace("torch==2.8.0+cpu", "torch==2.9.0+cpu")
        self.assertNotEqual(self._pin_map(mutated)[0], worker)
        loosened = worker_text.replace("torch==2.8.0", "torch>=2.8.0")
        self.assertNotEqual(self._pin_map(loosened)[1], [])

    def test_tokenise_flags_match_the_worker_script(self):
        text = (JOBS / "voicetune-fresh.sh").read_text()
        src = (JOBS / "omnivoice-ft" / "home_realpath_check.py").read_text()
        self.assertIn('TOKENISE_FLAGS = ["--samples_per_shard", "1000", "--skip_errors"]', src)
        self.assertIn("--samples_per_shard 1000 --skip_errors", text)

    def test_exit_code_13_is_documented(self):
        self.assertEqual(self.pr.EXIT["realpath"], 13)
        self.assertIn("13 ", (JOBS.parent / "docs" / "cloud-worker.md").read_text().split("Exit codes:")[1][:1500])


class RunLogging(Base):
    """run.log, the pre-destroy diagnostics pull, and stage-tail.log (round 8 item 5)."""

    def paid(self, name):
        return self.tmp / "paid" / "t1" / name

    def gate_fail_then_pass(self, **extra):
        self.scenario([dict(preflight="cuda", spent=0.1, hours=0.1), dict(spent=0.2, hours=0.2)],
                      **extra)

    def test_run_log_has_a_header_per_attempt_and_the_full_output(self):
        self.gate_fail_then_pass()
        self.assertEqual(self.run_orch(), 0, self.out)
        log = self.paid("run.log").read_text()
        self.assertIn("===== attempt 1 of 3", log)
        self.assertIn("===== attempt 2 of 3", log)
        self.assertIn("worker-preflight: CUDA version is 12.4", log)  # the worker's own output
        self.assertIn("total actual spend", log)
        self.assertIn("===== train stage", log)

    def test_failed_attempt_leaves_diagnostics_and_a_good_one_does_not(self):
        self.gate_fail_then_pass()
        self.assertEqual(self.run_orch(), 0, self.out)
        diag = self.paid("diag-attempt-1-worker-gate.txt")
        self.assertIn("CUDA Version: N/A", diag.read_text())
        self.assertIn("torch==2.8.0", diag.read_text())
        self.assertEqual(len(list(self.paid("x").parent.glob("diag-attempt-*"))), 1)

    def test_diagnostics_come_before_destroy(self):
        self.gate_fail_then_pass()
        self.run_orch()
        names = [c[0] for c in self.calls() if c[0] in ("ssh", "down")]
        self.assertEqual(names[:2], ["ssh", "down"])

    def test_hanging_fetch_never_blocks_destroy(self):
        self.gate_fail_then_pass(diag="hang")
        self.env["PAID_RUN_DIAG_SECONDS"] = "1"
        t0 = time.time()
        self.assertEqual(self.run_orch(), 0, self.out)
        self.assertLess(time.time() - t0, 30)
        self.assertGreaterEqual(len(self.downs()), 2)  # both attempts were destroyed
        text = self.paid("diag-attempt-1-worker-gate.txt").read_text()
        self.assertIn("cut off after 1 s", text)
        self.assertIn("partial nvidia-smi line", text)
        pid = int((self.sd / "diag.pid").read_text())
        with self.assertRaises(OSError):
            os.kill(pid, 0)  # the hung fetch process was killed, not left running

    def test_failing_fetch_still_destroys(self):
        self.gate_fail_then_pass(diag="fail")
        self.assertEqual(self.run_orch(), 0, self.out)
        self.assertIn("exited 255", self.paid("diag-attempt-1-worker-gate.txt").read_text())
        self.assertGreaterEqual(len(self.downs()), 2)

    def test_diagnostics_size_is_bounded(self):
        self.gate_fail_then_pass(diag="big")
        self.assertEqual(self.run_orch(), 0, self.out)
        size = self.paid("diag-attempt-1-worker-gate.txt").stat().st_size
        self.assertLess(size, load_paid_run().DIAG_MAX_BYTES + 200)
        self.assertIn("truncated", self.paid("diag-attempt-1-worker-gate.txt").read_text())

    def test_stage_tail_is_written_for_the_preflight_and_status_reads_it(self):
        self.scenario([dict(preflight="cuda", spent=0.1)], offer_list=offers((1, 0.20, "US", 800)))
        self.assertEqual(self.run_orch(max_attempts=1), 5, self.out)
        self.assertIn("worker-preflight: CUDA version", self.paid("stage-tail.log").read_text())
        rc = subprocess.run([sys.executable, "-I", str(SCRIPT), "status", "--run-id", "t1", "--json"],
                            env=self.env, text=True, capture_output=True, timeout=60)
        self.assertTrue(any("worker-preflight: CUDA version" in ln
                            for ln in json.loads(rc.stdout)["stage_log_tail"]), rc.stdout)


if __name__ == "__main__":
    unittest.main()
