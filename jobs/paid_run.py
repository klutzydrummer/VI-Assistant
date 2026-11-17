#!/usr/bin/env python3
"""One unattended command for the paid OmniVoice fine-tune flow (standard library only).

    python3 -I jobs/paid_run.py --run-id ID --cap USD --max-hours H [options]

It drives the existing `vi` binary ($VI_BIN, default `vi`) so a stub can stand in for tests:
rent the cheapest qualifying offer, run the preflight stage (tunnel, receiver, token push, the
worker gates), then the train stage, then release the worker and print what it cost. If ONLY the
push-speed gate fails (stderr says `worker push (MB/s)`), the machine is released, that offer is
excluded, and the next cheapest is tried, up to --max-attempts. Any other failure releases the
worker and stops.

--cap and --max-hours cover ALL attempts together. A state file records every attempt, so a
killed run can be started again with the same --run-id: a worker still up is adopted (if the
train stage had started) or destroyed first, and a second rental is never made on top of it.

Exit codes:
  0 done        2 bad arguments         3 `vi cloud up` failed (boot failure included)
  4 a preflight failure that is not the push gate
  5 the push gate failed on every attempt allowed (--max-attempts)
  6 the train stage failed             7 train finished but home checkpoints/listening sets
                                         did not verify
  8 interrupted (SIGINT/SIGTERM); worker released
  9 THE WORKER COULD NOT BE RELEASED: check `vi cloud status` now
 10 no offer or budget left to try (cap, hours, or offers exhausted)
  1 anything unexpected (worker release is still attempted)

Heavy work happens inside `vi dataset run` (stages `omnivoice_preflight`, `omnivoice_train`).
No Vast credential file is read here; only the `vi` CLI is called.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO / "jobs" / "paid-run" / "omnivoice-run-cloud.toml"
DEFAULT_STAGES = "characters/tali/omnivoice-ft.toml"
PUSH_GATE_TEXT = "worker push (MB/s)"
EU_COUNTRIES = ("US AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI "
                "ES SE GB CH NO").split()
# Home keeps the two newest checkpoints plus the exempt steps (jobs/checkpoints.py
# EXEMPT_STEPS = 1000, 2500) when a 5000-step run ends.
EXPECTED_STEPS = (1000, 2500, 4500, 5000)
LISTENING_STEPS = (1000, 2500, 5000)
EXPECT_EPOCHS = 4

OFFER_RE = re.compile(
    r"^\s*(\d+)\s+(.+?)\s+(\d+) GB\s+\$([\d.]+)/hr\s+reliability ([\d.]+)\s+"
    r"net down\s+(\d+) / up\s+(\d+) Mbps\s+([\d.]+) cores\s+(\d+) GB ram\s+"
    r"disk (\d+) MB/s\s*(.*)$")

EXIT = dict(ok=0, args=2, up=3, preflight=4, push=5, train=6, verify=7, interrupted=8,
            release=9, budget=10, other=1)


class Stop(Exception):
    """Finish the run with this exit code and message."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


class Interrupted(Exception):
    pass


def say(msg):
    print(f"paid-run: {msg}", flush=True)


def state_root():
    for var in ("VI_PAID_RUN_STATE_DIR",):
        if os.environ.get(var):
            return Path(os.environ[var])
    if os.environ.get("VI_STATE_DIR"):
        base = Path(os.environ["VI_STATE_DIR"])
    elif os.environ.get("XDG_STATE_HOME"):
        base = Path(os.environ["XDG_STATE_HOME"]) / "vi"
    else:
        base = Path.home() / ".local" / "state" / "vi"
    return base / "paid-run"


def vi_state_dir():
    if os.environ.get("VI_STATE_DIR"):
        return Path(os.environ["VI_STATE_DIR"])
    if os.environ.get("XDG_STATE_HOME"):
        return Path(os.environ["XDG_STATE_HOME"]) / "vi"
    return Path.home() / ".local" / "state" / "vi"


def clock():
    """Wall clock. Tests add a fake offset (seconds) kept in $PAID_RUN_CLOCK_FILE."""
    t = time.time()
    f = os.environ.get("PAID_RUN_CLOCK_FILE")
    if f:
        try:
            t += float(Path(f).read_text().strip() or 0)
        except (OSError, ValueError):
            pass
    return t


def rfc3339(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def parse_offers(text):
    """[{id, gpu, dph, up, cores, ram, loc, sig}] in the order `vi cloud offers` printed them."""
    out = []
    for line in text.splitlines():
        m = OFFER_RE.match(line)
        if not m:
            continue
        loc = m.group(11).strip()
        out.append(dict(
            id=int(m.group(1)), gpu=m.group(2).strip(), dph=float(m.group(4)),
            up=int(m.group(7)), cores=m.group(8), ram=m.group(9), loc=loc,
            # `vi cloud offers` does not print machine_id, so a relisted machine is recognised by
            # what it is and where it is.
            sig=f"{m.group(2).strip()}|{loc}|{m.group(8)}|{m.group(9)}|{m.group(7)}"))
    return out


def set_toml_values(text, max_hours=None, countries=None):
    """A copy of a cloud config with top-level max_hours and/or search.countries replaced."""
    lines = text.splitlines()
    if max_hours is not None:
        new = f"max_hours = {max_hours:.4f}"
        for i, ln in enumerate(lines):
            if ln.startswith("[") :
                lines.insert(i, new)
                break
            if re.match(r"\s*max_hours\s*=", ln):
                lines[i] = new
                break
        else:
            lines.append(new)
    if countries is not None:
        new = "countries = [" + ", ".join(f'"{c}"' for c in countries) + "]"
        in_search, at = False, None
        for i, ln in enumerate(lines):
            s = ln.strip()
            if s.startswith("["):
                if in_search:
                    break
                in_search = s == "[search]"
                if in_search:
                    at = i + 1
            elif in_search and re.match(r"countries\s*=", s):
                lines[i] = new
                at = None
                break
        else:
            pass
        if at is not None:
            lines.insert(at, new)
        elif not any(ln.strip() == "[search]" for ln in lines):
            lines += ["[search]", new]
    return "\n".join(lines) + "\n"


class PaidRun:
    def __init__(self, args):
        self.a = args
        self.vi = os.environ.get("VI_BIN", "vi")
        self.dir = state_root() / args.run_id
        self.path = state_root() / f"{args.run_id}.json"
        self.child = None
        self.worker_maybe_up = False
        self.in_teardown = False
        self.config_path = None  # the per-attempt config currently in force
        self.s = None

    # ----- state ---------------------------------------------------------------------------
    def load_or_create(self):
        if self.path.exists():
            self.s = json.loads(self.path.read_text())
            for key, want in (("cap", self.a.cap), ("max_hours", self.a.max_hours)):
                if abs(self.s[key] - want) > 1e-9:
                    raise Stop(EXIT["args"], f"run {self.a.run_id} was started with {key} "
                               f"{self.s[key]}, not {want}; use the same limits or a new --run-id")
            say(f"resuming run {self.a.run_id} from {self.path}")
            return True
        self.s = dict(run_id=self.a.run_id, cap=self.a.cap, max_hours=self.a.max_hours,
                      max_attempts=self.a.max_attempts, created=clock(), status="running",
                      exit_code=None, message="", attempts=[], failed_offers=[], failed_sigs=[],
                      spent_total=0.0, hours_total=0.0, credit_before_run=None,
                      credit_after_run=None)
        return False

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.s, indent=1) + "\n")
        os.replace(tmp, self.path)

    # ----- running vi ----------------------------------------------------------------------
    def env(self):
        env = dict(os.environ)
        if self.config_path:
            env["VI_CLOUD_CONFIG"] = str(self.config_path)
        return env

    def run(self, argv, stream=True):
        """Run a command, returning (returncode, combined stdout+stderr text)."""
        p = subprocess.Popen(argv, cwd=self.a.repo, env=self.env(), text=True, bufsize=1,
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        self.child = p
        buf = []
        try:
            for line in p.stdout:
                buf.append(line)
                if stream:
                    print(line, end="", flush=True)
            p.wait()
        except BaseException:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            raise
        finally:
            self.child = None
        return p.returncode, "".join(buf)

    def vi_cmd(self, *args, stream=False):
        return self.run([self.vi, *args], stream=stream)

    def credit(self):
        rc, out = self.vi_cmd("cloud", "status")
        m = re.search(r"^credit \$([\d.]+)", out, re.M)
        return float(m.group(1)) if rc == 0 and m else None

    def worker_present(self):
        """(present, text, tracked): is any vi-worker instance on the account?"""
        rc, out = self.vi_cmd("cloud", "status")
        if rc != 0:
            raise Stop(EXIT["release"], "`vi cloud status` failed, so a billing worker cannot be "
                       f"ruled out. Check it by hand.\n{out.strip()}")
        tracked = bool(re.search(r"^worker \d+ (?!is gone)", out, re.M))
        orphan = bool(re.search(r"^untracked .* instance \d+", out, re.M))
        return tracked or orphan, out, tracked

    # ----- the plan ------------------------------------------------------------------------
    def plan_lines(self):
        a = self.a
        countries = a.countries or "(from the config)"
        return [
            f"run id          : {a.run_id}",
            f"cap (all tries) : ${a.cap:.2f}",
            f"max hours (all) : {a.max_hours} h",
            f"max attempts    : {a.max_attempts} (a retry only after a push-speed gate failure)",
            f"vi binary       : {self.vi}",
            f"run config      : {a.config}",
            f"countries       : {countries}",
            f"stage file      : {a.stages}",
            f"state file      : {self.path}",
            "flow            : offers -> cheapest not-failed offer -> up -> preflight stage "
            "(tunnel, receiver, token push, gates) -> train stage -> verify -> down",
            "retry rule      : only `worker push (MB/s)` failures retry; every other failure "
            "releases the worker and stops",
            f"keeps at home   : checkpoints {', '.join(map(str, EXPECTED_STEPS))}; listening sets "
            f"{', '.join(map(str, LISTENING_STEPS))} in {a.platter}",
        ]

    # ----- config --------------------------------------------------------------------------
    def base_config_text(self):
        text = Path(self.a.config).read_text()
        if self.a.countries:
            cs = EU_COUNTRIES if self.a.countries.strip().lower() == "eu" else [
                c.strip().upper() for c in self.a.countries.split(",") if c.strip()]
            text = set_toml_values(text, countries=cs)
        return text

    def write_attempt_config(self, n, hours):
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / f"attempt-{n}.toml"
        p.write_text(set_toml_values(self.base_config_text(), max_hours=hours))
        self.config_path = p
        return p

    # ----- bookkeeping ---------------------------------------------------------------------
    def remaining(self):
        return (self.s["cap"] - self.s["spent_total"], self.s["max_hours"] - self.s["hours_total"])

    def open_attempt(self):
        for at in self.s["attempts"]:
            if at.get("ended") is None:
                return at
        return None

    def close_attempt(self, at, outcome, spent, spent_estimated=False):
        at["ended"] = clock()
        at["hours"] = max(0.0, (at["ended"] - at["started"]) / 3600)
        at["outcome"] = outcome
        at["spent"] = spent
        at["spent_estimated"] = spent_estimated
        self.s["spent_total"] = round(sum(x.get("spent") or 0 for x in self.s["attempts"]), 4)
        self.s["hours_total"] = round(sum(x.get("hours") or 0 for x in self.s["attempts"]), 4)
        self.save()

    def release(self, at, outcome):
        """`vi cloud down`, then close the attempt with its spend. Raises Stop(9) if the worker
        cannot be released."""
        self.in_teardown = True
        say("releasing the worker (vi cloud down)")
        rc, out = self.vi_cmd("cloud", "down", stream=True)
        if rc != 0:
            say("plain `down` refused; trying `down --force` so nothing keeps billing")
            rc, out = self.vi_cmd("cloud", "down", "--force", stream=True)
        if rc != 0:
            self.worker_maybe_up = True
            if at is not None:
                at["outcome"] = outcome + "+release-failed"
                self.save()
            raise Stop(EXIT["release"], "COULD NOT RELEASE THE WORKER. It may still be billing. "
                       "Run `vi cloud status` and `vi cloud down --force` now.")
        self.worker_maybe_up = False
        spent, est = None, False
        m = re.search(r"\(\$([\d.]+) spent\)", out)
        if m:
            spent = float(m.group(1))
        elif at is not None and at.get("credit_start") is not None:
            now = self.credit()
            if now is not None:
                spent = max(0.0, at["credit_start"] - now)
        if spent is None and at is not None:
            spent, est = round((at.get("dph") or 0) * (clock() - at["started"]) / 3600, 4), True
        if at is not None:
            self.close_attempt(at, outcome, spent or 0.0, est)
        self.in_teardown = False
        return spent

    # ----- resume --------------------------------------------------------------------------
    def settle_previous(self):
        """Handle an attempt left open by a killed run. Returns the attempt to continue (adopted
        worker in the train phase) or None."""
        at = self.open_attempt()
        present, out, tracked = self.worker_present()
        if at is None:
            if present:
                say("a vi-worker instance exists that this run did not record; destroying it "
                    "before renting anything")
                self.worker_maybe_up = True
                self.release(None, "orphan")
            return None
        say(f"attempt {at['n']} was left open (phase {at['phase']})")
        if present and at["phase"] == "train" and tracked:
            say("the worker is still up and the train stage had started: adopting it, the train "
                "stage reattaches")
            self.worker_maybe_up = True
            return at
        if present:
            self.worker_maybe_up = True
            self.release(at, "interrupted")
        else:
            now = self.credit()
            spent = None
            if now is not None and at.get("credit_start") is not None:
                spent = max(0.0, at["credit_start"] - now)
            est = spent is None
            if est:
                spent = round((at.get("dph") or 0) * (clock() - at["started"]) / 3600, 4)
            self.close_attempt(at, "interrupted", spent, est)
        if at["phase"] == "train":
            raise Stop(EXIT["train"], "the run was interrupted during the train stage and the "
                       "worker is gone; checkpoints that came home are under "
                       f"{self.a.checkpoint_root}. Decide by hand whether to start a new run.")
        return None

    # ----- the flow ------------------------------------------------------------------------
    def choose_offer(self, hours_left, cap_left):
        rc, out = self.vi_cmd("cloud", "offers")
        if rc != 0:
            raise Stop(EXIT["up"], f"`vi cloud offers` failed:\n{out.strip()}")
        offers = sorted(parse_offers(out), key=lambda o: o["dph"])
        if not offers:
            raise Stop(EXIT["budget"], "no offers match the config right now")
        fresh = [o for o in offers if o["id"] not in self.s["failed_offers"]
                 and o["sig"] not in self.s["failed_sigs"]]
        if not fresh:
            raise Stop(EXIT["budget"], "every listed offer has already failed this run "
                       f"({len(offers)} listed)")
        affordable = [o for o in fresh if o["dph"] * hours_left <= cap_left + 1e-9]
        if not affordable:
            cheapest = fresh[0]
            raise Stop(EXIT["budget"], f"no offer fits what is left: ${cap_left:.2f} for "
                       f"{hours_left:.2f} h (cheapest is ${cheapest['dph']:.3f}/hr, worst case "
                       f"${cheapest['dph'] * hours_left:.2f})")
        return affordable[0]

    def execute(self):
        a = self.a
        adopted = self.settle_previous()
        if self.s["credit_before_run"] is None:
            self.s["credit_before_run"] = self.credit()
            self.save()
        at = adopted
        while True:
            if at is None:
                n = len(self.s["attempts"]) + 1
                cap_left, hours_left = self.remaining()
                if n > a.max_attempts:
                    raise Stop(EXIT["push"], f"the push-speed gate failed on all {a.max_attempts}"
                               " attempts allowed")
                if hours_left < a.min_hours:
                    raise Stop(EXIT["budget"], f"only {hours_left:.2f} h of the {a.max_hours} h "
                               f"left (under --min-hours {a.min_hours}); not renting again")
                if cap_left <= 0:
                    raise Stop(EXIT["budget"], f"the ${a.cap:.2f} cap is used up")
                say(f"attempt {n}: ${cap_left:.2f} and {hours_left:.2f} h left")
                self.write_attempt_config(n, hours_left)
                offer = self.choose_offer(hours_left, cap_left)
                say(f"cheapest offer not tried: {offer['id']} {offer['gpu']} {offer['loc']} "
                    f"${offer['dph']:.3f}/hr")
                at = dict(n=n, offer=offer["id"], sig=offer["sig"], dph=offer["dph"],
                          phase="up", started=clock(), ended=None, cap=round(cap_left, 4),
                          hours=None, credit_start=self.credit())
                self.s["attempts"].append(at)
                self.save()
                self.worker_maybe_up = True
                until = rfc3339(clock() + 3600)
                rc, out = self.vi_cmd("cloud", "up", "--offer", str(offer["id"]), "--cap",
                                      f"{cap_left:.2f}", "--until", until, "--yes", stream=True)
                if rc != 0:
                    self.release(at, "up-failed")
                    raise Stop(EXIT["up"], f"`vi cloud up` failed (exit {rc}); the machine was "
                               "released and nothing else was rented")
                at["phase"] = "preflight"
                self.save()
                rc, out = self.run([self.vi, "dataset", "run", a.stages, "omnivoice_preflight"])
                if rc != 0:
                    if PUSH_GATE_TEXT in out:
                        say(f"offer {offer['id']} failed the push-speed gate")
                        self.s["failed_offers"].append(offer["id"])
                        self.s["failed_sigs"].append(offer["sig"])
                        spent = self.release(at, "push-gate")
                        say(f"attempt {n} cost ${spent:.2f}; total ${self.s['spent_total']:.2f} "
                            f"of ${a.cap:.2f}, {self.s['hours_total']:.2f} of {a.max_hours} h")
                        at = None
                        continue
                    self.release(at, "preflight-failed")
                    raise Stop(EXIT["preflight"], f"the preflight failed (exit {rc}) on a gate "
                               "other than the push speed; the worker was released, no retry")
                at["phase"] = "train"
                self.save()
            # -- train (new or adopted) --
            say("preflight passed; starting the train stage")
            rc, out = self.run([self.vi, "dataset", "run", a.stages, "omnivoice_train"])
            if rc != 0:
                self.release(at, "train-failed")
                raise Stop(EXIT["train"], f"the train stage failed (exit {rc}); the worker was "
                           f"released. Any checkpoints that came home are under "
                           f"{a.checkpoint_root}")
            at["phase"] = "verify"
            self.save()
            problems = self.verify_final()
            spent = self.release(at, "verify-failed" if problems else "done")
            if problems:
                raise Stop(EXIT["verify"], "the train stage finished but the results at home do "
                           "not check out:\n  " + "\n  ".join(problems))
            return spent

    # ----- home results --------------------------------------------------------------------
    def run_dir(self):
        root = Path(self.a.checkpoint_root) / "omnivoice_train"
        runs = [d for d in root.glob("*") if d.is_dir()] if root.is_dir() else []
        return max(runs, key=lambda d: d.stat().st_mtime) if runs else None

    def verify_final(self):
        """What home should hold after a good train stage; a list of problems (empty = fine)."""
        problems = []
        d = self.run_dir()
        if d is None:
            return [f"no run folder under {self.a.checkpoint_root}/omnivoice_train"]
        steps = sorted(int(m.group(1)) for p in d.iterdir()
                       if (m := re.fullmatch(r"checkpoint-(\d+)", p.name)) and p.is_dir())
        if tuple(steps) != EXPECTED_STEPS:
            problems.append(f"{d}: checkpoints present {steps}, expected {list(EXPECTED_STEPS)}")
        if not (d / "VERIFIED").is_file():
            problems.append(f"{d}: no VERIFIED file (the stage's final verify did not finish)")
        markers = vi_state_dir() / "unverified"
        if markers.is_dir():
            for m in markers.iterdir():
                if m.is_file() and str(d) in m.read_text():
                    problems.append(f"unverified marker {m} still names {d}")
        for step in LISTENING_STEPS:
            s = Path(self.a.platter) / f"step-{step}"
            gone = [need for need in ("listening.jsonl", "README.txt") if not (s / need).is_file()]
            problems += [f"listening set step {step}: {s / need} is missing" for need in gone]
            if not gone:
                problems += self.write_clip_list(step, s)
        if not problems:
            say(f"verified: checkpoints {steps} in {d}; listening sets in {self.a.platter}")
        return problems

    def write_clip_list(self, step, s):
        """Make step-N listable in the review hub: write `clips.toml` beside its wav files.

        The hub (`vi review corpus/tali`) already finds any `clips.toml` under a stage's current
        output (crates/review/src/hub.rs `walk`, depth 4) and plays its clips unblinded, with a
        label and the line text, saving his answers beside it. A single-model set has nothing to
        blind against, so `blind` stays off. Only local files are written; returns problems."""
        wavs = []
        try:
            rows = [json.loads(ln) for ln in (s / "listening.jsonl").read_text().splitlines() if ln.strip()]
        except ValueError as e:
            return [f"listening set step {step}: listening.jsonl is not valid JSON lines ({e})"]
        for row in rows:
            label = row.get("label") if isinstance(row, dict) else None
            # The row's `file` is a worker path; the wav came home beside the set under wav/.
            wav = s / "wav" / f"{label}.wav" if isinstance(label, str) and label else None
            if wav is None or Path(label).name != label or not wav.is_file():
                return [f"listening set step {step}: no audio {wav or row} for a row in listening.jsonl"]
            wavs.append((label, str(row.get("text") or "")))
        if not wavs:
            return [f"listening set step {step}: listening.jsonl has no clips"]
        q = json.dumps  # a JSON string is a valid TOML basic string
        toml = (f"title = {q(f'OmniVoice fine-tune, step {step} (run {self.a.run_id})')}\n"
                f"question = {q('Does this sound like her? Listen for the voice; nothing here is scored.')}\n"
                f"answers = [\"her\", \"partly\", \"not her\"]\n"
                f"asked_by = \"paid_run.py\"\n")
        for label, text in wavs:
            toml += f"\n[[clip]]\npath = {q(f'wav/{label}.wav')}\nlabel = {q(label)}\ntext = {q(text)}\n"
        (s / "clips.toml").write_text(toml)
        return []

    # ----- main ----------------------------------------------------------------------------
    def signal_handler(self, signum, _frame):
        if self.in_teardown:
            say(f"signal {signum} ignored while releasing the worker")
            return
        say(f"signal {signum}: stopping and releasing the worker")
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()
        raise Interrupted(signum)

    def report(self):
        s = self.s
        say("attempts:")
        for at in s["attempts"]:
            say(f"  #{at['n']} offer {at['offer']} ${at['dph']:.3f}/hr: {at.get('outcome')}, "
                f"${(at.get('spent') or 0):.2f}{' (estimated)' if at.get('spent_estimated') else ''}"
                f", {(at.get('hours') or 0):.2f} h")
        after = self.credit()
        s["credit_after_run"] = after
        before = s["credit_before_run"]
        line = f"total actual spend: ${s['spent_total']:.2f} of ${s['cap']:.2f}; {s['hours_total']:.2f} h"
        if before is not None and after is not None:
            line += f"; account credit ${before:.2f} before, ${after:.2f} now (${before - after:.2f})"
        say(line + " (credit can lag the last billing slice)")
        d = self.run_dir()
        say(f"home results: checkpoints {d}; listening sets {self.a.platter}")

    def main(self):
        for ln in self.plan_lines():
            say(ln)
        if self.a.plan:
            say("--plan: nothing was called; exiting")
            return 0
        try:
            resumed = self.load_or_create()
        except Stop as e:
            print(f"paid-run: STOPPED (exit {e.code}): {e.message}", file=sys.stderr, flush=True)
            return e.code
        if resumed and self.s["status"] not in ("running", "interrupted"):
            say(f"run already finished: {self.s['status']} (exit {self.s['exit_code']}): "
                f"{self.s['message']}")
            return self.s["exit_code"]
        self.s["status"] = "running"
        self.save()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self.signal_handler)
        code, message = EXIT["ok"], "done"
        try:
            try:
                self.execute()
            except Stop as e:
                code, message = e.code, e.message
            except Interrupted:
                code, message = EXIT["interrupted"], "interrupted by a signal"
            except Exception as e:  # noqa: BLE001: always fall through to the release below
                code, message = EXIT["other"], f"unexpected error: {type(e).__name__}: {e}"
        finally:
            if self.worker_maybe_up:
                try:
                    self.in_teardown = True
                    self.release(self.open_attempt(), "aborted")
                except Stop as e:
                    code, message = e.code, e.message
                except BaseException as e:  # noqa: BLE001
                    code, message = EXIT["release"], (
                        f"release failed unexpectedly ({e}); run `vi cloud status` now")
        self.s["status"] = {0: "done", EXIT["interrupted"]: "interrupted"}.get(code, "failed")
        self.s["exit_code"], self.s["message"] = code, message
        self.save()
        try:
            self.report()
        except Exception as e:  # noqa: BLE001
            say(f"(report incomplete: {e})")
        (say if code == 0 else lambda m: print(f"paid-run: STOPPED (exit {code}): {m}",
                                               file=sys.stderr, flush=True))(message)
        return code


def parse(argv):
    p = argparse.ArgumentParser(prog="paid_run.py", description=__doc__.split("\n\n")[0])
    p.add_argument("--run-id", required=True, help="names the state file; reuse it to resume")
    p.add_argument("--cap", type=float, required=True, help="most USD all attempts together")
    p.add_argument("--max-hours", type=float, required=True, help="most hours, all attempts")
    p.add_argument("--max-attempts", type=int, default=3)
    p.add_argument("--min-hours", type=float, default=0.5,
                   help="do not rent again with fewer hours than this left")
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help="the run's vi cloud config (default jobs/paid-run/omnivoice-run-cloud.toml)")
    p.add_argument("--countries", help='e.g. "US,DE" or "eu"; written into a copy of the config')
    p.add_argument("--stages", default=DEFAULT_STAGES)
    p.add_argument("--repo", default=str(REPO), help="working directory for vi")
    p.add_argument("--checkpoint-root", default=os.environ.get(
        "VI_CHECKPOINT_ROOT", str(Path.home() / ".cache" / "vi" / "checkpoints")))
    p.add_argument("--platter", default=os.environ.get("VI_PLATTER_DIR", str(
        REPO / "corpus" / "tali" / "stages" / "omnivoice_train" / "current" / "out" / "platter")))
    p.add_argument("--plan", action="store_true", help="print the plan and exit; no vi calls")
    a = p.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", a.run_id):
        p.error("--run-id may only hold letters, digits, dot, dash and underscore")
    if a.cap <= 0 or a.max_hours <= 0 or a.max_attempts < 1:
        p.error("--cap, --max-hours and --max-attempts must be positive")
    if not Path(a.config).is_file():
        p.error(f"--config {a.config} is not a file")
    return a


def main(argv=None):
    try:
        args = parse(sys.argv[1:] if argv is None else argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else EXIT["args"]
    return PaidRun(args).main()


if __name__ == "__main__":
    sys.exit(main())
