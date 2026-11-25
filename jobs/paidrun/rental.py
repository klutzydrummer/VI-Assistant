"""Renting and training: offer choice, failed machines, VRAM floors, in-place train recovery."""

import json
import os
import re
from pathlib import Path

from paidrun.common import EXIT, Stop, StopRequested, clock, nap, rfc3339, say
from paidrun.limits import (
    FATAL_QUIET_SECONDS, INPLACE_MAX_RESTARTS, INPLACE_MAX_SECONDS, PREFLIGHT_TIMEOUT_SECONDS,
    REATTACH_MAX, REATTACH_WAIT_SECONDS, UP_EXTRA_SECONDS,
)
from paidrun.signals import BOOT_FAIL_RE, failure_rc_text, gate_kind, parse_offers, retryable_gate
from paidrun.config import config_boot_minutes, search_value


INPLACE_MIN_DISK_GB = 5.0         # a worker with less free disk than this is treated as ENOSPC
VRAM_FLOOR_STEP_GB = 4.0          # F-OOM-1: how much a memory failure raises the next offer's VRAM floor
# Worker check for an in-place train restart, run through `vi cloud ssh`.
INPLACE_SCRIPT = r"""
exec 2>&1
# INPLACE_CHECK
if timeout 20 nvidia-smi -L >/dev/null 2>&1; then echo "inplace: gpu=ok"; else echo "inplace: gpu=missing"; fi
echo "inplace: disk_free_gb=$(df -BG --output=avail "$HOME" 2>/dev/null | tail -n1 | tr -dc 0-9)"
n=$(pgrep -fc '[w]orker-job.sh|[t]rain_ft.py|[v]oicetune-fresh.sh' 2>/dev/null); echo "inplace: procs=${n:-0}"
ab=$(find "$HOME" /tmp /workspace -maxdepth 8 -name abort.json -path '*voicetune-fresh*' 2>/dev/null | head -n1)
echo "inplace: abort=${ab:-none}"
if find "$HOME" /tmp /workspace -maxdepth 6 -name job.log 2>/dev/null | head -n3 | xargs -r grep -l 'No space left on device' >/dev/null 2>&1; then
  echo "inplace: enospc=1"; else echo "inplace: enospc=0"; fi
ck=$(find "$HOME" /tmp /workspace -maxdepth 8 -path '*voicetune-fresh/run/checkpoint-*' -name SHA256SUMS 2>/dev/null \
  | sed -E 's#.*/checkpoint-([0-9]+)/SHA256SUMS#\1#' | sort -n | tail -n1)
echo "inplace: ckpt=${ck:-none}"
"""


class RentalMixin:
    # ----- the flow ------------------------------------------------------------------------
    def choose_offer(self, hours_left, cap_left):
        excl = [x for m in self.s["failed_machines"] for x in ("--exclude-machine", str(m))]
        rc, out = self.vi_cmd("cloud", "offers", *excl)
        if rc != 0:
            raise Stop(EXIT["up"], f"`vi cloud offers` failed:\n{out.strip()}")
        offers = sorted(parse_offers(out), key=lambda o: o["dph"])
        if not offers:
            raise Stop(EXIT["budget"], "no offers match the config right now")

        def failed(o):
            if o["id"] in self.s["failed_offers"]:
                return True
            if o["machine"] is not None and o["machine"] in self.s["failed_machines"]:
                return True
            if o["host"] is not None and o["host"] in self.s["failed_hosts"]:
                return True
            # without a machine id in the listing, fall back to what the machine looks like
            return o["machine"] is None and o["sig"] in self.s["failed_sigs"]

        fresh = [o for o in offers if not failed(o)]
        if not fresh:
            on_failed = len(offers) - len(fresh)
            raise Stop(EXIT["budget"], "every listed offer has already failed this run "
                       f"({len(offers)} offers listed, {on_failed} on failed machines; "
                       f"{len(self.s['failed_machines'])} machines excluded in the search)")
        affordable = [o for o in fresh if o["dph"] * hours_left <= cap_left + 1e-9]
        if not affordable:
            cheapest = fresh[0]
            raise Stop(EXIT["budget"], f"no offer fits what is left: ${cap_left:.2f} for "
                       f"{hours_left:.2f} h (cheapest is ${cheapest['dph']:.3f}/hr, worst case "
                       f"${cheapest['dph'] * hours_left:.2f})")
        return affordable[0]

    # ----- failed machines and memory floors ----------------------------------------------
    def mark_machine(self, at, host_too=False):
        """Never rent this machine again in this run (F-RENT-4); `host_too` also drops every offer
        from the same host, used for CUDA/driver failures where the whole box is suspect."""
        m, h = at.get("machine"), at.get("host")
        if m is not None and m not in self.s["failed_machines"]:
            self.s["failed_machines"].append(m)
        if host_too and h is not None and h not in self.s["failed_hosts"]:
            self.s["failed_hosts"].append(h)
        self.save()

    def raise_vram_floor(self, why):
        """F-OOM-1: after a memory failure the next offer must have more GPU memory."""
        cur = self.s.get("vram_floor_gb") or search_value(self.base_config_text(),
                                                          "min_gpu_ram_gb", 24.0)
        self.s["vram_floor_gb"] = cur + VRAM_FLOOR_STEP_GB
        self.save()
        say(f"{why}: the next offer needs at least {self.s['vram_floor_gb']:g} GB of GPU memory "
            f"(was {cur:g})")

    # ----- train stage with in-place recovery ----------------------------------------------
    def inplace_check(self):
        """The worker facts for an in-place restart as a dict, or None if they cannot be read."""
        rc, out = self.run([self.vi, "cloud", "ssh", INPLACE_SCRIPT], stream=False, timeout=120)
        facts = dict(re.findall(r"^inplace: (\w+)=(\S*)", out, re.M))
        return facts if rc == 0 and {"gpu", "disk_free_gb", "procs"} <= facts.keys() else None

    def train_blocker(self, rc, out, restarts, first_fail):
        """Why the train stage cannot be restarted in place, or None when it can. A string
        starting with 'VANISHED' means the instance is gone."""
        if self.timed_out:
            return "the train stage ran out of its hours and was killed"
        # worker-job.sh reads exit 76 for an OOM even through accelerate (which turns the
        # trainer's 76 into 1) when oom.json was written in the attempt (r23); the string is
        # the fallback for an rc lost on the way home.
        if rc == 76 or "GPU out of memory" in out:
            return "the GPU ran out of memory (exit 76); a restart on this card would do the same"
        if restarts >= INPLACE_MAX_RESTARTS:
            return f"{INPLACE_MAX_RESTARTS} in-place restarts were already used"
        if clock() - first_fail > INPLACE_MAX_SECONDS:
            return (f"the first failure was more than {INPLACE_MAX_SECONDS // 60} min ago "
                    "(the in-place recovery window)")
        present, _, tracked = self.worker_present()
        if not tracked:
            return "VANISHED: the instance is gone"
        if "No space left on device" in out:
            return "the worker disk is full (ENOSPC)"
        facts = self.inplace_check()
        if facts is None:
            return "the worker could not be inspected (`vi cloud ssh` failed)"
        if facts.get("abort", "none") != "none":
            return f"the run was aborted by its own rule ({facts['abort']})"
        if facts.get("enospc") == "1":
            return "the worker disk is full (ENOSPC in the job log)"
        if facts.get("gpu") != "ok":
            return "the GPU is not visible on the worker"
        try:
            if float(facts["disk_free_gb"]) < INPLACE_MIN_DISK_GB:
                return f"the worker has under {INPLACE_MIN_DISK_GB:g} GB of free disk"
        except ValueError:
            return "the worker's free disk could not be read"
        if facts.get("ckpt", "none") == "none":
            return "no verified checkpoint on the worker to restart from"
        self.last_facts = facts
        return None

    def run_train(self, at):
        """The train stage; a failure is recovered in place on this worker (never by renting
        another one) within INPLACE_MAX_RESTARTS and INPLACE_MAX_SECONDS, else the worker is
        released and the run stops so that a person decides. A stage process that exited while
        the trainer still runs on the worker (the supervisor died) is reattached, not counted."""
        a = self.a
        restarts = reattaches = 0
        first_fail = None
        while True:
            _, hours_left = self.remaining()
            left = max(1 / 60, hours_left - (clock() - at["started"]) / 3600)
            # the worker's ETA warning reads this (on-worker.sh forwards it); refreshed on every
            # in-place restart, so it shrinks as the run uses its hours. The stale "projected"
            # marker ($work/eta.warned) is not reset: a restart that is still too slow was
            # already warned about once.
            self.extra_env = {**getattr(self, "extra_env", {}), "VI_HOURS_LEFT": f"{left:.2f}"}
            rc, out = self.run([self.vi, "dataset", "run", a.stages, "omnivoice_train"],
                               watch=True, train=True, timeout=left * 3600)
            if rc == 0:
                return
            if first_fail is None:
                first_fail = clock()
            if rc == 76 or "GPU out of memory" in out:  # see train_blocker for the two signals
                self.raise_vram_floor("train ran out of GPU memory")
            why = self.train_blocker(rc, out, restarts, first_fail)
            if why is None and self.last_facts.get("procs") not in ("0", ""):
                if reattaches >= REATTACH_MAX:
                    why = f"the stage process died {REATTACH_MAX} times while the trainer lived"
                else:
                    reattaches += 1
                    say(f"the stage exited (exit {rc}) but the trainer still runs on the worker: "
                        f"reattaching ({reattaches} of {REATTACH_MAX})")
                    nap(REATTACH_WAIT_SECONDS)
                    continue
            if why is None:
                restarts += 1
                say(f"train stage failed (exit {rc}); the worker is healthy (GPU visible, disk "
                    f"free, stage process gone, newest checkpoint {self.last_facts['ckpt']}): "
                    f"in-place restart {restarts} of {INPLACE_MAX_RESTARTS} on the same worker")
                continue
            vanished = why.startswith("VANISHED")
            self.mark_machine(at, host_too=False)
            say(f"train stage failed (exit {rc}) and cannot be recovered in place: {why}")
            self.release(at, "train-vanished" if vanished else "train-failed")
            raise Stop(EXIT["train"], f"the train stage failed (exit {rc}): {why}. No other "
                       "worker was rented; the worker was "
                       f"{'already gone' if vanished else 'released'}. Checkpoints that came home "
                       f"are under {a.checkpoint_root}. STOPPED so that you decide what next "
                       "(this machine is on the run's failed list).")

    def execute(self):
        a = self.a
        adopted = self.settle_previous()
        if self.s["credit_before_run"] is None:
            self.s["credit_before_run"] = self.credit()
            self.save()
        at = adopted
        while True:
            if at is None:
                if self.stop_file.exists():
                    raise StopRequested("stop requested; nothing was rented")
                n = len(self.s["attempts"]) + 1
                cap_left, hours_left = self.remaining()
                # Every attempt counts, including one whose `vi cloud up` failed: attempts bound
                # the number of rentals (a boot timeout bills real minutes), while the cap and
                # the hours bound the spend. Otherwise a long run of dead offers is unbounded.
                if len(self.s["attempts"]) >= a.max_attempts:
                    raise Stop(EXIT["push"], f"all {a.max_attempts} attempts allowed are used "
                               "(each ended in a failed rental or a failed cheap worker gate)")
                if hours_left < a.min_hours:
                    raise Stop(EXIT["budget"], f"only {hours_left:.2f} h of the {a.max_hours} h "
                               f"left (under --min-hours {a.min_hours}); not renting again")
                if cap_left <= 0:
                    raise Stop(EXIT["budget"], f"the ${a.cap:.2f} cap is used up")
                say(f"===== attempt {n} of {a.max_attempts} at {rfc3339(clock())} =====")
                say(f"attempt {n}: ${cap_left:.2f} and {hours_left:.2f} h left")
                self.write_attempt_config(n, hours_left)
                offer = self.choose_offer(hours_left, cap_left)
                say(f"cheapest offer not tried: {offer['id']} {offer['gpu']} {offer['loc']} "
                    f"${offer['dph']:.3f}/hr")
                at = dict(n=n, offer=offer["id"], sig=offer["sig"], dph=offer["dph"],
                          machine=offer["machine"], host=offer["host"],
                          phase="up", started=clock(), ended=None, cap=round(cap_left, 4),
                          hours=None, credit_start=self.credit())
                self.s["attempts"].append(at)
                self.save()
                self.worker_maybe_up = True
                until = rfc3339(clock() + 3600)
                boot_s = config_boot_minutes(self.base_config_text()) * 60 + UP_EXTRA_SECONDS
                rc, out = self.run([self.vi, "cloud", "up", "--offer", str(offer["id"]), "--cap",
                                    f"{cap_left:.2f}", "--until", until, "--yes"], watch=True,
                                   timeout=boot_s)
                self.cache_instance(at, out)
                if rc != 0:
                    # F-RENT-1/2: the offer is gone, or the box did not boot. Release whatever
                    # exists (a no-op when nothing was created), skip this offer, try the next.
                    boot_fault = self.timed_out or bool(BOOT_FAIL_RE.search(out))
                    say(f"`vi cloud up` failed (exit {rc}) for offer {offer['id']}"
                        f"{' (boot failure: machine marked failed)' if boot_fault else ''}")
                    self.s["failed_offers"].append(offer["id"])
                    if boot_fault:
                        self.s["failed_sigs"].append(offer["sig"])
                        self.mark_machine(at)
                    self.release(at, "up-failed")
                    at = None
                    continue
                self.set_phase(at, "preflight")
                quiet = float(os.environ.get("PAID_RUN_FATAL_QUIET_SECONDS", FATAL_QUIET_SECONDS))
                rc, out = self.run([self.vi, "dataset", "run", a.stages, "omnivoice_preflight"],
                                   watch=True, timeout=PREFLIGHT_TIMEOUT_SECONDS, fatal_quiet=quiet)
                fatal = self.fatal_stopped  # release() runs other commands, which reset it
                if rc != 0 and fatal:
                    self.release(at, "preflight-failed")
                    raise Stop(EXIT["preflight"], "the preflight printed a fatal line ("
                               f"{fatal.strip()[:200]}) and then nothing for "
                               f"{quiet:.0f} s, so it was stopped. A crash is not a per-offer gate: "
                               "the worker was released, no retry; the diag-attempt file in the run "
                               "folder holds the worker's limits and job tree.")
                if rc != 0:
                    gate = retryable_gate(out) or ("stage-timeout" if self.timed_out else None)
                    if gate:
                        say(f"offer {offer['id']} failed a cheap worker gate ({gate})")
                        self.s["failed_offers"].append(offer["id"])
                        self.s["failed_sigs"].append(offer["sig"])
                        kind = gate_kind(out)
                        self.mark_machine(at, host_too=kind == "cuda")
                        if kind == "memory":
                            self.raise_vram_floor("a memory gate failed")
                        spent = self.release(at, "worker-gate")
                        say(f"attempt {n} cost ${spent:.2f}; total ${self.s['spent_total']:.2f} "
                            f"of ${a.cap:.2f}, {self.s['hours_total']:.2f} of {a.max_hours} h")
                        at = None
                        continue
                    # A2: name a host drop as one (the full run never rents again for it)
                    dropped, drop_text = (False, None) if self.timed_out else \
                        self.host_drop(at, rc, out)
                    self.release(at, "host-drop" if dropped else "preflight-failed")
                    if dropped:
                        raise Stop(EXIT["preflight"], f"the preflight lost its worker: {drop_text}; "
                                   f"{failure_rc_text(rc, out)}. The worker was released; the "
                                   "full run never rents again after a host drop")
                    raise Stop(EXIT["preflight"], f"the preflight failed ({failure_rc_text(rc, out)}) on a gate "
                               "that is not a retryable worker gate; the worker was released, no retry")
                self.set_phase(at, "train")
            # -- train (new or adopted) --
            say(f"===== train stage (attempt {at['n']}) at {rfc3339(clock())} =====")
            say("preflight passed; starting the train stage")
            self.run_train(at)
            self.set_phase(at, "verify")
            # r32 WP7: destroy first; the home-side verify needs no worker and can take minutes
            spent = self.release(at, "done")
            problems = self.verify_final()
            if problems:
                at["outcome"] = "verify-failed"
                self.save()
                raise Stop(EXIT["verify"], "the train stage finished but the results at home do "
                           "not check out:\n  " + "\n  ".join(problems))
            return spent

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
