"""r32 WP3 (C.1, C.4, C.6): the rental validity state machine, layer L1 of the layered destroy.

One Validity per rental. It answers one question, "is this rental still worth paying for?", as
VALID, SUSPECT or INVALID(reason class):

- VALID: the stage's required evidence is fresh and its progress counter (when it has one) moves.
- SUSPECT: a stall window passed with no counter advance, or required evidence went stale.
  Recovery is bounded per stage by `tries` (try_recover) and `recover_s` (time spent SUSPECT).
  Only NEW positive evidence, a progress counter strictly increasing, makes it VALID again.
  Liveness (vast_running, ssh_ok, return_push_ok, a live process, an mtime) never does.
- INVALID: recovery exhausted or the stage deadline passed (stalled), the effective cap reached
  (cap), a confirmed host drop (host-drop), a failed host qualification (host-unqualified, WP5),
  or our own failure (our-failure). INVALID is final: the first reason wins.

The effective cap is min(wall_cap_s, dollar_cap / (dph + storage_dph) hours), fixed when the
rental is armed and never extended. The clock is injectable, so tests never sleep.

Nothing here runs a command or talks to Vast; the orchestrator feeds evidence, polls, and on
INVALID releases the rental (destroy first, see ReleaseMixin.release)."""

from dataclasses import dataclass

from paidrun.common import EXIT, Stop, clock as default_clock


VALID, SUSPECT, INVALID = "VALID", "SUSPECT", "INVALID"
STATES = (VALID, SUSPECT, INVALID)

REASON_HOST_DROP = "host-drop"
REASON_HOST_UNQUALIFIED = "host-unqualified"
REASON_STALLED = "stalled"
REASON_OUR_FAILURE = "our-failure"
REASON_CAP = "cap"
REASONS = (REASON_HOST_DROP, REASON_HOST_UNQUALIFIED, REASON_STALLED, REASON_OUR_FAILURE,
           REASON_CAP)
# Reasons the machine reaches by itself while a stage runs (poll); the orchestrator stops the
# stage at once for these. The others are recorded by the code that saw them.
SELF_REASONS = (REASON_CAP, REASON_STALLED)

EV_VAST_RUNNING = "vast_running"
EV_SSH_OK = "ssh_ok"
EV_RETURN_PUSH_OK = "return_push_ok"
EV_PROGRESS = "progress"
LIVENESS_KINDS = (EV_VAST_RUNNING, EV_SSH_OK, EV_RETURN_PUSH_OK)
EVIDENCE_KINDS = LIVENESS_KINDS + (EV_PROGRESS,)


@dataclass(frozen=True)
class StageSpec:
    """One row of the C.2 table. Seconds; None means the stage has no such limit. `counter` is
    the progress counter WP4 will report (`VI-PROGRESS <stage> <counter> <value>`)."""
    deadline_s: float | None
    stall_s: float | None
    counter: str | None
    tries: int = 1
    recover_s: float | None = None
    # units per second: an advance slower than this, measured since the last counted reading,
    # is not progress (C.2: upload "60 s at < 0.5 MB/s", base-model fetch "120 s at < 1 MB/s")
    min_rate: float | None = None


# C.2 (w33 report), about 3x the worst good past run and at least 120 s. WP4: the worker now
# reports these counters (`VI-PROGRESS <stage> <counter> <value>`), and PaidRun.note_progress
# enters each reported stage with its stall_s and min_rate, so a stalled or trickling stage turns
# SUSPECT. The deadlines, tries and recover_s are still not enforced: the existing ad-hoc limits
# (see PaidRun.run and on-worker.sh) keep deciding when a stage is stopped.
STAGE_TABLE = {
    "create-running": StageSpec(600, None, None),
    "boot-ssh": StageSpec(900, None, None),
    "host-qual": StageSpec(240, 60, "probe-bytes"),
    "upload": StageSpec(300, 60, "bytes-acked", min_rate=0.5e6),
    # pip reports no package count it can be asked for; the worker reports bytes of wheels
    # and installed files instead (VI-PROGRESS venv install-bytes)
    "venv": StageSpec(300, 120, "install-bytes"),
    "base-model-fetch": StageSpec(300, 120, "bytes-on-disk", min_rate=1e6),
    # upstream's token extractor reports no item count; the worker reports shard bytes written
    "tokenise": StageSpec(450, 120, "shard-bytes"),
    "cuda-load": StageSpec(300, None, "loaded"),
    # first step within 300 s; after that the whole-run cap, stall max(600, 5x median step)
    "train": StageSpec(None, 600, "step"),
    # deadline max(600, 1.5x bytes/rate) is computed per push
    "checkpoint-push": StageSpec(600, 120, "files-pushed"),
}
TRAIN_FIRST_STEP_SECONDS = 300
TRAIN_STALL_MIN_SECONDS = 600
TRAIN_STALL_MEDIAN_FACTOR = 5
# return_push_ok must arrive once per lost_limit in every stage
RETURN_PUSH_LOST_SECONDS = {"debug": 5 * 60, "full": 30 * 60}


def train_stall_seconds(median_step_s):
    """C.2: a train stage is SUSPECT after max(600 s, 5x the median step time) with no step."""
    return max(TRAIN_STALL_MIN_SECONDS, TRAIN_STALL_MEDIAN_FACTOR * (median_step_s or 0))


def checkpoint_push_deadline_seconds(nbytes, bytes_per_s):
    """C.2: a checkpoint push gets max(600 s, 1.5x bytes/rate)."""
    if not bytes_per_s:
        return STAGE_TABLE["checkpoint-push"].deadline_s
    return max(STAGE_TABLE["checkpoint-push"].deadline_s, 1.5 * nbytes / bytes_per_s)


def effective_cap_seconds(wall_cap_s, dollar_cap=None, dph=None, storage_dph=0.0):
    """C.4: min(wall cap, $cap / (price/hr + storage/hr)), in seconds. A missing $ side leaves
    the wall cap; a missing wall side leaves the $ cap; both missing is None (no cap)."""
    caps = []
    if wall_cap_s is not None:
        caps.append(max(0.0, float(wall_cap_s)))
    rate = (dph or 0.0) + (storage_dph or 0.0)
    if dollar_cap is not None and rate > 0:
        caps.append(max(0.0, float(dollar_cap)) / rate * 3600)
    return min(caps) if caps else None


def host_unqualified_allows_rental2():
    """WP5 / fork D3 (decided): a failed host qualification may buy rental 2, like a confirmed
    host drop. The failed host's host id and machine id are excluded from the next offer search,
    and rental 2 stays inside DEBUG_MAX_RENTALS and the $ cap left."""
    return True


def rental2_allowed(reason):
    """C.6: whether an INVALID rental with this reason class may be followed by rental 2
    (still bounded by DEBUG_MAX_RENTALS and the $ cap left). A confirmed host drop and a failed
    host qualification do (D3, through the WP5 hook); stalled, our-failure and cap never."""
    if reason == REASON_HOST_DROP:
        return True
    if reason == REASON_HOST_UNQUALIFIED:
        return host_unqualified_allows_rental2()
    return False


# Release outcomes (ReleaseMixin.release) -> the reason class recorded for that rental. None:
# not a failed rental (a good end, an orphan sweep) or not attributable (a user stop or signal,
# a worker that vanished without a confirmed host drop).
OUTCOME_REASONS = {
    "host-drop": REASON_HOST_DROP,
    "host-unqualified": REASON_HOST_UNQUALIFIED,
    "worker-gate": REASON_HOST_UNQUALIFIED,
    "up-failed": REASON_HOST_UNQUALIFIED,
    "preflight-failed": REASON_OUR_FAILURE,
    "train-failed": REASON_OUR_FAILURE,
    "debug-failed": REASON_OUR_FAILURE,
    "aborted": REASON_OUR_FAILURE,
}


def reason_for_outcome(outcome):
    return OUTCOME_REASONS.get(outcome)


class Invalidated(Stop):
    """Raised out of a running stage when the rental turned INVALID by itself (cap, stalled).
    The worker is then released by the usual teardown, destroy first."""

    def __init__(self, reason, detail, stage=None):
        code = EXIT["budget"] if reason == REASON_CAP else EXIT["other"]
        where = f" during {stage}" if stage else ""
        super().__init__(code, f"the rental became INVALID ({reason}){where}: {detail}; the "
                         "worker was destroyed first, diagnostics only after")
        self.reason, self.detail, self.stage = reason, detail, stage


class Validity:
    def __init__(self, started=None, wall_cap_s=None, dollar_cap=None, dph=None, storage_dph=0.0,
                 clock=None, attempt=None):
        self.clock = clock or default_clock
        self.attempt = attempt
        self.started = self.clock() if started is None else started
        self.cap_s = effective_cap_seconds(wall_cap_s, dollar_cap, dph, storage_dph)
        self.cap_at = None if self.cap_s is None else self.started + self.cap_s
        self.cap_detail = (f"effective cap {self.cap_s:.0f} s = min(wall {wall_cap_s}, "
                           f"${dollar_cap} at ${(dph or 0) + (storage_dph or 0):.3f}/hr)"
                           if self.cap_s is not None else "no cap")
        self.state = VALID
        self.reason = self.detail = None
        self.invalid_at = None
        self.stage = None
        self.counters = {}
        self.rate_base = {}
        self.seen = {}
        self._enter_stage(None, self.started, None, None, 1, None, None)

    # ----- stages ---------------------------------------------------------------------------
    def _enter_stage(self, stage, now, deadline_s, stall_s, tries, recover_s, required):
        self.stage = stage
        self.stage_started = now
        self.deadline_s = deadline_s
        self.stall_s = stall_s
        self.tries = tries
        self.tries_used = 0
        self.recover_s = recover_s
        self.recover_used = 0.0
        self.required = dict(required or {})
        self.last_progress_at = now
        self.suspect_since = None
        if self.state == SUSPECT:
            self.state = VALID

    def enter(self, stage, deadline_s=None, stall_s=None, tries=1, recover_s=None, required=None,
              now=None):
        """Start `stage`. deadline_s: the whole stage, else INVALID(stalled). stall_s: no counter
        advance this long -> SUSPECT. tries / recover_s: the recovery bound for this stage.
        required: {evidence kind: max age s}; stale or missing -> SUSPECT. The cap is kept."""
        now = self.clock() if now is None else now
        if self.state != INVALID:
            self._enter_stage(stage, now, deadline_s, stall_s, tries, recover_s, required)
        return self.state

    # ----- evidence -------------------------------------------------------------------------
    def evidence(self, kind, value=None, counter=None, now=None, min_rate=None):
        """Record positive evidence. Liveness kinds only refresh their timestamp. `progress`
        counts only when its counter strictly increases; that alone moves SUSPECT to VALID.
        min_rate (units/s): the increase must also average at least this since the last counted
        reading, so a trickle never counts however long it keeps creeping up."""
        if kind not in EVIDENCE_KINDS:
            raise ValueError(f"unknown evidence kind {kind!r}")
        now = self.clock() if now is None else now
        self.seen[kind] = now
        if kind != EV_PROGRESS or self.state == INVALID:
            return self.state
        name = counter or self.stage or "progress"
        try:
            value = float(value)
        except (TypeError, ValueError):
            return self.state
        prev = self.counters.get(name)
        if prev is not None and value <= prev:
            return self.state
        self.counters[name] = value
        if prev is None:
            # the first reading of a counter is a baseline, not an advance
            self.rate_base[name] = (value, now)
            return self.state
        if min_rate:
            base_value, base_at = self.rate_base.get(name, (prev, now))
            if now <= base_at or (value - base_value) / (now - base_at) < min_rate:
                return self.state
        self.rate_base[name] = (value, now)
        self.last_progress_at = now
        if self.state == SUSPECT:
            self.recover_used += now - self.suspect_since
            self.suspect_since = None
            self.state = VALID
        return self.state

    # ----- decisions ------------------------------------------------------------------------
    def invalidate(self, reason, detail="", now=None):
        """Mark the rental INVALID. Final: a later call keeps the first reason."""
        if reason not in REASONS:
            raise ValueError(f"unknown reason class {reason!r}")
        if self.state != INVALID:
            self.state, self.reason, self.detail = INVALID, reason, detail
            self.invalid_at = self.clock() if now is None else now
        return self.state

    def try_recover(self, now=None):
        """Use one recovery try in a SUSPECT stage. True when the try may go ahead; False (and
        INVALID(stalled)) when the stage's tries are used up."""
        now = self.clock() if now is None else now
        if self.poll(now) != SUSPECT:
            return False
        if self.tries_used >= self.tries:
            self.invalidate(REASON_STALLED, f"{self.stage or 'stage'}: {self.tries} recovery "
                            "tries used and no new progress", now)
            return False
        self.tries_used += 1
        return True

    def poll(self, now=None):
        now = self.clock() if now is None else now
        if self.state == INVALID:
            return self.state
        if self.cap_at is not None and now >= self.cap_at:
            return self.invalidate(REASON_CAP, self.cap_detail, now)
        stage = self.stage or "stage"
        if self.deadline_s is not None and now - self.stage_started >= self.deadline_s:
            return self.invalidate(REASON_STALLED, f"{stage} passed its {self.deadline_s:.0f} s "
                                   "deadline", now)
        why = None
        if self.stall_s is not None and now - self.last_progress_at >= self.stall_s:
            why = f"no progress for {self.stall_s:.0f} s"
        for kind, age in self.required.items():
            at = self.seen.get(kind)
            if at is None or now - at > age:
                why = why or f"{kind} older than {age:.0f} s"
        if self.state == VALID and why is None:
            return self.state
        # SUSPECT stays SUSPECT until a counter advances (evidence); liveness coming back,
        # or simply no error, never ends it
        if self.state == VALID:
            self.state, self.suspect_since, self.detail = SUSPECT, now, why
        if self.recover_s is not None and \
                self.recover_used + (now - self.suspect_since) >= self.recover_s:
            return self.invalidate(REASON_STALLED, f"{stage}: {why}; recovery took longer than "
                                   f"{self.recover_s:.0f} s", now)
        return self.state

    def stop(self):
        """The exception that ends the running stage for a self-reached INVALID."""
        return Invalidated(self.reason, self.detail, self.stage)

    def snapshot(self):
        return dict(attempt=self.attempt, state=self.state, reason=self.reason,
                    detail=self.detail, stage=self.stage, cap_at=self.cap_at,
                    cap_seconds=self.cap_s, invalid_at=self.invalid_at)
