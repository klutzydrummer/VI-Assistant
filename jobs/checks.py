"""Confound reports and acceptance checks for the dataset jobs.

Standard library only, so it runs in CI with the fake models.

The smoke tests showed every off-the-shelf model breaking an assumption on
Tali's audio (docs/design/methodology-review.md, section 1): an embedding
that mostly encoded clip length, a speaker check that recognised the game,
an audio LLM that refused short clips or tagged every clip the same way.
These were found by hand. This module makes finding them routine:

- confounds(): how much of a signal clip length, game and bark-vs-
  conversation explain. A signal that mostly tracks them would teach a vector
  "long line" or "ME3 recording" instead of a feeling.
- Check: a named pass/fail test with the measured value and its limit. Each
  stage writes its checks to checks.json; with settings `checks = "fail"`
  (the smoke pipeline) any failure fails the stage, so problems surface on
  40 clips before a full run pays for them. The default, "warn", only logs.
"""

import json
import math
import sys
from dataclasses import asdict, dataclass

import fusion
from vi_job import log


def eta_squared(values, groups):
    """Share of the variance in values explained by group membership
    (between-group sum of squares over total), 0 to 1."""
    pairs = [(v, g) for v, g in zip(values, groups) if v is not None]
    if len(pairs) < 3:
        return None
    mean = sum(v for v, _ in pairs) / len(pairs)
    total = sum((v - mean) ** 2 for v, _ in pairs)
    if total == 0:
        return None
    by = {}
    for v, g in pairs:
        by.setdefault(g, []).append(v)
    between = sum(len(vs) * (sum(vs) / len(vs) - mean) ** 2 for vs in by.values())
    return round(between / total, 3)


def confounds(values, clips):
    """How much of a per-clip signal three non-emotion facts explain:
    Spearman with log-duration, and eta squared for game and for bark vs
    conversation. `values` lines up with `clips` (None where unscored)."""
    return {
        "duration_spearman": fusion.spearman(values, [math.log(max(c.duration_s, 0.05)) for c in clips]),
        "game_eta2": eta_squared(values, [c.game for c in clips]),
        "bark_eta2": eta_squared(values, ["bark" in c.flags for c in clips]),
    }


# A confound this strong on a signal gets a warning in the report.
CONFOUND_LIMITS = {"duration_spearman": 0.5, "game_eta2": 0.2, "bark_eta2": 0.3}


@dataclass
class Check:
    name: str
    value: float | None
    limit: float
    # "max": value must be <= limit; "min": value must be >= limit.
    kind: str = "max"
    note: str = ""

    @property
    def ok(self):
        if self.value is None:
            return True
        return self.value <= self.limit if self.kind == "max" else self.value >= self.limit


def confound_checks(prefix, report):
    """Checks for a confounds() report: each measure within its limit."""
    out = []
    for key, limit in CONFOUND_LIMITS.items():
        v = report.get(key)
        out.append(Check(f"{prefix}: {key}", None if v is None else abs(v), limit,
                         note="a signal this tied to it may teach the vector the wrong thing"))  # fmt: skip
    return out


def finish(out, checks, mode):
    """Write checks.json, log failures, and exit 1 if mode is "fail" and any
    check failed."""
    failed = [c for c in checks if not c.ok]
    (out / "checks.json").write_text(
        json.dumps({"mode": mode, "failed": len(failed), "checks": [asdict(c) | {"ok": c.ok} for c in checks]},
                   indent=1) + "\n"  # fmt: skip
    )
    for c in failed:
        log(f"CHECK FAILED: {c.name} = {c.value} (limit {c.kind} {c.limit}) {c.note}")
    if failed and mode == "fail":
        log(f"{len(failed)} acceptance check(s) failed; see checks.json")
        sys.exit(1)
