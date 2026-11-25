"""Run and disk config: TOML text edits, disk gate and cost math, stage settings and step constants."""

import os
import re
from pathlib import Path

from paidrun.common import REPO
from paidrun.limits import DEFAULT_BOOT_MINUTES


EU_COUNTRIES = ("US AT BE BG HR CY CZ DK EE FI FR DE GR HU IE IT LV LT LU MT NL PL PT RO SK SI "
                "ES SE GB CH NO").split()
# Home keeps the two newest checkpoints plus the exempt steps (jobs/checkpoints.py
# EXEMPT_STEPS = 1000, 2500) when a 5000-step run ends.
EXPECTED_STEPS = (1000, 2500, 4500, 5000)
LISTENING_STEPS = (1000, 2500, 5000)
EXPECT_EPOCHS = 4


CKPT_EVERY = 500


# The worker's preflight (jobs/voicetune-fresh.sh, `min_disk = float(os.environ.get(
# "VI_MIN_DISK_GB", "100"))`) refuses a worker with less free disk than this many GiB. The
# default is read from that script so the two cannot drift; GATE_FALLBACK is only used if the
# script is missing, and a test fails if it differs from the script.
GATE_SCRIPT = REPO / "jobs" / "voicetune-fresh.sh"
GATE_FALLBACK_GB = 100.0
# Measured on omnivoice-ft-r4: a 100 GB rented disk showed 95 GB free at the gate.
DISK_OVERHEAD_GB = 5.0
# Vast's offer filter default when a run config sets no disk_gb (crates/cloud/src/config.rs).
CONFIG_DEFAULT_DISK_GB = 60.0


# C: a debug pass rents at most this many workers; the second only after a confirmed host drop.
DEBUG_MAX_RENTALS = 2


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


def set_search_value(text, key, value):
    """A copy of a cloud config with `key = value` set inside its [search] table."""
    lines = text.splitlines()
    new = f"{key} = {value}"
    in_search, at = False, None
    for i, ln in enumerate(lines):
        t = ln.strip()
        if t.startswith("["):
            if in_search:
                break
            in_search = t == "[search]"
            if in_search:
                at = i + 1
        elif in_search and re.match(rf"{re.escape(key)}\s*=", t):
            lines[i], at = new, None
            break
    if at is not None:
        lines.insert(at, new)
    elif not any(ln.strip() == "[search]" for ln in lines):
        lines += ["[search]", new]
    return "\n".join(lines) + "\n"


def search_value(text, key, default=None):
    in_search = False
    for ln in text.splitlines():
        t = ln.strip()
        if t.startswith("["):
            in_search = t == "[search]"
        elif in_search and re.match(rf"{re.escape(key)}\s*=", t):
            try:
                return float(t.split("=", 1)[1].split("#")[0].strip())
            except ValueError:
                return default
    return default


def config_boot_minutes(text):
    for ln in text.splitlines():
        if ln.startswith("["):
            break
        m = re.match(r"boot_minutes\s*=\s*(\d+)", ln)
        if m:
            return int(m.group(1))
    return DEFAULT_BOOT_MINUTES


def gate_default_gb(script=GATE_SCRIPT):
    """The free-disk gate default (GiB) written in the worker preflight script."""
    try:
        m = re.search(r'"VI_MIN_DISK_GB",\s*"([\d.]+)"', Path(script).read_text())
    except OSError:
        m = None
    return float(m.group(1)) if m else GATE_FALLBACK_GB


def config_disk_gb(text):
    """The top-level disk_gb of a cloud config (the Vast default when it sets none)."""
    for ln in text.splitlines():
        if ln.startswith("["):
            break
        m = re.match(r"\s*disk_gb\s*=\s*([\d.]+)", ln)
        if m:
            return float(m.group(1))
    return CONFIG_DEFAULT_DISK_GB


def disk_check(config_text, env=None):
    """(ok, disk_gb, gate_gb, line): can the rented disk pass the worker's free-disk gate?"""
    env = os.environ if env is None else env
    disk = config_disk_gb(config_text)
    try:
        gate = float(env.get("VI_MIN_DISK_GB") or gate_default_gb())
    except ValueError:
        gate = gate_default_gb()
    ok = disk - DISK_OVERHEAD_GB >= gate
    line = (f"worker disk     : disk_gb {disk:g} - overhead {DISK_OVERHEAD_GB:g} vs gate "
            f"{gate:g} GiB free: {'OK' if ok else 'FAIL'}")
    return ok, disk, gate, line


# Assumed ceiling for Vast's storage price, in $/GB/month. The real storage_cost is only known
# from the live offer, so this is a bound for the plan, not a quote.
DISK_CEILING_PER_GB_MONTH = 1.00
HOURS_PER_MONTH = 730


def disk_cost_line(disk_gb, hours):
    """The plan line about disk billing: Vast charges storage on top of the GPU price, per GB and
    second, until the instance is destroyed (stopping does not end it)."""
    cost = disk_gb * DISK_CEILING_PER_GB_MONTH / HOURS_PER_MONTH * hours
    return (f"disk billing    : Vast bills disk on top of the GPU price, per GB-hour until the "
            f"instance is destroyed; worst case for {disk_gb:g} GB over {hours:g} h is about "
            f"${cost:.2f} at an assumed ${DISK_CEILING_PER_GB_MONTH:.2f}/GB/month (estimate; "
            "the real storage_cost is only known from the live offer)")


def debug_worst_case(cap, disk_gb, hours, max_dph=None):
    """(total, gpu, disk) worst-case dollars of a debug pass of up to DEBUG_MAX_RENTALS rentals of
    `hours` each (r21: the hours are per rental). The GPU part is at most 2 x hours x max_dph (the
    config's price ceiling) and never more than the shared cap: an offer is rented only if its
    $/hr x the rental's hours fits the cap left, and `vi cloud up` gets that cap. Disk is billed
    on top for both rentals' hours (disk_cost_line)."""
    gpu = cap if max_dph is None else min(cap, DEBUG_MAX_RENTALS * hours * max_dph)
    disk = disk_gb * DISK_CEILING_PER_GB_MONTH / HOURS_PER_MONTH * hours * DEBUG_MAX_RENTALS
    return gpu + disk, gpu, disk


def config_max_dph(text):
    """search.max_dph of a cloud config ($/hr ceiling of an offer), or None when it is not set."""
    m = re.search(r"^\s*max_dph\s*=\s*([0-9.]+)", text or "", re.M)
    try:
        return float(m.group(1)) if m else None
    except ValueError:
        return None


def stage_settings(stages, repo, stage):
    """The [stage.<stage>.settings] table of the stage file, or None when it cannot be read."""
    try:
        import tomllib
        p = Path(stages)
        p = p if p.is_absolute() else Path(repo) / p
        return tomllib.loads(p.read_text())["stage"][stage].get("settings", {})
    except (ImportError, OSError, KeyError, ValueError):
        return None


CHECKPOINT_BYTES_CALCULATED = 3 * 2_450_344_112  # weights + two AdamW states (WP3 / w27 report)
WEIGHTS_BYTES_CALCULATED = 2_450_344_112


def push_hours_line(settings, bytes_full=CHECKPOINT_BYTES_CALCULATED,
                    bytes_weights=WEIGHTS_BYTES_CALCULATED):
    """`push ≈ X h at warn rate` for one full checkpoint and one weights-only checkpoint, at the
    stage's warn_push_mb_s (a warning threshold since round 19, never a refusal; the worker
    preflight prints the real measured rate at run time). The sizes are CALCULATED (3 x the
    2,450,344,112-byte weights file), not measured; jobs/checkpoints.py only enforces lower bounds."""
    st = settings or {}
    floor = float(st.get("warn_push_mb_s", st.get("min_push_mb_s", 2.5)))
    full, weights = bytes_full / (floor * 1e6) / 3600, bytes_weights / (floor * 1e6) / 3600
    return (f"push time       : push ≈ {full:.2f} h at warn rate ({floor:g} MB/s, a warning, "
            f"not a gate; measured at run time) per full checkpoint "
            f"({bytes_full / 1e9:.2f} GB), {weights:.2f} h per weights-only one "
            f"({bytes_weights / 1e9:.2f} GB); sizes calculated, not measured (jobs/checkpoints.py "
            f"floors {'2.0 GB weights + 4.0 GB optimizer'})")
