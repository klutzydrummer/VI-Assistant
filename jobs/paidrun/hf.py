"""The Hugging Face base-model gate (F-NET-4)."""

import importlib.util
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from paidrun.common import EXIT, REPO, nap
from paidrun.limits import HF_BACKOFF, HF_TIMEOUT_SECONDS, HF_TRIES
from paidrun.config import GATE_SCRIPT


HF_REPO = "k2-fsa/OmniVoice"
HF_API = "https://huggingface.co/api/models"  # the tests point it at a local stub server


def base_model_revision(script=GATE_SCRIPT):
    """The base-model revision pinned in the stage script (shown at --plan, not checked)."""
    try:
        m = re.search(r"VI_BASE_MODEL_REVISION:-([0-9a-f]{12,40})", Path(script).read_text())
    except OSError:
        m = None
    return m.group(1) if m else "unknown"


# F-NET-4: the pinned base-model revision must still exist upstream, with the pinned file sizes.


def load_size_table():
    """(SIZE_TABLE, DEFAULT_REVISION_PREFIX) from jobs/omnivoice-ft/fetch_base_model.py."""
    spec = importlib.util.spec_from_file_location(
        "fetch_base_model_pins", REPO / "jobs" / "omnivoice-ft" / "fetch_base_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.SIZE_TABLE), mod.DEFAULT_REVISION_PREFIX


class RevisionNotFoundError(Exception):
    """The Hub answered 404 for the pinned revision (X-Error-Code: RevisionNotFound)."""


class RepositoryNotFoundError(Exception):
    """The Hub answered 401 or 403: the repo is gone, private or gated."""


def hf_fetch(url, body, timeout):
    """GET url (POST when body is a dict) and return the parsed JSON reply. Standard library only,
    because the paid run starts under the system python3 -I, which has no huggingface_hub (r8)."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        why = e.headers.get("X-Error-Message") or e.reason
        if e.code == 404:
            raise RevisionNotFoundError(f"HTTP 404 {e.headers.get('X-Error-Code', '')}: {why}")
        if e.code in (401, 403):
            raise RepositoryNotFoundError(f"HTTP {e.code}: {why}")
        raise  # 5xx, 429 and the like are retried as network failures


def hf_model_info(repo, revision, timeout, fetch=None):
    """The one network step of the gate (tests replace it): the commit the pinned revision
    resolves to, and the sizes of the pinned files at that commit.

    Both endpoints are in the official Hub API spec, https://huggingface.co/.well-known/openapi.md
    (linked from https://huggingface.co/docs/hub/api, read 2026-11-20):
      GET  /api/models/{namespace}/{repo}/commits/{rev}     -> [{"id": <commit sha>, ...}]
      POST /api/models/{namespace}/{repo}/paths-info/{rev}  {"paths": [...]}
                                                            -> [{"path": ..., "size": ...}]
    The spec lists no response fields, so the field names were checked against live replies on
    2026-11-20: commits gives 404 with X-Error-Code RevisionNotFound for an unknown revision,
    an unknown repo gives 401, and paths-info leaves out paths it does not have."""
    fetch = fetch or hf_fetch
    base = f"{HF_API}/{repo}"
    commits = fetch(f"{base}/commits/{revision}?limit=1", None, timeout)
    sha = commits[0]["id"] if commits else None
    table, _ = load_size_table()
    files = fetch(f"{base}/paths-info/{sha or revision}", {"paths": sorted(table)}, timeout)
    return SimpleNamespace(sha=sha, siblings=[{"rfilename": f.get("path"), "size": f.get("size")}
                                              for f in files])


def hf_offline_check(revision=None):
    """(ok, line) for --plan: runs the whole gate on canned Hub replies, with no network, so a
    missing module or a broken parser shows up before anything is rented."""
    rev = revision or base_model_revision()
    urls = []

    def canned(url, body, timeout):
        urls.append(url)
        if "/commits/" in url:
            return [{"id": rev, "title": "canned"}]
        table, _ = load_size_table()
        return [{"type": "file", "path": p, "size": table[p]} for p in body["paths"]]

    try:
        ok, lines = hf_revision_gate(revision=rev, sleep=lambda _s: None,
                                     model_info=lambda r, v, t: hf_model_info(r, v, t, canned))
    except Exception as e:  # noqa: BLE001: any crash here would crash the real gate too
        ok, lines = False, [f"{type(e).__name__}: {e}"]
    label = "hf check code   : "
    if ok and len(urls) == 2:
        return True, f"{label}OK offline on canned Hub replies (no network call in --plan)"
    return False, (f"{label}BROKEN offline ({lines[0].removeprefix('hf revision    : ')}); the "
                   f"real check before renting would stop with exit {EXIT['hf']}")


def hf_revision_gate(home=None, revision=None, model_info=None, sleep=None):
    """(ok, lines). A revision that is gone or has other file sizes fails. A network failure
    after HF_TRIES tries is only a warning when the home HF cache holds the snapshot."""
    model_info = model_info or hf_model_info
    sleep = sleep or nap
    label = "hf revision    : "
    rev = revision or base_model_revision()
    home = Path(home or Path.home())
    snap = (home / ".cache" / "huggingface" / "hub" / "models--k2-fsa--OmniVoice" / "snapshots" / rev)
    last = None
    for i in range(HF_TRIES):
        try:
            info = model_info(HF_REPO, rev, HF_TIMEOUT_SECONDS)
            break
        except Exception as e:  # noqa: BLE001: classified below
            last = e
            if isinstance(e, ImportError):  # ModuleNotFoundError included
                # Retrying cannot fix a missing library, and a cached snapshot does not prove
                # the pinned revision still exists, so this is a hard stop (r7 lost its gate here).
                return False, [f"{label}could not be checked: {type(e).__name__}: {e}. The check "
                               "needs only the Python standard library, so this is a bug in "
                               "jobs/paid_run.py; fix it and run again. Nothing was rented."]
            if type(e).__name__ in ("RevisionNotFoundError", "RepositoryNotFoundError",
                                    "EntryNotFoundError", "GatedRepoError"):
                return False, [f"{label}NOT FOUND: {HF_REPO} @ {rev} ({type(e).__name__}: {e}); the "
                               "worker would fail after being rented. Nothing was rented."]
            if i + 1 < HF_TRIES:
                sleep(HF_BACKOFF[min(i, len(HF_BACKOFF) - 1)])
    else:
        if snap.is_dir():
            return True, [f"{label}WARNING not checked ({type(last).__name__}: {last}); the home "
                          f"cache holds the snapshot {snap}"]
        return False, [f"{label}could not be checked after {HF_TRIES} tries ({type(last).__name__}: "
                       f"{last}) and the home cache has no snapshot at {snap}. Nothing was rented."]
    sha = getattr(info, "sha", None)
    if not sha or not sha.startswith(rev):
        return False, [f"{label}MISMATCH: {HF_REPO} @ {rev} resolved to commit {sha}, not the "
                       "pinned one. Nothing was rented."]
    table, prefix = load_size_table()
    if not rev.startswith(prefix):
        return True, [f"{label}ok ({HF_REPO} @ {rev} exists; the pinned size table is for {prefix}, "
                      "so sizes were not compared)"]
    sizes = {}
    for f in info.siblings or []:
        name, size = (f.get("rfilename"), f.get("size")) if isinstance(f, dict) else (
            getattr(f, "rfilename", None), getattr(f, "size", None))
        sizes[name] = size
    bad = [f"{name}: {sizes.get(name)} bytes upstream, pinned {want}"
           for name, want in table.items() if sizes.get(name) != want]
    if bad:
        return False, [f"{label}CHANGED upstream: " + "; ".join(bad) + ". Nothing was rented."]
    return True, [f"{label}ok ({HF_REPO} @ {rev}; pinned file sizes match)"]
