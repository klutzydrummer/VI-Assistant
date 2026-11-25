# Paid-run coding guide

Written 2026-11-25 for round 33. Covers the code that rents, watches and releases a Vast.ai worker: `jobs/paid_run.py` (the entry point) and the `jobs/paidrun/` package it loads, `jobs/on-worker.sh`, `jobs/worker-onstart.sh`, `jobs/debug-pass.sh`, `jobs/push-probe.sh`, `jobs/voicetune-fresh.sh`, `jobs/worker-job.sh`, `jobs/worker-return.sh` and `crates/cloud`. It is for anyone changing that code and anyone writing or reviewing a brief for it. The user-facing commands are in `cloud-worker.md`; why the work is split between home and worker is in `design/local-cloud-split.md`.

The design behind it is the round 32 audit (rental-wait table, validity design C.1-C.7, work packages WP1-WP7). Every rule below applies to new work now, including the parts not yet built, so the later work packages fit together.

## What is built and what is planned

| Piece | Status |
|---|---|
| WP1: destroy-first release, `vi cloud down --force --verify`, exit 9 = destroy not verified | **Built** (PR #185) |
| WP7: verify-epoch bounded at 300 s, `verify_final` after release, instance id cached from `up` | **Built** (PR #185) |
| WP3: `jobs/paidrun/validity.py` state machine and the cap | Planned |
| WP4: worker progress counters (`VI-PROGRESS stage counter value`) | Planned |
| WP2: home dead-man timer (L2) | Planned |
| WP6: worker watchdog fixes (L3) | Planned |
| WP5: host qualification gate in every mode, rental-2 rules, host exclusion | Planned |

Until WP3 and WP4 land, progress is still judged by "some file changed" (`on-worker.sh`), and stage limits are still spread across the files listed in "Where limits live today" below.

## Architecture

Three layers each decide on their own whether to destroy a rental. Each one only adds protection, so a later layer never relies on an earlier one having worked.

- **L1, the validity state machine** (`jobs/paidrun/validity.py`, owned by the paid run; planned in WP3). This is the normal path. Every stage registers its deadline and the evidence it will produce. The machine moves through these states:
  - **VALID** while fresh positive evidence keeps arriving.
  - **SUSPECT** when an item goes stale. It then runs bounded recovery using only the actions that stage registered.
  - **VALID** again only on *new* positive evidence. "No error" does not count.
  - **INVALID** when recovery is used up, the cap is reached or a host drop is confirmed. It records one reason: `host-drop`, `host-unqualified`, `stalled`, `our-failure` or `cap`.
  - From INVALID it goes to **DESTROY**, then **VERIFY**, then **DIAG**, then **BOOKKEEPING**.
- **L2, the home dead-man timer** (planned in WP2). This is a transient `systemd-run --user` timer, armed by `vi cloud up` with the instance id right after the create call.
  - It fires at `min(until, start + wall cap, start + $ cap ÷ hourly price)`.
  - L1's heartbeats push it forward, but never past that hard cap.
  - When it fires it runs `vi cloud down --force --verify <id>`.
  - It is cancelled only after a destroy has been verified.
  - It survives a crash, OOM kill, SIGKILL or hang of `paid_run.py`. It does not survive a home reboot. That is deliberate (decision D1): there is no boot-time settle unit, and L3 covers a reboot.
- **L3, the worker watchdog** (`jobs/worker-onstart.sh`, rendered by `crates/cloud/src/onstart.rs`; fixes planned in WP6).
  - It self-destroys at the hard cap + 10 min with the instance's own key.
  - It ignores home's tunnel ssh session when deciding whether the worker is idle.
  - If DELETE is refused it falls back to the documented `vastai stop instance $CONTAINER_ID` (decision D4). That ends compute billing but not storage billing, so L1 or L2 still has to destroy.
  - It is the only layer that works when home is gone for good.

**Who owns what**

| Thing | Owner | Nobody else may |
|---|---|---|
| Which instance is ours | `crates/cloud/src/state.rs` (`~/.local/state/vi/cloud.json`) | create, track or forget an instance |
| Talking to the Vast API with the account key | `crates/cloud` (`vast.rs`, `worker.rs`) through `vi cloud ...` | call the API from Python or bash at home |
| Whether to keep paying | L1 `validity.py` (until then, `jobs/paidrun/`) | decide to continue; a bash script may only *report* |
| Run progress, attempts, outcome, spend | `paid_run.py` run state (`<state root>/<run_id>.json`) | write it |
| Moving bytes, the tunnel, the receiver, stage execution | `jobs/on-worker.sh` | open its own tunnel or receiver |
| The worker's own deadline | L3 watchdog | — |

**The single source of truth for rental state** is `cloud.json`, reached only through `vi cloud`. `paid_run.py` caches the instance id it reads from `up` output, which saves a status lookup during release. That cache is only a hint: `down --verify` also destroys the tracked id and every labelled instance. Files on the worker (`.boot`, `.this-boot`, `.stopped-at`) belong to L3 alone, and home never trusts them to mean billing has stopped.

**Data and control flow**

1. `paid_run.py` → `vi cloud up` → create → id saved to `cloud.json` → L2 armed (planned) → boot wait → ssh.
2. `paid_run.py` → `on-worker.sh` → home opens the reverse tunnel and the restricted receiver → the worker pushes a random token → inputs go up → the stage runs.
3. The worker writes `VI-*` status lines, `on-worker.sh` forwards them, and `paid_run.py` (L1) reads them as evidence and heartbeats L2.
4. Outputs and checkpoints are pushed from the worker to home through the tunnel. Home verifies them, then acks.
5. Release: optional ssh diagnostics (≤ 25 s, only while the worker answers), then `vi cloud down --force --verify`, then Vast evidence, spend and markers, then `verify_final`.

## Invariants

These hold in every change. A reviewer rejects a diff that breaks one.

1. **Default deny.** A rental stays only on fresh positive evidence: Vast reports running, ssh answers, the return push lands, or a progress counter goes up. A changed mtime, a live process or an absent error is liveness, not progress, and never counts on its own.
2. **Every wait has a deadline and a progress signal.** That covers loops, subprocesses, ssh calls, HTTP calls, pid-file waits and retries, with no exceptions. A retry loop has a bounded try count *and* a bounded total time.
3. **Destroy first.** Nothing comes before DELETE except bounded ssh diagnostics (≤ 25 s). Markers, unverified checkpoints, summaries and spend reports change *what is reported*, never *whether* or *when* to destroy.
4. **Billing stops only when the list says so.** A DELETE that returned 2xx or 404 is not proof. The instance must be absent from the list (or marked `destroyed`) within `VERIFY_SECS` (120 s). If it is not, the exit code is 9, which alerts Brandon. Exit 9 is the only outcome allowed to block.
5. **Cap = min(wall clock, $).** The $ cap is the approved budget ÷ (price/hr + storage/hr). L1 checks it on every poll, L2 holds it as the absolute fire time, and L3 holds it as MAX. Nothing may extend it.
6. **Destroy, don't stop.** A stopped instance still bills storage. A stop is allowed only as L3's fallback, and as the D2 checkpoint hold: full runs only, only when the unpushed checkpoint is more than 1 h newer than the newest home copy, and for at most 30 min. Debug passes always destroy.
7. **Rental 2 depends on the reason class.** `host-drop` and `host-unqualified` may rent once more within the cap. `host-unqualified` must also exclude the failed host and machine id from the next search (D3). `stalled` and `our-failure` never rent again automatically, and neither does `cap`. A failed rental is never replaced silently.
8. **Rules carried from `CLAUDE.md`:**
   - Home opens the reverse tunnel and the receiver, and the worker must push a random token before inputs or training.
   - The worker never gets home's keys, agent, credentials or any long-lived identity.
   - Listeners bind to loopback only. The receiver accepts only the exact output and checkpoint directories, is write-only, never deletes, stays hidden, allows one connection and munges symlinks. Both the tunnel and the receiver are stopped afterwards.
   - Copyrighted audio and text move only between home and the worker.
   - The Vast key is never printed or logged.
   - Diagnose workers with the Vast CLI, never by reading its credential, config or state files.
   - Nothing here learns from or selects by legacy mood labels.

## Adding a stage, a wait point or a check

Every new stage, wait or check must state all five of the following, both in the brief and in the code next to it:

1. **Deadline.** A number in seconds. Set it at about 3× the worst good past run and at least 120 s. Give the source, for example "dbg2-dbg5 took 55-91 s". Put the unit in the name (see Naming).
2. **Progress counter.** A monotonic number that proves the work is advancing, such as bytes on disk, files fetched, packages installed, items tokenised, train steps or bytes acked home. Also state the stall window: how long the counter may stay still before the stage is SUSPECT. If no counter exists, say so, and the stage gets a deadline only.
3. **Recovery bound.** Name the allowed recovery actions (reconnect the tunnel, retry in place, refresh the endpoint) with a maximum try count and a maximum total time. "Retry until it works" is not allowed.
4. **Fault-injection test.** A local test that makes the failure happen and checks the outcome: SUSPECT, then recovery, then VALID, or INVALID with the right reason, followed by DESTROY.
5. **Where it registers.**
   - After WP3: one `validity.enter(stage, deadline_s, evidence_fn)` call, plus the stage's row in the stage table in `validity.py`.
   - Until then: one named constant in `jobs/paidrun/limits.py`, passed to the worker script as a `VI_*` environment variable, not repeated as a separate default in bash.
   - The table in round 32's design C.2 holds the current proposed values.

A check that can fail a rental must also say which reason class it reports, because that decides whether rental 2 is allowed.

## Language boundary

| Goes in | What | Why |
|---|---|---|
| **Rust (`crates/cloud`, `vi cloud`)** | Every Vast API call with the account key, offer search, create, boot wait, host key, `cloud.json`, markers, destroy and verify, reap, rendering the onstart script, and the L2 timer arm and cancel | One typed, tested owner for money and identity. The key never reaches Python or bash. |
| **Python (`jobs/paid_run.py`, `jobs/paidrun/`, including the planned `validity.py`)** | Orchestration: stage order, the validity state machine, the cap, reason classes, rental-2 rules, run state, reports | It holds the whole run and is easy to test with a fake clock |
| **Bash on home (`jobs/on-worker.sh`)** | The tunnel, the restricted receiver, rsync, running a stage over ssh, and forwarding `VI-*` lines | It is a thin layer over ssh and rsync |
| **Bash on the worker (`worker-onstart.sh`, `debug-pass.sh`, `push-probe.sh`, `voicetune-fresh.sh`, `worker-job.sh`, `worker-return.sh`)** | Doing the work and *emitting* evidence | The worker image only promises bash, coreutils and curl |

Rules:

- Bash scripts report evidence. They do not decide to keep paying, and they do not destroy, with one exception: the L3 watchdog's self-destroy at the hard cap.
- Python never parses Rust's TOML or state files itself. Ask `vi cloud` for the value. (`config_boot_minutes` in `jobs/paidrun/config.py` is an existing exception, flagged in the refactor assessment.)
- **The onstart limit.** Vast's onstart field holds 4048 characters. `crates/cloud/src/onstart.rs` tests keep the rendered script under 3000 bytes and its live logic under 2700. Any change to `worker-onstart.sh` must keep those tests green. Anything that will not fit goes into a script pushed over ssh after boot, not into onstart.

## Testing

- **Local fakes only.** Use the fake Vast API (`FakeApi` in `crates/cloud`), the `vi` stub in `jobs/test_paid_run.py`, fake sleeping binaries, a throttled local HTTP server, and a fake ssh that exits 255. No test may reach Vast or rent anything.
- **No GPU in CI.** Tests that need a GPU, or more space than CI has, skip themselves with a clear reason.
- **Every failure mode gets an injection test**, not just a code read. At minimum, cover the failures that apply:
  - hang (SIGSTOP the script),
  - stall (trickle bytes),
  - ssh drop,
  - Vast API errors (500, timeout, still listed after delete),
  - home crash (SIGKILL `paid_run.py`, then assert L2 fires).
- **Order assertions.** A test for release checks that DELETE happens before Vast evidence, markers and checkpoint verify.
- **Use a fake clock** for deadlines in Python. Patch limits down to about 1 s in shell tests through the `VI_*` overrides rather than sleeping for real.
- **The CI shard count.** A new Python test changes `EXPECTED_TESTS` in `.github/ci_shards.py`, and every module must be in exactly one shard. Run `python .github/ci_shards.py check`.
- **A paid observation** (for example, proving that the watchdog's DELETE works) is never a test. It needs Brandon's approval for that specific run and budget.

## Logging and diagnostic format

- **Human lines** start with the program name and a colon: `paid-run: `, `on-worker.sh: `, `debug-pass: `, `push-probe: `. Lines that need Brandon's attention add `ALERT: ` after the prefix.
- **Machine lines** that the other side reads are one of two kinds:
  - `VI-<NAME> <fields>`, one per line, space-separated, with no prose. Examples: `VI-PROGRESS <stage> <counter> <value>` and `VI-STAGE-RC`.
  - `<program>: json {...}` for structured results, like `push-probe: json`.
- **New home↔worker signals must use a machine line.** Do not add another regex over a prose sentence. `paid_run.py` already matches about 20 prose patterns, and a reworded message silently breaks them.
- **Every release logs** the reason class, the instance id, whether the destroy was verified, and the spend.
- **Diagnostics are bounded.** Use the existing `DIAG_TIMEOUT_SECONDS`, `VAST_EVIDENCE_SECONDS` and `DIAG_MAX_BYTES`, and never wait on them before DELETE except for the ≤ 25 s ssh diagnostics.
- **Never log** the Vast key, home credentials, or game text and audio content.

## Naming

- Put the unit in every limit's name: Python `<THING>_SECONDS` or `<THING>_MINUTES`, Rust `<THING>_SECS`, environment `VI_<THING>_SECONDS`.
- An environment override uses the same name as the Python constant with a `VI_` prefix wherever possible.
- Reason classes, stage names and counter names are lowercase with hyphens (`host-drop`, `base-model-fetch`, `bytes-acked`), and the same spelling is used in every language.
- Outcomes passed to `release()` name what happened, not what to do next.

## Planner checklist

For a brief or a review that touches this area:

- [ ] Does the brief name the layer or layers it changes (L1, L2 or L3) and the owner from the table above? Does it avoid adding a new place that owns rental state or decides to keep paying?
- [ ] Does every new or changed wait state its deadline (with the past-run source), progress counter, stall window, recovery bound, fault-injection test and registration point?
- [ ] Is destroy still first? Is billing judged stopped only by a verified list? Is exit 9 still the only blocking outcome?
- [ ] Is the cap still min(wall, $) and impossible to extend? Is a stop used only as the L3 fallback or the D2 full-run hold?
- [ ] Does every new failure path map to a reason class and the rental-2 rule for that class?
- [ ] Is each piece in the right language? Does `worker-onstart.sh` still pass the onstart size tests?
- [ ] Are new signals `VI-*` or `json` lines rather than prose regexes? Are new limits named with their units and not duplicated across languages?
- [ ] Are the tests local fakes with no GPU and no Vast calls? Was the `ci_shards.py` count updated?
- [ ] Are the `CLAUDE.md` worker, tunnel, copyright, data and listening-set rules intact?
- [ ] Is nothing paid run without Brandon's approval for that run and budget? Does the brief list the free steps to finish first?
- [ ] Does the done-when condition measure the literal ask (for example, a real fault-injection run, not "the code looks right")?

## Where limits live today, and the refactor assessment

Limits and rental state are currently spread across Python, bash and Rust, and some values are copied between languages. The measured list (file sizes, every place that owns a limit or rental state, and the values copied between languages) is in the round 33 planning notes, which are kept outside the repo. In short:

- Make targeted fixes, folded into WP3 and WP4.
- Round 35 split `paid_run.py` into the `jobs/paidrun/` package as a behavior-preserving move (no logic changed). `jobs/paid_run.py` is now the entry point and re-exports every old name. Line references into `paid_run.py` in older round notes (w33, w35) point at the pre-split file and are stale.
- The fixes: WP3's stage table becomes the one home for stage deadlines, and `paid_run.py` passes them to bash as `VI_*` variables. WP4's `VI-PROGRESS` lines replace mtime checks. Values copied between languages are removed as each work package touches them.
- Prose regexes are moved to machine lines only when their code is touched anyway.
