# Rehearsals

Two local rehearsals live here. Neither rents anything or touches Vast.

- `run.sh` is the stub rehearsal of the whole worker flow (no GPU, no network, synthetic data). See
  its header for the scenarios.
- `vast-limits.sh` is the Vast-conditions gate described below. It is opt-in and is not in
  `run.sh`'s default scenario list or in GitHub CI, because it needs real clips and the 2 GB home
  CPU venv.

## Vast-limits rehearsal (`vast-limits.sh`)

### Purpose

The r7 preflight died on a rented box that showed 48 CPUs but allowed only a few thousand tasks
(processes plus threads). Each Python process started a 48-thread OpenBLAS pool; a thread could not
be created; OpenBLAS raised SIGINT; the tokeniser's process pool broke; and the stage then hung
for good behind `| tee`. The stub rehearsal never loads real numpy or scipy, and the home
real-path check runs on 4 CPUs with a huge task limit, so neither could see it. This gate
reproduces the failure at home, for free, before any paid run that touches the tokenise or train
path.

### How to run (one command)

```bash
jobs/rehearsal/vast-limits.sh --expect red     # on a base WITHOUT the fixes: must reproduce r7
jobs/rehearsal/vast-limits.sh --expect green   # on a base WITH the fixes: must run clean
```

It prints one summary line (limits, timings, pids numbers, verdict) and exits 0 only when the
result matches what you expected. `--help` lists every option; each also has an environment
variable (`FAKE_NPROC`, `TASKS_MAX`, `CPU_QUOTA`, `N_CLIPS`, `BUDGET`, `VENV`, `MANIFEST`,
`THREADS`, `NJ`).

It runs the real `tokenise` function from `jobs/voicetune-fresh.sh` (its real watched launcher,
`vi_run_watched`, and its count guard) and the real upstream
`omnivoice.scripts.extract_audio_tokens`, on 8 real home clips, on CPU. Before that it sources
`jobs/worker-limits.sh` and runs `worker_limits`, exactly as the worker does, so the real limits
rule picks the thread and worker counts. `voicetune-fresh.sh` is not edited: the script cuts the `tokenise`,
`tokenise_count_guard` and `audio_tokenizer_dir` functions out of the file by name, so a later fix
to them is rehearsed automatically. If a fix moves logic into another function that `tokenise`
calls, name it in `EXTRA_FUNCS`.

### The limits and what they mean

| Setting | Default | Meaning |
|---|---|---|
| `FAKE_NPROC` | 48 | CPUs the process sees (an LD_PRELOAD shim, `fakenproc.c`, makes `sched_getaffinity`, `sysconf` and `get_nprocs` say 48 on this 4-core box). OpenBLAS sizes one thread pool per process from it. |
| `TASKS_MAX` | 2800 for red, 512 for green | The scope's cgroup `pids.max`. Threads count as tasks. The uncapped run peaks near 2840 tasks, so anything below that fails; 2750 to 2800 is the narrow band where the exact r7 hang (Manager "can't start new thread", `BrokenProcessPool`, stage never exits) appears. 512 is a small sandbox, which a fixed stage (2 BLAS threads, 2 workers) passes at about 160 tasks. |
| `CPU_QUOTA` | 1152% | r7's 11.52 cores. It changes nothing on 4 CPUs but keeps the limits line like the r7 box. |
| `N_CLIPS` | 8 | The first N rows of the manifest, in file order. No labels are read. |
| `BUDGET` | 300 s | The stage is killed after this long. A red run that hangs ends here (exit 124). |

Red passes only when the r7 failure shows: `pthread_create failed ... of 48` together with
`KeyboardInterrupt` or `BrokenProcessPool`, or the stage hangs until `BUDGET` kills it. Green
passes only when all of these hold: exit 0, exactly one `Processing Complete` line with
`Successful: 8`, cgroup `pids.events` `max` = 0, no `pthread_create failed`, `BrokenProcessPool`,
`can't start new thread` or `Traceback` line, and no process left in the scope after the stage
returns. Both verdicts also require that nothing survives teardown.

To try other caps, force them from outside: `--threads N` exports
`OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=N` over what the limits step chose, and `--nj N` sets
`VI_LIMIT_NJ=N`, which `tokenise` passes as `--nj_per_gpu`. A forced run is only an experiment: it
never writes a pass record.

### Pass record and the paid-run gate (exit 15)

A run writes a pass record only when all of these hold:

- it was `--expect green` and the verdict was PASS;
- it used the default limits: no `--threads`, `--nj` or `--tasks-max`, and no `THREADS`, `NJ` or
  `TASKS_MAX` in the environment (the fixed code has to pass with the limits it picks itself);
- the tracked files match HEAD (`git status --porcelain --untracked-files=no` is empty; untracked
  files do not matter).

The record is `~/.local/state/vi/rehearsal/<HEAD sha>.pass` (folder mode 0700, file 0600, written
to a temporary file and renamed). It is JSON with the head and tree shas, `tasks_max`,
`fake_nproc`, `cpu_quota`, `clips`, the threads and worker count used, `stage_secs`, `pids_peak`,
`events_max`, `base_red` (the newest red-run log kept in `vast-limits-logs`, with its time, or
`"none"`), the time and the script version. Any other run prints `no pass record written:` and the
reason.

`jobs/paid_run.py` checks the record for the HEAD of its `--repo` before it makes any `vi` or
Vast call. The record must exist, be valid JSON, name that HEAD both in its file name and inside,
match `git rev-parse HEAD^{tree}`, and be a green PASS. Otherwise a real run stops with exit 15
(nothing rented) and says whether the record is missing, stale (made for another commit or tree)
or invalid, and which command to run. `--plan` shows the same `rehearsal` line and does not fail
on it. There is no switch to skip the gate. `VI_REHEARSAL_STATE_DIR` moves the record folder for
the tests only, in both scripts.

So the order before a paid run is: commit, run `jobs/rehearsal/vast-limits.sh --expect green`,
then start `paid_run.py`. Any new commit needs a new green run (about a minute).

### Safety

CPU only (`CUDA_VISIBLE_DEVICES=""`), `nice -n 19`, `ionice -c2 -n7`. It refuses to start under
4 GB of free RAM or under 20 GB free on `/home`, and when `systemd-run --user --scope` is not
reachable (it sets `XDG_RUNTIME_DIR=/run/user/$UID` if unset). Clips are read in place. Derived
token shards go to a scratch folder under `~/.cache/vi-rehearsal` that is deleted at the end. The
log kept in `~/.cache/vi-rehearsal/vast-limits-logs/` has lines that carry clip text removed.
Nothing leaves this machine. Only the scope it started is stopped (by unit name and process
group, never by process name). The shim is built once into `~/.cache/vi-rehearsal` (with `cc`,
`gcc`, or `nix shell nixpkgs#gcc`) and never goes to a worker or into the repo.

### Runtime

One run takes about 1 to 5 minutes: a clean 8-clip run is roughly 45 to 75 s, a red run that
fails fast takes seconds to a minute, and a red run that hangs ends at `BUDGET` (300 s). Expect
a red and a green pair to fit in about 5 to 8 minutes.

### Expected result today

On a base without the fixes (`5287635`, and `74f37a3` where this script was added), `--expect
red` passes and `--expect green` fails. With the limits fixes (`jobs/worker-limits.sh`, the watched
tokenise launcher), `--expect green` passes with the default limits and writes the pass record,
and `--expect red` fails its verdict by design: at `--tasks-max 2800` the capped stage no longer
reproduces r7. Green going from fail to pass is the proof of the fix.

### What it does not cover

- CUDA: no GPU init threads, NCCL or cuDNN threads. The run is CPU only.
- The worker's Python build: home uses a uv CPython 3.11 venv with the same pinned wheels, the
  worker uses conda Python 3.11 in `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`.
- Torch's own thread pool: under the shim torch still sees 4 CPUs, so home slightly undercounts
  threads compared with a real 48-CPU view.
- The real Vast `pids.max`: it is unknown. r7 only proves it was below roughly 2.8k tasks at that
  scale. The exact hang band is timing-dependent and moves with clip count and machine speed.
- Full scale (1961 clips): the peak and the failure band grow with the clip count.
- Train-time DataLoader workers (`train_ft.py`) are not exercised.

### Future, optional: Docker worker-image mode

A `--docker` mode that runs the same stage in the real worker image with `--pids-limit` would
close the Python and torch-binary gaps. It still needs the shim, and the image is several GB, so
it is not built yet.
