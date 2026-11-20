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

It runs the real `tokenise` function from `jobs/voicetune-fresh.sh` (its real `tee` pipeline and
count guard) and the real upstream `omnivoice.scripts.extract_audio_tokens`, on 8 real home clips,
on CPU. `voicetune-fresh.sh` is not edited: the script cuts the `tokenise`,
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

To rehearse a fix without editing the stage, force the caps from outside: `--threads 2 --nj 2`
exports `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=2` and adds `--nj_per_gpu 2` to the upstream call
(through a small python wrapper). That passes at `--tasks-max 512` even on the unfixed base, which
shows the cap works and the base only lacks it.

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

On a base without the fixes (`5287635` and its descendants until the limits and tokenise fixes
land), `--expect red` passes and `--expect green` fails. That is the point: green going from fail
to pass is the proof of the fix.

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
