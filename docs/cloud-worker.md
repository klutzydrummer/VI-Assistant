# The cloud worker

Heavy jobs run on a GPU rented from Vast.ai. The worker keeps nothing that matters (see `design/local-cloud-split.md`); home remembers only which instance is ours, in `~/.local/state/vi/cloud.json`.

## Commands
- `vi cloud offers`: the ten cheapest machines that fit the config. Read only.
- `vi cloud up --cap USD --until TIME [--offer ID] [--yes]`: shows the cheapest offer, its worst case next to your cap, and your credit, asks, then rents it and waits until ssh answers. `--cap` is the most the rental may cost and `--until` when that approval expires (an RFC 3339 timestamp, or `HH:MM` meaning today). It refuses before renting anything if the deadline has passed, the worst case (`max_hours` at this price) would exceed the cap, or the account holds less than the cap. It rents at most one offer: if that machine fails to boot it's given back and `up` exits nonzero rather than renting a replacement. Run it again while a worker is up and it just reuses it (the flags are still required).
- `vi cloud down`: destroys our worker, plus any instance labelled `vi-worker` that home lost track of. It never touches other instances on the account. It prints the credit before, after, and the difference, so you can see what the rental actually cost.
- `vi cloud status`: state, price, time up, rough spend so far, idle minutes left, credit, and any other instances on the account (a stopped instance still bills for its disk).
- `vi cloud reap`: releases the worker if it has been idle for `idle_minutes` or up for `max_hours`. Before an idle release it asks the worker over ssh whether an interactive session or another ssh connection is using it (`who`, plus `ss` for established inbound ssh connections), and keeps the worker when the answer is yes or when the check can't run; the `max_hours` limit releases it regardless.
- `vi cloud ssh [command]`: a shell on the worker.
- `jobs/check-worker.sh`: checks the full round trip a GPU stage makes. It sends a random token up through `on-worker.sh` as a stage input, runs a tiny job that copies it into its output and its checkpoint, and passes only if both copies come home matching. It prints what the worker has (GPU, disk free, Python, torch). Run it after `vi cloud up` to confirm home can reach the worker, send files, run code, and receive worker-pushed results and checkpoints. Home creates a reverse SSH tunnel and a restricted, temporary rsync receiver; no home credentials or agent are stored on the worker.
- `vi cloud env` and `vi cloud touch`: used by `jobs/on-worker.sh`.
- `python3 jobs/progress.py corpus/tali [--watch 60]`: how far a running stage has got, as a percentage per pass, read from its log at home; `--watch` adds the time left in the current pass. It reads only logs, so it works for GPU and home stages alike and changes no stage's key.

## Keeping the budget safe
Two guards, so a forgotten worker can't drain the credit:
1. **On the worker.** `jobs/worker-onstart.sh` runs at boot and starts a watchdog. It destroys the instance after `idle_minutes` with no running job and no ssh session, or `max_hours` after boot whatever it's doing. This works while the home machine is asleep. It calls the Vast API with the instance's own `CONTAINER_API_KEY`; if destroying fails it stops the instance instead, which ends the GPU charge but still bills disk until `vi cloud down` or the next `vi cloud up` cleans it up.
2. **At home.** `vi cloud reap` applies the same limits from home's own clock. Its idle release first checks the worker over ssh for an interactive session (`who`) or another established ssh connection (`ss`); only a worker it can see is idle is released, and `max_hours` still releases regardless. Run it from a timer:

   ```nix
   systemd.user.services.vi-cloud-reap.script = "vi cloud reap";
   systemd.user.timers.vi-cloud-reap = {
     wantedBy = [ "timers.target" ];
     timerConfig.OnCalendar = "*:0/5";
   };
   ```

The worker-side watchdog hasn't run on a real instance yet. The first `vi cloud up` should be followed by `vi cloud ssh cat /var/log/vi-watchdog.log` after the idle timeout on a short `idle_minutes`, to confirm it really releases itself.

## Config
`~/.config/vi/cloud.toml` (or `$VI_CLOUD_CONFIG`). Every field is optional; these are the defaults:

```toml
image = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"
disk_gb = 60
idle_minutes = 20
max_hours = 4          # 0 = no limit
boot_minutes = 15      # give up on a machine that hasn't answered ssh by then
label = "vi-worker"

[search]
gpu_names = []         # e.g. ["RTX 3090", "RTX 4090"]; empty = any
min_gpu_ram_gb = 24
max_dph = 0.40         # dollars per hour
min_reliability = 0.98
min_inet_down_mbps = 200
min_cuda = 13.0
min_compute_cap = 800  # Ampere or newer, for bf16 and flash attention
verified = true
countries = []         # e.g. ["US", "CA"]; empty = anywhere
```

`countries` takes two-letter codes (Vast.ai's geolocation). A nearer worker uploads a little faster: one rsync stream slows down as the round trip grows.

`min_cuda` must cover the CUDA that the jobs' torch was built for. textscore's vLLM brings torch built for CUDA 13, hence the 13.0 default; on an older driver torch sees no GPU, and textscore stops at once rather than crawl on the CPU. Jobs built for CUDA 12 run on these newer drivers too. The boot script also installs gcc, which Triton (inside vLLM) needs at run time; `check-worker.sh` reports whether it's there. For textscore's FP8 LLMs, prefer GPUs with native FP8 (Ada or newer: `"L40S"`, `"RTX 6000Ada"`, `"RTX PRO 5000"`); on Ampere cards such as the A6000, vLLM has to compile a fallback kernel first.

The API key comes from `$VAST_API_KEY` or `~/.config/vastai/vast_api_key` (where the `vastai` CLI saves it). Your ssh public key must be on your Vast account (Account, SSH keys); new instances get it automatically.

## Running a dataset stage on the worker
Prefix the stage's command with `jobs/on-worker.sh`:

```toml
[stage.label]
command = ["../../jobs/on-worker.sh", "--put", "states.toml", "python3", "jobs/label.py"]
inputs = ["clean"]
files = ["../../jobs/label.py", "states.toml"]
```

It rsyncs `jobs/`, the settings, each input folder and the `--put` files into a fresh folder on the worker, runs the command there with the usual `VI_*` variables pointing at the copies (`--put` files land in `files/`), and the worker pushes `VI_OUT` home. It deletes the folder on the worker whether the job worked or not. Game audio only ever travels between home and the worker over the direct SSH connection and its reverse tunnel, and doesn't outlive the job there. Destroying the instance wipes its disk too.

**Worker-initiated returns.** Before any inputs or paid job run, `on-worker.sh` starts `jobs/worker-return.sh` on home, which launches a temporary rsync daemon bound to `127.0.0.1`. Home then opens a second SSH connection with `ExitOnForwardFailure=yes` and a reverse forward whose worker listener is also bound to `127.0.0.1`. The worker must push a random token through that tunnel and home must read the exact token back; otherwise the stage stops before training. The worker pushes checkpoints every `VI_SYNC_SECONDS` and final output at the end. Home does not pull them.

The receiver has separate `vi-out`, `vi-checkpoint`, and `vi-publish` modules rooted at their exact destination folders; `vi-publish` is the durable per-attempt checkpoint dir where finished epochs land to be verified. All three are write-only, refuse deletion, allow only localhost, are hidden from module listings, and limit connections to one. `use chroot = false` keeps the daemon unprivileged; `munge symlinks = true` prevents an uploaded symlink from escaping the destination. No `auth users` or secrets file is used: the loopback-only SSH tunnel provides encryption and access control, and both receiver and tunnel close when the stage ends. No home SSH login, private key, or forwarded agent is available to the worker. `VI_RETURN_PORT` can override the automatically selected loopback port. Jobs can push additional files with `VI_RETURN_PORT=... jobs/worker-return.sh push SOURCE vi-out` (or `vi-checkpoint`, `vi-publish`).

`VI_WORKER_HOST=local` uses the same worker-push path without SSH, which lets CI exercise the receiver, output/checkpoint transfer, and failure handling. `jobs/test_worker_return.py` also starts a disposable OpenSSH server to test a real reverse tunnel.

**Uploads.** Home's disk is the slow end: the corpus sits on spinning, encrypted disks that other services also use, and reading a stage's thousands of small audio files ran at about 0.7 MB/s on 2026-10-17, against about 2.2 MB/s of uplink. So `on-worker.sh` does three things before the upload:
- **Packs first.** Each input over `VI_PACK_MIN_KB` (default 1024) becomes one zstd'd tar, before the worker is even looked up, so the slow reading happens while nothing is billed. Bundles are kept in `VI_BUNDLE_DIR` (default `~/.cache/vi/bundles`, at most `VI_BUNDLE_CACHE_MB`, default 4000, least recently used dropped first), keyed by the folder's path and file listing, so a rerun or a second stage with the same input reuses them. Each is uploaded as one file and unpacked on the worker (or streamed and unpacked there if the worker has no zstd). zstd saves only about 13% on her WAVs; FLAC would save about 36% but needs `flac` on both ends, so it isn't used yet.
- **Can wait for the worker.** `VI_WAIT_FOR_WORKER=MINUTES` makes it wait that long for `vi cloud up` instead of failing, so you can start `vi dataset run`, let it pack, and rent when it says it's waiting. Without it, a run with no worker packs, keeps the bundles and stops; the run after `vi cloud up` starts straight at the upload.
- **Checks the uplink.** A timed 8 MB upload estimates how long the inputs will take; if that's over `VI_MAX_UPLOAD_MINUTES` (default 30) it stops before sending anything, unless `VI_UPLOAD_ANYWAY=1`. It logs `upload: <inputs done>/<inputs>`, which `jobs/progress.py` shows.

**If the worker dies part way.** Jobs save each clip's results in `VI_CHECKPOINT` as they go, and the worker pushes that folder home every `VI_SYNC_SECONDS` (default 60) and once more when the job ends, however it ends. If the instance is lost, the watchdog or `max_hours` ends it, or the credit runs out, the stage fails with its checkpoint at home in `stages/<stage>/.checkpoint-<key>/`. Run `vi cloud up` and `vi dataset run` again: the checkpoint goes up to the new worker and the job skips clips already done. At most about a minute of work is lost. Editing the job's code or settings in between starts the stage over.

For tests, `VI_WORKER_HOST=local` with `VI_WORKER_ROOT=<folder>` runs the "worker" in a local folder; `crates/vi/tests/on_worker.rs` does this, so CI needs no GPU or rented machine.

## One command for the paid OmniVoice fine-tune
`jobs/paid_run.py` runs the whole paid flow unattended, driving the `vi` binary (`$VI_BIN`; if unset, `<repo>/target/release/vi` when that executable exists; otherwise `vi` on PATH). It rents the cheapest offer that has not failed yet, runs the `omnivoice_preflight` stage (tunnel, receiver, token push, worker gates), then `omnivoice_train`, checks what came home, and releases the worker. Start it only with Brandon's approval of the cap and hours for that run.

```sh
python3 -I jobs/paid_run.py --run-id omnivoice-ft-1 --cap 5.00 --max-hours 8 \
    --config jobs/paid-run/omnivoice-run-cloud.toml [--countries "US,DE" | --countries eu] [--max-attempts 3]
```

- `--cap` and `--max-hours` cover all attempts together. Each attempt gets the remaining cap on `vi cloud up --cap` and a config copy whose `max_hours` is the hours left.
- Only a push-speed gate failure (stderr contains `worker push (MB/s)`) is retried: the worker is released, that offer and any relisting of the same machine (same GPU, place, cores, RAM and uplink; `vi cloud offers` does not print machine ids) is excluded, and the next cheapest is rented, up to `--max-attempts`. Any other failure releases the worker and stops.
- `--plan` prints the effective plan and exits without calling `vi`.
- State is kept in `~/.local/state/vi/paid-run/<run-id>.json` (`$VI_PAID_RUN_STATE_DIR` overrides). A killed run is started again with the same `--run-id` and the same limits: a worker still up whose train stage had started is adopted (the stage reattaches); any other leftover worker is destroyed first, so a second rental is never made on top of a billing one.
- SIGINT and SIGTERM stop the child stage and release the worker (`down`, then `down --force` if it refuses).
- Exit codes: 0 done, 2 bad arguments, 3 `up` failed, 4 other preflight failure, 5 push gate failed on all attempts, 6 train failed, 7 home results did not verify, 8 interrupted, 9 worker could not be released (check `vi cloud status`), 10 no offer or budget left, 1 unexpected.
- After training it checks that home holds checkpoints 1000, 2500, 4500 and 5000 with `VERIFIED` and no unverified marker, and that `step-1000`, `step-2500` and `step-5000` listening sets (with `listening.jsonl` and `README.txt`) are in the stage's `platter` (`--platter`, default `corpus/tali/stages/omnivoice_train/current/out/platter`). Human listening decides voice quality. For each set it also writes `step-N/clips.toml` (one clip per `listening.jsonl` row, audio at `step-N/wav/<label>.wav`, unblinded, answers her / partly / not her). The review hub (`vi review corpus/tali`, port 8737) already finds any `clips.toml` under a stage's current output, so the three sets then show up there as clip lists; a single model has nothing to blind against, so there is no A/B. If a row has no wav at home, the run exits 7. This needs the default `--platter` (or any folder under that stage's current output); the hub display itself has not been checked on the real service.
- Tests: `jobs/test_paid_run.py` (stub `vi`, no Vast, no network).

### Status and stop

Agents can read a run's progress and ask it to stop safely. Neither command can change the cap, the hours limit, or the attempt count, and neither rents or retries anything.

```sh
python3 -I jobs/paid_run.py status --run-id omnivoice-ft-1 [--json] [--credit]
python3 -I jobs/paid_run.py stop --run-id omnivoice-ft-1 [--now]
```

- `status` reads the run's state file only and makes no `vi` call. `--credit` adds one `vi cloud status` call to show the account credit. Spend for the open attempt and the ETA are labelled estimates, and the ETA appears only after 3 step samples. An unknown run id exits 2 and changes nothing.
- The training step comes from the trainer output that `paid_run.py` already streams. It reads a tqdm bar (`150/5000 [02:10<...`) or a text line such as `Step 150/5000` or `step: 150 of 5000`. **Assumption:** the real trainer's line format has not been checked against a live run. If neither shape matches, `status` says "no step read yet" and `stop` releases the worker at once, so check `status` early in the first real run.
- `stop` writes `<run-id>.stop` beside the state file. It refuses (exit 2) if the run is not running. Before training starts (offer, up, preflight) it releases the worker immediately. During training it waits for the newest checkpoint (saved every 500 steps) to reach home and be acknowledged, for at most 10 minutes, then tears the worker down and exits 8. It also releases at once if no checkpoint is pending or the step is unreadable.
- `stop --now` skips the wait. It sends SIGTERM to the recorded `paid_run.py` pid, but only after checking that the process is this run's (command line holds `paid_run.py` and the run id, and the start time matches). Otherwise it signals nothing.
- A stop file left from an older run is deleted when a run starts, so it never stops a new run with the same id.
