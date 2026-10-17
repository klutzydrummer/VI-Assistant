# The cloud worker

Heavy jobs run on a GPU rented from Vast.ai. The worker keeps nothing that matters (see `design/local-cloud-split.md`); home remembers only which instance is ours, in `~/.local/state/vi/cloud.json`.

## Commands
- `vi cloud offers`: the ten cheapest machines that fit the config. Read only.
- `vi cloud up [--offer ID] [--yes]`: shows the cheapest offer, your credit and how many hours it buys, asks, then rents it and waits until ssh answers. If a machine fails to boot it's given back and the next one is tried (three at most). Run it again while a worker is up and it just reuses it.
- `vi cloud down`: destroys our worker, plus any instance labelled `vi-worker` that home lost track of. It never touches other instances on the account.
- `vi cloud status`: state, price, time up, rough spend so far, idle minutes left, credit, and any other instances on the account (a stopped instance still bills for its disk).
- `vi cloud reap`: releases the worker if it has been idle for `idle_minutes` or up for `max_hours`.
- `vi cloud ssh [command]`: a shell on the worker.
- `jobs/check-worker.sh`: checks the full round trip a GPU stage makes. It sends a random token up through `on-worker.sh` as a stage input, runs a tiny job that copies it into its output and its checkpoint, and passes only if both copies come home matching. It prints what the worker has (GPU, disk free, Python, torch). Run it after `vi cloud up` to confirm home can reach the worker, send files, run code and get results and checkpoints back. All connections are home's ssh; the worker never connects to home.
- `vi cloud env` and `vi cloud touch`: used by `jobs/on-worker.sh`.
- `python3 jobs/progress.py corpus/tali [--watch 60]`: how far a running stage has got, as a percentage per pass, read from its log at home; `--watch` adds the time left in the current pass. It reads only logs, so it works for GPU and home stages alike and changes no stage's key.

## Keeping the budget safe
Two guards, so a forgotten worker can't drain the credit:
1. **On the worker.** `jobs/worker-onstart.sh` runs at boot and starts a watchdog. It destroys the instance after `idle_minutes` with no running job and no ssh session, or `max_hours` after boot whatever it's doing. This works while the home machine is asleep. It calls the Vast API with the instance's own `CONTAINER_API_KEY`; if destroying fails it stops the instance instead, which ends the GPU charge but still bills disk until `vi cloud down` or the next `vi cloud up` cleans it up.
2. **At home.** `vi cloud reap` applies the same limits from home's own clock. Run it from a timer:

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

`min_cuda` must cover the CUDA that the jobs' torch was built for. textscore's vLLM brings torch built for CUDA 13, hence the 13.0 default; on an older driver torch sees no GPU, and textscore stops at once rather than crawl on the CPU. Jobs built for CUDA 12 run on these newer drivers too. The boot script also installs gcc, which Triton (inside vLLM) needs at run time; `check-worker.sh` reports whether it's there.

The API key comes from `$VAST_API_KEY` or `~/.config/vastai/vast_api_key` (where the `vastai` CLI saves it). Your ssh public key must be on your Vast account (Account, SSH keys); new instances get it automatically.

## Running a dataset stage on the worker
Prefix the stage's command with `jobs/on-worker.sh`:

```toml
[stage.label]
command = ["../../jobs/on-worker.sh", "--put", "states.toml", "python3", "jobs/label.py"]
inputs = ["clean"]
files = ["../../jobs/label.py", "states.toml"]
```

It rsyncs `jobs/`, the settings, each input folder and the `--put` files into a fresh folder on the worker, runs the command there with the usual `VI_*` variables pointing at the copies (`--put` files land in `files/`), rsyncs `VI_OUT` back, and deletes the folder on the worker whether the job worked or not. Game audio only ever travels between home and the worker over ssh, and doesn't outlive the job there. Destroying the instance wipes its disk too.

**Uploads.** Home's disk is the slow end: the corpus sits on spinning, encrypted disks that other services also use, and reading a stage's thousands of small audio files ran at about 0.7 MB/s on 2026-10-17, against about 2.2 MB/s of uplink. So `on-worker.sh` does three things before the upload:
- **Packs first.** Each input over `VI_PACK_MIN_KB` (default 1024) becomes one zstd'd tar, before the worker is even looked up, so the slow reading happens while nothing is billed. Bundles are kept in `VI_BUNDLE_DIR` (default `~/.cache/vi/bundles`, at most `VI_BUNDLE_CACHE_MB`, default 4000, least recently used dropped first), keyed by the folder's path and file listing, so a rerun or a second stage with the same input reuses them. Each is uploaded as one file and unpacked on the worker (or streamed and unpacked there if the worker has no zstd). zstd saves only about 13% on her WAVs; FLAC would save about 36% but needs `flac` on both ends, so it isn't used yet.
- **Can wait for the worker.** `VI_WAIT_FOR_WORKER=MINUTES` makes it wait that long for `vi cloud up` instead of failing, so you can start `vi dataset run`, let it pack, and rent when it says it's waiting. Without it, a run with no worker packs, keeps the bundles and stops; the run after `vi cloud up` starts straight at the upload.
- **Checks the uplink.** A timed 8 MB upload estimates how long the inputs will take; if that's over `VI_MAX_UPLOAD_MINUTES` (default 30) it stops before sending anything, unless `VI_UPLOAD_ANYWAY=1`. It logs `upload: <inputs done>/<inputs>`, which `jobs/progress.py` shows.

**If the worker dies part way.** Jobs save each clip's results in `VI_CHECKPOINT` as they go, and `on-worker.sh` copies that folder home every `VI_SYNC_SECONDS` (default 60) and once more when the job ends, however it ends. If the instance is lost, the watchdog or `max_hours` ends it, or the credit runs out, the stage fails with its checkpoint at home in `stages/<stage>/.checkpoint-<key>/`. Run `vi cloud up` and `vi dataset run` again: the checkpoint goes up to the new worker and the job skips clips already done. At most about a minute of work is lost. Editing the job's code or settings in between starts the stage over.

For tests, `VI_WORKER_HOST=local` with `VI_WORKER_ROOT=<folder>` runs the "worker" in a local folder; `crates/vi/tests/on_worker.rs` does this, so CI needs no GPU or rented machine.
