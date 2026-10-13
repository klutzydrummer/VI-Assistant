# The cloud worker

Heavy jobs run on a GPU rented from Vast.ai. The worker keeps nothing that matters (see `design/local-cloud-split.md`); home remembers only which instance is ours, in `~/.local/state/vi/cloud.json`.

## Commands
- `vi cloud offers`: the ten cheapest machines that fit the config. Read only.
- `vi cloud up [--offer ID] [--yes]`: shows the cheapest offer, your credit and how many hours it buys, asks, then rents it and waits until ssh answers. If a machine fails to boot it's given back and the next one is tried (three at most). Run it again while a worker is up and it just reuses it.
- `vi cloud down`: destroys our worker, plus any instance labelled `vi-worker` that home lost track of. It never touches other instances on the account.
- `vi cloud status`: state, price, time up, rough spend so far, idle minutes left, credit, and any other instances on the account (a stopped instance still bills for its disk).
- `vi cloud reap`: releases the worker if it has been idle for `idle_minutes` or up for `max_hours`.
- `vi cloud ssh [command]`: a shell on the worker.
- `vi cloud env` and `vi cloud touch`: used by `jobs/on-worker.sh`.

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
min_cuda = 12.8
min_compute_cap = 800  # Ampere or newer, for bf16 and flash attention
verified = true
```

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

### If the credit runs out mid-job
The stage's checkpoint folder (`VI_CHECKPOINT`, see `design/corpus-format.md`) travels with the job and comes back home every `VI_CHECKPOINT_EVERY` seconds (default 120) and when the job ends. The jobs save per-clip results there as they go. So if the worker is destroyed partway (credit gone, watchdog, a lost machine), the stage fails at home with its progress on disk. To resume: top up the credit, `vi cloud up`, and run `vi dataset run` again. The stage starts from the last copy that reached home, losing at most a couple of minutes of work. Model downloads and package installs aren't checkpointed; a fresh worker repeats them.

The checkpoint belongs to the stage's key, so changing a setting or a job script before resuming starts that stage over.

For tests, `VI_WORKER_HOST=local` with `VI_WORKER_ROOT=<folder>` runs the "worker" in a local folder; `crates/vi/tests/on_worker.rs` does this, so CI needs no GPU or rented machine.
