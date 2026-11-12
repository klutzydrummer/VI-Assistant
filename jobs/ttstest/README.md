# TTS listening test (rented GPU, one run)

Compares OmniVoice, MOSS-TTS-v1.5 (8B) and Qwen3-TTS Base on 13 test lines in two
voices: **T** (a Tali game clip, private) and **P** (one public LibriTTS-R control). Human
listening decides; nothing here scores quality. Nothing in this folder rents anything: a rental
needs Brandon's explicit approval of that run and cap, and the hardening PR #126 merged first.

## Files
| File | What it does |
|---|---|
| `lines.json` | the 13 lines (public text) |
| `pick_refs.py` | picks vT (label-free, seeded hash) and vP; writes `refs/` |
| `generate.py` | one process per (model, voice, seed); writes WAVs, `manifest.sha256`, `expected-count.txt`, `run-info.json`. Model `stub` = sine WAVs for dry runs. Does not push |
| `worker_run.py` | the whole job on the worker (venvs, smoke, generation, per-model push home, keep copy); run by `on-worker.sh` |
| `requirements-moss15.txt` | full pins for MOSS-TTS-v1.5's own Python 3.12 venv (official pyproject + torchcodec) |
| `moss15_check.py` | import + WAV round-trip check run in that venv before any weights are fetched |
| `smoke_moss15_local.py` | free CPU-only home proof that the venv is complete (see below) |
| `test_inject.py` | local injection test of the job, with stub models (see "Injection test") |
| `verify.py` | home check of pushed outputs; names the culprit file and exits nonzero |
| `blind.py` | 24 kHz mono, level-matched, shuffled letters; writes `sheet.csv` and the key |
| `cloud-tts-test.toml` | test-only config (offer is picked by id, never "cheapest") |
| `reap-loop.sh` | home `vi cloud reap` every 300 s; loud if it stops. Uses `VI_BIN`, else `vi` on PATH, else `target/release/vi` in this repo; refuses to start if none exists |

Tests: `python3 -I -m unittest discover -s jobs -p test_ttstest.py` (standard library only).
Clip count: 2 voices x 13 lines x 3 models = **78** (39 more with the optional seed-2 pass for vT).

## Where things live (outside the repo, never committed)
```
/home/brandon/Projects/VI-Assistant-listen/tts-test-2026-11-10/
  refs/          vT.wav vP.wav refs.json ref-sources.md
  raw/           <model>__v<T|P>__s<seed>/ or <model>/<model>__v<T|P>__s<seed>/ (any depth), exactly as generated, plus manifest etc.
  listen-blind/  v<V>/<LINE>/{A,B,C,D}.wav, sheet.csv (ratings are saved here), refs/
  listen-key/    key.json     (do not open until ratings.csv is saved)
```
Tali audio is copyrighted: it goes only between home and the worker (as training does) and no
permanent copy may remain off Brandon's machine.

## Steps
1. **Refs** (home, free). Download ONE LibriTTS-R shard (URL in `pick_refs.py`) and the LibriTTS
   `speakers.tsv` (OpenSLR 60) into a NEW empty temp dir, then:
   `nice -n 19 python3 jobs/ttstest/pick_refs.py --corpus <corpus/tali> --shard <dir>/<shard>.parquet --speakers-tsv <dir>/speakers.tsv --out <listen root>/refs --delete-shard`.
   Listen to both clips, check the LibriVox book page named in `ref-sources.md`, delete the temp dir.
2. **Dry run** (home): `generate.py --model stub ...` for T and P, then `verify.py`, then `blind.py`.
3. **On the worker** (after approval): one stage runs `worker_run.py` under `on-worker.sh`. It gets the
   refs as a stage input named `refs` (`VI_INPUT_REFS`: `vP.wav`, `vT.wav`, `refs.json`), copied from
   home straight to the worker. Example pipeline (the `refs` stage just copies the refs into its output):
   ```toml
   [stage.refs]
   command = ["sh", "-c", "cp <listen root>/refs/* \"$VI_OUT\"/"]
   [stage.tts]
   command = ["../../jobs/on-worker.sh", "python3", "jobs/ttstest/worker_run.py"]
   inputs = ["refs"]
   settings = { expect_epochs = 0, min_vram_gb = 22 }
   ```
   `--seed2` adds the seed-2 pass for vT (`TTSTEST_SEED2=1` set at home does NOT reach the worker, which only
   gets `VI_*` variables: put the flag in the command). `--models moss15 --seed2` is the **moss15-only run**:
   13 lines x (vP s1, vT s1, vT s2) = **39 clips**, expected-count 39; verify with
   `verify.py <raw> --expect moss15:P:1,moss15:T:1,moss15:T:2`.
   **Run notes**: `export VI_BIN=<path to the vi binary>` first (`on-worker.sh` and `reap-loop.sh` call bare `vi`
   otherwise); `vi cloud up --until` needs a full RFC3339 time (`2026-11-12T19:15:00Z`, not `19:15Z`); `vi cloud down`
   refuses the empty placeholder checkpoint dir of this job, so first run `python3 jobs/checkpoints.py verify --expect 0 <that dir>`
   (usage: `checkpoints.py verify --expect 0 <dir>`), then `down` works without `--force`.
   What the job does, per model and each in its own process:
   - **Environment**: Qwen3 in the venv `vi_job.ensure_venv` builds from `jobs/requirements-tune.txt`;
     OmniVoice in a venv that reuses the image's torch (`pip install omnivoice`); MOSS-TTS-v1.5 in its own
     Python 3.12 venv with torch 2.9.1 and transformers 5.0.0 (`uv` or `python3.12` if the image has one; otherwise the job `pip install`s a pinned uv into its own venv and runs
     `uv python install 3.12`; if that fails the model is dropped at once, exit 3, before any weights; no flash-attn build).
     torchaudio 2.9 reads and writes audio through **torchcodec 0.8.1, which needs FFmpeg shared libraries** (4-8): the job
     `apt-get install`s `ffmpeg` if `ldconfig` shows none, builds the venv, then runs `moss15_check.py` (every import the
     model's code uses plus a save/load/resample of a tiny WAV). A failed check drops moss15 at once, exit 3, before the
     ~24 GB download (TTS 17.0 GB + the separate `MOSS-Audio-Tokenizer` repo 7.1 GB). No Hugging Face token is used or passed. Downloads are retried 3 times with
     backoff and a model whose setup/download is still unfinished after 20 minutes is dropped.
   - **Smoke first**: line L01, voice P, seed 1, 15 minutes at most, checked with `verify.py`'s WAV
     checks. Failing twice drops the model; the reason is recorded in `run-info.json`. Other models go on.
   - **Full run**: voices P and T (seed 1; T also seed 2 if asked). Each finished folder is checked, copied
     to `/root/vi-keep` (the job-folder cleanup never touches it), then pushed home through
     `worker-return.sh` (manifest and count last). After a failed push the other pushes wait only 30 s,
     and every folder not yet home is tried once more at the end.
   - **Last**: top-level `manifest.sha256`, `expected-count.txt` (the planned 78, so a dropped model shows
     as a shortfall) and `run-info.json` (versions, GPU, seed, timings, dropped models, culprits), pushed
     last. `vT.wav` (and its transcript) are deleted from the worker when generation ends, on any error,
     on SIGTERM, and by a detached watcher even if the job is SIGKILLed. Never copy it anywhere.
   - **Exit codes**: 0 all good; 3 a model was dropped or a folder failed its check; 4 a push failed;
     2 bad setup. The culprits are printed under `CULPRITS` and the worker folder is kept (`on-worker.sh`).
   The three real adapters are untested guesses from vendor READMEs; the smoke test is what proves them.
4. **At home**: `verify.py <raw> --expect omnivoice:T:1,omnivoice:P:1,...,omnivoice:T:2,... --min-free-gb 5`.
   Fix every culprit before anything is destroyed.
5. **Worker-copy cleanup**: only after `verify.py` says `VERIFY OK` for everything, delete the worker's
   kept copy (`rm -rf /root/vi-keep` over ssh, and the job output folder), then `vi cloud down`, read the
   spend, confirm `vi cloud status` is empty and no receiver or tunnel remains. If verify fails, keep the
   copy and the worker until the files are recovered or the loss is accepted.

   **Re-push recipe** (a push failed, or home lost a folder; the worker still has `/root/vi-keep`):
   1. At home, start a receiver into the raw folder and a tunnel to the worker (loopback only):
      `state=$(mktemp -d)/s; port=$(jobs/worker-return.sh start <raw dir> $(mktemp -d) $state)`, then
      `ssh -p <worker port> root@<worker host> -R 127.0.0.1:<rport>:127.0.0.1:$port -N &`
      (use `vi cloud env` for host and port; `<rport>` is any free port on the worker).
   2. On the worker: `VI_RETURN_PORT=<rport> python3 /root/vi-jobs/<stage dir>/jobs/ttstest/worker_run.py --repush`
      (add a path to push a different kept folder). It re-sends every kept folder, manifest last.
   3. At home run `verify.py` again. Then stop the tunnel (`kill %1`) and the receiver
      (`jobs/worker-return.sh stop $state`). Rerunning the same stage also reattaches and re-proves the path.
6. **Blind**: `blind.py <raw> <listen root> --refs <listen root>/refs` (re-run it whenever a model
   folder is added under `raw/`, into a fresh root; it refuses to overwrite). Then listen in the browser:
   `vi review --listen <root>/listen-blind --key <root>/listen-key/key.json --refs <root>/refs`
   (add `--port N` / `--host ADDR` as for other review modes). The set can live in any folder. The page
   plays each A/B/C/D clip and shows the refs (vT, vP) in a "Reference clips" box, outside the rated
   groups. Answers save to `listen-blind/sheet.csv` as you go. The key is never served; the model names
   appear on the page only after every clip is answered, and `results.md` is written then. The page's
   questions are the voicearms ones (her voice y/n, mood 0-3, words clear y/n, notes): for this test
   read "mood" as a general quality score and leave the mood-only extras blank.

## Injection test (free, no GPU, no network)
`test_inject.py` runs `on-worker.sh` + `worker_run.py` with the worker in a local folder
(`VI_WORKER_HOST=local`, `VI_WORKER_ROOT=<tmp>`), home in temp folders, and sine-WAV stub models
(`TTSTEST_STUB=1`: no venvs, downloads or real audio). Run it under `nice` (about five minutes):

    nice -n 19 python3 -I jobs/ttstest/test_inject.py -v

Cases: a clean baseline (78 clips, `VERIFY OK`); (a) the job SIGKILLed halfway; (b) the return port
occupied and the receiver killed so pushes fail; (c) a WAV byte flipped after the manifest; (d)
`MemoryError` in one model (it is dropped after two smoke tries, the others finish); (e) one model's
output deleted; a re-push check, a moss15-only run (39 clips) and moss15 failing fast when Python 3.12 cannot be provisioned. Each failure case must give a nonzero exit, a kept worker folder
and keep dir, `vT.wav` gone from the worker, and a `verify.py` result naming the right folder or file.
A single case: `... test_inject.py Injection.test_d_memory_error_in_one_model`.

## moss15 local smoke (free, CPU, no weights)
    LD_LIBRARY_PATH=<dir with FFmpeg .so files> nice -n 19 ionice -c2 -n7 python3 -I jobs/ttstest/smoke_moss15_local.py

Builds the worker's venv (uv, Python 3.12, the same pins, CPU torch wheels) in a new temp dir and runs `moss15_check.py`.
It proves the pins resolve together, every import works and torchaudio/torchcodec can save, load and resample a WAV.
It cannot prove CUDA, flash-attn, the real model forward, or that the Vast image's apt gives working FFmpeg libs.
