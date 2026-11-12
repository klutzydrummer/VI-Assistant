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
| `requirements-moss15.txt` | pins for MOSS-TTS-v1.5's own Python 3.12 venv |
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
   `TTSTEST_SEED2=1` adds the seed-2 pass for vT. What the job does, per model and each in its own process:
   - **Environment**: Qwen3 in the venv `vi_job.ensure_venv` builds from `jobs/requirements-tune.txt`;
     OmniVoice in a venv that reuses the image's torch (`pip install omnivoice`); MOSS-TTS-v1.5 in its own
     Python 3.12 venv with torch 2.9.1 and transformers 5.0.0 (`uv` if the image has it, else `python3.12`;
     no flash-attn build). No Hugging Face token is used or passed. Downloads are retried 3 times with
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
output deleted; and a re-push check. Each failure case must give a nonzero exit, a kept worker folder
and keep dir, `vT.wav` gone from the worker, and a `verify.py` result naming the right folder or file.
A single case: `... test_inject.py Injection.test_d_memory_error_in_one_model`.
