# TTS listening test (rented GPU, one run)

Compares OmniVoice, MOSS-TTS-v1.5 (8B), Qwen3-TTS Base and MOSS-TTS-Nano on 13 test lines in two
voices: **T** (a Tali game clip, private) and **P** (one public LibriTTS-R control). Human
listening decides; nothing here scores quality. Nothing in this folder rents anything: a rental
needs Brandon's explicit approval of that run and cap, and the hardening PR #126 merged first.

## Files
| File | What it does |
|---|---|
| `lines.json` | the 13 lines (public text) |
| `pick_refs.py` | picks vT (label-free, seeded hash) and vP; writes `refs/` |
| `generate.py` | one process per (model, voice, seed); writes WAVs, `manifest.sha256`, `expected-count.txt`, `run-info.json`. Model `stub` = sine WAVs for dry runs. Does not push |
| `verify.py` | home check of pushed outputs; names the culprit file and exits nonzero |
| `blind.py` | 24 kHz mono, level-matched, shuffled letters; writes `sheet.csv` and the key |
| `cloud-tts-test.toml` | test-only config (offer is picked by id, never "cheapest") |
| `reap-loop.sh` | home `vi cloud reap` every 300 s; loud if it stops |

Tests: `python3 -m unittest discover -s jobs -p test_ttstest.py` (standard library only).

## Where things live (outside the repo, never committed)
```
/home/brandon/Projects/VI-Assistant-listen/tts-test-2026-11-10/
  refs/          vT.wav vP.wav refs.json ref-sources.md
  raw/           <model>__v<T|P>__s<seed>/   exactly as generated, plus manifest etc.
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
3. **On the worker** (after approval): per model, in its own venv, smoke test first with
   `--only L01`, then the full run: `generate.py --model M --voice V --seed 1 --out /root/job-out --refs <refs>`
   (voice T also with `--seed 2`). The four real adapters are untested guesses from vendor READMEs.
   Pushing each finished folder home (`worker-return.sh`) and keeping a copy in `/root/vi-keep/` is phase B.
4. **At home**: `verify.py <raw> --expect omnivoice:T:1,omnivoice:P:1,...,omnivoice:T:2,... --min-free-gb 5`.
   Fix every culprit before anything is destroyed.
5. **Worker-copy cleanup**: only after `verify.py` says `VERIFY OK` for everything, delete the worker's
   kept copy (`rm -rf /root/vi-keep` over ssh, and the job output folder), then `vi cloud down`, read the
   spend, confirm `vi cloud status` is empty and no receiver or tunnel remains. If verify fails, keep the
   copy and the worker until the files are recovered or the loss is accepted.
6. **Blind**: `blind.py <raw> <listen root> --refs <listen root>/refs` (re-run it whenever a model
   folder is added under `raw/`, into a fresh root; it refuses to overwrite). Then listen in the browser:
   `vi review --listen <root>/listen-blind --key <root>/listen-key/key.json --refs <root>/refs`
   (add `--port N` / `--host ADDR` as for other review modes). The set can live in any folder. The page
   plays each A/B/C/D clip and shows the refs (vT, vP) in a "Reference clips" box, outside the rated
   groups. Answers save to `listen-blind/sheet.csv` as you go. The key is never served; the model names
   appear on the page only after every clip is answered, and `results.md` is written then. The page's
   questions are the voicearms ones (her voice y/n, mood 0-3, words clear y/n, notes): for this test
   read "mood" as a general quality score and leave the mood-only extras blank.
