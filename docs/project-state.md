# VI Assistant: project state and handoff

Last updated 2026-10-31. This page is for an agent or person picking the project up cold. It covers the goal, the rules Brandon (the owner, GitHub `klutzydrummer`) has set, where everything lives, what is done, and what comes next. Live state moves quickly: when this page and GitHub disagree, trust GitHub and ask Brandon.

## Where things stand right now

- **Nothing is running.** No rental is up, and no paid job is waiting.
- **Latest result:** "run B v2" of the voice fine-tune finished on 2026-10-31. It ran 20 more epochs from run B's epoch-9 weights, with the fixed style encoder (PRs #98 and #99). Held-out loss improved to 4.55 (run B was 4.78, run A 4.84) and flattened out by the end. The style code now varies between clips, so the collapse is fixed. It cost $1.69, the rental was destroyed, and the checkpoints (best is epoch 19) are on Brandon's PC.
- **Waiting on Brandon's ears:** a listening "platter" on the review hub, titled "Fine-tune continued (fixed style code): epochs 4, 9, 19". It has 18 clips, with no reference clip and no RVC. The question is whether the three moods now sound different from each other. His verdict decides the next step for the voice.
- **Open PR:** draft #100 speeds up style-encoder runs. It awaits his review.
- **Working voice today:** the "interim voice", which is the Qwen3-TTS Base model cloning from one of her real clips, then passed through the Collige01 fan RVC model. Brandon rated it "acceptable". The fine-tune clips from runs A and B were all rated "not yet".
- **Money:** Vast.ai credit is $8.66. The Claude API labelling budget is $22.24 of $25 spent, and that run is finished.

## The goal

A conversational assistant that feels like a friend and an executive assistant while embodying a character, with that character's persona, lore and voice. The pilot is Tali'Zorah from Mass Effect. A second character (Carl from Aqua Teen Hunger Force) comes only after Tali works end to end.

- **First milestone:** Tali replies in character, in her cloned voice, to typed text (`vi chat`). Not reached yet.
- **Later:** speech input with emotion and speaker diarization, and Tali sending out tool agents (calendar, email).
- **Voice end state, Brandon's ranking (2026-10-29):**
  1. **Target:** a fine-tuned TTS that needs no RVC and no reference clips, where the app's mood state drives the mood directly.
  2. No reference clips, with RVC at the end for a consistent voice.
  3. **Worst case:** a library of her reference clips plus RVC. This is roughly the interim voice.
- **Cheap character swaps:** zero-shot voice cloning plus a persona pack. Fine-tuning is an optional upgrade per character.

## Core design (Brandon's decisions; don't change without his explicit yes)

- **Moods come from the character's own data.** Her clips' audio embeddings are clustered to find states nobody asked for. There is also a target list of states that matter for friend/assistant chat (`docs/design/must-have-moods.md`). The result is one flat multi-label mood list (77 families; a line can count toward several), plus separate knobs for energy (calm to urgent) and intoxication (sober to drunk). Thin moods may be expanded with synthetic data or made as blends.
- **Control vectors steer both the chat model and the voice,** and they are mixed to get blends. Playing stored clips is only a fallback for engines that can't be steered.
- **The chat model writes plain speech only.** Mood travels separately, in a continuous, predictable state machine (`characters/tali/states.toml`).
- **Label-less voice route (2026-10-30, his ask):** a style encoder learns each clip's state from the audio during the voice fine-tune. Those states get clustered and named by Brandon's ears, and a small text model predicts the state from the chat. Claude's labels are now used only as names and directions. His own 150-line hand-labelled "gold" set must never touch the model (not in training, scoring, prompts or selection).
- **Full fine-tune of the voice model, not a LoRA** (decided 2026-10-30). One model covers all moods, trained on her real clips to keep her voice, accent and suit effect. A KL term on public LibriTTS-R replay lines guards against forgetting. There is no separate strength knob, because the mood state vector, through a projector, controls strength and mixing.
- **No mood is dropped** on the strength of pre-fine-tune results. Every mood stays until the fine-tuned model has been tested.
- **The suit effect** (her helmet filter) is removed only for analysis and clustering. The cloned voice keeps it.
- **Models:** chat is Qwen3.8-27B Q8_0 with thinking on, and it must stay easy to swap. Voice is Qwen3-TTS 1.7B Base. Chat vectors come from our own builder (`vi vectors`, see `docs/vectors.md`).
- **RVC:** Collige01 (a fan-made RVC model) is the default and the quality bar. It beat AI-Wheelz in Brandon's blind A/B. It is unlicensed, so it can't ship. Our own RVC model is planned.

## Ground rules

- **Self-hosted by default:** rented Vast.ai GPUs now, Brandon's own hardware later. The only hosted model API allowed is Claude, and only for the text-only labelling run.
- **Game assets are copyrighted.** Game audio and text never go to third-party hosts (Hugging Face included) or into this public repo. Audio moves only by direct rsync from Brandon's PC to a rented GPU. The corpus lives in `corpus/` on his PC and is gitignored.
- **No paid run without Brandon's typed go for that specific run.** That includes any new rental or GPU job. One exception: if an already-approved paid job has an obvious bug that would waste money, fix it first and tell him after. If his approval is unclear, pause the instance (don't destroy it) until he answers.
- **Plan ahead, don't chase errors.** Before any rental, verify the whole environment end to end: driver and CUDA version, compilers, libraries, every model's dependencies, and a tiny smoke test of each model. Fix everything in one change. Never send him one fix-and-rerun after another.
- **Never touch Vast instances this project didn't rent.** Brandon runs his own instances on the same account.
- **Carry his concerns through.** When he raises a concern, apply it to every part of the plan (scoring, prompts, examples, metrics) and to every later version, and say where it was applied. Don't make him repeat himself.
- **Any run that makes voice output also makes a small listening set** for his ears. He prefers small, unblinded, clearly labelled "sample platters" to big blind rating sets.
- **Brandon's PC limits:** heavy jobs run at `nice -n 19`. Use `ionice -c3` only for reads of the game disk; anything on the system disk uses `ionice -c2 -n7`. Nothing may overfill the RX 480's 8 GB of VRAM or hold several big models in RAM, since that has crashed the whole machine.
- **CI** (`.github/workflows/ci.yml`: fmt, clippy `-D warnings`, cargo test, plus the Python job tests) runs without a GPU. GPU jobs stay out of CI.
- **When he sequences work** ("do X first, then Y"), don't start Y until X has landed, even for prep.
- **Communication:** any line he should copy and paste goes in a code block. His phone keyboard autocorrects badly, so read typos charitably and confirm before acting on an odd-looking "go" or a bare "Yes".

## Repo layout

- `crates/`: the Rust core.
  - `vi`: the CLI (`vi chat`, `vi pack check`, `vi cloud`, `vi review`, `vi vectors`, …).
  - `engine`: the turn pipeline and mood state machine.
  - `llm`: in-process llama.cpp with control vectors.
  - `charpack`: character pack loading and checks.
  - `cloud`: Vast.ai rentals.
  - `review`: the review hub web app.
  - `harvest`: extracting lines and audio from the games.
  - `dataset`: the corpus format.
- `jobs/`: Python jobs that run on the rented GPU or at home, each with a `test_*.py`. Key ones:
  - `voicetune.py`: the full fine-tune of Qwen3-TTS with the style encoder, KL replay, `--init-from` and `--warmup`.
  - `voicearms.py`, `voicesteer.py` and `voicelayers.py`: voice steering experiments.
  - `refscreen.py`: ranks her clips as the voice-clone reference per mood.
  - `codectrip.py` and `rvctest.py`: suit-effect checks.
  - `steer.py`: chat-model steering checks.
  - `prep.py`, `clean.py`, `embed.py`, `confounds.py`, `cluster.py` and `classes.py`: the corpus pipeline.
  - `on-worker.sh`, `worker-onstart.sh` and `check-worker.sh`: the rental worker scripts.
- `characters/tali/`: the character pack. It holds `character.toml`, `persona.md`, `examples.md`, `states.toml`, `states/` and `drones.toml`. Voice clips are extracted locally and never committed.
- `models.toml`: which models each stage uses.
- `docs/`: how-tos (`cloud-worker.md`, `extracting-game-audio.md`, `vectors.md`, `tali-naming.md`) and `docs/design/` (plans and research, including `qwen3tts-finetune-guide.md`, `labelling-run-plan.md`, `must-have-moods.md`, `voice-consistency-check.md`, `after-labels-plan.md`, `steering-builder-*.md` and `review-page.md`).

## How the work runs

- **Brandon's PC** (NixOS; 4 cores; RX 480 8 GB; repo at `~/Projects/VI-Assistant`) holds the corpus, all results and checkpoints, and runs the review hub. All app state persists there, so a fresh cloud instance can resume at once.
- **Building at home** needs a nix-shell with libclang and the Vulkan packages. Always pass `--features vulkan`, otherwise the build silently replaces the Vulkan binary with a CPU-only one. The exact command is in `README.md`. Only llama.cpp uses the GPU at home; the Python models (Qwen3-TTS, emotion2vec, Whisper) run on the CPU or the rental.
- **Rentals:** `vi cloud offers`, `vi cloud up` and `vi cloud down` (see `docs/cloud-worker.md`). Audio is rsynced from his PC to the worker, and results are rsynced back. The worker releases itself when idle. Recent fine-tunes used RTX 6000 Ada class cards at about $0.64/h.
- **Review hub:** `vi review corpus/tali`, served at https://vi-review.home.internal on his LAN. Jobs add listening tasks with `listen/` sets, `clips.toml`, `task.toml` or qtcheck folders. His ratings come back through the hub.
- **Agents on his PC:** earlier work ran Claude Code on his PC through Remote Control (Claude Desktop 2.1.302 or newer). Those sessions park when idle and only wake from his own message.

## Pipeline status

| Stage | State |
|---|---|
| Harvest Tali from the games | Done. 3,516 lines, 202 minutes, with scene tags and conversation context. Metadata is hard data from the game, never LLM-inferred. |
| Corpus pipeline (clean, embed, cluster) | Done. The clusters found no mood the labels missed. |
| Claude labelling run | Done. 8,326 trusted positives over 77 families. 11 moods are thin. |
| Radio and other effect clips | Done. 71 effect clips are kept off the voice side. |
| Chat steering vectors (27B) | Built, but the first check was flat (12 of 80 steer), most likely from too little strength. The fix (strength sweep above 1.0, layer sweep, pairwise judge, random control) is merged in PR #74. **The sweep has not run yet.** |
| Voice steering on the base model (voicearms 1–3) | Done. Her voice survives only with her real clip as the prompt (in-context cloning). Indignation steers clearly; alarm and urgency are weaker; sadness, playful and blends don't steer. These results are the base-model baseline. |
| Reference screening (refscreen) | Done. Every mood found clips of hers that carry it. RVC versions through Collige01 were made too. |
| Suit checks | Done. The codec keeps the suit on LE2 and LE3 clips but degrades LE1. RVC puts her voice and suit back on 21 of 23 TTS clips. |
| Voice fine-tune | Runs A and B done (rated "not yet"; the style encoder had collapsed). Run B v2 done with the fix; **its platter awaits Brandon's ears.** |
| Vulkan on the RX 480 | Works for `vi chat` and for Qwen3-TTS trials at Q8. Final voice verdicts come from the rental. |
| `vi chat` milestone | Not yet. It needs the steering and voice results wired into the persona pack. |

## The Tali voice plan (current plan)

The full plan lives in a private Claude page, "Tali voice plan". Here is a summary.

**Aim:** one fine-tuned TTS in her voice, with her suit and accent, driven by the mood state, with no reference clips and no RVC at run time.

1. **Reference screening:** done (2026-10-30).
2. **Interim voice:** the best example clip per mood plus Collige01 RVC on the worker. This is in use, and GPU RVC is fast enough for live replies: 0.2–0.4 s per sentence when warm, after about 35 s of warm-up once.
3. **Our own RVC model,** trained on her LE2 and LE3 clips, then used to convert the LE1 clips to the same suit. It must match or beat Collige01 in a blind A/B, for under $1. Not started. The fine-tune was moved ahead of it.
4. **Training lines:** every candidate clip plus steered outputs at known settings, passed through our RVC, with mood-fitting text and a listening sample. There is no automatic score filter, because the probe scores didn't predict his ratings. The duds (about 1 in 7) were quiet or word-carried moods on neutral sentences, and mood-fitting text should fix them.
5. **Full fine-tune with no reference clip** (`jobs/voicetune.py`), with the KL guard. A projector from the app's mood state is trained on top as the only mood input. Brandon's yes is needed before the projector replaces the TTS control vectors. Estimated $5–10. **This is in progress**; see run B v2 above.
6. **Judge blind** between end-state tiers 1 and 2 and the baseline, at home for free.
7. **Into the app** as a voice adapter.

- **Training data:** 3,445 effect-free clips, 3.3 hours. That is 726 from LE1, 1,145 from LE2 and 1,574 from LE3, and 1,653 of them are 3–15 s long.
- **Projector input states:** the Claude pass-2 mood scores scaled 0–1, intoxication 0–3, and energy measured from the audio, with neutral as all zeros.
- **Chat-model track:** run the 27B strength sweep, then a linearity test. A learned vector map comes only if the blends prove non-linear.
- **Fallbacks:** if the fine-tune route fails, the options include an IndexTTS-2 bake-off, which needs his yes.

## Voice fine-tune history

- **Smoke test** ($0.34): all three test runs were comparable, and he heard "the beginnings of Tali" in each. Peak memory was 32.5 GB with KL on.
- **Runs A and B** ($1.30, PR #97): these had style-encoder guards (big-chunk shuffle, style dropout, a light delivery helper) and `--style-lr 1e-4`. He rated every clip "not yet" without a reference clip. The cadence was getting close, but the mood deliveries sounded robotic.
- **Cause:** the style encoder collapsed. Every clip got the same code, pinned at its limits, and the plain-voice inference was missing a trained offset. PR #99 fixes both by normalising codes across the batch, making the plain voice code 0, and logging `code_std` to catch a collapse early. PR #98 adds `--init-from` with LR warmup.
- **Run B v2** ($1.69): continued from B's epoch 9 for 20 epochs with a fresh style encoder. Held-out loss reached 4.55, and the style code stayed alive.
- **Inference:** `--speaker style` with `speak --style-from <clip>` uses a clip's style code. The style code is 16 numbers in the speaker slot.

## Next steps and decisions waiting on Brandon

1. **His verdict on the run B v2 platter.**
   - If the moods differ, the next steps are: cluster and name the style codes by ear, then train the text-to-style predictor and the mood-state projector (his yes is needed on the projector replacing the TTS vectors), then stage 6.
   - If they don't differ, diagnose before spending more.
2. **Review draft PR #100** (faster style runs).
3. **Stage 3, our own RVC,** which needs a go for a rental under $1.
4. **The 27B chat strength sweep,** which needs a go line once it has been built and dry-run.
5. **Home cloning checks on the real model** (cpu-f32, then vk-q8), on his go. These must respect the home memory limits.
6. **Cluster purity review** (0 of 10 done on the hub): is it still wanted? The suggestion is to retire it, since the Claude labels replaced it and the clusters found no missing mood.
7. **The one-page goals doc** (`what-brandon-wants.md`) needs a final read with the answered questions folded in.
8. **Leftover listening sets** are on the hub but no longer needed: 14 refscreen lists, and a 128-clip voicearms set stopped at 35.

## Pull requests

| PR | What | State |
|---|---|---|
| #100 | Speeds up style-encoder runs | Draft, open, awaiting review |
| #99 | Style-encoder collapse fix | Merged |
| #98 | `--init-from` with LR warmup | Merged |
| #97 | Full fine-tune with the style-encoder guards and `--style-lr` | Merged |
| #96 | Earlier fine-tune code | Merged; superseded by #97 |
| #95 | Style encoder (label-less route) | Merged |
| #94 | Voice training job (full fine-tune, KL replay, frame-offset fix) | Merged |
| #93 | refscreen excludes screams, pain lines and combat callouts | Merged |
| #92 | Duplicate voice plan | Closed |
| #91 | codectrip and rvctest suit checks | Merged |
| #90 | refrank (duplicate of #89) | Closed |
| #89 | refscreen | Merged |
| #87, #88 | voicearms3 and her neutral voice | Merged |

## Budget

| Pot | State |
|---|---|
| Vast.ai GPU rentals | $8.66 credit left. Recent costs: run B v2 $1.69, runs A and B $1.30, fine-tune smoke $0.34, RVC rental $0.06, refscreen $1.61, voicearms3 about $3.30. |
| Claude API (text-only labelling) | $22.24 of $25 spent. The run is finished, and the rest is held in reserve. |
| Overall | The project budget is about $25 for GPUs, kept separate from the Claude credit. |

## Known issues and gotchas

- **The pre-rental check must cover compilers.** It once missed a missing C++ compiler, which made an RVC step fail on the worker.
- **The resume file from a full fine-tune is too big to bring home.** Continuing therefore restarts from the saved weights with a fresh optimizer (`--init-from`).
- **Brandon's disk is tight:** the system disk had 12 GB free on 2026-10-26, and `/home` had 148 GB free. Big outputs and pip temp files go on `/home`.
- **Vulkan:** anything that overfills the 8 GB RX 480 can crash the PC. F32 TTS runs on the CPU or a rental only.
- **Cloud sandboxes can't download from Hugging Face.** Real-model tests run on his PC or the rental.
- **Remote Control** needs Claude Desktop 2.1.302 or newer. Older versions silently dropped messages.
- **Rental CUDA:** textscore rentals needed min_cuda 13.0. Check the CUDA version on every new worker image.

## Where to find more

- **In this repo:** `README.md`, `docs/cloud-worker.md`, `docs/vectors.md`, `docs/design/qwen3tts-finetune-guide.md`, `docs/design/must-have-moods.md`, `docs/design/labelling-run-plan.md`, `docs/design/after-labels-plan.md` and `docs/design/voice-consistency-check.md`.
- **On Brandon's PC (gitignored):** `corpus/tali/`, including `analysis/` (after-labels, refscreen and RVC results, plus the duds table at `analysis/refscreen_rvc/duds.md`), `stages/` (voicearms and the fine-tune runs) and `persona/`.
- **Private to Brandon's Claude account:**
  - the "Tali voice plan" page;
  - the live "VI Assistant State" page, which this file mirrors;
  - research notes on steering Qwen3-TTS and Qwen3.8-27B, voicearms against the literature, suit and accent options, and fan Tali voice recreations.

  Ask him for copies if you need them.
