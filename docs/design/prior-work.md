# Prior work: what others tried, and what it means for us

Status: research primer, 2026-10-13. Aim: turn the Tali pipeline into a method that works for any character with recorded lines and a wiki. Each section says what others did, what worked, what didn't, and what we should take from it. Our own results from the 40-clip smoke test (2026-10-13) are included where they agree or disagree.

Sources are linked inline. Numbers are as the papers report them.

## 1. Labeling expressive speech at scale

**What others built.** Several groups label large speech corpora with style and emotion automatically, for training expressive TTS:
- [ParaSpeechCaps](https://arxiv.org/abs/2503.04713) (EMNLP 2025) scaled 59 style tags from 282 h of human labels to 2,427 h automatically. Situational tags such as happy or whispered went through a three-step cascade. First, a dominance/valence/arousal model kept only clips with extreme values. Second, a text embedder kept transcripts that match the emotion semantically, with keyword filters so a line that merely says "angry" doesn't count. Third, Gemini 1.5 Flash rated how well the delivery fits on a 5-point scale, and only clips rated 5 were kept. Each step mattered: tag recall was 71.3% with all three, versus 61.0%, 66.1% and 63.3% with each step removed. That matched their human-labelled baseline (68.1%). They could not scale "clarity" (slurred, stammering) or "expressiveness" automatically.
- [SpeechCraft](https://arxiv.org/abs/2408.13608) (ACM MM 2024): expert classifiers and captioning models measure each clip, then a fine-tuned LLaMA writes the natural-language description. About 2,000 h, 2M clips.
- [Data-Speech](https://github.com/huggingface/dataspeech), the Parler-TTS pipeline: measure continuous features (speaking rate, pitch mean and spread, SNR, reverberation), map them to word bins ("quite slowly", "very monotone"), then have an LLM write the sentence.

**What worked.** Measure with specialist models and use the LLM to *verify* or *phrase*, not to *discover*. In ParaSpeechCaps the audio LLM was the last filter, asked a narrow question ("how well does this delivery fit *angry*, 1 to 5") about clips the other models had already shortlisted, and only its top rating counted.

**What didn't.** Free-form audio-LLM labelling as the primary signal. See section 2.

**For us.** Discover asks Gemma an open question ("describe it, tag it, score six axes"). The smoke test shows the cost of that:
- `other_voice` on 29 of 29 clips, even though the prompt says the helmet filter is normal.
- Refusals on 11 of 40 clips, nearly all under 1.6 s.
- Constant zeros for joy, playfulness and intoxication once the words were shown.

Move toward the cascade: acoustic models and the text judge shortlist candidates per axis, and Gemma rates fit on a rubric only for the shortlist. Keep its open captions for discovery, where being wrong is cheap.

## 2. How far to trust an audio LLM on emotion

- [LISTEN](https://arxiv.org/abs/2510.10444) (EACL 2026) tested six audio LLMs. They lean on the words. When the words are neutral or missing they answer "neutral". When words and tone conflict they fail. On paralinguistic-only cues they are near chance.
- Audio LLMs hallucinate sound events that aren't in the audio and get their order wrong ([Kuan and Lee](https://arxiv.org/abs/2410.16130)).
- [BRACE](https://arxiv.org/abs/2512.10403): even the best audio LLM scored 63.19 F1 at judging whether a caption matches its audio.

**Our data agrees.** Showing Gemma the words barely changed its scores (mean absolute change 0 to 0.12 per axis). On the axes the words decide, its scores were constant. Its captions are near-identical ("steady, slightly filtered…").

**For us.** Treat the audio LLM's axis scores as weak (fusion weight 0.5 is right, or lower), and keep `audio_llm_axes` limited to axes where it agrees with the acoustic models on reviewed clips. Short clips need either padding with silence or skipping. Test which on the next run rather than assume.

## 3. Acoustic emotion models and the limits of "ground truth"

- Humans disagree a lot on speech emotion. On IEMOCAP, the standard benchmark, categorical labels reach Fleiss' kappa 0.48, a quarter of clips get no agreed label, and human accuracy is about 69% ([as summarized here](https://arxiv.org/abs/2210.07642)).
- Self-supervised speech models carry emotion best in their middle layers, and a simple linear model on the right layer is competitive ([a study on emotional corpora](https://arxiv.org/abs/2210.02595); [a probing-head benchmark](https://arxiv.org/abs/2308.14456)).
- LLM annotators: GPT-4 emotion labels are often preferred by human raters, and work well for flagging bad labels. They fail on emotions that need pragmatic inference, and agreement *between* LLMs overstates accuracy against human gold labels ([Rethinking Emotion Annotations in the Era of LLMs](https://arxiv.org/abs/2412.07906)).

**For us.**
1. Calibrate against Brandon's reviews, not model-vs-model agreement. `fusion.json`'s Spearman table is a health check, not a measure of correctness.
2. Train per-axis probes on the saved embeddings (`embeddings.jsonl`, added 2026-10-13) once there are a few hundred reviewed clips. This also covers the axes no emotion2vec+ class maps to.
3. Keep ambiguous clips out of contrast sets rather than forcing a label. Low fused confidence already marks them.
4. The smoke test's top emotion2vec+ class was "angry" for 14 of 40 clips. Check against reviews whether the helmet filter is biasing it.

## 4. Cleaning a character's audio

- [Emilia-Pipe](https://arxiv.org/abs/2407.05361) turns arbitrary audio into TTS training clips in six steps: standardize (24 kHz, loudness), separate sources, speaker diarization (pyannote), VAD segmentation to 3–30 s, Whisper ASR, then filter on DNSMOS. Average DNSMOS went from 2.52 to 3.26.
- ParaSpeechCaps found a perceptual speaker-similarity model (VoxSim, cosine ≥ 0.8) recovered more correct speaker-level tags than a standard WavLM ECAPA embedder (50.3% vs 45.3% recall).
- Hobbyist voice cloning (e.g. [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)) uses the same steps on a small scale: UVR5 vocal separation and de-reverb, slicing to roughly 2–10 s, and ASR to label. It needs only minutes of clean audio.

**For us.** Game dialogue is already isolated, one line per file, with the game's own transcript, which is a large advantage over any in-the-wild source. What's left:
- **Speaker check by embedding, not by LLM.** Compute a speaker embedding per clip, take the character's median, and flag clips far from it. That replaces Gemma's `other_voice` tag, and it's what the voice-clone set needs. For characters whose voice changes (masks, helmets, radio), the median may need computing per condition.
- **Speech-quality score** (DNSMOS or similar) per clip, for choosing TTS references.
- **Clips under ~1 s** (barks) are a known weak spot for every model here. Tag them and keep them out of anything that needs a reliable label.

## 5. Building the persona (the LLM side)

- [ChatHaruhi](https://arxiv.org/abs/2308.09597): extracts each character's lines from scripts and novels into a memory, retrieves similar past dialogue as in-context examples, and adds LLM-generated dialogue. 32 characters, about 54k dialogues. Retrieval of the character's real lines is the core of its style imitation.
- [RoleLLM](https://arxiv.org/abs/2310.00746): builds profiles for 100 roles, generates role-specific Q&A from the source text ("Context-Instruct"), imitates speaking style with prompting, then fine-tunes. The fine-tuned open models came close to GPT-4 prompting.
- [Character-LLM](https://arxiv.org/abs/2310.10158): reconstructs a character's experiences as scenes and trains on them. Trained agents were more vivid than prompted ones.
- [Ditto](https://arxiv.org/abs/2401.12474): builds profiles for 4,000 characters from Wikipedia and Wikidata, has the model simulate dialogues as reading comprehension, and self-tunes on them. It kept role identity across multi-turn chats better than open baselines.
- [CharacterEval](https://arxiv.org/abs/2401.01275): extracts dialogue with GPT-4, then has humans check it. Its trained reward model tracks human judgement better than GPT-4 as a judge does.
- [TimeChara](https://arxiv.org/abs/2405.18027): role-play models, GPT-4o included, leak knowledge from later in the story ("point-in-time hallucination"). Splitting the reasoning across narrative experts reduces it.

**What worked.** Real lines from the source, retrieved at chat time, beat descriptions of the character for capturing voice. Profiles from wikis work for knowledge. Human checking of automatic extraction remains standard.

**What didn't.** Untimed knowledge: characters "know" things from later in the story. LLM-as-judge for character quality is weaker than a reward model trained on human ratings.

**For us.**
- **The harvest is already a ChatHaruhi-style memory**, with better provenance: every line has conversation context and game. Index it for retrieval of example lines by situation and emotion, using the label scores.
- **Tag wiki facts and lines with a point in the story** (game, and act where known). Then a persona can be set to "Tali after ME2's loyalty mission" and avoid leaking ME3 events. This generalizes to any serialized character.
- **Treat wiki extraction like CharacterEval did:** have an LLM extract, then do a short human pass.

## 6. Steering vectors (control vectors)

- [Tan et al.](https://arxiv.org/abs/2407.12404) (NeurIPS 2024): steering effect varies a lot per input. Some datasets push almost 50% of inputs the *opposite* way, and some behaviours can't be steered at any layer or strength. Steerability depends mostly on the dataset, not the model. Generalization to new prompts is decent but brittle.
- [A sober look at steering](https://www.lesswrong.com/posts/QQP4nq7TXg89CJGBh/a-sober-look-at-steering-vectors-for-llms): steering costs quality (one study likens it to halving pre-training compute on hard questions). Evaluations on multiple-choice benchmarks overstate how well it works in open-ended dialogue.
- [Persona vectors](https://www.anthropic.com/research/persona-vectors) (Anthropic, 2025): trait directions extracted from contrasting system prompts. They are useful for *monitoring* drift and screening training data as well as steering.
- TTS: [EmoSteer-TTS](https://arxiv.org/abs/2508.03543) steers emotion in flow-matching TTS (F5-TTS, CosyVoice2, E2-TTS) with no training. It takes the difference in mean activations between emotional and neutral clips (about 1,000 clips per emotion), restricted to the tokens that emotion2vec says matter most. Strength 2 worked; strength 3 could make speech unintelligible. The authors note it depends on high-quality emotional samples.
- Alternatives for TTS emotion without vectors:
  - Reference selection: [ExpPro](https://arxiv.org/abs/2409.18512) improved both emotion and speaker similarity by choosing the right reference clip per line, and found the best reference depends on the model.
  - Models that separate timbre from emotion, e.g. IndexTTS2.

**For us.**
- Expect some axes to be unsteerable or anti-steerable for a given model. The evaluate stage's strength sweep is the right defence, and it must run on open-ended chat prompts, not single sentences.
- Minimal pairs (same content, one axis changed) are the dataset design these papers point to. Keep them.
- **Decision for you:** for TTS emotion, steering vectors (our plan, closest to EmoSteer-TTS) compete with simply choosing an emotion-matched reference clip per line from the labelled corpus. The second needs no extraction stage and uses the labels directly. Worth prototyping both on one axis before building the TTS extract stage.
- Persona-vector-style monitoring could watch for the assistant drifting out of character over a long chat (section 7).

## 7. Staying in character over long conversations

- [Li et al.](https://arxiv.org/abs/2402.10962): system-prompt personas drift noticeably within 8 rounds, which the authors tie to attention decay. They propose split-softmax as a mitigation.
- [Style Amnesia](https://arxiv.org/abs/2512.23578) (ACL 2026 Findings): spoken language models lose a requested speaking style (emotion, volume, pace) after a few turns, though they can still recall the instruction. Putting the style in a user message works better than in the system message, and asking the model to recall it helps.

**For us.** The mood state director should re-assert persona and current mood each turn (or every few) rather than once in a system prompt. Control vectors help here because they don't decay with context length. That only holds if section 6's per-axis checks pass.

## 8. The method, generalized

For any new character:
1. **Gather sources by tier.**
   - Gold: line-level audio with transcript and dialogue context from game or show files.
   - Silver: wiki and transcripts for knowledge and timeline.
   - Bronze: community clips that need Emilia-style cleaning.
2. **Clean and verify.**
   - ASR against the transcript.
   - Speaker-embedding distance from the character's median.
   - A quality score.
   - Duration flags.
3. **Discover** with embeddings and clustering, using the audio LLM to name clusters, not to label clips.
4. **Label with a cascade:**
   - acoustic and dimensional models shortlist,
   - the text judge scores in context,
   - the audio LLM rates fit only for the shortlist,
   - priors from stage directions and scene.
5. **Review about an hour** with the review queue ordered by uncertainty. Then train per-axis probes on embeddings and recalibrate the fusion weights against the reviews.
6. **Build artifacts:**
   - a retrieval index of real lines by situation and emotion,
   - point-in-time persona facts,
   - contrast sets and minimal pairs,
   - TTS references ranked by quality and emotion fit.
7. **Extract and evaluate per axis and per model.** Sweep strength on open-ended prompts. Record unsteerable axes and fall back to prompting or reference selection for them.
8. **At runtime:**
   - retrieve example lines,
   - re-assert persona and mood each turn,
   - monitor drift.

Character-specific settings live in the pack. Examples: the helmet note in the caption prompt (a voice filter is normal), scene priors, stage-direction patterns, and which axes exist. The pipeline code stays the same.

## Open questions to test, not assume
- Does padding short clips with silence fix Gemma's refusals, or does it just hallucinate on the padding?
- Do Gemma's fit ratings on a shortlist (ParaSpeechCaps-style) agree with reviews better than its open captions do?
- Is emotion2vec+ biased toward "angry" by the helmet filter? Is a probe on its embeddings better?
- TTS: steering vectors vs emotion-matched reference selection, on one axis.
- Which LE3 stage-direction words actually occur? The current regexes fired for 1 of 6 smoke-test clips.
