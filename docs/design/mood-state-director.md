# Moods, states and voice direction

Status: the state machine (`crates/charpack/src/machine.rs`), the rules director and the streaming turn pipeline (`crates/engine`) are implemented, runnable with `vi chat` against placeholder dialogue and voice. The model director, real dialogue and voice adapters are not built yet.

Decided 2026-10-12: state is continuous. Harvest Tali's game lines and build per-character control vectors for her voice (TTS) and her behavior (LLM). A mixing table in the pack turns the continuous variables into each vector's strength, per model, so voice and behavior blend smoothly instead of switching. Discrete states are only a readout for what has to be yes/no.

Where this runs: see `local-cloud-split.md`. State lives at home, models on a stateless rented GPU.

## Goals
- The dialogue model writes only what the character says. No app tags in its output or its context.
- States are a real state machine: defined per character pack, validated in code, predictable.
- Moods are flexible and continuous, and can be driven by a small model, a second LLM slot, or plain rules, behind one interface.
- Voice direction travels out of band, in whatever form each TTS engine wants.
- First audio stays fast: target under 1.5 s from the end of the user's message.

## Variables, events and states
Everything that changes over a conversation lives in the pack's `states.toml`:

| Piece | What it is | Example |
|---|---|---|
| **Variable** | Continuous value with a range, a resting value and a per-turn decay toward rest | `intoxication` 0..1, decays 0.06 a turn |
| **Event** | Named thing that happened, with effects on variables. The only thing a director can emit | `drink`: intoxication +0.3, joy +0.15 |
| **State** | Discrete mode derived through transitions. Carries a prompt overlay, a fallback clip bank and flags | `drunk`: `states/drunk.md`, flags `impaired`, `no_risky_tools` |
| **Transition** | Fires when all its triggers hold: an event (`on`), a variable threshold (`when`), or time in state (`after_turns`) | tipsy to drunk when intoxication >= 0.6 |
| **Mix** | One row of the mixing table: a target model, a reference vector, input weights per variable, a bias, a curve and a cap | `llm`/`affectionate` = 0.6 intoxication + 0.4 joy, capped at 0.5 |

Mood is just variables (joy, sadness, anger, worry, playfulness) that rest at zero and fade over a few turns. So "a little sad and quite drunk" is sadness 0.3 and intoxication 0.7.

The mixing table maps that state to strengths for each model: strength = curve(clamp(bias + sum(weight x variable), 0, 1)) x max. One variable can feed several vectors and one vector can draw on several variables, with negative weights to suppress (joy damps `sad`). LLM vectors get lower caps than voice ones, because strong steering makes an LLM incoherent. A richer state later (more variables, or a learned embedding) just means more inputs to the same table.

Applying the strengths: llama.cpp's `llama_set_adapter_cvec` takes the whole blended vector as a float buffer, so we compute sum(strength x vector) ourselves and set it before generating. For an 8B model that's about half a megabyte, so changing it every turn costs nothing noticeable. Stock `llama-server` only loads vectors at startup, so the worker embeds llama.cpp directly. Vectors apply during generation only, not prompt processing, so the cached persona prompt stays valid when the blend changes. On the TTS side a PyTorch forward hook adds the blended vector; updating it is a tensor write.

Rules:
- Unknown events and states are rejected, so no model can put her in an invalid state.
- Thresholds going up are higher than coming down (hysteresis), so she doesn't flicker between states.
- Flags are what code checks: `no_risky_tools` blocks send, delete and spend tools in Rust, not in the prompt.
- The user can force a state (`/state sober`). The snapshot (state, variables, turns in state) is saved with the conversation.
- `vi pack check` validates the graph: duplicate ids, unknown references, transitions with no trigger, unreachable states.

Per turn: the director's events are applied, then transitions settle (at most 8 chained). After her reply, `end_turn` decays variables, advances timers and settles again.

## The director
One trait, several implementations:

```rust
trait Director {
    /// Before the reply: which pack events did the user's message trigger?
    async fn on_user_turn(&self, ctx: &TurnContext) -> Vec<EventId>;
    /// Per sentence of the reply, as it streams: how should it sound?
    async fn on_sentence(&self, ctx: &SentenceContext) -> Delivery;
}
```

Implementations, in the order we'd build them:
1. **Rules**: keyword and pattern triggers from the pack, plus a default delivery from the current mood. No model, always available, a fallback.
2. **Small local model**: a 1.5 to 4B instruct model (Qwen3 class) with grammar-constrained JSON output, so it can only answer with event ids from the pack and emotion names from the voice bank. It's small enough for CPU or the RX 480 via llama.cpp Vulkan, or alongside everything else on the rented GPU.
3. **Second LLM slot**: same prompt and schema, pointed at a bigger model when quality matters more than cost.

Model output is a proposal: events go through the state machine, and their effects are fixed amounts set in the pack, so one odd guess nudges her mood rather than swinging it.

## The turn pipeline

```
user message
  │
  ├─▶ Director.on_user_turn ──▶ state machine applies events
  │      (small model, ~100-300 ms)
  ▼
dialogue LLM, streaming plain text
  prompt = persona + active state overlay + mood summary in plain words
  │
  ├─ sentence 1 done ─▶ Director.on_sentence ─▶ normalize text ─▶ TTS ─▶ play
  ├─ sentence 2 done ─▶ ... (runs while sentence 1 is playing)
  ▼
end of turn: variables decay, timers tick, snapshot saved
```

**Is there a second pass that rewrites the dialogue? No.** Rewriting means waiting for the whole reply and risks a second model drifting her voice. The two extra steps are:
- **Director annotation**: reads each finished sentence and decides how it should sound. It never changes the words. It runs per sentence while earlier sentences are already playing, so it adds almost no latency.
- **Deterministic text normalization** (Rust, no model): strip anything unspeakable, expand numbers and units, split long sentences, and apply per-engine quirks.

The one step that must precede generation is `on_user_turn`, because a state change (she just took the drink) changes the prompt. That's a small, fast call. If it runs over a time budget, we proceed with the current state and apply the event on the next turn.

## Voice adapters
Delivery is engine-neutral:

```rust
struct Delivery {
    vectors: BTreeMap<String, f32>, // from the machine's variables, plus per-sentence nudges
    emotion: String,                // dominant emotion, for clip-based fallbacks
    style: Option<String>,          // free text, e.g. "slurred, giggling"
    pace: f32,
}
```

Each TTS engine gets an adapter that maps it to what that engine takes:
- **Control vectors** (primary path): add each vector times its weight to the TTS model's activations.
- **Reference-clip engines** without hooks (fallback): pick the clip from the state's voice bank and dominant emotion.
- **IndexTTS-2**: an emotion vector or emotion text, plus a weight from intensity.
- **Chatterbox**: an exaggeration level from intensity, plus the clip.

### Combining a state with an emotion (sad-drunk, happy-drunk)
Engines differ in whether they can mix references:
- **IndexTTS-2** takes the speaker clip and the emotion separately: a speaker reference, plus either an emotion reference clip with a 0 to 1 strength (`emo_alpha`) or an 8-value emotion vector (happy, angry, sad, afraid, disgusted, melancholic, surprised, calm). Sad-drunk = drunk clip as speaker + sad weight. The vector is continuous, so blends and gradual shifts come free. Open question for the bake-off: whether the slur survives when the emotion comes from elsewhere.
- **CosyVoice 2/3** instruct mode takes a reference clip plus a text instruction, e.g. drunk clip + "speak sadly".
- **XTTS-v2** accepts several reference files at once and lists interpolation between speakers as a feature, but it's older and lower quality.
- **Qwen3-TTS, OmniVoice, F5-TTS and Chatterbox** take one reference, which carries timbre and delivery together. Mixing means picking the nearest clip from a bank tagged by state and emotion, or generating combination clips once with an engine that can mix and caching them.

### Character-specific emotion (Tali's sadness, not generic sadness)
Three levels, cheapest first. None needs fine-tuning.
1. **Her own clips as the emotion reference.** IndexTTS-2's emotion prompt can be one of Tali's sad lines instead of the generic vector: drunk clip as speaker, her sad clip as emotion.
2. **Cached emotion embeddings.** Run each of her labelled clips through the engine's own emotion and speaker encoders once, then average per emotion to get a "Tali-sad" embedding, and blend embeddings for mixes. Needs a small hook into the engine's Python code.
3. **Activation steering (control vectors).** The shortlisted engines are transformer LMs over audio tokens, so the text-LLM trick applies: generate the same lines with her neutral and her sad clips as references, record hidden states at a few layers, and take the mean difference as a "Tali-sad" direction. At inference add strength x direction to the residual stream with a forward hook. One GPU pass per character, composable (drunk reference + sad direction), adjustable strength. This is research: too much strength may cause audio artifacts, and we test it on whichever engine wins the bake-off.

All three need many labelled clips per emotion. The game has thousands of Tali lines, so after extraction an emotion classifier (emotion2vec class) can auto-label the corpus, and we spot-check.

Level 3 is the chosen path. Levels 1 and 2 stay as fallbacks for engines we can't hook.

## Build order
1. ~~State machine, `states.toml` parsing, graph validation, mixing table.~~ Done.
2. ~~Rules director (event cues in `states.toml`), and the sentence streaming pipeline with a stub TTS that prints each sentence's delivery.~~ Done.
3. Harvest and auto-label Tali's game lines (needs the game on Brandon's machine).
4. Bake-off, now also scoring how well each engine takes control vectors.
5. Compute control vectors per axis: TTS by contrasting her clips, LLM with llama.cpp's `cvector-generator` on contrasting dialogue. Then the real adapters.
6. Stateless cloud worker and `vi cloud up/down` (see `local-cloud-split.md`).
7. Small-model director with constrained output.
