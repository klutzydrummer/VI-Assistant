# Moods, states and voice direction

Status: proposal for review. Nothing here is implemented yet.

## Goals
- The dialogue model writes only what the character says. No app tags in its output or its context.
- States are a real state machine: defined per character pack, validated in code, predictable.
- Moods are flexible and continuous, and can be driven by a small model, a second LLM slot, or plain rules, behind one interface.
- Voice direction travels out of band, in whatever form each TTS engine wants.
- First audio stays fast: target under 1.5 s from the end of the user's message.

## Three layers

| Layer | What it is | Who changes it | Lifetime |
|---|---|---|---|
| **State** | Discrete mode from the pack's state machine: `sober`, `tipsy`, `drunk`, `hungover` | Events, validated by the machine | Until a transition fires |
| **Mood** | Continuous vector, e.g. valence, arousal, plus named emotion weights (worried 0.6) | Director proposals, smoothed | Decays toward the character's baseline each turn |
| **Delivery** | Per-sentence voice direction: emotion, intensity, style text, pace | Director, per sentence | One sentence |

State changes what she *is* (prompt overlay, voice bank, tool permissions). Mood colors *how* she says things. Delivery is the concrete instruction to the voice engine.

## The state machine (in the pack)
Each pack gets a `states.toml`. The pack defines the vocabulary; the engine enforces it.

```toml
initial = "sober"

[[event]]
id = "drink"
description = "Tali accepts and drinks alcohol."

[[event]]
id = "sober_up"
description = "Tali drinks water, eats, or time passes without drinks."

[[state]]
id = "tipsy"
prompt = "states/tipsy.md"
voice = "drunk"            # voice bank to draw clips from
voice_intensity = 0.4      # how hard to push it, for engines that take intensity
flags = ["impaired"]

[[state]]
id = "drunk"
prompt = "states/drunk.md"
voice = "drunk"
voice_intensity = 0.9
flags = ["impaired", "no_risky_tools"]

[[transition]]
from = "sober"
on = "drink"
to = "tipsy"

[[transition]]
from = "tipsy"
on = "drink"
to = "drunk"

[[transition]]            # time-based: no event needed
from = "drunk"
after_turns = 6
to = "hungover"
```

Rules:
- Transitions fire on **events** (named in the pack) or **timers** (`after_turns`, `after_minutes`). Anything not listed is rejected, so a model can never put her in an invalid state.
- **Flags** are what code checks. `no_risky_tools` blocks send, delete and spend tools in Rust, not in the prompt.
- The user can always force an event or state with a command (`/state sober`). The machine is the single source of truth, saved with the conversation.
- `vi pack check` validates the graph: unknown states or events, unreachable states, states with no way out.

## The director
One trait, several implementations:

```rust
trait Director {
    /// Before the reply: which pack events did the user's message trigger,
    /// and how does mood shift?
    async fn on_user_turn(&self, ctx: &TurnContext) -> TurnDirection; // events + mood delta
    /// Per sentence of the reply, as it streams: how should it sound?
    async fn on_sentence(&self, ctx: &SentenceContext) -> Delivery;
}
```

Implementations, in the order we'd build them:
1. **Rules**: keyword and pattern triggers from the pack, plus a default delivery from the current mood. No model, always available, a fallback.
2. **Small local model**: a 1.5 to 4B instruct model (Qwen3 class) with grammar-constrained JSON output, so it can only answer with event ids from the pack and emotion names from the voice bank. It's small enough for CPU or the RX 480 via llama.cpp Vulkan, or alongside everything else on the rented GPU.
3. **Second LLM slot**: same prompt and schema, pointed at a bigger model when quality matters more than cost.

Model output is a proposal. Mood deltas are clamped and smoothed (exponential moving average) so one odd guess doesn't swing her voice, and events still go through the state machine.

## The turn pipeline

```
user message
  │
  ├─▶ Director.on_user_turn ──▶ state machine applies events ──▶ mood update
  │      (small model, ~100-300 ms)
  ▼
dialogue LLM, streaming plain text
  prompt = persona + active state overlay + mood summary in plain words
  │
  ├─ sentence 1 done ─▶ Director.on_sentence ─▶ normalize text ─▶ TTS ─▶ play
  ├─ sentence 2 done ─▶ ... (runs while sentence 1 is playing)
  ▼
end of turn: timers tick, mood decays, state saved
```

**Is there a second pass that rewrites the dialogue? No.** Rewriting means waiting for the whole reply and risks a second model drifting her voice. The two extra steps are:
- **Director annotation**: reads each finished sentence and decides how it should sound. It never changes the words. It runs per sentence while earlier sentences are already playing, so it adds almost no latency.
- **Deterministic text normalization** (Rust, no model): strip anything unspeakable, expand numbers and units, split long sentences, and apply per-engine quirks.

The one step that must precede generation is `on_user_turn`, because a state change (she just took the drink) changes the prompt. That's a small, fast call. If it runs over a time budget, we proceed with the current state and apply the event on the next turn.

## Voice adapters
Delivery is engine-neutral:

```rust
struct Delivery {
    emotion: String,       // from the pack's voice bank
    intensity: f32,        // 0..1
    style: Option<String>, // free text, e.g. "slurred, giggling"
    pace: f32,
}
```

Each TTS engine gets an adapter that maps it to what that engine takes. Examples of the expected mappings, to confirm in the bake-off:
- **Reference-clip engines** (Qwen3-TTS, OmniVoice): pick the clip from the state's voice bank and emotion; the clip carries the delivery.
- **IndexTTS-2**: an emotion vector or emotion text, plus a weight from intensity.
- **Chatterbox**: an exaggeration level from intensity, plus the clip.

The voice bank grows from "one clip per emotion" to clips tagged with emotion, intensity and state. The adapter picks the closest match.

## Changes to what's in PR #1
- Remove the `[emotion]` and `[state:...]` tag instructions from `persona.md` and the examples. Mood goes into the prompt as a plain-language line instead ("You're feeling a little worried").
- Replace the flat `[[states]]` list in `character.toml` with `states.toml` as above, and add events.
- Keep `drunk.md` as the overlay for `drunk`, and add `tipsy.md` and `hungover.md`.

## Build order
1. State machine, `states.toml` parsing and graph validation, rules director. Pure Rust, testable now.
2. Sentence streaming pipeline with a stub TTS that prints the delivery for each sentence.
3. Small-model director with constrained output.
4. Real TTS adapters, after the bake-off picks the engine.
