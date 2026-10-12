# Local state, cloud compute

Status: proposal. Nothing here is built yet.

The heavy models (LLM, TTS, control vector generation) run on rented Vast.ai GPUs, and those instances can vanish at any moment. So everything durable lives on Brandon's machine, and the cloud worker is stateless: a fresh instance can pick up the next turn with nothing carried over.

## Who holds what

| Home (NixOS machine, `vi`) | Cloud worker (Vast.ai, `vi-worker`) |
|---|---|
| Character packs | Model weights (pulled on boot) |
| Conversation log and state snapshot | Warm caches only (prompt KV cache, loaded vectors) |
| Reference control vectors, per model | Nothing written to disk that matters |
| Mixing table evaluation (state to strengths) | LLM inference with blended control vectors |
| Director (rules, or a small model on CPU or the RX 480) | TTS with blended control vectors |
| Microphone, speakers, speech-to-text | |
| Agents and their credentials (calendar, email) | |

Credentials never leave home: when Tali sends a drone to your inbox, the agent runs locally and only its results go into the conversation.

## A turn

1. Home records your message, runs the director, and applies its events to the state machine in memory. Nothing is committed yet.
2. Home sends the worker one self-contained request: the prompt (persona, state overlay, recent history), the mixing table output for each target (`llm`: drunk 0.42, ...; `tts`: sad 0.3, drunk 0.6, ...), the content hashes of the vectors it needs, and sampling settings.
3. If the worker lacks a vector, it asks for it by hash and home uploads it (about half a megabyte for an 8B LLM vector). Vectors are cached by hash, so a re-rented worker re-fetches once.
4. The worker streams text back, and audio sentence by sentence.
5. When the turn completes, home commits everything in one transaction: your message, the events, her reply, and the new snapshot.

If the worker dies mid-turn, nothing was committed, so home resends the same request to a new worker. The only cost of a fresh worker is re-reading the persona prompt and re-fetching vectors, which is a second or two.

## Storage at home
SQLite in one file: conversations, turns, snapshots, and a vector index (name, target, model hash, content hash, path). Vector files sit next to it. Back up one folder and you have everything.

## Connection
Home always dials out to the worker, so home can stay behind NAT. Traffic goes over an SSH tunnel or WireGuard to the instance's exposed port, plus a per-instance token. Text and audio are visible to the rented machine while it computes, so nothing is written to its disk and logs stay off.

## Lifecycle
- `vi cloud up` rents an instance through the Vast API from a saved template, waits for the models to load, and connects.
- `vi cloud down` releases it, and an idle timeout releases it automatically to protect the budget.
- Weights come from Hugging Face on boot. If that's too slow, we bake them into the instance image.
- Offline jobs (harvest labelling, control vector generation, the bake-off) use the same worker with a batch endpoint, and their outputs are pulled home.

## Later: owning the hardware
When the models run on local hardware, `vi-worker` runs on the same machine or LAN and nothing else changes.
