"""codectrip stage: does Qwen3-TTS's own audio codec already lose her suit effect and accent?
(Brandon, 2026-10-29.) Even plain neutral clones lose most of the suit, so before blaming the
model, pass her real clips through the 12 Hz speech tokenizer alone (encode, then decode) and
listen. If the round trip loses the suit, no steering or LoRA can bring it back, and the fix is
after the TTS (an effects chain or a voice-conversion pass).

Clips: from her neutral clips and every mood's positives (3-15 s, effect clips already out of
voice_sets), the `clips` with the strongest suit colouring, i.e. the spectrum shape furthest from
flat (suit_curve's RMS), at most `per_game` from each game so all three games' suits are heard.

Outputs: wav/<id>.wav (the round trip, at the codec's 24 kHz), report.md (each clip's suit gap
between original and round trip, in dB), and clips.toml, a blind hub list with each original
beside its round trip. Light enough for the home CPU at nice 19: one ~0.7 GB model.
"""

import json
import os
from pathlib import Path

import voicearms
from vi_job import ensure_venv, fake_models, load_clips, log, out_dir, settings, unit_hash
from voicearms import suit_curve


def main():
    import numpy as np

    s = settings()
    fake = fake_models()
    if not fake:
        ensure_venv("requirements-steer.txt")
    out = out_dir()
    files = Path(os.environ.get("VI_FILES", "files"))
    sets = json.loads((files / "voice_sets.json").read_text())
    clips = {c.id: c for c in load_clips()}
    lo, hi = float(s.get("ref_min_s", 3.0)), float(s.get("ref_max_s", 15.0))
    ids = list(sets["neutral"]) + [p for m in sets["moods"].values() for p, _ in m["pairs"]]
    pool = []
    for i in dict.fromkeys(ids):
        c = clips.get(i)
        if c is None or not lo <= c.duration_s <= hi:
            continue
        curve = suit_curve(c.audio)
        if curve:
            pool.append((float(np.sqrt(np.nanmean(np.square(curve)))), c, curve))
    pool.sort(key=lambda x: (-x[0], x[1].id))
    per_game, want = int(s.get("per_game", 3)), int(s.get("clips", 8))
    picked, games = [], {}
    for strength, c, curve in pool:
        g = c.id.split(".")[0]
        if games.get(g, 0) < per_game:
            games[g] = games.get(g, 0) + 1
            picked.append((strength, c, curve))
        if len(picked) == want:
            break
    log(f"codectrip: {len(picked)} clips from {len(pool)}: " + ", ".join(f"{g} {n}" for g, n in sorted(games.items())))

    codec = FakeCodec() if fake else Codec(s.get("tokenizer", "Qwen/Qwen3-TTS-Tokenizer-12Hz"))
    (out / "wav").mkdir(exist_ok=True)
    rows = []
    for strength, c, curve in picked:
        path = out / "wav" / f"{c.id}.wav"
        codec.round_trip(c.audio, path)
        after = suit_curve(path)
        gap = round(float(np.sqrt(np.nanmean((np.array(after) - np.array(curve)) ** 2))), 2) if after else None
        rows.append({"clip": c.id, "seconds": round(c.duration_s, 2), "suit_strength_db": round(strength, 2), "suit_gap_db": gap})
    (out / "report.json").write_text(json.dumps({"tokenizer": s.get("tokenizer", "Qwen/Qwen3-TTS-Tokenizer-12Hz"),
                                                 "clips": rows}, indent=1) + "\n")
    lines = ["# Codec round trip (codectrip)", "",
             "Her real clips through Qwen3-TTS's 12 Hz speech tokenizer alone. Suit strength: how far the clip's spectrum "
             "shape is from flat. Gap: RMS dB between the original's shape and the round trip's (0 = unchanged; "
             f"voicearms' neutral clones sat about 2.75 dB from her real clips).", "",
             "| clip | s | suit strength dB | gap after round trip dB |", "|---|---|---|---|"]
    lines += [f"| {r['clip']} | {r['seconds']} | {r['suit_strength_db']} | {r['suit_gap_db']} |" for r in rows]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    hub = ['title = "Codec round trip: does the codec keep her suit?"',
           'question = "Is her suit effect there? Note anything about the accent or artifacts."',
           'answers = ["fully", "partly", "missing"]', "blind = true",
           f"asked_by = {json.dumps(s.get('asked_by', 'Modeling Tali' + chr(39) + 's suit effect'))}", ""]
    entries = []
    for r in rows:
        entries.append(f'[[clip]]\nid = "{r["clip"]}"\nlabel = "{r["clip"]}: original"\n')
        entries.append(f'[[clip]]\npath = "wav/{r["clip"]}.wav"\nlabel = "{r["clip"]}: codec round trip"\n')
    entries.sort(key=lambda e: unit_hash("codectrip", e))
    (out / "clips.toml").write_text("\n".join(hub) + "\n" + "\n".join(entries))


class Codec:
    def __init__(self, model_id):
        from qwen_tts import Qwen3TTSTokenizer

        self.tok = Qwen3TTSTokenizer.from_pretrained(model_id, device_map="cpu")

    def round_trip(self, src, dst):
        import soundfile as sf

        wavs, sr = self.tok.decode(self.tok.encode(str(src)))
        sf.write(str(dst), wavs[0], sr)


class FakeCodec:
    def round_trip(self, src, dst):
        import shutil

        shutil.copy(src, dst)


if __name__ == "__main__":
    main()
