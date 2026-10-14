"""embed stage: every candidate embedding and the backup scores, in one GPU
pass. The one paid job of the corpus pipeline: it runs only after Brandon has
seen the exact command and said ok to that run.

Plan: docs/design/corpus-pipeline.md, step 3. Format:
docs/design/corpus-format.md, embed. Per clip, from prep's trimmed, levelled
16 kHz copy:

  wavlm_L06..L18          WavLM-large hidden states, mean-pooled, per layer
  emotion2vec_plus_large  emotion2vec+ large (and its 9 class scores)
  emotion2vec_base        plain emotion2vec, retried now silence is trimmed
  bude_whisper            the BUD-E-Whisper encoder pooled over the frames
                          that hold audio only (Whisper pads to 30 s)
  eiv, eiv_quality        every Empathic-Insight-Voice-Small head and the
                          Plus quality heads (not content enjoyment): the
                          backup labelling and naming help
  avd                     audeering arousal/dominance/valence, a guard
  prosody, minilm         pitch, loudness and pace; text embeddings of the
                          lines with words. For reporting and naming.

Crops: the centred 0.5, 1, 1.5 and 2 s of up to `crops.sources` long clips
are embedded too (crops/<name>.npy, ids `<id>@<seconds>`), for the
shortest-usable-clip test (spec, "Sample sizes, derived" 1).

Choosing among the candidates happens at home (confounds). Every model is
fetched at the revision the settings pin; embed.json records them.

Outputs: embeddings/<name>.npy (float16, one row per clip) + .ids,
crops/<name>.npy + .ids, scores.jsonl, prosody.jsonl, embed.json,
checks.json.

Settings: backend ("fake" for tests), wavlm, wavlm_layers, emotion2vec_plus,
emotion2vec_base, bude_whisper, eiv_small, eiv_plus, eiv_skip, avd,
text_embed, revisions, crops (lengths, sources, min_source_s), batch, limit.
"""

import json
import time
from dataclasses import replace

import checks
import models
from vi_job import (
    ArrayCheckpoint,
    Checkpoint,
    ensure_packages,
    fake_models,
    load_clips,
    log,
    out_dir,
    settings,
    spread,
    write_jsonl,
)

DEFAULTS = {
    "wavlm": "microsoft/wavlm-large",
    "emotion2vec_plus": "emotion2vec/emotion2vec_plus_large",
    "emotion2vec_base": "emotion2vec/emotion2vec_base",
    "bude_whisper": "laion/BUD-E-Whisper",
    "eiv_small": "laion/Empathic-Insight-Voice-Small",
    "eiv_plus": "laion/Empathic-Insight-Voice-Plus",
    "avd": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    "text_embed": "sentence-transformers/all-MiniLM-L6-v2",
}
# Kept out of the stored vectors and into scores.jsonl, with names.
SCORES = ("eiv", "eiv_quality", "avd", "emotion2vec_plus_large_classes")


def main():
    s = settings()
    ensure_packages()
    fake = fake_models()
    out = out_dir()
    clips = [c for c in load_clips() if c.audio16k is not None]
    if not clips:
        raise SystemExit("embed needs prep as an input (its 16 kHz copies)")
    repos = {k: s.get(k, v) for k, v in DEFAULTS.items()}
    revisions = s.get("revisions", {})
    rev = lambda key: revisions.get(repos[key])  # noqa: E731
    layers = list(s.get("wavlm_layers", range(6, 19)))
    batch = int(s.get("batch", 16))
    crop = s.get("crops", {})
    lengths = [float(x) for x in crop.get("lengths", [0.5, 1, 1.5, 2])]
    long = [c for c in clips if c.duration_s >= float(crop.get("min_source_s", 4.0)) and "bark" not in c.flags]
    crops = [replace(c, id=f"{c.id}@{n:g}") for c in spread(long, int(crop.get("sources", 300))) for n in lengths]
    everything = clips + crops
    log(f"embed: {len(clips)} clips and {len(crops)} crops")
    started = time.time()

    names = {}  # score columns, filled by the models that know them
    vectors = {}
    if fake:
        model = models.FakeEmbedder(layers)
        names = {"eiv": model.HEADS, "eiv_quality": model.QUALITY, "avd": ["arousal", "dominance", "valence"],
                 "emotion2vec_plus_large_classes": [f"class{i}" for i in range(9)]}  # fmt: skip
        merge(vectors, ArrayCheckpoint("fake").run(everything, lambda part: [model(wav(c)) for c in part]))
    else:
        passes = [
            ("wavlm", lambda: models.WavLMLayers(repos["wavlm"], rev("wavlm"), layers), one_by_one),
            ("emotion2vec_plus", lambda: models.Emotion2VecEmbed(
                repos["emotion2vec_plus"], rev("emotion2vec_plus"), "emotion2vec_plus_large"), one_by_one),
            ("emotion2vec_base", lambda: models.Emotion2VecEmbed(
                repos["emotion2vec_base"], rev("emotion2vec_base"), "emotion2vec_base"), one_by_one),
            ("bude_whisper", lambda: models.BudEWhisper(
                repos["bude_whisper"], rev("bude_whisper"), repos["eiv_small"], rev("eiv_small"),
                repos["eiv_plus"], rev("eiv_plus"), skip=s.get("eiv_skip", ["score_content_enjoyment"])), batched),
            ("avd", lambda: models.AVD(repos["avd"], rev("avd")), one_by_one),
        ]  # fmt: skip
        for name, load, call in passes:
            saved = ArrayCheckpoint(name)
            model = load() if saved.todo(everything) else None
            merge(vectors, saved.run(everything, lambda part, m=model, f=call: f(m, part), chunk=batch))
            if isinstance(model, models.BudEWhisper):
                names["eiv"], names["eiv_quality"] = model.head_names, model.quality_names
            if isinstance(model, models.Emotion2VecEmbed) and model.labels:
                names["emotion2vec_plus_large_classes"] = model.labels
            del model
            free()
        names["avd"] = ["arousal", "dominance", "valence"]
        # A resumed run that skipped a model reads the names it saved.
        names = {**saved_names(), **names}
        save_names(names)

    # Prosody and text, small and cheap; on the worker since librosa and
    # sentence-transformers are there.
    heard = {c.id: replace(c, audio=c.audio16k) for c in clips}
    measure = models.fake_prosody if fake else models.prosody
    prosody = Checkpoint("prosody").run(clips, lambda part: [measure(heard[c.id]) for c in part])
    worded = [c for c in clips if c.has_words and "no_subtitle" not in c.flags]
    embedder = None
    saved = Checkpoint("text")
    if saved.todo(worded):
        embedder = (models.FakeTextEmbedder if fake else models.TextEmbedder)(repos["text_embed"])
    text = saved.run(worded, lambda part: embedder([c.text for c in part]), chunk=256)

    # Outputs.
    ids = [c.id for c in clips]
    stored = {}
    for key in sorted({k for v in vectors.values() for k in v} - set(SCORES)):
        stored[key] = write_matrix(out / "embeddings", key, ids, [vectors[i][key] for i in ids])
    crop_ids = [c.id for c in crops]
    for key in sorted({k for i in crop_ids for k in vectors[i]} - set(SCORES)):
        write_matrix(out / "crops", key, crop_ids, [vectors[i][key] for i in crop_ids])
    if text:
        tids = [c.id for c in worded]
        stored["minilm"] = write_matrix(out / "embeddings", "minilm", tids, [text[i] for i in tids])
    scores = []
    for i in ids:
        r = {"id": i}
        for key in SCORES:
            if key in vectors[i]:
                r[key] = {n: round(float(x), 4) for n, x in zip(names.get(key, []), vectors[i][key], strict=True)}
        scores.append(r)
    write_jsonl(out / "scores.jsonl", scores)
    write_jsonl(out / "prosody.jsonl", [{"id": i, **prosody[i]} for i in ids])
    report = {
        "clips": len(ids),
        "embeddings": stored,
        "crops": {"lengths": lengths, "sources": len(crops) // max(1, len(lengths)), "clips": len(crops),
                  "placement": "centred"},  # fmt: skip
        "models": {k: {"repo": repos[k], "revision": rev(k), "resolved": resolved(repos[k], rev(k), fake)}
                   for k in DEFAULTS},  # fmt: skip
        "wavlm_layers": layers,
        "score_names": names,
        "pooling": {"wavlm": "mean over frames", "bude_whisper": "mean over the first ceil(duration x 50) "
                    "of 1500 frames", "eiv": "full padded encoder output, as trained",
                    "eiv_quality": "mean+min+max+std over all 1500 frames, as trained"},  # fmt: skip
        "packages": packages(fake),
        "seconds": round(time.time() - started, 1),
    }
    (out / "embed.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

    import numpy as np

    bad = sum(1 for i in ids + crop_ids for k, v in vectors[i].items() if not np.isfinite(np.asarray(v, dtype=np.float32)).all())
    long = sum(c.duration_s > 30 for c in clips)
    results = [
        checks.Check("non-finite vectors", bad, 0),
        checks.Check("clips over 30 s (Whisper hears only the first 30 s)", long, 5),
    ]
    checks.finish(out, results, s.get("checks", "warn"))
    log(f"embed: {len(ids)} clips, {len(stored)} embeddings, {report['seconds']} s")


def wav(clip):
    """The clip's 16 kHz samples; for a crop (`<id>@<seconds>`), the centred
    window of that length."""
    import numpy as np
    import soundfile as sf

    data, rate = sf.read(str(clip.audio16k), dtype="float32", always_2d=True)
    if rate != 16_000:
        raise ValueError(f"{clip.id}: prep's copy is {rate} Hz, not 16 kHz")
    data = data.mean(axis=1)
    if "@" in clip.id:
        n = int(round(float(clip.id.rsplit("@", 1)[1]) * rate))
        start = max(0, (len(data) - n) // 2)
        data = data[start : start + n]
    return np.ascontiguousarray(data)


def one_by_one(model, part):
    return [model(wav(c)) for c in part]


def batched(model, part):
    return model([wav(c) for c in part])


def merge(into, results):
    for cid, r in results.items():
        into.setdefault(cid, {}).update(r)


def write_matrix(folder, name, ids, rows):
    """<name>.npy (float16, one row per id) and <name>.ids."""
    import numpy as np

    folder.mkdir(parents=True, exist_ok=True)
    m = np.stack([np.asarray(r, dtype=np.float16) for r in rows])
    np.save(folder / f"{name}.npy", m)
    (folder / f"{name}.ids").write_text("".join(f"{i}\n" for i in ids))
    return {"clips": len(ids), "dim": int(m.shape[1])}


def names_path():
    from vi_job import env_path

    folder = env_path("VI_CHECKPOINT")
    return folder / "names.json" if folder else None


def saved_names():
    path = names_path()
    return json.loads(path.read_text()) if path and path.exists() else {}


def save_names(names):
    path = names_path()
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(names))


def resolved(repo, revision, fake):
    """The commit a repo resolves to at the pinned revision (or now, if none
    is pinned), so an unpinned run still records exactly what it used."""
    if fake:
        return None
    from huggingface_hub import HfApi

    return HfApi().model_info(repo, revision=revision).sha


def packages(fake):
    from importlib.metadata import PackageNotFoundError, version

    found = {}
    for name in ["numpy", "soundfile"] + ([] if fake else ["torch", "transformers", "funasr", "librosa",
                                                          "sentence-transformers", "huggingface-hub"]):  # fmt: skip
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            pass
    return found


def free():
    import gc

    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
