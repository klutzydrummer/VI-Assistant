"""discover stage: let the data say what's in it before anything is scored.

Embeds every clip by emotion (emotion2vec) and prosody, clusters them, has
the audio LLM caption each clip from its sound alone, and summarizes each
cluster. Plan: docs/design/control-vector-datasets.md, step 3. Outputs
(corpus-format.md, discover):

  captions.jsonl   {id, caption, emotions, tags, scores, heard_without_text,
                    with_text?}  scores are the audio LLM's per-axis guesses
                    without the words; label uses them as its audio_llm signal
  clusters.jsonl   {id, cluster, x, y, text_cluster}  cluster -1 is noise
  prosody.jsonl    {id, duration_s, f0_median_hz, ..., words_per_s}
  clusters.md      what each cluster seems to be, and the emotion vocabulary
  discover.json    counts and the with/without-words comparison
  embeddings/      emotion.npy (emotion2vec+ per clip) and text.npy (lines
                   with words), each with a .ids file giving the row order

Settings: backend ("fake" for tests), emotion (embedding model), text_embed,
audio_llm, batch, quant, compare (how many clips to also caption with their
words shown), min_cluster, limit.
"""

import gc
import json
import math
import tempfile
from collections import Counter
from pathlib import Path

import fusion
import models
from vi_job import (
    Checkpoint,
    ensure_packages,
    fake_models,
    load_axes,
    load_clips,
    log,
    out_dir,
    settings,
    spread,
    unit_hash,
    write_embeddings,
    write_jsonl,
)


def free():
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


def main():
    s = settings()
    ensure_packages()
    fake = fake_models()
    out = out_dir()
    clips = load_clips()
    axes = load_axes()
    log(f"discover: {len(clips)} clips, axes {', '.join(a.id for a in axes)}")

    # 1. Acoustic emotion embeddings.
    emotion_model = s.get("emotion", "emotion2vec/emotion2vec_plus_large")
    embedded = Checkpoint("emotion")
    if embedded.todo(clips):
        e2v = (models.FakeEmotion2Vec if fake else models.Emotion2Vec)(emotion_model)
        embedded.run(clips, lambda part: [e2v(c)[1] for c in part])
        del e2v
        free()
    emotion = embedded.run(clips, None)
    embeddings = {"emotion": write_embeddings(out / "embeddings", "emotion", emotion_model, emotion)}

    # 2. Prosody.
    measure = models.fake_prosody if fake else models.prosody
    pros = Checkpoint("prosody").run(clips, lambda part: [measure(c) for c in part])
    write_jsonl(out / "prosody.jsonl", [{"id": cid, **p} for cid, p in pros.items()])

    # 3. Text embeddings, for lines with words.
    worded = [c for c in clips if c.text.strip() and not c.nonverbal]
    text_model = s.get("text_embed", "sentence-transformers/all-MiniLM-L6-v2")
    embedder = (models.FakeTextEmbedder if fake else models.TextEmbedder)(text_model)
    text_emb = dict(zip([c.id for c in worded], embedder([c.text for c in worded])))
    embeddings["text"] = write_embeddings(out / "embeddings", "text", text_model, text_emb)

    # 4. Captions from the audio alone, and a sample with the words shown.
    gemma = models.Judge(
        (models.FakeGemma if fake else models.Gemma)(
            s.get("audio_llm", "google/gemma-4-12B-it"),
            batch=int(s.get("batch", 8)),
            quant=s.get("quant", "auto"),
        )
    )
    hearable = [c for c in clips if c.duration_s <= models.AUDIO_LLM_MAX_S]
    log(f"captioning {len(hearable)} clips from audio alone")
    # Short clips padded with silence, outside VI_OUT so they aren't outputs.
    pad_dir = tempfile.TemporaryDirectory()
    audio = {c.id: c.audio if fake else models.padded(c, Path(pad_dir.name)) for c in hearable}
    replies = Checkpoint("captions").run(
        hearable,
        lambda part: gemma.json([models.caption_prompt(c, axes, audio=audio[c.id]) for c in part], max_new_tokens=200),
        chunk=64,
    )
    captions = {}
    failed = 0
    for clip in hearable:
        r = replies[clip.id]
        if not isinstance(r, dict):
            failed += 1
            continue
        captions[clip.id] = clean_caption(r, axes)

    sample = spread([c for c in hearable if c.id in captions and c.id in text_emb], int(s.get("compare", 200)))
    log(f"captioning {len(sample)} clips again with their words shown")
    replies = Checkpoint("captions_with_text").run(
        sample,
        lambda part: gemma.json(
            [models.caption_prompt(c, axes, with_text=True, audio=audio[c.id]) for c in part], max_new_tokens=200
        ),
        chunk=64,
    )
    for clip in sample:
        if isinstance(replies[clip.id], dict):
            captions[clip.id]["with_text"] = clean_caption(replies[clip.id], axes)

    write_jsonl(
        out / "captions.jsonl",
        [{"id": c.id, "heard_without_text": True, **captions[c.id]} for c in clips if c.id in captions],
    )

    # 5. Clusters.
    ids = [c.id for c in clips]
    if fake:
        acoustic = fake_clusters(ids, "acoustic")
        by_text = fake_clusters(list(text_emb), "text")
    else:
        acoustic = cluster(ids, [acoustic_features(emotion[i], pros[i]) for i in ids], int(s.get("min_cluster", 15)))
        by_text = cluster(list(text_emb), list(text_emb.values()), int(s.get("min_cluster", 15)))
    write_jsonl(
        out / "clusters.jsonl",
        [
            {"id": i, "cluster": acoustic[i][0], "x": acoustic[i][1], "y": acoustic[i][2],
             "text_cluster": by_text[i][0] if i in by_text else None}
            for i in ids
        ],
    )  # fmt: skip

    # 6. What each acoustic cluster is.
    by_id = {c.id: c for c in clips}
    groups = {}
    for i in ids:
        groups.setdefault(acoustic[i][0], []).append(i)
    order = sorted((k for k in groups if k != -1), key=lambda k: -len(groups[k]))
    prompts = []
    for k in order:
        members = groups[k]
        cx = sum(acoustic[i][1] for i in members) / len(members)
        cy = sum(acoustic[i][2] for i in members) / len(members)
        near = sorted(members, key=lambda i: (acoustic[i][1] - cx) ** 2 + (acoustic[i][2] - cy) ** 2)
        exemplars = [
            {"id": i, "text": by_id[i].text, "caption": captions.get(i, {}).get("caption", "?"),
             "emotions": captions.get(i, {}).get("emotions", [])}
            for i in near[:10]
        ]  # fmt: skip
        prompts.append(exemplars)
    summaries = gemma.json([models.summary_prompt(e, axes) for e in prompts], max_new_tokens=200)
    del gemma
    free()

    vocab = vocabulary(captions, fake, s)
    comparison = compare(captions, axes)
    write_report(out, clips, groups, order, prompts, summaries, by_id, captions, vocab, comparison)
    (out / "discover.json").write_text(
        json.dumps(
            {
                "clips": len(clips),
                "captioned": len(captions),
                "caption_failures": failed,
                "too_long_for_audio_llm": len(clips) - len(hearable),
                "clusters": len(order),
                "embeddings": embeddings,
                "noise": len(groups.get(-1, [])),
                "with_text_compared": sum(1 for c in captions.values() if "with_text" in c),
                "words_change_scores": comparison,
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    log(f"discover: {len(order)} clusters, {len(captions)} captions, {failed} unparsed replies")


def clean_caption(r, axes):
    scores = r.get("scores") if isinstance(r.get("scores"), dict) else {}
    delivery = r.get("delivery") if isinstance(r.get("delivery"), list) else []
    emotions = r.get("emotions") if isinstance(r.get("emotions"), list) else []
    return {
        "caption": str(r.get("caption", "")).strip(),
        "emotions": [str(e).strip().lower() for e in emotions if str(e).strip()][:4],
        "tags": sorted({str(d).strip().lower() for d in delivery} & set(models.DELIVERY)),
        "scores": {a.id: fusion.clamp01(scores.get(a.id)) for a in axes},
    }


def acoustic_features(emb, p):
    """The emotion embedding plus prosody, prosody scaled to matter about as
    much as the embedding does after standardizing (see cluster)."""
    keys = ["f0_median_hz", "f0_range_st", "loudness_db", "loudness_range_db", "voiced_fraction"]
    feats = [p[k] if p[k] is not None else 0.0 for k in keys]
    feats.append(math.log(max(p["duration_s"], 0.05)))
    return list(emb) + feats


def cluster(ids, vectors, min_cluster):
    """id -> (cluster, x, y): standardize, PCA, UMAP to 10 dimensions for
    HDBSCAN and to 2 for plotting."""
    import numpy as np
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA

    if len(ids) < 5:
        return {i: (-1, 0.0, 0.0) for i in ids}
    x = np.asarray(vectors, dtype=np.float32)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    x = PCA(n_components=min(50, x.shape[0] - 1, x.shape[1]), random_state=0).fit_transform(x)
    if len(ids) >= 30:
        import umap

        nn = min(30, len(ids) - 1)
        z = umap.UMAP(n_components=10, n_neighbors=nn, min_dist=0.0, random_state=0).fit_transform(x)
        xy = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=0.1, random_state=0).fit_transform(x)
    else:
        z = x
        xy = x[:, :2] if x.shape[1] >= 2 else np.pad(x, ((0, 0), (0, 2 - x.shape[1])))
    labels = HDBSCAN(min_cluster_size=max(2, min(min_cluster, len(ids) // 4)), copy=True).fit_predict(z)
    return {i: (int(l), round(float(p[0]), 4), round(float(p[1]), 4)) for i, l, p in zip(ids, labels, xy)}


def fake_clusters(ids, salt):
    return {
        i: (int(unit_hash(salt, i) * 4) - 1, round(unit_hash(salt, i, "x") * 10, 4), round(unit_hash(salt, i, "y") * 10, 4))
        for i in ids
    }


def vocabulary(captions, fake, s):
    """The audio LLM's emotion words, grouped by meaning, with counts."""
    counts = Counter(w for c in captions.values() for w in c["emotions"])
    words = [w for w, n in counts.most_common(200) if n >= 2]
    if fake or len(words) < 4:
        return [[(w, counts[w])] for w in words]
    from sklearn.cluster import AgglomerativeClustering

    emb = models.TextEmbedder(s.get("text_embed", "sentence-transformers/all-MiniLM-L6-v2"))(words)
    labels = AgglomerativeClustering(
        n_clusters=None, distance_threshold=0.6, metric="cosine", linkage="average"
    ).fit_predict(emb)
    groups = {}
    for w, l in zip(words, labels):
        groups.setdefault(int(l), []).append((w, counts[w]))
    return sorted(groups.values(), key=lambda g: -sum(n for _, n in g))


def compare(captions, axes):
    """Per axis, how much showing the words moves the audio LLM's score: the
    mean absolute change, and the rank correlation between the two. A large
    change means its no-words score was mostly not what it would say once it
    knows the words, which is fine; what matters for trusting it is the label
    stage's agreement with the acoustic models (labels' fusion.json)."""
    out = {}
    pairs = [(c["scores"], c["with_text"]["scores"]) for c in captions.values() if "with_text" in c]
    for a in axes:
        xs = [p[0].get(a.id) for p in pairs]
        ys = [p[1].get(a.id) for p in pairs]
        diffs = [abs(x - y) for x, y in zip(xs, ys) if x is not None and y is not None]
        out[a.id] = {
            "mean_abs_change": round(sum(diffs) / len(diffs), 3) if diffs else None,
            "spearman": fusion.spearman(xs, ys),
        }
    return out


def write_report(out, clips, groups, order, prompts, summaries, by_id, captions, vocab, comparison):
    md = ["# Discovery", "", f"{len(clips)} clips, {len(order)} clusters by sound, {len(groups.get(-1, []))} unclustered.", ""]
    for k, exemplars, summary in zip(order, prompts, summaries):
        members = groups[k]
        summary = summary or {}
        flags = Counter(f for i in members for f in by_id[i].flags)
        tags = Counter(t for i in members for t in captions.get(i, {}).get("tags", []))
        games = Counter(by_id[i].game for i in members)
        md += [
            f"## Cluster {k}: {summary.get('name', '?')} ({len(members)} clips)",
            "",
            summary.get("summary", "(no summary)"),
            "",
            f"- closest axis: {summary.get('axis', '?')}",
            f"- games: {', '.join(f'{g} {n}' for g, n in games.most_common())}",
            f"- flags: {', '.join(f'{f} {n}' for f, n in flags.most_common()) or 'none'}",
            f"- heard: {', '.join(f'{t} {n}' for t, n in tags.most_common(6)) or 'nothing notable'}",
            "- examples:",
        ]
        md += [f"  - `{e['id']}` \"{e['text'] or '(no words)'}\": {e['caption']}" for e in exemplars[:5]]
        md.append("")
    md += ["## Emotion words the audio model used", ""]
    md += ["- " + ", ".join(f"{w} ({n})" for w, n in g) for g in vocab[:40]]
    md += ["", "## Showing the words", "", "| axis | mean change | rank correlation |", "|---|---|---|"]
    md += [f"| {a} | {v['mean_abs_change']} | {v['spearman']} |" for a, v in comparison.items()]
    (out / "clusters.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
