"""Label-blind buildset tests: stdlib only, synthetic rows, no game data."""

import unittest

import buildset


def line(i, game="le2", conversation="c0", text="hello", flags=None):
    return {"id": i, "source": {"game": game, "conversation": conversation},
            "text": text, "flags": flags}


def clean(i, keep=True, flags=None, duration=2.0):
    return {"id": i, "keep": keep, "flags": flags,
            "audio": {"duration_s": duration, "path": f"audio/{i}.flac"}}


def rows(n_convs, clips_per_conv, game="le2", prefix="c"):
    """Synthetic rows in the built schema, one `conversation` per group."""
    out = []
    for k in range(n_convs):
        cid = f"{prefix}{k}"
        for j in range(clips_per_conv):
            out.append({"id": f"{prefix}{k}_{j}", "audio": f"wav/{prefix}{k}_{j}.wav",
                        "text": "hello", "ref_audio": f"wav/{prefix}{k}_{j}.wav",
                        "game": game, "language": "English", "duration_s": 2.0,
                        "conversation": cid})
    return out


class DropFilterTests(unittest.TestCase):
    def _apply(self, lines, clean_=None, effect=(), excl=()):
        return buildset.drop_filters(lines, clean_ or {}, set(effect), set(excl))

    def test_le1_dropped(self):
        kept, why = self._apply({"a": line("a", game="le1")})
        self.assertEqual(kept, [])
        self.assertEqual(why["le1"], 1)

    def test_clean_missing_dropped(self):
        kept, why = self._apply({"a": line("a")})
        self.assertEqual(kept, [])
        self.assertEqual(why["clean dropped/flagged"], 1)

    def test_clean_keep_false_dropped(self):
        kept, why = self._apply({"a": line("a")}, {"a": clean("a", keep=False)})
        self.assertEqual(kept, [])
        self.assertEqual(why["clean dropped/flagged"], 1)

    def test_clean_bad_flag_dropped(self):
        kept, why = self._apply({"a": line("a")}, {"a": clean("a", flags=["far_voice"])})
        self.assertEqual(kept, [])
        self.assertEqual(why["clean dropped/flagged"], 1)

    def test_harvest_bark_dropped(self):
        kept, why = self._apply({"a": line("a", flags=["bark"])}, {"a": clean("a")})
        self.assertEqual(kept, [])
        self.assertEqual(why["bark/nonverbal/no subtitle"], 1)

    def test_effect_clip_dropped(self):
        kept, why = self._apply({"a": line("a")}, {"a": clean("a")}, effect=["a"])
        self.assertEqual(kept, [])
        self.assertEqual(why["effect clip"], 1)

    def test_refscreen_exclude_dropped(self):
        kept, why = self._apply({"a": line("a")}, {"a": clean("a")}, excl=["a"])
        self.assertEqual(kept, [])
        self.assertEqual(why["refscreen exclude"], 1)

    def test_interjection_dropped(self):
        kept, why = self._apply({"a": line("a", text="Aaah!")}, {"a": clean("a")})
        self.assertEqual(kept, [])
        self.assertEqual(why["scream/pain/interjection"], 1)

    def test_pain_dropped(self):
        kept, why = self._apply({"a": line("a", text="I'm on fire!")}, {"a": clean("a")})
        self.assertEqual(kept, [])
        self.assertEqual(why["scream/pain/interjection"], 1)

    def test_empty_text_dropped(self):
        kept, why = self._apply({"a": line("a", text="")}, {"a": clean("a")})
        self.assertEqual(kept, [])
        self.assertEqual(why["scream/pain/interjection"], 1)

    def test_length_dropped(self):
        kept, why = self._apply({"a": line("a")}, {"a": clean("a", duration=0.5)})
        self.assertEqual(kept, [])
        self.assertEqual(why["length outside 0.8-15 s"], 1)
        kept, why = self._apply({"b": line("b")}, {"b": clean("b", duration=15.5)})
        self.assertEqual(kept, [])
        self.assertEqual(why["length outside 0.8-15 s"], 1)

    def test_kept(self):
        kept, why = self._apply({"a": line("a")}, {"a": clean("a")})
        self.assertEqual(why, {})
        self.assertEqual(len(kept), 1)
        r = kept[0]
        self.assertEqual(r["id"], "a")
        self.assertEqual(r["audio"], "wav/a.wav")
        self.assertEqual(r["ref_audio"], "wav/a.wav")
        self.assertEqual(r["game"], "le2")
        self.assertEqual(r["language"], "English")
        self.assertEqual(r["duration_s"], 2.0)
        self.assertEqual(r["conversation"], "c0")


class SplitTests(unittest.TestCase):
    def test_deterministic(self):
        data = rows(100, 3)
        t1, v1 = buildset.split_conversations(data)
        t2, v2 = buildset.split_conversations(data)
        self.assertEqual([r["id"] for r in t1], [r["id"] for r in t2])
        self.assertEqual([r["id"] for r in v1], [r["id"] for r in v2])

    def test_whole_conversations_stay_together(self):
        data = rows(100, 3)
        train, val = buildset.split_conversations(data)
        train_convs = {r["conversation"] for r in train}
        val_convs = {r["conversation"] for r in val}
        self.assertTrue(train_convs.isdisjoint(val_convs))
        self.assertEqual(len(train_convs) + len(val_convs), 100)

    def test_roughly_ten_percent_per_game(self):
        data = rows(100, 3, game="le2", prefix="c") + rows(100, 3, game="le3", prefix="d")
        train, val = buildset.split_conversations(data)
        self.assertEqual(len(train) + len(val), len(data))
        for game in ("le2", "le3"):
            val_convs = {r["conversation"] for r in val if r["game"] == game}
            # round(100 * 0.10) == 10 conversations per game.
            self.assertEqual(len(val_convs), 10)

    def test_label_blind(self):
        # Extra mood-like fields and a fake "rated" flag must not change the
        # split: the builder reads only game and conversation.
        base = rows(50, 2)
        extra = [dict(r, mood="angry", rated=True) for r in base]
        t1, v1 = buildset.split_conversations(base)
        t2, v2 = buildset.split_conversations(extra)
        self.assertEqual([r["id"] for r in t1], [r["id"] for r in t2])
        self.assertEqual([r["id"] for r in v1], [r["id"] for r in v2])

    def test_val100_deterministic_and_sized(self):
        val = rows(50, 3)  # 150 clips -> sample caps at 100
        sample = buildset.sample_val100(val)
        sample2 = buildset.sample_val100(val)
        self.assertEqual(len(sample), 100)
        self.assertEqual([r["id"] for r in sample], [r["id"] for r in sample2])

    def test_val100_under_size_returns_all(self):
        small = rows(3, 1)
        self.assertEqual(len(buildset.sample_val100(small)), 3)


if __name__ == "__main__":
    unittest.main()
