"""Tests for clean.py's pure helpers. Run with `python3 -m unittest discover -s jobs`."""

import unittest

import clean


class BaselineTest(unittest.TestCase):
    def test_a_game_recorded_differently_gets_its_own_centre(self):
        # Two "games": the same voice through two different mixes.
        voice = {f"a{i}": [1.0, 0.1 * (i % 3), 0.0] for i in range(25)}
        voice |= {f"b{i}": [0.6, 0.8, 0.1 * (i % 3)] for i in range(25)}
        judged = set(voice)
        everyone = clean.baseline(voice, judged, list(voice), mads=4)
        game_b = clean.baseline(voice, judged, [k for k in voice if k.startswith("b")], mads=4)
        b = clean.unit(voice["b0"])
        self.assertGreater(clean.dot(b, game_b["centre"]), clean.dot(b, everyone["centre"]))
        self.assertEqual(game_b["references"], 25)

    def test_unjudged_clips_dont_set_the_centre_when_judged_ones_exist(self):
        voice = {"long": [1.0, 0.0], "sigh": [0.0, 1.0]}
        b = clean.baseline(voice, {"long"}, ["long", "sigh"], mads=4)
        self.assertEqual(b["references"], 1)
        self.assertAlmostEqual(clean.dot(clean.unit(voice["long"]), b["centre"]), 1.0)


class WordsTest(unittest.TestCase):
    def test_stage_notes_and_punctuation_are_ignored(self):
        self.assertEqual(clean.words("{Sighs} Keelah se'lai, Shepard!"), ["keelah", "se'lai", "shepard"])

    def test_wer(self):
        self.assertAlmostEqual(clean.wer(["a", "b", "c"], ["a", "x", "c", "d"]), 2 / 3)
        self.assertEqual(clean.wer([], []), 0.0)


if __name__ == "__main__":
    unittest.main()
