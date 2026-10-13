"""Tests for label.py's helpers. Run with `python3 -m unittest discover -s jobs`."""

import unittest

import label


class ShortlistTest(unittest.TestCase):
    def test_top_and_a_control_from_the_lower_half(self):
        prelim = {f"c{i:02}": i / 100 for i in range(100)}
        s = label.shortlist(prelim, top=10, low=5)
        self.assertEqual(s["top"], [f"c{i:02}" for i in range(99, 89, -1)])
        self.assertEqual(len(s["low"]), 5)
        self.assertTrue(all(prelim[i] < 0.5 for i in s["low"]))
        self.assertEqual(s["all"], set(s["top"]) | set(s["low"]))

    def test_small_sets_dont_overlap(self):
        prelim = {"a": 0.9, "b": 0.5, "c": 0.1}
        s = label.shortlist(prelim, top=2, low=5)
        self.assertEqual(s["top"], ["a", "b"])
        self.assertEqual(s["low"], ["c"])


class RatingTest(unittest.TestCase):
    def test_ratings(self):
        self.assertEqual(label.rating({"rating": 4}), 4)
        self.assertEqual(label.rating({"rating": "2"}), 2)
        self.assertIsNone(label.rating({"rating": 9}))
        self.assertIsNone(label.rating(None))
        self.assertIsNone(label.rating({"score": 3}))


if __name__ == "__main__":
    unittest.main()
