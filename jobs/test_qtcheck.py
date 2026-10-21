import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

import qtcheck


class DumpTest(unittest.TestCase):
    def test_reads_qwentts_dump_format(self):
        data = np.arange(6, dtype="<f4").reshape(2, 3)
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "x.bin"
            f.write_bytes(struct.pack("<i", 2) + struct.pack("<ii", 2, 3) + data.tobytes())
            got = qtcheck.read_dump(f)
        self.assertEqual(got.shape, (2, 3))
        self.assertTrue(np.array_equal(got, data))

    def test_cosine_and_shift(self):
        self.assertAlmostEqual(qtcheck.cosine([1, 2], [2, 4]), 1.0)
        x = [3.0, 4.0, 0.0, 0.0]
        d = np.array(qtcheck.shifted(x)) - np.array(x)
        self.assertAlmostEqual(float(np.linalg.norm(d)), 0.5 * 5.0, places=5)
        self.assertEqual(qtcheck.shifted(x), qtcheck.shifted(x))  # fixed seed


if __name__ == "__main__":
    unittest.main()
