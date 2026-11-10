from unittest import TestCase

import numpy as np

from codec import FILTERS, HEADER, MAGIC, decode, encode


class CodecTests(TestCase):
    def test_every_strategy_is_exact_for_grayscale_and_rgb(self):
        rng = np.random.default_rng(42)
        for shape in [(1, 1), (7, 9), (9, 1, 3), (7, 9, 3)]:
            data = rng.integers(0, 256, size=shape, dtype=np.uint8)
            for strategy in FILTERS:
                with self.subTest(shape=shape, strategy=strategy):
                    np.testing.assert_array_equal(decode(encode(data, strategy)), data)

    def test_rejects_truncated_oversized_and_concatenated_streams(self):
        encoded = encode(np.zeros((7, 9), dtype=np.uint8))
        for invalid in (encoded[:-1], encoded + b"x", HEADER.pack(MAGIC, 99999, 99999, 3, 0)):
            with self.assertRaises(ValueError):
                decode(invalid)

    def test_rejects_pixel_checksum_mismatch(self):
        encoded = bytearray(encode(np.zeros((7, 9), dtype=np.uint8)))
        encoded[HEADER.size - 1] ^= 1
        with self.assertRaisesRegex(ValueError, "checksum"):
            decode(encoded)
