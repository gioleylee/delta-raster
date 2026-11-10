import io
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from sklearn.datasets import load_sample_image

from codec import FILTERS, decode, encode


def compare(pixels):
    entries = {}
    for strategy in FILTERS:
        start = perf_counter()
        encoded = encode(pixels, strategy)
        encode_ms = 1000 * (perf_counter() - start)
        np.testing.assert_array_equal(decode(encoded), pixels)
        entries[strategy] = {"bytes": len(encoded), "encode_ms": round(encode_ms, 2)}
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG", compress_level=6)
    entries["pillow_png"] = {"bytes": len(buffer.getvalue())}
    return {"raw_bytes": pixels.nbytes, "formats": entries}


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    gradient = np.broadcast_to(np.arange(256, dtype=np.uint8), (256, 256)).copy()
    report = {
        "china": compare(load_sample_image("china.jpg")),
        "flower": compare(load_sample_image("flower.jpg")),
        "gradient": compare(gradient),
        "uniform_noise": compare(rng.integers(0, 256, (256, 256), dtype=np.uint8)),
    }
    payload = json.dumps(report, indent=2) + "\n"
    Path("benchmark-results.json").write_text(payload, encoding="utf-8")
    print(payload, end="")
