# Delta Raster

A lossless grayscale and RGB codec for comparing row predictors. It stores byte residuals with DEFLATE compression and verifies the decoded pixels with CRC32.

```sh
python -m pip install -r requirements.txt
python codec.py encode input.png packed.dlt --filter adaptive
python codec.py decode packed.dlt recovered.png
python -m unittest -v
python benchmark.py
```

Use Python 3.13. The codec preserves decoded 8-bit pixels. File-level metadata such as EXIF, color profiles, and text chunks is outside the format.

## Predictor benchmark

Sizes are bytes at compression level 6. Every custom variant was decoded and checked against the original pixel array. See [benchmark-results.json](benchmark-results.json) for local encode times.

| Input | None | Sub | Up | Adaptive | Pillow PNG |
| --- | ---: | ---: | ---: | ---: | ---: |
| China sample | 539,801 | 497,628 | 500,220 | 491,161 | 490,868 |
| Flower sample | 516,762 | 383,553 | 366,191 | 367,421 | 365,079 |
| Horizontal gradient | 640 | 522 | 866 | 524 | 464 |
| Uniform noise | 65,840 | 65,840 | 65,840 | 65,840 | 65,917 |

Adaptive filtering saved 9.0% over no filter on the China sample. On the flower sample, a fixed vertical predictor beat the adaptive heuristic. Uniform noise did not compress below its 65,536 raw pixel bytes.

## Format

The header holds a format marker, width, height, channel count, and pixel checksum. The compressed stream contains one filter byte per row followed by that row's residuals. Sub subtracts the prior pixel in the same channel. Up subtracts the pixel above. Adaptive selects the filter with the smallest sum of absolute signed byte residuals. All predictor arithmetic wraps modulo 256.

The decoder validates the declared dimensions, limits decompressed data to 64 MiB, and rejects missing or trailing compressed bytes. The format is inspired by the [PNG filtering specification](https://www.w3.org/TR/PNG/#9Filters). The natural images come from [scikit-learn's bundled sample images](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_sample_image.html).
