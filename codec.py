"""A small lossless raster format for experimenting with row predictors."""

import argparse
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image


HEADER = struct.Struct(">4sIIBI")
MAGIC = b"DLT1"
MAX_FILTERED = 64 * 1024 * 1024
FILTERS = ("none", "sub", "up", "adaptive")


def dimensions(width, height, channels):
    if width < 1 or height < 1 or channels not in (1, 3):
        raise ValueError("expected positive dimensions and one or three channels")
    size = height * (1 + width * channels)
    if size > MAX_FILTERED:
        raise ValueError("decoded image exceeds 64 MiB")
    return size


def encode(pixels, strategy="adaptive", level=6):
    if pixels.dtype != np.uint8 or pixels.ndim not in (2, 3):
        raise ValueError("expected a uint8 grayscale or RGB array")
    if pixels.ndim == 3 and pixels.shape[2] != 3:
        raise ValueError("three-dimensional input must have three RGB channels")
    height, width = pixels.shape[:2]
    channels = pixels.shape[2] if pixels.ndim == 3 else 1
    dimensions(width, height, channels)
    if strategy not in FILTERS or not 0 <= level <= 9:
        raise ValueError("invalid filter or compression level")
    rows = pixels.reshape(height, width * channels)
    filtered = np.empty((height, width * channels + 1), dtype=np.uint8)
    previous = np.zeros(width * channels, dtype=np.uint8)
    for target, row in zip(filtered, rows):
        if strategy == "none":
            code, residual = 0, row
        else:
            sub = row.copy()
            sub[channels:] -= row[:-channels]
            up = row - previous
            options = (row, sub, up)
            if strategy == "adaptive":
                code = min(range(len(options)), key=lambda i: np.abs(options[i].view(np.int8).astype(np.int16)).sum())
            else:
                code = FILTERS.index(strategy)
            residual = options[code]
        target[0], target[1:] = code, residual
        previous = row
    crc = zlib.crc32(rows.tobytes())
    return HEADER.pack(MAGIC, width, height, channels, crc) + zlib.compress(filtered.tobytes(), level)


def decode(encoded):
    if len(encoded) < HEADER.size:
        raise ValueError("incomplete header")
    magic, width, height, channels, expected_crc = HEADER.unpack_from(encoded)
    if magic != MAGIC:
        raise ValueError("unrecognized raster format")
    expected_size = dimensions(width, height, channels)
    decompressor = zlib.decompressobj()
    try:
        filtered = decompressor.decompress(encoded[HEADER.size:], expected_size + 1)
    except zlib.error as error:
        raise ValueError("invalid compressed raster") from error
    if len(filtered) != expected_size or not decompressor.eof or decompressor.unused_data:
        raise ValueError("compressed raster has an invalid length or trailing data")
    data = np.frombuffer(filtered, dtype=np.uint8).reshape(height, width * channels + 1)
    result = np.empty((height, width * channels), dtype=np.uint8)
    previous = np.zeros(width * channels, dtype=np.uint8)
    for target, row in zip(result, data):
        code, residual = row[0], row[1:]
        if code == 0:
            target[:] = residual
        elif code == 1:
            target[:] = np.cumsum(residual.reshape(width, channels), axis=0, dtype=np.uint8).ravel()
        elif code == 2:
            target[:] = residual + previous
        else:
            raise ValueError("unknown filter code")
        previous = target
    if zlib.crc32(result) != expected_crc:
        raise ValueError("decoded pixel checksum mismatch")
    return result.reshape((height, width) if channels == 1 else (height, width, channels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("encode")
    pack.add_argument("--filter", choices=FILTERS, default="adaptive")
    unpack = commands.add_parser("decode")
    for command in (pack, unpack):
        command.add_argument("source", type=Path)
        command.add_argument("target", type=Path)
    args = parser.parse_args()
    try:
        if args.source.resolve() == args.target.resolve():
            raise ValueError("source and target must differ")
        if args.command == "encode":
            with Image.open(args.source) as image:
                if image.mode not in ("L", "RGB"):
                    raise ValueError("input must be 8-bit grayscale or RGB")
                dimensions(image.width, image.height, 1 if image.mode == "L" else 3)
                data = encode(np.asarray(image), args.filter)
            args.target.write_bytes(data)
        else:
            Image.fromarray(decode(args.source.read_bytes())).save(args.target, format="PNG")
    except (OSError, ValueError) as error:
        parser.error(str(error))
