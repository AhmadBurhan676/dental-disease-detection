"""
Stream unpacker for DENTEX disease training xrays from curl range stream.
Reads local ZIP entries from stdin and saves decompressed PNGs directly to disk.
"""

import os
import sys
import struct
import zlib
from pathlib import Path
from tqdm import tqdm

OUT_DIR = Path("data/dentex/training/training_data/quadrant-enumeration-disease/xrays")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_EXPECTED = 705
saved_count = 0

print(f"[Extractor] Streaming and extracting {TOTAL_EXPECTED} disease xrays into {OUT_DIR}...")

stream = sys.stdin.buffer

def read_exact(n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)

with tqdm(total=TOTAL_EXPECTED, desc="Extracted images") as pbar:
    while saved_count < TOTAL_EXPECTED:
        sig = read_exact(4)
        if not sig or sig != b'PK\x03\x04':
            # Check if reached EOF or central directory
            if sig.startswith(b'PK\x01\x02') or not sig:
                break
            # Align if any padding
            break

        header = read_exact(26)
        if len(header) < 26:
            break

        comp_method = struct.unpack('<H', header[4:6])[0]
        comp_size = struct.unpack('<I', header[14:18])[0]
        uncomp_size = struct.unpack('<I', header[18:22])[0]
        fn_len, extra_len = struct.unpack('<HH', header[22:26])

        fname_bytes = read_exact(fn_len)
        fname = fname_bytes.decode('utf-8', errors='ignore')

        if extra_len > 0:
            read_exact(extra_len)

        compressed_data = read_exact(comp_size)

        if 'quadrant-enumeration-disease/xrays' in fname and not fname.endswith('/'):
            basename = Path(fname).name
            if comp_method == 8:
                raw_data = zlib.decompress(compressed_data, -15)
            else:
                raw_data = compressed_data

            dest_path = OUT_DIR / basename
            dest_path.write_bytes(raw_data)
            saved_count += 1
            pbar.update(1)

print(f"[Extractor] Successfully extracted {saved_count} images to {OUT_DIR}.")
