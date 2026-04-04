"""
split_zip.py
============
Splits a large zip file into chunks small enough to upload to Claude (< 90 MB each).
Each chunk is a valid zip file containing a subset of the original entries.

Usage:
    python split_zip.py praxis.zip
    python split_zip.py praxis.zip --chunk-mb 50
    python split_zip.py praxis.zip --output-dir chunks/ --chunk-mb 50

Output:
    praxis_part001.zip, praxis_part002.zip, ...
    praxis_manifest.txt  — lists which files are in which chunk

The chunks can be uploaded to Claude one at a time.
Claude reassembles context from the manifest + each chunk.
"""

import argparse
import os
import zipfile
from pathlib import Path


def split_zip(
    input_path: str,
    output_dir: str | None = None,
    chunk_mb: float = 50.0,
) -> list[str]:
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Not found: {input_path}")

    chunk_bytes = int(chunk_mb * 1024 * 1024)
    stem = input_path.stem

    out_dir = Path(output_dir) if output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Read all entries ----
    print(f"Reading {input_path.name} …")
    with zipfile.ZipFile(input_path, "r") as zf:
        all_entries = zf.infolist()
        total = len(all_entries)
        print(f"  {total} entries, "
              f"{input_path.stat().st_size / 1024 / 1024:.1f} MB total")

    # ---- Sort entries by size descending so big files are spread evenly ----
    all_entries.sort(key=lambda e: e.file_size, reverse=True)

    # ---- Bin-pack into chunks ----
    chunks: list[list[zipfile.ZipInfo]] = []
    chunk_sizes: list[int] = []

    for entry in all_entries:
        # Compressed size determines actual chunk file size
        entry_size = entry.compress_size or entry.file_size or 0

        # Try to add to existing chunk
        placed = False
        for i, (chunk, csize) in enumerate(zip(chunks, chunk_sizes)):
            if csize + entry_size <= chunk_bytes:
                chunk.append(entry)
                chunk_sizes[i] += entry_size
                placed = True
                break

        if not placed:
            chunks.append([entry])
            chunk_sizes.append(entry_size)

    print(f"  Splitting into {len(chunks)} chunks of ≤ {chunk_mb} MB each")

    # ---- Write chunks ----
    output_paths = []
    manifest_lines = [
        f"Source: {input_path.name}",
        f"Total entries: {total}",
        f"Chunks: {len(chunks)}",
        "",
    ]

    with zipfile.ZipFile(input_path, "r") as src_zf:
        for i, chunk_entries in enumerate(chunks, start=1):
            part_name = f"{stem}_part{i:03d}.zip"
            part_path = out_dir / part_name

            manifest_lines.append(f"=== {part_name} ({len(chunk_entries)} files) ===")

            with zipfile.ZipFile(part_path, "w", compression=zipfile.ZIP_DEFLATED) as dst_zf:
                for entry in chunk_entries:
                    data = src_zf.read(entry.filename)
                    dst_zf.writestr(entry, data)
                    manifest_lines.append(f"  {entry.filename}")

            size_mb = part_path.stat().st_size / 1024 / 1024
            print(f"  Wrote {part_name}  ({len(chunk_entries)} files, {size_mb:.1f} MB)")
            manifest_lines.append(f"  -> {size_mb:.1f} MB")
            manifest_lines.append("")
            output_paths.append(str(part_path))

    # ---- Write manifest ----
    manifest_path = out_dir / f"{stem}_manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
    print(f"\nManifest written to: {manifest_path.name}")
    print(f"\nDone. Upload each part zip to Claude separately.")
    print(f"Upload the manifest first so Claude knows the full structure.")

    return output_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a zip into uploadable chunks")
    parser.add_argument("zip_file", help="Path to the zip file to split")
    parser.add_argument(
        "--chunk-mb", type=float, default=50.0,
        help="Max size per chunk in MB (default: 50)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for output chunks (default: same as input)"
    )
    args = parser.parse_args()

    split_zip(
        input_path=args.zip_file,
        output_dir=args.output_dir,
        chunk_mb=args.chunk_mb,
    )
