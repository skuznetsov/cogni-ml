#!/usr/bin/env python3
"""Convert `bin/nomic_embedding_export` TSV output to NumPy arrays.

Input TSV format:
  id<TAB>dim<TAB>norm<TAB>[comma-separated vector]

The script is intentionally small glue for offline retrieval/index experiments.
It does not generate embeddings and does not validate retrieval quality.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_vector(text: str, dim: int, row_num: int) -> np.ndarray:
    value = text.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    arr = np.fromstring(value, sep=",", dtype=np.float32)
    if arr.size != dim:
        raise SystemExit(f"row {row_num}: expected dim {dim}, parsed {arr.size}")
    if not np.isfinite(arr).all():
        raise SystemExit(f"row {row_num}: vector contains NaN or Inf")
    return arr


def read_export(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids: list[str] = []
    norms: list[float] = []
    vectors: list[np.ndarray] = []

    with path.open() as f:
        for row_num, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if row_num == 1 and parts[:3] == ["id", "dim", "norm"]:
                continue
            if len(parts) != 4:
                raise SystemExit(f"row {row_num}: expected 4 TSV columns, got {len(parts)}")
            item_id, dim_text, norm_text, vector_text = parts
            try:
                dim = int(dim_text)
                norm = float(norm_text)
            except ValueError as exc:
                raise SystemExit(f"row {row_num}: invalid dim/norm: {exc}") from exc
            ids.append(item_id)
            norms.append(norm)
            vectors.append(parse_vector(vector_text, dim, row_num))

    if not vectors:
        raise SystemExit("input TSV has no vector rows")

    dims = {vec.size for vec in vectors}
    if len(dims) != 1:
        raise SystemExit(f"mixed vector dims: {sorted(dims)}")
    return ids, np.asarray(norms, dtype=np.float32), np.vstack(vectors).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_tsv", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="Output .npy path for vectors")
    ap.add_argument("--ids-out", type=Path, help="Optional output path for one id per line")
    ap.add_argument("--norms-out", type=Path, help="Optional output .npy path for exported norms")
    args = ap.parse_args()

    ids, norms, vectors = read_export(args.input_tsv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, vectors)
    if args.ids_out:
        args.ids_out.parent.mkdir(parents=True, exist_ok=True)
        args.ids_out.write_text("\n".join(ids) + "\n")
    if args.norms_out:
        args.norms_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.norms_out, norms)
    print(
        f"rows={vectors.shape[0]} dim={vectors.shape[1]} "
        f"norm_min={float(norms.min()):.9f} norm_max={float(norms.max()):.9f} out={args.out}"
    )


if __name__ == "__main__":
    main()
