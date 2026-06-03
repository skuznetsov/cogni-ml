#!/usr/bin/env python3
"""Probe second-stage rerank codecs on Nomic vectors and Hadamard shortlists.

This is an offline diagnostic. It answers whether candidate misses come from the
packed shortlist or from the compressed rerank representation.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class ShortlistMethod(Protocol):
    name: str

    def fit(self, base: np.ndarray) -> None: ...
    def search(self, query: np.ndarray, k: int) -> np.ndarray: ...
    def bytes_per_vec(self) -> float: ...


def load_clustered_pg_eval(root: Path):
    path = root / "scripts" / "bench_turboquant_retrieval.py"
    if not path.exists():
        raise SystemExit(f"clustered_pg evaluator not found: {path}")
    spec = importlib.util.spec_from_file_location("clustered_pg_turboquant_eval", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= scores.size:
        return np.argsort(scores)[::-1]
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(scores[idx])[::-1]]


def exact_gt(base: np.ndarray, queries: np.ndarray, k: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    ids: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for q in queries:
        s = base @ q
        idx = topk(s, k)
        ids.append(idx)
        scores.append(s[idx])
    return ids, scores


class RerankCodec:
    name = "base"
    bytes_per_vec = 0.0

    def fit(self, base_normalized: np.ndarray) -> None:
        raise NotImplementedError

    def scores(self, candidate_ids: np.ndarray, query_normalized: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Fp32Rerank(RerankCodec):
    name = "fp32"

    def fit(self, base_normalized: np.ndarray) -> None:
        self.base = base_normalized.astype(np.float32, copy=True)
        self.bytes_per_vec = float(self.base.shape[1] * 4)

    def scores(self, candidate_ids: np.ndarray, query_normalized: np.ndarray) -> np.ndarray:
        return self.base[candidate_ids] @ query_normalized


class Fp16Rerank(RerankCodec):
    name = "fp16"

    def fit(self, base_normalized: np.ndarray) -> None:
        self.codes = base_normalized.astype(np.float16)
        self.bytes_per_vec = float(self.codes.shape[1] * 2)

    def scores(self, candidate_ids: np.ndarray, query_normalized: np.ndarray) -> np.ndarray:
        return self.codes[candidate_ids].astype(np.float32) @ query_normalized


class DimSq8Rerank(RerankCodec):
    name = "dim_sq8"

    def fit(self, base_normalized: np.ndarray) -> None:
        col_min = base_normalized.min(axis=0).astype(np.float32)
        col_max = base_normalized.max(axis=0).astype(np.float32)
        col_range = np.maximum(col_max - col_min, 1e-8)
        self.codes = np.clip(np.rint((base_normalized - col_min) / col_range * 255.0), 0, 255).astype(np.uint8)
        self.mins = col_min
        self.scales = col_range / 255.0
        self.bytes_per_vec = float(self.codes.shape[1])

    def scores(self, candidate_ids: np.ndarray, query_normalized: np.ndarray) -> np.ndarray:
        decoded = self.codes[candidate_ids].astype(np.float32) * self.scales + self.mins
        return decoded @ query_normalized


class RowSymI8Rerank(RerankCodec):
    name = "row_i8"

    def fit(self, base_normalized: np.ndarray) -> None:
        max_abs = np.maximum(np.max(np.abs(base_normalized), axis=1), 1e-8).astype(np.float32)
        scales = max_abs / 127.0
        self.codes = np.clip(np.rint(base_normalized / scales[:, None]), -127, 127).astype(np.int8)
        self.scales = scales
        self.bytes_per_vec = float(self.codes.shape[1] + 4)

    def scores(self, candidate_ids: np.ndarray, query_normalized: np.ndarray) -> np.ndarray:
        decoded = self.codes[candidate_ids].astype(np.float32) * self.scales[candidate_ids, None]
        return decoded @ query_normalized


class BlockSymI8Rerank(RerankCodec):
    def __init__(self, block: int) -> None:
        self.block = block
        self.name = f"block{block}_i8"

    def fit(self, base_normalized: np.ndarray) -> None:
        n, dim = base_normalized.shape
        padded = ((dim + self.block - 1) // self.block) * self.block
        work = np.zeros((n, padded), dtype=np.float32)
        work[:, :dim] = base_normalized
        grouped = work.reshape(n, padded // self.block, self.block)
        max_abs = np.maximum(np.max(np.abs(grouped), axis=2), 1e-8).astype(np.float32)
        scales = max_abs / 127.0
        codes = np.clip(np.rint(grouped / scales[:, :, None]), -127, 127).astype(np.int8)
        self.codes = codes.reshape(n, padded)[:, :dim]
        self.scales = scales
        self.dim = dim
        self.bytes_per_vec = float(dim + scales.shape[1] * 4)

    def scores(self, candidate_ids: np.ndarray, query_normalized: np.ndarray) -> np.ndarray:
        codes = self.codes[candidate_ids].astype(np.float32)
        expanded = np.repeat(self.scales[candidate_ids], self.block, axis=1)[:, : self.dim]
        decoded = codes * expanded
        return decoded @ query_normalized


@dataclass
class ProbeRow:
    shortlist: str
    shortlist_m: int
    codec: str
    bytes_per_vec: float
    hit1: float
    recall: float
    shortlist_miss: int
    rerank_miss: int
    avg_exact_margin_loss: float


def build_shortlist_method(module, name: str, seed: int) -> ShortlistMethod:
    if name == "block16":
        return module.TurboQuantBlock32PackedTopKMethod(4, seed, group_size=16)
    if name == "block32":
        return module.TurboQuantBlock32PackedTopKMethod(4, seed, group_size=32)
    if name == "higgs16":
        return module.TurboQuantBlockHiggs2PackedMethod(4, seed, group_size=16, grid_samples=20_000)
    if name == "higgs32":
        return module.TurboQuantBlockHiggs2PackedMethod(4, seed, group_size=32, grid_samples=20_000)
    raise SystemExit(f"unknown shortlist method {name!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--clustered-pg", type=Path, default=Path("/Users/sergey/Projects/C/clustered_pg"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--shortlists", default="block16,higgs16")
    ap.add_argument("--shortlist-ms", default="20,50,100")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fail-report", type=Path)
    args = ap.parse_args()

    module = load_clustered_pg_eval(args.clustered_pg)
    base = normalize_rows(np.load(args.base).astype(np.float32))
    queries = normalize_rows(np.load(args.queries).astype(np.float32))
    if base.ndim != 2 or queries.ndim != 2 or base.shape[1] != queries.shape[1]:
        raise SystemExit(f"shape mismatch: base={base.shape} queries={queries.shape}")

    gt_ids, gt_scores = exact_gt(base, queries, args.k)
    codecs: list[RerankCodec] = [
        Fp32Rerank(),
        Fp16Rerank(),
        DimSq8Rerank(),
        RowSymI8Rerank(),
        BlockSymI8Rerank(16),
        BlockSymI8Rerank(32),
        BlockSymI8Rerank(64),
    ]
    for codec in codecs:
        codec.fit(base)

    print("shortlist\tM\tcodec\tbytes_per_vec\thit1\trecall@k\tshortlist_miss\trerank_miss\tavg_exact_margin_loss")
    report_lines = ["query\tshortlist\tM\tcodec\texact_top1\tchosen\texact_top1_score\tchosen_exact_score\tmargin_loss\tshortlist_has_exact"]

    for sl_name in [x.strip() for x in args.shortlists.split(",") if x.strip()]:
        method = build_shortlist_method(module, sl_name, args.seed)
        method.fit(base)
        max_m = max(int(x) for x in args.shortlist_ms.split(",") if x.strip())
        all_shortlists = [method.search(q, max_m) for q in queries]
        for m in [int(x) for x in args.shortlist_ms.split(",") if x.strip()]:
            shortlists = [ids[:m] for ids in all_shortlists]
            shortlist_miss = sum(1 for ids, gt in zip(shortlists, gt_ids) if gt[0] not in set(ids.tolist()))
            for codec in codecs:
                hit1 = 0
                recall_total = 0.0
                rerank_miss = 0
                margin_losses: list[float] = []
                for qi, (ids, gt, gt_score_row) in enumerate(zip(shortlists, gt_ids, gt_scores)):
                    scores = codec.scores(ids, queries[qi])
                    chosen_local = topk(scores, min(args.k, ids.size))
                    found = ids[chosen_local]
                    exact_top1 = int(gt[0])
                    chosen = int(found[0])
                    if chosen == exact_top1:
                        hit1 += 1
                    else:
                        rerank_miss += 1
                        exact_scores = base[[exact_top1, chosen]] @ queries[qi]
                        loss = float(exact_scores[0] - exact_scores[1])
                        margin_losses.append(loss)
                        if args.fail_report:
                            report_lines.append(
                                f"{qi}\t{sl_name}\t{m}\t{codec.name}\t{exact_top1}\t{chosen}\t"
                                f"{float(exact_scores[0]):.9f}\t{float(exact_scores[1]):.9f}\t{loss:.9f}\t"
                                f"{1 if exact_top1 in set(ids.tolist()) else 0}"
                            )
                    recall_total += len(set(found.tolist()) & set(gt.tolist())) / float(args.k)
                n = len(queries)
                row = ProbeRow(
                    shortlist=sl_name,
                    shortlist_m=m,
                    codec=codec.name,
                    bytes_per_vec=codec.bytes_per_vec,
                    hit1=100.0 * hit1 / n,
                    recall=100.0 * recall_total / n,
                    shortlist_miss=shortlist_miss,
                    rerank_miss=rerank_miss,
                    avg_exact_margin_loss=float(np.mean(margin_losses)) if margin_losses else 0.0,
                )
                print(
                    f"{row.shortlist}\t{row.shortlist_m}\t{row.codec}\t{row.bytes_per_vec:.1f}\t"
                    f"{row.hit1:.2f}\t{row.recall:.2f}\t{row.shortlist_miss}\t{row.rerank_miss}\t"
                    f"{row.avg_exact_margin_loss:.9f}"
                )

    if args.fail_report:
        args.fail_report.write_text("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
