#!/usr/bin/env python3
"""Stage true full-J FFN boundaries from a saved upstream attention replay.

Only the upstream block-zero projection/norm/MLP suffix is evaluated on MPS.
This is a source-method parity oracle, not an independent arithmetic backend.
No attention or full decoder/generation is executed. Native timing is separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

SCHEMA = "cogni_ml.pixelumm_ffn_isolated_reference/v1"
UPSTREAM_REVISION = "911cea6458a89c2663513ff7099561ec190dd491"
MODEL_SHA = "bea01a8490f8e9a4076da83d4f9fbb3cba7bd6701df5c28fb2e2c04f00131eec"
SUBSET_SHA = "ae82bb41c121a955493f5b1502dc512ac8aaba05a6cec67e98edeaedfe720485"
COMPARISON = {"atol": 2e-5, "rtol": 2e-5}
WEIGHTS = {
    "attn.o_proj.weight": [4096, 4096],
    "post_attn_norm.weight": [4096],
    "mlp.gate_proj.weight": [12288, 4096],
    "mlp.up_proj.weight": [12288, 4096],
    "mlp.down_proj.weight": [4096, 12288],
}


class FFNReferenceError(ValueError):
    pass


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_parent(document: dict) -> tuple[int, dict]:
    """Validate the selected typed corridor; do not load unrelated 36-block data."""
    if (document.get("upstream_revision") != UPSTREAM_REVISION
            or document.get("upstream_model_sha256") != MODEL_SHA
            or document.get("comparison") != COMPARISON
            or document.get("token_order") != "tile_major"
            or document.get("tile_tokens") != 8):
        raise FFNReferenceError("parent source, tolerance or token layout mismatch")
    grid = document.get("grid")
    if grid not in ([2, 32, 32], [6, 44, 80]):
        raise FFNReferenceError("unsupported parent grid")
    count = math.prod(grid)
    if count == 2048:
        if (document.get("schema") != "cogni_ml.pixelumm_block_reference/v1"
                or document.get("checkpoint_subset_sha256") != SUBSET_SHA):
            raise FFNReferenceError("small block-zero source identity mismatch")
        prefix = ""
    else:
        blocks = document.get("blocks")
        if (document.get("schema") != "cogni_ml.pixelumm_thirty_six_block_reference/v1"
                or document.get("runtime_dtype") != "BF16"
                or document.get("input_provenance", {}).get("kind") != "real_latent_xt_pixelumm_stem"
                or document.get("full36_capture_policy") != "two_bank_survivors"
                or not isinstance(blocks, list) or len(blocks) != 36
                or blocks[0].get("index") != 0
                or blocks[0].get("checkpoint_subset_sha256") != SUBSET_SHA):
            raise FFNReferenceError("full-J block-zero source identity mismatch")
        prefix = "block0."
    payload_bytes = document.get("payload_bytes")
    if (type(payload_bytes) is not int or not 0 < payload_bytes <= 32 * 1024**3
            or not _digest(document.get("payload_sha256"))):
        raise FFNReferenceError("parent payload identity mismatch")
    records = {
        "x": (document["inputs"]["x"], [1, count, 4096]),
        "attention": (document["captures"][prefix + "attention"], [1, count, 32, 128]),
        **{name: (document["weights"][prefix + name], shape) for name, shape in WEIGHTS.items()},
    }
    cursor = 0
    for name, (record, shape) in sorted(records.items(), key=lambda item: item[1][0].get("offset", -1)):
        offset, nbytes = record.get("offset"), record.get("nbytes")
        if (record.get("dtype") != "BF16" or record.get("shape") != shape
                or type(offset) is not int or type(nbytes) is not int
                or offset < cursor or nbytes != math.prod(shape) * 2
                or offset > payload_bytes or nbytes > payload_bytes - offset
                or not _digest(record.get("sha256"))):
            raise FFNReferenceError(f"parent record shape, extent or digest mismatch: {name}")
        cursor = offset + nbytes
    return count, {name: record for name, (record, _) in records.items()}


def load_parent(path: Path, expected_sha: str):
    path = path.absolute()
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 1024**2:
        raise FFNReferenceError("parent must be a bounded regular metadata file")
    raw = path.read_bytes()
    if not _digest(expected_sha) or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise FFNReferenceError("parent trusted metadata digest mismatch")
    document = json.loads(raw)
    count, records = validate_parent(document)
    basename = document.get("payload")
    if (not isinstance(basename, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", basename)
            or basename in {".", ".."}):
        raise FFNReferenceError("parent payload must be a sibling basename")
    payload = path.parent / basename
    if not payload.is_file() or payload.is_symlink() or payload.stat().st_size != document["payload_bytes"]:
        raise FFNReferenceError("parent payload is missing or has a changed extent")
    return document, count, records, payload


def make_fixture(*, reference: Path, reference_sha256: str, upstream: Path, output: Path,
                 include_output: bool = False):
    document, count, records, payload = load_parent(reference, reference_sha256)
    output = output.absolute()
    target_payload = output.with_suffix(".bin")
    if (not str(output.resolve()).startswith("/Volumes/")
            or not output.parent.is_dir() or output.suffix != ".json"
            or output.exists() or output.is_symlink()
            or target_payload.exists() or target_payload.is_symlink()):
        raise FFNReferenceError("output must be a fresh external-SSD JSON/payload pair")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise FFNReferenceError("MPS CPU fallback is forbidden")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import torch
    import hy_nomas_pixelumm_reference as block
    if not torch.backends.mps.is_available():
        raise FFNReferenceError("MPS is required")
    source, model_sha = block.load_upstream(upstream)
    if model_sha != MODEL_SHA:
        raise FFNReferenceError("pinned model source digest mismatch")
    before = payload.stat()

    def tensor(name):
        record = records[name]
        with payload.open("rb") as stream:
            stream.seek(record["offset"])
            data = bytearray(stream.read(record["nbytes"]))
        if len(data) != record["nbytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise FFNReferenceError(f"selected parent tensor digest mismatch: {name}")
        return torch.frombuffer(data, dtype=torch.bfloat16).reshape(record["shape"]).to("mps")

    with torch.device("meta"):
        layer = source.GenLayer(source.QWEN3_8B_GEN)
    state = {name: tensor(name) for name in WEIGHTS}
    loaded = layer.load_state_dict(state, strict=False, assign=True)
    if set(loaded.missing_keys) != set(layer.state_dict()) - set(WEIGHTS) or loaded.unexpected_keys:
        raise FFNReferenceError("upstream suffix parameter assignment mismatch")
    layer.eval()
    layer.requires_grad_(False)
    x, attention = tensor("x"), tensor("attention")
    with torch.inference_mode(), torch.autocast("mps", dtype=torch.bfloat16):
        projected = layer.attn.o_proj(attention.reshape(1, count, 4096))
        residual = x + projected
        normalized = layer.post_attn_norm(residual)
        gate = layer.mlp.gate_proj(normalized)
        up = layer.mlp.up_proj(normalized)
        product = torch.nn.functional.silu(gate) * up
        down = layer.mlp.down_proj(product)
        # GenLayer.forward's final residual, after the same BF16 MLP boundary.
        final_output = residual + down if include_output else None
    tensors = {
        "inputs": {"post_attn_norm": normalized, "product": product},
        "weights": {name: state[name] for name in WEIGHTS if name.startswith("mlp.")},
        "captures": {"gate_proj": gate, "up_proj": up, "down_proj": down},
    }
    if include_output:
        tensors["captures"]["output"] = final_output
    flat = [value for group in tensors.values() for value in group.values()]
    if not all(value.device.type == "mps" and value.dtype == torch.bfloat16 and value.is_contiguous() for value in flat):
        raise FFNReferenceError("suffix tensors must remain contiguous BF16 on MPS")
    finite = torch.stack([torch.isfinite(value).all() for value in flat]).all()
    torch.mps.synchronize()  # Separate oracle staging; never in the native measured path.
    if not bool(finite.item()):
        raise FFNReferenceError("nonfinite upstream suffix boundary")
    del x, attention, projected, residual, layer
    after = payload.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise FFNReferenceError("parent payload changed during selected-record replay")
    result = {
        "schema": ("cogni_ml.pixelumm_ffn_suffix_reference/v1" if include_output else SCHEMA),
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_model_sha256": model_sha, "model_sha256": model_sha, "grid": document["grid"],
        "comparison": COMPARISON, "token_order": "tile_major", "tile_tokens": 8,
        "runtime_dtype": "BF16", "torch_version": str(torch.__version__),
        "source_replay": {
            "kind": "saved_block0_attention_upstream_suffix_mps",
            "parent_manifest_path": str(reference.resolve()),
            "parent_manifest_sha256": reference_sha256,
            "parent_schema": document["schema"],
            "parent_payload_sha256": document["payload_sha256"],
            "parent_payload_bytes": document["payload_bytes"],
            "selected_parent_records": records,
            "selected_record_hashes_verified": True,
            "parent_payload_full_hash_recomputed": False,
            "source_metadata_sha256": reference_sha256,
            "source_payload_sha256": document["payload_sha256"],
            "source_records": {
                "inputs.x": records["x"],
                "captures.block0.attention": records["attention"],
                **{"weights.block0." + name: records[name] for name in WEIGHTS},
            },
            "source_receipt_sha256": (
                document["blocks"][0]["source_receipt_sha256"] if count > 2048 else None
            ),
            "block0_subset_sha256": SUBSET_SHA,
            "parent_input_provenance": document.get("input_provenance"),
            "parent_block0": document.get("blocks", [None])[0],
            "scope": "source-method MPS suffix parity; saved locally adapted attention; not an independent backend",
            "cpu_fallback": False, "explicit_pre_staging_waits": 1,
        },
        "capture_order": list(tensors["captures"]),
        "payload": target_payload.name,
    }
    digest = hashlib.sha256()
    cursor = 0
    try:
        with target_payload.open("xb") as stream:
            for group, values in tensors.items():
                result[group] = {}
                for name, value in values.items():
                    data = value.detach().cpu().view(torch.uint16).numpy().tobytes()
                    record = {"dtype": "BF16", "shape": list(value.shape), "offset": cursor,
                              "nbytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                    result[group][name] = record
                    stream.write(data)
                    digest.update(data)
                    cursor += len(data)
                    del data
            stream.flush()
            os.fsync(stream.fileno())
        result.update(payload_bytes=cursor, payload_sha256=digest.hexdigest())
        with output.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        target_payload.unlink(missing_ok=True)
        raise
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-output", action="store_true",
                        help="Also stage GenLayer's final residual for connected-block comparisons")
    args = parser.parse_args()
    try:
        result = make_fixture(reference=args.reference, reference_sha256=args.reference_sha256,
                              upstream=args.upstream, output=args.output,
                              include_output=args.include_output)
    except (FFNReferenceError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"FFN reference rejected: {error}\n")
    print(json.dumps({"schema": result["schema"], "payload_bytes": result["payload_bytes"],
                      "output": str(args.output.resolve()), "sha256": sha_file(args.output)}))


if __name__ == "__main__":
    main()
