"""Qualify the frozen T0 block-0 cache store and reader on MPS."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import hy_nomas_reference_block0_compare as artifact_io
import hy_nomas_reference_history as history
import hy_nomas_reference_preblock_compare as preblock


ATOL = RTOL = 2e-5
IMAGE_ROWS = 4320
STEMS = ("k", "v")
NAMES = tuple(f"history.{stem}_bf16" for stem in STEMS)
READ_NAMES = tuple(f"history.{stem}_read_f32" for stem in STEMS)
PREBLOCK_NAMES = (
    "reference.latent_bf16", "cache.noisy_latent_f32", "block.img_f32",
    "block.txt_valid_f32", "block.vec_f32", "block.vec_txt_f32",
    "block.rope_cos_f32", "block.rope_sin_f32", "cache.timestep_f32",
)


def pinned_json(path, sha256):
    path = Path(path)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError(f"JSON digest differs from pin: {path}")
    return json.loads(payload, object_pairs_hook=artifact_io._unique_object)


def artifact_for(receipt_path, receipt, names):
    name = receipt["artifact"]
    if Path(name).name != name:
        raise ValueError("history artifact must be a sibling of its receipt")
    return artifact_io.TensorArtifact(
        Path(receipt_path).parent / name, expected_sha256=receipt["artifact_sha256"],
        expected_names=names,
    )


def verify_refiner_pair(native_pin, source_pin, native, source):
    outputs = []
    for pin, receipt in ((native_pin, native), (source_pin, source)):
        if pin[1] != receipt["refiner_blocks_manifest_sha256"]:
            raise ValueError("history receipt does not bind the selected refiner capture")
        manifest = pinned_json(*pin)
        if (manifest["processed_tokens"] != 1280
                or manifest["valid_text_tokens"] != receipt["valid_text_tokens"]
                or manifest["blocks_executed"] != 2):
            raise ValueError("history refiner capture has different geometry")
        record = manifest["tensors"]["block1.output"]
        if record["shape"] != [1, 1280, 2048] or record["dtype"] != "F32" or record["nbytes"] != 1280 * 2048 * 4:
            raise ValueError("refiner capture lacks the complete second-block output")
        payload = manifest["tensor_artifact"]["basename"] if receipt is native else manifest["payload"]
        if Path(payload).name != payload:
            raise ValueError("refiner payload must be a sibling of its manifest")
        with (Path(pin[0]).parent / payload).open("rb") as stream:
            stream.seek(record["offset"])
            actual = hashlib.sha256(stream.read(record["nbytes"])).hexdigest()
        if actual != record["sha256"]:
            raise ValueError("refiner block1 output bytes differ from the manifest")
        outputs.append(actual)
        if receipt is native and manifest["cache_fixture_manifest_sha256"] != native["preblock_manifest_sha256"]:
            raise ValueError("native refiner and native preblock receipts refer to different runs")
    if outputs[0] != outputs[1]:
        raise ValueError("refiner text including padding differs between history writers")
    return outputs[0]


def compare(native, source, shape):
    import numpy as np
    import torch

    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0" or not torch.backends.mps.is_available():
        raise ValueError("qualification requires MPS with CPU fallback disabled")

    def load(artifact, name, dtype):
        record = artifact.tensors[name]
        if record["shape"] != shape or record["dtype"] != dtype:
            raise ValueError(f"unexpected history tensor geometry: {name}")
        with artifact.path.open("rb") as stream:
            stream.seek(artifact.data_offset + record["offset"])
            payload = stream.read(record["nbytes"])
        artifact.check_identity()
        host = np.frombuffer(payload, dtype="<u2" if dtype == "BF16" else "<f4").copy()
        tensor = torch.from_numpy(host)
        if dtype == "BF16":
            tensor = tensor.view(torch.bfloat16)
        return tensor.to("mps").reshape(shape), hashlib.sha256(payload).hexdigest()

    records = {}

    def enqueue(name, actual, expected, *, exact=False):
        atol, rtol = (0.0, 0.0) if exact else (ATOL, RTOL)
        stats = preblock._metric_stat_tensors(torch, actual, expected, atol=atol, rtol=rtol)
        records[name] = (atol, rtol, tuple(stats.values()))

    hashes = {}
    with torch.inference_mode():
        for stem, stored, reader in zip(STEMS, NAMES, READ_NAMES, strict=True):
            native_store, native_sha = load(native, stored, "BF16")
            source_store, source_sha = load(source, stored, "BF16")
            native_read, _ = load(native, reader, "F32")
            native_f32, source_f32 = native_store.float(), source_store.float()
            hashes[stem] = {"native": native_sha, "source": source_sha, "equal": native_sha == source_sha}
            enqueue(f"{stem}.reader_vs_native_store", native_read, native_f32, exact=True)
            for region, rows in (("image", slice(0, IMAGE_ROWS)), ("text", slice(IMAGE_ROWS, None))):
                enqueue(f"{stem}.store.{region}", native_f32[:, rows], source_f32[:, rows])
                enqueue(f"{stem}.reader.{region}", native_read[:, rows], source_f32[:, rows])
        sample = source_f32[0, 0, 0, :16]
        enqueue("control.identity", sample, sample, exact=True)
        enqueue("control.perturbation", sample + 0.01, sample)
        # Read the GPU comparison only after every region and control is encoded.
        host_stats = torch.stack([value for _, _, stats in records.values() for value in stats])
        host_stats = host_stats.reshape(len(records), -1).cpu().tolist()

    rows = {
        name: preblock._stats_record(values, atol=atol, rtol=rtol)
        for (name, (atol, rtol, _)), values in zip(records.items(), host_stats, strict=True)
    }
    if not rows["control.identity"]["passed"] or rows["control.perturbation"]["passed"]:
        raise ValueError("history comparison sensitivity controls failed")
    return {
        "physical_sha256": hashes,
        "tensors": rows,
        "numeric_gate_passed": all(row["passed"] for name, row in rows.items() if not name.startswith("control.")),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", nargs=2, required=True, metavar=("RECEIPT", "SHA256"))
    parser.add_argument("--source", nargs=2, required=True, metavar=("RECEIPT", "SHA256"))
    parser.add_argument("--preblock", nargs=2, required=True, metavar=("COMPARISON", "SHA256"))
    parser.add_argument("--native-refiner", nargs=2, required=True, metavar=("MANIFEST", "SHA256"))
    parser.add_argument("--source-refiner", nargs=2, required=True, metavar=("MANIFEST", "SHA256"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native, source, inputs = (pinned_json(*pin) for pin in (args.native, args.source, args.preblock))
    if (native["schema"] != "cogni_ml.hy_nomas_connected_reference_history/v1"
            or source["schema"] != "cogni_ml.hy_nomas_connected_reference_history_source/v1"
            or native["precision"] != history.REFERENCE_BF16_PRECISION
            or native["weights_sha256"] != history.BLOCK_WEIGHTS_SHA256
            or source["upstream_revision"] != history.SOURCE_REVISION
            or type(source["mps_bf16_autocast"]) is not bool):
        raise ValueError("unexpected connected history execution contract")
    execution = ("encode_allocations", "encode_readbacks", "preterminal_waits", "terminal_waits")
    if tuple(native[key] for key in execution) != (0, 0, 0, 1):
        raise ValueError("native history changed the encoded allocation or wait contract")
    if (native["preblock_manifest_sha256"] != inputs["native"]["manifest_sha256"]
            or source["preblock_manifest_sha256"] != inputs["source"]["manifest_sha256"]
            or native["weights_sha256"] != source["weights_sha256"]
            or native["valid_text_tokens"] != source["valid_text_tokens"]
            or native["dit_text_tokens"] != source["dit_text_tokens"]
            or not inputs["sensitivity_controls"]["passed"]):
        raise ValueError("history receipts do not bind to the same effective inputs")
    image = "block.img_f32"
    input_rows = [row for row in inputs["raw_tensors"] if row["tensor"] != image]
    input_rows.extend(row for row in inputs["native_bf16_rounded_views"]["tensors"] if row["tensor"] == image)
    if (sorted(row["tensor"] for row in input_rows) != sorted(PREBLOCK_NAMES)
            or any(not row["passed"] or row["max_abs"] != 0 for row in input_rows)):
        raise ValueError("effective preblock input comparison was not exact")
    text_sha256 = verify_refiner_pair(args.native_refiner, args.source_refiner, native, source)
    active_rows = IMAGE_ROWS + native["valid_text_tokens"]
    physical_rows = IMAGE_ROWS + native["dit_text_tokens"]
    if (not IMAGE_ROWS < active_rows <= physical_rows
            or native["active_rows"] != active_rows or source["active_rows"] != active_rows
            or source["physical_rows"] != physical_rows
            or source["masked_rows"] != physical_rows - active_rows):
        raise ValueError("history active prefix differs")
    result = compare(
        artifact_for(args.native[0], native, (*NAMES, *READ_NAMES)),
        artifact_for(args.source[0], source, NAMES),
        (1, active_rows, 16, 128),
    )
    result.update(schema="cogni_ml.hy_nomas_connected_reference_history_compare/v1",
                  native_receipt_sha256=args.native[1], source_receipt_sha256=args.source[1],
                  source_mps_bf16_autocast=source["mps_bf16_autocast"],
                  preblock_comparison_sha256=args.preblock[1], full_refiner_text_sha256=text_sha256)
    preblock._publish_fresh(args.output, result)
    return 0 if result["numeric_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
