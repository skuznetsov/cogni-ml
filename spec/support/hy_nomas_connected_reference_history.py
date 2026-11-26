"""Replay the pinned Hy-Nomas block-0 cache store on frozen source T0 inputs."""

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import hy_nomas_reference_cache as cache
import hy_nomas_reference_history as history
import hy_nomas_reference_preblock_compare as preblock
from hy_nomas_connected_reference_history_compare import pinned_json


DIT_TEXT_ROWS = 320
CACHE_STORAGE_SHA256 = "53fdb67aaa967b9195f17c1d829feb12cd1e17dfa21a4e5bc0a8492790f8f0c3"


def fixture(pin, schema):
    doc = pinned_json(*pin)
    if doc["schema"] != schema or Path(doc["payload"]).name != doc["payload"]:
        raise ValueError("unexpected source fixture schema or payload name")
    payload = Path(pin[0]).parent / doc["payload"]
    if (payload.is_symlink() or payload.stat().st_size != doc["payload_bytes"]
            or history._hash_file(payload) != doc["payload_sha256"]):
        raise ValueError("source fixture payload differs from the pinned manifest")
    return doc, payload


def upload(torch, fixture_pair, name, shape, dtype):
    doc, path = fixture_pair
    record = doc["tensors"][name]
    if record["dtype"] != "F32" or record["shape"] != list(shape):
        raise ValueError(f"unexpected source input geometry: {name}")
    with path.open("rb") as stream:
        stream.seek(record["offset"])
        raw = stream.read(record["nbytes"])
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError(f"source tensor bytes differ from the manifest: {name}")
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape).to("mps", dtype=dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source-preblock", nargs=2, required=True, metavar=("MANIFEST", "SHA256"))
    parser.add_argument("--source-refiner", nargs=2, required=True, metavar=("MANIFEST", "SHA256"))
    parser.add_argument("--bf16-autocast", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = args.output.with_suffix(".safetensors")
    if args.output.suffix != ".json" or any(
            path.exists() or path.is_symlink() for path in (args.output, artifact)):
        raise ValueError("output requires a fresh JSON receipt and sibling safetensors")

    source_preblock = fixture(args.source_preblock, "cogni_ml.hy_nomas_reference_preblock_source_trace/v1")
    source_refiner = fixture(args.source_refiner, "cogni_ml.hy_nomas_reference_refiner_blocks_source_trace/v1")
    valid = source_preblock[0]["valid_text_tokens"]
    if (not 0 < valid <= DIT_TEXT_ROWS or valid != source_refiner[0]["valid_text_tokens"]
            or source_refiner[0]["blocks_executed"] != 2):
        raise ValueError("unexpected frozen T0 source geometry")
    upstream = args.upstream.resolve(strict=True)
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != history.SOURCE_REVISION:
        raise ValueError("upstream revision differs from the pinned source")
    support = cache._load_support_oracle()
    for name, expected in {
        **support.PINNED_SOURCE_SHA256,
        "hymm/models/dit/modules/kv_cache_storage.py": CACHE_STORAGE_SHA256,
    }.items():
        if (upstream / name).is_symlink() or history._hash_file(upstream / name) != expected:
            raise ValueError(f"pinned upstream source differs: {name}")

    import torch
    from safetensors.torch import save

    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0" or not torch.backends.mps.is_available():
        raise ValueError("the source replay requires MPS with CPU fallback disabled")
    dtype = torch.bfloat16
    weights_sha = history.BLOCK_WEIGHTS_SHA256
    _, state, _ = cache._load_source_weights(torch, support, args.weights, weights_sha, parameter_dtype=dtype)
    helpers = support._install_upstream_modules(upstream)

    def unexpected_attention(*_args, **_kwargs):
        raise RuntimeError("unexpected non-cache attention in the T0 writer")

    block_type = support._extract_pinned_block(upstream, torch, unexpected_attention,
                                               helpers[-1].apply_rotary_emb, helpers)
    block = block_type(hidden_size=2048, heads_num=16, mlp_width_ratio=4.0, mlp_act_type="gelu_tanh",
                       attn_mode="magi_mp_causal", qk_norm=True, qk_norm_type="rms", qkv_bias=True,
                       dtype=dtype, device="meta")
    block.load_state_dict(state, strict=True, assign=True)
    block.to("mps").eval().requires_grad_(False)
    del state

    physical_rows = history.IMAGE_TOKENS + DIT_TEXT_ROWS
    active_rows = history.IMAGE_TOKENS + valid
    captures = {}

    def stop(query, key, value, **_kwargs):
        if captures or any(t.device.type != "mps" for t in (query, key, value)):
            raise RuntimeError("unexpected attention consumer")
        captures.update(k=key.transpose(1, 2), v=value.transpose(1, 2))
        raise history._AttentionConsumerIntercepted("source-T0")

    writer, identity = history._load_function_from_pinned_ast(
        upstream, support, torch, SimpleNamespace(scaled_dot_product_attention=stop))
    block.forward_with_cache.__globals__["causal_attention_with_cache_mp"] = writer
    physical_shape = (1, physical_rows, 16, 128)
    buffers = {name: torch.full(physical_shape, float("nan"), dtype=dtype, device="mps") for name in ("k", "v")}
    cache_mask = torch.zeros((1, physical_rows), dtype=torch.bool, device="mps")
    text_mask = (torch.arange(DIT_TEXT_ROWS, device="mps") < valid).view(1, -1)
    image = upload(torch, source_preblock, "block.img_f32", (1, history.IMAGE_TOKENS, 2048), dtype)
    text = upload(torch, source_refiner, "block1.output", (1, 1280, 2048), dtype)[:, :DIT_TEXT_ROWS].contiguous()
    vector = upload(torch, source_preblock, "block.vec_f32", (1, 2048), dtype)
    rope = tuple(upload(torch, source_preblock, f"block.rope_{name}_f32",
                        (history.IMAGE_TOKENS, 128), torch.float32) for name in ("cos", "sin"))
    with torch.inference_mode(), torch.autocast("mps", dtype=dtype, enabled=args.bf16_autocast):
        try:
            block.forward_with_cache(img=image, txt=text, vec=vector, vec_txt=vector,
                                     freqs_cis=rope, text_mask=text_mask, kv_cache=buffers,
                                     cache_valid_len=0, update_cache=True, attn_mode="torch",
                                     cache_txt=True, cache_mask=cache_mask)
        except history._AttentionConsumerIntercepted as error:
            if error.phase != "source-T0":
                raise
        else:
            raise RuntimeError("the source writer did not reach the attention consumer")
        if any(t.shape != physical_shape for t in captures.values()):
            raise RuntimeError("unexpected source physical cache shape")
        checks = [(captures[name] == tensor).all() for name, tensor in buffers.items()]
        checks.extend(torch.isfinite(tensor).all() for tensor in buffers.values())
        checks.append((cache_mask == (torch.arange(physical_rows, device="mps") < active_rows)).all())
        # Export actual stored BF16 bytes and small checks in a single terminal read.
        packed = [tensor[:, :active_rows].contiguous().view(torch.int16).flatten() for tensor in buffers.values()]
        terminal = torch.cat([*packed, torch.stack(checks).to(torch.int16)]).cpu()
    if terminal[-len(checks):].tolist() != [1] * len(checks):
        raise RuntimeError("physical source cache or active mask failed validation")
    tensors = {f"history.{name}_bf16": tensor.view(dtype).reshape(1, active_rows, 16, 128).contiguous()
               for name, tensor in zip(buffers, terminal[:-len(checks)].chunk(2), strict=True)}
    payload = save(tensors)
    with artifact.open("xb") as stream:
        stream.write(payload)
    preblock._publish_fresh(args.output, {
        "schema": "cogni_ml.hy_nomas_connected_reference_history_source/v1",
        "artifact": artifact.name, "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        "preblock_manifest_sha256": args.source_preblock[1],
        "refiner_blocks_manifest_sha256": args.source_refiner[1],
        "weights_sha256": weights_sha, "valid_text_tokens": valid, "dit_text_tokens": DIT_TEXT_ROWS,
        "active_rows": active_rows, "physical_rows": physical_rows, "masked_rows": physical_rows - active_rows,
        "upstream_revision": revision, "torch_version": torch.__version__, "pinned_cache_function": identity,
        "mps_bf16_autocast": args.bf16_autocast, "terminal_exports": 1,
    })


if __name__ == "__main__":
    main()
