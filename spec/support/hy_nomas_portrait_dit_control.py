#!/usr/bin/env python3
"""Replay the frozen portrait's shared-scene DiT with pinned upstream equations.

The supplied clean T0 and saved Qwen states are inputs, not upstream encoder
parity claims. No random tensor is redrawn. Model arithmetic and comparisons
stay on MPS until one terminal packed export. This diagnostic expects an
external cache-aware memory guard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

SUPPORT = Path(__file__).resolve().parent
sys.path.insert(0, str(SUPPORT))
import hy_nomas_reference_continuation_control as source

DEFAULT_ROOT = Path('/opt/mldata_ssd/cogni-ml/hy-nomas-native-context-20261106.ui7736ue')
MANIFEST_SHA = '0f35bb09964f2b28d28069e8d39a8a1d6194607d4269de6bb824ff54d828e2ac'
RECEIPT_SHA = 'ca8be135ef659497594dec9647c72240f911be8cfd6be989f0740fbf76cd66f5'
INDEX_SHA = 'a235ca592b47161472c7fd94cd3d886fce164480cd0571179c8860002063cbb0'
ATOL = RTOL = 2e-5
VALID = 263
TEXT_ROWS = 320
VIDEO_ROWS = 4320
CACHE_ROWS = VIDEO_ROWS + TEXT_ROWS
SHAPES = {
    'input.latent_bank': ('F32', [1, 32, 180, 9, 16]),
    'input.qwen_bf16': ('BF16', [1, 1280, 3584]),
    'input.rope_cos': ('F32', [6, 4320, 128]),
    'input.rope_sin': ('F32', [6, 4320, 128]),
    'cache.eps_bf16': ('BF16', [6, 1, 32, 30, 9, 16]),
}


def shared_scene_plan():
    return [dict(kind='writer' if index == 0 else 'denoise',
                 condition='t0_caption' if index == 0 else 't1_shared_scene',
                 valid_text_tokens=VALID, cache_valid_len=0 if index == 0 else CACHE_ROWS,
                 update_cache=index == 0, cache_txt=index == 0,
                 rope_bank=0 if index == 0 else 1)
            for index in range(5)]


def checked_json(path, digest):
    if source._sha256(path) != digest:
        raise ValueError(f'input digest mismatch: {path.name}')
    return source.attention_reference._json_no_duplicates(path.read_bytes(), path.name)


def admit(args):
    root = args.input_root.resolve(strict=True)
    manifest_path = root / 'photo-continuation-v2.json'
    manifest = checked_json(manifest_path, MANIFEST_SHA)
    receipt_path = root / 'photo-native-v2/receipt.json'
    receipt = checked_json(receipt_path, RECEIPT_SHA)
    payload = source._regular_file(root / manifest['payload'], 'conditioning payload')
    if (payload.stat().st_size != manifest['payload_bytes']
            or source._sha256(payload) != manifest['payload_sha256']):
        raise ValueError('conditioning payload changed')
    mask = manifest['qwen_source']['cropped_attention_mask']
    source._validate_mask(mask, 'shared scene', VALID)
    if (receipt['inputs_sha256'] != MANIFEST_SHA
            or receipt['dit_weight_index_sha256'] != INDEX_SHA
            or receipt['text_conditioning_mode'] != 'shared_scene'
            or receipt['text_cache_policy'] != 'once_per_scene'
            or receipt['source_bf16'] is not True
            or receipt['steps_per_segment'] != 4 or receipt['segments'] != 2
            or receipt['reference_initialization'] is not True):
        raise ValueError('native portrait contract differs')
    for name, (dtype, shape) in SHAPES.items():
        record = manifest['tensors'][name]
        source._validate_tensor_record(record, dtype, shape)
        if source._sha256_range(payload, record['offset'], record['nbytes']) != record['sha256']:
            raise ValueError(f'conditioning range mismatch: {name}')
    bank = root / 'photo-native-v2/latent-bank.f32le'
    if (bank.stat().st_size != math.prod(source.LATENT_BANK_SHAPE) * 4
            or source._sha256(bank) != receipt['diagnostics']['latent_bank_sha256']):
        raise ValueError('native latent bank changed')
    scene = json.loads(receipt['prompt'])
    if len(scene['segments']) != 1 or scene['segments'][0]['contained_actions']:
        raise ValueError('shared scene must not contain separate actions')
    model = source._authenticate_model_sources(SimpleNamespace(
        upstream=args.upstream, cache_index=args.cache_index, cache_index_sha256=INDEX_SHA))
    return dict(root=root, manifest_path=manifest_path, document=manifest,
                receipt_path=receipt_path, receipt=receipt, payload_path=payload,
                path=bank, latent_record=manifest['tensors']['input.latent_bank'],
                rope_cos_record=manifest['tensors']['input.rope_cos'],
                rope_sin_record=manifest['tensors']['input.rope_sin'],
                cache_epsilon_record=manifest['tensors']['cache.eps_bf16'],
                qwen_record=manifest['tensors']['input.qwen_bf16'], mask=mask,
                **model)


def execute(admitted, output):
    import numpy as np
    from safetensors import safe_open

    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0':
        raise ValueError('explicit MPS fallback=0 required')
    source.single._configure_mps_runtime()
    torch, scheduler, diffusers_version = source.single._prepare_cpu_scheduler(
        Path(admitted['upstream_path']), 'source-bf16')
    if torch.__version__ != '2.8.0' or not torch.backends.mps.is_available():
        raise ValueError('this DiT control requires the admitted Torch 2.8.0 MPS lineage')
    # Process-local adapter geometry; the upstream block/cache bodies stay pinned.
    source.DIT_TEXT_ROWS, source.CACHE_CAPACITY = TEXT_ROWS, CACHE_ROWS
    source.QUERY_CHUNK = 256
    device, dtype = torch.device('mps'), torch.bfloat16
    state = dict(dtype=dtype, device_type='mps', call=None, attention_layout='compact',
                 t0_valid_tokens=VALID,
                 receipt=dict(attention_calls=0, attention_query_chunks=0))
    helpers = source.block_reference._install_upstream_modules(Path(admitted['upstream_path']))
    regular_state = dict(mode='bf16', compare=source.single._NoopCompare(),
                         receipt=dict(attention_calls=0, attention_query_chunks=0))
    regular = source.attention_reference._build_attention_adapter(torch, regular_state)
    refiner_state = dict(mode='bf16', valid_text_tokens=VALID, attention_block=0,
                         attention_query_chunks=0, compare=source.single._NoopCompare())
    _, refiner = source.attention_reference._install_refiner_attention(torch, refiner_state)
    upstream = Path(admitted['upstream_path'])
    source.single._bind_refiner_attention_adapter(upstream, refiner, refiner_state)
    globals_ = source.attention_reference._load_prelude_weights(
        torch, Path(admitted['global_shard']['path']), upstream, helpers, device, dtype)
    source.single._bind_refiner_attention_adapter(upstream, refiner, refiner_state, globals_['txt_in'])
    block_type = source.block_reference._extract_pinned_block(
        upstream, torch, regular, helpers[-1].apply_rotary_emb, helpers)
    block = block_type(hidden_size=2048, heads_num=16, mlp_width_ratio=4.0,
                       mlp_act_type='gelu_tanh', attn_mode='magi_mp_causal',
                       qk_norm=True, qk_norm_type='rms', qkv_bias=True,
                       dtype=dtype, device=torch.device('meta'))
    block.to_empty(device=device).eval()
    expected = source.cache_reference._BLOCK_RELATIVE_SHAPES
    if set(block.state_dict()) != set(expected):
        raise ValueError('upstream block tensor inventory changed')
    cache_function = source._build_pinned_cache_function(torch, upstream, state)
    source.two_segment._bind_runtime_cache_functions([block], None, cache_function)
    caches, masks = source._new_caches(torch, device)
    qwen = source._load_qwen(torch, np, admitted, device)
    qmask = torch.tensor([admitted['mask']], dtype=torch.bool, device=device)
    text_mask = qmask[:, :TEXT_ROWS]
    clean = source._load_t0_clean(torch, np, admitted, device)
    epsilon = source._load_cache_epsilon(torch, np, admitted, device)
    ropes = [tuple(source._load_segment_rope(torch, np, admitted, name, segment, device)
                   for name in ('cos', 'sin')) for segment in (0, 1)]
    initial = source._load_t1_noise(torch, np, admitted, device)
    times = scheduler.timesteps.to(device=device, dtype=torch.float32)
    deltas = (scheduler.sigmas[1:] - scheduler.sigmas[:-1]).to(device=device, dtype=torch.float32)
    native_values = np.memmap(admitted['path'], dtype='<f4', mode='r', shape=source.LATENT_BANK_SHAPE)
    native = torch.from_numpy(np.array(native_values[:, :, 30:60], copy=True)).to(device)
    del native_values
    captured = []

    def capture(name, value):
        if value.device.type != 'mps':
            raise ValueError('model capture escaped GPU')
        captured.append((name, value.detach().contiguous().clone()))

    started = time.monotonic()
    with torch.inference_mode():
        capture('t0.clean_latent', clean)
        with torch.autocast('mps', enabled=False):
            writer = (clean.float() * (1.0 - source.WRITER_SIGMA)
                      + epsilon.float() * source.WRITER_SIGMA).to(dtype)
        capture('t0.noised_latent', writer)
        latent = initial
        for index, call in enumerate(shared_scene_plan()):
            state['call'] = call
            timestep = (torch.full((1,), source.WRITER_TIMESTEP, device=device, dtype=torch.float32)
                        if index == 0 else times[index - 1].reshape(1))
            label = 't0' if index == 0 else f't1.step{index}'
            refiner_state['attention_block'] = 0
            refiner_state['attention_query_chunks'] = 0
            with torch.autocast('mps', dtype=dtype):
                vector = globals_['time_in'](timestep)
                text_full = globals_['txt_in'](qwen, timestep, qmask)
                text = text_full[:, :TEXT_ROWS].contiguous()
                image = globals_['img_in'](writer if index == 0 else latent)
                if refiner_state['attention_block'] != 2:
                    raise ValueError('incomplete source text refiner')
                capture(f'{label}.vector', vector)
                capture(f'{label}.text_valid', text[:, :VALID])
                if index < 2:
                    capture(f'{label}.patch', image)
                for layer, entry in enumerate(admitted['cache_index']['blocks']):
                    path = source.single._canonical_file(
                        Path(admitted['cache_index_path']).parent / entry['bf16'], 'BF16 block')
                    before = admitted['cache_identities'][str(path)]
                    if source.single._file_identity(path) != before:
                        raise ValueError('BF16 block changed after authentication')
                    with safe_open(str(path), framework='pt', device='cpu') as weights:
                        if set(weights.keys()) != {f'double_blocks.{layer}.{name}' for name in expected}:
                            raise ValueError('source block inventory changed')
                        for name, target in block.state_dict().items():
                            value = weights.get_tensor(f'double_blocks.{layer}.{name}')
                            if value.dtype != dtype or list(value.shape) != expected[name]:
                                raise ValueError('source block geometry changed')
                            target.copy_(value, non_blocking=True)
                    if source.single._file_identity(path) != before:
                        raise ValueError('BF16 block changed during staging')
                    image, text = block(
                        img=image, txt=text, vec=vector, vec_txt=vector,
                        freqs_cis=ropes[call['rope_bank']], text_mask=text_mask,
                        attn_param=None, is_flash=False, txt_tokens_per_block=TEXT_ROWS,
                        prompt_video_boundaries=[0, VIDEO_ROWS], kv_cache=caches[layer],
                        **source._block_cache_contract(call), cache_attn_mode='torch',
                        cache_mask=masks[layer])
                    if index < 2 and layer in (0, 53):
                        capture(f'{label}.block{layer}.image', image)
                        capture(f'{label}.block{layer}.text_valid', text[:, :VALID])
                    if index == 0 and layer == 0:
                        for key in ('k', 'v'):
                            capture(f't0.block0.cache_{key}', caches[layer][key][:, :VIDEO_ROWS + VALID])
                prediction = globals_['final_layer'](image, vector)
            if index > 0:
                velocity = prediction.reshape(1, 30, 9, 16, 32).permute(0, 4, 1, 2, 3).contiguous()
                capture(f'{label}.velocity', velocity)
                with torch.autocast('mps', enabled=False):
                    latent = (latent.float() + velocity.float() * deltas[index - 1]).to(dtype)
                capture(f'{label}.after_euler', latent)
            print(json.dumps(dict(forward=index + 1, forwards=5, stage=label,
                                  enqueue_seconds=time.monotonic() - started)), flush=True)
        if state['receipt']['attention_calls'] != 270:
            raise ValueError('incomplete upstream 54-block five-forward schedule')
        expected_inventory = {f'{CACHE_ROWS}->{VIDEO_ROWS + VALID}': 54,
                              f'{CACHE_ROWS + VIDEO_ROWS}->{2 * VIDEO_ROWS + VALID}': 216}
        if state['receipt']['attention_row_inventory'] != expected_inventory:
            raise ValueError('shared-scene attention included unadmitted text keys')
        actual = latent.float()
        delta = actual - native
        bad = (~torch.isfinite(actual)) | (delta.abs() > ATOL + RTOL * native.abs())
        stats = torch.stack((delta.abs().max(), (delta.square().mean()).sqrt(),
                             bad.float().sum(), (delta != 0).float().sum()))
        terminal = torch.cat((clean, latent), dim=2).float()
        capture('result.latent', terminal)
        capture('comparison.native_t1', stats)
        # One blocking terminal transfer; no scalar observations between forwards.
        packed = torch.cat([value.float().reshape(-1) for _, value in captured]).cpu().numpy()
    elapsed = time.monotonic() - started
    output.mkdir(mode=0o700)
    packed_path = output / 'captures.f32le'
    with packed_path.open('xb') as stream:
        np.asarray(packed, dtype='<f4').tofile(stream)
    records, offset = [], 0
    for name, value in captured:
        size = value.numel() * 4
        records.append(dict(name=name, shape=list(value.shape), offset=offset, nbytes=size,
                            sha256=hashlib.sha256(memoryview(packed).cast('B')[offset:offset + size]).hexdigest()))
        offset += size
    by_name = {record['name']: record for record in records}
    record = by_name['result.latent']
    terminal_raw = memoryview(packed).cast('B')[record['offset']:record['offset'] + record['nbytes']]
    latent_path = output / 'latent.f32le'
    with latent_path.open('xb') as stream:
        stream.write(terminal_raw)
    stats_record = by_name['comparison.native_t1']
    stats_values = packed[stats_record['offset'] // 4:stats_record['offset'] // 4 + 4].tolist()
    receipt = dict(schema='cogni_ml.hy_nomas_portrait_dit_control/v1',
                   runtime=dict(torch_version=torch.__version__, device='mps', fallback=0,
                                diffusers_version=diffusers_version),
                   producer_sha256=source._sha256(Path(__file__)),
                   conditioning_manifest_sha256=MANIFEST_SHA,
                   conditioning_payload_sha256=admitted['document']['payload_sha256'],
                   native_receipt_sha256=RECEIPT_SHA,
                   native_latent_sha256=admitted['receipt']['diagnostics']['latent_bank_sha256'],
                   cache_index_sha256=INDEX_SHA, upstream_revision=source.PINNED_REVISION,
                   upstream_sources=admitted['source_hashes'],
                   shared_scene_plan=shared_scene_plan(), schedule=source.single._schedule_record(scheduler),
                   attention=state['receipt'], seconds_through_terminal=elapsed,
                   intermediate_readbacks=0, explicit_mps_synchronize_calls=0, terminal_transfer_count=1,
                   one_resident_bf16_block=True, persistent_layer_caches=54,
                   scope='supplied native clean T0 and saved Qwen; upstream five-forward DiT; encoders excluded',
                   comparison=dict(atol=ATOL, rtol=RTOL, max_abs=stats_values[0],
                                   rmse=stats_values[1], outside_tolerance=int(stats_values[2]),
                                   unequal=int(stats_values[3]), values=native.numel()),
                   result=dict(path=str(latent_path), shape=list(terminal.shape),
                               sha256=source._sha256(latent_path)),
                   capture_payload=dict(path=str(packed_path), sha256=source._sha256(packed_path), records=records))
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: receipt[key] for key in ('comparison', 'seconds_through_terminal', 'result')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--upstream', type=Path, default=Path('/opt/mldata_ssd/models/hy-nomas/upstream-911cea6458a8'))
    parser.add_argument('--cache-index', type=Path, default=source.single.DEFAULT_CACHE_INDEX)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.output_dir.is_absolute() or args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error('output-dir must be a fresh absolute path')
    args.output_dir = args.output_dir.parent.resolve(strict=True) / args.output_dir.name
    admitted = admit(args)
    if args.execute:
        execute(admitted, args.output_dir)
    else:
        print(json.dumps(dict(model_launched=False, validated=True, conditioning_sha256=MANIFEST_SHA,
                              total_bf16_weight_bytes=admitted['total_weight_bytes'],
                              forward_plan=shared_scene_plan(), output_dir=str(args.output_dir))))


if __name__ == '__main__':
    main()
