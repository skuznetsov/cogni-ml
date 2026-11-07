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
import hy_nomas_reference_continuation_control as source  # noqa: E402

DEFAULT_ROOT = Path('/opt/mldata_ssd/cogni-ml/hy-nomas-native-context-20261106.ui7736ue')
MANIFEST_SHA = '0f35bb09964f2b28d28069e8d39a8a1d6194607d4269de6bb824ff54d828e2ac'
RECEIPT_SHA = 'ca8be135ef659497594dec9647c72240f911be8cfd6be989f0740fbf76cd66f5'
INDEX_SHA = 'a235ca592b47161472c7fd94cd3d886fce164480cd0571179c8860002063cbb0'
ATOL = RTOL = 2e-5
VALID = 263
TEXT_ROWS = 320
VIDEO_ROWS = 4320
LATENTS_PER_SEGMENT = 30
CACHE_ROWS = VIDEO_ROWS + TEXT_ROWS
SHAPES = {
    'input.latent_bank': ('F32', [1, 32, 180, 9, 16]),
    'input.qwen_bf16': ('BF16', [1, 1280, 3584]),
    'input.rope_cos': ('F32', [6, 4320, 128]),
    'input.rope_sin': ('F32', [6, 4320, 128]),
    'cache.eps_bf16': ('BF16', [6, 1, 32, 30, 9, 16]),
}


def shared_scene_plan(t1_current_text_kv=False):
    return [dict(kind='writer' if index == 0 else 'denoise',
                 condition='t0_caption' if index == 0 else 't1_shared_scene',
                 valid_text_tokens=VALID, cache_valid_len=0 if index == 0 else CACHE_ROWS,
                 update_cache=index == 0,
                 cache_txt=index == 0 or t1_current_text_kv,
                 rope_bank=0 if index == 0 else 1)
            for index in range(5)]


def checked_json(path, digest):
    if source._sha256(path) != digest:
        raise ValueError(f'input digest mismatch: {path.name}')
    return source.attention_reference._json_no_duplicates(path.read_bytes(), path.name)


def _admit_comparison_latents(path_arg, digest):
    if path_arg is None and digest is None:
        return None
    if path_arg is None or not isinstance(digest, str) or source._HEX.fullmatch(digest) is None:
        raise ValueError('comparison native latents require a path and lowercase SHA-256')
    path = source._regular_file(path_arg, 'comparison native T1 latents')
    expected_bytes = math.prod(source.LATENT_SEGMENT_SHAPE) * 4
    if path.stat().st_size != expected_bytes or source._sha256(path) != digest:
        raise ValueError('comparison native T1 latent shape or digest differs')
    return dict(path=path, sha256=digest, shape=list(source.LATENT_SEGMENT_SHAPE))


def history_visibility_record(
    keep_latents,
    t0_hide_caption_keys=False,
    t1_current_text_kv=False,
):
    if not isinstance(t0_hide_caption_keys, bool):
        raise ValueError('t0_hide_caption_keys must be boolean')
    if not isinstance(t1_current_text_kv, bool):
        raise ValueError('t1_current_text_kv must be boolean')
    if t1_current_text_kv and not t0_hide_caption_keys:
        raise ValueError('--t1-current-text-kv requires --hide-t0-caption-keys')
    if keep_latents is None:
        keep_latents = LATENTS_PER_SEGMENT
        explicit_mask = False
    else:
        explicit_mask = True
    if t0_hide_caption_keys and not explicit_mask:
        raise ValueError('--hide-t0-caption-keys requires --t0-history-keep-latents 30')
    if keep_latents == 0 and not (t0_hide_caption_keys and t1_current_text_kv):
        raise ValueError(
            '--t0-history-keep-latents 0 requires --hide-t0-caption-keys '
            'and --t1-current-text-kv'
        )
    if t0_hide_caption_keys and keep_latents == 10:
        raise ValueError('--hide-t0-caption-keys requires --t0-history-keep-latents 30')
    layout = source.compact_t0_history_visibility_layout(
        keep_latents,
        VALID,
        t0_hide_caption_keys=t0_hide_caption_keys,
        t1_current_text_kv=t1_current_text_kv,
    )
    return dict(
        kind=(
            'all T0 keys hidden with current T1 text K/V'
            if keep_latents == 0
            else 'T0 caption mask with current T1 text K/V rescue'
            if t1_current_text_kv
            else 'fixed-shape T0 key visibility diagnostic'
            if t0_hide_caption_keys
            else 'fixed-shape T0 video-key visibility diagnostic'
        ),
        diagnostic_only=True,
        keep_latents=keep_latents,
        t0_hide_caption_keys=t0_hide_caption_keys,
        t1_current_text_kv=t1_current_text_kv,
        explicit_all_true_mask=(
            explicit_mask and keep_latents == LATENTS_PER_SEGMENT
            and not t0_hide_caption_keys and not t1_current_text_kv
        ),
        t0_video_latents=LATENTS_PER_SEGMENT,
        t0_video_key_rows=layout['t0_video_rows'],
        hidden_t0_video_key_rows=layout['hidden_t0_video_prefix_rows'],
        visible_t0_video_key_rows=layout['visible_t0_video_rows'],
        all_t0_video_keys_hidden=layout['visible_t0_video_rows'] == 0,
        hidden_t0_caption_key_rows=layout['hidden_t0_caption_key_rows'],
        retained_t0_caption_key_rows=layout['retained_t0_caption_key_rows'],
        all_t0_keys_hidden=(
            layout['hidden_t0_video_prefix_rows'] + layout['hidden_t0_caption_key_rows']
            == layout['t0_video_rows'] + layout['t0_text_end'] - layout['t0_text_start']
        ),
        retained_current_t1_video_key_rows=VIDEO_ROWS,
        current_t1_text_key_rows=layout['current_t1_text_key_rows'],
        visible_current_t1_text_key_rows=layout['current_t1_text_key_rows'],
        current_t1_text_key_start=layout['current_t1_text_start'],
        current_t1_text_key_end=layout['current_t1_text_end'],
        compact_attention_key_rows=layout['key_rows'],
        t1_query_rows=VIDEO_ROWS + VALID,
        t0_cache_content_unchanged=True,
        t0_cache_writer_unchanged=True,
        t1_query_rows_unchanged=True,
        t1_text_refinement_unchanged=True,
        t1_rope_and_query_geometry_unchanged=True,
        trained_stock_contract=False,
        t1_text_kv_attention_only=t1_current_text_kv,
        interpretation=(
            'Leaves the T0 writer unchanged and hides all T0 video and caption keys on T1 reads. '
            'Each T1 denoise forward includes its current text K/V in attention only '
            '(update_cache=False); all T1 video and current text keys remain visible, and T1 '
            'query rows, text refinement, RoPE, and geometry remain unchanged. This adds no '
            'persistent cache write or storage.'
            if keep_latents == 0
            else 'Leaves the T0 writer unchanged and hides all admitted T0 caption keys on T1 reads. '
            'Each T1 denoise forward includes its current text K/V in attention only '
            '(update_cache=False); all T0 and T1 video keys remain visible, and T1 query rows, '
            'text refinement, RoPE, and geometry remain unchanged. This adds no persistent '
            'cache write or storage.'
            if t1_current_text_kv
            else 'Leaves the T0 writer unmasked, then hides only all admitted T0 caption keys from '
            'T1 attention. All T0 and T1 video keys remain visible; T1 query rows, text '
            'refinement, RoPE, and geometry remain unchanged. All T0 K/V remain cached; this '
            'does not reduce cache storage or dense attention extent.'
            if t0_hide_caption_keys
            else 'Masks visibility to the oldest repeated-photo T0 image keys. All T0 K/V '
            'remain cached; this does not reduce cache storage or dense attention extent.'
        ),
    )


def build_t0_history_attention_mask(
    torch,
    device,
    keep_latents,
    t0_hide_caption_keys=False,
    t1_current_text_kv=False,
):
    layout = source.compact_t0_history_visibility_layout(
        keep_latents,
        VALID,
        t0_hide_caption_keys=t0_hide_caption_keys,
        t1_current_text_kv=t1_current_text_kv,
    )
    mask = torch.ones((1, layout['key_rows']), dtype=torch.bool, device=device)
    hidden_rows = layout['hidden_t0_video_prefix_rows']
    if hidden_rows:
        mask[:, :hidden_rows] = False
    if layout['hidden_t0_caption_key_rows']:
        mask[:, layout['t0_text_start']:layout['t0_text_end']] = False
    return layout, mask


def admit(args):
    visibility = history_visibility_record(
        getattr(args, 't0_history_keep_latents', None),
        getattr(args, 'hide_t0_caption_keys', False),
        getattr(args, 't1_current_text_kv', False),
    )
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
    comparison_latents = _admit_comparison_latents(
        getattr(args, 'native_latents', None),
        getattr(args, 'native_latents_sha256', None),
    )
    model = source._authenticate_model_sources(SimpleNamespace(
        upstream=args.upstream, cache_index=args.cache_index, cache_index_sha256=INDEX_SHA))
    return dict(root=root, manifest_path=manifest_path, document=manifest,
                receipt_path=receipt_path, receipt=receipt, payload_path=payload,
                path=bank, latent_record=manifest['tensors']['input.latent_bank'],
                rope_cos_record=manifest['tensors']['input.rope_cos'],
                rope_sin_record=manifest['tensors']['input.rope_sin'],
                cache_epsilon_record=manifest['tensors']['cache.eps_bf16'],
                qwen_record=manifest['tensors']['input.qwen_bf16'], mask=mask,
                comparison_latents=comparison_latents, history_visibility=visibility,
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
                 t0_history_hide_caption_keys=admitted['history_visibility']['t0_hide_caption_keys'],
                 t1_current_text_kv=admitted['history_visibility']['t1_current_text_kv'],
                 t0_history_all_keys_hidden=admitted['history_visibility']['all_t0_keys_hidden'],
                 t0_history_keep_latents=(
                     admitted['history_visibility']['keep_latents']
                     if admitted['history_visibility']['explicit_all_true_mask']
                     or admitted['history_visibility']['hidden_t0_video_key_rows'] > 0
                     or admitted['history_visibility']['hidden_t0_caption_key_rows'] > 0
                     else None
                 ),
                 receipt=dict(attention_calls=0, attention_query_chunks=0,
                              t0_history_visibility_mask_calls=0))
    if state['t0_history_keep_latents'] is not None:
        _, visibility_mask = build_t0_history_attention_mask(
            torch, device, state['t0_history_keep_latents'],
            t0_hide_caption_keys=state['t0_history_hide_caption_keys'],
            t1_current_text_kv=state['t1_current_text_kv'])
        state['t0_history_attention_mask'] = visibility_mask
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
    comparison_latents = admitted['comparison_latents']
    if comparison_latents is None:
        native_values = np.memmap(admitted['path'], dtype='<f4', mode='r', shape=source.LATENT_BANK_SHAPE)
        native_cpu = np.array(native_values[:, :, 30:60], copy=True)
        del native_values
        comparison_reference = dict(
            kind='authenticated historical native T1 bank slice',
            path=str(admitted['path']),
            sha256=admitted['receipt']['diagnostics']['latent_bank_sha256'],
        )
    else:
        native_values = np.memmap(
            comparison_latents['path'], dtype='<f4', mode='r',
            shape=source.LATENT_SEGMENT_SHAPE)
        native_cpu = np.array(native_values, copy=True)
        del native_values
        comparison_reference = dict(
            kind='authenticated same-arm native T1 latent segment',
            path=str(comparison_latents['path']),
            sha256=comparison_latents['sha256'],
        )
    native = torch.from_numpy(native_cpu).to(device)
    del native_cpu
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
        plan = shared_scene_plan(state['t1_current_text_kv'])
        for index, call in enumerate(plan):
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
        current_text_rows = VALID if state['t1_current_text_kv'] else 0
        t1_full_k_rows = CACHE_ROWS + VIDEO_ROWS + (
            TEXT_ROWS if state['t1_current_text_kv'] else 0
        )
        t1_compact_k_rows = 2 * VIDEO_ROWS + VALID + current_text_rows
        expected_inventory = {f'{CACHE_ROWS}->{VIDEO_ROWS + VALID}': 54,
                              f'{t1_full_k_rows}->{t1_compact_k_rows}': 216}
        if state['receipt']['attention_row_inventory'] != expected_inventory:
            raise ValueError('shared-scene attention included unadmitted text keys')
        expected_visibility_calls = 216 if state['t0_history_keep_latents'] is not None else 0
        if state['receipt']['t0_history_visibility_mask_calls'] != expected_visibility_calls:
            raise ValueError('T0 history visibility mask call count differs from the 4x54 T1 schedule')
        actual = latent.float()
        delta = actual - native
        bad = (~torch.isfinite(actual)) | (~torch.isfinite(native)) | (
            delta.abs() > ATOL + RTOL * native.abs()
        )
        stats = torch.stack((delta.abs().max(), (delta.square().mean()).sqrt(),
                             bad.float().sum(), (delta != 0).float().sum()))
        terminal = torch.cat((clean, latent), dim=2).float()
        capture('result.latent', terminal)
        capture('comparison.native_t1', stats)
        # One blocking terminal transfer; no scalar observations between forwards.
        packed = torch.cat([value.float().reshape(-1) for _, value in captured]).cpu().numpy()
    stats_offset = 0
    for name, value in captured:
        if name == 'comparison.native_t1':
            break
        stats_offset += value.numel()
    stats_values = packed[stats_offset:stats_offset + 4].tolist()
    comparison_passed = int(stats_values[2]) == 0
    max_abs = float(stats_values[0])
    rmse = float(stats_values[1])
    if not math.isfinite(max_abs):
        max_abs = None
    if not math.isfinite(rmse):
        rmse = None
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
    receipt = dict(schema='cogni_ml.hy_nomas_portrait_dit_control/v1',
                   runtime=dict(torch_version=torch.__version__, device='mps', fallback=0,
                                diffusers_version=diffusers_version),
                   producer_sha256=source._sha256(Path(__file__)),
                   conditioning_manifest_sha256=MANIFEST_SHA,
                   conditioning_payload_sha256=admitted['document']['payload_sha256'],
                   native_receipt_sha256=RECEIPT_SHA,
                   native_latent_sha256=admitted['receipt']['diagnostics']['latent_bank_sha256'],
                   comparison_reference=comparison_reference,
                   cache_index_sha256=INDEX_SHA, upstream_revision=source.PINNED_REVISION,
                   upstream_sources=admitted['source_hashes'],
                   shared_scene_plan=plan, schedule=source.single._schedule_record(scheduler),
                   history_visibility=admitted['history_visibility'],
                   attention=state['receipt'], seconds_through_terminal=elapsed,
                   intermediate_readbacks=0, explicit_mps_synchronize_calls=0, terminal_transfer_count=1,
                   one_resident_bf16_block=True, persistent_layer_caches=54,
                   scope='supplied native clean T0 and saved Qwen; upstream five-forward DiT; encoders excluded',
                   comparison=dict(atol=ATOL, rtol=RTOL, passed=comparison_passed,
                                   max_abs=max_abs, rmse=rmse,
                                   outside_tolerance=int(stats_values[2]),
                                   unequal=int(stats_values[3]), values=native.numel()),
                   result=dict(path=str(latent_path), shape=list(terminal.shape),
                               sha256=source._sha256(latent_path)),
                   capture_payload=dict(path=str(packed_path), sha256=source._sha256(packed_path), records=records))
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: receipt[key] for key in ('comparison', 'seconds_through_terminal', 'result')}), flush=True)
    if not comparison_passed:
        raise ValueError('upstream/native T1 comparison exceeded the admitted 2e-5 tolerance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--upstream', type=Path, default=Path('/opt/mldata_ssd/models/hy-nomas/upstream-911cea6458a8'))
    parser.add_argument('--cache-index', type=Path, default=source.single.DEFAULT_CACHE_INDEX)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--t0-history-keep-latents', type=int, choices=(0, 10, 30),
                        help='diagnostic T1 visibility mask; 30 retains all T0 video, 10 hides oldest 20, 0 hides all T0 keys only with both rescue flags')
    parser.add_argument('--hide-t0-caption-keys', action='store_true',
                        help='hide cached T0 caption keys; requires explicit keep-latents 30, or keep 0 with current T1 text K/V rescue')
    parser.add_argument('--t1-current-text-kv', action='store_true',
                        help='add current T1 text K/V to attention while T0 caption keys are hidden')
    parser.add_argument('--native-latents', type=Path,
                        help='optional authenticated same-arm native T1 raw F32 segment for the 2e-5 gate')
    parser.add_argument('--native-latents-sha256',
                        help='required SHA-256 when --native-latents is supplied')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    try:
        history_visibility_record(
            args.t0_history_keep_latents,
            args.hide_t0_caption_keys,
            args.t1_current_text_kv,
        )
    except ValueError as error:
        parser.error(str(error))
    if not args.output_dir.is_absolute() or args.output_dir.exists() or args.output_dir.is_symlink():
        parser.error('output-dir must be a fresh absolute path')
    args.output_dir = args.output_dir.parent.resolve(strict=True) / args.output_dir.name
    admitted = admit(args)
    if args.execute:
        execute(admitted, args.output_dir)
    else:
        print(json.dumps(dict(
            model_launched=False,
            validated=True,
            conditioning_sha256=MANIFEST_SHA,
            total_bf16_weight_bytes=admitted['total_weight_bytes'],
            forward_plan=shared_scene_plan(admitted['history_visibility']['t1_current_text_kv']),
            history_visibility=admitted['history_visibility'],
            comparison_native=(
                dict(
                    kind='authenticated same-arm native T1 latent segment',
                    path=str(admitted['comparison_latents']['path']),
                    sha256=admitted['comparison_latents']['sha256'],
                    shape=admitted['comparison_latents']['shape'],
                )
                if admitted['comparison_latents'] is not None
                else dict(kind='authenticated historical native T1 bank slice', sha256=RECEIPT_SHA)
            ),
            output_dir=str(args.output_dir),
        )))


if __name__ == '__main__':
    main()
