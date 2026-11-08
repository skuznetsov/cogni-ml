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
LATENT_SEGMENT_BYTES = 1 * 32 * LATENTS_PER_SEGMENT * 9 * 16 * 4
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


def generated_context_plan():
    """Return stock T0 denoising, its separate cache write, and frozen T1 reads."""
    t0_denoise = [dict(kind='denoise', condition='t0_shared_scene', step=step,
                       valid_text_tokens=VALID, cache_valid_len=0,
                       update_cache=False, cache_txt=True, rope_bank=0)
                  for step in range(source.STEPS)]
    writer_and_t1 = shared_scene_plan()
    t1_denoise = [dict(call, step=step) for step, call in enumerate(writer_and_t1[1:])]
    return t0_denoise + [writer_and_t1[0]] + t1_denoise


def validate_t0_context(context, keep_latents=None, hide_caption=False, current_text_kv=False):
    if context not in ('photo', 'generated'):
        raise ValueError('--t0-context must be photo or generated')
    if context == 'generated' and (
        keep_latents is not None or hide_caption or current_text_kv
    ):
        raise ValueError('--t0-context generated requires ordinary unmasked T1 history')


def admit_native_endpoint(path_arg, digest, label):
    if path_arg is None and digest is None:
        return None
    if path_arg is None or not isinstance(digest, str) or source._HEX.fullmatch(digest) is None:
        raise ValueError(f'native {label} endpoint requires both a path and lowercase SHA-256')
    path = source._regular_file(path_arg, f'native {label} endpoint')
    if path.stat().st_size != LATENT_SEGMENT_BYTES or source._sha256(path) != digest:
        raise ValueError(f'native {label} endpoint shape or digest differs')
    return dict(path=path, sha256=digest, shape=list(source.LATENT_SEGMENT_SHAPE))


def tensor_time_slice_sha256(path, record, start, stop, itemsize):
    shape = record.get('shape')
    if (not isinstance(shape, list) or len(shape) != 5 or shape[0] != 1
            or type(start) is not int or type(stop) is not int
            or not 0 <= start < stop <= shape[2] or type(itemsize) is not int or itemsize <= 0):
        raise ValueError('time-slice hash requires a valid [1,C,T,H,W] tensor extent')
    digest = hashlib.sha256()
    channel_stride = shape[2] * shape[3] * shape[4] * itemsize
    slice_bytes = (stop - start) * shape[3] * shape[4] * itemsize
    slice_start = record['offset'] + start * shape[3] * shape[4] * itemsize
    with Path(path).open('rb') as stream:
        for channel in range(shape[1]):
            stream.seek(slice_start + channel * channel_stride)
            chunk = stream.read(slice_bytes)
            if len(chunk) != slice_bytes:
                raise ValueError('time-slice hash range is truncated')
            digest.update(chunk)
    return digest.hexdigest()


def frozen_input_identity(admitted):
    """Hash the exact fixed T0/T1 noise, scene text, RoPE, and T0 writer epsilon."""
    payload = admitted['payload_path']
    latent = admitted['latent_record']
    rope_cos, rope_sin = admitted['rope_cos_record'], admitted['rope_sin_record']
    cache_epsilon = admitted['cache_epsilon_record']
    rope_segment_bytes = math.prod(rope_cos['shape'][1:]) * 4
    epsilon_segment_bytes = math.prod(cache_epsilon['shape'][1:]) * 2
    return dict(
        t0_initial_noise_sha256=tensor_time_slice_sha256(
            payload, latent, 0, LATENTS_PER_SEGMENT, 4),
        t1_initial_noise_sha256=tensor_time_slice_sha256(
            payload, latent, LATENTS_PER_SEGMENT, 2 * LATENTS_PER_SEGMENT, 4),
        scene_qwen_bf16_sha256=admitted['qwen_record']['sha256'],
        t0_rope_cos_sha256=source._sha256_range(
            payload, rope_cos['offset'], rope_segment_bytes),
        t0_rope_sin_sha256=source._sha256_range(
            payload, rope_sin['offset'], rope_segment_bytes),
        t1_rope_cos_sha256=source._sha256_range(
            payload, rope_cos['offset'] + rope_segment_bytes, rope_segment_bytes),
        t1_rope_sin_sha256=source._sha256_range(
            payload, rope_sin['offset'] + rope_segment_bytes, rope_segment_bytes),
        t0_cache_epsilon_bf16_sha256=source._sha256_range(
            payload, cache_epsilon['offset'], epsilon_segment_bytes),
    )


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
    context = getattr(args, 't0_context', 'photo')
    validate_t0_context(
        context,
        getattr(args, 't0_history_keep_latents', None),
        getattr(args, 'hide_t0_caption_keys', False),
        getattr(args, 't1_current_text_kv', False),
    )
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
    native_t0_endpoint = admit_native_endpoint(
        getattr(args, 'native_t0_latents', None),
        getattr(args, 'native_t0_latents_sha256', None),
        'T0',
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
                comparison_latents=comparison_latents,
                native_t0_endpoint=native_t0_endpoint,
                t0_context=context,
                frozen_input_identity=frozen_input_identity(dict(
                    payload_path=payload,
                    latent_record=manifest['tensors']['input.latent_bank'],
                    qwen_record=manifest['tensors']['input.qwen_bf16'],
                    rope_cos_record=manifest['tensors']['input.rope_cos'],
                    rope_sin_record=manifest['tensors']['input.rope_sin'],
                    cache_epsilon_record=manifest['tensors']['cache.eps_bf16'],
                )),
                history_visibility=visibility,
                **model)


def _load_t0_initial_noise(torch, np, admitted, device):
    record = admitted['latent_record']
    raw = np.memmap(
        admitted['payload_path'], dtype='<f4', mode='r',
        offset=record['offset'], shape=tuple(record['shape']),
    )
    value = np.array(raw[:, :, :LATENTS_PER_SEGMENT, :, :], copy=True)
    del raw
    latent = torch.from_numpy(value).to(device=device, dtype=torch.bfloat16)
    if tuple(latent.shape) != source.LATENT_SEGMENT_SHAPE:
        raise ValueError('authenticated T0 noise segment has an unexpected extent')
    return latent


def _load_native_endpoint(torch, np, endpoint, device):
    if endpoint is None:
        return None
    values = np.memmap(
        endpoint['path'], dtype='<f4', mode='r', shape=tuple(endpoint['shape']))
    cpu = np.array(values, copy=True)
    del values
    return torch.from_numpy(cpu).to(device)


def _comparison_stats(torch, actual, native):
    if native is None:
        return None
    delta = actual.float() - native
    bad = (~torch.isfinite(actual)) | (~torch.isfinite(native)) | (
        delta.abs() > ATOL + RTOL * native.abs()
    )
    return torch.stack((delta.abs().max(), (delta.square().mean()).sqrt(),
                        bad.float().sum(), (delta != 0).float().sum()))


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
    if admitted['t0_context'] == 'generated':
        clean = _load_t0_initial_noise(torch, np, admitted, device)
    else:
        clean = source._load_t0_clean(torch, np, admitted, device)
    epsilon = source._load_cache_epsilon(torch, np, admitted, device)
    ropes = [tuple(source._load_segment_rope(torch, np, admitted, name, segment, device)
                   for name in ('cos', 'sin')) for segment in (0, 1)]
    initial = source._load_t1_noise(torch, np, admitted, device)
    times = scheduler.timesteps.to(device=device, dtype=torch.float32)
    deltas = (scheduler.sigmas[1:] - scheduler.sigmas[:-1]).to(device=device, dtype=torch.float32)
    comparison_latents = admitted['comparison_latents']
    if comparison_latents is None and admitted['t0_context'] == 'photo':
        native_values = np.memmap(admitted['path'], dtype='<f4', mode='r', shape=source.LATENT_BANK_SHAPE)
        native_cpu = np.array(native_values[:, :, 30:60], copy=True)
        del native_values
        comparison_reference = dict(
            kind='authenticated historical native T1 bank slice',
            path=str(admitted['path']),
            sha256=admitted['receipt']['diagnostics']['latent_bank_sha256'],
        )
    elif comparison_latents is not None:
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
    else:
        native_cpu = None
        comparison_reference = None
    native = torch.from_numpy(native_cpu).to(device) if native_cpu is not None else None
    if native_cpu is not None:
        del native_cpu
    native_t0 = _load_native_endpoint(torch, np, admitted['native_t0_endpoint'], device)
    native_t0_reference = (
        dict(kind='authenticated native T0 endpoint',
             path=str(admitted['native_t0_endpoint']['path']),
             sha256=admitted['native_t0_endpoint']['sha256'])
        if admitted['native_t0_endpoint'] is not None else None
    )
    captured = []

    def capture(name, value):
        if admitted['t0_context'] == 'generated' and not name.startswith(
            ('result.', 'comparison.')
        ):
            return
        if value.device.type != 'mps':
            raise ValueError('model capture escaped GPU')
        captured.append((name, value.detach().contiguous().clone()))

    started = time.monotonic()
    with torch.inference_mode():
        if admitted['t0_context'] == 'generated':
            t0_denoise_plan = generated_context_plan()[:source.STEPS]
            for step, call in enumerate(t0_denoise_plan):
                state['call'] = call
                timestep = times[step].reshape(1)
                refiner_state['attention_block'] = 0
                refiner_state['attention_query_chunks'] = 0
                with torch.autocast('mps', dtype=dtype):
                    vector = globals_['time_in'](timestep)
                    text_full = globals_['txt_in'](qwen, timestep, qmask)
                    text = text_full[:, :TEXT_ROWS].contiguous()
                    image = globals_['img_in'](clean)
                    if refiner_state['attention_block'] != 2:
                        raise ValueError('incomplete T0 text refiner during generated-context denoising')
                    for layer, entry in enumerate(admitted['cache_index']['blocks']):
                        path = source.single._canonical_file(
                            Path(admitted['cache_index_path']).parent / entry['bf16'], 'BF16 block')
                        before = admitted['cache_identities'][str(path)]
                        if source.single._file_identity(path) != before:
                            raise ValueError('BF16 block changed after authentication')
                        with safe_open(str(path), framework='pt', device='cpu') as weights:
                            if set(weights.keys()) != {
                                f'double_blocks.{layer}.{name}' for name in expected
                            }:
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
                            freqs_cis=ropes[0], text_mask=text_mask,
                            attn_param=None, is_flash=False, txt_tokens_per_block=TEXT_ROWS,
                            prompt_video_boundaries=[0, VIDEO_ROWS], kv_cache=caches[layer],
                            **source._block_cache_contract(call), cache_attn_mode='torch',
                            cache_mask=masks[layer])
                    prediction = globals_['final_layer'](image, vector)
                if tuple(prediction.shape) != (1, VIDEO_ROWS, 32):
                    raise ValueError('generated T0 final head returned unexpected geometry')
                velocity = prediction.reshape(1, LATENTS_PER_SEGMENT, 9, 16, 32)
                velocity = velocity.permute(0, 4, 1, 2, 3).contiguous()
                with torch.autocast('mps', enabled=False):
                    clean = (clean.float() + velocity.float() * deltas[step]).to(dtype)
                del image, text, text_full, vector, prediction, velocity
                print(json.dumps(dict(forward=step + 1, forwards=2 * source.STEPS + 1,
                                      stage=f't0.denoise{step + 1}',
                                      enqueue_seconds=time.monotonic() - started)), flush=True)

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
            forward_offset = source.STEPS if admitted['t0_context'] == 'generated' else 0
            total_forwards = 2 * source.STEPS + 1 if admitted['t0_context'] == 'generated' else 5
            print(json.dumps(dict(forward=forward_offset + index + 1,
                                  forwards=total_forwards, stage=label,
                                  enqueue_seconds=time.monotonic() - started)), flush=True)
        expected_forwards = 2 * source.STEPS + 1 if admitted['t0_context'] == 'generated' else 5
        if state['receipt']['attention_calls'] != expected_forwards * 54:
            raise ValueError('incomplete upstream source-BF16 forward schedule')
        current_text_rows = VALID if state['t1_current_text_kv'] else 0
        t1_full_k_rows = CACHE_ROWS + VIDEO_ROWS + (
            TEXT_ROWS if state['t1_current_text_kv'] else 0
        )
        t1_compact_k_rows = 2 * VIDEO_ROWS + VALID + current_text_rows
        expected_inventory = {f'{CACHE_ROWS}->{VIDEO_ROWS + VALID}':
                              (source.STEPS + 1) * 54
                              if admitted['t0_context'] == 'generated' else 54,
                              f'{t1_full_k_rows}->{t1_compact_k_rows}': 216}
        if state['receipt']['attention_row_inventory'] != expected_inventory:
            raise ValueError('shared-scene attention included unadmitted text keys')
        expected_visibility_calls = 216 if state['t0_history_keep_latents'] is not None else 0
        if state['receipt']['t0_history_visibility_mask_calls'] != expected_visibility_calls:
            raise ValueError('T0 history visibility mask call count differs from the 4x54 T1 schedule')
        t0_stats = _comparison_stats(torch, clean, native_t0)
        t1_stats = _comparison_stats(torch, latent, native)
        if admitted['t0_context'] == 'generated':
            capture('result.t0', clean.float())
            capture('result.t1', latent.float())
            result_shape = [1, 32, 2 * LATENTS_PER_SEGMENT, 9, 16]
        else:
            terminal = torch.cat((clean, latent), dim=2).float()
            capture('result.latent', terminal)
            result_shape = list(terminal.shape)
        if t0_stats is not None:
            capture('comparison.native_t0', t0_stats)
        if t1_stats is not None:
            capture('comparison.native_t1', t1_stats)
        # One blocking terminal transfer; no scalar observations between forwards.
        packed = torch.cat([value.float().reshape(-1) for _, value in captured]).cpu().numpy()
    comparison_values = {}
    stats_offset = 0
    for name, value in captured:
        if name.startswith('comparison.'):
            comparison_values[name] = packed[stats_offset:stats_offset + 4].tolist()
        stats_offset += value.numel()

    def comparison_record(name, reference):
        values = comparison_values.get(name)
        if values is None:
            return dict(reference=reference, atol=ATOL, rtol=RTOL, passed=None)
        max_abs = float(values[0])
        rmse = float(values[1])
        return dict(
            reference=reference,
            atol=ATOL,
            rtol=RTOL,
            passed=int(values[2]) == 0,
            max_abs=max_abs if math.isfinite(max_abs) else None,
            rmse=rmse if math.isfinite(rmse) else None,
            outside_tolerance=int(values[2]),
            unequal=int(values[3]),
            values=math.prod(source.LATENT_SEGMENT_SHAPE),
        )

    t0_comparison = comparison_record('comparison.native_t0', native_t0_reference)
    t1_comparison = comparison_record('comparison.native_t1', comparison_reference)
    comparison_results = (t0_comparison, t1_comparison)
    compared_endpoints = [result for result in comparison_results if result['passed'] is not None]
    comparisons_passed = (
        all(result['passed'] for result in compared_endpoints)
        if compared_endpoints else None
    )
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
    packed_bytes = memoryview(packed).cast('B')

    def write_capture(name, filename):
        record = by_name[name]
        raw = packed_bytes[record['offset']:record['offset'] + record['nbytes']]
        path = output / filename
        with path.open('xb') as stream:
            stream.write(raw)
        return dict(path=str(path), shape=record['shape'], sha256=source._sha256(path))

    latent_path = output / 'latent.f32le'
    endpoints = None
    if admitted['t0_context'] == 'generated':
        t0_endpoint = write_capture('result.t0', 't0.f32le')
        t1_endpoint = write_capture('result.t1', 't1.f32le')
        t0_record, t1_record = by_name['result.t0'], by_name['result.t1']
        t0_values = np.frombuffer(
            packed_bytes[t0_record['offset']:t0_record['offset'] + t0_record['nbytes']],
            dtype='<f4',
        ).reshape(source.LATENT_SEGMENT_SHAPE)
        t1_values = np.frombuffer(
            packed_bytes[t1_record['offset']:t1_record['offset'] + t1_record['nbytes']],
            dtype='<f4',
        ).reshape(source.LATENT_SEGMENT_SHAPE)
        combined = np.concatenate((t0_values, t1_values), axis=2)
        with latent_path.open('xb') as stream:
            combined.tofile(stream)
        endpoints = dict(t0=t0_endpoint, t1=t1_endpoint)
    else:
        record = by_name['result.latent']
        terminal_raw = packed_bytes[record['offset']:record['offset'] + record['nbytes']]
        with latent_path.open('xb') as stream:
            stream.write(terminal_raw)
    t0_context_record = dict(
        kind=(
            'stock-generated T0 from the authenticated saved noise segment; no photo conditioning'
            if admitted['t0_context'] == 'generated'
            else 'technical photo-reference T0 supplied as a clean latent'
        ),
        mode=admitted['t0_context'],
        t0_denoise_forwards=source.STEPS if admitted['t0_context'] == 'generated' else 0,
        t0_cache_writer_forwards=1,
        writer_sigma=source.WRITER_SIGMA,
        writer_timestep=source.WRITER_TIMESTEP,
        generated_identity_claim='none' if admitted['t0_context'] == 'generated' else None,
        source_rng_parity='saved MPS noise; stock upstream CPU RNG values are not claimed',
    )
    receipt = dict(schema='cogni_ml.hy_nomas_portrait_dit_control/v1',
                   runtime=dict(torch_version=torch.__version__, device='mps', fallback=0,
                                diffusers_version=diffusers_version),
                   producer_sha256=source._sha256(Path(__file__)),
                   conditioning_manifest_sha256=MANIFEST_SHA,
                   conditioning_payload_sha256=admitted['document']['payload_sha256'],
                   native_receipt_sha256=RECEIPT_SHA,
                   native_latent_sha256=admitted['receipt']['diagnostics']['latent_bank_sha256'],
                   comparison_reference=comparison_reference,
                   native_t0_reference=native_t0_reference,
                   cache_index_sha256=INDEX_SHA, upstream_revision=source.PINNED_REVISION,
                   upstream_sources=admitted['source_hashes'],
                   shared_scene_plan=plan, schedule=source.single._schedule_record(scheduler),
                   context_forward_plan=(
                       generated_context_plan() if admitted['t0_context'] == 'generated' else plan
                   ),
                   t0_context=t0_context_record,
                   frozen_input_identity=admitted['frozen_input_identity'],
                   history_visibility=admitted['history_visibility'],
                   attention=state['receipt'], seconds_through_terminal=elapsed,
                   intermediate_readbacks=0, explicit_mps_synchronize_calls=0, terminal_transfer_count=1,
                   one_resident_bf16_block=True, persistent_layer_caches=54,
                   scope=(
                       'saved MPS T0 noise denoised with pinned upstream equations, then separate writer '
                       'and frozen-noise T1 continuation; no photo conditioning or likeness claim'
                       if admitted['t0_context'] == 'generated'
                       else 'supplied clean photo-reference T0 plus saved Qwen; upstream five-forward DiT; encoders excluded'
                   ),
                   comparison=t1_comparison,
                   endpoint_comparisons=dict(t0=t0_comparison, t1=t1_comparison),
                   comparisons_passed=comparisons_passed,
                   result=dict(path=str(latent_path), shape=result_shape,
                               sha256=source._sha256(latent_path), endpoints=endpoints),
                   capture_payload=dict(path=str(packed_path), sha256=source._sha256(packed_path), records=records))
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: receipt[key] for key in (
        'endpoint_comparisons', 'seconds_through_terminal', 'result'
    )}), flush=True)
    if comparisons_passed is False:
        raise ValueError('upstream/native endpoint comparison exceeded the admitted 2e-5 tolerance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--upstream', type=Path, default=Path('/opt/mldata_ssd/models/hy-nomas/upstream-911cea6458a8'))
    parser.add_argument('--cache-index', type=Path, default=source.single.DEFAULT_CACHE_INDEX)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--t0-context', choices=('photo', 'generated'), default='photo',
                        help='use the technical photo T0 by default, or denoise saved T0 noise with pinned source equations')
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
    parser.add_argument('--native-t0-latents', type=Path,
                        help='optional same-arm native clean T0 raw F32 segment for the 2e-5 gate')
    parser.add_argument('--native-t0-latents-sha256',
                        help='required SHA-256 when --native-t0-latents is supplied')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    try:
        validate_t0_context(
            args.t0_context,
            args.t0_history_keep_latents,
            args.hide_t0_caption_keys,
            args.t1_current_text_kv,
        )
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
            t0_context=admitted['t0_context'],
            forward_plan=(
                generated_context_plan()
                if admitted['t0_context'] == 'generated'
                else shared_scene_plan(admitted['history_visibility']['t1_current_text_kv'])
            ),
            history_visibility=admitted['history_visibility'],
            frozen_input_identity=admitted['frozen_input_identity'],
            comparison_native=(
                dict(
                    kind='authenticated same-arm native T1 latent segment',
                    path=str(admitted['comparison_latents']['path']),
                    sha256=admitted['comparison_latents']['sha256'],
                    shape=admitted['comparison_latents']['shape'],
                )
                if admitted['comparison_latents'] is not None
                else dict(kind='authenticated historical native T1 bank slice', sha256=RECEIPT_SHA)
                if admitted['t0_context'] == 'photo'
                else None
            ),
            comparison_native_t0=(
                dict(kind='authenticated native T0 endpoint',
                     path=str(admitted['native_t0_endpoint']['path']),
                     sha256=admitted['native_t0_endpoint']['sha256'],
                     shape=admitted['native_t0_endpoint']['shape'])
                if admitted['native_t0_endpoint'] is not None else None
            ),
            output_dir=str(args.output_dir),
        )))


if __name__ == '__main__':
    main()
