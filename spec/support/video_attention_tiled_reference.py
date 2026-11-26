#!/usr/bin/env python3
"""Pinned PyTorch 2.8 MPS SDPA fixtures for opt-in native tiled F32 attention.

No model inference or CPU neural fallback; inputs and SDPA execute on MPS.
The optional long case uses the official first-request Wan key length, while
its query count remains bounded. This is a component fixture, not video speed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--long', action='store_true')
args = parser.parse_args()
assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
assert torch.__version__ == '2.8.0' and torch.backends.mps.is_available()
args.output.mkdir(parents=True, exist_ok=True)
torch.manual_seed(28371)
cases = [(2, 13, 97, 3, 64, 'normal'), (1, 64, 1024, 40, 128, 'normal'),
         (1, 9, 67, 2, 128, 'sharp'), (2, 1, 1, 1, 64, 'single'),
         (1, 9, 65, 2, 64, 'near-tie')]
if args.long:
    cases.append((1, 64, 36608, 40, 128, 'normal'))
manifest = {'torch': torch.__version__, 'device': 'mps', 'fallback_disabled': True, 'cases': []}
with torch.no_grad():
    for index, (b, nq, nk, h, d, profile) in enumerate(cases):
        name = f'b{b}-q{nq}-k{nk}-h{h}-d{d}'
        if profile != 'normal':
            name += f'-{profile}'
        q = torch.randn(b, nq, h, d, device='mps', dtype=torch.float32)
        k = torch.randn(b, nk, h, d, device='mps', dtype=torch.float32)
        v = torch.randn(b, nk, h, d, device='mps', dtype=torch.float32)
        if profile == 'sharp':
            q = q * 8
            k = k * 8
        elif profile == 'near-tie':
            k = k * 1e-5
        y = F.scaled_dot_product_attention(q.transpose(1,2).contiguous(),
            k.transpose(1,2).contiguous(), v.transpose(1,2).contiguous(),
            dropout_p=0.0, is_causal=False).transpose(1,2).contiguous()
        record = {'name': name, 'batches': b, 'query_rows': nq, 'key_rows': nk, 'heads': h, 'dim': d}
        for key, tensor in [('q',q), ('k',k), ('v',v), ('expected',y)]:
            assert torch.isfinite(tensor).all()
            target = args.output / f'{name}-{key}.bin'
            tensor.cpu().numpy().tofile(target)
            digest = hashlib.sha256()
            with target.open('rb') as stream:
                while chunk := stream.read(1024*1024): digest.update(chunk)
            record[key] = {'path': str(target.resolve()), 'bytes': target.stat().st_size, 'sha256': digest.hexdigest()}
        manifest['cases'].append(record)
        print(json.dumps({'generated': name, 'values': y.numel()}), flush=True)
        del q,k,v,y
        torch.mps.empty_cache()
(args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
