"""Independent tiny-block oracle for spec/qwen_image21_block_spec.cr."""

import math
import torch
import torch.nn.functional as F

torch.set_printoptions(precision=8, linewidth=200)

tokens, dim, head_dim, intermediate = 4, 6, 6, 5


def matrix(out_dim: int, in_dim: int, phase: int) -> torch.Tensor:
    values = [((i * 17 + phase * 13) % 29 - 14) / 37.0 for i in range(out_dim * in_dim)]
    return torch.tensor(values, dtype=torch.float32).reshape(out_dim, in_dim)


hidden = torch.tensor([((i * 11) % 23 - 11) / 19.0 for i in range(tokens * dim)]).reshape(tokens, dim)
mod = torch.tensor([((i * 7) % 31 - 15) / 101.0 for i in range(tokens * 4 * dim)]).reshape(tokens, 4 * dim)
positions = torch.tensor([[0, 0, 0], [1, 1, 1], [2, -1, 0], [2, 0, 0]])
image_ids = torch.tensor([-1, -1, 0, 0])

q_weight, k_weight, v_weight, out_weight = (matrix(6, 6, phase) for phase in range(1, 5))
norm_q = torch.tensor([1.0, 0.9, 1.1, 0.8, 1.2, 0.95])
norm_k = torch.tensor([0.85, 1.05, 0.9, 1.15, 0.8, 1.1])
gate_up = matrix(10, 6, 5)
mlp_out = matrix(6, 5, 6)

mod1_scale, mod1_gate, mod2_scale, mod2_gate = mod.chunk(4, dim=-1)
x = F.layer_norm(hidden, (dim,), eps=1e-6) * (1 + mod1_scale)
q, k, v = F.linear(x, q_weight), F.linear(x, k_weight), F.linear(x, v_weight)
q = F.rms_norm(q.reshape(tokens, 1, head_dim), (head_dim,), norm_q, 1e-6)
k = F.rms_norm(k.reshape(tokens, 1, head_dim), (head_dim,), norm_k, 1e-6)
v = v.reshape(tokens, 1, head_dim)

for values in (q, k):
    for token in range(tokens):
        axis_offset = 0
        for axis, axis_dim in enumerate((2, 2, 2)):
            for pair in range(axis_dim // 2):
                angle = positions[token, axis] / (10000.0 ** (2 * pair / axis_dim))
                real, imag = values[token, 0, axis_offset + 2 * pair : axis_offset + 2 * pair + 2].clone()
                values[token, 0, axis_offset + 2 * pair] = real * math.cos(angle) - imag * math.sin(angle)
                values[token, 0, axis_offset + 2 * pair + 1] = real * math.sin(angle) + imag * math.cos(angle)
            axis_offset += axis_dim

scores = torch.einsum("thd,shd->hts", q, k) / math.sqrt(head_dim)
mask = torch.zeros(tokens, tokens, dtype=torch.bool)
for qi in range(tokens):
    for ki in range(tokens):
        mask[qi, ki] = qi >= ki or (image_ids[qi] >= 0 and image_ids[qi] == image_ids[ki])
scores.masked_fill_(~mask.unsqueeze(0), float("-inf"))
attended = torch.einsum("hts,shd->thd", scores.softmax(-1), v).flatten(1)
state = hidden + mod1_gate.tanh() * F.linear(attended, out_weight)

x = F.layer_norm(state, (dim,), eps=1e-6) * (1 + mod2_scale)
fused = F.linear(x, gate_up)
gate, projection = fused.chunk(2, dim=-1)
result = state + mod2_gate.tanh() * F.linear(F.silu(gate) * projection, mlp_out)
print(result.flatten())
