"""Independent tiny outer-transformer oracle for Qwen-Image 2.1."""

import math

import torch
import torch.nn.functional as F


torch.set_printoptions(precision=9, linewidth=240, sci_mode=False)

hidden_dim = 6
input_dim = 4
context_dim = 5
time_dim = 4
intermediate_dim = 3
axes_dims = (2, 2, 2)


def matrix(out_dim: int, in_dim: int, phase: int) -> torch.Tensor:
    values = [((i * 19 + phase * 11) % 37 - 18) / 43.0 for i in range(out_dim * in_dim)]
    return torch.tensor(values, dtype=torch.float32).reshape(out_dim, in_dim)


image_latents = torch.tensor(
    [((i * 13) % 31 - 15) / 29.0 for i in range(8 * input_dim)], dtype=torch.float32
).reshape(8, input_dim)
encoder_hidden = torch.tensor(
    [((i * 7) % 29 - 14) / 23.0 for i in range(4 * context_dim)], dtype=torch.float32
).reshape(4, context_dim)
encoder_valid = torch.tensor([True, True, False, True])
img_mask = torch.tensor([False, True, False, False, True])
img_shapes = [(1, 2, 2), (1, 2, 2)]
timestep = torch.tensor([0.625], dtype=torch.float32)

img_in = matrix(hidden_dim, input_dim, 1)
modulation_weight = matrix(4 * hidden_dim, hidden_dim, 2)
norm_out_weight = matrix(hidden_dim, hidden_dim, 3)
proj_out = matrix(input_dim, hidden_dim, 4)
time_linear_1 = matrix(hidden_dim, time_dim, 5)
time_linear_2 = matrix(hidden_dim, hidden_dim, 6)
text_in = matrix(hidden_dim, context_dim, 7)
text_out = matrix(hidden_dim, hidden_dim, 8)
text_norm = torch.tensor([0.10, -0.08, 0.04, 0.0, 0.12], dtype=torch.float32)

to_q = matrix(hidden_dim, hidden_dim, 9)
to_k = matrix(hidden_dim, hidden_dim, 10)
to_v = matrix(hidden_dim, hidden_dim, 11)
to_out = matrix(hidden_dim, hidden_dim, 12)
norm_q = torch.tensor([1.0, 0.9, 1.1, 0.8, 1.2, 0.95], dtype=torch.float32)
norm_k = torch.tensor([0.85, 1.05, 0.9, 1.15, 0.8, 1.1], dtype=torch.float32)
gate_up = matrix(2 * intermediate_dim, hidden_dim, 13)
mlp_out = matrix(hidden_dim, intermediate_dim, 14)

# Exact outer input projection and text projection.
projected_images = F.linear(image_latents, img_in)
text_rms = encoder_hidden.float() * torch.rsqrt(encoder_hidden.float().square().mean(-1, keepdim=True) + 1e-6)
projected_text = F.linear(F.gelu(F.linear(text_rms * (text_norm + 1), text_in), approximate="tanh"), text_out)

# The final True slot is a target placeholder appended by the pipeline. Each
# image slot expands to four latent tokens before the actual latents overwrite it.
repeats = torch.where(img_mask, 4, 1)
image_pad_mask = torch.repeat_interleave(img_mask, repeats)
base = torch.cat([projected_text, projected_text.new_zeros(1, hidden_dim)], dim=0)
joint = torch.repeat_interleave(base, repeats, dim=0)
joint[image_pad_mask] = projected_images

# Metadata uses shape boundaries, not runs of True values.
image_positions = image_pad_mask.nonzero(as_tuple=True)[0]
block_lengths = [math.prod(shape) for shape in img_shapes]
image_ids = torch.full((joint.shape[0],), -1, dtype=torch.long)
image_ids[image_positions] = torch.repeat_interleave(torch.arange(len(block_lengths)), torch.tensor(block_lengths))
target_mask = torch.zeros(joint.shape[0], dtype=torch.bool)
target_mask[image_positions[-block_lengths[-1] :]] = True

# Exact three-axis position construction.
positions = []
cursor = 0
position = 0
mask_list = image_pad_mask.tolist()
for _, height, width in img_shapes:
    block_start = mask_list.index(True, cursor)
    text_len = block_start - cursor
    positions.extend((p, p, p) for p in range(position, position + text_len))
    position += text_len
    for h in range(-(height - height // 2), height // 2):
        for w in range(-(width - width // 2), width // 2):
            positions.append((position, h, w))
    cursor = block_start + height * width
    position += max(height, width)
if cursor < joint.shape[0]:
    positions.extend((p, p, p) for p in range(position, position + joint.shape[0] - cursor))
positions = torch.tensor(positions, dtype=torch.long)

key_valid = torch.ones(joint.shape[0], dtype=torch.bool)
text_positions = (~image_pad_mask).nonzero(as_tuple=True)[0]
vlm_text_positions = ~img_mask[: encoder_valid.shape[0]]
key_valid[text_positions] = encoder_valid[vlm_text_positions]


def time_embedding(t: torch.Tensor) -> torch.Tensor:
    half = time_dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half)
    args = 1000.0 * t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


time_rows = torch.cat([timestep, timestep.new_zeros(1)])
temb = F.linear(F.silu(F.linear(time_embedding(time_rows), time_linear_1)), time_linear_2)
modulation = F.linear(F.silu(temb), modulation_weight)
selected_modulation = torch.where(target_mask[:, None], modulation[0], modulation[1])


def apply_rope(values: torch.Tensor) -> torch.Tensor:
    values = values.clone().reshape(joint.shape[0], 1, hidden_dim)
    for token in range(joint.shape[0]):
        offset = 0
        for axis, axis_dim in enumerate(axes_dims):
            for pair in range(axis_dim // 2):
                angle = positions[token, axis] / (10000.0 ** (2 * pair / axis_dim))
                real, imag = values[token, 0, offset + 2 * pair : offset + 2 * pair + 2].clone()
                values[token, 0, offset + 2 * pair] = real * math.cos(angle) - imag * math.sin(angle)
                values[token, 0, offset + 2 * pair + 1] = real * math.sin(angle) + imag * math.cos(angle)
            offset += axis_dim
    return values


mod1_scale, mod1_gate, mod2_scale, mod2_gate = selected_modulation.chunk(4, dim=-1)
x = F.layer_norm(joint, (hidden_dim,), eps=1e-6) * (1 + mod1_scale)
q = F.rms_norm(F.linear(x, to_q).reshape(-1, 1, hidden_dim), (hidden_dim,), norm_q, 1e-6)
k = F.rms_norm(F.linear(x, to_k).reshape(-1, 1, hidden_dim), (hidden_dim,), norm_k, 1e-6)
v = F.linear(x, to_v).reshape(-1, 1, hidden_dim)
q, k = apply_rope(q), apply_rope(k)

scores = torch.einsum("thd,shd->hts", q, k) / math.sqrt(hidden_dim)
mask = torch.zeros(joint.shape[0], joint.shape[0], dtype=torch.bool)
for qi in range(joint.shape[0]):
    for ki in range(joint.shape[0]):
        same_image = image_ids[qi] >= 0 and image_ids[qi] == image_ids[ki]
        mask[qi, ki] = (qi >= ki or same_image) and key_valid[ki]
scores.masked_fill_(~mask.unsqueeze(0), float("-inf"))
attended = torch.einsum("hts,shd->thd", scores.softmax(-1), v).flatten(1)
state = joint + mod1_gate.tanh() * F.linear(attended, to_out)

x = F.layer_norm(state, (hidden_dim,), eps=1e-6) * (1 + mod2_scale)
gate, projection = F.linear(x, gate_up).chunk(2, dim=-1)
joint = state + mod2_gate.tanh() * F.linear(F.silu(gate) * projection, mlp_out)

scale_rows = F.linear(F.silu(temb), norm_out_weight)
selected_scale = torch.where(target_mask[:, None], scale_rows[0], scale_rows[1])
output = F.linear(F.layer_norm(joint, (hidden_dim,), eps=1e-6) * (1 + selected_scale), proj_out)

print("image_pad_mask", image_pad_mask.tolist())
print("image_ids", image_ids.tolist())
print("target_mask", target_mask.tolist())
print("positions", positions.tolist())
print("key_valid", key_valid.tolist())
print("output", output.flatten())
