"""Independent Qwen-Image 2.1 FlowMatch schedule and Euler oracle."""

import math

import numpy as np


steps = 4
image_seq_len = 256
base_seq_len = 256
max_seq_len = 8192
base_shift = 0.5
max_shift = 0.9
shift_terminal = 0.02

slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
mu = image_seq_len * slope + (base_shift - slope * base_seq_len)
sigmas = np.linspace(1.0, 1.0 / steps, steps).astype(np.float32)
exponential = math.exp(mu)
sigmas = exponential / (exponential + (1.0 / sigmas - 1.0))
scale = (1.0 - sigmas[-1]) / (1.0 - shift_terminal)
sigmas = 1.0 - (1.0 - sigmas) / scale
sigmas = np.concatenate([sigmas, np.zeros(1, dtype=np.float32)])

latents = np.array([0.25, -0.5, 1.25], dtype=np.float32)
for index in range(steps):
    model_output = latents * np.float32(0.25) + np.float32(sigmas[index] * (index + 1))
    latents = latents + np.float32(sigmas[index + 1] - sigmas[index]) * model_output

print("mu", mu)
print("sigmas", sigmas.tolist())
print("timesteps", (sigmas[:-1] * 1000.0).tolist())
print("latents", latents.tolist())
