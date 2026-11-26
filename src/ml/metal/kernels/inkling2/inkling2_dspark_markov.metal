#include <metal_stdlib>
using namespace metal;

static inline float inkling2_dspark_markov_bf16_to_f32(ushort value) {
    return as_type<float>(uint(value) << 16);
}

// Assemble [state, embedding(previous), hidden(candidate), last_context] for
// the trained rnn_proj matrix. The first transition deliberately ignores the
// uninitialized scratch state and starts from zeros on every encode call.
kernel void inkling2_dspark_markov_input(
    device const float *hidden [[buffer(0)]],
    device const float *last_context [[buffer(1)]],
    device const float *state [[buffer(2)]],
    device const ushort *markov_embedding [[buffer(3)]],
    device const int *output_ids [[buffer(4)]],
    device float *input [[buffer(5)]],
    constant uint &anchor_id [[buffer(6)]],
    constant uint &step [[buffer(7)]],
    constant uint &hidden_size [[buffer(8)]],
    constant uint &rank [[buffer(9)]],
    constant uint &vocab_size [[buffer(10)]],
    uint index [[thread_position_in_grid]]) {
    const uint width = 2u * rank + 2u * hidden_size;
    if (index >= width) return;

    if (index < rank) {
        input[index] = step == 0u ? 0.0f : state[index];
        return;
    }
    if (index < 2u * rank) {
        const int previous_id = step == 0u ? int(anchor_id) : output_ids[step - 1u];
        input[index] = previous_id >= 0 && uint(previous_id) < vocab_size
            ? inkling2_dspark_markov_bf16_to_f32(
                markov_embedding[uint(previous_id) * rank + index - rank])
            : 0.0f;
        return;
    }

    const uint hidden_index = index - 2u * rank;
    if (hidden_index < hidden_size) {
        input[index] = hidden[(step + 1u) * hidden_size + hidden_index];
    } else {
        input[index] = last_context[hidden_index - hidden_size];
    }
}

// Apply the three rnn_proj slices, updating recurrent state and producing the
// tanh latent. FP32 state/intermediates are retained for the resident path.
kernel void inkling2_dspark_markov_transition(
    device const float *raw [[buffer(0)]],
    device const ushort *bias [[buffer(1)]],
    device float *state [[buffer(2)]],
    device float *latent [[buffer(3)]],
    constant uint &rank [[buffer(4)]],
    constant uint &step [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= rank) return;

    const float gate_pre = raw[index] + inkling2_dspark_markov_bf16_to_f32(bias[index]);
    const float candidate_pre = raw[rank + index] + inkling2_dspark_markov_bf16_to_f32(bias[rank + index]);
    const float output_pre = raw[2u * rank + index] + inkling2_dspark_markov_bf16_to_f32(bias[2u * rank + index]);
    const float gate = 1.0f / (1.0f + exp(-gate_pre));
    const float prior = step == 0u ? 0.0f : state[index];
    const float candidate = tanh(candidate_pre);
    state[index] = gate * prior + (1.0f - gate) * candidate;
    latent[index] = tanh(output_pre);
}

// Form the direct hidden state x + hidden_out(latent), including its learned
// bias, before the target LM head projection.
kernel void inkling2_dspark_markov_adjust_hidden(
    device const float *hidden [[buffer(0)]],
    device const float *delta [[buffer(1)]],
    device const ushort *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &hidden_size [[buffer(4)]],
    constant uint &step [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= hidden_size) return;
    output[index] = hidden[(step + 1u) * hidden_size + index] + delta[index] +
        inkling2_dspark_markov_bf16_to_f32(bias[index]);
}
