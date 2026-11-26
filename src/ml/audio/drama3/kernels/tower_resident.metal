#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Compact kernels used by the resident Drama 3 tower. All activations and
// token/code ids are FP32 buffers; integer ids are exactly representable.

kernel void drama3_tower_embed(
    device const float *table [[buffer(0)]],
    device const float *code [[buffer(1)]],
    device const float *main [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant ulong *p [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  const uint vocab = uint(p[0]);
  const uint dim = uint(p[1]);
  const int semantic_begin = int(p[2]);
  const int semantic_end = int(p[3]);
  if (i >= dim) return;
  const int token = int(main[0]);
  const int id = clamp(int(code[0]), 0, int(vocab) - 1);
  output[i] = token >= semantic_begin && token <= semantic_end
      ? table[uint(id) * dim + i] : 0.0f;
}

kernel void drama3_tower_clip_code(
    device const float *code [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant ulong *p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i == 0) output[0] = float(clamp(int(code[0]), 0, int(p[0]) - 1));
}

kernel void drama3_tower_codebook_row(
    device const float *frame [[buffer(0)]],
    device float *row [[buffer(1)]],
    constant ulong *p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i != 0) return;
  const uint vocab = uint(p[0]);
  const uint codebook = uint(p[1]);
  const uint frame_code_index = uint(p[2]);
  const int code = clamp(int(frame[frame_code_index]), 0, int(vocab) - 1);
  row[0] = float(codebook * vocab + uint(code));
}

kernel void drama3_tower_mask_semantic(
    device const float *input [[buffer(0)]],
    device const float *main [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant ulong *p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= uint(p[0])) return;
  const int token = int(main[0]);
  output[i] = token >= int(p[1]) && token <= int(p[2]) ? input[i] : 0.0f;
}

inline uint drama3_tower_load_u32(device const uchar *bytes, ulong offset) {
  return uint(bytes[offset]) | uint(bytes[offset + 1]) << 8 |
      uint(bytes[offset + 2]) << 16 | uint(bytes[offset + 3]) << 24;
}

inline float drama3_tower_decode(device const uchar *bytes, ulong offset,
                                 ulong index, ulong dtype) {
  if (dtype == 0) return as_type<float>(drama3_tower_load_u32(bytes, offset + index * 4));
  const ulong position = offset + index * 2;
  const ushort word = ushort(bytes[position]) | ushort(bytes[position + 1]) << 8;
  return dtype == 1 ? as_type<float>(uint(word) << 16) : float(as_type<half>(word));
}

kernel void drama3_tower_unseen_embedding(
    device const uchar *table [[buffer(0)]],
    device const float *frame [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant ulong *p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  const uint vocab = uint(p[0]);
  const uint dim = uint(p[1]);
  const uint first_codebook = uint(p[2]);
  const uint end_codebook = uint(p[3]);
  const int semantic_begin = int(p[4]);
  const int semantic_end = int(p[5]);
  const ulong table_offset = p[6];
  const ulong dtype = p[7];
  if (i >= dim) return;
  const int main = int(frame[0]);
  float sum = 0.0f;
  if (main >= semantic_begin && main <= semantic_end) {
    for (uint cb = first_codebook; cb < end_codebook; ++cb) {
      const int id = clamp(int(frame[cb + 1]), 0, int(vocab) - 1);
      const ulong index = (ulong(cb) * vocab + uint(id)) * dim + i;
      sum += drama3_tower_decode(table, table_offset, index, dtype);
    }
  }
  output[i] = sum;
}

kernel void drama3_tower_embed_backbone(
    device const float *frame [[buffer(0)]],
    device const float *text [[buffer(1)]],
    device const float *extra_embedding [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant ulong *p [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  const uint dim = uint(p[1]);
  const uint vq_codebooks = uint(p[2]);
  const int semantic_begin = int(p[3]);
  const int semantic_end = int(p[4]);
  const float scale = as_type<float>(uint(p[5]));
  if (i >= dim) return;
  const int main = int(frame[0]);
  if (main < semantic_begin || main > semantic_end || vq_codebooks == 0) {
    output[i] = text[i];
    return;
  }
  output[i] = (text[i] + extra_embedding[i]) / scale;
}

kernel void drama3_tower_add_embedding(
    device float *embedding [[buffer(0)]],
    device const float *unseen [[buffer(1)]],
    constant ulong *p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  if (i < uint(p[0])) embedding[i] += unseen[i];
}

kernel void drama3_tower_concat(
    device const float *embedding [[buffer(0)]],
    device const float *hidden [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant ulong *p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  const uint dim = uint(p[0]);
  if (i < dim) output[i] = embedding[i];
  else if (i < 2 * dim) output[i] = hidden[i - dim];
}

kernel void drama3_tower_append_kv(
    device const float *key [[buffer(0)]],
    device const float *value [[buffer(1)]],
    device float *key_cache [[buffer(2)]],
    device float *value_cache [[buffer(3)]],
    constant ulong *p [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  const uint heads = uint(p[0]);
  const uint head_dim = uint(p[1]);
  const uint capacity = uint(p[2]);
  const uint position = uint(p[3]);
  const uint width = heads * head_dim;
  if (i >= width) return;
  const uint head = i / head_dim;
  const uint dimension = i % head_dim;
  const uint destination = (head * capacity + position) * head_dim + dimension;
  key_cache[destination] = key[i];
  value_cache[destination] = value[i];
}

kernel void drama3_tower_relocate_kv(
    device const float *old_key_cache [[buffer(0)]],
    device const float *old_value_cache [[buffer(1)]],
    device float *new_key_cache [[buffer(2)]],
    device float *new_value_cache [[buffer(3)]],
    constant ulong *p [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  const uint heads = uint(p[0]);
  const uint head_dim = uint(p[1]);
  const uint old_capacity = uint(p[2]);
  const uint new_capacity = uint(p[3]);
  const uint length = uint(p[4]);
  const uint count = heads * length * head_dim;
  if (i >= count) return;
  const uint head = i / (length * head_dim);
  const uint time = (i / head_dim) % length;
  const uint dimension = i % head_dim;
  const uint old_index = (head * old_capacity + time) * head_dim + dimension;
  const uint new_index = (head * new_capacity + time) * head_dim + dimension;
  new_key_cache[new_index] = old_key_cache[old_index];
  new_value_cache[new_index] = old_value_cache[old_index];
}

kernel void drama3_tower_compact_cache(
    device const float *cache [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant ulong *p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  const uint heads = uint(p[0]);
  const uint head_dim = uint(p[1]);
  const uint capacity = uint(p[2]);
  const uint length = uint(p[3]);
  const uint count = heads * head_dim * length;
  if (i >= count) return;
  const uint head = i / (head_dim * length);
  const uint time = (i / head_dim) % length;
  const uint dimension = i % head_dim;
  output[i] = cache[(head * capacity + time) * head_dim + dimension];
}

kernel void drama3_tower_attention_scores(
    device const float *query [[buffer(0)]],
    device const float *key_cache [[buffer(1)]],
    device float *scores [[buffer(2)]],
    constant ulong *p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  const uint n_head = uint(p[0]);
  const uint n_local_heads = uint(p[1]);
  const uint head_dim = uint(p[2]);
  const uint heads_per_kv = uint(p[3]);
  const uint length = uint(p[4]);
  const uint capacity = uint(p[5]);
  const float scale = as_type<float>(uint(p[6]));
  if (i >= n_head * length) return;
  const uint head = i / length;
  const uint time = i % length;
  const uint kv_head = head / heads_per_kv;
  float score = 0.0f;
  for (uint d = 0; d < head_dim; ++d) {
    score += query[head * head_dim + d] * key_cache[(kv_head * capacity + time) * head_dim + d];
  }
  scores[i] = score * scale;
}

inline float drama3_tower_sigmoid(float x) {
  return x >= 0.0f ? 1.0f / (1.0f + exp(-x)) : exp(x) / (1.0f + exp(x));
}

kernel void drama3_tower_attention_context(
    device const float *scores [[buffer(0)]],
    device const float *value_cache [[buffer(1)]],
    device const float *gate [[buffer(2)]],
    device float *context [[buffer(3)]],
    constant ulong *p [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  const uint n_head = uint(p[0]);
  const uint head_dim = uint(p[1]);
  const uint heads_per_kv = uint(p[2]);
  const uint length = uint(p[3]);
  const uint capacity = uint(p[4]);
  const uint count = n_head * head_dim;
  if (i >= count) return;
  const uint head = i / head_dim;
  const uint dimension = i % head_dim;
  const uint kv_head = head / heads_per_kv;
  float maximum = -INFINITY;
  for (uint time = 0; time < length; ++time) {
    maximum = max(maximum, scores[head * length + time]);
  }
  float denominator = 0.0f;
  for (uint time = 0; time < length; ++time) {
    denominator += exp(scores[head * length + time] - maximum);
  }
  float value = 0.0f;
  for (uint time = 0; time < length; ++time) {
    const float weight = exp(scores[head * length + time] - maximum);
    value += (weight / denominator) * value_cache[(kv_head * capacity + time) * head_dim + dimension];
  }
  context[i] = value * drama3_tower_sigmoid(gate[i]);
}

kernel void drama3_tower_initial_frame(
    device const float *main [[buffer(0)]],
    device float *frame [[buffer(1)]],
    constant ulong *p [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
  const uint count = uint(p[0]);
  const int semantic_begin = int(p[1]);
  const uint vocab = uint(p[2]);
  if (i >= count) return;
  if (i == 0) frame[i] = main[0];
  else if (i == 1) frame[i] = float(clamp(int(main[0]) - semantic_begin, 0, int(vocab) - 1));
  else frame[i] = 0.0f;
}
