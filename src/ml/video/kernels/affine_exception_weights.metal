#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

// Private fixed-budget affine weights. All integers/floats in the payload are
// little endian. A 256-value affine group is codes[128], scale_f32, offset_f32.
static inline uint ax_u16(device const uchar *p) {
  return uint(p[0]) | (uint(p[1]) << 8);
}
static inline uint ax_u32(device const uchar *p) {
  return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) |
         (uint(p[3]) << 24);
}
static inline void ax_store16(device uchar *p, uint v) {
  p[0] = uchar(v);
  p[1] = uchar(v >> 8);
}
static inline void ax_store32(device uchar *p, uint v) {
  p[0] = uchar(v);
  p[1] = uchar(v >> 8);
  p[2] = uchar(v >> 16);
  p[3] = uchar(v >> 24);
}
static inline float ax_f32(device const uchar *p) {
  return as_type<float>(ax_u32(p));
}
static inline float ax_bf16(device const uchar *p) {
  return as_type<float>(ax_u16(p) << 16);
}
static inline float ax_base_value(device const uchar *group, uint local) {
  uint packed = uint(group[local >> 1]);
  uint code = (local & 1u) == 0u ? (packed & 15u) : (packed >> 4);
  float scale = ax_f32(group + 128u);
  float offset = ax_f32(group + 132u);
  float scaled = scale * float(code);
  return offset + scaled;
}
static inline bool ax_affine_valid(device const uchar *group) {
  float scale = ax_f32(group + 128u);
  float offset = ax_f32(group + 132u);
  float top_scaled = scale * 15.0f;
  float top = offset + top_scaled;
  return isfinite(scale) && scale >= 0.0f && isfinite(offset) &&
         isfinite(top_scaled) && isfinite(top);
}
static inline uint ax_quant(float value, float offset, float scale) {
  if (scale == 0.0f)
    return 0u;
  return uint(clamp(floor((value - offset) / scale + 0.5f), 0.0f, 15.0f));
}
static inline ulong ax_key(float score, uint index) {
  return (ulong(as_type<uint>(score)) << 32) | ulong(~index);
}
static inline ulong ax_threshold(device const uint *state) {
  return (ulong(state[0]) << 32) | ulong(state[1]);
}
static inline bool ax_pack_failed(device const atomic_uint *status) {
  return atomic_load_explicit(status + 0, memory_order_relaxed) != 0u ||
         atomic_load_explicit(status + 2, memory_order_relaxed) != 0u ||
         atomic_load_explicit(status + 3, memory_order_relaxed) != 0u;
}
static inline bool ax_decode_failed(device const atomic_uint *status) {
  return ax_pack_failed(status) ||
         atomic_load_explicit(status + 4, memory_order_relaxed) != 0u;
}

kernel void affine_exception_base_score(
    device const uchar *original [[buffer(0)]],
    device const float *calibration [[buffer(1)]],
    device uchar *affine [[buffer(2)]], device float *scores [[buffer(3)]],
    device atomic_uint *status [[buffer(4)]], constant uint &n [[buffer(5)]],
    constant uint &k [[buffer(6)]], constant uint &mode [[buffer(7)]],
    constant uint &fold [[buffer(8)]], uint t [[thread_index_in_threadgroup]],
    uint group_index [[threadgroup_position_in_grid]]) {
  threadgroup float mins[256];
  threadgroup float maxs[256];
  uint flat = group_index * 256u + t;
  float value = ax_bf16(original + flat * 2u);
  bool finite_value = isfinite(value);
  if (!finite_value) {
    atomic_fetch_add_explicit(status + 2, 1u, memory_order_relaxed);
    value = 0.0f;
  }
  mins[t] = value;
  maxs[t] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (t < stride) {
      mins[t] = min(mins[t], mins[t + stride]);
      maxs[t] = max(maxs[t], maxs[t + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float offset = mins[0];
  float scale = (maxs[0] - offset) / 15.0f;
  if (!isfinite(scale) || scale < 0.0f) {
    atomic_fetch_add_explicit(status + 2, 1u, memory_order_relaxed);
    scale = 0.0f;
  }
  device uchar *packed_group = affine + group_index * 136u;
  if (t == 0u) {
    ax_store32(packed_group + 128u, as_type<uint>(scale));
    ax_store32(packed_group + 132u, as_type<uint>(offset));
  }
  if (t < 128u) {
    uint lo = t * 2u;
    float a = ax_bf16(original + (group_index * 256u + lo) * 2u);
    float b = ax_bf16(original + (group_index * 256u + lo + 1u) * 2u);
    if (!isfinite(a))
      a = 0.0f;
    if (!isfinite(b))
      b = 0.0f;
    uint qa = ax_quant(a, offset, scale);
    uint qb = ax_quant(b, offset, scale);
    packed_group[t] = uchar(qa | (qb << 4));
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  float error = value - ax_base_value(packed_group, t);
  float score = error * error;
  if (mode == 1u) {
    uint channel = flat % k;
    float x0 = calibration[(fold * 2u) * k + channel];
    float x1 = calibration[(fold * 2u + 1u) * k + channel];
    float x0sq = x0 * x0;
    float x1sq = x1 * x1;
    float mean_x2 = (x0sq + x1sq) * 0.5f;
    score = score * mean_x2;
    if (!isfinite(x0) || !isfinite(x1) || !isfinite(score)) {
      atomic_fetch_add_explicit(status + 2, 1u, memory_order_relaxed);
      score = 0.0f;
    }
  } else if (!isfinite(score)) {
    atomic_fetch_add_explicit(status + 2, 1u, memory_order_relaxed);
    score = 0.0f;
  }
  scores[flat] = score;
}

kernel void affine_exception_candidate_score(
    device const float *scores [[buffer(0)]],
    device float *candidates [[buffer(1)]],
    device atomic_uint *status [[buffer(2)]], constant uint &n [[buffer(3)]],
    constant uint &k [[buffer(4)]], constant uint &mode [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
  uint count = mode == 2u ? k : n;
  if (i >= count)
    return;
  float sum = 0.0f;
  if (mode == 2u) {
    for (uint row = 0u; row < n; ++row)
      sum = sum + scores[row * k + i];
  } else {
    for (uint channel = 0u; channel < k; ++channel)
      sum = sum + scores[i * k + channel];
  }
  if (!isfinite(sum) || sum < 0.0f) {
    atomic_fetch_add_explicit(status + 2, 1u, memory_order_relaxed);
    sum = 0.0f;
  }
  candidates[i] = sum;
}

kernel void affine_exception_rank_init(device uint *state [[buffer(0)]],
                                       constant uint &target [[buffer(1)]],
                                       uint i [[thread_position_in_grid]]) {
  if (i == 0u) {
    state[0] = 0u;
    state[1] = 0u;
    state[2] = target == 0u ? 0u : target - 1u;
    state[3] = 0u;
  }
}

kernel void affine_exception_rank_hist(device const float *scores [[buffer(0)]],
                                       device uint *histograms [[buffer(1)]],
                                       device const uint *state [[buffer(2)]],
                                       constant uint &count [[buffer(3)]],
                                       constant uint &shift [[buffer(4)]],
                                       uint t [[thread_index_in_threadgroup]],
                                       uint group
                                       [[threadgroup_position_in_grid]]) {
  threadgroup atomic_uint bins[256];
  atomic_store_explicit(bins + t, 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong prefix = (ulong(state[0]) << 32) | ulong(state[1]);
  uint prior_bits = 56u - shift;
  uint start = group * 4096u + t;
  uint group_end = min(count, (group + 1u) * 4096u);
  for (uint i = start; i < group_end; i += 256u) {
    ulong key = ax_key(scores[i], i);
    bool matches = prior_bits == 0u || (key >> (64u - prior_bits)) ==
                                           (prefix >> (64u - prior_bits));
    if (matches) {
      uint digit = uint((key >> shift) & 255ul);
      atomic_fetch_add_explicit(bins + digit, 1u, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  histograms[group * 256u + t] =
      atomic_load_explicit(bins + t, memory_order_relaxed);
}

kernel void affine_exception_rank_reduce(device const uint *histograms
                                         [[buffer(0)]],
                                         device uint *digits [[buffer(1)]],
                                         constant uint &groups [[buffer(2)]],
                                         uint bin
                                         [[thread_position_in_threadgroup]]) {
  uint total = 0u;
  for (uint group = 0u; group < groups; ++group)
    total += histograms[group * 256u + bin];
  digits[bin] = total;
}

kernel void affine_exception_rank_digit(device const uint *digits [[buffer(0)]],
                                        device uint *state [[buffer(1)]],
                                        constant uint &shift [[buffer(2)]],
                                        uint i [[thread_position_in_grid]]) {
  if (i != 0u)
    return;
  uint rank = state[2];
  uint chosen = 0u;
  bool found = false;
  for (int bin = 255; bin >= 0; --bin) {
    uint amount = digits[uint(bin)];
    if (rank < amount) {
      chosen = uint(bin);
      found = true;
      break;
    }
    rank -= amount;
  }
  if (!found) {
    state[3] = 1u;
    return;
  }
  if (shift >= 32u)
    state[0] |= chosen << (shift - 32u);
  else
    state[1] |= chosen << shift;
  state[2] = rank;
}

kernel void affine_exception_zero_payload(device uchar *payload [[buffer(0)]],
                                          constant uint &bytes [[buffer(1)]],
                                          uint i [[thread_position_in_grid]]) {
  if (i < bytes)
    payload[i] = 0u;
}

kernel void affine_exception_candidate_select(
    device const float *scores [[buffer(0)]],
    device const uint *state [[buffer(1)]], device uint *flags [[buffer(2)]],
    constant uint &count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= count)
    return;
  ulong threshold = ax_threshold(state);
  flags[i] = state[3] == 0u && ax_key(scores[i], i) >= threshold ? 1u : 0u;
}

kernel void affine_exception_candidate_scan(
    device const uint *flags [[buffer(0)]], device uint *prefix [[buffer(1)]],
    device uint *group_sums [[buffer(2)]], constant uint &count [[buffer(3)]],
    uint t [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  threadgroup uint scan[256];
  uint index = group * 256u + t;
  uint flag = index < count ? flags[index] : 0u;
  scan[t] = flag;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 1u; stride < 256u; stride <<= 1u) {
    uint add = t >= stride ? scan[t - stride] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scan[t] += add;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (index < count)
    prefix[index] = scan[t] - flag;
  if (t == 255u)
    group_sums[group] = scan[t];
}

kernel void affine_exception_candidate_scan_groups(
    device const uint *group_sums [[buffer(0)]],
    device uint *group_offsets [[buffer(1)]],
    device atomic_uint *status [[buffer(3)]],
    constant uint &groups [[buffer(2)]], constant uint &target [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  if (i != 0u)
    return;
  uint total = 0u;
  for (uint group = 0u; group < groups; ++group) {
    group_offsets[group] = total;
    total += group_sums[group];
  }
  if (total != target)
    atomic_fetch_add_explicit(status + 3, 1u, memory_order_relaxed);
}

kernel void affine_exception_candidate_scan_finish(
    device uint *prefix [[buffer(0)]],
    device const uint *group_offsets [[buffer(1)]],
    constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < count)
    prefix[i] += group_offsets[i / 256u];
}

kernel void affine_exception_copy_base(device const uchar *affine [[buffer(0)]],
                                       device uchar *payload [[buffer(1)]],
                                       constant uint &bytes [[buffer(2)]],
                                       uint i [[thread_position_in_grid]]) {
  if (i < bytes)
    payload[i] = affine[i];
}

kernel void affine_exception_sparse_counts(
    device const float *scores [[buffer(0)]],
    device const uint *state [[buffer(1)]], device uchar *payload [[buffer(2)]],
    device atomic_uint *status [[buffer(3)]],
    constant uint &records [[buffer(4)]], constant uint &target [[buffer(5)]],
    constant uint &counts_offset [[buffer(6)]],
    uint t [[thread_index_in_threadgroup]],
    uint record [[threadgroup_position_in_grid]]) {
  uint count = 0u;
  ulong threshold = ax_threshold(state);
  for (uint local = t; local < 1024u; local += 256u) {
    uint index = record * 1024u + local;
    count += uint(ax_key(scores[index], index) >= threshold && state[3] == 0u);
  }
  threadgroup uint counts[256];
  counts[t] = count;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (t < stride)
      counts[t] += counts[t + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (t == 0u) {
    device uchar *dst = payload + counts_offset + record * 4u;
    ax_store32(dst, counts[0]);
    if (counts[0] > 1024u || record >= records || target > records * 1024u)
      atomic_fetch_add_explicit(status + 3, 1u, memory_order_relaxed);
  }
}

kernel void
affine_exception_sparse_offsets(device uchar *payload [[buffer(0)]],
                                device atomic_uint *status [[buffer(1)]],
                                constant uint &records [[buffer(2)]],
                                constant uint &target [[buffer(3)]],
                                constant uint &offsets_offset [[buffer(4)]],
                                uint i [[thread_position_in_grid]]) {
  if (i != 0u)
    return;
  uint running = 0u;
  ax_store32(payload + offsets_offset, 0u);
  for (uint record = 0u; record < records; ++record) {
    uint count = ax_u32(payload + records * 544u + record * 4u);
    if (count > 1024u || running > target || count > target - running) {
      atomic_fetch_add_explicit(status + 3, 1u, memory_order_relaxed);
      count = 0u;
    }
    running += count;
    ax_store32(payload + offsets_offset + (record + 1u) * 4u, running);
  }
  if (running != target)
    atomic_fetch_add_explicit(status + 3, 1u, memory_order_relaxed);
}

kernel void affine_exception_sparse_pack(
    device const uchar *original [[buffer(0)]],
    device const float *scores [[buffer(1)]],
    device const uint *state [[buffer(2)]], device uchar *payload [[buffer(3)]],
    device atomic_uint *status [[buffer(4)]],
    constant uint &records [[buffer(5)]], constant uint &target [[buffer(6)]],
    constant uint &offsets_offset [[buffer(7)]],
    constant uint &patches_offset [[buffer(8)]],
    uint t [[thread_index_in_threadgroup]],
    uint record [[threadgroup_position_in_grid]]) {
  if (record >= records || state[3] != 0u)
    return;
  uint start = ax_u32(payload + offsets_offset + record * 4u);
  uint end = ax_u32(payload + offsets_offset + (record + 1u) * 4u);
  uint expected = ax_u32(payload + records * 544u + record * 4u);
  if (start > target || end > target || end < start ||
      end - start != expected) {
    if (t == 0u)
      atomic_fetch_add_explicit(status + 3, 1u, memory_order_relaxed);
    return;
  }
  threadgroup uint scan[256];
  threadgroup uint prior;
  if (t == 0u)
    prior = 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong threshold = ax_threshold(state);
  for (uint chunk = 0u; chunk < 4u; ++chunk) {
    uint local = chunk * 256u + t;
    uint index = record * 1024u + local;
    uint selected = uint(ax_key(scores[index], index) >= threshold);
    scan[t] = selected;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < 256u; stride <<= 1u) {
      uint add = t >= stride ? scan[t - stride] : 0u;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      scan[t] += add;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (selected != 0u) {
      uint rank = prior + scan[t] - 1u;
      if (rank >= expected) {
        atomic_fetch_add_explicit(status + 3, 1u, memory_order_relaxed);
      } else {
        device uchar *entry = payload + patches_offset + (start + rank) * 4u;
        ax_store16(entry, local);
        uint original_bits = ax_u16(original + index * 2u);
        ax_store16(entry + 2u, original_bits);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 255u)
      prior += scan[255];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

kernel void affine_exception_k_layout(device const uint *flags [[buffer(0)]],
                                      device const uint *prefix [[buffer(1)]],
                                      device uchar *payload [[buffer(2)]],
                                      device atomic_uint *status [[buffer(3)]],
                                      constant uint &k [[buffer(4)]],
                                      constant uint &count [[buffer(5)]],
                                      constant uint &map_offset [[buffer(6)]],
                                      constant uint &ids_offset [[buffer(7)]],
                                      constant uint &sidecar_offset
                                      [[buffer(8)]],
                                      uint i [[thread_position_in_grid]]) {
  if (i >= k)
    return;
  uint ordinal = prefix[i];
  if (flags[i] != 0u) {
    if (ordinal >= count || i > 0xffffu) {
      atomic_fetch_add_explicit(status + 3, 1u, memory_order_relaxed);
      ax_store16(payload + map_offset + i * 2u, 0xffffu);
    } else {
      ax_store16(payload + map_offset + i * 2u, ordinal);
      ax_store16(payload + ids_offset + ordinal * 2u, i);
    }
  } else {
    ax_store16(payload + map_offset + i * 2u, 0xffffu);
  }
}

kernel void affine_exception_k_sidecar(
    device const uchar *original [[buffer(0)]],
    device uchar *payload [[buffer(1)]], device const uint *flags [[buffer(2)]],
    device const uint *prefix [[buffer(3)]], constant uint &n [[buffer(4)]],
    constant uint &k [[buffer(5)]], constant uint &count [[buffer(6)]],
    constant uint &map_offset [[buffer(7)]],
    constant uint &sidecar_offset [[buffer(8)]],
    uint i [[thread_position_in_grid]]) {
  uint total = n * k;
  if (i >= total)
    return;
  uint channel = i % k;
  if (flags[channel] == 0u)
    return;
  uint ordinal = prefix[channel];
  if (ordinal >= count)
    return;
  uint row = i / k;
  uint bits = ax_u16(original + i * 2u);
  ax_store16(payload + sidecar_offset + (row * count + ordinal) * 2u, bits);
}

kernel void affine_exception_n_directory(
    device const uint *flags [[buffer(0)]],
    device const uint *prefix [[buffer(1)]],
    device uchar *payload [[buffer(2)]],
    device atomic_uint *status [[buffer(3)]], constant uint &n [[buffer(4)]],
    constant uint &count [[buffer(5)]],
    constant uint &directory_bytes [[buffer(6)]],
    constant uint &row_base_bytes [[buffer(7)]], constant uint &k [[buffer(8)]],
    uint i [[thread_position_in_grid]]) {
  if (i > n)
    return;
  uint selected_before = i == n ? count : prefix[i];
  ulong offset = ulong(directory_bytes) + ulong(i) * row_base_bytes +
                 ulong(selected_before) * (ulong(k) * 2ul - row_base_bytes);
  if (offset > 0xfffffffful) {
    atomic_fetch_add_explicit(status + 3, 1u, memory_order_relaxed);
    offset = directory_bytes;
  }
  ax_store32(payload + i * 4u, uint(offset));
}

kernel void affine_exception_n_rows(device const uchar *original [[buffer(0)]],
                                    device const uchar *affine [[buffer(1)]],
                                    device const uint *flags [[buffer(2)]],
                                    device uchar *payload [[buffer(3)]],
                                    device atomic_uint *status [[buffer(4)]],
                                    constant uint &n [[buffer(5)]],
                                    constant uint &k [[buffer(6)]],
                                    constant uint &directory_bytes
                                    [[buffer(7)]],
                                    constant uint &row_base_bytes [[buffer(8)]],
                                    uint t [[thread_index_in_threadgroup]],
                                    uint row [[threadgroup_position_in_grid]]) {
  if (row >= n)
    return;
  uint start = ax_u32(payload + row * 4u);
  uint end = ax_u32(payload + (row + 1u) * 4u);
  bool raw = flags[row] != 0u;
  uint length = raw ? k * 2u : row_base_bytes;
  uint payload_bytes = (n * k / 256u) * 144u;
  if (start < directory_bytes || end < start || end - start != length ||
      end > payload_bytes) {
    if (t == 0u)
      atomic_fetch_add_explicit(status + 3, 1u, memory_order_relaxed);
    return;
  }
  device const uchar *source =
      raw ? original + row * k * 2u : affine + row * row_base_bytes;
  for (uint byte = t; byte < length; byte += 256u)
    payload[start + byte] = source[byte];
}

kernel void affine_exception_validate_init(device atomic_uint *status
                                           [[buffer(0)]],
                                           uint i [[thread_position_in_grid]]) {
  if (i == 0u)
    atomic_store_explicit(status + 4, 0u, memory_order_relaxed);
}

kernel void
affine_exception_validate_sparse(device const uchar *payload [[buffer(0)]],
                                 device atomic_uint *status [[buffer(1)]],
                                 constant uint &records [[buffer(2)]],
                                 constant uint &patch_count [[buffer(3)]],
                                 uint t [[thread_index_in_threadgroup]],
                                 uint record [[threadgroup_position_in_grid]]) {
  if (record >= records)
    return;
  uint base_bytes = records * 544u;
  uint counts_offset = base_bytes;
  uint offsets_offset = counts_offset + records * 4u;
  uint patches_offset = offsets_offset + (records + 1u) * 4u;
  uint bad = uint(ax_pack_failed(status));
  uint count = ax_u32(payload + counts_offset + record * 4u);
  uint start = ax_u32(payload + offsets_offset + record * 4u);
  uint end = ax_u32(payload + offsets_offset + (record + 1u) * 4u);
  bool safe_patches = count <= 1024u && start <= patch_count &&
                      end <= patch_count && end >= start &&
                      end - start == count;
  bad |= uint(!safe_patches);
  if (record == 0u && start != 0u)
    bad = 1u;
  if (record + 1u == records && end != patch_count)
    bad = 1u;
  if (safe_patches) {
    for (uint j = t; j < count; j += 256u) {
      device const uchar *entry = payload + patches_offset + (start + j) * 4u;
      uint position = ax_u16(entry);
      float replacement = ax_bf16(entry + 2u);
      uint previous = j == 0u ? 0u : ax_u16(entry - 4u);
      if (position >= 1024u || (j > 0u && position <= previous) ||
          !isfinite(replacement))
        bad = 1u;
    }
  }
  for (uint group = t; group < 4u; group += 256u) {
    device const uchar *affine = payload + record * 544u + group * 136u;
    if (!ax_affine_valid(affine))
      bad = 1u;
  }
  threadgroup uint errors[256];
  errors[t] = bad;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (t < stride)
      errors[t] |= errors[t + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (t == 0u && errors[0] != 0u) {
    atomic_fetch_add_explicit(status + 1, 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(status + 4, 1u, memory_order_relaxed);
  }
}

kernel void affine_exception_validate_k(
    device const uchar *payload [[buffer(0)]],
    device atomic_uint *status [[buffer(1)]], constant uint &n [[buffer(2)]],
    constant uint &k [[buffer(3)]], constant uint &count [[buffer(4)]],
    constant uint &base_bytes [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
  uint map_offset = base_bytes;
  uint ids_offset = map_offset + k * 2u;
  uint sidecar_offset = ids_offset + ((count * 2u + 3u) & ~3u);
  uint sidecar_count = n * count;
  uint group_count = n * k / 256u;
  uint maximum = max(max(k, sidecar_count), group_count);
  if (i >= maximum)
    return;
  bool bad = ax_pack_failed(status);
  if (i < k) {
    uint ordinal = ax_u16(payload + map_offset + i * 2u);
    if (ordinal != 0xffffu && ordinal >= count)
      bad = true;
    else if (ordinal != 0xffffu &&
             ax_u16(payload + ids_offset + ordinal * 2u) != i)
      bad = true;
  }
  if (i < count) {
    uint channel = ax_u16(payload + ids_offset + i * 2u);
    if (channel >= k || ax_u16(payload + map_offset + channel * 2u) != i)
      bad = true;
    if (i > 0u && channel <= ax_u16(payload + ids_offset + (i - 1u) * 2u))
      bad = true;
  }
  if (i < sidecar_count &&
      !isfinite(ax_bf16(payload + sidecar_offset + i * 2u)))
    bad = true;
  if (i < group_count) {
    device const uchar *group = payload + i * 136u;
    if (!ax_affine_valid(group))
      bad = true;
  }
  if (bad) {
    atomic_fetch_add_explicit(status + 1, 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(status + 4, 1u, memory_order_relaxed);
  }
}

kernel void affine_exception_validate_n(
    device const uchar *payload [[buffer(0)]],
    device atomic_uint *status [[buffer(1)]], constant uint &n [[buffer(2)]],
    constant uint &k [[buffer(3)]], constant uint &row_base_bytes [[buffer(4)]],
    constant uint &directory_bytes [[buffer(5)]],
    constant uint &expected_end [[buffer(6)]],
    uint t [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
  if (row >= n)
    return;
  uint payload_bytes = (n * k / 256u) * 144u;
  uint start = ax_u32(payload + row * 4u);
  uint end = ax_u32(payload + (row + 1u) * 4u);
  uint raw_bytes = k * 2u;
  bool raw = end >= start && end - start == raw_bytes;
  bool affine = end >= start && end - start == row_base_bytes;
  bool bad = ax_pack_failed(status) || start < directory_bytes || end < start ||
             end > payload_bytes || (!raw && !affine) ||
             (row == 0u && start != directory_bytes) ||
             (row + 1u == n && end != expected_end);
  if (!bad && raw) {
    for (uint j = t; j < k; j += 256u)
      if (!isfinite(ax_bf16(payload + start + j * 2u)))
        bad = true;
  } else if (!bad && affine) {
    for (uint group = t; group < k / 256u; group += 256u) {
      device const uchar *affine_group = payload + start + group * 136u;
      if (!ax_affine_valid(affine_group))
        bad = true;
    }
  }
  threadgroup uint errors[256];
  errors[t] = uint(bad);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128u; stride > 0u; stride >>= 1u) {
    if (t < stride)
      errors[t] |= errors[t + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (t == 0u && errors[0] != 0u) {
    atomic_fetch_add_explicit(status + 1, 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(status + 4, 1u, memory_order_relaxed);
  }
}

kernel void affine_exception_decode_sparse_base(
    device const uchar *payload [[buffer(0)]],
    device float *stage [[buffer(1)]], device atomic_uint *status [[buffer(2)]],
    constant uint &groups [[buffer(3)]], constant uint &n [[buffer(4)]],
    constant uint &k [[buffer(5)]], uint t [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
  uint index = group * 256u + t;
  uint record = group / 4u;
  if (ax_decode_failed(status)) {
    stage[index] = as_type<float>(0x7fc00000u);
    return;
  }
  float value = ax_base_value(payload + group * 136u, t);
  if (!isfinite(value)) {
    atomic_fetch_add_explicit(status + 2, 1u, memory_order_relaxed);
    value = as_type<float>(0x7fc00000u);
  }
  stage[index] = value;
}

kernel void affine_exception_decode_sparse_patch(
    device const uchar *payload [[buffer(0)]],
    device float *stage [[buffer(1)]], device atomic_uint *status [[buffer(2)]],
    constant uint &records [[buffer(3)]],
    uint t [[thread_index_in_threadgroup]],
    uint record [[threadgroup_position_in_grid]]) {
  if (ax_decode_failed(status))
    return;
  uint base_bytes = records * 544u;
  uint counts_offset = base_bytes;
  uint offsets_offset = counts_offset + records * 4u;
  uint patches_offset = offsets_offset + (records + 1u) * 4u;
  uint start = ax_u32(payload + offsets_offset + record * 4u);
  uint end = ax_u32(payload + offsets_offset + (record + 1u) * 4u);
  for (uint j = start + t; j < end; j += 256u) {
    device const uchar *entry = payload + patches_offset + j * 4u;
    uint position = ax_u16(entry);
    float value = ax_bf16(entry + 2u);
    if (position < 1024u && isfinite(value))
      stage[record * 1024u + position] = value;
    else
      atomic_fetch_add_explicit(status + 2, 1u, memory_order_relaxed);
  }
}

kernel void affine_exception_decode_k(
    device const uchar *payload [[buffer(0)]],
    device float *stage [[buffer(1)]], device atomic_uint *status [[buffer(2)]],
    constant uint &n [[buffer(3)]], constant uint &k [[buffer(4)]],
    constant uint &count [[buffer(5)]], constant uint &base_bytes [[buffer(6)]],
    constant uint &map_offset [[buffer(7)]],
    constant uint &ids_offset [[buffer(8)]],
    constant uint &sidecar_offset [[buffer(9)]],
    uint i [[thread_position_in_grid]]) {
  if (ax_decode_failed(status)) {
    stage[i] = as_type<float>(0x7fc00000u);
    return;
  }
  uint row = i / k, channel = i % k;
  uint ordinal = ax_u16(payload + map_offset + channel * 2u);
  float value;
  if (ordinal == 0xffffu) {
    uint group_index = i / 256u;
    uint local = i & 255u;
    value = ax_base_value(payload + group_index * 136u, local);
  } else if (ordinal < count) {
    value = ax_bf16(payload + sidecar_offset + (row * count + ordinal) * 2u);
  } else {
    atomic_fetch_add_explicit(status + 2, 1u, memory_order_relaxed);
    value = as_type<float>(0x7fc00000u);
  }
  stage[i] = isfinite(value) ? value : as_type<float>(0x7fc00000u);
}

kernel void affine_exception_decode_n(
    device const uchar *payload [[buffer(0)]],
    device float *stage [[buffer(1)]], device atomic_uint *status [[buffer(2)]],
    constant uint &n [[buffer(3)]], constant uint &k [[buffer(4)]],
    constant uint &row_base_bytes [[buffer(5)]],
    constant uint &directory_bytes [[buffer(6)]],
    uint t [[thread_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
  uint start = ax_u32(payload + row * 4u);
  uint end = ax_u32(payload + (row + 1u) * 4u);
  uint row_bytes = end >= start ? end - start : 0u;
  bool raw = row_bytes == k * 2u;
  if (ax_decode_failed(status)) {
    for (uint j = t; j < k; j += 256u)
      stage[row * k + j] = as_type<float>(0x7fc00000u);
    return;
  }
  for (uint j = t; j < k; j += 256u) {
    float value;
    if (raw)
      value = ax_bf16(payload + start + j * 2u);
    else
      value = ax_base_value(payload + start + (j / 256u) * 136u, j & 255u);
    stage[row * k + j] = isfinite(value) ? value : as_type<float>(0x7fc00000u);
  }
}
