#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <vulkan/vulkan.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#define VK_CHECK(expr) do { VkResult _r = (expr); if (_r != VK_SUCCESS) die_vk(#expr, _r); } while (0)

typedef struct {
  uint32_t out_dim;
  uint32_t in_dim;
  uint32_t row_bytes;
  uint32_t repeats;
} Push;

typedef struct {
  VkBuffer buffer;
  VkDeviceMemory memory;
  void *mapped;
  size_t size;
} Buf;

typedef struct {
  uint32_t type_id;
  uint32_t out_dim;
  uint32_t in_dim;
  uint64_t raw_bytes;
  uint8_t *raw;
} TensorFile;

static void die(const char *msg) { fprintf(stderr, "error: %s\n", msg); exit(1); }
static void die_vk(const char *expr, VkResult r) { fprintf(stderr, "vulkan error: %s -> %d\n", expr, r); exit(1); }

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static uint32_t env_u32(const char *name, uint32_t fallback) {
  const char *s = getenv(name);
  if (!s || !*s) return fallback;
  char *end = NULL;
  unsigned long v = strtoul(s, &end, 10);
  if (end == s || *end != '\0' || v > UINT32_MAX) die("invalid unsigned integer environment value");
  return (uint32_t)v;
}

static char *read_file(const char *path, size_t *size_out) {
  FILE *f = fopen(path, "rb");
  if (!f) die("cannot open SPIR-V file");
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  rewind(f);
  if (n <= 0) die("empty SPIR-V file");
  char *buf = (char *)malloc((size_t)n);
  if (!buf) die("malloc failed");
  if (fread(buf, 1, (size_t)n, f) != (size_t)n) die("fread failed");
  fclose(f);
  *size_out = (size_t)n;
  return buf;
}

static void read_file_exact(const char *path, void *dst, size_t expected_size) {
  FILE *f = fopen(path, "rb");
  if (!f) die("cannot open exact input file");
  if (fseek(f, 0, SEEK_END) != 0) die("fseek exact input failed");
  long n = ftell(f);
  if (n < 0) die("ftell exact input failed");
  rewind(f);
  if ((size_t)n != expected_size) die("exact input file size mismatch");
  if (fread(dst, 1, expected_size, f) != expected_size) die("exact input fread failed");
  fclose(f);
}

static void write_file_exact(const char *path, const void *src, size_t size) {
  FILE *f = fopen(path, "wb");
  if (!f) die("cannot open exact output file");
  if (fwrite(src, 1, size, f) != size) die("exact output fwrite failed");
  if (fclose(f) != 0) die("exact output fclose failed");
}

static uint32_t rd_u32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t rd_u64(const uint8_t *p) {
  uint64_t v = 0;
  for (uint32_t i = 0; i < 8; i++) v |= ((uint64_t)p[i]) << (i * 8u);
  return v;
}

static int cmp_u32(const void *a, const void *b) {
  uint32_t av = *(const uint32_t *)a;
  uint32_t bv = *(const uint32_t *)b;
  return (av > bv) - (av < bv);
}

static void parse_row_ids_csv(const char *csv, uint32_t *ids, uint32_t count, uint32_t max_row) {
  const char *p = csv;
  for (uint32_t i = 0; i < count; i++) {
    char *end = NULL;
    unsigned long v = strtoul(p, &end, 10);
    if (end == p) die("invalid RPI5_ROW_IDS_CSV entry");
    if (v >= max_row) die("RPI5_ROW_IDS_CSV row id out of range");
    ids[i] = (uint32_t)v;
    p = end;
    if (i + 1u < count) {
      if (*p != ',') die("RPI5_ROW_IDS_CSV has too few entries");
      p++;
    }
  }
  if (*p != '\0') die("RPI5_ROW_IDS_CSV has too many entries");
}

static const char *parse_row_ids_group(const char *p, uint32_t *ids, uint32_t *count_out, uint32_t max_count, uint32_t max_row) {
  uint32_t count = 0;
  if (*p == '\0' || *p == ';' || *p == ':') die("empty RPI5_ROW_IDS_CSV_BATCH group");
  while (*p != '\0' && *p != ';' && *p != ':') {
    if (count >= max_count) die("RPI5_ROW_IDS_CSV_BATCH group exceeds q6idx max row count");
    char *end = NULL;
    unsigned long v = strtoul(p, &end, 10);
    if (end == p) die("invalid RPI5_ROW_IDS_CSV_BATCH entry");
    if (v >= max_row) die("RPI5_ROW_IDS_CSV_BATCH row id out of range");
    ids[count++] = (uint32_t)v;
    p = end;
    if (*p == ',') {
      p++;
    } else if (*p != '\0' && *p != ';' && *p != ':') {
      die("invalid RPI5_ROW_IDS_CSV_BATCH separator");
    }
  }
  *count_out = count;
  return p;
}

static void parse_row_id_groups_csv(const char *csv, uint32_t *ids, uint32_t *meta, uint32_t batch, uint32_t max_count, uint32_t max_row) {
  const char *p = csv;
  for (uint32_t b = 0; b < batch; b++) {
    uint32_t offset = b * max_count;
    uint32_t count = 0;
    p = parse_row_ids_group(p, ids + offset, &count, max_count, max_row);
    meta[b * 2u + 0u] = offset;
    meta[b * 2u + 1u] = count;
    for (uint32_t i = count; i < max_count; i++) ids[offset + i] = 0u;
    if (b + 1u < batch) {
      if (*p != ';' && *p != ':') die("RPI5_ROW_IDS_CSV_BATCH has too few groups");
      p++;
    }
  }
  if (*p != '\0') die("RPI5_ROW_IDS_CSV_BATCH has too many groups");
}

static TensorFile read_tensor_file(const char *path) {
  size_t n = 0;
  uint8_t *buf = (uint8_t *)read_file(path, &n);
  if (n < 24) die("tensor file too small");
  if (rd_u32(buf) != 0x43564750u) die("bad tensor file magic");
  TensorFile tf;
  tf.type_id = rd_u32(buf + 4);
  tf.out_dim = rd_u32(buf + 8);
  tf.in_dim = rd_u32(buf + 12);
  tf.raw_bytes = rd_u64(buf + 16);
  if (tf.type_id != 12u && tf.type_id != 14u) die("expected Q4_K or Q6_K tensor file");
  if (24u + tf.raw_bytes != n) die("tensor file size mismatch");
  tf.raw = (uint8_t *)malloc((size_t)tf.raw_bytes);
  if (!tf.raw) die("tensor raw alloc failed");
  memcpy(tf.raw, buf + 24, (size_t)tf.raw_bytes);
  free(buf);
  return tf;
}

static uint32_t find_memory_type(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags flags) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    if ((bits & (1u << i)) && ((mp.memoryTypes[i].propertyFlags & flags) == flags)) return i;
  }
  die("no compatible memory type");
  return 0;
}

static Buf make_buffer(VkDevice dev, VkPhysicalDevice pd, size_t size, VkBufferUsageFlags usage) {
  Buf b;
  memset(&b, 0, sizeof(b));
  b.size = size;
  VkBufferCreateInfo bi = {0};
  bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bi.size = size;
  bi.usage = usage;
  bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK(vkCreateBuffer(dev, &bi, NULL, &b.buffer));
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(dev, b.buffer, &mr);
  VkMemoryAllocateInfo ai = {0};
  ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  ai.allocationSize = mr.size;
  ai.memoryTypeIndex = find_memory_type(pd, mr.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VK_CHECK(vkAllocateMemory(dev, &ai, NULL, &b.memory));
  VK_CHECK(vkBindBufferMemory(dev, b.buffer, b.memory, 0));
  VK_CHECK(vkMapMemory(dev, b.memory, 0, size, 0, &b.mapped));
  return b;
}

static void free_buffer(VkDevice dev, Buf *b) {
  if (b->mapped) vkUnmapMemory(dev, b->memory);
  if (b->buffer) vkDestroyBuffer(dev, b->buffer, NULL);
  if (b->memory) vkFreeMemory(dev, b->memory, NULL);
}

static uint16_t f32_to_f16(float f) {
  union { float f; uint32_t u; } v = { f };
  uint32_t sign = (v.u >> 16) & 0x8000u;
  int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = v.u & 0x7fffffu;
  if (exp <= 0) return (uint16_t)sign;
  if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
  return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static uint32_t f32_bits(float f) {
  union { float f; uint32_t u; } v = { f };
  return v.u;
}

static float f16_to_f32(uint16_t h) {
  uint32_t sign = (h >> 15) & 1u;
  uint32_t exp = (h >> 10) & 31u;
  uint32_t frac = h & 1023u;
  float v;
  if (exp == 0) {
    v = frac == 0 ? 0.0f : ldexpf((float)frac, -24);
  } else if (exp == 31) {
    v = frac == 0 ? INFINITY : NAN;
  } else {
    v = ldexpf(1.0f + (float)frac / 1024.0f, (int)exp - 15);
  }
  return sign ? -v : v;
}

static void put_u16(uint8_t *p, uint16_t v) {
  p[0] = (uint8_t)(v & 0xffu);
  p[1] = (uint8_t)(v >> 8);
}

static void set_scale_min(uint8_t *s, uint32_t j, uint32_t scale, uint32_t minv) {
  if (j < 4) {
    s[j] = (uint8_t)((s[j] & 0xc0u) | (scale & 63u));
    s[j + 4] = (uint8_t)((s[j + 4] & 0xc0u) | (minv & 63u));
  } else {
    s[j - 4] = (uint8_t)((s[j - 4] & 0x3fu) | ((scale >> 4) << 6));
    s[j] = (uint8_t)((s[j] & 0x3fu) | ((minv >> 4) << 6));
    s[j + 4] = (uint8_t)((scale & 0x0fu) | ((minv & 0x0fu) << 4));
  }
}

static void get_scale_min(const uint8_t *s, uint32_t j, uint32_t *scale, uint32_t *minv) {
  if (j < 4) {
    *scale = s[j] & 63u;
    *minv = s[j + 4] & 63u;
  } else {
    *scale = (s[j + 4] & 0x0fu) | ((s[j - 4] >> 6) << 4);
    *minv = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
  }
}

static void fill_q4k(uint8_t *w, float *x, uint32_t out_dim, uint32_t in_dim) {
  uint32_t blocks = in_dim / 256u;
  uint32_t row_bytes = blocks * 144u;
  for (uint32_t i = 0; i < in_dim; i++) x[i] = ((int)(i % 29) - 14) * 0.03125f;

  for (uint32_t row = 0; row < out_dim; row++) {
    for (uint32_t blk = 0; blk < blocks; blk++) {
      uint8_t *bp = w + row * row_bytes + blk * 144u;
      memset(bp, 0, 144);
      put_u16(bp, f32_to_f16(0.03125f));
      put_u16(bp + 2, f32_to_f16(0.0f));
      uint8_t *scales = bp + 4;
      uint8_t *qs = bp + 16;
      for (uint32_t j = 0; j < 8; j++) set_scale_min(scales, j, 1u + ((row + blk + j) % 7u), 0u);
      for (uint32_t q = 0; q < 128; q++) {
        uint8_t lo = (uint8_t)((row * 3u + blk * 5u + q) & 15u);
        uint8_t hi = (uint8_t)((row * 7u + blk * 11u + q * 3u) & 15u);
        qs[q] = (uint8_t)(lo | (hi << 4));
      }
    }
  }
}

static void fill_q4_prepacked(uint8_t *w, float *x, uint32_t out_dim, uint32_t in_dim) {
  uint32_t groups = in_dim / 32u;
  uint32_t row_bytes = groups * 24u;
  for (uint32_t i = 0; i < in_dim; i++) x[i] = ((int)(i % 29) - 14) * 0.03125f;
  for (uint32_t row = 0; row < out_dim; row++) {
    for (uint32_t g = 0; g < groups; g++) {
      uint32_t *bp = (uint32_t *)(void *)(w + row * row_bytes + g * 24u);
      float scale = 0.03125f * (float)(1u + ((row + g) % 7u));
      float minv = 0.0f;
      memcpy(&bp[0], &scale, sizeof(float));
      memcpy(&bp[1], &minv, sizeof(float));
      for (uint32_t q = 0; q < 4; q++) {
        uint32_t word = 0;
        for (uint32_t lane = 0; lane < 8; lane++) {
          uint32_t j = q * 8u + lane;
          uint8_t v = (uint8_t)((row * 3u + g * 5u + j * 7u) & 15u);
          word |= ((uint32_t)v) << (lane * 4u);
        }
        bp[2 + q] = word;
      }
    }
  }
}

static void cpu_q4k(float *y, const uint8_t *w, const float *x, uint32_t out_dim, uint32_t in_dim) {
  uint32_t blocks = in_dim / 256u;
  uint32_t row_bytes = blocks * 144u;
  for (uint32_t row = 0; row < out_dim; row++) {
    double sum = 0.0;
    const uint8_t *wr = w + row * row_bytes;
    for (uint32_t blk = 0; blk < blocks; blk++) {
      const uint8_t *bp = wr + blk * 144u;
      float d = f16_to_f32((uint16_t)(bp[0] | (bp[1] << 8)));
      float dmin = f16_to_f32((uint16_t)(bp[2] | (bp[3] << 8)));
      const uint8_t *scales = bp + 4;
      const uint8_t *qs = bp + 16;
      uint32_t is = 0, qoff = 0, base_j = blk * 256u;
      for (uint32_t group = 0; group < 4; group++) {
        uint32_t sc1, m1, sc2, m2;
        get_scale_min(scales, is, &sc1, &m1);
        get_scale_min(scales, is + 1, &sc2, &m2);
        float d1 = d * (float)sc1, min1 = dmin * (float)m1;
        float d2 = d * (float)sc2, min2 = dmin * (float)m2;
        uint32_t j0 = base_j + (is / 2u) * 64u;
        for (uint32_t l = 0; l < 32; l++) sum += x[j0 + l] * (d1 * (float)(qs[qoff + l] & 0x0fu) - min1);
        for (uint32_t l = 0; l < 32; l++) sum += x[j0 + 32u + l] * (d2 * (float)(qs[qoff + l] >> 4) - min2);
        qoff += 32;
        is += 2;
      }
    }
    y[row] = (float)sum;
  }
}

static void cpu_q4_prepacked(float *y, const uint8_t *w, const float *x, uint32_t out_dim, uint32_t in_dim) {
  uint32_t groups = in_dim / 32u;
  uint32_t row_bytes = groups * 24u;
  for (uint32_t row = 0; row < out_dim; row++) {
    double sum = 0.0;
    const uint8_t *wr = w + row * row_bytes;
    for (uint32_t g = 0; g < groups; g++) {
      const uint32_t *bp = (const uint32_t *)(const void *)(wr + g * 24u);
      float scale, minv;
      memcpy(&scale, &bp[0], sizeof(float));
      memcpy(&minv, &bp[1], sizeof(float));
      for (uint32_t q = 0; q < 4; q++) {
        uint32_t word = bp[2 + q];
        for (uint32_t lane = 0; lane < 8; lane++) {
          uint32_t v = (word >> (lane * 4u)) & 0x0fu;
          sum += x[g * 32u + q * 8u + lane] * ((double)scale * (double)v - (double)minv);
        }
      }
    }
    y[row] = (float)sum;
  }
}

#if defined(__aarch64__)
static inline float neon_sum4(float32x4_t v) {
  return vaddvq_f32(v);
}

static inline float32x4_t q4_word_lo_to_f32(uint32_t word) {
  uint32x4_t q = {
    (word >> 0) & 0x0fu,
    (word >> 4) & 0x0fu,
    (word >> 8) & 0x0fu,
    (word >> 12) & 0x0fu,
  };
  return vcvtq_f32_u32(q);
}

static inline float32x4_t q4_word_hi_to_f32(uint32_t word) {
  uint32x4_t q = {
    (word >> 16) & 0x0fu,
    (word >> 20) & 0x0fu,
    (word >> 24) & 0x0fu,
    (word >> 28) & 0x0fu,
  };
  return vcvtq_f32_u32(q);
}

static void cpu_q4_prepacked_neon(float *y, const uint8_t *w, const float *x, uint32_t out_dim, uint32_t in_dim) {
  uint32_t groups = in_dim / 32u;
  uint32_t row_bytes = groups * 24u;
  for (uint32_t row = 0; row < out_dim; row++) {
    float32x4_t accv = vdupq_n_f32(0.0f);
    const uint8_t *wr = w + row * row_bytes;
    for (uint32_t g = 0; g < groups; g++) {
      const uint32_t *bp = (const uint32_t *)(const void *)(wr + g * 24u);
      float scale, minv;
      memcpy(&scale, &bp[0], sizeof(float));
      memcpy(&minv, &bp[1], sizeof(float));
      float32x4_t sv = vdupq_n_f32(scale);
      float32x4_t mv = vdupq_n_f32(minv);
      const float *xp = x + g * 32u;
      for (uint32_t q = 0; q < 4; q++) {
        uint32_t word = bp[2 + q];
        float32x4_t qlo = vsubq_f32(vmulq_f32(sv, q4_word_lo_to_f32(word)), mv);
        float32x4_t qhi = vsubq_f32(vmulq_f32(sv, q4_word_hi_to_f32(word)), mv);
        accv = vfmaq_f32(accv, vld1q_f32(xp + q * 8u), qlo);
        accv = vfmaq_f32(accv, vld1q_f32(xp + q * 8u + 4u), qhi);
      }
    }
    y[row] = neon_sum4(accv);
  }
}

typedef struct {
  float *y;
  const uint8_t *w;
  const float *x;
  uint32_t out_dim;
  uint32_t in_dim;
  uint32_t row_start;
  uint32_t row_end;
} NeonTask;

static void *cpu_q4_prepacked_neon_worker(void *arg) {
  NeonTask *t = (NeonTask *)arg;
  uint32_t groups = t->in_dim / 32u;
  uint32_t row_bytes = groups * 24u;
  for (uint32_t row = t->row_start; row < t->row_end; row++) {
    float32x4_t accv = vdupq_n_f32(0.0f);
    const uint8_t *wr = t->w + row * row_bytes;
    for (uint32_t g = 0; g < groups; g++) {
      const uint32_t *bp = (const uint32_t *)(const void *)(wr + g * 24u);
      float scale, minv;
      memcpy(&scale, &bp[0], sizeof(float));
      memcpy(&minv, &bp[1], sizeof(float));
      float32x4_t sv = vdupq_n_f32(scale);
      float32x4_t mv = vdupq_n_f32(minv);
      const float *xp = t->x + g * 32u;
      for (uint32_t q = 0; q < 4; q++) {
        uint32_t word = bp[2 + q];
        float32x4_t qlo = vsubq_f32(vmulq_f32(sv, q4_word_lo_to_f32(word)), mv);
        float32x4_t qhi = vsubq_f32(vmulq_f32(sv, q4_word_hi_to_f32(word)), mv);
        accv = vfmaq_f32(accv, vld1q_f32(xp + q * 8u), qlo);
        accv = vfmaq_f32(accv, vld1q_f32(xp + q * 8u + 4u), qhi);
      }
    }
    t->y[row] = neon_sum4(accv);
  }
  return NULL;
}

static void cpu_q4_prepacked_neon_threads(float *y, const uint8_t *w, const float *x,
                                          uint32_t out_dim, uint32_t in_dim, uint32_t n_threads) {
  if (n_threads < 1) n_threads = 1;
  if (n_threads > out_dim) n_threads = out_dim;
  pthread_t *threads = (pthread_t *)calloc(n_threads, sizeof(pthread_t));
  NeonTask *tasks = (NeonTask *)calloc(n_threads, sizeof(NeonTask));
  if (!threads || !tasks) die("thread alloc failed");
  uint32_t chunk = (out_dim + n_threads - 1u) / n_threads;
  for (uint32_t i = 0; i < n_threads; i++) {
    uint32_t start = i * chunk;
    uint32_t end = start + chunk;
    if (end > out_dim) end = out_dim;
    tasks[i] = (NeonTask){y, w, x, out_dim, in_dim, start, end};
    if (pthread_create(&threads[i], NULL, cpu_q4_prepacked_neon_worker, &tasks[i]) != 0) die("pthread_create failed");
  }
  for (uint32_t i = 0; i < n_threads; i++) pthread_join(threads[i], NULL);
  free(tasks);
  free(threads);
}
#endif

static void prepack_q4k_to_q4_pre(uint8_t *dst, const uint8_t *src, uint32_t out_dim, uint32_t in_dim) {
  uint32_t src_blocks = in_dim / 256u;
  uint32_t src_row_bytes = src_blocks * 144u;
  uint32_t dst_groups = in_dim / 32u;
  uint32_t dst_row_bytes = dst_groups * 24u;
  for (uint32_t row = 0; row < out_dim; row++) {
    const uint8_t *sr = src + row * src_row_bytes;
    uint8_t *dr = dst + row * dst_row_bytes;
    for (uint32_t blk = 0; blk < src_blocks; blk++) {
      const uint8_t *bp = sr + blk * 144u;
      float d = f16_to_f32((uint16_t)(bp[0] | (bp[1] << 8)));
      float dmin = f16_to_f32((uint16_t)(bp[2] | (bp[3] << 8)));
      const uint8_t *scales = bp + 4;
      const uint8_t *qs = bp + 16;
      uint32_t is = 0, qoff = 0;
      for (uint32_t group = 0; group < 4; group++) {
        uint32_t sc1, m1, sc2, m2;
        get_scale_min(scales, is, &sc1, &m1);
        get_scale_min(scales, is + 1, &sc2, &m2);
        uint32_t dst_group0 = blk * 8u + group * 2u;
        uint32_t *d0 = (uint32_t *)(void *)(dr + dst_group0 * 24u);
        uint32_t *d1 = (uint32_t *)(void *)(dr + (dst_group0 + 1u) * 24u);
        float scale1 = d * (float)sc1, min1 = dmin * (float)m1;
        float scale2 = d * (float)sc2, min2 = dmin * (float)m2;
        memcpy(&d0[0], &scale1, sizeof(float));
        memcpy(&d0[1], &min1, sizeof(float));
        memcpy(&d1[0], &scale2, sizeof(float));
        memcpy(&d1[1], &min2, sizeof(float));
        for (uint32_t q = 0; q < 4; q++) {
          uint32_t word0 = 0, word1 = 0;
          for (uint32_t lane = 0; lane < 8; lane++) {
            uint8_t b = qs[qoff + q * 8u + lane];
            word0 |= ((uint32_t)(b & 0x0fu)) << (lane * 4u);
            word1 |= ((uint32_t)(b >> 4)) << (lane * 4u);
          }
          d0[2 + q] = word0;
          d1[2 + q] = word1;
        }
        qoff += 32u;
        is += 2u;
      }
    }
  }
}

static void prepack_q6k_to_q6_pre(uint8_t *dst, const uint8_t *src, uint32_t out_dim, uint32_t in_dim) {
  uint32_t src_blocks = in_dim / 256u;
  uint32_t src_row_bytes = src_blocks * 210u;
  uint32_t dst_groups = in_dim / 16u;
  uint32_t dst_row_bytes = dst_groups * 20u;
  memset(dst, 0, (size_t)out_dim * dst_row_bytes);
  for (uint32_t row = 0; row < out_dim; row++) {
    const uint8_t *sr = src + row * src_row_bytes;
    uint8_t *dr = dst + row * dst_row_bytes;
    for (uint32_t blk = 0; blk < src_blocks; blk++) {
      const uint8_t *bp = sr + blk * 210u;
      const uint8_t *ql = bp;
      const uint8_t *qh = bp + 128u;
      const int8_t *sc = (const int8_t *)(const void *)(bp + 192u);
      float d = f16_to_f32((uint16_t)(bp[208] | (bp[209] << 8)));
      uint32_t ql_off = 0, qh_off = 0, sc_off = 0;
      for (uint32_t n_iter = 0; n_iter < 2u; n_iter++) {
        for (uint32_t l = 0; l < 32u; l++) {
          uint32_t is = l / 16u;
          uint32_t q1 = ((ql[ql_off + l] & 0x0fu) | (((qh[qh_off + l] >> 0) & 3u) << 4));
          uint32_t q2 = ((ql[ql_off + l + 32u] & 0x0fu) | (((qh[qh_off + l] >> 2) & 3u) << 4));
          uint32_t q3 = ((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 3u) << 4));
          uint32_t q4 = ((ql[ql_off + l + 32u] >> 4) | (((qh[qh_off + l] >> 6) & 3u) << 4));
          float ds[4] = {
            d * (float)sc[sc_off + is],
            d * (float)sc[sc_off + is + 2u],
            d * (float)sc[sc_off + is + 4u],
            d * (float)sc[sc_off + is + 6u],
          };
          uint32_t base = blk * 256u + n_iter * 128u;
          uint32_t pos[4] = {base + l, base + l + 32u, base + l + 64u, base + l + 96u};
          uint32_t qs[4] = {q1, q2, q3, q4};
          for (uint32_t k = 0; k < 4u; k++) {
            uint32_t g = pos[k] / 16u;
            uint32_t lane = pos[k] & 15u;
            uint32_t *dw = (uint32_t *)(void *)(dr + g * 20u);
            if (lane == 0u) dw[0] = f32_bits(ds[k]);
            uint32_t word = 1u + lane / 4u;
            uint32_t shift = (lane & 3u) * 8u;
            dw[word] = (dw[word] & ~(0xffu << shift)) | ((qs[k] & 0x3fu) << shift);
          }
        }
        ql_off += 64u;
        qh_off += 32u;
        sc_off += 8u;
      }
    }
  }
}

static void cpu_q6_prepacked(float *y, const uint8_t *w, const float *x, uint32_t out_dim, uint32_t in_dim) {
  uint32_t groups = in_dim / 16u;
  uint32_t row_bytes = groups * 20u;
  for (uint32_t row = 0; row < out_dim; row++) {
    double sum = 0.0;
    const uint8_t *wr = w + row * row_bytes;
    for (uint32_t g = 0; g < groups; g++) {
      const uint32_t *bp = (const uint32_t *)(const void *)(wr + g * 20u);
      float dscale;
      memcpy(&dscale, &bp[0], sizeof(float));
      for (uint32_t q = 0; q < 4u; q++) {
        uint32_t word = bp[1u + q];
        for (uint32_t lane = 0; lane < 4u; lane++) {
          uint32_t v = (word >> (lane * 8u)) & 0x3fu;
          sum += x[g * 16u + q * 4u + lane] * ((double)dscale * ((double)v - 32.0));
        }
      }
    }
    y[row] = (float)sum;
  }
}

static void cpu_q6_prepacked_indexed(float *y, const uint8_t *w, const float *x, const uint32_t *row_ids,
                                     uint32_t out_dim, uint32_t in_dim) {
  uint32_t groups = in_dim / 16u;
  uint32_t row_bytes = groups * 20u;
  for (uint32_t out_row = 0; out_row < out_dim; out_row++) {
    uint32_t src_row = row_ids[out_row];
    double sum = 0.0;
    const uint8_t *wr = w + src_row * row_bytes;
    for (uint32_t g = 0; g < groups; g++) {
      const uint32_t *bp = (const uint32_t *)(const void *)(wr + g * 20u);
      float dscale;
      memcpy(&dscale, &bp[0], sizeof(float));
      for (uint32_t q = 0; q < 4u; q++) {
        uint32_t word = bp[1u + q];
        for (uint32_t lane = 0; lane < 4u; lane++) {
          uint32_t v = (word >> (lane * 8u)) & 0x3fu;
          sum += x[g * 16u + q * 4u + lane] * ((double)dscale * ((double)v - 32.0));
        }
      }
    }
    y[out_row] = (float)sum;
  }
}

static void cpu_q6_prepacked_indexed_meta(float *y, const uint8_t *w, const float *x, const uint32_t *row_ids,
                                          const uint32_t *row_meta, uint32_t batch, uint32_t out_dim, uint32_t in_dim) {
  uint32_t groups = in_dim / 16u;
  uint32_t row_bytes = groups * 20u;
  memset(y, 0, (size_t)batch * out_dim * sizeof(float));
  for (uint32_t b = 0; b < batch; b++) {
    uint32_t ids_off = row_meta[b * 2u + 0u];
    uint32_t count = row_meta[b * 2u + 1u];
    for (uint32_t out_row = 0; out_row < count; out_row++) {
      uint32_t src_row = row_ids[ids_off + out_row];
      double sum = 0.0;
      const uint8_t *wr = w + src_row * row_bytes;
      for (uint32_t g = 0; g < groups; g++) {
        const uint32_t *bp = (const uint32_t *)(const void *)(wr + g * 20u);
        float dscale;
        memcpy(&dscale, &bp[0], sizeof(float));
        for (uint32_t q = 0; q < 4u; q++) {
          uint32_t word = bp[1u + q];
          for (uint32_t lane = 0; lane < 4u; lane++) {
            uint32_t v = (word >> (lane * 8u)) & 0x3fu;
            sum += x[(size_t)b * in_dim + g * 16u + q * 4u + lane] * ((double)dscale * ((double)v - 32.0));
          }
        }
      }
      y[(size_t)b * out_dim + out_row] = (float)sum;
    }
  }
}

static void prepack_q4k_to_q4_inflated(uint8_t *dst, const uint8_t *src, uint32_t out_dim, uint32_t in_dim) {
  uint32_t src_blocks = in_dim / 256u;
  uint32_t src_row_bytes = src_blocks * 144u;
  uint32_t dst_groups = in_dim / 32u;
  uint32_t dst_row_bytes = dst_groups * 136u;
  for (uint32_t row = 0; row < out_dim; row++) {
    const uint8_t *sr = src + row * src_row_bytes;
    uint8_t *dr = dst + row * dst_row_bytes;
    for (uint32_t blk = 0; blk < src_blocks; blk++) {
      const uint8_t *bp = sr + blk * 144u;
      float d = f16_to_f32((uint16_t)(bp[0] | (bp[1] << 8)));
      float dmin = f16_to_f32((uint16_t)(bp[2] | (bp[3] << 8)));
      const uint8_t *scales = bp + 4;
      const uint8_t *qs = bp + 16;
      uint32_t is = 0, qoff = 0;
      for (uint32_t group = 0; group < 4; group++) {
        uint32_t sc1, m1, sc2, m2;
        get_scale_min(scales, is, &sc1, &m1);
        get_scale_min(scales, is + 1, &sc2, &m2);
        uint32_t dst_group0 = blk * 8u + group * 2u;
        uint32_t *d0 = (uint32_t *)(void *)(dr + dst_group0 * 136u);
        uint32_t *d1 = (uint32_t *)(void *)(dr + (dst_group0 + 1u) * 136u);
        float scale1 = d * (float)sc1, min1 = dmin * (float)m1;
        float scale2 = d * (float)sc2, min2 = dmin * (float)m2;
        d0[0] = f32_bits(scale1);
        d0[1] = f32_bits(min1);
        d1[0] = f32_bits(scale2);
        d1[1] = f32_bits(min2);
        for (uint32_t i = 0; i < 32u; i++) {
          uint8_t packed0 = qs[qoff + i];
          d0[2u + i] = packed0 & 0x0fu;
          d1[2u + i] = packed0 >> 4;
        }
        is += 2u;
        qoff += 32u;
      }
    }
  }
}

static float max_abs_diff(const float *a, const float *b, uint32_t n) {
  float md = 0.0f;
  for (uint32_t i = 0; i < n; i++) {
    float d = fabsf(a[i] - b[i]);
    if (d > md) md = d;
  }
  return md;
}

static uint32_t argmax_f32(const float *v, uint32_t n) {
  uint32_t best = 0;
  float best_v = v[0];
  for (uint32_t i = 1; i < n; i++) {
    if (v[i] > best_v) {
      best_v = v[i];
      best = i;
    }
  }
  return best;
}

int main(int argc, char **argv) {
  const char *spv_path = argc > 1 ? argv[1] : "rpi5_q4k_matvec.spv";
  int file_mode = argc > 2 && strcmp(argv[2], "file") == 0;
  TensorFile tf;
  memset(&tf, 0, sizeof(tf));
  uint32_t out_dim;
  uint32_t in_dim;
  uint32_t repeats;
  int workgroup_per_row = 0;
  int prepacked = 0;
  uint32_t dispatch_local_size = 64u;
  uint32_t warmups = env_u32("RPI5_WARMUPS", 0u);
  const char *mode = "raw";
  if (file_mode) {
    if (argc < 6) die("usage: probe SPV file TENSOR.cvgp REPEATS raw|pre|pre_lN|wg");
    tf = read_tensor_file(argv[3]);
    out_dim = tf.out_dim;
    in_dim = tf.in_dim;
    repeats = (uint32_t)strtoul(argv[4], NULL, 10);
    mode = argv[5];
  } else {
    out_dim = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 4096;
    in_dim = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 4096;
    repeats = argc > 4 ? (uint32_t)strtoul(argv[4], NULL, 10) : 50;
    mode = argc > 5 ? argv[5] : "raw";
  }
  workgroup_per_row = strcmp(mode, "wg") == 0;
  int sumx_mode = strncmp(mode, "pre_sx_l", 8) == 0;
  int batch_mode = strncmp(mode, "pre_b", 5) == 0;
  int token_pack_mode = strncmp(mode, "pre_t", 5) == 0;
  int inflated_mode = strncmp(mode, "pre_i_l", 7) == 0;
  int q6_pre_mode = strncmp(mode, "q6pre_l", 7) == 0;
  int q6_idx_sorted_mode = strncmp(mode, "q6idxs", 6) == 0;
  int q6_idx_mode = q6_idx_sorted_mode || strncmp(mode, "q6idx", 5) == 0;
  uint32_t batch = 1u;
  uint32_t token_pack = 1u;
  uint32_t allowed_count = 0u;
  prepacked = strcmp(mode, "pre") == 0 || strncmp(mode, "pre_l", 5) == 0 || sumx_mode || batch_mode || token_pack_mode || inflated_mode || q6_pre_mode || q6_idx_mode;
  if (strncmp(mode, "pre_l", 5) == 0) {
    dispatch_local_size = (uint32_t)strtoul(mode + 5, NULL, 10);
  } else if (sumx_mode) {
    dispatch_local_size = (uint32_t)strtoul(mode + 8, NULL, 10);
  } else if (batch_mode) {
    char *end = NULL;
    batch = (uint32_t)strtoul(mode + 5, &end, 10);
    if (!end || strncmp(end, "_l", 2) != 0) die("invalid batch mode; expected pre_bN_lM");
    dispatch_local_size = (uint32_t)strtoul(end + 2, NULL, 10);
  } else if (token_pack_mode) {
    char *end = NULL;
    token_pack = (uint32_t)strtoul(mode + 5, &end, 10);
    if (!end || strncmp(end, "_b", 2) != 0) die("invalid token-pack mode; expected pre_tP_bN_lM");
    batch = (uint32_t)strtoul(end + 2, &end, 10);
    if (!end || strncmp(end, "_l", 2) != 0) die("invalid token-pack mode; expected pre_tP_bN_lM");
    dispatch_local_size = (uint32_t)strtoul(end + 2, NULL, 10);
  } else if (inflated_mode) {
    dispatch_local_size = (uint32_t)strtoul(mode + 7, NULL, 10);
  } else if (q6_pre_mode) {
    dispatch_local_size = (uint32_t)strtoul(mode + 7, NULL, 10);
  } else if (q6_idx_mode) {
    char *end = NULL;
    allowed_count = (uint32_t)strtoul(mode + (q6_idx_sorted_mode ? 6 : 5), &end, 10);
    if (!end || strncmp(end, "_l", 2) != 0) die("invalid q6idx mode; expected q6idxN_lM");
    dispatch_local_size = (uint32_t)strtoul(end + 2, NULL, 10);
    batch = env_u32("RPI5_BATCH", batch);
  }
  if (batch == 0u) die("invalid batch mode");
  if (token_pack == 0u) die("invalid token-pack mode");
  if (!workgroup_per_row && dispatch_local_size == 0u) die("invalid local size mode");
  if (!out_dim || !in_dim || in_dim % 256u != 0 || !repeats) die("usage: probe SPV OUT_DIM IN_DIM REPEATS; IN_DIM must be multiple of 256");
  uint32_t src_out_dim = out_dim;
  if (q6_idx_mode) {
    if (!file_mode) die("q6idx mode requires file mode");
    if (allowed_count == 0u || allowed_count > src_out_dim) die("invalid q6idx allowed row count");
    out_dim = allowed_count;
  }

  VkApplicationInfo app = {0};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "rpi5_vulkan_q4k_probe";
  app.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici = {0};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pApplicationInfo = &app;
  VkInstance inst;
  VK_CHECK(vkCreateInstance(&ici, NULL, &inst));

  uint32_t pd_count = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(inst, &pd_count, NULL));
  if (!pd_count) die("no Vulkan physical devices");
  VkPhysicalDevice pd;
  VK_CHECK(vkEnumeratePhysicalDevices(inst, &pd_count, &pd));
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(pd, &props);

  uint32_t q_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &q_count, NULL);
  VkQueueFamilyProperties *qprops = (VkQueueFamilyProperties *)calloc(q_count, sizeof(*qprops));
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &q_count, qprops);
  uint32_t qfam = UINT32_MAX;
  for (uint32_t i = 0; i < q_count; i++) if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfam = i; break; }
  free(qprops);
  if (qfam == UINT32_MAX) die("no compute queue");

  float qprio = 1.0f;
  VkDeviceQueueCreateInfo qci = {0};
  qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  qci.queueFamilyIndex = qfam;
  qci.queueCount = 1;
  qci.pQueuePriorities = &qprio;
  VkDeviceCreateInfo dci = {0};
  dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  VkDevice dev;
  VK_CHECK(vkCreateDevice(pd, &dci, NULL, &dev));
  VkQueue queue;
  vkGetDeviceQueue(dev, qfam, 0, &queue);

  uint32_t blocks = in_dim / 256u;
  uint32_t row_bytes = (q6_pre_mode || q6_idx_mode) ? (in_dim / 16u) * 20u : (inflated_mode ? (in_dim / 32u) * 136u : (prepacked ? (in_dim / 32u) * 24u : blocks * 144u));
  size_t w_bytes = (size_t)src_out_dim * row_bytes;
  size_t x_bytes = (size_t)batch * in_dim * sizeof(float);
  size_t y_bytes = (size_t)batch * out_dim * sizeof(float);
  size_t sumx_bytes = (size_t)(in_dim / 32u) * sizeof(float);
  Buf wb = make_buffer(dev, pd, w_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  Buf xb = make_buffer(dev, pd, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  Buf yb = make_buffer(dev, pd, y_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  Buf sumxb = {0};
  Buf rowidb = {0};
  Buf rowmetab = {0};
  if (sumx_mode) sumxb = make_buffer(dev, pd, sumx_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  if (q6_idx_mode) {
    rowidb = make_buffer(dev, pd, (size_t)batch * out_dim * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    rowmetab = make_buffer(dev, pd, (size_t)batch * 2u * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    uint32_t *ids = (uint32_t *)rowidb.mapped;
    uint32_t *meta = (uint32_t *)rowmetab.mapped;
    const char *row_ids_batch_csv = getenv("RPI5_ROW_IDS_CSV_BATCH");
    const char *row_ids_csv = getenv("RPI5_ROW_IDS_CSV");
    if (row_ids_batch_csv && row_ids_batch_csv[0]) {
      parse_row_id_groups_csv(row_ids_batch_csv, ids, meta, batch, out_dim, src_out_dim);
    } else if (row_ids_csv && row_ids_csv[0]) {
      parse_row_ids_csv(row_ids_csv, ids, out_dim, src_out_dim);
      if (q6_idx_sorted_mode) qsort(ids, out_dim, sizeof(uint32_t), cmp_u32);
      for (uint32_t b = 0; b < batch; b++) {
        uint32_t offset = b * out_dim;
        if (b > 0) memcpy(ids + offset, ids, (size_t)out_dim * sizeof(uint32_t));
        meta[b * 2u + 0u] = offset;
        meta[b * 2u + 1u] = out_dim;
      }
    } else {
      for (uint32_t b = 0; b < batch; b++) {
        uint32_t offset = b * out_dim;
        for (uint32_t i = 0; i < out_dim; i++) ids[offset + i] = (i * 241u + 17u) % src_out_dim;
        if (q6_idx_sorted_mode) qsort(ids + offset, out_dim, sizeof(uint32_t), cmp_u32);
        meta[b * 2u + 0u] = offset;
        meta[b * 2u + 1u] = out_dim;
      }
    }
  }
  float *cpu = (float *)malloc(y_bytes);
  if (!cpu) die("cpu alloc failed");
  double prepack_ms = 0.0;
  const char *q6_prepack_load = getenv("RPI5_Q6_PREPACK_LOAD");
  const char *q6_prepack_save = getenv("RPI5_Q6_PREPACK_SAVE");
  if (file_mode && (q6_pre_mode || q6_idx_mode) && q6_prepack_load && q6_prepack_load[0]) {
    if (tf.type_id != 14u) die("q6 prepack load requires Q6_K tensor file");
    for (uint32_t b = 0; b < batch; b++) {
      for (uint32_t i = 0; i < in_dim; i++) ((float *)xb.mapped)[(size_t)b * in_dim + i] = ((int)((i + b * 7u) % 29) - 14) * 0.03125f;
    }
    double lt0 = now_ms();
    read_file_exact(q6_prepack_load, wb.mapped, w_bytes);
    prepack_ms = -(now_ms() - lt0);
  } else if (file_mode && (q6_pre_mode || q6_idx_mode)) {
    if (tf.type_id != 14u) die("q6pre mode requires Q6_K tensor file");
    for (uint32_t b = 0; b < batch; b++) {
      for (uint32_t i = 0; i < in_dim; i++) ((float *)xb.mapped)[(size_t)b * in_dim + i] = ((int)((i + b * 7u) % 29) - 14) * 0.03125f;
    }
    double pt0 = now_ms();
    prepack_q6k_to_q6_pre((uint8_t *)wb.mapped, tf.raw, src_out_dim, in_dim);
    prepack_ms = now_ms() - pt0;
    if (q6_prepack_save && q6_prepack_save[0]) write_file_exact(q6_prepack_save, wb.mapped, w_bytes);
  } else if (file_mode && prepacked) {
    if (tf.type_id != 12u) die("Q4 pre mode requires Q4_K tensor file");
    for (uint32_t b = 0; b < batch; b++) {
      for (uint32_t i = 0; i < in_dim; i++) ((float *)xb.mapped)[(size_t)b * in_dim + i] = ((int)((i + b * 7u) % 29) - 14) * 0.03125f;
    }
    double pt0 = now_ms();
    if (inflated_mode) {
      prepack_q4k_to_q4_inflated((uint8_t *)wb.mapped, tf.raw, out_dim, in_dim);
    } else {
      prepack_q4k_to_q4_pre((uint8_t *)wb.mapped, tf.raw, out_dim, in_dim);
    }
    prepack_ms = now_ms() - pt0;
  } else if (file_mode) {
    memcpy(wb.mapped, tf.raw, (size_t)tf.raw_bytes);
    for (uint32_t b = 0; b < batch; b++) {
      for (uint32_t i = 0; i < in_dim; i++) ((float *)xb.mapped)[(size_t)b * in_dim + i] = ((int)((i + b * 7u) % 29) - 14) * 0.03125f;
    }
  } else if (prepacked) {
    fill_q4_prepacked((uint8_t *)wb.mapped, (float *)xb.mapped, out_dim, in_dim);
  } else {
    fill_q4k((uint8_t *)wb.mapped, (float *)xb.mapped, out_dim, in_dim);
  }
  if (sumx_mode) {
    float *sx = (float *)sumxb.mapped;
    const float *xv = (const float *)xb.mapped;
    for (uint32_t g = 0; g < in_dim / 32u; g++) {
      float s = 0.0f;
      for (uint32_t i = 0; i < 32u; i++) s += xv[g * 32u + i];
      sx[g] = s;
    }
  }
  memset(yb.mapped, 0, y_bytes);

  size_t spv_size;
  char *spv = read_file(spv_path, &spv_size);
  VkShaderModuleCreateInfo smi = {0};
  smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  smi.codeSize = spv_size;
  smi.pCode = (const uint32_t *)spv;
  VkShaderModule shader;
  VK_CHECK(vkCreateShaderModule(dev, &smi, NULL, &shader));
  free(spv);

  uint32_t descriptor_count = q6_idx_mode ? 5u : (sumx_mode ? 4u : 3u);
  VkDescriptorSetLayoutBinding binds[5];
  memset(binds, 0, sizeof(binds));
  for (uint32_t i = 0; i < descriptor_count; i++) {
    binds[i].binding = i;
    binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[i].descriptorCount = 1;
    binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dsli = {0};
  dsli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dsli.bindingCount = descriptor_count;
  dsli.pBindings = binds;
  VkDescriptorSetLayout dsl;
  VK_CHECK(vkCreateDescriptorSetLayout(dev, &dsli, NULL, &dsl));

  VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push)};
  VkPipelineLayoutCreateInfo pli = {0};
  pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pli.setLayoutCount = 1;
  pli.pSetLayouts = &dsl;
  pli.pushConstantRangeCount = 1;
  pli.pPushConstantRanges = &pcr;
  VkPipelineLayout pl;
  VK_CHECK(vkCreatePipelineLayout(dev, &pli, NULL, &pl));

  VkPipelineShaderStageCreateInfo stage = {0};
  stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = shader;
  stage.pName = "main";
  VkComputePipelineCreateInfo cpi = {0};
  cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpi.stage = stage;
  cpi.layout = pl;
  VkPipeline pipe;
  VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpi, NULL, &pipe));

  VkDescriptorPoolSize pool_size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptor_count};
  VkDescriptorPoolCreateInfo dpci = {0};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.maxSets = 1;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &pool_size;
  VkDescriptorPool dp;
  VK_CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
  VkDescriptorSetAllocateInfo dsai = {0};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = dp;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &dsl;
  VkDescriptorSet ds;
  VK_CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
  VkDescriptorBufferInfo infos[5] = {
    {wb.buffer, 0, w_bytes}, {xb.buffer, 0, x_bytes}, {yb.buffer, 0, y_bytes}, {sumxb.buffer, 0, sumx_bytes}
  };
  if (q6_idx_mode) {
    infos[3] = (VkDescriptorBufferInfo){rowidb.buffer, 0, (size_t)batch * out_dim * sizeof(uint32_t)};
    infos[4] = (VkDescriptorBufferInfo){rowmetab.buffer, 0, (size_t)batch * 2u * sizeof(uint32_t)};
  }
  VkWriteDescriptorSet writes[5];
  memset(writes, 0, sizeof(writes));
  for (uint32_t i = 0; i < descriptor_count; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = ds;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(dev, descriptor_count, writes, 0, NULL);

  VkCommandPoolCreateInfo cpci = {0};
  cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cpci.queueFamilyIndex = qfam;
  cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  VkCommandPool cp;
  VK_CHECK(vkCreateCommandPool(dev, &cpci, NULL, &cp));
  VkCommandBufferAllocateInfo cbai = {0};
  cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbai.commandPool = cp;
  cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbai.commandBufferCount = 1;
  VkCommandBuffer cmd;
  VK_CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));

  Push push = {out_dim, in_dim, row_bytes, repeats};
  VkCommandBufferBeginInfo cbi = {0};
  cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  VK_CHECK(vkBeginCommandBuffer(cmd, &cbi));
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &push);
  if (workgroup_per_row) {
    vkCmdDispatch(cmd, out_dim, 1, 1);
  } else {
    vkCmdDispatch(cmd, (out_dim + dispatch_local_size - 1u) / dispatch_local_size, (batch + token_pack - 1u) / token_pack, 1);
  }
  VK_CHECK(vkEndCommandBuffer(cmd));
  VkSubmitInfo si = {0};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;

  for (uint32_t r = 0; r < warmups; r++) {
    memset(yb.mapped, 0, y_bytes);
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
  }

  double gpu_total = 0.0;
  for (uint32_t r = 0; r < repeats; r++) {
    memset(yb.mapped, 0, y_bytes);
    double t0 = now_ms();
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
    gpu_total += now_ms() - t0;
  }

  double t0 = now_ms();
  if (file_mode && q6_idx_mode) {
    cpu_q6_prepacked_indexed_meta(cpu, (const uint8_t *)wb.mapped, (const float *)xb.mapped,
                                  (const uint32_t *)rowidb.mapped, (const uint32_t *)rowmetab.mapped,
                                  batch, out_dim, in_dim);
  } else if (file_mode && q6_pre_mode) {
    for (uint32_t b = 0; b < batch; b++) {
      cpu_q6_prepacked(cpu + (size_t)b * out_dim, (const uint8_t *)wb.mapped, (const float *)xb.mapped + (size_t)b * in_dim, out_dim, in_dim);
    }
  } else if (file_mode && inflated_mode) {
    for (uint32_t b = 0; b < batch; b++) {
      cpu_q4k(cpu + (size_t)b * out_dim, tf.raw, (const float *)xb.mapped + (size_t)b * in_dim, out_dim, in_dim);
    }
  } else if (file_mode && prepacked) {
    for (uint32_t b = 0; b < batch; b++) {
      cpu_q4_prepacked(cpu + (size_t)b * out_dim, (const uint8_t *)wb.mapped, (const float *)xb.mapped + (size_t)b * in_dim, out_dim, in_dim);
    }
  } else if (file_mode) {
    cpu_q4k(cpu, (const uint8_t *)wb.mapped, (const float *)xb.mapped, out_dim, in_dim);
  } else if (prepacked) {
    for (uint32_t b = 0; b < batch; b++) {
      cpu_q4_prepacked(cpu + (size_t)b * out_dim, (const uint8_t *)wb.mapped, (const float *)xb.mapped + (size_t)b * in_dim, out_dim, in_dim);
    }
  } else {
    cpu_q4k(cpu, (const uint8_t *)wb.mapped, (const float *)xb.mapped, out_dim, in_dim);
  }
  double cpu_ms = now_ms() - t0;
#if defined(__aarch64__)
  double cpu_neon_ms = 0.0;
  double cpu_neon4_ms = 0.0;
  float *cpu_neon = NULL;
  float *cpu_neon4 = NULL;
  if (prepacked && !inflated_mode && !q6_pre_mode && !q6_idx_mode) {
    cpu_neon = (float *)malloc(y_bytes);
    if (!cpu_neon) die("cpu neon alloc failed");
    double nt0 = now_ms();
    for (uint32_t b = 0; b < batch; b++) {
      cpu_q4_prepacked_neon(cpu_neon + (size_t)b * out_dim, (const uint8_t *)wb.mapped, (const float *)xb.mapped + (size_t)b * in_dim, out_dim, in_dim);
    }
    cpu_neon_ms = now_ms() - nt0;
    cpu_neon4 = (float *)malloc(y_bytes);
    if (!cpu_neon4) die("cpu neon4 alloc failed");
    double mt0 = now_ms();
    for (uint32_t b = 0; b < batch; b++) {
      cpu_q4_prepacked_neon_threads(cpu_neon4 + (size_t)b * out_dim, (const uint8_t *)wb.mapped, (const float *)xb.mapped + (size_t)b * in_dim, out_dim, in_dim, 4);
    }
    cpu_neon4_ms = now_ms() - mt0;
  }
#endif
  float diff = max_abs_diff(cpu, (const float *)yb.mapped, batch * out_dim);
  double top1_scan_ms = 0.0;
  uint32_t gpu_top1 = 0, cpu_top1 = 0, gpu_top1_src = 0, cpu_top1_src = 0;
  int top1_match = 1;
  if (q6_idx_mode && batch == 1u && out_dim > 0u) {
    double st0 = now_ms();
    gpu_top1 = argmax_f32((const float *)yb.mapped, out_dim);
    top1_scan_ms = now_ms() - st0;
    cpu_top1 = argmax_f32(cpu, out_dim);
    const uint32_t *ids = (const uint32_t *)rowidb.mapped;
    gpu_top1_src = ids[gpu_top1];
    cpu_top1_src = ids[cpu_top1];
    top1_match = gpu_top1_src == cpu_top1_src;
  }
#if defined(__aarch64__)
  float neon_diff = 0.0f;
  float neon4_diff = 0.0f;
  if (cpu_neon) neon_diff = max_abs_diff(cpu, cpu_neon, batch * out_dim);
  if (cpu_neon4) neon4_diff = max_abs_diff(cpu, cpu_neon4, batch * out_dim);
#endif
  double ops = 2.0 * (double)batch * (double)out_dim * (double)in_dim;
  double gpu_ms = gpu_total / (double)repeats;
  double q_bytes = (double)w_bytes + (double)x_bytes + (double)y_bytes;

  printf("device=%s %s mode=%s batch=%u out=%u src_out=%u in=%u weight_mib=%.3f repeats=%u warmups=%u\n",
         props.deviceName, (q6_pre_mode || q6_idx_mode) ? "q6_matvec" : "q4_matvec", prepacked ? mode : (workgroup_per_row ? "wg64" : "row1"), batch, out_dim, src_out_dim, in_dim, (double)w_bytes / 1048576.0, repeats, warmups);
  if (prepack_ms > 0.0) printf("prepack_ms=%.3f\n", prepack_ms);
  if (prepack_ms < 0.0) printf("prepack_load_ms=%.3f\n", -prepack_ms);
  printf("max_abs_diff=%g gpu_ms_avg=%.3f gpu_gops=%.3f approx_stream_gib_s=%.3f cpu_ms=%.3f cpu_gops=%.3f speedup=%.3fx\n",
         diff, gpu_ms, ops / (gpu_ms / 1000.0) / 1e9, q_bytes / (gpu_ms / 1000.0) / 1073741824.0,
         cpu_ms, ops / (cpu_ms / 1000.0) / 1e9, cpu_ms / gpu_ms);
  if (q6_idx_mode && batch == 1u && out_dim > 0u) {
    printf("top1_match=%s gpu_top1_pos=%u gpu_top1_src=%u cpu_top1_pos=%u cpu_top1_src=%u top1_scan_ms=%.6f\n",
           top1_match ? "true" : "false", gpu_top1, gpu_top1_src, cpu_top1, cpu_top1_src, top1_scan_ms);
  }
#if defined(__aarch64__)
  if (cpu_neon) {
    printf("cpu_neon_ms=%.3f cpu_neon_gops=%.3f gpu_vs_neon=%.3fx neon_max_abs_diff=%g\n",
           cpu_neon_ms, ops / (cpu_neon_ms / 1000.0) / 1e9, cpu_neon_ms / gpu_ms, neon_diff);
    printf("cpu_neon4_ms=%.3f cpu_neon4_gops=%.3f gpu_vs_neon4=%.3fx neon4_max_abs_diff=%g\n",
           cpu_neon4_ms, ops / (cpu_neon4_ms / 1000.0) / 1e9, cpu_neon4_ms / gpu_ms, neon4_diff);
    free(cpu_neon);
    free(cpu_neon4);
  }
#endif

  free(cpu);
  vkDeviceWaitIdle(dev);
  vkDestroyCommandPool(dev, cp, NULL);
  vkDestroyDescriptorPool(dev, dp, NULL);
  vkDestroyPipeline(dev, pipe, NULL);
  vkDestroyPipelineLayout(dev, pl, NULL);
  vkDestroyDescriptorSetLayout(dev, dsl, NULL);
  vkDestroyShaderModule(dev, shader, NULL);
  free_buffer(dev, &yb);
  free_buffer(dev, &xb);
  free_buffer(dev, &wb);
  free_buffer(dev, &rowidb);
  free_buffer(dev, &rowmetab);
  vkDestroyDevice(dev, NULL);
  vkDestroyInstance(inst, NULL);
  free(tf.raw);
  return diff <= 1e-3f ? 0 : 2;
}
