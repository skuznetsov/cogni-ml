#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#define VK_CHECK(expr) do { VkResult _r = (expr); if (_r != VK_SUCCESS) die_vk(#expr, _r); } while (0)

typedef struct { VkBuffer buffer; VkDeviceMemory memory; void *mapped; size_t size; } Buf;
typedef struct { uint32_t type_id, out_dim, in_dim; uint64_t raw_bytes; uint8_t *raw; } TensorFile;
typedef struct { uint32_t out_dim, in_dim, row_bytes, repeats; } MatPush;
typedef struct { uint32_t n; } SwigluPush;

static void die(const char *msg) { fprintf(stderr, "error: %s\n", msg); exit(1); }
static void die_vk(const char *expr, VkResult r) { fprintf(stderr, "vulkan error: %s -> %d\n", expr, r); exit(1); }

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static char *read_file(const char *path, size_t *size_out) {
  FILE *f = fopen(path, "rb");
  if (!f) die("cannot open file");
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  rewind(f);
  if (n <= 0) die("empty file");
  char *buf = (char *)malloc((size_t)n);
  if (!buf) die("malloc failed");
  if (fread(buf, 1, (size_t)n, f) != (size_t)n) die("fread failed");
  fclose(f);
  *size_out = (size_t)n;
  return buf;
}

static uint32_t rd_u32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t rd_u64(const uint8_t *p) {
  uint64_t v = 0;
  for (uint32_t i = 0; i < 8; i++) v |= ((uint64_t)p[i]) << (i * 8u);
  return v;
}

static TensorFile read_tensor_file(const char *path) {
  size_t n = 0;
  uint8_t *buf = (uint8_t *)read_file(path, &n);
  if (n < 24) die("tensor file too small");
  if (rd_u32(buf) != 0x43564750u) die("bad tensor magic");
  TensorFile tf;
  tf.type_id = rd_u32(buf + 4);
  tf.out_dim = rd_u32(buf + 8);
  tf.in_dim = rd_u32(buf + 12);
  tf.raw_bytes = rd_u64(buf + 16);
  if (tf.type_id != 12u) die("expected Q4_K tensor");
  if (24u + tf.raw_bytes != n) die("tensor size mismatch");
  tf.raw = (uint8_t *)malloc((size_t)tf.raw_bytes);
  if (!tf.raw) die("tensor alloc failed");
  memcpy(tf.raw, buf + 24, (size_t)tf.raw_bytes);
  free(buf);
  return tf;
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

static float f16_to_f32(uint16_t h) {
  uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 31u, frac = h & 1023u;
  float v;
  if (exp == 0) v = frac == 0 ? 0.0f : ldexpf((float)frac, -24);
  else if (exp == 31) v = frac == 0 ? INFINITY : NAN;
  else v = ldexpf(1.0f + (float)frac / 1024.0f, (int)exp - 15);
  return sign ? -v : v;
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

static void prepack_q4k_to_q4_pre(uint8_t *dst, const uint8_t *src, uint32_t out_dim, uint32_t in_dim) {
  uint32_t src_blocks = in_dim / 256u;
  uint32_t src_row_bytes = src_blocks * 144u;
  uint32_t dst_row_bytes = (in_dim / 32u) * 24u;
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

static void cpu_q4_pre(float *y, const uint8_t *w, const float *x, uint32_t out_dim, uint32_t in_dim) {
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

static float max_abs_diff(const float *a, const float *b, uint32_t n) {
  float md = 0.0f;
  for (uint32_t i = 0; i < n; i++) {
    float d = fabsf(a[i] - b[i]);
    if (d > md) md = d;
  }
  return md;
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

static Buf make_buffer(VkDevice dev, VkPhysicalDevice pd, size_t size) {
  Buf b;
  memset(&b, 0, sizeof(b));
  b.size = size;
  VkBufferCreateInfo bi = {0};
  bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bi.size = size;
  bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
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

static VkShaderModule make_shader(VkDevice dev, const char *path) {
  size_t n = 0;
  char *spv = read_file(path, &n);
  VkShaderModuleCreateInfo smi = {0};
  smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  smi.codeSize = n;
  smi.pCode = (const uint32_t *)spv;
  VkShaderModule sh;
  VK_CHECK(vkCreateShaderModule(dev, &smi, NULL, &sh));
  free(spv);
  return sh;
}

static VkDescriptorSetLayout make_dsl(VkDevice dev, uint32_t count) {
  VkDescriptorSetLayoutBinding binds[5];
  memset(binds, 0, sizeof(binds));
  for (uint32_t i = 0; i < count; i++) {
    binds[i].binding = i;
    binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[i].descriptorCount = 1;
    binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo ci = {0};
  ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  ci.bindingCount = count;
  ci.pBindings = binds;
  VkDescriptorSetLayout dsl;
  VK_CHECK(vkCreateDescriptorSetLayout(dev, &ci, NULL, &dsl));
  return dsl;
}

static VkPipeline make_pipeline(VkDevice dev, VkShaderModule sh, VkPipelineLayout pl) {
  VkPipelineShaderStageCreateInfo stage = {0};
  stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stage.module = sh;
  stage.pName = "main";
  VkComputePipelineCreateInfo ci = {0};
  ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  ci.stage = stage;
  ci.layout = pl;
  VkPipeline p;
  VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, NULL, &p));
  return p;
}

static VkDescriptorSet make_ds(VkDevice dev, VkDescriptorPool dp, VkDescriptorSetLayout dsl,
                               Buf *a, Buf *b, Buf *c) {
  VkDescriptorSetAllocateInfo ai = {0};
  ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  ai.descriptorPool = dp;
  ai.descriptorSetCount = 1;
  ai.pSetLayouts = &dsl;
  VkDescriptorSet ds;
  VK_CHECK(vkAllocateDescriptorSets(dev, &ai, &ds));
  VkDescriptorBufferInfo infos[3] = {
    {a->buffer, 0, a->size}, {b->buffer, 0, b->size}, {c->buffer, 0, c->size}
  };
  VkWriteDescriptorSet writes[3];
  memset(writes, 0, sizeof(writes));
  for (uint32_t i = 0; i < 3; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = ds;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(dev, 3, writes, 0, NULL);
  return ds;
}

static VkDescriptorSet make_ds5(VkDevice dev, VkDescriptorPool dp, VkDescriptorSetLayout dsl,
                                Buf *a, Buf *b, Buf *c, Buf *d, Buf *e) {
  VkDescriptorSetAllocateInfo ai = {0};
  ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  ai.descriptorPool = dp;
  ai.descriptorSetCount = 1;
  ai.pSetLayouts = &dsl;
  VkDescriptorSet ds;
  VK_CHECK(vkAllocateDescriptorSets(dev, &ai, &ds));
  VkDescriptorBufferInfo infos[5] = {
    {a->buffer, 0, a->size}, {b->buffer, 0, b->size}, {c->buffer, 0, c->size},
    {d->buffer, 0, d->size}, {e->buffer, 0, e->size}
  };
  VkWriteDescriptorSet writes[5];
  memset(writes, 0, sizeof(writes));
  for (uint32_t i = 0; i < 5; i++) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = ds;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &infos[i];
  }
  vkUpdateDescriptorSets(dev, 5, writes, 0, NULL);
  return ds;
}

static void barrier_buf(VkCommandBuffer cmd, VkBuffer buf) {
  VkBufferMemoryBarrier b = {0};
  b.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  b.buffer = buf;
  b.offset = 0;
  b.size = VK_WHOLE_SIZE;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 0, NULL, 1, &b, 0, NULL);
}

int main(int argc, char **argv) {
  if (argc < 7) die("usage: ffn_probe MATVEC.spv SWIGLU.spv GATE.cvgp UP.cvgp DOWN.cvgp REPEATS [MATVEC_LOCAL_SIZE] [DOWN_MATVEC.spv] [BATCH] [TOKEN_PACK] [DUAL_GATE_UP.spv]");
  const char *mat_spv = argv[1], *swiglu_spv = argv[2];
  TensorFile gate_tf = read_tensor_file(argv[3]);
  TensorFile up_tf = read_tensor_file(argv[4]);
  TensorFile down_tf = read_tensor_file(argv[5]);
  uint32_t repeats = (uint32_t)strtoul(argv[6], NULL, 10);
  uint32_t matvec_local_size = argc > 7 ? (uint32_t)strtoul(argv[7], NULL, 10) : 64u;
  const char *down_mat_spv = argc > 8 ? argv[8] : mat_spv;
  uint32_t batch = 1u;
  uint32_t token_pack = 1u;
  const char *dual_mat_spv = NULL;
  if (argc > 9) {
    char c = argv[9][0];
    if (c >= '0' && c <= '9') {
      batch = (uint32_t)strtoul(argv[9], NULL, 10);
      if (argc > 10) token_pack = (uint32_t)strtoul(argv[10], NULL, 10);
      if (argc > 11) dual_mat_spv = argv[11];
    } else {
      dual_mat_spv = argv[9];
    }
  }
  if (matvec_local_size == 0u) die("MATVEC_LOCAL_SIZE must be > 0");
  if (batch == 0u || token_pack == 0u) die("batch and token_pack must be > 0");
  if (batch % token_pack != 0u) die("batch must be divisible by token_pack");
  if (!repeats) die("repeats must be positive");
  if (gate_tf.out_dim != up_tf.out_dim || gate_tf.in_dim != up_tf.in_dim) die("gate/up shape mismatch");
  if (down_tf.in_dim != gate_tf.out_dim || down_tf.out_dim == 0) die("down shape mismatch");
  uint32_t hidden = gate_tf.in_dim;
  uint32_t ffn = gate_tf.out_dim;
  uint32_t out_dim = down_tf.out_dim;
  uint32_t gate_row_bytes = (hidden / 32u) * 24u;
  uint32_t down_row_bytes = (ffn / 32u) * 24u;

  VkApplicationInfo app = {0};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "rpi5_vulkan_ffn_probe";
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

  Buf w_gate = make_buffer(dev, pd, (size_t)ffn * gate_row_bytes);
  Buf w_up = make_buffer(dev, pd, (size_t)ffn * gate_row_bytes);
  Buf w_down = make_buffer(dev, pd, (size_t)out_dim * down_row_bytes);
  Buf x = make_buffer(dev, pd, (size_t)batch * hidden * sizeof(float));
  Buf gate_y = make_buffer(dev, pd, (size_t)batch * ffn * sizeof(float));
  Buf up_y = make_buffer(dev, pd, (size_t)batch * ffn * sizeof(float));
  Buf mid = make_buffer(dev, pd, (size_t)batch * ffn * sizeof(float));
  Buf out = make_buffer(dev, pd, (size_t)batch * out_dim * sizeof(float));
  for (uint32_t b = 0; b < batch; b++) {
    for (uint32_t i = 0; i < hidden; i++) {
      ((float *)x.mapped)[(size_t)b * hidden + i] = ((int)((i + b * 7u) % 29) - 14) * 0.03125f;
    }
  }
  double pt0 = now_ms();
  prepack_q4k_to_q4_pre((uint8_t *)w_gate.mapped, gate_tf.raw, gate_tf.out_dim, gate_tf.in_dim);
  prepack_q4k_to_q4_pre((uint8_t *)w_up.mapped, up_tf.raw, up_tf.out_dim, up_tf.in_dim);
  prepack_q4k_to_q4_pre((uint8_t *)w_down.mapped, down_tf.raw, down_tf.out_dim, down_tf.in_dim);
  double prepack_ms = now_ms() - pt0;

  VkDescriptorSetLayout dsl = make_dsl(dev, 3);
  VkDescriptorSetLayout dual_dsl = make_dsl(dev, 5);
  VkPushConstantRange mat_pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MatPush)};
  VkPipelineLayoutCreateInfo mat_pli = {0};
  mat_pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  mat_pli.setLayoutCount = 1;
  mat_pli.pSetLayouts = &dsl;
  mat_pli.pushConstantRangeCount = 1;
  mat_pli.pPushConstantRanges = &mat_pcr;
  VkPipelineLayout mat_pl;
  VK_CHECK(vkCreatePipelineLayout(dev, &mat_pli, NULL, &mat_pl));
  VkPipelineLayoutCreateInfo dual_pli = mat_pli;
  dual_pli.pSetLayouts = &dual_dsl;
  VkPipelineLayout dual_pl;
  VK_CHECK(vkCreatePipelineLayout(dev, &dual_pli, NULL, &dual_pl));
  VkPushConstantRange sw_pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SwigluPush)};
  VkPipelineLayoutCreateInfo sw_pli = mat_pli;
  sw_pli.pPushConstantRanges = &sw_pcr;
  VkPipelineLayout sw_pl;
  VK_CHECK(vkCreatePipelineLayout(dev, &sw_pli, NULL, &sw_pl));
  VkShaderModule mat_sh = make_shader(dev, mat_spv);
  VkShaderModule down_mat_sh = make_shader(dev, down_mat_spv);
  VkShaderModule dual_mat_sh = dual_mat_spv ? make_shader(dev, dual_mat_spv) : VK_NULL_HANDLE;
  VkShaderModule sw_sh = make_shader(dev, swiglu_spv);
  VkPipeline mat_pipe = make_pipeline(dev, mat_sh, mat_pl);
  VkPipeline down_mat_pipe = make_pipeline(dev, down_mat_sh, mat_pl);
  VkPipeline dual_mat_pipe = dual_mat_spv ? make_pipeline(dev, dual_mat_sh, dual_pl) : VK_NULL_HANDLE;
  VkPipeline sw_pipe = make_pipeline(dev, sw_sh, sw_pl);
  VkDescriptorPoolSize pool_size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 20};
  VkDescriptorPoolCreateInfo dpci = {0};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.maxSets = 5;
  dpci.poolSizeCount = 1;
  dpci.pPoolSizes = &pool_size;
  VkDescriptorPool dp;
  VK_CHECK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
  VkDescriptorSet ds_gate = make_ds(dev, dp, dsl, &w_gate, &x, &gate_y);
  VkDescriptorSet ds_up = make_ds(dev, dp, dsl, &w_up, &x, &up_y);
  VkDescriptorSet ds_sw = make_ds(dev, dp, dsl, &gate_y, &up_y, &mid);
  VkDescriptorSet ds_down = make_ds(dev, dp, dsl, &w_down, &mid, &out);
  VkDescriptorSet ds_dual = make_ds5(dev, dp, dual_dsl, &w_gate, &w_up, &x, &gate_y, &up_y);

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
  VkCommandBufferBeginInfo cbi = {0};
  cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  VK_CHECK(vkBeginCommandBuffer(cmd, &cbi));
  MatPush gp = {ffn, hidden, gate_row_bytes, repeats};
  MatPush dpv = {out_dim, ffn, down_row_bytes, repeats};
  SwigluPush sp = {batch * ffn};
  uint32_t batch_groups = (batch + token_pack - 1u) / token_pack;
  if (dual_mat_spv) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dual_mat_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, dual_pl, 0, 1, &ds_dual, 0, NULL);
    vkCmdPushConstants(cmd, dual_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gp), &gp);
    vkCmdDispatch(cmd, (ffn + matvec_local_size - 1u) / matvec_local_size, batch_groups, 1);
    barrier_buf(cmd, gate_y.buffer);
    barrier_buf(cmd, up_y.buffer);
  } else {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mat_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mat_pl, 0, 1, &ds_gate, 0, NULL);
    vkCmdPushConstants(cmd, mat_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gp), &gp);
    vkCmdDispatch(cmd, (ffn + matvec_local_size - 1u) / matvec_local_size, batch_groups, 1);
    barrier_buf(cmd, gate_y.buffer);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mat_pl, 0, 1, &ds_up, 0, NULL);
    vkCmdPushConstants(cmd, mat_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gp), &gp);
    vkCmdDispatch(cmd, (ffn + matvec_local_size - 1u) / matvec_local_size, batch_groups, 1);
    barrier_buf(cmd, up_y.buffer);
  }
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sw_pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sw_pl, 0, 1, &ds_sw, 0, NULL);
  vkCmdPushConstants(cmd, sw_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(sp), &sp);
  vkCmdDispatch(cmd, (batch * ffn + 63u) / 64u, 1, 1);
  barrier_buf(cmd, mid.buffer);
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, down_mat_pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, mat_pl, 0, 1, &ds_down, 0, NULL);
  vkCmdPushConstants(cmd, mat_pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dpv), &dpv);
  vkCmdDispatch(cmd, (out_dim + matvec_local_size - 1u) / matvec_local_size, batch_groups, 1);
  VK_CHECK(vkEndCommandBuffer(cmd));
  VkSubmitInfo si = {0};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;
  double gpu_total = 0.0;
  for (uint32_t r = 0; r < repeats; r++) {
    memset(gate_y.mapped, 0, gate_y.size);
    memset(up_y.mapped, 0, up_y.size);
    memset(mid.mapped, 0, mid.size);
    memset(out.mapped, 0, out.size);
    double t0 = now_ms();
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
    gpu_total += now_ms() - t0;
  }

  float *cpu_gate = (float *)malloc(gate_y.size);
  float *cpu_up = (float *)malloc(up_y.size);
  float *cpu_mid = (float *)malloc(mid.size);
  float *cpu_out = (float *)malloc(out.size);
  if (!cpu_gate || !cpu_up || !cpu_mid || !cpu_out) die("cpu alloc failed");
  double ct0 = now_ms();
  for (uint32_t b = 0; b < batch; b++) {
    cpu_q4_pre(cpu_gate + (size_t)b * ffn, (const uint8_t *)w_gate.mapped, (const float *)x.mapped + (size_t)b * hidden, ffn, hidden);
    cpu_q4_pre(cpu_up + (size_t)b * ffn, (const uint8_t *)w_up.mapped, (const float *)x.mapped + (size_t)b * hidden, ffn, hidden);
    for (uint32_t i = 0; i < ffn; i++) {
      size_t off = (size_t)b * ffn + i;
      cpu_mid[off] = (cpu_gate[off] / (1.0f + expf(-cpu_gate[off]))) * cpu_up[off];
    }
    cpu_q4_pre(cpu_out + (size_t)b * out_dim, (const uint8_t *)w_down.mapped, cpu_mid + (size_t)b * ffn, out_dim, ffn);
  }
  double cpu_ms = now_ms() - ct0;
  float diff = max_abs_diff(cpu_out, (const float *)out.mapped, batch * out_dim);
  double gpu_ms = gpu_total / (double)repeats;
  double ops = (double)batch * (2.0 * (double)ffn * (double)hidden * 2.0 + 2.0 * (double)out_dim * (double)ffn);
  printf("device=%s q4_ffn_pre hidden=%u ffn=%u out=%u batch=%u token_pack=%u repeats=%u matvec_local_size=%u mat_spv=%s down_mat_spv=%s dual_mat_spv=%s\n",
         props.deviceName, hidden, ffn, out_dim, batch, token_pack, repeats, matvec_local_size, mat_spv, down_mat_spv,
         dual_mat_spv ? dual_mat_spv : "none");
  printf("prepack_ms=%.3f\n", prepack_ms);
  printf("max_abs_diff=%g gpu_ms_avg=%.3f gpu_gops=%.3f cpu_ms=%.3f speedup=%.3fx\n",
         diff, gpu_ms, ops / (gpu_ms / 1000.0) / 1e9, cpu_ms, cpu_ms / gpu_ms);

  free(cpu_gate); free(cpu_up); free(cpu_mid); free(cpu_out);
  vkDeviceWaitIdle(dev);
  vkDestroyCommandPool(dev, cp, NULL);
  vkDestroyDescriptorPool(dev, dp, NULL);
  vkDestroyPipeline(dev, mat_pipe, NULL);
  vkDestroyPipeline(dev, sw_pipe, NULL);
  vkDestroyShaderModule(dev, mat_sh, NULL);
  vkDestroyShaderModule(dev, sw_sh, NULL);
  vkDestroyPipelineLayout(dev, mat_pl, NULL);
  vkDestroyPipelineLayout(dev, sw_pl, NULL);
  vkDestroyDescriptorSetLayout(dev, dsl, NULL);
  free_buffer(dev, &out); free_buffer(dev, &mid); free_buffer(dev, &up_y); free_buffer(dev, &gate_y);
  free_buffer(dev, &x); free_buffer(dev, &w_down); free_buffer(dev, &w_up); free_buffer(dev, &w_gate);
  vkDestroyDevice(dev, NULL);
  vkDestroyInstance(inst, NULL);
  free(gate_tf.raw); free(up_tf.raw); free(down_tf.raw);
  return diff <= 3e-3f ? 0 : 2;
}
