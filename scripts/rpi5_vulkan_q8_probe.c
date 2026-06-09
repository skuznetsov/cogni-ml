#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

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

static void die(const char *msg) { fprintf(stderr, "error: %s\n", msg); exit(1); }
static void die_vk(const char *expr, VkResult r) { fprintf(stderr, "vulkan error: %s -> %d\n", expr, r); exit(1); }

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
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

static void fill_q8(uint8_t *w, float *x, uint32_t out_dim, uint32_t in_dim) {
  uint32_t blocks = in_dim / 32u;
  uint32_t row_bytes = blocks * 34u;
  for (uint32_t i = 0; i < in_dim; i++) x[i] = ((int)(i % 29) - 14) * 0.03125f;
  for (uint32_t row = 0; row < out_dim; row++) {
    for (uint32_t blk = 0; blk < blocks; blk++) {
      uint8_t *bp = w + row * row_bytes + blk * 34u;
      put_u16(bp, f32_to_f16(0.015625f + (float)((row + blk) & 3u) * 0.001953125f));
      for (uint32_t j = 0; j < 32; j++) {
        int v = (int)((row * 3u + blk * 5u + j * 7u) & 255u) - 128;
        bp[2 + j] = (uint8_t)(int8_t)v;
      }
    }
  }
}

static void fill_q8_prepacked(uint8_t *w, float *x, uint32_t out_dim, uint32_t in_dim) {
  uint32_t blocks = in_dim / 32u;
  uint32_t row_bytes = blocks * 36u;
  for (uint32_t i = 0; i < in_dim; i++) x[i] = ((int)(i % 29) - 14) * 0.03125f;
  for (uint32_t row = 0; row < out_dim; row++) {
    for (uint32_t blk = 0; blk < blocks; blk++) {
      uint32_t *bp = (uint32_t *)(void *)(w + row * row_bytes + blk * 36u);
      float d = 0.015625f + (float)((row + blk) & 3u) * 0.001953125f;
      memcpy(&bp[0], &d, sizeof(float));
      for (uint32_t q = 0; q < 8; q++) {
        uint32_t word = 0;
        for (uint32_t lane = 0; lane < 4; lane++) {
          uint32_t j = q * 4u + lane;
          int v = (int)((row * 3u + blk * 5u + j * 7u) & 255u) - 128;
          word |= ((uint32_t)(uint8_t)(int8_t)v) << (lane * 8u);
        }
        bp[1 + q] = word;
      }
    }
  }
}

static void cpu_q8(float *y, const uint8_t *w, const float *x, uint32_t out_dim, uint32_t in_dim) {
  uint32_t blocks = in_dim / 32u;
  uint32_t row_bytes = blocks * 34u;
  for (uint32_t row = 0; row < out_dim; row++) {
    double sum = 0.0;
    const uint8_t *wr = w + row * row_bytes;
    for (uint32_t blk = 0; blk < blocks; blk++) {
      const uint8_t *bp = wr + blk * 34u;
      float d = f16_to_f32((uint16_t)(bp[0] | (bp[1] << 8)));
      for (uint32_t j = 0; j < 32; j++) {
        sum += x[blk * 32u + j] * d * (float)((int8_t)bp[2 + j]);
      }
    }
    y[row] = (float)sum;
  }
}

static void cpu_q8_prepacked(float *y, const uint8_t *w, const float *x, uint32_t out_dim, uint32_t in_dim) {
  uint32_t blocks = in_dim / 32u;
  uint32_t row_bytes = blocks * 36u;
  for (uint32_t row = 0; row < out_dim; row++) {
    double sum = 0.0;
    const uint8_t *wr = w + row * row_bytes;
    for (uint32_t blk = 0; blk < blocks; blk++) {
      const uint32_t *bp = (const uint32_t *)(const void *)(wr + blk * 36u);
      float d;
      memcpy(&d, &bp[0], sizeof(float));
      for (uint32_t q = 0; q < 8; q++) {
        uint32_t word = bp[1 + q];
        for (uint32_t lane = 0; lane < 4; lane++) {
          uint8_t b = (uint8_t)((word >> (lane * 8u)) & 0xffu);
          sum += x[blk * 32u + q * 4u + lane] * d * (float)((int8_t)b);
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

int main(int argc, char **argv) {
  const char *spv_path = argc > 1 ? argv[1] : "rpi5_q8_matvec.spv";
  uint32_t out_dim = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 4096;
  uint32_t in_dim = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : 4096;
  uint32_t repeats = argc > 4 ? (uint32_t)strtoul(argv[4], NULL, 10) : 50;
  const char *mode = argc > 5 ? argv[5] : "raw";
  int row_group4 = strcmp(mode, "rg4") == 0;
  int prepacked = strcmp(mode, "pre") == 0 || strncmp(mode, "pre_l", 5) == 0;
  uint32_t dispatch_local_size = 64u;
  if (strncmp(mode, "pre_l", 5) == 0) {
    dispatch_local_size = (uint32_t)strtoul(mode + 5, NULL, 10);
  }
  if (!row_group4 && dispatch_local_size == 0u) die("invalid local size mode");
  if (!out_dim || !in_dim || in_dim % 32u != 0 || !repeats) die("usage: probe SPV OUT_DIM IN_DIM REPEATS; IN_DIM must be multiple of 32");

  VkApplicationInfo app = {0};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "rpi5_vulkan_q8_probe";
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

  uint32_t row_bytes = (in_dim / 32u) * (prepacked ? 36u : 34u);
  size_t w_bytes = (size_t)out_dim * row_bytes;
  size_t x_bytes = (size_t)in_dim * sizeof(float);
  size_t y_bytes = (size_t)out_dim * sizeof(float);
  Buf wb = make_buffer(dev, pd, w_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  Buf xb = make_buffer(dev, pd, x_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  Buf yb = make_buffer(dev, pd, y_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  float *cpu = (float *)malloc(y_bytes);
  if (!cpu) die("cpu alloc failed");
  if (prepacked) {
    fill_q8_prepacked((uint8_t *)wb.mapped, (float *)xb.mapped, out_dim, in_dim);
  } else {
    fill_q8((uint8_t *)wb.mapped, (float *)xb.mapped, out_dim, in_dim);
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

  VkDescriptorSetLayoutBinding binds[3];
  memset(binds, 0, sizeof(binds));
  for (uint32_t i = 0; i < 3; i++) {
    binds[i].binding = i;
    binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    binds[i].descriptorCount = 1;
    binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dsli = {0};
  dsli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dsli.bindingCount = 3;
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

  VkDescriptorPoolSize pool_size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
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
  VkDescriptorBufferInfo infos[3] = {{wb.buffer, 0, w_bytes}, {xb.buffer, 0, x_bytes}, {yb.buffer, 0, y_bytes}};
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
  if (row_group4) {
    vkCmdDispatch(cmd, (out_dim + 3u) / 4u, 1, 1);
  } else {
    vkCmdDispatch(cmd, (out_dim + dispatch_local_size - 1u) / dispatch_local_size, 1, 1);
  }
  VK_CHECK(vkEndCommandBuffer(cmd));
  VkSubmitInfo si = {0};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;

  double gpu_total = 0.0;
  for (uint32_t r = 0; r < repeats; r++) {
    memset(yb.mapped, 0, y_bytes);
    double t0 = now_ms();
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
    gpu_total += now_ms() - t0;
  }

  double t0 = now_ms();
  if (prepacked) {
    cpu_q8_prepacked(cpu, (const uint8_t *)wb.mapped, (const float *)xb.mapped, out_dim, in_dim);
  } else {
    cpu_q8(cpu, (const uint8_t *)wb.mapped, (const float *)xb.mapped, out_dim, in_dim);
  }
  double cpu_ms = now_ms() - t0;
  float diff = max_abs_diff(cpu, (const float *)yb.mapped, out_dim);
  double ops = 2.0 * (double)out_dim * (double)in_dim;
  double gpu_ms = gpu_total / (double)repeats;
  double q_bytes = (double)w_bytes + (double)x_bytes + (double)y_bytes;

  printf("device=%s q8_matvec mode=%s out=%u in=%u weight_mib=%.3f repeats=%u\n",
         props.deviceName, prepacked ? mode : (row_group4 ? "rg4" : "row1"), out_dim, in_dim, (double)w_bytes / 1048576.0, repeats);
  printf("max_abs_diff=%g gpu_ms_avg=%.3f gpu_gops=%.3f approx_stream_gib_s=%.3f cpu_ms=%.3f cpu_gops=%.3f speedup=%.3fx\n",
         diff, gpu_ms, ops / (gpu_ms / 1000.0) / 1e9, q_bytes / (gpu_ms / 1000.0) / 1073741824.0,
         cpu_ms, ops / (cpu_ms / 1000.0) / 1e9, cpu_ms / gpu_ms);

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
  vkDestroyDevice(dev, NULL);
  vkDestroyInstance(inst, NULL);
  return diff <= 1e-3f ? 0 : 2;
}
