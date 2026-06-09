#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#define VK_CHECK(expr) do { VkResult _r = (expr); if (_r != VK_SUCCESS) die_vk(#expr, _r); } while (0)

typedef struct {
  uint32_t m;
  uint32_t n;
  uint32_t k;
} Push;

static void die(const char *msg) {
  fprintf(stderr, "error: %s\n", msg);
  exit(1);
}

static void die_vk(const char *expr, VkResult r) {
  fprintf(stderr, "vulkan error: %s -> %d\n", expr, r);
  exit(1);
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static uint32_t ceil_div(uint32_t a, uint32_t b) {
  return (a + b - 1) / b;
}

static uint32_t find_memory_type(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags flags) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    if ((bits & (1u << i)) && ((mp.memoryTypes[i].propertyFlags & flags) == flags)) {
      return i;
    }
  }
  die("no compatible memory type");
  return 0;
}

static char *read_file(const char *path, size_t *size_out) {
  FILE *f = fopen(path, "rb");
  if (!f) die("cannot open SPIR-V file");
  if (fseek(f, 0, SEEK_END) != 0) die("fseek failed");
  long n = ftell(f);
  if (n <= 0) die("empty SPIR-V file");
  rewind(f);
  char *buf = (char *)malloc((size_t)n);
  if (!buf) die("malloc failed");
  if (fread(buf, 1, (size_t)n, f) != (size_t)n) die("fread failed");
  fclose(f);
  *size_out = (size_t)n;
  return buf;
}

typedef struct {
  VkBuffer buffer;
  VkDeviceMemory memory;
  void *mapped;
  size_t size;
} Buf;

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

static void fill_inputs(float *a, float *b, uint32_t m, uint32_t n, uint32_t k) {
  for (uint32_t i = 0; i < m * k; i++) {
    a[i] = (float)((int)(i % 17) - 8) * 0.03125f;
  }
  for (uint32_t i = 0; i < k * n; i++) {
    b[i] = (float)((int)(i % 23) - 11) * 0.015625f;
  }
}

static void cpu_matmul(float *c, const float *a, const float *b, uint32_t m, uint32_t n, uint32_t k) {
  for (uint32_t row = 0; row < m; row++) {
    for (uint32_t col = 0; col < n; col++) {
      float acc = 0.0f;
      for (uint32_t x = 0; x < k; x++) {
        acc += a[row * k + x] * b[x * n + col];
      }
      c[row * n + col] = acc;
    }
  }
}

static float max_abs_diff(const float *x, const float *y, size_t n) {
  float md = 0.0f;
  for (size_t i = 0; i < n; i++) {
    float d = fabsf(x[i] - y[i]);
    if (d > md) md = d;
  }
  return md;
}

int main(int argc, char **argv) {
  const char *spv_path = argc > 1 ? argv[1] : "rpi5_matmul.spv";
  uint32_t m = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 256;
  uint32_t n = argc > 3 ? (uint32_t)strtoul(argv[3], NULL, 10) : m;
  uint32_t k = argc > 4 ? (uint32_t)strtoul(argv[4], NULL, 10) : m;
  uint32_t repeats = argc > 5 ? (uint32_t)strtoul(argv[5], NULL, 10) : 10;
  if (!m || !n || !k || !repeats) die("shape and repeats must be positive");

  VkApplicationInfo app = {0};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = "rpi5_vulkan_matmul_probe";
  app.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo ici = {0};
  ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  ici.pApplicationInfo = &app;
  VkInstance inst;
  VK_CHECK(vkCreateInstance(&ici, NULL, &inst));

  uint32_t pd_count = 0;
  VK_CHECK(vkEnumeratePhysicalDevices(inst, &pd_count, NULL));
  if (pd_count == 0) die("no Vulkan physical devices");
  VkPhysicalDevice *pds = (VkPhysicalDevice *)calloc(pd_count, sizeof(VkPhysicalDevice));
  VK_CHECK(vkEnumeratePhysicalDevices(inst, &pd_count, pds));
  VkPhysicalDevice pd = pds[0];
  free(pds);

  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(pd, &props);
  VkPhysicalDeviceMemoryProperties mem_props;
  vkGetPhysicalDeviceMemoryProperties(pd, &mem_props);

  uint32_t q_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &q_count, NULL);
  VkQueueFamilyProperties *qprops = (VkQueueFamilyProperties *)calloc(q_count, sizeof(VkQueueFamilyProperties));
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &q_count, qprops);
  uint32_t qfam = UINT32_MAX;
  for (uint32_t i = 0; i < q_count; i++) {
    if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      qfam = i;
      break;
    }
  }
  free(qprops);
  if (qfam == UINT32_MAX) die("no compute queue family");

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

  size_t a_bytes = (size_t)m * k * sizeof(float);
  size_t b_bytes = (size_t)k * n * sizeof(float);
  size_t c_bytes = (size_t)m * n * sizeof(float);
  Buf ab = make_buffer(dev, pd, a_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  Buf bb = make_buffer(dev, pd, b_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  Buf cb = make_buffer(dev, pd, c_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  float *cpu = (float *)malloc(c_bytes);
  if (!cpu) die("malloc cpu output failed");
  fill_inputs((float *)ab.mapped, (float *)bb.mapped, m, n, k);
  memset(cb.mapped, 0, c_bytes);

  size_t spv_size = 0;
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

  VkPushConstantRange pcr = {0};
  pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pcr.offset = 0;
  pcr.size = sizeof(Push);
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

  VkDescriptorBufferInfo infos[3] = {
    {ab.buffer, 0, a_bytes},
    {bb.buffer, 0, b_bytes},
    {cb.buffer, 0, c_bytes},
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

  Push push = {m, n, k};
  VkCommandBufferBeginInfo cbi = {0};
  cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  VK_CHECK(vkBeginCommandBuffer(cmd, &cbi));
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
  vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &push);
  vkCmdDispatch(cmd, ceil_div(n, 8), ceil_div(m, 8), 1);
  VK_CHECK(vkEndCommandBuffer(cmd));

  VkSubmitInfo si = {0};
  si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cmd;

  double gpu_total = 0.0;
  for (uint32_t r = 0; r < repeats; r++) {
    memset(cb.mapped, 0, c_bytes);
    double t0 = now_ms();
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
    gpu_total += now_ms() - t0;
  }

  double t0 = now_ms();
  cpu_matmul(cpu, (const float *)ab.mapped, (const float *)bb.mapped, m, n, k);
  double cpu_ms = now_ms() - t0;
  float diff = max_abs_diff(cpu, (const float *)cb.mapped, (size_t)m * n);
  double flops = 2.0 * (double)m * (double)n * (double)k;
  double gpu_ms = gpu_total / (double)repeats;

  printf("device=%s api=%u.%u.%u driver=%u heap0_bytes=%llu\n",
         props.deviceName,
         VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion),
         props.driverVersion,
         (unsigned long long)mem_props.memoryHeaps[0].size);
  printf("shape=%ux%u x %ux%u repeats=%u max_abs_diff=%g\n", m, k, k, n, repeats, diff);
  printf("gpu_ms_avg=%.3f gpu_gflops=%.3f cpu_ms=%.3f cpu_gflops=%.3f speedup=%.3fx\n",
         gpu_ms, flops / (gpu_ms / 1000.0) / 1e9,
         cpu_ms, flops / (cpu_ms / 1000.0) / 1e9,
         cpu_ms / gpu_ms);

  free(cpu);
  vkDeviceWaitIdle(dev);
  vkDestroyCommandPool(dev, cp, NULL);
  vkDestroyDescriptorPool(dev, dp, NULL);
  vkDestroyPipeline(dev, pipe, NULL);
  vkDestroyPipelineLayout(dev, pl, NULL);
  vkDestroyDescriptorSetLayout(dev, dsl, NULL);
  vkDestroyShaderModule(dev, shader, NULL);
  free_buffer(dev, &cb);
  free_buffer(dev, &bb);
  free_buffer(dev, &ab);
  vkDestroyDevice(dev, NULL);
  vkDestroyInstance(inst, NULL);
  return diff <= 1e-3f ? 0 : 2;
}
