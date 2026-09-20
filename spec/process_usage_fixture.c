// Own-process controls only: <32MiB anonymous memory, 8MiB temporary file, no GPU.
#define main sampler_main
#include "../bin/process_usage_sample.c"
#undef main
#include <fcntl.h>
#include <sys/mman.h>

static void require(int ok, const char *message) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", message); exit(1); }
}

static struct rusage_info_v2 usage(void) {
    struct rusage_info_v2 value = {0};
    require(proc_pid_rusage(getpid(), RUSAGE_INFO_V2, (rusage_info_t *)&value) == 0, "self usage");
    return value;
}

int main(void) {
    require(mach_timebase_info(&timebase) == KERN_SUCCESS, "timebase");
    puts("ready"); fflush(stdout);
    const double started = mach_seconds();
    usleep(1000000);
    struct rusage_info_v2 before = usage();
    struct rusage r0, r1;
    require(getrusage(RUSAGE_SELF, &r0) == 0, "getrusage before");
    volatile uint64_t total = 0;
    double until = mach_seconds() + .12;
    while (mach_seconds() < until) for (int i = 0; i < 10000; ++i) total += i;
    require(getrusage(RUSAGE_SELF, &r1) == 0, "getrusage after");
    struct rusage_info_v2 busy = usage();
    const double cpu_ms = (nanoseconds(busy.ri_user_time - before.ri_user_time) +
        nanoseconds(busy.ri_system_time - before.ri_system_time)) / 1e6;
    const double rusage_ms = (r1.ru_utime.tv_sec - r0.ru_utime.tv_sec +
        r1.ru_stime.tv_sec - r0.ru_stime.tv_sec) * 1000.0 +
        (r1.ru_utime.tv_usec - r0.ru_utime.tv_usec + r1.ru_stime.tv_usec - r0.ru_stime.tv_usec) / 1000.0;
    const size_t bytes = 32ULL << 20;
    volatile unsigned char *memory = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    require(memory != MAP_FAILED, "map");
    for (size_t i = 0; i < bytes; i += getpagesize()) memory[i] = 123;
    struct rusage_info_v2 touched = usage();
    // A small uncached read qualifies counter sensitivity, not model mmap I/O attribution.
    char path[] = "/private/tmp/cogni-usage-fixture-XXXXXX";
    int fd = mkstemp(path);
    require(fd >= 0, "temporary file");
    require(unlink(path) == 0 && fcntl(fd, F_NOCACHE, 1) == 0, "private uncached file");
    char block[65536]; memset(block, 0x5a, sizeof(block));
    for (int i = 0; i < 128; ++i) require(write(fd, block, sizeof(block)) == sizeof(block), "write");
    require(fsync(fd) == 0 && lseek(fd, 0, SEEK_SET) == 0, "sync/seek");
    struct rusage_info_v2 io_before = usage();
    for (int i = 0; i < 128; ++i) require(read(fd, block, sizeof(block)) == sizeof(block), "read");
    struct rusage_info_v2 io_after = usage();
    require(close(fd) == 0, "close");
    while (mach_seconds() < started + 5) usleep(10000);
    printf("{\"cpu_ms\":%.6f,\"cpu_to_getrusage_ratio\":%.6f,\"resident_delta\":%" PRIu64
        ",\"disk_read_delta\":%" PRIu64 ",\"pagein_delta\":%" PRIu64 "}\n", cpu_ms,
        cpu_ms / rusage_ms, touched.ri_resident_size - before.ri_resident_size,
        io_after.ri_diskio_bytesread - io_before.ri_diskio_bytesread,
        io_after.ri_pageins - io_before.ri_pageins);
    require(munmap((void *)memory, bytes) == 0, "unmap");
    return total == 0;
}
