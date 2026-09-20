// Read-only macOS process counters, not a system-wide I/O or GPU residency probe.
// Usage: sampler PID EXPECTED_PPID /absolute/executable SECONDS (1..30).
#include <errno.h>
#include <inttypes.h>
#include <libproc.h>
#include <limits.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

static mach_timebase_info_data_t timebase;

static double mach_seconds(void) {
    return (double)mach_absolute_time() * timebase.numer / timebase.denom / 1e9;
}

static int number(const char *s, int low, int high) {
    char *end;
    errno = 0;
    long value = strtol(s, &end, 10);
    if (errno || !*s || *end || value < low || value > high) return -1;
    return (int)value;
}

// The kernel supplies Mach CPU ticks. Keep raw ticks and timebase too, so units
// remain auditable. The companion fixture cross-checks against getrusage time.
static uint64_t nanoseconds(uint64_t ticks) {
    return (uint64_t)((__uint128_t)ticks * timebase.numer / timebase.denom);
}

int main(int argc, char **argv) {
    int pid, ppid, seconds;
    if (argc != 5 || (pid = number(argv[1], 2, INT_MAX)) < 0 ||
        (ppid = number(argv[2], 2, INT_MAX)) < 0 || argv[3][0] != '/' ||
        (seconds = number(argv[4], 1, 30)) < 0) {
        fprintf(stderr, "expected PID PPID /absolute/executable SECONDS(1..30)\n");
        return 64;
    }
    if (mach_timebase_info(&timebase) != KERN_SUCCESS || !timebase.denom) return 65;
    const double deadline = mach_seconds() + seconds;
    uint64_t start = 0;
    unsigned samples = 0;
    const char *reason = "deadline";
    while (mach_seconds() < deadline) {
        const double begin = mach_seconds();
        struct rusage_info_v2 before = {0}, after = {0};
        struct proc_bsdinfo bsd = {0};
        char path[PROC_PIDPATHINFO_MAXSIZE] = {0};
        errno = 0;
        if (proc_pid_rusage(pid, RUSAGE_INFO_V2, (rusage_info_t *)&before) != 0) {
            if (errno == ESRCH && samples) { reason = "exited"; break; }
            perror("proc_pid_rusage");
            return 65;
        }
        if (before.ri_proc_exit_abstime && samples && before.ri_proc_start_abstime == start) {
            reason = "exited";
            break;
        }
        // Bracket path/parent checks with start identity checks. Any race or
        // unavailable identity is inconclusive, never a silently valid sample.
        if (proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &bsd, sizeof(bsd)) != sizeof(bsd) ||
            proc_pidpath(pid, path, sizeof(path)) <= 0 || strcmp(path, argv[3]) ||
            bsd.pbi_ppid != (uint32_t)ppid || bsd.pbi_uid != getuid() ||
            proc_pid_rusage(pid, RUSAGE_INFO_V2, (rusage_info_t *)&after) != 0 ||
            !before.ri_proc_start_abstime || before.ri_proc_start_abstime != after.ri_proc_start_abstime ||
            (samples && before.ri_proc_start_abstime != start)) {
            fprintf(stderr, "identity mismatch or unavailable; discard observation\n");
            return 65;
        }
        start = before.ri_proc_start_abstime;
        const double end = mach_seconds();
        printf("{\"event\":\"usage\",\"pid\":%d,\"ppid\":%d,\"start_abstime\":%" PRIu64
            ",\"mach_begin_s\":%.9f,\"mach_end_s\":%.9f,\"timebase_numer\":%u,\"timebase_denom\":%u,"
            "\"user_ticks\":%" PRIu64 ",\"system_ticks\":%" PRIu64
            ",\"user_ns\":%" PRIu64 ",\"system_ns\":%" PRIu64
            ",\"pageins\":%" PRIu64 ",\"disk_read_bytes\":%" PRIu64
            ",\"disk_write_bytes\":%" PRIu64 ",\"resident_bytes\":%" PRIu64
            ",\"footprint_bytes\":%" PRIu64 "}\n",
            pid, ppid, start, begin, end, timebase.numer, timebase.denom,
            before.ri_user_time, before.ri_system_time,
            nanoseconds(before.ri_user_time), nanoseconds(before.ri_system_time),
            before.ri_pageins, before.ri_diskio_bytesread, before.ri_diskio_byteswritten,
            before.ri_resident_size, before.ri_phys_footprint);
        if (fflush(stdout) != 0 || ferror(stdout)) return 65;
        ++samples;
        struct timespec delay = {0, 100000000};
        while (nanosleep(&delay, &delay) != 0) if (errno != EINTR) return 65;
    }
    printf("{\"event\":\"end\",\"reason\":\"%s\",\"samples\":%u}\n", reason, samples);
    return fflush(stdout) == 0 && !ferror(stdout) && samples ? 0 : 65;
}
