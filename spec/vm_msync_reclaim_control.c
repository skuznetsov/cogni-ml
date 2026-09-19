// One-page owned-file instrument qualification. No Metal or memory pressure.
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

static char path[] = "/private/tmp/qwen-reclaim-page-XXXXXX";
static int created;
static void cleanup(void) { if (created) unlink(path); }
static void check(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "FAIL %s errno=%d\n", what, errno); exit(65); }
}
static int present(void *p, size_t n) {
    char v = 0;
    check(mincore(p, n, &v) == 0, "mincore");
    return (v & MINCORE_INCORE) != 0;
}
static void invalidate(void *p, size_t n, const char *phase) {
    errno = 0;
    int rc = msync(p, n, MS_SYNC | MS_INVALIDATE);
    int error = errno;
    int in_core = present(p, n);
    printf("phase=%s msync_rc=%d errno=%d incore=%d\n", phase, rc, error, in_core);
    fflush(stdout);
}
int main(void) {
    size_t n = (size_t)getpagesize();
    check(n == 16384, "bounded page size");
    check(atexit(cleanup) == 0, "atexit");
    int writer = mkstemp(path);
    check(writer >= 0, "mkstemp");
    created = 1;
    fprintf(stderr, "fixture=%s bytes=%zu\n", path, n);
    unsigned char *data = malloc(n);
    check(data != NULL, "malloc");
    memset(data, 0x5a, n);
    check(write(writer, data, n) == (ssize_t)n, "write");
    free(data);
    check(fsync(writer) == 0 && close(writer) == 0, "sync/close");
    int fd = open(path, O_RDONLY);
    check(fd >= 0, "open");
    void *p = mmap(NULL, n, PROT_READ, MAP_PRIVATE, fd, 0);
    check(p != MAP_FAILED, "mmap");
    check(*(volatile unsigned char *)p == 0x5a && present(p, n), "touch positive");
    invalidate(p, n, "clean_unlocked");
    check(*(volatile unsigned char *)p == 0x5a && present(p, n), "retouch before lock");
    check(mlock(p, n) == 0, "mlock positive");
    invalidate(p, n, "locked");
    check(munlock(p, n) == 0, "munlock");
    invalidate(p, n, "unlocked");
    check(*(volatile unsigned char *)p == 0x5a, "refault payload");
    printf("phase=refault incore=%d payload_ok=1\n", present(p, n));
    check(munmap(p, n) == 0 && close(fd) == 0, "unmap/close");
    check(unlink(path) == 0, "unlink");
    created = 0;
    return 0;
}
