#ifndef COGNIGRAPH_SCHEDULE_H
#define COGNIGRAPH_SCHEDULE_H

#include <stdint.h>

typedef struct GsCogniGraphAccess {
  uint64_t buffer_id;
  int64_t offset;
  int64_t length;
  int32_t access;     /* 0 = read, 1 = write, 2 = read-write */
  int32_t partition;  /* negative means unpartitioned */
} GsCogniGraphAccess;

/* Called only during native graph preparation, never during request encoding. */
typedef int32_t (*GsCogniGraphCompile)(const GsCogniGraphAccess *accesses,
                                      const uint32_t *op_offsets,
                                      uint32_t op_count,
                                      uint32_t access_count,
                                      int32_t *wave_of,
                                      uint32_t *wave_count);

#endif
