#ifndef PIXELUMM_GC_OBSERVER_H
#define PIXELUMM_GC_OBSERVER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum PixelUMMGcObserverStatus {
  PIXELUMM_GC_OBSERVER_OK = 0,
  PIXELUMM_GC_OBSERVER_INVALID_ARGUMENT = 1,
  PIXELUMM_GC_OBSERVER_UNSUPPORTED = 2,
  PIXELUMM_GC_OBSERVER_CALLBACK_OCCUPIED = 3,
  PIXELUMM_GC_OBSERVER_ALREADY_ACTIVE = 4,
  PIXELUMM_GC_OBSERVER_NOT_ACTIVE = 5,
  PIXELUMM_GC_OBSERVER_CALLBACK_CHANGED = 6,
  PIXELUMM_GC_OBSERVER_CONTROL_BUSY = 7,
  PIXELUMM_GC_OBSERVER_CLOCK_FAILURE = 8
} PixelUMMGcObserverStatus;

typedef struct PixelUMMGcObserverSnapshot {
  /* GC_EVENT_START -> GC_EVENT_END collector envelope; not mutator pause time. */
  uint64_t collection_count;
  uint64_t collection_total_ns;
  uint64_t collection_max_ns;
  /* GC_EVENT_PRE_STOP_WORLD -> GC_EVENT_POST_START_WORLD, including rendezvous. */
  uint64_t rendezvous_pause_count;
  uint64_t rendezvous_pause_total_ns;
  uint64_t rendezvous_pause_max_ns;
  /* GC_EVENT_POST_STOP_WORLD -> GC_EVENT_PRE_START_WORLD, while fully stopped. */
  uint64_t stopped_interval_count;
  uint64_t stopped_interval_total_ns;
  uint64_t stopped_interval_max_ns;
  uint64_t unmatched_event_count;
  uint32_t active;
  uint32_t callback_installed;
} PixelUMMGcObserverSnapshot;

/*
 * Timing fields use monotonic nanoseconds. The observer records collector
 * events only; it does not measure allocated bytes. The three durations are
 * distinct windows and must not be summed as if they were disjoint.
 */
/*
 * Install an allocation-free Boehm collection-event observer. Start fails if
 * another callback already owns the global event slot. The caller must not
 * concurrently register or replace a callback. Snapshot is for phase
 * boundaries outside a Boehm callback because it reads the callback slot
 * under Boehm's lock.
 */
int pixelumm_gc_observer_start(void);

/* Restore the initially empty callback slot. A changed slot is left untouched. */
int pixelumm_gc_observer_stop(void);

/* Read a coherent counter snapshot and the live callback-slot identity. */
int pixelumm_gc_observer_snapshot(PixelUMMGcObserverSnapshot *out);

#ifdef __cplusplus
}
#endif

#endif
