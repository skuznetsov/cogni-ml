#include "pixelumm_gc_observer.h"

#include <gc/gc.h>

#include <atomic>
#include <cstdint>

#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif

static_assert(std::atomic<uint64_t>::is_always_lock_free,
              "PixelUMM GC observer requires lock-free 64-bit atomics");
static_assert(std::atomic<uint32_t>::is_always_lock_free,
              "PixelUMM GC observer requires lock-free 32-bit atomics");

namespace {

std::atomic_flag control_lock = ATOMIC_FLAG_INIT;
std::atomic<bool> active{false};
std::atomic<uint64_t> write_sequence{0};

std::atomic<uint64_t> collection_count{0};
std::atomic<uint64_t> collection_total_ns{0};
std::atomic<uint64_t> collection_max_ns{0};
std::atomic<uint64_t> rendezvous_pause_count{0};
std::atomic<uint64_t> rendezvous_pause_total_ns{0};
std::atomic<uint64_t> rendezvous_pause_max_ns{0};
std::atomic<uint64_t> stopped_interval_count{0};
std::atomic<uint64_t> stopped_interval_total_ns{0};
std::atomic<uint64_t> stopped_interval_max_ns{0};
std::atomic<uint64_t> unmatched_event_count{0};

bool collection_open = false;
uint64_t collection_start_ticks = 0;
bool rendezvous_open = false;
uint64_t rendezvous_start_ticks = 0;
bool stopped_open = false;
uint64_t stopped_start_ticks = 0;

#if defined(__APPLE__)
uint32_t timebase_numer = 0;
uint32_t timebase_denom = 0;
#endif

bool try_control_lock() {
  return !control_lock.test_and_set(std::memory_order_acquire);
}

void release_control_lock() {
  control_lock.clear(std::memory_order_release);
}

void begin_write() {
  write_sequence.fetch_add(1, std::memory_order_acq_rel);
}

void end_write() {
  write_sequence.fetch_add(1, std::memory_order_release);
}

void reset_counters() {
  collection_count.store(0, std::memory_order_relaxed);
  collection_total_ns.store(0, std::memory_order_relaxed);
  collection_max_ns.store(0, std::memory_order_relaxed);
  rendezvous_pause_count.store(0, std::memory_order_relaxed);
  rendezvous_pause_total_ns.store(0, std::memory_order_relaxed);
  rendezvous_pause_max_ns.store(0, std::memory_order_relaxed);
  stopped_interval_count.store(0, std::memory_order_relaxed);
  stopped_interval_total_ns.store(0, std::memory_order_relaxed);
  stopped_interval_max_ns.store(0, std::memory_order_relaxed);
  unmatched_event_count.store(0, std::memory_order_relaxed);
  collection_open = false;
  collection_start_ticks = 0;
  rendezvous_open = false;
  rendezvous_start_ticks = 0;
  stopped_open = false;
  stopped_start_ticks = 0;
}

#if defined(__APPLE__)
uint64_t duration_ns(uint64_t start_ticks, uint64_t end_ticks) {
  if (end_ticks < start_ticks || timebase_denom == 0) {
    return 0;
  }
  const uint64_t ticks = end_ticks - start_ticks;
  const uint64_t whole = ticks / timebase_denom;
  const uint64_t remainder = ticks % timebase_denom;
  return whole * timebase_numer + (remainder * timebase_numer) / timebase_denom;
}
#else
uint64_t duration_ns(uint64_t, uint64_t) { return 0; }
#endif

void update_max(std::atomic<uint64_t> &target, uint64_t value) {
  uint64_t current = target.load(std::memory_order_relaxed);
  while (current < value &&
         !target.compare_exchange_weak(current, value, std::memory_order_relaxed,
                                       std::memory_order_relaxed)) {
  }
}

void record_collection(uint64_t start_ticks, uint64_t end_ticks) {
  const uint64_t elapsed = duration_ns(start_ticks, end_ticks);
  collection_count.fetch_add(1, std::memory_order_relaxed);
  collection_total_ns.fetch_add(elapsed, std::memory_order_relaxed);
  update_max(collection_max_ns, elapsed);
}

void record_rendezvous_pause(uint64_t start_ticks, uint64_t end_ticks) {
  const uint64_t elapsed = duration_ns(start_ticks, end_ticks);
  rendezvous_pause_count.fetch_add(1, std::memory_order_relaxed);
  rendezvous_pause_total_ns.fetch_add(elapsed, std::memory_order_relaxed);
  update_max(rendezvous_pause_max_ns, elapsed);
}

void record_stopped_interval(uint64_t start_ticks, uint64_t end_ticks) {
  const uint64_t elapsed = duration_ns(start_ticks, end_ticks);
  stopped_interval_count.fetch_add(1, std::memory_order_relaxed);
  stopped_interval_total_ns.fetch_add(elapsed, std::memory_order_relaxed);
  update_max(stopped_interval_max_ns, elapsed);
}

void GC_CALLBACK collection_event(GC_EventType event) {
  if (!active.load(std::memory_order_acquire)) {
    return;
  }

#if defined(__APPLE__)
  const uint64_t now = mach_absolute_time();
#else
  const uint64_t now = 0;
#endif

  begin_write();
  switch (event) {
  case GC_EVENT_START:
    if (collection_open) {
      unmatched_event_count.fetch_add(1, std::memory_order_relaxed);
    }
    collection_open = true;
    collection_start_ticks = now;
    break;
  case GC_EVENT_END:
    if (collection_open) {
      record_collection(collection_start_ticks, now);
      collection_open = false;
    } else {
      unmatched_event_count.fetch_add(1, std::memory_order_relaxed);
    }
    if (rendezvous_open || stopped_open) {
      unmatched_event_count.fetch_add(1, std::memory_order_relaxed);
      rendezvous_open = false;
      stopped_open = false;
    }
    break;
  case GC_EVENT_PRE_STOP_WORLD:
    if (rendezvous_open || stopped_open) {
      unmatched_event_count.fetch_add(1, std::memory_order_relaxed);
    }
    rendezvous_open = true;
    rendezvous_start_ticks = now;
    stopped_open = false;
    break;
  case GC_EVENT_POST_STOP_WORLD:
    if (rendezvous_open && !stopped_open) {
      stopped_open = true;
      stopped_start_ticks = now;
    } else {
      unmatched_event_count.fetch_add(1, std::memory_order_relaxed);
    }
    break;
  case GC_EVENT_PRE_START_WORLD:
    if (rendezvous_open && stopped_open) {
      record_stopped_interval(stopped_start_ticks, now);
      stopped_open = false;
    } else {
      unmatched_event_count.fetch_add(1, std::memory_order_relaxed);
    }
    break;
  case GC_EVENT_POST_START_WORLD:
    if (rendezvous_open && !stopped_open) {
      record_rendezvous_pause(rendezvous_start_ticks, now);
      rendezvous_open = false;
    } else {
      unmatched_event_count.fetch_add(1, std::memory_order_relaxed);
      rendezvous_open = false;
      stopped_open = false;
    }
    break;
  default:
    break;
  }
  end_write();
}

int validate_runtime() {
#if !defined(__APPLE__)
  return PIXELUMM_GC_OBSERVER_UNSUPPORTED;
#else
  const unsigned expected = (GC_VERSION_MAJOR << 16) | (GC_VERSION_MINOR << 8) |
                            GC_VERSION_MICRO;
  if (GC_get_version() != expected) {
    return PIXELUMM_GC_OBSERVER_UNSUPPORTED;
  }
  mach_timebase_info_data_t timebase;
  if (mach_timebase_info(&timebase) != KERN_SUCCESS || timebase.numer == 0 ||
      timebase.denom == 0) {
    return PIXELUMM_GC_OBSERVER_CLOCK_FAILURE;
  }
  timebase_numer = timebase.numer;
  timebase_denom = timebase.denom;
  return PIXELUMM_GC_OBSERVER_OK;
#endif
}

} // namespace

extern "C" int pixelumm_gc_observer_start(void) {
  if (!try_control_lock()) {
    return PIXELUMM_GC_OBSERVER_CONTROL_BUSY;
  }
  if (active.load(std::memory_order_acquire)) {
    release_control_lock();
    return PIXELUMM_GC_OBSERVER_ALREADY_ACTIVE;
  }
  const int runtime_status = validate_runtime();
  if (runtime_status != PIXELUMM_GC_OBSERVER_OK) {
    release_control_lock();
    return runtime_status;
  }
  if (GC_get_on_collection_event() != nullptr) {
    release_control_lock();
    return PIXELUMM_GC_OBSERVER_CALLBACK_OCCUPIED;
  }

  begin_write();
  reset_counters();
  end_write();

  active.store(true, std::memory_order_release);
  GC_set_on_collection_event(collection_event);
  if (GC_get_on_collection_event() != collection_event) {
    active.store(false, std::memory_order_release);
    release_control_lock();
    return PIXELUMM_GC_OBSERVER_CALLBACK_CHANGED;
  }
  release_control_lock();
  return PIXELUMM_GC_OBSERVER_OK;
}

extern "C" int pixelumm_gc_observer_stop(void) {
  if (!try_control_lock()) {
    return PIXELUMM_GC_OBSERVER_CONTROL_BUSY;
  }
  if (!active.load(std::memory_order_acquire)) {
    release_control_lock();
    return PIXELUMM_GC_OBSERVER_NOT_ACTIVE;
  }
  if (GC_get_on_collection_event() != collection_event) {
    active.store(false, std::memory_order_release);
    release_control_lock();
    return PIXELUMM_GC_OBSERVER_CALLBACK_CHANGED;
  }

  GC_set_on_collection_event(nullptr);
  active.store(false, std::memory_order_release);
  if (collection_open || rendezvous_open || stopped_open) {
    begin_write();
    unmatched_event_count.fetch_add(1, std::memory_order_relaxed);
    collection_open = false;
    rendezvous_open = false;
    stopped_open = false;
    end_write();
  }
  release_control_lock();
  return PIXELUMM_GC_OBSERVER_OK;
}

extern "C" int pixelumm_gc_observer_snapshot(PixelUMMGcObserverSnapshot *out) {
  if (out == nullptr) {
    return PIXELUMM_GC_OBSERVER_INVALID_ARGUMENT;
  }

  for (;;) {
    const uint64_t before = write_sequence.load(std::memory_order_acquire);
    if ((before & 1U) != 0) {
      continue;
    }
    out->collection_count = collection_count.load(std::memory_order_relaxed);
    out->collection_total_ns = collection_total_ns.load(std::memory_order_relaxed);
    out->collection_max_ns = collection_max_ns.load(std::memory_order_relaxed);
    out->rendezvous_pause_count =
        rendezvous_pause_count.load(std::memory_order_relaxed);
    out->rendezvous_pause_total_ns =
        rendezvous_pause_total_ns.load(std::memory_order_relaxed);
    out->rendezvous_pause_max_ns =
        rendezvous_pause_max_ns.load(std::memory_order_relaxed);
    out->stopped_interval_count =
        stopped_interval_count.load(std::memory_order_relaxed);
    out->stopped_interval_total_ns =
        stopped_interval_total_ns.load(std::memory_order_relaxed);
    out->stopped_interval_max_ns =
        stopped_interval_max_ns.load(std::memory_order_relaxed);
    out->unmatched_event_count =
        unmatched_event_count.load(std::memory_order_relaxed);
    out->active = active.load(std::memory_order_acquire) ? 1U : 0U;
    out->callback_installed =
        GC_get_on_collection_event() == collection_event ? 1U : 0U;
    const uint64_t after = write_sequence.load(std::memory_order_acquire);
    if (before == after && (after & 1U) == 0) {
      return PIXELUMM_GC_OBSERVER_OK;
    }
  }
}
