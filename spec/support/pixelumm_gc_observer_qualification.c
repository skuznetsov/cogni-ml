#include "../../examples/support/pixelumm_gc_observer.h"

#include <gc/gc.h>

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

static void occupied_callback(GC_EventType event) { (void)event; }
static void replacement_callback(GC_EventType event) { (void)event; }

static int fail(const char *message) {
  fprintf(stderr, "pixelumm_gc_observer_qualification: %s\n", message);
  return 1;
}

int main(void) {
  GC_INIT();

  GC_set_on_collection_event(occupied_callback);
  if (pixelumm_gc_observer_start() != PIXELUMM_GC_OBSERVER_CALLBACK_OCCUPIED) {
    return fail("observer did not reject an occupied callback slot");
  }
  if (GC_get_on_collection_event() != occupied_callback) {
    return fail("occupied callback was overwritten");
  }
  GC_set_on_collection_event(NULL);

  if (pixelumm_gc_observer_start() != PIXELUMM_GC_OBSERVER_OK) {
    return fail("observer did not start with an empty callback slot");
  }
  if (pixelumm_gc_observer_start() != PIXELUMM_GC_OBSERVER_ALREADY_ACTIVE) {
    return fail("observer accepted a duplicate start");
  }
  if (pixelumm_gc_observer_snapshot(NULL) !=
      PIXELUMM_GC_OBSERVER_INVALID_ARGUMENT) {
    return fail("observer accepted a null snapshot destination");
  }

  void *roots[128];
  for (size_t i = 0; i < sizeof(roots) / sizeof(roots[0]); ++i) {
    roots[i] = GC_MALLOC(4096);
    if (roots[i] == NULL) {
      return fail("Boehm allocation failed during the CPU qualification");
    }
  }

  PixelUMMGcObserverSnapshot before;
  PixelUMMGcObserverSnapshot after;
  if (pixelumm_gc_observer_snapshot(&before) != PIXELUMM_GC_OBSERVER_OK) {
    return fail("initial snapshot failed");
  }
  if (before.active != 1 || before.callback_installed != 1) {
    return fail("active snapshot did not report the installed observer");
  }
  GC_gcollect();
  if (pixelumm_gc_observer_snapshot(&after) != PIXELUMM_GC_OBSERVER_OK) {
    return fail("snapshot after explicit collection failed");
  }
  if (after.collection_count <= before.collection_count) {
    return fail("observer did not count an explicit collection");
  }
  if (after.collection_total_ns == 0 || after.collection_max_ns == 0) {
    return fail("observer did not time the explicit collection");
  }
  if (after.rendezvous_pause_count == 0 ||
      after.stopped_interval_count == 0 || after.unmatched_event_count != 0) {
    return fail("observer did not produce matched stop-world intervals");
  }

  const uint64_t count_before_disable = after.collection_count;
  GC_disable();
  if (!GC_is_disabled()) {
    return fail("GC_disable did not disable the collector");
  }
  GC_gcollect();
  if (pixelumm_gc_observer_snapshot(&after) != PIXELUMM_GC_OBSERVER_OK) {
    return fail("snapshot during disabled collection failed");
  }
  if (after.collection_count != count_before_disable) {
    return fail("observer counted a collection suppressed by GC_disable");
  }
  GC_enable();

  if (pixelumm_gc_observer_stop() != PIXELUMM_GC_OBSERVER_OK) {
    return fail("observer did not stop cleanly");
  }
  if (GC_get_on_collection_event() != NULL) {
    return fail("stop did not restore the initially empty callback slot");
  }
  if (pixelumm_gc_observer_snapshot(&before) != PIXELUMM_GC_OBSERVER_OK ||
      before.active != 0 || before.callback_installed != 0) {
    return fail("snapshot still reports the observer active after stop");
  }

  const uint64_t stopped_count = before.collection_count;
  GC_gcollect();
  if (pixelumm_gc_observer_snapshot(&after) != PIXELUMM_GC_OBSERVER_OK ||
      after.collection_count != stopped_count) {
    return fail("observer continued counting after stop");
  }

  if (pixelumm_gc_observer_start() != PIXELUMM_GC_OBSERVER_OK) {
    return fail("observer did not restart after a clean stop");
  }
  GC_set_on_collection_event(replacement_callback);
  if (pixelumm_gc_observer_snapshot(&after) != PIXELUMM_GC_OBSERVER_OK ||
      after.active != 1 || after.callback_installed != 0) {
    return fail("snapshot did not detect a replaced callback slot");
  }
  if (pixelumm_gc_observer_stop() != PIXELUMM_GC_OBSERVER_CALLBACK_CHANGED) {
    return fail("stop did not reject a replaced callback slot");
  }
  if (GC_get_on_collection_event() != replacement_callback) {
    return fail("stop overwrote the replacement callback");
  }
  if (pixelumm_gc_observer_snapshot(&after) != PIXELUMM_GC_OBSERVER_OK ||
      after.active != 0 || after.callback_installed != 0) {
    return fail("snapshot did not report inactive state after callback loss");
  }
  GC_set_on_collection_event(NULL);

  printf("pixelumm_gc_observer_qualification: PASS collections=%" PRIu64
         " envelope_ns=%" PRIu64 " max_envelope_ns=%" PRIu64
         " rendezvous_pauses=%" PRIu64 " stopped_intervals=%" PRIu64
         " unmatched_events=%" PRIu64 "\n",
         before.collection_count, before.collection_total_ns,
         before.collection_max_ns, before.rendezvous_pause_count,
         before.stopped_interval_count, before.unmatched_event_count);
  return 0;
}
