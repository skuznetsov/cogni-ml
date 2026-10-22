#pragma once

#include <stdint.h>

// Optional per-chain API timing for the WAN MPS command-chain diagnostic.
// Each optional bridge provides a weak no-op definition, overridden by the
// strong implementation in metal_mps_command_chain.mm when that object links.
#if defined(__APPLE__) && !defined(GS_MPS_API_PROFILE_IMPLEMENTATION) &&      \
    !defined(GS_MPS_API_PROFILE_WEAK_FALLBACK_IMPLEMENTATION)
#define GS_MPS_API_PROFILE_WEAK_IMPORT __attribute__((weak_import))
#else
#define GS_MPS_API_PROFILE_WEAK_IMPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum {
  GS_MPS_API_PROFILE_PHASE_UNASSIGNED = 0,
  GS_MPS_API_PROFILE_PHASE_SELF_PREPARE = 1,
  GS_MPS_API_PROFILE_PHASE_SELF_ATTENTION = 2,
  GS_MPS_API_PROFILE_PHASE_SELF_PROJECTION_RESIDUAL = 3,
  GS_MPS_API_PROFILE_PHASE_CROSS_ATTENTION_PROJECTION_RESIDUAL = 4,
  GS_MPS_API_PROFILE_PHASE_FFN = 5,
  GS_MPS_API_PROFILE_PHASE_COUNT = 6,
  GS_MPS_API_PROFILE_PHASE_CAPACITY = 16,
};

enum {
  GS_MPS_API_PROFILE_OP_GRAPH_ENCODE = 0,
  GS_MPS_API_PROFILE_OP_COMMIT_AND_CONTINUE = 1,
  GS_MPS_API_PROFILE_OP_T5_LINEAR_ENCODE = 2,
  GS_MPS_API_PROFILE_OP_COUNT = 3,
};

enum {
  GS_MPS_CHAIN_TRACE_EVENT_CAPACITY = 4096,
  GS_MPS_CHAIN_TRACE_EVENT_NESTED_IN_GRAPH = 1u << 0,
  GS_MPS_CHAIN_TRACE_EVENT_BEFORE_ROOT_VALID = 1u << 1,
  GS_MPS_CHAIN_TRACE_EVENT_BEFORE_STATUS_AFTER_CALL_VALID = 1u << 2,
  GS_MPS_CHAIN_TRACE_EVENT_AFTER_ROOT_VALID = 1u << 3,
};

enum {
  GS_MPS_CHAIN_TRACE_COMPLETION_UNRECORDED = 0,
  GS_MPS_CHAIN_TRACE_COMPLETION_CALLBACK = 1,
  GS_MPS_CHAIN_TRACE_COMPLETION_TERMINAL_FALLBACK = 2,
};

typedef struct gs_mps_api_profile_scope {
  void *owner_data;
  uint64_t start_wall_ns;
  uint64_t start_thread_cpu_ns;
  uint64_t serial;
  uint64_t parent_graph_serial;
  uint32_t operation;
  uint32_t phase;
  uint32_t active;
  uint32_t clocks_valid;
  uint32_t nested_in_graph;
  uint32_t reserved;
} gs_mps_api_profile_scope;

// All fields are uint64_t so the Crystal FFI layout is stable and simple.
// `count` and wall/CPU totals/maxima describe this operation's measured span.
// `nested_*` describes commitAndContinue spans nested inside Graph encode;
// for commitAndContinue it is the subset of this operation called under Graph.
// Graph exclusive totals subtract those nested commit spans. Commit and linear
// exclusive totals equal their own totals.
typedef struct gs_mps_api_profile_stats {
  uint64_t count;
  uint64_t wall_total_ns;
  uint64_t wall_max_ns;
  uint64_t thread_cpu_total_ns;
  uint64_t thread_cpu_max_ns;
  uint64_t nested_count;
  uint64_t nested_wall_total_ns;
  uint64_t nested_thread_cpu_total_ns;
  uint64_t exclusive_wall_total_ns;
  uint64_t exclusive_thread_cpu_total_ns;
} gs_mps_api_profile_stats;

// Trace records are available only after the chain's existing terminal wait.
// Trace enable rejects CLOCK_MONOTONIC resolutions above 1 us; its per-call
// wall intervals therefore retain at most that clock's timestamp quantization.
// Host ticks are mach_absolute_time() ticks; convert them to seconds with
// ticks * mach_timebase_numer / mach_timebase_denom / 1e9. Metal GPU times are
// already seconds in the corresponding host clock domain. Root sequence 0 is
// the original root observed under the tracker lock at trace start; later
// sequences record actual tracker insertion times. A continuation event
// bounds the call to commitAndContinue; it does not claim an exact submission
// instant.
typedef struct gs_mps_chain_trace_info {
  uint64_t owner_thread_id;
  uint64_t mach_timebase_numer;
  uint64_t mach_timebase_denom;
  uint64_t trace_start_mach_ticks;
  uint64_t event_capacity;
  uint64_t event_count;
  uint64_t root_count;
  uint64_t failed;
} gs_mps_chain_trace_info;

typedef struct gs_mps_chain_trace_event {
  uint64_t serial;
  uint64_t begin_mach_ticks;
  uint64_t end_mach_ticks;
  uint64_t begin_thread_cpu_ns;
  uint64_t end_thread_cpu_ns;
  uint64_t api_scope_serial;
  uint64_t parent_graph_serial;
  uint64_t before_root_sequence;
  uint64_t after_root_sequence;
  uint32_t phase;
  uint32_t graph_depth;
  uint32_t before_root_status;
  uint32_t before_root_status_after_call;
  uint32_t after_root_status;
  uint32_t flags;
} gs_mps_chain_trace_event;

typedef struct gs_mps_chain_trace_root {
  uint64_t sequence;
  uint64_t tracked_or_observed_mach_ticks;
  uint64_t completion_observed_mach_ticks;
  uint64_t gpu_start_time_bits;
  uint64_t gpu_end_time_bits;
  uint32_t status_at_track_or_enable;
  uint32_t final_status;
  uint32_t has_error;
  uint32_t completion_source;
} gs_mps_chain_trace_root;

int32_t gs_mps_chain_api_profile_enable(void *chain);
int32_t gs_mps_chain_api_profile_set_phase(void *chain, int32_t phase);
// Valid only after the chain's existing terminal wait has completed.
int32_t gs_mps_chain_api_profile_snapshot(
    void *chain, int32_t phase, int32_t operation,
    gs_mps_api_profile_stats *stats);

// Explicit opt-in before encoding; snapshots require the existing terminal
// wait to have completed. The fixed event buffer is allocated by enable.
int32_t gs_mps_chain_api_trace_enable(void *chain);
int32_t gs_mps_chain_api_trace_info(void *chain,
                                    gs_mps_chain_trace_info *info);
int32_t gs_mps_chain_api_trace_event(void *chain, uint64_t index,
                                     gs_mps_chain_trace_event *event);
int32_t gs_mps_chain_api_trace_root(void *chain, uint64_t index,
                                    gs_mps_chain_trace_root *root);

// CPU-only controls for timestamp, nesting, ordering, and overflow behavior.
int32_t gs_mps_chain_api_trace_self_test(uint64_t *positive_events,
                                         uint64_t *rejected_controls);

void gs_mps_api_profile_scope_begin(
    int32_t operation, void *command_context,
    gs_mps_api_profile_scope *scope)
    GS_MPS_API_PROFILE_WEAK_IMPORT;
void gs_mps_api_profile_scope_end(gs_mps_api_profile_scope *scope)
    GS_MPS_API_PROFILE_WEAK_IMPORT;

// CPU-only native discriminator; it performs no Metal work or waits.
int32_t gs_mps_api_profile_cpu_probe(uint64_t *wall_ns,
                                     uint64_t *thread_cpu_ns);

#ifdef __cplusplus
}
#endif

#undef GS_MPS_API_PROFILE_WEAK_IMPORT
