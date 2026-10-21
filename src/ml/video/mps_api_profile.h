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

int32_t gs_mps_chain_api_profile_enable(void *chain);
int32_t gs_mps_chain_api_profile_set_phase(void *chain, int32_t phase);
// Valid only after the chain's existing terminal wait has completed.
int32_t gs_mps_chain_api_profile_snapshot(
    void *chain, int32_t phase, int32_t operation,
    gs_mps_api_profile_stats *stats);

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
