#pragma once

#include "mps_api_profile.h"

// Optional MPSGraph API timing around one executable encode call. The weak
// imports keep bridges that omit the command-chain profiler linkable, and the
// profiler returns immediately when no per-chain API profile is enabled.
class DCAEVGraphAPIScope {
public:
  explicit DCAEVGraphAPIScope(void *commandContext) {
    if (gs_mps_api_profile_scope_begin && gs_mps_api_profile_scope_end)
      gs_mps_api_profile_scope_begin(GS_MPS_API_PROFILE_OP_GRAPH_ENCODE,
                                     commandContext, &scope_);
  }

  ~DCAEVGraphAPIScope() {
    if (scope_.active && gs_mps_api_profile_scope_end)
      gs_mps_api_profile_scope_end(&scope_);
  }

  DCAEVGraphAPIScope(const DCAEVGraphAPIScope &) = delete;
  DCAEVGraphAPIScope &operator=(const DCAEVGraphAPIScope &) = delete;
  DCAEVGraphAPIScope(DCAEVGraphAPIScope &&) = delete;
  DCAEVGraphAPIScope &operator=(DCAEVGraphAPIScope &&) = delete;

private:
  gs_mps_api_profile_scope scope_{};
};
