CRYSTAL ?= crystal
BUILD_DIR ?= build
LLAMA_DIR ?= ../llama.cpp
LLAMA_BUILD ?= $(LLAMA_DIR)/build
BUILD_SENTINEL := $(BUILD_DIR)/.dir
ARGS ?=
UNAME_S := $(shell uname -s)
IS_DARWIN := $(filter Darwin,$(UNAME_S))
BREW_PREFIX := $(shell if command -v brew >/dev/null 2>&1; then brew --prefix; fi)
LLAMA_LIB_DIR ?= $(firstword \
	$(dir $(wildcard $(LLAMA_BUILD)/bin/libllama.*)) \
	$(dir $(wildcard $(LLAMA_BUILD)/src/libllama.*)) \
	$(dir $(wildcard $(LLAMA_BUILD)/lib/libllama.*)) \
	$(dir $(wildcard $(LLAMA_DIR)/build/bin/libllama.*)) \
	$(dir $(wildcard $(LLAMA_DIR)/build/src/libllama.*)) \
	$(dir $(wildcard $(LLAMA_DIR)/build/lib/libllama.*)) \
	$(if $(BREW_PREFIX),$(dir $(wildcard $(BREW_PREFIX)/lib/libllama.*)),) \
	$(dir $(wildcard /opt/homebrew/lib/libllama.*)) \
	$(dir $(wildcard /usr/local/lib/libllama.*)) \
	$(dir $(wildcard /usr/lib/libllama.*)) \
	$(dir $(wildcard /lib/libllama.*)) \
)

OBJC_SOURCES := src/ml/metal/bridge.mm
BRIDGE_OBJ := $(BUILD_DIR)/bridge.o
WAN_MPS_CHAIN_OBJ := $(BUILD_DIR)/metal_mps_command_chain.o
WAN_STAGE_IO_OBJ := $(BUILD_DIR)/metal_stage_io.o
WAN_T5_MPS_OBJ := $(BUILD_DIR)/wan_t5_mps_bridge.o
WAN_T5_RMS_MPS_OBJ := $(BUILD_DIR)/wan_t5_rms_mps_bridge.o
WAN_T5_ATTENTION_MPS_OBJ := $(BUILD_DIR)/wan_t5_attention_mps_bridge.o
WAN_ATTENTION_MPS_OBJ := $(BUILD_DIR)/wan_attention_mps_bridge.o
HY_NOMAS_MPS_OBJ := $(BUILD_DIR)/hy_nomas_mps_reductions.o
HY_NOMAS_BF16_LINEAR_OBJ := $(BUILD_DIR)/hy_nomas_mps_bf16_linear.o
HY_NOMAS_COMPACT_ATTENTION_OBJ := $(BUILD_DIR)/hy_nomas_attention_compact.o
HY_NOMAS_HISTORY_ATTENTION_OBJ := $(BUILD_DIR)/hy_nomas_history_attention.o
QWEN25_MPS_OBJ := $(BUILD_DIR)/qwen25_mps_bridge.o
PIXELUMM_MPS_OBJ := $(BUILD_DIR)/pixelumm_decoder_block_mps_bridge.o
PIXELUMM_HEAD_OBJ := $(BUILD_DIR)/pixelumm_decoder_head_mps_bridge.o
PIXELUMM_FFN_PROBE_OBJ := $(BUILD_DIR)/pixelumm_ffn_projection_bridge.o
PIXELUMM_FFN_ISOLATED_PROBE_OBJ := $(BUILD_DIR)/pixelumm_ffn_isolated_bridge.o
PIXELUMM_FFN_AA_OBJ := $(BUILD_DIR)/pixelumm_ffn_aa_bridge.o
PIXELUMM_FFN_GRAPH_OBJ := $(BUILD_DIR)/pixelumm_ffn_graph_bridge.o
PIXELUMM_CONNECTED_RESOURCE_PROBE_OBJ := $(BUILD_DIR)/pixelumm_connected_resource_bridge.o
PIXELUMM_FFN_WEIGHT_REUSE_OBJ := $(BUILD_DIR)/pixelumm_ffn_weight_reuse_bridge.o
PIXELUMM_PACKED_GATE_UP_OBJ := $(BUILD_DIR)/pixelumm_packed_gate_up_bridge.o
PIXELUMM_FFN_ORIENTATION_OBJ := $(BUILD_DIR)/pixelumm_ffn_orientation_bridge.o
PIXELUMM_QUERY_PAIR_OBJ := $(BUILD_DIR)/pixelumm_query_pair_bridge.o
PIXELUMM_WINDOW_BATCH_OBJ := $(BUILD_DIR)/pixelumm_window_batch_bridge.o
PIXELUMM_FFN_DIRECT_MPS_OBJ := $(BUILD_DIR)/pixelumm_ffn_direct_mps_bridge.o
PIXELUMM_GC_OBSERVER_OBJ := $(BUILD_DIR)/pixelumm_gc_observer.o
PIXELUMM_RECOVERY_HOST_OBJ := $(BUILD_DIR)/pixelumm_recovery_host.o
COGNIGRAPH_NATIVE_HEADERS := src/ml/metal/cognigraph_native.h src/ml/metal/cognigraph_schedule.h
PIXELUMM_FFN_ISOLATED_SOURCES := $(wildcard src/ml/core/*.cr src/ml/metal/*.cr src/ml/video/*.cr)
QWEN25_PREFIX_SOURCES := $(wildcard src/ml/gguf/*.cr src/ml/core/*.cr src/ml/metal/*.cr src/ml/video/*.cr src/ml/video/kernels/*.metal)
WAN_MPS_OBJS := $(WAN_T5_MPS_OBJ) $(WAN_T5_RMS_MPS_OBJ) $(WAN_T5_ATTENTION_MPS_OBJ) $(WAN_ATTENTION_MPS_OBJ) $(WAN_MPS_CHAIN_OBJ) $(WAN_STAGE_IO_OBJ)
WAN_SPEC_OBJS := $(if $(IS_DARWIN),$(WAN_MPS_OBJS) $(HY_NOMAS_MPS_OBJ) $(HY_NOMAS_COMPACT_ATTENTION_OBJ) $(QWEN25_MPS_OBJ),)
WAN_SPEC_LINK_FLAGS := $(if $(IS_DARWIN),-framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -fuse-ld=/usr/bin/ld,)
WAN_BENCHMARK_SOURCES := $(wildcard src/ml/video/*.cr src/ml/metal/*.cr src/ml/gguf/*.cr src/ml/*.cr) spec/support/video_allocation_counter.cr

LINK_FLAGS := -framework Metal -framework Foundation -lc++

.PHONY: all spec_video_staging spec_wan spec_hy_nomas spec_hy_nomas_residency spec_hy_nomas_bf16_cache spec_qwen25 hy_nomas_block_benchmark hy_nomas_serial_benchmark hy_nomas_pretrained_serial_probe hy_nomas_bf16_cache_probe hy_nomas_bf16_serial_probe hy_nomas_residency_plan spec build spec_cpu build_cpu drama3_studio llama llama_env profile_nomic profile_nomic_layers profile_nomic_vs_llama benchmark_qwen_prefill_vs_llama_same_token benchmark_qwen_decode_vs_llama_same_token q4k_ref clean help

all: spec

# Each of these translation units includes the PixelUMM block bridge.
$(PIXELUMM_MPS_OBJ) $(PIXELUMM_HEAD_OBJ) $(PIXELUMM_FFN_PROBE_OBJ) $(PIXELUMM_FFN_ISOLATED_PROBE_OBJ) $(PIXELUMM_CONNECTED_RESOURCE_PROBE_OBJ) $(PIXELUMM_FFN_WEIGHT_REUSE_OBJ) $(PIXELUMM_PACKED_GATE_UP_OBJ) $(PIXELUMM_FFN_ORIENTATION_OBJ) $(PIXELUMM_QUERY_PAIR_OBJ) $(PIXELUMM_FFN_DIRECT_MPS_OBJ): $(COGNIGRAPH_NATIVE_HEADERS)

$(BUILD_SENTINEL):
	@mkdir -p $(BUILD_DIR)
	@touch $(BUILD_SENTINEL)

$(BRIDGE_OBJ): $(OBJC_SOURCES) | $(BUILD_SENTINEL)
	clang++ -c $(OBJC_SOURCES) -o $(BRIDGE_OBJ) \
		-std=c++17 -fobjc-arc -fPIC


$(WAN_T5_MPS_OBJ): src/ml/video/wan_t5_mps_bridge.mm | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "Wan MPS requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

$(WAN_T5_RMS_MPS_OBJ): src/ml/video/wan_t5_rms_mps_bridge.mm | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "Wan MPS requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

$(WAN_T5_ATTENTION_MPS_OBJ): src/ml/video/wan_t5_attention_mps_bridge.mm | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "Wan MPS requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

$(WAN_ATTENTION_MPS_OBJ): src/ml/video/wan_attention_mps_bridge.mm | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "Wan MPS requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

$(HY_NOMAS_MPS_OBJ): src/ml/video/hy_nomas_mps_reductions_bridge.mm | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "Hy-Nomas MPS requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

$(HY_NOMAS_BF16_LINEAR_OBJ): src/ml/video/hy_nomas_mps_bf16_linear_bridge.mm | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "Hy-Nomas BF16 linear MPSGraph requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

$(HY_NOMAS_COMPACT_ATTENTION_OBJ): src/ml/video/hy_nomas_attention_compact_bridge.mm | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "Hy-Nomas compact MPS attention requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

$(HY_NOMAS_HISTORY_ATTENTION_OBJ): src/ml/video/hy_nomas_history_attention_bridge.mm | $(BUILD_SENTINEL)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

$(QWEN25_MPS_OBJ): src/ml/gguf/qwen25_mps_bridge.mm | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "Qwen2.5-VL MPS requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

$(PIXELUMM_MPS_OBJ): src/ml/video/pixelumm_decoder_block_mps_bridge.mm src/ml/video/mps_api_profile.h | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "PixelUMM MPS requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

.PHONY: pixelumm_decoder_head_probe spec_pixelumm_decoder_head
pixelumm_decoder_head_probe: $(BUILD_DIR)/pixelumm_decoder_head_probe

$(PIXELUMM_HEAD_OBJ): src/ml/video/pixelumm_decoder_head_mps_bridge.mm src/ml/video/pixelumm_decoder_block_mps_bridge.mm src/ml/video/mps_api_profile.h | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "PixelUMM head MPS requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

# The head translation unit reuses the block graph helpers and includes its ABI.
# Do not also link PIXELUMM_MPS_OBJ into this diagnostic executable.
$(BUILD_DIR)/pixelumm_decoder_head_probe: examples/pixelumm_decoder_head_probe.cr examples/support/pixelumm_head_probe.cr examples/support/pixelumm_block_probe.cr examples/support/pixelumm_memory_probe.cr examples/support/hy_nomas_probe_instrumentation.cr src/ml/video/pixelumm_decoder_head_mps.cr src/ml/video/pixelumm_decoder_block_mps.cr src/ml/video/metal_mps_command_chain.cr $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_HEAD_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_HEAD_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

spec_pixelumm_decoder_head:
	$(CRYSTAL) spec -Dcpu_only spec/pixelumm_decoder_head_admission_spec.cr spec/pixelumm_head_probe_spec.cr $(ARGS)

.PHONY: pixelumm_decoder_integrated_probe spec_pixelumm_decoder_integrated
pixelumm_decoder_integrated_probe: $(BUILD_DIR)/pixelumm_decoder_integrated_probe

# The head object includes the block ABI; do not link PIXELUMM_MPS_OBJ twice.
$(BUILD_DIR)/pixelumm_decoder_integrated_probe: examples/pixelumm_decoder_integrated_probe.cr examples/support/pixelumm_decoder_probe.cr examples/support/pixelumm_cpu_diagnostic_probe.cr examples/support/pixelumm_tiled_gate_up_probe.cr examples/support/pixelumm_two_block_probe.cr examples/support/pixelumm_head_probe.cr examples/support/pixelumm_block_probe.cr examples/support/pixelumm_memory_probe.cr examples/support/hy_nomas_probe_instrumentation.cr src/ml/video/pixelumm_decoder_head_mps.cr src/ml/video/pixelumm_decoder_block_mps.cr src/ml/video/metal_mps_command_chain.cr src/ml/video/pixelumm_decoder_profile.cr src/ml/video/pixelumm_block_profile.cr src/ml/video/mps_api_profile.cr src/ml/video/mps_command_phase_profile.cr src/ml/video/mps_continuation_trace.cr $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_HEAD_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_HEAD_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

spec_pixelumm_decoder_integrated:
	$(CRYSTAL) spec -Dcpu_only spec/pixelumm_cpu_diagnostic_probe_spec.cr spec/pixelumm_tiled_gate_up_probe_spec.cr spec/pixelumm_decoder_probe_spec.cr spec/pixelumm_decoder_head_admission_spec.cr spec/pixelumm_head_probe_spec.cr spec/pixelumm_two_block_probe_spec.cr spec/pixelumm_decoder_profile_spec.cr $(ARGS)

.PHONY: pixelumm_decoder_block_probe pixelumm_decoder_two_block_probe pixelumm_decoder_four_block_probe pixelumm_decoder_eight_block_probe spec_pixelumm_decoder_block spec_pixelumm_two_block_probe spec_pixelumm_block_profile spec_pixelumm_memory_probe
pixelumm_decoder_block_probe: $(BUILD_DIR)/pixelumm_decoder_block_probe

.PHONY: pixelumm_ffn_projection_probe
pixelumm_ffn_projection_probe: $(BUILD_DIR)/pixelumm_ffn_projection_probe

.PHONY: pixelumm_ffn_f32_kernel_probe spec_pixelumm_ffn_f32_kernel_probe
pixelumm_ffn_f32_kernel_probe: $(BUILD_DIR)/pixelumm_ffn_f32_kernel_probe

spec_pixelumm_ffn_f32_kernel_probe:
	$(CRYSTAL) spec -Dcpu_only spec/pixelumm_ffn_f32_kernel_probe_spec.cr

.PHONY: spec_pixelumm_ffn_projection
spec_pixelumm_ffn_projection:
	$(CRYSTAL) spec -Dcpu_only spec/pixelumm_ffn_projection_spec.cr

.PHONY: pixelumm_ffn_isolated_probe spec_pixelumm_ffn_isolated_probe
pixelumm_ffn_isolated_probe: $(BUILD_DIR)/pixelumm_ffn_isolated_probe

spec_pixelumm_ffn_isolated_probe:
	$(CRYSTAL) spec -Dcpu_only spec/pixelumm_ffn_isolated_probe_spec.cr

$(PIXELUMM_FFN_PROBE_OBJ): examples/support/pixelumm_ffn_projection_bridge.mm src/ml/video/pixelumm_decoder_block_mps_bridge.mm src/ml/video/mps_api_profile.h | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "PixelUMM FFN probe requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

$(PIXELUMM_FFN_ISOLATED_PROBE_OBJ): examples/support/pixelumm_ffn_isolated_bridge.mm src/ml/video/pixelumm_decoder_block_mps_bridge.mm src/ml/video/mps_api_profile.h | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "PixelUMM isolated FFN probe requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

.PHONY: pixelumm_connected_resource_probe spec_pixelumm_connected_resource_probe
pixelumm_connected_resource_probe: $(BUILD_DIR)/pixelumm_connected_resource_probe

spec_pixelumm_connected_resource_probe:
	$(CRYSTAL) spec -Dcpu_only spec/pixelumm_connected_resource_probe_spec.cr

.PHONY: pixelumm_plan_reuse_probe spec_pixelumm_plan_reuse_probe
pixelumm_plan_reuse_probe: $(BUILD_DIR)/pixelumm_plan_reuse_probe

spec_pixelumm_plan_reuse_probe:
	$(CRYSTAL) spec -Dcpu_only spec/pixelumm_plan_reuse_probe_spec.cr

.PHONY: pixelumm_ffn_weight_reuse_probe spec_pixelumm_ffn_weight_reuse_probe
pixelumm_ffn_weight_reuse_probe: $(BUILD_DIR)/pixelumm_ffn_weight_reuse_probe

spec_pixelumm_ffn_weight_reuse_probe:
	$(CRYSTAL) spec -Dcpu_only spec/pixelumm_ffn_weight_reuse_probe_spec.cr $(ARGS)

$(PIXELUMM_FFN_WEIGHT_REUSE_OBJ): examples/support/pixelumm_ffn_weight_reuse_bridge.mm src/ml/video/pixelumm_decoder_block_mps_bridge.mm src/ml/video/mps_api_profile.h | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "PixelUMM FFN weight-reuse probe requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

$(PIXELUMM_PACKED_GATE_UP_OBJ): examples/support/pixelumm_packed_gate_up_bridge.mm examples/support/pixelumm_ffn_weight_reuse_bridge.mm src/ml/video/pixelumm_decoder_block_mps_bridge.mm src/ml/video/mps_api_profile.h | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "PixelUMM packed gate/up probe requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

$(PIXELUMM_FFN_ORIENTATION_OBJ): examples/support/pixelumm_ffn_orientation_bridge.mm examples/support/pixelumm_ffn_weight_reuse_bridge.mm src/ml/video/pixelumm_decoder_block_mps_bridge.mm src/ml/video/mps_api_profile.h | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "PixelUMM FFN orientation probe requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

$(PIXELUMM_QUERY_PAIR_OBJ): examples/support/pixelumm_query_pair_bridge.mm examples/support/pixelumm_ffn_weight_reuse_bridge.mm src/ml/video/pixelumm_decoder_block_mps_bridge.mm src/ml/video/mps_api_profile.h | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "PixelUMM query pair probe requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

.PHONY: pixelumm_window_batch_probe spec_pixelumm_window_batch_probe pixelumm_window_batch_full_probe spec_pixelumm_window_batch_full_probe
pixelumm_window_batch_probe: $(BUILD_DIR)/pixelumm_window_batch_probe
pixelumm_window_batch_full_probe: $(BUILD_DIR)/pixelumm_window_batch_full_probe

$(PIXELUMM_WINDOW_BATCH_OBJ): examples/support/pixelumm_window_batch_bridge.mm src/ml/video/pixelumm_decoder_head_mps_bridge.mm src/ml/video/pixelumm_decoder_block_mps_bridge.mm src/ml/video/mps_api_profile.h $(COGNIGRAPH_NATIVE_HEADERS) | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "PixelUMM window batch probe requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

# The diagnostic includes both decoder ABIs; link neither block nor head twice.
$(BUILD_DIR)/pixelumm_window_batch_probe: examples/pixelumm_window_batch_probe.cr examples/support/pixelumm_window_batch_probe.cr examples/support/pixelumm_block_probe.cr examples/support/pixelumm_memory_probe.cr examples/support/hy_nomas_probe_instrumentation.cr $(PIXELUMM_FFN_ISOLATED_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_WINDOW_BATCH_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_WINDOW_BATCH_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

spec_pixelumm_window_batch_probe:
	$(CRYSTAL) spec -Dcpu_only spec/pixelumm_window_batch_probe_spec.cr $(ARGS)

# Full36 B2 diagnostic composes the existing decoder/head session with one
# shared B2 candidate panel and per-plan metrics; link the combined ABI once.
$(BUILD_DIR)/pixelumm_window_batch_full_probe: examples/pixelumm_window_batch_full_probe.cr examples/support/pixelumm_window_batch_full_probe.cr examples/support/pixelumm_window_batch_probe.cr examples/support/pixelumm_decoder_probe.cr examples/support/pixelumm_two_block_probe.cr examples/support/pixelumm_head_probe.cr examples/support/pixelumm_block_probe.cr examples/support/pixelumm_memory_probe.cr examples/support/hy_nomas_probe_instrumentation.cr $(PIXELUMM_FFN_ISOLATED_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_WINDOW_BATCH_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_WINDOW_BATCH_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

spec_pixelumm_window_batch_full_probe:
	$(CRYSTAL) spec -Dcpu_only spec/pixelumm_window_batch_full_probe_spec.cr $(ARGS)

$(PIXELUMM_FFN_DIRECT_MPS_OBJ): examples/support/pixelumm_ffn_direct_mps_bridge.mm examples/support/pixelumm_ffn_weight_reuse_bridge.mm src/ml/video/pixelumm_decoder_block_mps_bridge.mm src/ml/video/mps_api_profile.h | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "PixelUMM direct MPS FFN probe requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

# This diagnostic translation unit includes the production block bridge;
# linking PIXELUMM_MPS_OBJ would duplicate its C ABI symbols.
$(BUILD_DIR)/pixelumm_ffn_weight_reuse_probe: examples/pixelumm_ffn_weight_reuse_probe.cr examples/support/pixelumm_ffn_weight_reuse_probe.cr examples/support/pixelumm_ffn_weight_reuse_bridge.mm examples/support/pixelumm_block_probe.cr examples/support/pixelumm_two_block_probe.cr examples/support/pixelumm_connected_resource_probe.cr examples/support/pixelumm_memory_probe.cr examples/support/hy_nomas_probe_instrumentation.cr examples/support/pixelumm_ffn_dense.metal src/ml/video/pixelumm_decoder_block_mps.cr src/ml/video/metal_mps_command_chain.cr src/ml/video/pixelumm_block_profile.cr src/ml/video/mps_api_profile.cr src/ml/video/mps_command_phase_profile.cr src/ml/video/mps_continuation_trace.cr $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_FFN_WEIGHT_REUSE_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_FFN_WEIGHT_REUSE_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

.PHONY: pixelumm_gc_probe spec_pixelumm_gc_probe
pixelumm_gc_probe: $(BUILD_DIR)/pixelumm_gc_probe

$(PIXELUMM_GC_OBSERVER_OBJ): examples/support/pixelumm_gc_observer.cc examples/support/pixelumm_gc_observer.h | $(BUILD_SENTINEL)
	clang++ -c $< -o $@ -std=c++17 -O2 -Wall -Wextra -Werror -I$(BREW_PREFIX)/include

$(BUILD_DIR)/pixelumm_gc_observer_qualification: spec/support/pixelumm_gc_observer_qualification.c $(PIXELUMM_GC_OBSERVER_OBJ)
	clang -c $< -o $(BUILD_DIR)/pixelumm_gc_observer_qualification.o -std=c11 -Wall -Wextra -Werror -I$(BREW_PREFIX)/include
	clang++ $(BUILD_DIR)/pixelumm_gc_observer_qualification.o $(PIXELUMM_GC_OBSERVER_OBJ) -L$(BREW_PREFIX)/lib -lgc -o $@

spec_pixelumm_gc_probe: $(PIXELUMM_GC_OBSERVER_OBJ) $(BUILD_DIR)/pixelumm_gc_observer_qualification
	$(BUILD_DIR)/pixelumm_gc_observer_qualification
	$(CRYSTAL) spec -Dcpu_only spec/pixelumm_gc_probe_spec.cr spec/pixelumm_ffn_weight_reuse_probe_spec.cr --link-flags="$(abspath $(PIXELUMM_GC_OBSERVER_OBJ))" $(ARGS)

$(BUILD_DIR)/pixelumm_gc_probe: examples/pixelumm_gc_probe.cr examples/support/pixelumm_gc_probe.cr examples/support/pixelumm_ffn_weight_reuse_probe.cr examples/support/pixelumm_block_probe.cr examples/support/pixelumm_two_block_probe.cr examples/support/pixelumm_connected_resource_probe.cr examples/support/pixelumm_memory_probe.cr examples/support/hy_nomas_probe_instrumentation.cr examples/support/pixelumm_ffn_dense.metal $(PIXELUMM_FFN_ISOLATED_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_FFN_WEIGHT_REUSE_OBJ) $(PIXELUMM_GC_OBSERVER_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_FFN_WEIGHT_REUSE_OBJ)) $(abspath $(PIXELUMM_GC_OBSERVER_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

.PHONY: pixelumm_drift_probe spec_pixelumm_drift_probe
pixelumm_drift_probe: $(BUILD_DIR)/pixelumm_drift_probe

spec_pixelumm_drift_probe:
	$(CRYSTAL) spec -Dcpu_only spec/pixelumm_drift_profile_spec.cr spec/pixelumm_drift_probe_spec.cr

$(BUILD_DIR)/pixelumm_drift_probe: examples/pixelumm_drift_probe.cr examples/support/pixelumm_drift_probe.cr examples/support/pixelumm_drift_profile.cr examples/support/pixelumm_gc_probe.cr examples/support/pixelumm_ffn_weight_reuse_probe.cr examples/support/pixelumm_ffn_weight_reuse_bridge.mm examples/support/pixelumm_block_probe.cr examples/support/pixelumm_two_block_probe.cr examples/support/pixelumm_connected_resource_probe.cr examples/support/pixelumm_memory_probe.cr examples/support/hy_nomas_probe_instrumentation.cr examples/support/pixelumm_ffn_dense.metal $(PIXELUMM_FFN_ISOLATED_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_FFN_WEIGHT_REUSE_OBJ) $(PIXELUMM_GC_OBSERVER_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_FFN_WEIGHT_REUSE_OBJ)) $(abspath $(PIXELUMM_GC_OBSERVER_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

.PHONY: pixelumm_cognigraph_probe spec_pixelumm_cognigraph_probe
pixelumm_cognigraph_probe: $(BUILD_DIR)/pixelumm_cognigraph_probe

spec_pixelumm_cognigraph_probe:
	$(CRYSTAL) spec -Dcpu_only spec/compute_graph_schedule_spec.cr spec/pixelumm_cognigraph_probe_spec.cr

# The diagnostic bridge includes the production block ABI exactly once.
$(BUILD_DIR)/pixelumm_cognigraph_probe: examples/pixelumm_cognigraph_probe.cr examples/support/pixelumm_cognigraph_probe.cr examples/support/pixelumm_ffn_weight_reuse_probe.cr examples/support/pixelumm_ffn_weight_reuse_bridge.mm examples/support/pixelumm_block_probe.cr examples/support/pixelumm_two_block_probe.cr examples/support/pixelumm_connected_resource_probe.cr examples/support/pixelumm_memory_probe.cr examples/support/hy_nomas_probe_instrumentation.cr examples/support/pixelumm_ffn_dense.metal $(PIXELUMM_FFN_ISOLATED_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_FFN_WEIGHT_REUSE_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_FFN_WEIGHT_REUSE_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

.PHONY: pixelumm_ffn_orientation_probe pixelumm_fused_kv_gather_probe
pixelumm_ffn_orientation_probe: $(BUILD_DIR)/pixelumm_ffn_orientation_probe

pixelumm_fused_kv_gather_probe: $(BUILD_DIR)/pixelumm_fused_kv_gather_probe

.PHONY: pixelumm_packed_gate_up_probe
pixelumm_packed_gate_up_probe: $(BUILD_DIR)/pixelumm_packed_gate_up_probe

.PHONY: pixelumm_output_only_pv_probe
pixelumm_output_only_pv_probe: $(BUILD_DIR)/pixelumm_output_only_pv_probe

.PHONY: pixelumm_scale_softmax_probe
pixelumm_scale_softmax_probe: $(BUILD_DIR)/pixelumm_scale_softmax_probe

# The ordinary diagnostic bridge includes the production block ABI once.
$(BUILD_DIR)/pixelumm_scale_softmax_probe: examples/pixelumm_scale_softmax_probe.cr examples/support/pixelumm_cognigraph_probe.cr examples/support/pixelumm_ffn_weight_reuse_probe.cr examples/support/pixelumm_ffn_weight_reuse_bridge.mm examples/support/pixelumm_block_probe.cr examples/support/pixelumm_two_block_probe.cr examples/support/pixelumm_connected_resource_probe.cr examples/support/pixelumm_memory_probe.cr examples/support/hy_nomas_probe_instrumentation.cr examples/support/pixelumm_ffn_dense.metal $(PIXELUMM_FFN_ISOLATED_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_FFN_WEIGHT_REUSE_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_FFN_WEIGHT_REUSE_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

# Output-only softmax/PV uses the ordinary diagnostic bridge.
$(BUILD_DIR)/pixelumm_output_only_pv_probe: examples/pixelumm_output_only_pv_probe.cr examples/support/pixelumm_cognigraph_probe.cr examples/support/pixelumm_ffn_weight_reuse_probe.cr examples/support/pixelumm_ffn_weight_reuse_bridge.mm examples/support/pixelumm_block_probe.cr examples/support/pixelumm_two_block_probe.cr examples/support/pixelumm_connected_resource_probe.cr examples/support/pixelumm_memory_probe.cr examples/support/hy_nomas_probe_instrumentation.cr examples/support/pixelumm_ffn_dense.metal $(PIXELUMM_FFN_ISOLATED_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_FFN_WEIGHT_REUSE_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_FFN_WEIGHT_REUSE_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

# This diagnostic translation unit includes the weight-reuse bridge, which
# includes the production block ABI. Do not also link either included object.
$(BUILD_DIR)/pixelumm_packed_gate_up_probe: examples/pixelumm_packed_gate_up_probe.cr examples/support/pixelumm_cognigraph_probe.cr examples/support/pixelumm_ffn_weight_reuse_probe.cr examples/support/pixelumm_ffn_weight_reuse_bridge.mm examples/support/pixelumm_block_probe.cr examples/support/pixelumm_two_block_probe.cr examples/support/pixelumm_connected_resource_probe.cr examples/support/pixelumm_memory_probe.cr examples/support/hy_nomas_probe_instrumentation.cr examples/support/pixelumm_ffn_dense.metal src/ml/video/pixelumm_decoder_block_mps.cr src/ml/video/metal_mps_command_chain.cr src/ml/video/pixelumm_block_profile.cr src/ml/video/mps_api_profile.cr src/ml/video/mps_command_phase_profile.cr src/ml/video/mps_continuation_trace.cr $(PIXELUMM_FFN_ISOLATED_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_PACKED_GATE_UP_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_PACKED_GATE_UP_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

# The orientation wrapper includes the ordinary diagnostic and production ABIs.
$(BUILD_DIR)/pixelumm_ffn_orientation_probe: examples/pixelumm_ffn_orientation_probe.cr examples/support/pixelumm_cognigraph_probe.cr examples/support/pixelumm_ffn_weight_reuse_probe.cr examples/support/pixelumm_ffn_weight_reuse_bridge.mm examples/support/pixelumm_block_probe.cr examples/support/pixelumm_two_block_probe.cr examples/support/pixelumm_connected_resource_probe.cr examples/support/pixelumm_memory_probe.cr examples/support/hy_nomas_probe_instrumentation.cr examples/support/pixelumm_ffn_dense.metal src/ml/video/pixelumm_decoder_block_mps.cr src/ml/video/metal_mps_command_chain.cr src/ml/video/pixelumm_block_profile.cr src/ml/video/mps_api_profile.cr src/ml/video/mps_command_phase_profile.cr src/ml/video/mps_continuation_trace.cr $(PIXELUMM_FFN_ISOLATED_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_FFN_ORIENTATION_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_FFN_ORIENTATION_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

.PHONY: pixelumm_query_pair_probe
pixelumm_query_pair_probe: $(BUILD_DIR)/pixelumm_query_pair_probe

.PHONY: pixelumm_ffn_direct_mps_probe
pixelumm_ffn_direct_mps_probe: $(BUILD_DIR)/pixelumm_ffn_direct_mps_probe

# This wrapper includes the ordinary diagnostic and production ABIs once.
$(BUILD_DIR)/pixelumm_ffn_direct_mps_probe: examples/pixelumm_ffn_direct_mps_probe.cr examples/support/pixelumm_cognigraph_probe.cr examples/support/pixelumm_ffn_weight_reuse_probe.cr examples/support/pixelumm_ffn_weight_reuse_bridge.mm examples/support/pixelumm_block_probe.cr examples/support/pixelumm_two_block_probe.cr examples/support/pixelumm_connected_resource_probe.cr examples/support/pixelumm_memory_probe.cr examples/support/hy_nomas_probe_instrumentation.cr examples/support/pixelumm_ffn_dense.metal src/ml/video/pixelumm_decoder_block_mps.cr src/ml/video/metal_mps_command_chain.cr src/ml/video/pixelumm_block_profile.cr src/ml/video/mps_api_profile.cr src/ml/video/mps_command_phase_profile.cr src/ml/video/mps_continuation_trace.cr $(PIXELUMM_FFN_ISOLATED_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_FFN_DIRECT_MPS_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_FFN_DIRECT_MPS_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

# The query-pair wrapper includes the ordinary diagnostic and production ABIs.
$(BUILD_DIR)/pixelumm_query_pair_probe: examples/pixelumm_query_pair_probe.cr examples/support/pixelumm_cognigraph_probe.cr examples/support/pixelumm_ffn_weight_reuse_probe.cr examples/support/pixelumm_ffn_weight_reuse_bridge.mm examples/support/pixelumm_block_probe.cr examples/support/pixelumm_two_block_probe.cr examples/support/pixelumm_connected_resource_probe.cr examples/support/pixelumm_memory_probe.cr examples/support/hy_nomas_probe_instrumentation.cr examples/support/pixelumm_ffn_dense.metal src/ml/video/pixelumm_decoder_block_mps.cr src/ml/video/metal_mps_command_chain.cr src/ml/video/pixelumm_block_profile.cr src/ml/video/mps_api_profile.cr src/ml/video/mps_command_phase_profile.cr src/ml/video/mps_continuation_trace.cr $(PIXELUMM_FFN_ISOLATED_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_QUERY_PAIR_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_QUERY_PAIR_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

# Uses the ordinary FFN-owner diagnostic object, which includes the production
# PixelUMM ABI and the opt-in dual-output K/V gather selector exactly once.
$(BUILD_DIR)/pixelumm_fused_kv_gather_probe: examples/pixelumm_fused_kv_gather_probe.cr examples/support/pixelumm_cognigraph_probe.cr examples/support/pixelumm_ffn_weight_reuse_probe.cr examples/support/pixelumm_ffn_weight_reuse_bridge.mm examples/support/pixelumm_block_probe.cr examples/support/pixelumm_two_block_probe.cr examples/support/pixelumm_connected_resource_probe.cr examples/support/pixelumm_memory_probe.cr examples/support/hy_nomas_probe_instrumentation.cr examples/support/pixelumm_ffn_dense.metal src/ml/video/pixelumm_decoder_block_mps.cr src/ml/video/metal_mps_command_chain.cr src/ml/video/pixelumm_block_profile.cr src/ml/video/mps_api_profile.cr src/ml/video/mps_command_phase_profile.cr src/ml/video/mps_continuation_trace.cr $(PIXELUMM_FFN_ISOLATED_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_FFN_WEIGHT_REUSE_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_FFN_WEIGHT_REUSE_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

# Independent native Plans bind one diagnostic-owned Workspace. The normal
# production bridge has no per-Graph resource collector in this experiment.
$(BUILD_DIR)/pixelumm_plan_reuse_probe: examples/pixelumm_plan_reuse_probe.cr examples/support/pixelumm_plan_reuse_probe.cr examples/support/pixelumm_connected_resource_probe.cr examples/support/pixelumm_block_probe.cr examples/support/pixelumm_memory_probe.cr examples/support/hy_nomas_probe_instrumentation.cr examples/support/pixelumm_ffn_dense.metal src/ml/core/buffer.cr src/ml/video/pixelumm_decoder_block_mps.cr src/ml/video/pixelumm_decoder_block_mps_bridge.mm src/ml/video/pixelumm_block_profile.cr src/ml/video/mps_api_profile.cr src/ml/video/mps_command_phase_profile.cr src/ml/video/mps_continuation_trace.cr src/ml/video/metal_mps_command_chain.cr $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_MPS_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_MPS_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

$(PIXELUMM_CONNECTED_RESOURCE_PROBE_OBJ): examples/support/pixelumm_connected_resource_bridge.mm src/ml/video/pixelumm_decoder_block_mps_bridge.mm src/ml/video/mps_api_profile.h | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "PixelUMM connected resource probe requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

# The diagnostic bridge includes the production decoder-block bridge exactly
# once so its resource hook can wrap the unchanged graph encode body.
$(BUILD_DIR)/pixelumm_connected_resource_probe: examples/pixelumm_connected_resource_probe.cr examples/support/pixelumm_connected_resource_probe.cr examples/support/pixelumm_connected_resource_bridge.mm examples/support/pixelumm_block_probe.cr examples/support/pixelumm_memory_probe.cr examples/support/hy_nomas_probe_instrumentation.cr examples/support/pixelumm_ffn_dense.metal src/ml/core/buffer.cr src/ml/video/pixelumm_decoder_block_mps.cr src/ml/video/pixelumm_decoder_block_mps_bridge.mm src/ml/video/pixelumm_block_profile.cr src/ml/video/mps_api_profile.cr src/ml/video/mps_command_phase_profile.cr src/ml/video/mps_continuation_trace.cr src/ml/video/metal_mps_command_chain.cr $(PIXELUMM_FFN_ISOLATED_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_CONNECTED_RESOURCE_PROBE_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_CONNECTED_RESOURCE_PROBE_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

# The diagnostic bridge includes the production bridge translation unit.
# Linking PIXELUMM_MPS_OBJ here would duplicate its existing C ABI symbols.
$(BUILD_DIR)/pixelumm_ffn_projection_probe: examples/pixelumm_ffn_projection_probe.cr examples/support/pixelumm_ffn_projection_probe.cr examples/support/pixelumm_ffn_dense.metal examples/support/pixelumm_block_probe.cr examples/support/hy_nomas_probe_instrumentation.cr src/ml/video/pixelumm_decoder_block_mps.cr src/ml/video/metal_mps_command_chain.cr src/ml/video/pixelumm_block_profile.cr $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_FFN_PROBE_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_FFN_PROBE_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

# Reuse the diagnostic builder matching the production F32 graph operations.
# The candidate reads BF16 directly without a global F32 conversion buffer.
$(BUILD_DIR)/pixelumm_ffn_f32_kernel_probe: examples/pixelumm_ffn_f32_kernel_probe.cr examples/support/pixelumm_ffn_f32_kernel_probe.cr examples/support/pixelumm_ffn_f32_gemm.metal examples/support/pixelumm_ffn_isolated_probe.cr examples/support/pixelumm_ffn_dense.metal examples/support/pixelumm_memory_probe.cr examples/support/hy_nomas_probe_instrumentation.cr $(PIXELUMM_FFN_ISOLATED_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_FFN_PROBE_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_FFN_PROBE_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

# The isolated diagnostic bridge includes production FFN builders; do not also
# link PIXELUMM_MPS_OBJ or PIXELUMM_FFN_PROBE_OBJ into its executable.
$(BUILD_DIR)/pixelumm_ffn_isolated_probe: examples/pixelumm_ffn_isolated_probe.cr examples/support/pixelumm_ffn_isolated_probe.cr examples/support/pixelumm_tiled_gate_up_probe.cr examples/support/pixelumm_ffn_dense.metal examples/support/pixelumm_memory_probe.cr examples/support/hy_nomas_probe_instrumentation.cr $(PIXELUMM_FFN_ISOLATED_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_FFN_ISOLATED_PROBE_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_FFN_ISOLATED_PROBE_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

.PHONY: pixelumm_ffn_aa_probe spec_pixelumm_ffn_aa_probe
pixelumm_ffn_aa_probe: $(BUILD_DIR)/pixelumm_ffn_aa_probe

spec_pixelumm_ffn_aa_probe:
	$(CRYSTAL) spec -Dcpu_only spec/pixelumm_ffn_aa_contract_spec.cr

# Includes the production bridge once; no other PixelUMM bridge object is linked.
$(PIXELUMM_FFN_AA_OBJ): examples/support/pixelumm_ffn_aa_bridge.mm src/ml/video/pixelumm_decoder_block_mps_bridge.mm src/ml/video/mps_api_profile.h $(COGNIGRAPH_NATIVE_HEADERS) | $(BUILD_SENTINEL)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

$(BUILD_DIR)/pixelumm_ffn_aa_probe: examples/pixelumm_ffn_aa_probe.cr examples/support/pixelumm_ffn_aa_probe.cr examples/support/pixelumm_ffn_aa_contract.cr examples/support/pixelumm_ffn_isolated_probe.cr examples/support/pixelumm_ffn_weight_reuse_probe.cr examples/support/pixelumm_ffn_dense.metal examples/support/pixelumm_memory_probe.cr examples/support/hy_nomas_probe_instrumentation.cr $(PIXELUMM_FFN_ISOLATED_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_FFN_AA_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_FFN_AA_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

.PHONY: pixelumm_ffn_graph_probe spec_pixelumm_ffn_graph_probe
pixelumm_ffn_graph_probe: $(BUILD_DIR)/pixelumm_ffn_graph_probe

spec_pixelumm_ffn_graph_probe:
	$(CRYSTAL) spec -Dcpu_only spec/pixelumm_ffn_graph_contract_spec.cr

# The whole-FFN diagnostic includes the AA and production bridges in one TU.
# Linking their separate objects would duplicate the production C ABI symbols.
$(PIXELUMM_FFN_GRAPH_OBJ): examples/support/pixelumm_ffn_graph_bridge.mm examples/support/pixelumm_ffn_aa_bridge.mm src/ml/video/pixelumm_decoder_block_mps_bridge.mm src/ml/video/mps_api_profile.h $(COGNIGRAPH_NATIVE_HEADERS) | $(BUILD_SENTINEL)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

$(BUILD_DIR)/pixelumm_ffn_graph_probe: examples/pixelumm_ffn_graph_probe.cr examples/support/pixelumm_ffn_graph_probe.cr examples/support/pixelumm_ffn_aa_probe.cr examples/support/pixelumm_ffn_aa_contract.cr examples/support/pixelumm_ffn_isolated_probe.cr examples/support/pixelumm_ffn_weight_reuse_probe.cr examples/support/pixelumm_ffn_dense.metal examples/support/pixelumm_memory_probe.cr examples/support/hy_nomas_probe_instrumentation.cr $(PIXELUMM_FFN_ISOLATED_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_FFN_GRAPH_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_FFN_GRAPH_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

.PHONY: pixelumm_ffn_stage_timing_probe spec_pixelumm_ffn_stage_timing_probe
pixelumm_ffn_stage_timing_probe: $(BUILD_DIR)/pixelumm_ffn_stage_timing_probe

spec_pixelumm_ffn_stage_timing_probe:
	$(CRYSTAL) spec -Dcpu_only spec/pixelumm_ffn_stage_timing_contract_spec.cr

$(BUILD_DIR)/pixelumm_ffn_stage_timing_probe: examples/pixelumm_ffn_stage_timing_probe.cr examples/support/pixelumm_ffn_stage_timing_contract.cr examples/support/pixelumm_ffn_graph_probe.cr examples/support/pixelumm_ffn_aa_probe.cr examples/support/pixelumm_ffn_aa_contract.cr examples/support/pixelumm_ffn_isolated_probe.cr examples/support/pixelumm_ffn_weight_reuse_probe.cr examples/support/pixelumm_ffn_dense.metal examples/support/pixelumm_memory_probe.cr examples/support/hy_nomas_probe_instrumentation.cr $(PIXELUMM_FFN_ISOLATED_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_FFN_GRAPH_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_FFN_GRAPH_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

spec_pixelumm_block_profile:
	$(CRYSTAL) spec -Dcpu_only spec/pixelumm_block_profile_spec.cr spec/wan_mps_api_profile_spec.cr spec/wan_mps_command_phase_profile_spec.cr spec/wan_mps_continuation_trace_spec.cr

.PHONY: spec_mps_api_profile_clock
spec_mps_api_profile_clock: $(WAN_MPS_CHAIN_OBJ)
	@test "$(IS_DARWIN)" = Darwin || (echo "MPS API clock probe requires macOS"; exit 1)
	$(CRYSTAL) spec spec/video_mps_api_profile_clock_bridge_spec.cr \
		--link-flags="$(abspath $(WAN_MPS_CHAIN_OBJ)) $(LINK_FLAGS) -framework MetalPerformanceShaders -fuse-ld=/usr/bin/ld"

pixelumm_decoder_two_block_probe: $(BUILD_DIR)/pixelumm_decoder_two_block_probe

pixelumm_decoder_four_block_probe: $(BUILD_DIR)/pixelumm_decoder_four_block_probe

pixelumm_decoder_eight_block_probe: $(BUILD_DIR)/pixelumm_decoder_eight_block_probe

.PHONY: pixelumm_decoder_sixteen_block_probe pixelumm_decoder_thirty_six_block_probe spec_pixelumm_thirty_six_block_probe
pixelumm_decoder_sixteen_block_probe: $(BUILD_DIR)/pixelumm_decoder_sixteen_block_probe

pixelumm_decoder_thirty_six_block_probe: $(BUILD_DIR)/pixelumm_decoder_thirty_six_block_probe

spec_pixelumm_thirty_six_block_probe:
	$(CRYSTAL) spec -Dcpu_only spec/pixelumm_two_block_probe_spec.cr spec/pixelumm_memory_probe_spec.cr \
		spec/pixelumm_serial_probe_runner_spec.cr

spec_pixelumm_memory_probe:
	$(CRYSTAL) spec -Dcpu_only spec/pixelumm_memory_probe_spec.cr

$(BUILD_DIR)/pixelumm_decoder_eight_block_probe: examples/pixelumm_decoder_eight_block_probe.cr examples/support/pixelumm_serial_probe_runner.cr examples/support/pixelumm_two_block_probe.cr examples/support/pixelumm_memory_probe.cr examples/support/pixelumm_block_probe.cr examples/support/hy_nomas_probe_instrumentation.cr src/ml/video/pixelumm_decoder_block_mps.cr src/ml/video/pixelumm_block_profile.cr $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_MPS_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_MPS_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

$(BUILD_DIR)/pixelumm_decoder_sixteen_block_probe: examples/pixelumm_decoder_sixteen_block_probe.cr examples/support/pixelumm_serial_probe_runner.cr examples/support/pixelumm_two_block_probe.cr examples/support/pixelumm_memory_probe.cr examples/support/pixelumm_block_probe.cr examples/support/hy_nomas_probe_instrumentation.cr src/ml/video/pixelumm_decoder_block_mps.cr src/ml/video/pixelumm_block_profile.cr $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_MPS_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_MPS_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

$(BUILD_DIR)/pixelumm_decoder_thirty_six_block_probe: examples/pixelumm_decoder_thirty_six_block_probe.cr examples/support/pixelumm_serial_probe_runner.cr examples/support/pixelumm_two_block_probe.cr examples/support/pixelumm_memory_probe.cr examples/support/pixelumm_block_probe.cr examples/support/hy_nomas_probe_instrumentation.cr src/ml/video/pixelumm_decoder_block_mps.cr src/ml/video/pixelumm_block_profile.cr $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_MPS_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_MPS_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

$(BUILD_DIR)/pixelumm_decoder_two_block_probe: examples/pixelumm_decoder_two_block_probe.cr examples/support/pixelumm_serial_probe_runner.cr examples/support/pixelumm_two_block_probe.cr examples/support/pixelumm_memory_probe.cr examples/support/pixelumm_block_probe.cr examples/support/hy_nomas_probe_instrumentation.cr src/ml/video/pixelumm_decoder_block_mps.cr src/ml/video/pixelumm_block_profile.cr $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_MPS_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_MPS_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

$(BUILD_DIR)/pixelumm_decoder_four_block_probe: examples/pixelumm_decoder_four_block_probe.cr examples/support/pixelumm_serial_probe_runner.cr examples/support/pixelumm_two_block_probe.cr examples/support/pixelumm_memory_probe.cr examples/support/pixelumm_block_probe.cr examples/support/hy_nomas_probe_instrumentation.cr src/ml/video/pixelumm_decoder_block_mps.cr src/ml/video/pixelumm_block_profile.cr $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_MPS_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_MPS_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

$(BUILD_DIR)/pixelumm_decoder_block_probe: examples/pixelumm_decoder_block_probe.cr examples/support/pixelumm_block_probe.cr examples/support/hy_nomas_probe_instrumentation.cr src/ml/video/pixelumm_decoder_block_mps.cr src/ml/video/pixelumm_block_profile.cr src/ml/video/mps_api_profile.cr src/ml/video/mps_continuation_trace.cr $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_MPS_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_MPS_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

spec_pixelumm_decoder_block: $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_MPS_OBJ)
	$(CRYSTAL) spec spec/pixelumm_decoder_block_admission_spec.cr spec/pixelumm_ffn_precision_spec.cr \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_MPS_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)" $(ARGS)

spec_pixelumm_two_block_probe: $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_MPS_OBJ)
	$(CRYSTAL) spec spec/pixelumm_two_block_probe_spec.cr \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_MPS_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)" $(ARGS)

.PHONY: spec_pixelumm_sixteen_block_probe
spec_pixelumm_sixteen_block_probe: $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(PIXELUMM_MPS_OBJ)
	$(CRYSTAL) spec spec/pixelumm_two_block_probe_spec.cr \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(PIXELUMM_MPS_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)" $(ARGS)

$(WAN_MPS_CHAIN_OBJ): src/ml/video/metal_mps_command_chain.mm src/ml/video/mps_api_profile.h | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "Wan MPS command chain requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

$(WAN_STAGE_IO_OBJ): src/ml/video/metal_stage_io.mm | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "Wan stage I/O requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

spec_video_staging: $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(WAN_STAGE_IO_OBJ)
	$(CRYSTAL) spec spec/video_metal_ops_spec.cr spec/video_stage_pool_spec.cr \
		spec/video_stage_io_spec.cr spec/video_mps_command_chain_spec.cr \
		spec/video_mps_command_chain_tracking_spec.cr \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(WAN_STAGE_IO_OBJ)) $(LINK_FLAGS) -framework MetalPerformanceShaders -fuse-ld=/usr/bin/ld" $(ARGS)

spec_qwen25: $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(QWEN25_MPS_OBJ)
	$(CRYSTAL) spec spec/qwen25_hy_nomas_spec.cr spec/qwen25_hy_nomas_mps_spec.cr spec/qwen25_prefix_spec.cr \
		spec/qwen25_vl_weights_spec.cr spec/qwen25_vl_embedding_spec.cr spec/qwen25_prefix_probe_spec.cr spec/qwen25_vl_embedding_mps_spec.cr \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(QWEN25_MPS_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)" $(ARGS)

$(BUILD_DIR)/qwen25_pretrained_prefix: examples/qwen25_pretrained_prefix.cr examples/support/qwen25_prefix_probe.cr examples/support/hy_nomas_text_probe.cr examples/support/hy_nomas_text_session.cr $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

$(BUILD_DIR)/qwen25_pretrained_block_probe: examples/qwen25_pretrained_block_probe.cr examples/support/qwen25_prefix_probe.cr $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(QWEN25_MPS_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(QWEN25_MPS_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

$(BUILD_DIR)/hy_nomas_text_conditioning_probe: examples/hy_nomas_text_conditioning_probe.cr examples/support/hy_nomas_text_probe.cr examples/support/hy_nomas_text_session.cr examples/support/hy_nomas_probe_instrumentation.cr examples/support/qwen25_prefix_probe.cr $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

$(BUILD_DIR)/hy_nomas_text_projection_replay: examples/hy_nomas_text_projection_replay.cr examples/support/hy_nomas_text_probe.cr examples/support/hy_nomas_text_session.cr examples/support/hy_nomas_probe_instrumentation.cr examples/support/qwen25_prefix_probe.cr $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

$(BUILD_DIR)/hy_nomas_pretrained_serial_probe: examples/hy_nomas_pretrained_serial_probe.cr examples/support/hy_nomas_pretrained_serial_probe.cr examples/support/hy_nomas_text_probe.cr examples/support/hy_nomas_text_session.cr examples/support/hy_nomas_probe_instrumentation.cr examples/support/qwen25_prefix_probe.cr $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

hy_nomas_pretrained_serial_probe: $(BUILD_DIR)/hy_nomas_pretrained_serial_probe

$(BUILD_DIR)/hy_nomas_bf16_cache_probe: examples/hy_nomas_bf16_cache_probe.cr $(wildcard examples/support/hy_nomas*.cr examples/support/qwen25*.cr) $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

hy_nomas_bf16_cache_probe: $(BUILD_DIR)/hy_nomas_bf16_cache_probe

$(BUILD_DIR)/hy_nomas_bf16_serial_probe: examples/hy_nomas_bf16_serial_probe.cr $(wildcard examples/support/hy_nomas*.cr examples/support/qwen25*.cr) $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

hy_nomas_bf16_serial_probe: $(BUILD_DIR)/hy_nomas_bf16_serial_probe

$(BUILD_DIR)/hy_nomas_full_j_probe: examples/hy_nomas_full_j_probe.cr $(wildcard examples/support/hy_nomas*.cr examples/support/qwen25*.cr) $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

.PHONY: hy_nomas_full_j_probe spec_hy_nomas_full_j
hy_nomas_full_j_probe: $(BUILD_DIR)/hy_nomas_full_j_probe

spec_hy_nomas_full_j: $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) spec spec/hy_nomas_full_j_admission_spec.cr \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)" $(ARGS)

$(BUILD_DIR)/hy_nomas_block47_cache_probe: examples/hy_nomas_block47_cache_probe.cr $(wildcard examples/support/hy_nomas*.cr examples/support/qwen25*.cr) $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

.PHONY: hy_nomas_block47_cache_probe
hy_nomas_block47_cache_probe: $(BUILD_DIR)/hy_nomas_block47_cache_probe

$(BUILD_DIR)/hy_nomas_four_block_probe: examples/hy_nomas_four_block_probe.cr $(wildcard examples/support/hy_nomas*.cr examples/support/qwen25*.cr) $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

$(BUILD_DIR)/hy_nomas_conditioned_four_probe: examples/hy_nomas_conditioned_four_probe.cr $(wildcard examples/support/hy_nomas*.cr examples/support/qwen25*.cr) $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

.PHONY: hy_nomas_conditioned_four_probe
hy_nomas_conditioned_four_probe: $(BUILD_DIR)/hy_nomas_conditioned_four_probe

.PHONY: spec_hy_nomas_conditioned_four
spec_hy_nomas_conditioned_four: $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) spec spec/hy_nomas_conditioned_four_admission_spec.cr spec/hy_nomas_padded_conditioning_admission_spec.cr \
		spec/hy_nomas_mps_padded_prefix_spec.cr spec/hy_nomas_mps_shared_modulation_spec.cr \
		spec/hy_nomas_serial_mps_spec.cr spec/hy_nomas_text_refiner_spec.cr \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)" $(ARGS)

$(BUILD_DIR)/hy_nomas_conditioned_eight_probe: examples/hy_nomas_conditioned_eight_probe.cr $(wildcard examples/support/hy_nomas*.cr examples/support/qwen25*.cr) $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

.PHONY: hy_nomas_q4_dit_pack hy_nomas_q4_dit_probe spec_hy_nomas_q4_dit
hy_nomas_q4_dit_pack: $(BUILD_DIR)/hy_nomas_q4_dit_pack
hy_nomas_q4_dit_probe: $(BUILD_DIR)/hy_nomas_q4_dit_probe

.PHONY: hy_nomas_generate
hy_nomas_generate: $(BUILD_DIR)/hy_nomas_generate

.PHONY: pixelumm_after_dit_probe spec_hy_nomas_decoder_replay
pixelumm_after_dit_probe: $(BUILD_DIR)/pixelumm_after_dit_probe

$(BUILD_DIR)/pixelumm_after_dit_probe: examples/pixelumm_after_dit_probe.cr examples/support/pixelumm_decoder_replay.metal $(wildcard examples/support/hy_nomas*.cr examples/support/pixelumm*.cr) $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS) $(PIXELUMM_HEAD_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(abspath $(PIXELUMM_HEAD_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

.PHONY: pixelumm_recovery_probe
pixelumm_recovery_probe: $(BUILD_DIR)/pixelumm_recovery_probe

$(PIXELUMM_RECOVERY_HOST_OBJ): examples/support/pixelumm_recovery_host.mm | $(BUILD_SENTINEL)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

$(BUILD_DIR)/pixelumm_recovery_probe: examples/pixelumm_recovery_probe.cr examples/support/pixelumm_recovery_host.mm examples/support/pixelumm_decoder_replay.metal $(wildcard examples/support/hy_nomas*.cr examples/support/pixelumm*.cr) $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS) $(PIXELUMM_HEAD_OBJ) $(PIXELUMM_RECOVERY_HOST_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(abspath $(PIXELUMM_HEAD_OBJ)) $(abspath $(PIXELUMM_RECOVERY_HOST_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

spec_hy_nomas_decoder_replay: $(BRIDGE_OBJ) $(WAN_SPEC_OBJS) $(PIXELUMM_HEAD_OBJ)
	$(CRYSTAL) spec spec/hy_nomas_decoder_replay_spec.cr \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(abspath $(PIXELUMM_HEAD_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc" $(ARGS)

.PHONY: pixelumm_generation_probe
pixelumm_generation_probe: $(BUILD_DIR)/pixelumm_generation_probe

.PHONY: hy_nomas_euler_probe
hy_nomas_euler_probe: $(BUILD_DIR)/hy_nomas_euler_probe

$(BUILD_DIR)/hy_nomas_euler_probe: examples/hy_nomas_euler_probe.cr examples/support/hy_nomas_probe_instrumentation.cr $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

$(BUILD_DIR)/pixelumm_generation_probe: examples/pixelumm_generation_probe.cr examples/support/pixelumm_generation_probe.metal examples/support/hy_nomas_generation.cr $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS) $(PIXELUMM_HEAD_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(abspath $(PIXELUMM_HEAD_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

.PHONY: pixelumm_global_assembly_probe hy_nomas_history_attention_probe hy_nomas_sequence_generate hy_nomas_timeline_ops_probe hy_nomas_continuation_history_probe hy_nomas_resume_probe hy_nomas_sequence_resume_probe hy_nomas_cache_qbit_probe spec_hy_nomas_resume hy_nomas_bf16_linear_probe hy_nomas_bf16_patch_probe hy_nomas_boundary_attention_probe hy_nomas_real_attention_probe hy_nomas_decode_latent
pixelumm_global_assembly_probe: $(BUILD_DIR)/pixelumm_global_assembly_probe
hy_nomas_history_attention_probe: $(BUILD_DIR)/hy_nomas_history_attention_probe
hy_nomas_sequence_generate: $(BUILD_DIR)/hy_nomas_sequence_generate
hy_nomas_timeline_ops_probe: $(BUILD_DIR)/hy_nomas_timeline_ops_probe
hy_nomas_continuation_history_probe: $(BUILD_DIR)/hy_nomas_continuation_history_probe
hy_nomas_resume_probe: $(BUILD_DIR)/hy_nomas_resume_probe
hy_nomas_sequence_resume_probe: $(BUILD_DIR)/hy_nomas_sequence_resume_probe
hy_nomas_cache_qbit_probe: $(BUILD_DIR)/hy_nomas_cache_qbit_probe

$(BUILD_DIR)/pixelumm_global_assembly_probe: examples/pixelumm_global_assembly_probe.cr examples/support/pixelumm_global_assembly_probe.cr examples/support/pixelumm_global_assembly_probe.metal $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS) $(PIXELUMM_HEAD_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(abspath $(PIXELUMM_HEAD_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

$(BUILD_DIR)/hy_nomas_history_attention_probe: examples/hy_nomas_history_attention_probe.cr $(wildcard examples/support/hy_nomas_history_attention*) $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS) $(HY_NOMAS_HISTORY_ATTENTION_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(abspath $(HY_NOMAS_HISTORY_ATTENTION_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

hy_nomas_bf16_linear_probe: $(BUILD_DIR)/hy_nomas_bf16_linear_probe
hy_nomas_bf16_patch_probe: $(BUILD_DIR)/hy_nomas_bf16_patch_probe
hy_nomas_boundary_attention_probe: $(BUILD_DIR)/hy_nomas_boundary_attention_probe
hy_nomas_real_attention_probe: $(BUILD_DIR)/hy_nomas_real_attention_probe
hy_nomas_decode_latent: $(BUILD_DIR)/hy_nomas_decode_latent

$(BUILD_DIR)/hy_nomas_bf16_linear_probe: examples/hy_nomas_bf16_linear_probe.cr examples/support/hy_nomas_probe_instrumentation.cr src/ml/video/hy_nomas_mps_bf16_linear.cr $(wildcard examples/support/hy_nomas*.cr examples/support/qwen25*.cr) $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS) $(HY_NOMAS_BF16_LINEAR_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(abspath $(HY_NOMAS_BF16_LINEAR_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

$(BUILD_DIR)/hy_nomas_bf16_patch_probe: examples/hy_nomas_bf16_patch_probe.cr examples/support/hy_nomas_probe_instrumentation.cr src/ml/video/hy_nomas_dit_boundaries.cr src/ml/video/hy_nomas_mps_bf16_linear.cr $(wildcard examples/support/hy_nomas*.cr examples/support/qwen25*.cr) $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS) $(HY_NOMAS_BF16_LINEAR_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(abspath $(HY_NOMAS_BF16_LINEAR_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

$(BUILD_DIR)/hy_nomas_boundary_attention_probe: examples/hy_nomas_boundary_attention_probe.cr $(wildcard examples/support/pixelumm*.cr examples/support/hy_nomas*.cr examples/support/qwen25*.cr) $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS) $(HY_NOMAS_HISTORY_ATTENTION_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(abspath $(HY_NOMAS_HISTORY_ATTENTION_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

$(BUILD_DIR)/hy_nomas_real_attention_probe: examples/hy_nomas_real_attention_probe.cr $(wildcard examples/support/pixelumm*.cr examples/support/hy_nomas*.cr examples/support/qwen25*.cr) $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

# The head object includes the PixelUMM block ABI; do not link the block twice.
$(BUILD_DIR)/hy_nomas_decode_latent: examples/hy_nomas_decode_latent.cr $(wildcard examples/support/pixelumm*.cr examples/support/hy_nomas*.cr examples/support/qwen25*.cr) $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS) $(PIXELUMM_HEAD_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(abspath $(PIXELUMM_HEAD_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

$(BUILD_DIR)/hy_nomas_generate: examples/hy_nomas_generate.cr examples/support/hy_nomas_generation.cr $(wildcard examples/support/pixelumm*.cr examples/support/hy_nomas*.cr) $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS) $(PIXELUMM_HEAD_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(abspath $(PIXELUMM_HEAD_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

$(BUILD_DIR)/hy_nomas_sequence_generate: examples/hy_nomas_sequence_generate.cr $(wildcard examples/support/pixelumm*.cr examples/support/hy_nomas*.cr) $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS) $(PIXELUMM_HEAD_OBJ) $(HY_NOMAS_HISTORY_ATTENTION_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(abspath $(PIXELUMM_HEAD_OBJ)) $(abspath $(HY_NOMAS_HISTORY_ATTENTION_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

$(BUILD_DIR)/hy_nomas_sequence_resume_probe: examples/hy_nomas_sequence_resume_probe.cr $(wildcard examples/support/pixelumm*.cr examples/support/hy_nomas*.cr) examples/support/hy_nomas_cache_qbit.metal src/ml/gguf/qwen_qbit_gaussian_codec.cr $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS) $(PIXELUMM_HEAD_OBJ) $(HY_NOMAS_HISTORY_ATTENTION_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(abspath $(PIXELUMM_HEAD_OBJ)) $(abspath $(HY_NOMAS_HISTORY_ATTENTION_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

$(BUILD_DIR)/hy_nomas_cache_qbit_probe: examples/hy_nomas_cache_qbit_probe.cr examples/support/hy_nomas_cache_qbit.cr examples/support/hy_nomas_cache_qbit.metal src/ml/gguf/qwen_qbit_gaussian_codec.cr $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

$(BUILD_DIR)/hy_nomas_compact_sequence_probe: examples/hy_nomas_compact_sequence_probe.cr $(wildcard examples/support/pixelumm*.cr examples/support/hy_nomas*.cr examples/support/hy_nomas*.metal) src/ml/gguf/qwen_qbit_gaussian_codec.cr $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS) $(PIXELUMM_HEAD_OBJ) $(HY_NOMAS_HISTORY_ATTENTION_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(abspath $(PIXELUMM_HEAD_OBJ)) $(abspath $(HY_NOMAS_HISTORY_ATTENTION_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

$(BUILD_DIR)/hy_nomas_cache_qbit_checkpoint_probe: examples/hy_nomas_cache_qbit_checkpoint_probe.cr $(wildcard examples/support/hy_nomas_cache_qbit*.cr examples/support/hy_nomas_cache_qbit*.metal) src/ml/gguf/qwen_qbit_gaussian_codec.cr $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

$(BUILD_DIR)/hy_nomas_timeline_ops_probe: examples/hy_nomas_timeline_ops_probe.cr $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

$(BUILD_DIR)/hy_nomas_segment_edit_probe: examples/hy_nomas_segment_edit_probe.cr $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

spec_hy_nomas_segment_edit: $(BRIDGE_OBJ) $(WAN_SPEC_OBJS) $(HY_NOMAS_HISTORY_ATTENTION_OBJ)
	$(CRYSTAL) spec spec/hy_nomas_segment_edit_spec.cr spec/hy_nomas_sequence_admission_spec.cr \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(abspath $(HY_NOMAS_HISTORY_ATTENTION_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc" $(ARGS)

$(BUILD_DIR)/hy_nomas_continuation_history_probe: examples/hy_nomas_continuation_history_probe.cr examples/support/hy_nomas_continuation_history_probe.cr examples/support/hy_nomas_continuation_history_probe.metal src/ml/video/kernels/hy_nomas_continuation_history_probe.metal $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS) $(HY_NOMAS_HISTORY_ATTENTION_OBJ)
	$(CRYSTAL) build $< --release -Dhy_nomas_history_probe -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(abspath $(HY_NOMAS_HISTORY_ATTENTION_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

# The resume probe reuses the extracted synthetic history helper from the old
# probe. Keep the old target as a prerequisite so its extraction/build boundary
# remains explicit when this source is compiled independently.
$(BUILD_DIR)/hy_nomas_resume_probe: examples/hy_nomas_resume_probe.cr examples/support/hy_nomas_continuation_history_probe.cr examples/support/hy_nomas_continuation_history_probe.metal $(BUILD_DIR)/hy_nomas_continuation_history_probe $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS) $(HY_NOMAS_HISTORY_ATTENTION_OBJ)
	$(CRYSTAL) build $< --release -Dhy_nomas_history_probe -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(abspath $(HY_NOMAS_HISTORY_ATTENTION_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc"

spec_hy_nomas_resume: $(BRIDGE_OBJ) $(WAN_SPEC_OBJS) $(HY_NOMAS_HISTORY_ATTENTION_OBJ)
	$(CRYSTAL) spec spec/hy_nomas_continuation_history_spec.cr spec/hy_nomas_timeline_ops_spec.cr spec/hy_nomas_resume_bundle_spec.cr \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(abspath $(HY_NOMAS_HISTORY_ATTENTION_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc" $(ARGS)

spec_hy_nomas_compact_resume: $(BRIDGE_OBJ) $(WAN_SPEC_OBJS) $(HY_NOMAS_HISTORY_ATTENTION_OBJ)
	$(CRYSTAL) spec spec/hy_nomas_continuation_history_spec.cr spec/hy_nomas_timeline_ops_spec.cr spec/hy_nomas_resume_bundle_spec.cr spec/hy_nomas_cache_qbit_checkpoint_spec.cr \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(abspath $(HY_NOMAS_HISTORY_ATTENTION_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS) -lproc" $(ARGS)

.PHONY: hy_nomas_q4_reduction_probe
hy_nomas_q4_reduction_probe: $(BUILD_DIR)/hy_nomas_q4_reduction_probe

$(BUILD_DIR)/hy_nomas_q4_reduction_probe: examples/hy_nomas_q4_reduction_probe.cr examples/support/hy_nomas_q4_reduction_probe.cr examples/support/hy_nomas_q4_reduction.metal examples/support/hy_nomas_probe_instrumentation.cr src/ml/gguf/kernels/gemm_q4k.metal $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

.PHONY: hy_nomas_q4_paired_benchmark
hy_nomas_q4_paired_benchmark: $(BUILD_DIR)/hy_nomas_q4_paired_benchmark

$(BUILD_DIR)/hy_nomas_q4_paired_benchmark: examples/hy_nomas_q4_paired_benchmark.cr examples/support/hy_nomas_q4_paired_benchmark.cr examples/support/hy_nomas_q4_reduction_probe.cr examples/support/hy_nomas_q4_reduction.metal examples/support/hy_nomas_probe_instrumentation.cr src/ml/gguf/kernels/gemm_q4k.metal $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

.PHONY: hy_nomas_q4_tile_reuse_benchmark
hy_nomas_q4_tile_reuse_benchmark: $(BUILD_DIR)/hy_nomas_q4_tile_reuse_benchmark

$(BUILD_DIR)/hy_nomas_q4_tile_reuse_benchmark: examples/hy_nomas_q4_tile_reuse_benchmark.cr examples/support/hy_nomas_q4_tile_reuse_benchmark.cr examples/support/hy_nomas_q4_paired_benchmark.cr examples/support/hy_nomas_q4_reduction_probe.cr examples/support/hy_nomas_q4_reduction.metal examples/support/hy_nomas_probe_instrumentation.cr src/ml/gguf/kernels/gemm_q4k.metal $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

.PHONY: hy_nomas_q4_tile_dense_benchmark
hy_nomas_q4_tile_dense_benchmark: $(BUILD_DIR)/hy_nomas_q4_tile_dense_benchmark

$(BUILD_DIR)/hy_nomas_q4_tile_dense_benchmark: examples/hy_nomas_q4_tile_dense_benchmark.cr examples/support/hy_nomas_q4_tile_dense_benchmark.cr examples/support/hy_nomas_q4_tile_reuse_benchmark.cr examples/support/hy_nomas_q4_paired_benchmark.cr examples/support/hy_nomas_q4_reduction_probe.cr examples/support/hy_nomas_q4_reduction.metal examples/support/hy_nomas_probe_instrumentation.cr src/ml/gguf/kernels/gemm_q4k.metal $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

.PHONY: hy_nomas_q4_b64_schedule_benchmark
hy_nomas_q4_b64_schedule_benchmark: $(BUILD_DIR)/hy_nomas_q4_b64_schedule_benchmark

$(BUILD_DIR)/hy_nomas_q4_b64_schedule_benchmark: examples/hy_nomas_q4_b64_schedule_benchmark.cr examples/support/hy_nomas_q4_b64_schedule_benchmark.cr examples/support/hy_nomas_q4_tile_dense_benchmark.cr examples/support/hy_nomas_q4_tile_reuse_benchmark.cr examples/support/hy_nomas_q4_paired_benchmark.cr examples/support/hy_nomas_q4_reduction_probe.cr examples/support/hy_nomas_q4_reduction.metal examples/support/hy_nomas_probe_instrumentation.cr src/ml/gguf/kernels/gemm_q4k.metal $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

.PHONY: hy_nomas_q4_kpanel_benchmark
hy_nomas_q4_kpanel_benchmark: $(BUILD_DIR)/hy_nomas_q4_kpanel_benchmark

$(BUILD_DIR)/hy_nomas_q4_kpanel_benchmark: examples/hy_nomas_q4_kpanel_benchmark.cr examples/support/hy_nomas_q4_kpanel_benchmark.cr examples/support/hy_nomas_q4_b64_schedule_benchmark.cr examples/support/hy_nomas_q4_tile_dense_benchmark.cr examples/support/hy_nomas_q4_tile_reuse_benchmark.cr examples/support/hy_nomas_q4_paired_benchmark.cr examples/support/hy_nomas_q4_reduction_probe.cr examples/support/hy_nomas_q4_reduction.metal examples/support/hy_nomas_probe_instrumentation.cr src/ml/gguf/kernels/gemm_q4k.metal $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

.PHONY: hy_nomas_q4_shared_reserve_benchmark
hy_nomas_q4_shared_reserve_benchmark: $(BUILD_DIR)/hy_nomas_q4_shared_reserve_benchmark

$(BUILD_DIR)/hy_nomas_q4_shared_reserve_benchmark: examples/hy_nomas_q4_shared_reserve_benchmark.cr examples/support/hy_nomas_q4_shared_reserve_benchmark.cr examples/support/hy_nomas_q4_kpanel_benchmark.cr examples/support/hy_nomas_q4_b64_schedule_benchmark.cr examples/support/hy_nomas_q4_tile_dense_benchmark.cr examples/support/hy_nomas_q4_tile_reuse_benchmark.cr examples/support/hy_nomas_q4_paired_benchmark.cr examples/support/hy_nomas_q4_reduction_probe.cr examples/support/hy_nomas_q4_reduction.metal examples/support/hy_nomas_probe_instrumentation.cr src/ml/gguf/kernels/gemm_q4k.metal $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

.PHONY: hy_nomas_q4_k16_benchmark
hy_nomas_q4_k16_benchmark: $(BUILD_DIR)/hy_nomas_q4_k16_benchmark

$(BUILD_DIR)/hy_nomas_q4_k16_benchmark: examples/hy_nomas_q4_k16_benchmark.cr examples/support/hy_nomas_q4_k16_benchmark.cr examples/support/hy_nomas_q4_kpanel_benchmark.cr examples/support/hy_nomas_q4_b64_schedule_benchmark.cr examples/support/hy_nomas_q4_tile_dense_benchmark.cr examples/support/hy_nomas_q4_tile_reuse_benchmark.cr examples/support/hy_nomas_q4_paired_benchmark.cr examples/support/hy_nomas_q4_reduction_probe.cr examples/support/hy_nomas_q4_reduction.metal examples/support/hy_nomas_probe_instrumentation.cr src/ml/gguf/kernels/gemm_q4k.metal $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

$(BUILD_DIR)/hy_nomas_q4_dit_pack: examples/hy_nomas_q4_dit_pack.cr $(wildcard examples/support/hy_nomas*.cr) $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

$(BUILD_DIR)/hy_nomas_q4_dit_probe: examples/hy_nomas_q4_dit_probe.cr $(wildcard examples/support/hy_nomas*.cr) $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

.PHONY: hy_nomas_text_compaction_benchmark
hy_nomas_text_compaction_benchmark: $(BUILD_DIR)/hy_nomas_text_compaction_benchmark

$(BUILD_DIR)/hy_nomas_text_compaction_benchmark: examples/hy_nomas_text_compaction_benchmark.cr $(wildcard examples/support/hy_nomas*.cr) $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

.PHONY: hy_nomas_text_row_projection_probe hy_nomas_text_row_stages_probe
hy_nomas_text_row_projection_probe: $(BUILD_DIR)/hy_nomas_text_row_projection_probe
hy_nomas_text_row_stages_probe: $(BUILD_DIR)/hy_nomas_text_row_stages_probe

$(BUILD_DIR)/hy_nomas_text_row_projection_probe $(BUILD_DIR)/hy_nomas_text_row_stages_probe: $(BUILD_DIR)/%: examples/%.cr $(wildcard examples/support/hy_nomas*.cr) $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

spec_hy_nomas_q4_dit: $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) spec spec/hy_nomas_dit_boundaries_spec.cr spec/hy_nomas_q4_cache_spec.cr spec/hy_nomas_q4_stack_spec.cr spec/hy_nomas_q4_bf16_finite_spec.cr spec/hy_nomas_serial_compaction_admission_spec.cr \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)" $(ARGS)

.PHONY: hy_nomas_conditioned_eight_probe spec_hy_nomas_conditioned_eight
hy_nomas_conditioned_eight_probe: $(BUILD_DIR)/hy_nomas_conditioned_eight_probe

spec_hy_nomas_conditioned_eight: $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) spec spec/hy_nomas_conditioned_eight_admission_spec.cr spec/hy_nomas_block47_cache_admission_spec.cr \
		spec/hy_nomas_serial_mps_spec.cr spec/hy_nomas_text_refiner_spec.cr \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)" $(ARGS)

$(BUILD_DIR)/hy_nomas_block23_cache_probe: examples/hy_nomas_block23_cache_probe.cr $(wildcard examples/support/hy_nomas*.cr examples/support/qwen25*.cr) $(QWEN25_PREFIX_SOURCES) $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

.PHONY: hy_nomas_four_block_probe hy_nomas_block23_cache_probe spec_hy_nomas_four_block
hy_nomas_four_block_probe: $(BUILD_DIR)/hy_nomas_four_block_probe
hy_nomas_block23_cache_probe: $(BUILD_DIR)/hy_nomas_block23_cache_probe

spec_hy_nomas_four_block: $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) spec spec/hy_nomas_four_block_admission_spec.cr spec/hy_nomas_block23_cache_admission_spec.cr spec/hy_nomas_serial_mps_spec.cr spec/hy_nomas_full_j_admission_spec.cr \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)" $(ARGS)

spec_hy_nomas_bf16_cache: $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) spec spec/video_bf16_weight_cache_spec.cr spec/hy_nomas_bf16_cache_admission_spec.cr spec/hy_nomas_bf16_serial_admission_spec.cr \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)" $(ARGS)

# These targets inspect metadata only; no bridge link or Metal device is needed.
$(BUILD_DIR)/hy_nomas_residency_plan: examples/hy_nomas_residency_plan.cr $(wildcard examples/support/hy_nomas*.cr examples/support/qwen25*.cr) $(QWEN25_PREFIX_SOURCES) spec/support/video_allocation_counter.cr | $(BUILD_SENTINEL)
	$(CRYSTAL) build $< --release -o $@

hy_nomas_residency_plan: $(BUILD_DIR)/hy_nomas_residency_plan

spec_hy_nomas_residency:
	$(CRYSTAL) spec spec/hy_nomas_residency_plan_spec.cr $(ARGS)

spec: $(BRIDGE_OBJ) $(WAN_SPEC_OBJS)
	$(CRYSTAL) spec \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_SPEC_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

spec_wan: $(BRIDGE_OBJ) $(WAN_MPS_OBJS)
	$(CRYSTAL) spec $(wildcard spec/wan*_spec.cr) $(wildcard spec/video_*_spec.cr) \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)" $(ARGS)

spec_hy_nomas: $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(WAN_STAGE_IO_OBJ) $(WAN_T5_MPS_OBJ) $(WAN_T5_RMS_MPS_OBJ) $(HY_NOMAS_MPS_OBJ) $(HY_NOMAS_COMPACT_ATTENTION_OBJ)
	$(CRYSTAL) spec spec/hy_nomas_block_spec.cr spec/hy_nomas_block_profile_spec.cr \
		spec/hy_nomas_serial_mps_spec.cr \
		spec/hy_nomas_attention_probe_spec.cr \
		spec/hy_nomas_pretrained_serial_admission_spec.cr \
		spec/hy_nomas_pretrained_serial_probe_spec.cr \
		spec/hy_nomas_residency_plan_spec.cr \
		spec/video_bf16_weight_cache_spec.cr spec/hy_nomas_bf16_cache_admission_spec.cr spec/hy_nomas_bf16_serial_admission_spec.cr \
		spec/hy_nomas_pointwise_probe_spec.cr spec/hy_nomas_compact_attention_spec.cr \
		spec/hy_nomas_compact_attention_oracle_spec.cr \
		spec/hy_nomas_mps_shared_modulation_spec.cr spec/hy_nomas_text_probe_spec.cr \
		spec/hy_nomas_text_refiner_spec.cr \
		spec/hy_nomas_text_residual_spec.cr \
		spec/hy_nomas_text_session_lifecycle_spec.cr \
		spec/hy_nomas_text_timestep_spec.cr \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(WAN_STAGE_IO_OBJ)) $(abspath $(WAN_T5_MPS_OBJ)) $(abspath $(WAN_T5_RMS_MPS_OBJ)) $(abspath $(HY_NOMAS_MPS_OBJ)) $(abspath $(HY_NOMAS_COMPACT_ATTENTION_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)" $(ARGS)

$(BUILD_DIR)/hy_nomas_attention_benchmark: examples/hy_nomas_attention_benchmark.cr $(WAN_BENCHMARK_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(WAN_STAGE_IO_OBJ) $(HY_NOMAS_MPS_OBJ) $(HY_NOMAS_COMPACT_ATTENTION_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(WAN_STAGE_IO_OBJ)) $(abspath $(HY_NOMAS_MPS_OBJ)) $(abspath $(HY_NOMAS_COMPACT_ATTENTION_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

$(BUILD_DIR)/hy_nomas_block_benchmark: examples/hy_nomas_block_benchmark.cr $(WAN_BENCHMARK_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(WAN_STAGE_IO_OBJ) $(WAN_T5_MPS_OBJ) $(WAN_T5_RMS_MPS_OBJ) $(HY_NOMAS_MPS_OBJ) $(HY_NOMAS_COMPACT_ATTENTION_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(WAN_STAGE_IO_OBJ)) $(abspath $(WAN_T5_MPS_OBJ)) $(abspath $(WAN_T5_RMS_MPS_OBJ)) $(abspath $(HY_NOMAS_MPS_OBJ)) $(abspath $(HY_NOMAS_COMPACT_ATTENTION_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

hy_nomas_block_benchmark: $(BUILD_DIR)/hy_nomas_block_benchmark

$(BUILD_DIR)/hy_nomas_serial_benchmark: examples/hy_nomas_serial_benchmark.cr $(WAN_BENCHMARK_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_CHAIN_OBJ) $(WAN_STAGE_IO_OBJ) $(WAN_T5_MPS_OBJ) $(WAN_T5_RMS_MPS_OBJ) $(HY_NOMAS_MPS_OBJ) $(HY_NOMAS_COMPACT_ATTENTION_OBJ)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_CHAIN_OBJ)) $(abspath $(WAN_STAGE_IO_OBJ)) $(abspath $(WAN_T5_MPS_OBJ)) $(abspath $(WAN_T5_RMS_MPS_OBJ)) $(abspath $(HY_NOMAS_MPS_OBJ)) $(abspath $(HY_NOMAS_COMPACT_ATTENTION_OBJ)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

hy_nomas_serial_benchmark: $(BUILD_DIR)/hy_nomas_serial_benchmark

$(BUILD_DIR)/wan_s2v_projection_benchmark: examples/wan_s2v_projection_benchmark.cr $(WAN_BENCHMARK_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

$(BUILD_DIR)/wan_s2v_block_benchmark: examples/wan_s2v_block_benchmark.cr $(WAN_BENCHMARK_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

$(BUILD_DIR)/wan_s2v_full_block_benchmark: examples/wan_s2v_full_block_benchmark.cr $(WAN_BENCHMARK_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

$(BUILD_DIR)/wan_s2v_self_attention_benchmark: examples/wan_s2v_self_attention_benchmark.cr $(WAN_BENCHMARK_SOURCES) $(BRIDGE_OBJ) $(WAN_MPS_OBJS)
	$(CRYSTAL) build $< --release -o $@ \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(abspath $(WAN_MPS_OBJS)) $(LINK_FLAGS) $(WAN_SPEC_LINK_FLAGS)"

build: $(BRIDGE_OBJ)
	$(CRYSTAL) build src/ml.cr \
		--link-flags="$(shell pwd)/$(BRIDGE_OBJ) $(LINK_FLAGS)"

drama3_studio: $(BRIDGE_OBJ) | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "DRAMA Studio requires macOS Metal"; exit 1)
	$(CRYSTAL) build examples/drama3_studio.cr --release -Ddrama3_metal \
		-o $(BUILD_DIR)/drama3_studio \
		--link-flags="$(abspath $(BRIDGE_OBJ)) $(LINK_FLAGS) -framework MetalPerformanceShaders -fuse-ld=/usr/bin/ld"

profile_nomic: $(BRIDGE_OBJ)
	$(CRYSTAL) run bin/profile_nomic_stages.cr \
		--link-flags="$(shell pwd)/$(BRIDGE_OBJ) $(LINK_FLAGS)" \
		-- $(ARGS)

profile_nomic_layers: $(BRIDGE_OBJ)
	$(CRYSTAL) run bin/profile_nomic_stages.cr \
		--link-flags="$(shell pwd)/$(BRIDGE_OBJ) $(LINK_FLAGS)" \
		-- --mode=layers $(ARGS)

profile_nomic_vs_llama: $(BRIDGE_OBJ)
	@if [ -z "$(LLAMA_LIB_DIR)" ]; then \
		echo "ERROR: libllama not detected. Set LLAMA_DIR or LLAMA_LIB_DIR."; \
		exit 1; \
	fi
	LIBRARY_PATH="$(LLAMA_LIB_DIR):$$LIBRARY_PATH" \
	LD_LIBRARY_PATH="$(LLAMA_LIB_DIR):$$LD_LIBRARY_PATH" \
	DYLD_LIBRARY_PATH="$(LLAMA_LIB_DIR):$$DYLD_LIBRARY_PATH" \
	$(CRYSTAL) run bin/profile_nomic_vs_llama.cr \
		--link-flags="$(shell pwd)/$(BRIDGE_OBJ) $(LINK_FLAGS) -L$(LLAMA_LIB_DIR)" \
		-- $(ARGS)

benchmark_qwen_prefill_vs_llama_same_token: $(BUILD_DIR)/benchmark_qwen_prefill_vs_llama_same_token
$(BUILD_DIR)/benchmark_qwen_prefill_vs_llama_same_token: bin/benchmark_qwen_prefill_vs_llama_same_token.cr $(BRIDGE_OBJ)
	@if [ -z "$(LLAMA_LIB_DIR)" ]; then \
		echo "ERROR: libllama not detected. Set LLAMA_DIR or LLAMA_LIB_DIR."; \
		exit 1; \
	fi
	LIBRARY_PATH="$(LLAMA_LIB_DIR):$$LIBRARY_PATH" \
	$(CRYSTAL) build bin/benchmark_qwen_prefill_vs_llama_same_token.cr \
		-o $(BUILD_DIR)/benchmark_qwen_prefill_vs_llama_same_token \
		--link-flags="$(shell pwd)/$(BRIDGE_OBJ) $(LINK_FLAGS) -L$(LLAMA_LIB_DIR) -Wl,-rpath,$(LLAMA_LIB_DIR)"

benchmark_qwen_decode_vs_llama_same_token: $(BUILD_DIR)/benchmark_qwen_decode_vs_llama_same_token
$(BUILD_DIR)/benchmark_qwen_decode_vs_llama_same_token: bin/benchmark_qwen_decode_vs_llama_same_token.cr $(BRIDGE_OBJ)
	@if [ -z "$(LLAMA_LIB_DIR)" ]; then \
		echo "ERROR: libllama not detected. Set LLAMA_DIR or LLAMA_LIB_DIR."; \
		exit 1; \
	fi
	LIBRARY_PATH="$(LLAMA_LIB_DIR):$$LIBRARY_PATH" \
	$(CRYSTAL) build bin/benchmark_qwen_decode_vs_llama_same_token.cr \
		-o $(BUILD_DIR)/benchmark_qwen_decode_vs_llama_same_token \
		--link-flags="$(shell pwd)/$(BRIDGE_OBJ) $(LINK_FLAGS) -L$(LLAMA_LIB_DIR) -Wl,-rpath,$(LLAMA_LIB_DIR)"

spec_cpu:
	$(CRYSTAL) spec -Dcpu_only

# Q4_K dequantization reference helper — links llama.cpp's libggml-base
# so specs can bit-compare Crystal output against the reference.
q4k_ref: spec/support/q4k_ref
spec/support/q4k_ref: spec/support/q4k_ref.c
	@if [ ! -d "$(LLAMA_DIR)/build/bin" ]; then \
		echo "ERROR: $(LLAMA_DIR)/build/bin not found. Run 'make llama' first or set LLAMA_DIR."; \
		exit 1; \
	fi
	clang -O2 -o spec/support/q4k_ref spec/support/q4k_ref.c \
		-L$(LLAMA_DIR)/build/bin -lggml-base \
		-Wl,-rpath,$(LLAMA_DIR)/build/bin

build_cpu:
	$(CRYSTAL) build src/ml.cr -Dcpu_only

llama:
	@if [ -d "$(LLAMA_DIR)" ]; then \
		echo "Building llama.cpp in $(LLAMA_DIR)"; \
		cmake -S $(LLAMA_DIR) -B $(LLAMA_BUILD) -DLLAMA_BUILD_SHARED=ON -DCMAKE_BUILD_TYPE=Release; \
		cmake --build $(LLAMA_BUILD) --config Release; \
	elif [ -n "$(LLAMA_LIB_DIR)" ]; then \
		echo "libllama detected at $(LLAMA_LIB_DIR) (skipping build)"; \
	else \
		echo "ERROR: $(LLAMA_DIR) not found and libllama not detected."; \
		echo "Set LLAMA_DIR or install llama.cpp (e.g., Homebrew on macOS)."; \
		exit 1; \
	fi

llama_env:
	@if [ -n "$(LLAMA_LIB_DIR)" ]; then \
		echo "export LIBRARY_PATH=$(LLAMA_LIB_DIR)"; \
		if [ "$(IS_DARWIN)" = "Darwin" ]; then \
			echo "export DYLD_LIBRARY_PATH=$(LLAMA_LIB_DIR)"; \
		else \
			echo "export LD_LIBRARY_PATH=$(LLAMA_LIB_DIR)"; \
		fi; \
	else \
		echo "ERROR: libllama not detected. Set LLAMA_DIR or LLAMA_LIB_DIR."; \
		exit 1; \
	fi

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "Targets:"
	@echo "  spec  - run specs with Metal bridge linked"
	@echo "  build - compile library entrypoint"
	@echo "  drama3_studio - build the local native Metal voice studio"
	@echo "  spec_cpu  - run specs in CPU-only mode"
	@echo "  build_cpu - build in CPU-only mode"
	@echo "  profile_nomic - run native Metal stage profiler for nomic GGUF"
	@echo "  profile_nomic_layers - run per-layer native Metal profiler for nomic GGUF"
	@echo "  profile_nomic_vs_llama - compare native Metal embeddings against llama.cpp"
	@echo "  benchmark_qwen_prefill_vs_llama_same_token - build same-token Qwen prefill benchmark"
	@echo "  benchmark_qwen_decode_vs_llama_same_token - build same-token forced Qwen decode benchmark"
	@echo "  llama - build llama.cpp shared library (requires LLAMA_DIR)"
	@echo "  llama_env - print env vars for libllama discovery"
	@echo "  clean - remove build artifacts"
