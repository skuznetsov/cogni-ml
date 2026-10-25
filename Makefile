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
HY_NOMAS_COMPACT_ATTENTION_OBJ := $(BUILD_DIR)/hy_nomas_attention_compact.o
QWEN25_MPS_OBJ := $(BUILD_DIR)/qwen25_mps_bridge.o
QWEN25_PREFIX_SOURCES := $(wildcard src/ml/gguf/*.cr src/ml/core/*.cr src/ml/metal/*.cr src/ml/video/*.cr src/ml/video/kernels/*.metal)
WAN_MPS_OBJS := $(WAN_T5_MPS_OBJ) $(WAN_T5_RMS_MPS_OBJ) $(WAN_T5_ATTENTION_MPS_OBJ) $(WAN_ATTENTION_MPS_OBJ) $(WAN_MPS_CHAIN_OBJ) $(WAN_STAGE_IO_OBJ)
WAN_SPEC_OBJS := $(if $(IS_DARWIN),$(WAN_MPS_OBJS) $(HY_NOMAS_MPS_OBJ) $(HY_NOMAS_COMPACT_ATTENTION_OBJ) $(QWEN25_MPS_OBJ),)
WAN_SPEC_LINK_FLAGS := $(if $(IS_DARWIN),-framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -fuse-ld=/usr/bin/ld,)
WAN_BENCHMARK_SOURCES := $(wildcard src/ml/video/*.cr src/ml/metal/*.cr src/ml/gguf/*.cr src/ml/*.cr) spec/support/video_allocation_counter.cr

LINK_FLAGS := -framework Metal -framework Foundation -lc++

.PHONY: all spec_video_staging spec_wan spec_hy_nomas spec_hy_nomas_residency spec_hy_nomas_bf16_cache spec_qwen25 hy_nomas_block_benchmark hy_nomas_serial_benchmark hy_nomas_pretrained_serial_probe hy_nomas_bf16_cache_probe hy_nomas_bf16_serial_probe hy_nomas_residency_plan spec build spec_cpu build_cpu drama3_studio llama llama_env profile_nomic profile_nomic_layers profile_nomic_vs_llama benchmark_qwen_prefill_vs_llama_same_token benchmark_qwen_decode_vs_llama_same_token q4k_ref clean help

all: spec

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

$(HY_NOMAS_COMPACT_ATTENTION_OBJ): src/ml/video/hy_nomas_attention_compact_bridge.mm | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "Hy-Nomas compact MPS attention requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

$(QWEN25_MPS_OBJ): src/ml/gguf/qwen25_mps_bridge.mm | $(BUILD_SENTINEL)
	@test "$(IS_DARWIN)" = Darwin || (echo "Qwen2.5-VL MPS requires macOS"; exit 1)
	clang++ -c $< -o $@ -std=c++17 -fobjc-arc -fPIC

$(WAN_MPS_CHAIN_OBJ): src/ml/video/metal_mps_command_chain.mm | $(BUILD_SENTINEL)
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
