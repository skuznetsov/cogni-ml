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
	$(dir $(wildcard $(LLAMA_BUILD)/src/libllama.*)) \
	$(dir $(wildcard $(LLAMA_BUILD)/lib/libllama.*)) \
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

LINK_FLAGS := -framework Metal -framework Foundation -lc++

.PHONY: all spec build spec_cpu build_cpu llama llama_env profile_nomic profile_nomic_layers profile_nomic_vs_llama clean help

all: spec

$(BUILD_SENTINEL):
	@mkdir -p $(BUILD_DIR)
	@touch $(BUILD_SENTINEL)

$(BRIDGE_OBJ): $(OBJC_SOURCES) | $(BUILD_SENTINEL)
	clang++ -c $(OBJC_SOURCES) -o $(BRIDGE_OBJ) \
		-std=c++17 -fobjc-arc -fPIC

spec: $(BRIDGE_OBJ)
	$(CRYSTAL) spec \
		--link-flags="$(shell pwd)/$(BRIDGE_OBJ) $(LINK_FLAGS)"

build: $(BRIDGE_OBJ)
	$(CRYSTAL) build src/ml.cr \
		--link-flags="$(shell pwd)/$(BRIDGE_OBJ) $(LINK_FLAGS)"

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

spec_cpu:
	$(CRYSTAL) spec -Dcpu_only

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
	@echo "  spec_cpu  - run specs in CPU-only mode"
	@echo "  build_cpu - build in CPU-only mode"
	@echo "  profile_nomic - run native Metal stage profiler for nomic GGUF"
	@echo "  profile_nomic_layers - run per-layer native Metal profiler for nomic GGUF"
	@echo "  profile_nomic_vs_llama - compare native Metal embeddings against llama.cpp"
	@echo "  llama - build llama.cpp shared library (requires LLAMA_DIR)"
	@echo "  llama_env - print env vars for libllama discovery"
	@echo "  clean - remove build artifacts"
