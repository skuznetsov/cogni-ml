# LLM (llama.cpp)

The bindings in `src/ml/llm/llama_ffi.cr` expect a shared library named `libllama` from llama.cpp.

## Build llama.cpp

```sh
# Build shared lib in ../llama.cpp (override with LLAMA_DIR)
make llama
```

This produces a shared library in `../llama.cpp/build` (typically `build/src/libllama.dylib`).
If `LLAMA_DIR` is missing but `libllama` is already installed (e.g., via Homebrew), `make llama` will detect it and skip the build.

## Link and run

When building or running a program that uses `ML::LLM`, make sure the library is visible:

```sh
eval "$(make llama_env)"
```

## Example

```sh
crystal run examples/llm_inference.cr -- /path/to/model.gguf "Hello"
```

If you want to embed LLM usage in another app, just `require "ml/llm/llama"` and use `ML::LLM::Model` / `ML::LLM::Generator`.
