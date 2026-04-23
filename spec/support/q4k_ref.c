// Q4_K dequantization reference helper.
//
// Usage: q4k_ref <raw_in> <n_elements> <floats_out>
// Reads raw Q4_K blocks from <raw_in>, dequantizes via libggml's
// dequantize_row_q4_K, writes little-endian Float32 to <floats_out>.
//
// Build:
//   clang -O2 -o q4k_ref q4k_ref.c \
//     -I$LLAMA_DIR/ggml/src -I$LLAMA_DIR/ggml/include \
//     -L$LLAMA_DIR/build/bin -lggml-base \
//     -Wl,-rpath,$LLAMA_DIR/build/bin

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// Forward declaration - avoid pulling in full ggml-quants.h which needs ggml-common.h
extern void dequantize_row_q4_K(const void *x, float *y, int64_t k);

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <raw_in> <n_elements> <floats_out>\n", argv[0]);
        return 2;
    }
    const char *in_path = argv[1];
    int64_t n = strtoll(argv[2], NULL, 10);
    const char *out_path = argv[3];

    // 144 bytes per 256-element Q4_K block
    if (n % 256 != 0) {
        fprintf(stderr, "n must be multiple of QK_K=256, got %lld\n", (long long)n);
        return 3;
    }
    int64_t nb = n / 256;
    size_t raw_size = (size_t)nb * 144;

    FILE *fi = fopen(in_path, "rb");
    if (!fi) { perror(in_path); return 4; }
    void *raw = malloc(raw_size);
    if (!raw) { perror("malloc raw"); return 5; }
    if (fread(raw, 1, raw_size, fi) != raw_size) {
        fprintf(stderr, "short read: expected %zu bytes\n", raw_size);
        return 6;
    }
    fclose(fi);

    float *out = (float *)malloc(sizeof(float) * (size_t)n);
    if (!out) { perror("malloc out"); return 7; }

    dequantize_row_q4_K(raw, out, n);

    FILE *fo = fopen(out_path, "wb");
    if (!fo) { perror(out_path); return 8; }
    if ((int64_t)fwrite(out, sizeof(float), n, fo) != n) {
        fprintf(stderr, "short write\n");
        return 9;
    }
    fclose(fo);

    free(raw);
    free(out);
    return 0;
}
