#include <metal_stdlib>
using namespace metal;

// Candidate index 0 is EOS; the remaining indices map onto the semantic token
// interval. The selected token and repeat-window mutation stay on the device.
kernel void dr3_choose_main(
    device const float* first_index [[buffer(0)]],
    device const float* retry_index [[buffer(1)]],
    device float* window [[buffer(2)]],
    device float* selected_token [[buffer(3)]],
    constant ulong* p [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;

    int eos = int(p[0]);
    int semantic_begin = int(p[1]);
    int semantic_end = int(p[2]);
    uint window_size = uint(p[3]);
    uint window_cursor = uint(p[4]);
    uint vocab_size = uint(p[5]);

    if (!isfinite(first_index[0]) || !isfinite(retry_index[0])) {
        selected_token[0] = NAN;
        return;
    }
    int first = int(first_index[0]);
    int retry = int(retry_index[0]);
    // The head contains EOS plus vocab_size semantic candidates, so the last
    // valid candidate index is vocab_size and maps to semantic_end.
    if (first < 0 || first > int(vocab_size) || retry < 0 || retry > int(vocab_size) ||
        window_size == 0 || window_size > 16 || window_cursor >= window_size) {
        selected_token[0] = NAN;
        return;
    }

    int first_token = first == 0 ? eos : semantic_begin + first - 1;
    int retry_token = retry == 0 ? eos : semantic_begin + retry - 1;
    bool repeated = false;
    if (first_token >= semantic_begin && first_token <= semantic_end) {
        for (uint i = 0; i < window_size; ++i) {
            if (int(window[i]) == first_token) {
                repeated = true;
                break;
            }
        }
    }

    int selected = repeated ? retry_token : first_token;
    if (selected != eos && selected >= semantic_begin && selected <= semantic_end) {
        window[window_cursor] = float(selected);
    }
    selected_token[0] = float(selected);
}
