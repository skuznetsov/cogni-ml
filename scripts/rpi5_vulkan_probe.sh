#!/usr/bin/env bash
set -euo pipefail

ROOT="${COGNI_VULKAN_ROOT:-$HOME/cogni-vulkan-runtime/root}"
ICD="${COGNI_VULKAN_ICD:-$HOME/cogni-vulkan-runtime/icd/broadcom_icd.user.json}"
SRC_DIR="${1:-$HOME/cogni-ml-vulkan-probe}"

export VK_ICD_FILENAMES="$ICD"
export LD_LIBRARY_PATH="$ROOT/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"

"$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_matmul.comp" -o "$SRC_DIR/rpi5_matmul.spv"
"$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q4k_matvec.comp" -o "$SRC_DIR/rpi5_q4k_matvec.spv"
"$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q4k_matvec_wg.comp" -o "$SRC_DIR/rpi5_q4k_matvec_wg.spv"
"$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q4_matvec_pre.comp" -o "$SRC_DIR/rpi5_q4_matvec_pre.spv"
"$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q8_matvec.comp" -o "$SRC_DIR/rpi5_q8_matvec.spv"
"$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q8_matvec_rg4.comp" -o "$SRC_DIR/rpi5_q8_matvec_rg4.spv"
"$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q8_matvec_pre.comp" -o "$SRC_DIR/rpi5_q8_matvec_pre.spv"
if [[ -f "$SRC_DIR/rpi5_swiglu.comp" ]]; then
  "$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_swiglu.comp" -o "$SRC_DIR/rpi5_swiglu.spv"
fi
if [[ -f "$SRC_DIR/rpi5_q4_matvec_pre_r2.comp" ]]; then
  "$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q4_matvec_pre_r2.comp" -o "$SRC_DIR/rpi5_q4_matvec_pre_r2.spv"
fi
if [[ -f "$SRC_DIR/rpi5_q4_matvec_pre_dot.comp" ]]; then
  "$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q4_matvec_pre_dot.comp" -o "$SRC_DIR/rpi5_q4_matvec_pre_dot.spv"
fi
if [[ -f "$SRC_DIR/rpi5_q4_matvec_pre_l256.comp" ]]; then
  "$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q4_matvec_pre_l256.comp" -o "$SRC_DIR/rpi5_q4_matvec_pre_l256.spv"
fi
if [[ -f "$SRC_DIR/rpi5_q4_matvec_pre_l256_g2048.comp" ]]; then
  "$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q4_matvec_pre_l256_g2048.comp" -o "$SRC_DIR/rpi5_q4_matvec_pre_l256_g2048.spv"
fi
if [[ -f "$SRC_DIR/rpi5_q4_matvec_pre_l256_g6144.comp" ]]; then
  "$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q4_matvec_pre_l256_g6144.comp" -o "$SRC_DIR/rpi5_q4_matvec_pre_l256_g6144.spv"
fi
if [[ -f "$SRC_DIR/rpi5_q4_matvec_pre_l256_g2048_noguard.comp" ]]; then
  "$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q4_matvec_pre_l256_g2048_noguard.comp" -o "$SRC_DIR/rpi5_q4_matvec_pre_l256_g2048_noguard.spv"
fi
if [[ -f "$SRC_DIR/rpi5_q4_matvec_pre_l256_g6144_noguard.comp" ]]; then
  "$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q4_matvec_pre_l256_g6144_noguard.comp" -o "$SRC_DIR/rpi5_q4_matvec_pre_l256_g6144_noguard.spv"
fi
if [[ -f "$SRC_DIR/rpi5_q4_dual_matvec_pre_l256_g2048_noguard.comp" ]]; then
  "$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q4_dual_matvec_pre_l256_g2048_noguard.comp" -o "$SRC_DIR/rpi5_q4_dual_matvec_pre_l256_g2048_noguard.spv"
fi
if [[ -f "$SRC_DIR/rpi5_q4_matvec_pre_sx_l256_g2048_noguard.comp" ]]; then
  "$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q4_matvec_pre_sx_l256_g2048_noguard.comp" -o "$SRC_DIR/rpi5_q4_matvec_pre_sx_l256_g2048_noguard.spv"
fi
if [[ -f "$SRC_DIR/rpi5_q4_matvec_pre_sx_l256_g6144_noguard.comp" ]]; then
  "$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q4_matvec_pre_sx_l256_g6144_noguard.comp" -o "$SRC_DIR/rpi5_q4_matvec_pre_sx_l256_g6144_noguard.spv"
fi
if [[ -f "$SRC_DIR/rpi5_q4_matmat_pre_l256_g2048_noguard.comp" ]]; then
  "$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q4_matmat_pre_l256_g2048_noguard.comp" -o "$SRC_DIR/rpi5_q4_matmat_pre_l256_g2048_noguard.spv"
fi
if [[ -f "$SRC_DIR/rpi5_q4_matmat_pre_l256_g6144_noguard.comp" ]]; then
  "$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q4_matmat_pre_l256_g6144_noguard.comp" -o "$SRC_DIR/rpi5_q4_matmat_pre_l256_g6144_noguard.spv"
fi
if [[ -f "$SRC_DIR/rpi5_q4_matmat_t2_pre_l256_g2048_noguard.comp" ]]; then
  "$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q4_matmat_t2_pre_l256_g2048_noguard.comp" -o "$SRC_DIR/rpi5_q4_matmat_t2_pre_l256_g2048_noguard.spv"
fi
if [[ -f "$SRC_DIR/rpi5_q4_matmat_t2_pre_l256_g6144_noguard.comp" ]]; then
  "$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q4_matmat_t2_pre_l256_g6144_noguard.comp" -o "$SRC_DIR/rpi5_q4_matmat_t2_pre_l256_g6144_noguard.spv"
fi
if [[ -f "$SRC_DIR/rpi5_q4_matvec_inflated_l256_g2048_noguard.comp" ]]; then
  "$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q4_matvec_inflated_l256_g2048_noguard.comp" -o "$SRC_DIR/rpi5_q4_matvec_inflated_l256_g2048_noguard.spv"
fi
if [[ -f "$SRC_DIR/rpi5_q4_matvec_inflated_l256_g6144_noguard.comp" ]]; then
  "$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q4_matvec_inflated_l256_g6144_noguard.comp" -o "$SRC_DIR/rpi5_q4_matvec_inflated_l256_g6144_noguard.spv"
fi
if [[ -f "$SRC_DIR/rpi5_q8_matvec_pre_l256.comp" ]]; then
  "$ROOT/usr/bin/glslangValidator" -V "$SRC_DIR/rpi5_q8_matvec_pre_l256.comp" -o "$SRC_DIR/rpi5_q8_matvec_pre_l256.spv"
fi
gcc -O3 \
  -I"$ROOT/usr/include" \
  "$SRC_DIR/rpi5_vulkan_matmul_probe.c" \
  /usr/lib/aarch64-linux-gnu/libvulkan.so.1 -lm \
  -Wl,-rpath,"$ROOT/usr/lib/aarch64-linux-gnu" \
  -o "$SRC_DIR/rpi5_vulkan_matmul_probe"
gcc -O3 \
  -I"$ROOT/usr/include" \
  "$SRC_DIR/rpi5_vulkan_q4k_probe.c" \
  /usr/lib/aarch64-linux-gnu/libvulkan.so.1 -lm \
  -Wl,-rpath,"$ROOT/usr/lib/aarch64-linux-gnu" \
  -o "$SRC_DIR/rpi5_vulkan_q4k_probe"
gcc -O3 \
  -I"$ROOT/usr/include" \
  "$SRC_DIR/rpi5_vulkan_q8_probe.c" \
  /usr/lib/aarch64-linux-gnu/libvulkan.so.1 -lm \
  -Wl,-rpath,"$ROOT/usr/lib/aarch64-linux-gnu" \
  -o "$SRC_DIR/rpi5_vulkan_q8_probe"
if [[ -f "$SRC_DIR/rpi5_vulkan_ffn_probe.c" ]]; then
  gcc -O3 \
    -I"$ROOT/usr/include" \
    "$SRC_DIR/rpi5_vulkan_ffn_probe.c" \
    /usr/lib/aarch64-linux-gnu/libvulkan.so.1 -lm \
    -Wl,-rpath,"$ROOT/usr/lib/aarch64-linux-gnu" \
    -o "$SRC_DIR/rpi5_vulkan_ffn_probe"
fi

"$ROOT/usr/bin/vulkaninfo" --summary | sed -n '/Devices:/,$p'
for size in 64 128 256; do
  "$SRC_DIR/rpi5_vulkan_matmul_probe" "$SRC_DIR/rpi5_matmul.spv" "$size" "$size" "$size" 10
done
for shape in "1024 1024 50" "2048 2048 30" "4096 4096 20"; do
  "$SRC_DIR/rpi5_vulkan_q4k_probe" "$SRC_DIR/rpi5_q4k_matvec.spv" $shape
  "$SRC_DIR/rpi5_vulkan_q4k_probe" "$SRC_DIR/rpi5_q4k_matvec_wg.spv" $shape wg
  "$SRC_DIR/rpi5_vulkan_q4k_probe" "$SRC_DIR/rpi5_q4_matvec_pre.spv" $shape pre
  if [[ -f "$SRC_DIR/rpi5_q4_matvec_pre_l256.spv" ]]; then
    "$SRC_DIR/rpi5_vulkan_q4k_probe" "$SRC_DIR/rpi5_q4_matvec_pre_l256.spv" $shape pre_l256
  fi
  "$SRC_DIR/rpi5_vulkan_q8_probe" "$SRC_DIR/rpi5_q8_matvec.spv" $shape
  "$SRC_DIR/rpi5_vulkan_q8_probe" "$SRC_DIR/rpi5_q8_matvec_rg4.spv" $shape rg4
  "$SRC_DIR/rpi5_vulkan_q8_probe" "$SRC_DIR/rpi5_q8_matvec_pre.spv" $shape pre
  if [[ -f "$SRC_DIR/rpi5_q8_matvec_pre_l256.spv" ]]; then
    "$SRC_DIR/rpi5_vulkan_q8_probe" "$SRC_DIR/rpi5_q8_matvec_pre_l256.spv" $shape pre_l256
  fi
done
