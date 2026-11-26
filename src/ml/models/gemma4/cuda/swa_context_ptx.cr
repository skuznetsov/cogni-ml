# Reusable CUDA PTX builders for CogniGemma SWA attention-context probes.
#
# The kernels are derived from the Qwen cache-backed attention probes, but are
# rewritten fail-closed for Gemma4's ungated SWA semantics and window-relative
# attention start.

module ML::CUDA::Gemma4SWAContextPTX
  FULL_ATTN_PTX               = {{ read_file("#{__DIR__}/../../../cuda/kernels/shared/fullattn_post_probe.ptx") }}
  GATED_ATTENTION_START       = "// Correctness-first gated GQA attention decode over a resident KV cache."
  GATED_ATTENTION_END         = "// Split-K long-context GQA attention decode, stage 1."
  GATED_ATTENTION_START_INDEX = FULL_ATTN_PTX.index(GATED_ATTENTION_START) || raise "gated attention kernel start not found"
  GATED_ATTENTION_END_INDEX   = FULL_ATTN_PTX.index(GATED_ATTENTION_END) || raise "gated attention kernel end not found"
  GATED_ATTENTION             = FULL_ATTN_PTX[GATED_ATTENTION_START_INDEX, GATED_ATTENTION_END_INDEX - GATED_ATTENTION_START_INDEX]
  UNGATED_ATTENTION           = GATED_ATTENTION
    .gsub("gated GQA attention decode", "ungated GQA attention decode")
    .gsub("full_attn_decode_cache_probe", "gemma4_ungated_attn_decode_cache_probe")
    .gsub("    .param .u64 gate_in,\n", "")
    .gsub("    .param .u32 max_seq,\n    .param .f32 scale", "    .param .u32 max_seq,\n    .param .u32 window_size,\n    .param .f32 scale")
    .gsub("    .reg .pred %p<9>;", "    .reg .pred %p<10>;")
    .gsub("    ld.param.u64 %rd2, [gate_in];\n", "")
    .gsub("    ld.param.u32 %r6, [max_seq];\n    ld.param.f32 %f1, [scale];", "    ld.param.u32 %r6, [max_seq];\n    ld.param.u32 %r40, [window_size];\n    ld.param.f32 %f1, [scale];")
    .gsub("    mul.lo.u32 %r16, %r15, %r3; // q/gate/out base", "    mul.lo.u32 %r16, %r15, %r3; // q/out base")
    .gsub("    add.u32 %r11, %r10, 1;      // cache_len = pos + 1", "    add.u32 %r11, %r10, 1;      // cache_len = pos + 1\n    mov.u32 %r41, 0;            // SWA attention start\n    setp.eq.u32 %p9, %r40, 0;\n    @%p9 bra A_START_DONE;\n    setp.lt.u32 %p9, %r10, %r40;\n    @%p9 bra A_START_DONE;\n    sub.u32 %r41, %r10, %r40;\n    add.u32 %r41, %r41, 1;\nA_START_DONE:")
    .gsub("    mov.u32 %r18, 0;            // p", "    mov.u32 %r18, %r41;         // p")
    .gsub("    mov.u32 %r25, 0;", "    mov.u32 %r25, %r41;")
    .gsub("    mov.u32 %r28, 0;", "    mov.u32 %r28, %r41;")
    .gsub(/A_GATE_WRITE:.*?st\.global\.f32 \[\%rd21\], \%f26;/m, %(A_GATE_WRITE:
    mul.rn.f32 %f17, %f14, %f13;
    add.u32 %r33, %r16, %r27;
    mul.wide.u32 %rd19, %r33, 4;
    add.s64 %rd21, %rd6, %rd19;
    st.global.f32 [%rd21], %f17;))
  raise "ungated attention rewrite failed" unless UNGATED_ATTENTION.includes?("gemma4_ungated_attn_decode_cache_probe") && !UNGATED_ATTENTION.includes?("gate_in") && !UNGATED_ATTENTION.includes?("%f26")

  SPLITK_ATTENTION_START       = "// Split-K long-context GQA attention decode, stage 1."
  SPLITK_ATTENTION_END         = "// Parallel exact GQA attention decode over a resident KV cache."
  SPLITK_ATTENTION_START_INDEX = FULL_ATTN_PTX.index(SPLITK_ATTENTION_START) || raise "split-K attention kernel start not found"
  SPLITK_ATTENTION_END_INDEX   = FULL_ATTN_PTX.index(SPLITK_ATTENTION_END) || raise "split-K attention kernel end not found"
  SPLITK_ATTENTION             = FULL_ATTN_PTX[SPLITK_ATTENTION_START_INDEX, SPLITK_ATTENTION_END_INDEX - SPLITK_ATTENTION_START_INDEX]
  UNGATED_SPLITK_ATTENTION     = SPLITK_ATTENTION
    .gsub("full_attn_decode_cache_splitk_part_probe", "gemma4_swa_ungated_attn_splitk_part_probe")
    .gsub("full_attn_decode_cache_splitk_reduce_probe", "gemma4_swa_ungated_attn_splitk_reduce_probe")
    .gsub("    .param .f32 scale,\n    .param .u32 chunk_size,", "    .param .u32 window_size,\n    .param .f32 scale,\n    .param .u32 chunk_size,")
    .gsub("    ld.param.u32 %r6, [max_seq];\n    ld.param.f32 %f1, [scale];", "    ld.param.u32 %r6, [max_seq];\n    ld.param.u32 %r55, [window_size];\n    ld.param.f32 %f1, [scale];")
    .gsub("    add.u32 %r12, %r11, 1;      // cache_len", "    add.u32 %r12, %r11, 1;      // cache_len\n    mov.u32 %r56, 0;            // SWA attention start\n    setp.eq.u32 %p30, %r55, 0;\n    @%p30 bra SK_START_DONE;\n    setp.lt.u32 %p30, %r11, %r55;\n    @%p30 bra SK_START_DONE;\n    sub.u32 %r56, %r11, %r55;\n    add.u32 %r56, %r56, 1;\nSK_START_DONE:")
    .gsub("    mul.lo.u32 %r53, %r52, %r50; // chunk_start", "    mul.lo.u32 %r53, %r52, %r50; // chunk_start\n    add.u32 %r53, %r53, %r56;    // window-relative chunk_start")
    .gsub("stable log-sum-exp and applies the attention gate.", "stable log-sum-exp without an attention gate.")
    .gsub("    .param .u64 gate_in,\n", "")
    .gsub("    ld.param.u64 %rd1, [gate_in];\n", "")
    .gsub("    mul.lo.u32 %r14, %r12, %r2; // gate/out base", "    mul.lo.u32 %r14, %r12, %r2; // out base")
    .gsub(/SKR_GATE_WRITE:.*?st\.global\.f32 \[\%rd30\], \%f33;/m, %(SKR_GATE_WRITE:
    mul.rn.f32 %f24, %f17, %f16;
    add.u32 %r28, %r14, %r23;
    mul.wide.u32 %rd28, %r28, 4;
    add.s64 %rd30, %rd5, %rd28;
    st.global.f32 [%rd30], %f24;))
  raise "ungated split-K rewrite failed" unless UNGATED_SPLITK_ATTENTION.includes?("gemma4_swa_ungated_attn_splitk_part_probe") && UNGATED_SPLITK_ATTENTION.includes?("window_size") && !UNGATED_SPLITK_ATTENTION.includes?("gate_in") && !UNGATED_SPLITK_ATTENTION.includes?("%f33")

  ATTN_PTX = (<<-PTX
  .version 8.0
  .target sm_80
  .address_size 64

  #{UNGATED_ATTENTION}

  #{UNGATED_SPLITK_ATTENTION}
  PTX

    ).split('\n').map(&.lstrip).join('\n')
  KV_WRITE_PTX = {{ read_file("#{__DIR__}/../../../cuda/kernels/gemma4/kv_cache_write_probe.ptx") }}
end
