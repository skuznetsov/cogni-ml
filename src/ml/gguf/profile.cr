module ML::GGUF
  record MetalEncodeProfile,
    graph_lookup_ms : Float64,
    cmd_setup_ms : Float64,
    token_write_ms : Float64,
    lengths_write_ms : Float64,
    prepass_encode_ms : Float64,
    graph_encode_ms : Float64,
    submit_wait_ms : Float64,
    readback_ms : Float64 do
    def total_ms : Float64
      @graph_lookup_ms + @cmd_setup_ms + @token_write_ms + @lengths_write_ms +
        @prepass_encode_ms + @graph_encode_ms + @submit_wait_ms + @readback_ms
    end

    def +(other : MetalEncodeProfile) : MetalEncodeProfile
      MetalEncodeProfile.new(
        graph_lookup_ms: @graph_lookup_ms + other.graph_lookup_ms,
        cmd_setup_ms: @cmd_setup_ms + other.cmd_setup_ms,
        token_write_ms: @token_write_ms + other.token_write_ms,
        lengths_write_ms: @lengths_write_ms + other.lengths_write_ms,
        prepass_encode_ms: @prepass_encode_ms + other.prepass_encode_ms,
        graph_encode_ms: @graph_encode_ms + other.graph_encode_ms,
        submit_wait_ms: @submit_wait_ms + other.submit_wait_ms,
        readback_ms: @readback_ms + other.readback_ms,
      )
    end

    def self.zero : MetalEncodeProfile
      MetalEncodeProfile.new(
        graph_lookup_ms: 0.0,
        cmd_setup_ms: 0.0,
        token_write_ms: 0.0,
        lengths_write_ms: 0.0,
        prepass_encode_ms: 0.0,
        graph_encode_ms: 0.0,
        submit_wait_ms: 0.0,
        readback_ms: 0.0,
      )
    end
  end

  record EmbedProfile,
    text_count : Int32,
    total_tokens : Int32,
    max_seq_len : Int32,
    tokenize_ms : Float64,
    prepare_ms : Float64,
    reorder_ms : Float64,
    backend : MetalEncodeProfile,
    total_ms : Float64 do
    def cpu_overhead_ms : Float64
      @tokenize_ms + @prepare_ms + @reorder_ms
    end
  end

  record LayerStageProfile,
    layer_index : Int32,
    kind : String,
    attn_proj_ms : Float64,
    attn_core_ms : Float64,
    attn_out_norm_ms : Float64,
    ffn_route_ms : Float64,
    ffn_up_ms : Float64,
    ffn_down_ms : Float64,
    ffn_scatter_norm_ms : Float64 do
    def attention_ms : Float64
      @attn_proj_ms + @attn_core_ms + @attn_out_norm_ms
    end

    def ffn_ms : Float64
      @ffn_route_ms + @ffn_up_ms + @ffn_down_ms + @ffn_scatter_norm_ms
    end

    def total_ms : Float64
      attention_ms + ffn_ms
    end

    def +(other : LayerStageProfile) : LayerStageProfile
      LayerStageProfile.new(
        layer_index: @layer_index,
        kind: @kind,
        attn_proj_ms: @attn_proj_ms + other.attn_proj_ms,
        attn_core_ms: @attn_core_ms + other.attn_core_ms,
        attn_out_norm_ms: @attn_out_norm_ms + other.attn_out_norm_ms,
        ffn_route_ms: @ffn_route_ms + other.ffn_route_ms,
        ffn_up_ms: @ffn_up_ms + other.ffn_up_ms,
        ffn_down_ms: @ffn_down_ms + other.ffn_down_ms,
        ffn_scatter_norm_ms: @ffn_scatter_norm_ms + other.ffn_scatter_norm_ms,
      )
    end
  end

  record LayerEmbedProfile,
    seq_len : Int32,
    tokenize_ms : Float64,
    prepare_ms : Float64,
    prepass_wait_ms : Float64,
    pool_wait_ms : Float64,
    readback_ms : Float64,
    layers : Array(LayerStageProfile) do
    def layer_wait_ms : Float64
      @layers.sum(&.total_ms)
    end

    def total_ms : Float64
      @tokenize_ms + @prepare_ms + @prepass_wait_ms + layer_wait_ms + @pool_wait_ms + @readback_ms
    end
  end
end
