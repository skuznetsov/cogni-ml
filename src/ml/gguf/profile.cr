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
end
