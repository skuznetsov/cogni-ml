require "./qwen_image21_gguf"
require "./qwen_image21_transformer"
{% unless flag?(:cpu_only) %}
  require "./qwen35_metal"
{% end %}

# mmap-backed Qwen-Image 2.1 DiT weight loader.
#
# QuantWeight byte slices point directly into the GGUF mapping. The mapping is
# therefore owned by this container and remains valid until `close` returns.
module ML::GGUF
  class QwenImage21Weights
    HIDDEN_DIM       =  4096
    HEADS            =    32
    HEAD_DIM         =   128
    INTERMEDIATE_DIM = 12288
    INPUT_DIM        =    64
    TIME_EMBED_DIM   =   256

    getter inventory : QwenImage21GGUFInventory
    getter block_config : QwenImage21BlockConfig
    getter gguf_file_type : Int64?

    getter img_in : QuantWeight
    getter modulation : QuantWeight
    getter norm_out_linear : QuantWeight
    getter proj_out : QuantWeight
    getter timestep_linear_1 : QuantWeight
    getter timestep_linear_2 : QuantWeight
    getter text_in_layer : QuantWeight
    getter text_out_layer : QuantWeight
    getter text_norm : Array(Float32)
    getter layers : Array(QwenImage21BlockWeights)

    @gguf : GGUFFile
    @closed : Bool
    @close_mutex : Mutex
    {% unless flag?(:cpu_only) %}
      @mmap_base : Pointer(UInt8)?
    {% end %}

    def initialize(@gguf : GGUFFile, @inventory : QwenImage21GGUFInventory)
      unless @inventory.reader_compatible?
        raise ArgumentError.new(
          "Qwen-Image 2.1 GGUF contains unsupported tensor types: #{@inventory.unsupported_type_labels.join(", ")}"
        )
      end
      @inventory.ensure_tensor_data_complete!(@gguf.data_offset, File.size(@gguf.path))

      @closed = false
      @close_mutex = Mutex.new
      @gguf_file_type = @gguf.get_int("general.file_type")
      {% unless flag?(:cpu_only) %}
        @mmap_base = nil
      {% end %}

      @block_config = QwenImage21BlockConfig.new(
        hidden_dim: HIDDEN_DIM,
        heads: HEADS,
        head_dim: HEAD_DIM,
        intermediate_dim: INTERMEDIATE_DIM,
        axes_dims: StaticArray[16, 56, 56],
      )

      @img_in = load_projection("img_in.weight", HIDDEN_DIM, INPUT_DIM)
      @modulation = load_projection("modulation.1.weight", 4 * HIDDEN_DIM, HIDDEN_DIM)
      @norm_out_linear = load_projection("norm_out.linear.weight", HIDDEN_DIM, HIDDEN_DIM)
      @proj_out = load_projection("proj_out.weight", INPUT_DIM, HIDDEN_DIM)
      @timestep_linear_1 = load_projection(
        "time_text_embed.timestep_embedder.linear_1.weight", HIDDEN_DIM, TIME_EMBED_DIM
      )
      @timestep_linear_2 = load_projection(
        "time_text_embed.timestep_embedder.linear_2.weight", HIDDEN_DIM, HIDDEN_DIM
      )
      @text_in_layer = load_projection("txt_in.in_layer.weight", HIDDEN_DIM, HIDDEN_DIM)
      @text_out_layer = load_projection("txt_in.out_layer.weight", HIDDEN_DIM, HIDDEN_DIM)
      @text_norm = load_vector("txt_in.text_norm.weight", HIDDEN_DIM)

      @layers = Array(QwenImage21BlockWeights).new(QwenImage21GGUFInventory::BLOCK_COUNT) do |layer|
        load_layer(layer)
      end

      # Reuse the established Qwen 3.5 whole-file Metal registration. This
      # creates one no-copy MetalBuffer for the mmap instead of uploading each
      # projection independently.
      {% unless flag?(:cpu_only) %}
        if Qwen35Metal.available?
          if region = @gguf.mmap_region
            base, size = region
            Qwen35Metal.register_mmap(base, size)
            @mmap_base = base
          end
        end
      {% end %}
    end

    def self.from_gguf(path : String) : self
      gguf = GGUFFile.new(path)
      begin
        inventory = QwenImage21GGUFInventory.from_file(gguf)
        new(gguf, inventory)
      rescue ex
        gguf.close
        raise ex
      end
    end

    def transformer_config : QwenImage21TransformerConfig
      QwenImage21TransformerConfig.new(
        input_dim: INPUT_DIM,
        output_dim: INPUT_DIM,
        context_dim: HIDDEN_DIM,
        time_input_dim: TIME_EMBED_DIM,
        block: @block_config,
        causal_condition: true,
      )
    end

    # Lightweight view over the mmap-backed tensors. The returned object does
    # not own the mapping and must not outlive this loader.
    def transformer_weights : QwenImage21TransformerWeights
      QwenImage21TransformerWeights.new(
        img_in: @img_in,
        modulation: @modulation,
        norm_out_linear: @norm_out_linear,
        proj_out: @proj_out,
        timestep_linear_1: @timestep_linear_1,
        timestep_linear_2: @timestep_linear_2,
        text_in_layer: @text_in_layer,
        text_out_layer: @text_out_layer,
        text_norm: @text_norm,
        layers: @layers,
      )
    end

    # Callers must quiesce in-flight Metal work before closing the loader.
    def close : Nil
      @close_mutex.synchronize do
        return if @closed

        {% unless flag?(:cpu_only) %}
          if base = @mmap_base
            Qwen35Metal.unregister_mmap(base)
            @mmap_base = nil
          end
        {% end %}
        @gguf.close
        @closed = true
      end
    end

    def finalize
      close
    end

    private def load_layer(layer : Int32) : QwenImage21BlockWeights
      prefix = "transformer_blocks.#{layer}"
      QwenImage21BlockWeights.new(
        to_q: load_projection("#{prefix}.attn.to_q.weight", HIDDEN_DIM, HIDDEN_DIM),
        to_k: load_projection("#{prefix}.attn.to_k.weight", HIDDEN_DIM, HIDDEN_DIM),
        to_v: load_projection("#{prefix}.attn.to_v.weight", HIDDEN_DIM, HIDDEN_DIM),
        to_out: load_projection("#{prefix}.attn.to_out.0.weight", HIDDEN_DIM, HIDDEN_DIM),
        norm_q: load_vector("#{prefix}.attn.norm_q.weight", HEAD_DIM),
        norm_k: load_vector("#{prefix}.attn.norm_k.weight", HEAD_DIM),
        gate_up: load_projection("#{prefix}.img_mlp.gate_up.weight", 2 * INTERMEDIATE_DIM, HIDDEN_DIM),
        mlp_out: load_projection("#{prefix}.img_mlp.out.weight", HIDDEN_DIM, INTERMEDIATE_DIM),
      )
    end

    private def load_projection(name : String, expected_out : Int32, expected_in : Int32) : QuantWeight
      info = @gguf.tensor(name) || raise ArgumentError.new("missing tensor #{name}")
      out_dim, in_dim = @inventory.projection_dims(name)
      unless out_dim == expected_out && in_dim == expected_in
        raise ArgumentError.new(
          "tensor #{name} has projection shape #{out_dim}x#{in_dim}, expected #{expected_out}x#{expected_in}"
        )
      end
      QuantWeight.new(
        @gguf.read_tensor_raw(info),
        info.type,
        out_dim,
        in_dim,
        "qwen_image21:#{name}",
      )
    end

    private def load_vector(name : String, expected_size : Int32) : Array(Float32)
      info = @gguf.tensor(name) || raise ArgumentError.new("missing tensor #{name}")
      unless info.dims == [expected_size.to_i64]
        raise ArgumentError.new("tensor #{name} has shape #{info.dims}, expected [#{expected_size}]")
      end
      @gguf.read_tensor_f32(info)
    end
  end
end
