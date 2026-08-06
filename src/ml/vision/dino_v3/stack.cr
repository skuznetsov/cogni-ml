# Bounded synthetic CPU composition of the admitted one-block DINOv3 contract.
#
# This is intentionally a stack, not a real DINOv3 encoder: it accepts at most
# four compatible synthetic blocks, keeps runtime and serialized layer paths
# separate, and never loads a checkpoint or constructs model-scale state.

require "./block"
require "./config"

module ML::Vision::DinoV3
  class StackError < BlockError
  end

  class StackBudgetError < StackError
  end

  class BlockStackCPU
    MAX_LAYERS = 4_i32

    @layers : Array(BlockCPU)
    @path_adapter : LayerPathAdapter

    getter runtime_path : String

    def initialize(
      layers : Array(BlockCPU),
      *,
      runtime_path : String = LayerPathAdapter::PINNED_RUNTIME_PATH,
    )
      if layers.empty?
        raise StackError.new("DINOv3 block stack requires at least one layer")
      end
      if layers.size > MAX_LAYERS
        raise StackBudgetError.new(
          "DINOv3 block stack exceeds maximum #{MAX_LAYERS} layers"
        )
      end

      adapter = LayerPathAdapter.new(runtime_path)
      reference = layers.first.config
      layers.each_with_index do |layer, index|
        unless compatible_config?(reference, layer.config)
          raise StackError.new(
            "DINOv3 block #{index} is not compatible with the stack configuration"
          )
        end
      end

      @layers = layers.dup
      @runtime_path = adapter.runtime_path
      @path_adapter = adapter
    end

    def depth : Int32
      @layers.size.to_i32
    end

    def layers : Array(BlockCPU)
      @layers.dup
    end

    def serialized_state_dict_prefix(layer_index : Int32) : String
      validate_layer_index!(layer_index)
      @path_adapter.serialized_state_dict_prefix(layer_index)
    end

    def layer_index_from_serialized_prefix(prefix : String) : Int32
      layer_index = @path_adapter.layer_index_from_serialized_prefix(prefix)
      validate_layer_index!(layer_index)
      layer_index
    end

    # Sequentially feeds each block's residual output to the next block and
    # applies the extractor final LayerNorm only to the final block output.
    def forward(input : Tensor, rope_cos : Tensor, rope_sin : Tensor) : Tensor
      current = input
      @layers.each_with_index do |layer, index|
        trace = layer.forward_with_trace(current, rope_cos, rope_sin)
        return trace.extractor_final if index == @layers.size - 1
        current = trace.block_output
      end
      raise StackError.new("DINOv3 block stack has no executable layers")
    end

    private def validate_layer_index!(layer_index : Int32) : Nil
      unless 0 <= layer_index < depth
        raise StackError.new(
          "DINOv3 layer index #{layer_index} is outside stack depth #{depth}"
        )
      end
    end

    private def compatible_config?(left : BlockConfig, right : BlockConfig) : Bool
      left.hidden_size == right.hidden_size &&
        left.intermediate_size == right.intermediate_size &&
        left.num_attention_heads == right.num_attention_heads &&
        left.num_register_tokens == right.num_register_tokens &&
        left.layer_norm_eps == right.layer_norm_eps &&
        left.hidden_act == right.hidden_act &&
        left.query_bias == right.query_bias &&
        left.key_bias == right.key_bias &&
        left.value_bias == right.value_bias &&
        left.proj_bias == right.proj_bias &&
        left.mlp_bias == right.mlp_bias &&
        left.attention_dropout == right.attention_dropout &&
        left.drop_path_rate == right.drop_path_rate &&
        left.use_gated_mlp == right.use_gated_mlp &&
        left.attention_backend == right.attention_backend &&
        left.training == right.training &&
        left.extractor_final_layer_norm_eps == right.extractor_final_layer_norm_eps &&
        left.head_dim == right.head_dim
    end
  end
end
