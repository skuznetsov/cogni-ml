# CPU metadata/accounting only. No tensor payload is opened, no Metal device
# is initialized, and no N-block execution support is implied by a byte estimate.
require "digest/sha256"
require "json"
require "set"
require "./hy_nomas_pretrained_serial_probe"

module HyNomasResidencyPlan
  extend self

  HEADER_BYTES  = 252_904
  HEADER_SHA256 = "cbfa83dc81af315b425fd783e0ce42d8ff09df772952ef2a8aff48493e21b200"
  PAYLOAD_BYTES = 34_213_738_368_i64
  SOURCE_BYTES  = 34_213_991_280_i64
  BLOCKS        =                 54
  GROUP_NAMES   = %w(action_modules c2ws_hidden_states_layer1 c2ws_hidden_states_layer2 double_blocks extrinsics_embedding_camctrl final_layer img_in time_in txt_in)

  record Group, tensors : Int32, f32_bytes : Int64

  class Inventory
    getter groups : Hash(String, Group)
    getter tensor_count : Int32
    getter block_count : Int32
    getter f32_bytes : Int64
    getter block_f32_bytes : Int64

    def self.load!(path : String) : self
      File.open(path, "rb") do |io|
        raise ArgumentError.new("pinned header length mismatch") unless io.size == HEADER_BYTES
        raw = io.read_string(HEADER_BYTES)
        from_authenticated_bytes!(raw)
      end
    end

    def self.from_authenticated_bytes!(raw : String) : self
      raise ArgumentError.new("pinned header length mismatch") unless raw.bytesize == HEADER_BYTES
      raise ArgumentError.new("pinned header digest mismatch") unless Digest::SHA256.hexdigest(raw) == HEADER_SHA256
      new(JSON.parse(raw).as_h)
    end

    private def initialize(tensors : Hash(String, JSON::Any))
      @groups = {} of String => Group
      @tensor_count = 0
      @block_count = BLOCKS
      @f32_bytes = 0_i64
      ranges = [] of Tuple(Int64, Int64)
      config = HyNomasResidencyPlan.config(6, 23)
      native = {} of String => Array(Int64)
      BLOCKS.times do |index|
        ML::Video::HyNomasDualStreamBlock.checkpoint_bindings(config,
          block_index: index).each { |binding| native[binding.source] = binding.shape }
      end
      seen_blocks = Set(String).new
      tensors.each do |name, value|
        next if name == "__metadata__"
        category = name.split('.').first
        raise ArgumentError.new("unaccounted checkpoint category: #{category}") unless GROUP_NAMES.includes?(category)
        raise ArgumentError.new("pinned checkpoint tensor must be F32") unless value["dtype"].as_s == "F32"
        shape = value["shape"].as_a.map(&.as_i64)
        raise ArgumentError.new("invalid tensor shape") if shape.empty? || shape.any? { |dim| dim <= 0 }
        bytes = shape.reduce(4_i64) { |count, dim| count * dim }
        offsets = value["data_offsets"].as_a.map(&.as_i64)
        unless offsets.size == 2 && offsets[0] >= 0 && offsets[1] - offsets[0] == bytes && offsets[1] <= PAYLOAD_BYTES
          raise ArgumentError.new("invalid checkpoint extent: #{name}")
        end
        if category == "double_blocks"
          raise ArgumentError.new("checkpoint/native block binding mismatch: #{name}") unless native[name]? == shape
          seen_blocks << name
        end
        ranges << {offsets[0], offsets[1]}
        previous = @groups[category]? || Group.new(0, 0_i64)
        @groups[category] = Group.new(previous.tensors + 1, previous.f32_bytes + bytes)
        @tensor_count += 1
        @f32_bytes += bytes
      end
      cursor = 0_i64
      ranges.sort.each do |first, last|
        raise ArgumentError.new("checkpoint payload has a gap or overlap") unless first == cursor
        cursor = last
      end
      unless @groups.keys.sort == GROUP_NAMES.sort && @tensor_count == 2268 &&
             @f32_bytes == PAYLOAD_BYTES && cursor == PAYLOAD_BYTES &&
             seen_blocks == native.keys.to_set
        raise ArgumentError.new("incomplete pinned checkpoint inventory")
      end
      @block_f32_bytes = @groups["double_blocks"].f32_bytes
    end

    def report
      {header_bytes: HEADER_BYTES, header_sha256: HEADER_SHA256,
       declared_checkpoint_bytes: SOURCE_BYTES, checkpoint_payload_verified: false,
       tensors: @tensor_count, blocks: @block_count, source_f32_bytes: @f32_bytes,
       bf16_parameter_bytes: @f32_bytes // 2,
       groups: @groups.transform_values { |value|
         {tensors: value.tensors, source_f32_bytes: value.f32_bytes,
          bf16_parameter_bytes: value.f32_bytes // 2}
       }}
    end
  end

  record Geometry, scratch_bytes : Int64, output_pair_bytes : Int64,
    weight_expansion_bytes : Int64, attention_stage_bytes : Int64,
    rms_bytes : Int64, input_bytes : Int64, score_tile_bytes : Int64,
    graph_calls : Int32, checkpoint_bytes_per_block : Int64 do
    def workspace_bytes : Int64
      weight_expansion_bytes + attention_stage_bytes + rms_bytes
    end

    # Proposed generalization retaining every block's final output pair.
    # Only block counts 1 and 2 are currently executable by the existing APIs.
    def shared_bytes(blocks : Int32) : Int64
      raise ArgumentError.new("block count must be in 1..54") unless 1 <= blocks <= BLOCKS
      scratch_bytes + (blocks.to_i64 - 1) * output_pair_bytes + workspace_bytes
    end

    def proposed_two_bank_bytes(blocks : Int32) : Int64
      raise ArgumentError.new("two-bank proposal requires 2..54 blocks") unless 2 <= blocks <= BLOCKS
      shared_bytes(2)
    end
  end

  def config(video : Int32, text : Int32) : ML::Video::HyNomasConfig
    ML::Video::HyNomasConfig.new(batches: 1, video_tokens: video,
      text_tokens: text, dim: 2048, heads: 16, head_dim: 128, ffn_dim: 8192)
  end

  def geometry(config : ML::Video::HyNomasConfig, query_chunk : Int32 = 0,
               heads_per_group : Int32 = 0) : Geometry
    config.validate!
    scratch = ML::Video::HyNomasDualStreamBlock.scratch_plan(config)
    scratch_bytes = scratch.sum { |_, count| count.to_i64 * sizeof(Float32) }
    output_pair = config.joint_rows.to_i64 * config.dim * sizeof(Float32)
    expansion = ML::Video::HyNomasMPSWorkspace.required_weight_stage_count(config).to_i64 * sizeof(Float32)
    attention, score, calls = if query_chunk == 0 && heads_per_group == 0
                                {4_i64 * output_pair,
                                 config.batches.to_i64 * config.heads * config.joint_tokens * config.joint_tokens * sizeof(Float32), 1}
                              else
                                compact = ML::Video::HyNomasCompactAttentionMPSPlan.workspace_geometry(
                                  config.batches, config.joint_tokens, config.heads, config.head_dim,
                                  query_chunk, heads_per_group)
                                {compact.resident_bytes, compact.max_score_elements * sizeof(Float32), compact.graph_calls}
                              end
    rms = 2_i64 * config.joint_rows * (config.dim.to_i64 + config.heads) * sizeof(Float32)
    inputs = (config.joint_rows.to_i64 * config.dim +
              (config.effective_img_modulation_rows.to_i64 + config.effective_txt_modulation_rows) * config.dim +
              2_i64 * config.video_tokens * config.head_dim) * sizeof(Float32)
    captures = HyNomasPretrainedSerialProbe::CAPTURE_NAMES.to_set
    checkpoint_bytes = scratch.sum do |name, count|
      captures.includes?(name.sub("hy_nomas.", "")) ? count.to_i64 * sizeof(Float32) : 0_i64
    end
    unless scratch.count { |name, _| captures.includes?(name.sub("hy_nomas.", "")) } == captures.size
      raise ArgumentError.new("diagnostic snapshot schema differs from native scratch")
    end
    Geometry.new(scratch_bytes, output_pair, expansion, attention, rms,
      inputs, score, calls, checkpoint_bytes)
  end

  class Scenario
    getter geometry : Geometry
    getter blocks : Int32
    getter f32_conversion_staging_bytes : Int64

    def initialize(@geometry : Geometry, @blocks : Int32, @f32_conversion_staging_bytes : Int64)
      @geometry.shared_bytes(@blocks)
      unless @f32_conversion_staging_bytes > 0 && @f32_conversion_staging_bytes % 4 == 0
        raise ArgumentError.new("invalid F32 parameter byte count")
      end
    end

    def bf16_weights_bytes : Int64
      @f32_conversion_staging_bytes // 2
    end

    def warm_named_bytes : Int64
      bf16_weights_bytes + @geometry.input_bytes + @geometry.shared_bytes(@blocks)
    end

    def cold_named_bytes : Int64
      warm_named_bytes + @f32_conversion_staging_bytes
    end

    def retained_checkpoint_bytes : Int64
      @blocks.to_i64 * @geometry.checkpoint_bytes_per_block
    end

    # Existing benchmark heuristic: three score tiles plus 256 MiB. It is
    # neither measured nor a bound on opaque/queued MPSGraph allocations.
    def reserve_heuristic_bytes : Int64
      3_i64 * @geometry.score_tile_bytes + 256_i64 * 1024 * 1024
    end

    def estimated_cold_bytes : Int64
      cold_named_bytes + reserve_heuristic_bytes
    end

    def estimated_warm_bytes : Int64
      warm_named_bytes + reserve_heuristic_bytes
    end

    def implemented? : Bool
      @blocks <= 2
    end

    def full_pipeline_accounted? : Bool
      false
    end

    def report(name : String, estimate_budget : Int64)
      {name: name, blocks: @blocks, serial_block_count_implemented: implemented?,
       full_pipeline_accounted: false, runtime_launch_authorized: false,
       retained_distinct_output_pairs: @blocks, diagnostic_snapshots_included: false,
       bf16_parameter_bytes: bf16_weights_bytes, f32_conversion_staging_bytes: @f32_conversion_staging_bytes,
       input_bytes: @geometry.input_bytes, scratch_and_workspace_bytes: @geometry.shared_bytes(@blocks),
       cold_named_bytes: cold_named_bytes, warm_named_bytes: warm_named_bytes,
       score_and_opaque_reserve_heuristic_bytes: reserve_heuristic_bytes,
       estimated_cold_bytes: estimated_cold_bytes, estimated_warm_bytes: estimated_warm_bytes,
       cold_estimate_within_budget: estimated_cold_bytes <= estimate_budget,
       warm_estimate_within_budget: estimated_warm_bytes <= estimate_budget,
       optional_all_checkpoint_snapshot_bytes: retained_checkpoint_bytes,
       proposed_two_bank_saving_bytes: @blocks > 2 ? @geometry.shared_bytes(@blocks) - @geometry.proposed_two_bank_bytes(@blocks) : 0_i64}
    end
  end

  def require_budget!(estimate : Int64, budget : Int64) : Nil
    unless estimate >= 0 && budget > 0 && estimate <= budget
      raise ArgumentError.new("residency estimate #{estimate} exceeds positive budget #{budget}")
    end
  end
end
