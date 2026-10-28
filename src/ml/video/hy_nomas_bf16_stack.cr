# Authenticated all-block BF16 control weights for the native serial DiT path.
# Each block remains resident in its own setup Arena; encode copies the next
# block into one fixed BF16 staging Arena on the caller's command chain.
require "set"
require "./bf16_weight_cache"
require "./hy_nomas_q4_stack"

module ML::Video
  class HyNomasBF16Stack < HyNomasSerialWeightLoader
    BLOCK_COUNT = HyNomasQ4Cache::DIT_BLOCK_COUNT

    private record CopyTensor, target : String, elements : Int32

    getter source_sha256 : String
    getter index_sha256 : String
    getter block_count : Int32 = BLOCK_COUNT
    getter cache_payload_bytes : Int64 = 0_i64
    getter admitted_bytes : Int64 = 0_i64
    getter budget_bytes : Int64
    getter stage_bytes : Int64
    getter reserve_bytes : Int64
    getter max_forwards_per_chain : Int32

    @source_sha256 = ""
    @index_sha256 = ""
    @block_pins = [] of HyNomasQ4StackIndex::BlockPin
    @control_files = [] of CheckpointFile
    @control_plans = [] of WeightPlan
    @copy_tensors = [] of Array(CopyTensor)
    @control_arenas = [] of Arena
    @prepared = [] of PreparedWeights
    @control_buffers = [] of Hash(String, ML::MetalBuffer)
    @stage_arena : Arena? = nil
    @stage_buffers = {} of String => ML::MetalBuffer
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @cursor = HyNomasFullPassCursor.new(BLOCK_COUNT)
    @loaded = false
    @released = false
    @poisoned = false

    # Admission authenticates the index, all 54 source cache files and each
    # exact BF16 control file before this owner creates any Metal allocation.
    def initialize(index_path : String, expected_index_sha256 : String,
                   expected_source_sha256 : String, config : HyNomasConfig,
                   @budget_bytes : Int64, *, @stage_bytes : Int64,
                   @reserve_bytes : Int64 = 0_i64,
                   @max_forwards_per_chain : Int32 = 1)
      validate_options!
      admission = HyNomasQ4StackIndex::Admission.new(index_path,
        expected_index_sha256, expected_source_sha256, config)
      initialize_from_admission(admission, expected_index_sha256,
        expected_source_sha256, config)
    end

    def initialize(admission : HyNomasQ4StackIndex::Admission,
                   expected_index_sha256 : String,
                   expected_source_sha256 : String, config : HyNomasConfig,
                   @budget_bytes : Int64, *, @stage_bytes : Int64,
                   @reserve_bytes : Int64 = 0_i64,
                   @max_forwards_per_chain : Int32 = 1)
      validate_options!
      initialize_from_admission(admission, expected_index_sha256,
        expected_source_sha256, config)
    end

    private def validate_options! : Nil
      unless (1..8).includes?(@max_forwards_per_chain)
        raise ArgumentError.new("Hy-Nomas BF16 max_forwards_per_chain must be between 1 and 8")
      end
      unless @budget_bytes > 0 && @stage_bytes > 0 && @reserve_bytes >= 0
        raise ArgumentError.new("Hy-Nomas BF16 stack memory budget is invalid")
      end
    end

    private def initialize_from_admission(admission : HyNomasQ4StackIndex::Admission,
                                          expected_index_sha256 : String,
                                          expected_source_sha256 : String,
                                          config : HyNomasConfig) : Nil
      admission.validate!(expected_index_sha256, expected_source_sha256, config)
      @cursor = HyNomasFullPassCursor.new(BLOCK_COUNT, @max_forwards_per_chain)
      pins = admission.blocks
      unless pins.size == BLOCK_COUNT &&
             pins.each_with_index.all? { |pin, position| pin.block_index == position }
        raise ArgumentError.new("Hy-Nomas BF16 stack must contain ordered blocks 0...54")
      end

      bindings0 = HyNomasDualStreamBlock.checkpoint_bindings(config,
        ML::GGUF::SafeTensorDType::BF16, 0)
      expected_stage_bytes = bindings0.sum do |binding|
        binding.shape.reduce(1_i64) { |count, dimension| count * dimension } *
          WeightDType::BF16.bytes_per_element
      end
      unless @stage_bytes == expected_stage_bytes
        raise ArgumentError.new("Hy-Nomas BF16 stage budget must equal one exact 32-tensor block")
      end

      files = [] of CheckpointFile
      plans = [] of WeightPlan
      copies = [] of Array(CopyTensor)
      payload_bytes = 0_i64
      begin
        pins.each_with_index do |pin, block_index|
          file = BF16WeightCache.authenticate(pin.bf16_path, pin.bf16_sha256)
          files << file
          bindings = HyNomasDualStreamBlock.checkpoint_bindings(config,
            ML::GGUF::SafeTensorDType::BF16, block_index.to_i32)
          validate_control_inventory!(file, bindings, block_index.to_i32)

          # BF16 source bytes copy directly into BF16 residents. Conversion is
          # disabled, so each plan has a zero-byte raw staging requirement.
          plan = WeightPlan.new([file], bindings, @budget_bytes,
            resident_dtype: WeightDType::BF16, require_all: true,
            allow_f32_to_bf16: false)
          unless plan.entries.size == bindings.size &&
                 plan.destination_bytes == @stage_bytes &&
                 plan.staging_bytes == 0 && !plan.allow_f32_to_bf16
            raise ArgumentError.new("Hy-Nomas BF16 control plan is not an exact direct-copy block")
          end
          if plan.destination_bytes > Int64::MAX - payload_bytes
            raise ArgumentError.new("Hy-Nomas BF16 cache byte count overflow")
          end
          payload_bytes += plan.destination_bytes
          copies << plan.entries.map do |entry|
            CopyTensor.new(entry.binding.target, entry.info.n_elements.to_i32)
          end
          plans << plan
        end

        unless payload_bytes <= @budget_bytes &&
               @stage_bytes <= @budget_bytes - payload_bytes &&
               @reserve_bytes <= @budget_bytes - payload_bytes - @stage_bytes
          raise ArgumentError.new("Hy-Nomas BF16 controls, shared stage, and reserve exceed the memory budget")
        end
      rescue ex
        files.each(&.close)
        raise ex
      end

      @source_sha256 = admission.source_sha256
      @index_sha256 = admission.sha256
      @block_pins = pins
      @control_files = files
      @control_plans = plans
      @copy_tensors = copies
      @cache_payload_bytes = payload_bytes
      @admitted_bytes = payload_bytes + @stage_bytes + @reserve_bytes
    end

    # The pins are detached metadata. Internal cache and BF16 reader objects
    # remain private to preserve their admitted identities.
    def blocks : Array(HyNomasQ4StackIndex::BlockPin)
      @block_pins.dup
    end

    def loaded? : Bool
      @loaded
    end

    def released? : Bool
      @released
    end

    def forwards_on_chain : Int32
      @cursor.forwards_on_chain
    end

    # Allocate all block-local BF16 residents only after CPU admission has
    # completed. No F32 source or conversion staging is retained.
    def load!(arena : Arena) : Nil
      raise ArgumentError.new("Hy-Nomas BF16 stack has been released") if @released
      raise ArgumentError.new("Hy-Nomas BF16 stack is already loaded") if @loaded
      raise ArgumentError.new("Hy-Nomas BF16 stack is poisoned") if @poisoned
      validate_stage_arena!(arena)
      unless arena.resident_bytes == @stage_bytes
        raise ArgumentError.new("Hy-Nomas BF16 staging Arena does not match its admitted byte size")
      end

      resident_buffers = arena.buffers.values.dup
      snapshots = [] of Hash(String, ML::MetalBuffer)
      begin
        HyNomasQ4Cache.prepare_decode!
        @control_plans.each_with_index do |plan, block_index|
          file = @control_files[block_index]
          file.unchanged!
          control_arena = Arena.new
          @control_arenas << control_arena
          prepared = PreparedWeights.new(plan, control_arena)
          @prepared << prepared
          file.unchanged!
          validate_control_arena!(block_index.to_i32, control_arena)

          snapshot = control_arena.buffers.dup
          snapshot.each_value do |buffer|
            resident_buffers.each { |prior| MetalOps.distinct(buffer, prior) }
            resident_buffers << buffer
          end
          snapshots << snapshot
        end
        @stage_arena = arena
        @stage_buffers = arena.buffers.dup
        @control_buffers = snapshots
        @loaded = true
      rescue ex
        cleanup_after_failed_load!
        @poisoned = true
        @released = true
        raise ex
      end
    end

    # Called during serial CPU preflight and again immediately before each
    # block copy. It allocates nothing and appends no native command.
    def validate_for!(block_index : Int32, arena : Arena) : Nil
      raise ArgumentError.new("Hy-Nomas BF16 stack has been released") if @released
      raise ArgumentError.new("Hy-Nomas BF16 stack is not loaded") unless @loaded
      raise ArgumentError.new("Hy-Nomas BF16 stack is poisoned") if @poisoned
      unless (0...BLOCK_COUNT).includes?(block_index) &&
             @block_pins[block_index].block_index == block_index
        raise ArgumentError.new("Hy-Nomas BF16 block index is outside its authenticated order")
      end
      owner = @stage_arena || raise ArgumentError.new("Hy-Nomas BF16 staging Arena is unavailable")
      unless owner.same?(arena)
        raise ArgumentError.new("Hy-Nomas BF16 serial weights changed staging Arena identity")
      end
      validate_stage_bindings!(arena)
      validate_control_arena!(block_index, @control_arenas[block_index])
      @control_files[block_index].unchanged!
    end

    # Copy exactly one already-resident BF16 block into the fixed shared stage
    # on the caller's existing serial encoder. No wait or readback is added.
    def encode!(chain : ML::Metal::MPSCommandChain,
                block_index : Int32, arena : Arena) : Nil
      validate_for!(block_index, arena)
      encoder = chain.encoder
      previous = @last_chain
      @cursor.validate_block!(chain.object_id.to_u64, block_index,
        previous_terminal: previous.try(&.completed?) || false,
        previous_wait_count: previous.try(&.wait_count) || 0)
      if block_index == 0
        @last_chain = chain
      end

      begin
        source_arena = @control_arenas[block_index]
        @copy_tensors[block_index].each do |tensor|
          source = source_arena[tensor.target]
          destination = arena[tensor.target]
          MetalOps.check_storage(source, tensor.elements, WeightDType::BF16)
          MetalOps.check_storage(destination, tensor.elements, WeightDType::BF16)
          MetalOps.distinct(source, destination)
          encoder.set_pipeline(HyNomasQ4Cache.copy_pipeline)
            .set_buffer(source, 0)
            .set_buffer(destination, 1, ML::Metal::BufferAccess::Write)
            .set_value(tensor.elements.to_u32, 2)
            .dispatch_1d(tensor.elements)
        end
        @cursor.finish_block!(chain.object_id.to_u64, block_index)
      rescue ex
        @poisoned = true
        raise ex
      end
    end

    def release : Nil
      return if @released
      if chain = @last_chain
        raise ArgumentError.new("Complete Hy-Nomas BF16 command chain before release") unless chain.completed?
      end
      cleanup_residents!
      @control_files.each(&.close)
      @control_files.clear
      @control_plans.clear
      @copy_tensors.clear
      @control_buffers.clear
      @stage_buffers.clear
      @stage_arena = nil
      @loaded = false
      @released = true
    end

    private def validate_control_inventory!(file : CheckpointFile,
                                            bindings : Array(WeightBinding),
                                            block_index : Int32) : Nil
      expected_prefix = "double_blocks.#{block_index}."
      expected_sources = bindings.map(&.source).to_set
      unless bindings.size == HyNomasQ4Cache::MATRIX_COUNT + HyNomasQ4Cache::VECTOR_COUNT &&
             bindings.all? { |binding| binding.source.starts_with?(expected_prefix) }
        raise ArgumentError.new("Hy-Nomas BF16 bindings are not the exact block #{block_index} prefix")
      end

      tensors = file.tensors
      unless tensors.size == bindings.size && tensors.keys.to_set == expected_sources
        raise ArgumentError.new("Hy-Nomas BF16 control must contain exactly the 32 block #{block_index} tensors")
      end
      bindings.each do |binding|
        info = tensors[binding.source]?
        unless info && info.shape == binding.shape && info.dtype.bf16?
          raise ArgumentError.new("Hy-Nomas BF16 control dtype or shape mismatch #{binding.source}")
        end
      end
    end

    private def validate_control_arena!(block_index : Int32, arena : Arena) : Nil
      unless block_index >= 0 && block_index < @copy_tensors.size &&
             arena.buffers.size == @copy_tensors[block_index].size
        raise ArgumentError.new("Hy-Nomas BF16 control Arena inventory mismatch")
      end
      unless arena.resident_bytes == @stage_bytes
        raise ArgumentError.new("Hy-Nomas BF16 control Arena does not match one block's admitted resident bytes")
      end
      @copy_tensors[block_index].each do |tensor|
        buffer = arena.buffers[tensor.target]?
        unless buffer && arena.dtype(tensor.target).bf16? && buffer.valid? &&
               buffer.storage_mode.shared? &&
               buffer.size == tensor.elements.to_i64 * WeightDType::BF16.bytes_per_element
          raise ArgumentError.new("Hy-Nomas BF16 control Arena binding is invalid #{tensor.target}")
        end
      end
      if snapshot = @control_buffers[block_index]?
        same_identity = snapshot.size == arena.buffers.size
        snapshot.each do |name, expected|
          current = arena.buffers[name]?
          unless current && expected.same?(current)
            same_identity = false
            break
          end
        end
        unless same_identity
          raise ArgumentError.new("Hy-Nomas BF16 control Arena identity changed for block #{block_index}")
        end
      end
    end

    private def validate_stage_arena!(arena : Arena) : Nil
      expected = @copy_tensors.first
      unless arena.buffers.size == expected.size
        raise ArgumentError.new("Hy-Nomas BF16 staging Arena tensor inventory mismatch")
      end
      prior = [] of ML::MetalBuffer
      expected.each do |tensor|
        buffer = arena.buffers[tensor.target]?
        unless buffer && arena.dtype(tensor.target).bf16? && buffer.valid? &&
               buffer.storage_mode.shared? &&
               buffer.size == tensor.elements.to_i64 * WeightDType::BF16.bytes_per_element
          raise ArgumentError.new("Hy-Nomas BF16 staging Arena binding is invalid #{tensor.target}")
        end
        prior.each { |other| MetalOps.distinct(buffer, other) }
        prior << buffer
      end
    end

    private def validate_stage_bindings!(arena : Arena) : Nil
      same_identity = arena.buffers.size == @stage_buffers.size
      @stage_buffers.each do |name, expected|
        current = arena.buffers[name]?
        unless current && expected.same?(current) && current.valid? && arena.dtype(name).bf16?
          same_identity = false
          break
        end
      end
      unless same_identity
        raise ArgumentError.new("Hy-Nomas BF16 staging Arena map or buffer identity changed")
      end
    end

    private def cleanup_residents! : Nil
      @prepared.each(&.release)
      @prepared.clear
      @control_arenas.each(&.release)
      @control_arenas.clear
    end

    private def cleanup_after_failed_load! : Nil
      cleanup_residents!
      @control_files.each(&.close)
      @control_files.clear
      @control_plans.clear
      @copy_tensors.clear
      @control_buffers.clear
      @stage_buffers.clear
      @stage_arena = nil
    end
  end
end
