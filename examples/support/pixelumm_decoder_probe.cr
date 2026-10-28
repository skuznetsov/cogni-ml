require "./pixelumm_two_block_probe"
require "./pixelumm_head_probe"

# A direct composition of the admitted 36-block decoder and its global head.
# The head borrows block 35's output buffer on the same command chain; it does
# not upload or copy the saved head-fixture input.
module PixelUMMDecoderProbe
  extend self

  alias Record = PixelUMMBlockProbe::Record
  alias Comparison = PixelUMMBlockProbe::Comparison
  alias BlockFixture = PixelUMMTwoBlockProbe::Fixture
  alias WeightCache = PixelUMMTwoBlockProbe::WeightCache
  alias BlockSession = PixelUMMTwoBlockProbe::Session
  alias HeadPlan = ML::Video::PixelUMMDecoderHeadMPS::Plan
  alias HeadConfig = ML::Video::PixelUMMDecoderHeadMPS::Config

  BLOCK_COUNT   = 36
  NAMED_LIMIT   = PixelUMMTwoBlockProbe.named_budget_limit(BLOCK_COUNT)
  CAPTURE_COUNT = 43
  HEAD_PREFIX   = "head."

  record RequestArm, kind : String, alternate : Bool, poisoned : Bool, profile : Bool,
    shared_gate_up : Bool = false

  # Run the candidate first, before baseline-correct capture bytes can exist.
  # Both graphs are warmed before the timed ABBA pair. The alternate-input
  # gates are separately poisoned for each mode and excluded from timing claims.
  def shared_gate_up_schedule : Array(RequestArm)
    [
      RequestArm.new("a_warm_shared", false, false, false, true),
      RequestArm.new("a_warm_baseline", false, false, false, false),
      RequestArm.new("a_timed_baseline_first", false, false, false, false),
      RequestArm.new("a_timed_shared_first", false, false, false, true),
      RequestArm.new("a_timed_shared_second", false, false, false, true),
      RequestArm.new("a_timed_baseline_second", false, false, false, false),
      RequestArm.new("b_poisoned_baseline", true, true, false, false),
      RequestArm.new("b_poisoned_shared", true, true, false, true),
    ]
  end

  # Both orders of the same-input profiler control bracket the diagnostic
  # input/poison pair. This is a drift discriminator, not a throughput estimate.
  def variance_schedule : Array(RequestArm)
    [
      RequestArm.new("a_clean_off_early", false, false, false),
      RequestArm.new("a_clean_on_early", false, false, true),
      RequestArm.new("b_clean_on", true, false, true),
      RequestArm.new("b_poisoned_on", true, true, true),
      RequestArm.new("a_clean_on_late", false, false, true),
      RequestArm.new("a_clean_off_late", false, false, false),
    ]
  end

  struct Admission
    getter block_named_bytes : Int64
    getter head_owned_named_bytes : Int64
    getter head_weight_named_bytes : Int64
    getter named_bytes : Int64

    def initialize(@block_named_bytes : Int64, @head_owned_named_bytes : Int64,
                   @head_weight_named_bytes : Int64, @named_bytes : Int64)
    end
  end

  class Fixture
    getter blocks : BlockFixture
    getter head : PixelUMMHeadProbe::Fixture
    getter capture_order : Array(String)

    def initialize(@blocks : BlockFixture, @head : PixelUMMHeadProbe::Fixture)
      @capture_order = @blocks.capture_order + @head.capture_order.map { |name| HEAD_PREFIX + name }
      validate!
    end

    # Nested fixture getters expose mutable metadata containers. Recheck the
    # cross-fixture edge and inventories at admission and replay boundaries.
    def validate! : Nil
      unless @blocks.block_count == BLOCK_COUNT && @blocks.grid == [6, 44, 80] &&
             @blocks.capture_order == PixelUMMTwoBlockProbe.thirty_six_survivor_capture_order &&
             @blocks.captures.keys.sort == PixelUMMTwoBlockProbe.thirty_six_survivor_capture_order.sort &&
             @head.grid == [6, 44, 80] && @head.capture_order == PixelUMMHeadProbe::CAPTURES
        raise ArgumentError.new("integrated PixelUMM fixture requires full-J 36-block and head inventories")
      end
      unless @blocks.inputs.keys.sort == ["cos", "sin", "x"] &&
             @head.inputs.keys == ["block35_output"] &&
             @head.weights.keys.sort == PixelUMMHeadProbe.weight_shapes.keys.sort &&
             @head.captures.keys.sort == PixelUMMHeadProbe::CAPTURES.sort
        raise ArgumentError.new("integrated PixelUMM fixture record inventory changed after authentication")
      end
      PixelUMMTwoBlockProbe.validate_weights!(@blocks.weights, BLOCK_COUNT)
      block_output = @blocks.captures["block35.output"]
      PixelUMMTwoBlockProbe.validate_capture!("block35.output", block_output, 21_120, BLOCK_COUNT)
      head_input = @head.inputs["block35_output"]
      unless head_input.dtype == "BF16" && head_input.shape == [1, 21_120, 4096]
        raise ArgumentError.new("integrated PixelUMM head input metadata is not tile-major BF16 full-J")
      end
      PixelUMMHeadProbe.weight_shapes.each do |name, shape|
        record = @head.weights[name]
        unless record.dtype == "BF16" && record.shape == shape
          raise ArgumentError.new("integrated PixelUMM head weight metadata changed after authentication")
        end
      end
      PixelUMMHeadProbe.capture_shapes(@head.grid).each do |name, shape|
        record = @head.captures[name]
        unless record.dtype == "BF16" && record.shape == shape
          raise ArgumentError.new("integrated PixelUMM head capture metadata changed after authentication")
        end
      end
      PixelUMMDecoderProbe.validate_pair!(
        block_output.sha256, head_input.sha256)
      expected_order = @blocks.capture_order + @head.capture_order.map { |name| HEAD_PREFIX + name }
      unless @capture_order == expected_order && @capture_order.size == CAPTURE_COUNT &&
             @capture_order.uniq.size == CAPTURE_COUNT
        raise ArgumentError.new("integrated PixelUMM capture inventory must contain 43 unique stages")
      end
    end

    def compatible_replay!(other : Fixture) : Nil
      validate!
      other.validate!
      @blocks.compatible_replay!(other.blocks)
      @head.compatible_replay!(other.head)
      PixelUMMTwoBlockProbe.validate_replay_output_distinguishability!(
        @blocks.captures, other.blocks.captures, BLOCK_COUNT)
      PixelUMMDecoderProbe.validate_pair!(
        other.blocks.captures["block35.output"].sha256,
        other.head.inputs["block35_output"].sha256)
    end
  end

  # Pure CPU inventory calculation. The head input is deliberately absent:
  # its exact buffer is already included in the block chain's unique bytes.
  def aggregate_named_bytes(*, block_named : Int64, head_weight_named : Int64,
                            head_owned_named : Int64) : Int64
    values = {block_named, head_weight_named, head_owned_named}
    unless values.all?(&.> 0)
      raise ArgumentError.new("integrated PixelUMM named-byte components must be positive")
    end
    total = 0_i64
    values.each do |value|
      raise ArgumentError.new("integrated PixelUMM named-byte total overflows") if total > Int64::MAX - value
      total += value
    end
    total
  end

  def validate_aggregate_budget!(required : Int64, max_named_bytes : Int64,
                                 limit : Int64 = NAMED_LIMIT) : Int64
    unless required > 0 && max_named_bytes > 0 && limit > 0 && max_named_bytes <= limit
      raise ArgumentError.new("integrated PixelUMM aggregate budget is outside the admitted bound")
    end
    unless required <= max_named_bytes
      raise ArgumentError.new("integrated PixelUMM aggregate named bytes exceed budget: required #{required}, admitted #{max_named_bytes}")
    end
    required
  end

  def validate_pair!(block35_output_sha256 : String, head_input_sha256 : String) : Nil
    unless PixelUMMBlockProbe.valid_digest?(block35_output_sha256) &&
           PixelUMMBlockProbe.valid_digest?(head_input_sha256) &&
           block35_output_sha256 == head_input_sha256
      raise ArgumentError.new("head block35_output must match the authenticated block35.output capture")
    end
  end

  # Kept as a CPU-testable guard and called before HeadPlan#encode. A failed
  # guard must happen before that method touches the command encoder.
  def validate_head_order!(*, blocks_encoded : Bool, same_chain : Bool,
                           chain_open : Bool) : Nil
    unless blocks_encoded && same_chain && chain_open
      raise ArgumentError.new("PixelUMM head must encode after all blocks on the same open command chain")
    end
  end

  def block_unique_bytes(fixture : BlockFixture) : Int64
    unless fixture.block_count == BLOCK_COUNT && fixture.grid == [6, 44, 80]
      raise ArgumentError.new("integrated PixelUMM admission requires full-J 36-block fixture")
    end
    config = PixelUMMTwoBlockProbe::Decoder::Config.new(
      temporal: fixture.grid[0], height: fixture.grid[1], width: fixture.grid[2])
    weights_by_slot = Array.new(BLOCK_COUNT) do |slot|
      PixelUMMBlockProbe.weight_shapes.keys.to_h do |name|
        {name, fixture.weights["block#{slot}.#{name}"].nbytes}
      end
    end
    captures_by_slot = fixture.captures_by_slot.map_with_index do |captures, slot|
      captures.select do |name|
        PixelUMMTwoBlockProbe::Decoder::Workspace.capture_survives?(
          name, slot, BLOCK_COUNT, reuse_outputs: true)
      end
    end
    PixelUMMTwoBlockProbe::Decoder::Workspace.validate_admission!(config,
      input_bytes: fixture.inputs["x"].nbytes,
      cosine_bytes: fixture.inputs["cos"].nbytes,
      sine_bytes: fixture.inputs["sin"].nbytes,
      weights_bytes_by_slot: weights_by_slot,
      captures_by_slot: captures_by_slot,
      max_unique_bytes: NAMED_LIMIT,
      reuse_outputs: true)
  end

  def admission(fixture : Fixture, max_named_bytes : Int64) : Admission
    fixture.validate!
    block_bytes = block_unique_bytes(fixture.blocks)
    head_config = HeadConfig.new(temporal: 6, height: 44, width: 80)
    head_weight_bytes = fixture.head.weights.values.sum(&.nbytes)
    expected_head_weights = PixelUMMHeadProbe.weight_shapes.values.sum do |shape|
      shape.reduce(2_i64) { |bytes, dimension| bytes * dimension }
    end
    unless head_weight_bytes == expected_head_weights
      raise ArgumentError.new("integrated PixelUMM head weight inventory byte total mismatch")
    end
    head_owned_bytes = HeadPlan.owned_bytes(head_config, fixture.head.capture_order)
    total = aggregate_named_bytes(block_named: block_bytes,
      head_weight_named: head_weight_bytes, head_owned_named: head_owned_bytes)
    validate_aggregate_budget!(total, max_named_bytes)
    Admission.new(block_bytes, head_owned_bytes, head_weight_bytes, total)
  end

  class Session
    getter fixture : Fixture
    getter named_bytes : Int64
    getter block_named_bytes : Int64
    getter head_owned_named_bytes : Int64
    getter head_weight_named_bytes : Int64
    getter capture_order : Array(String)
    getter phase_observer : PixelUMMMemoryProbe::Recorder?

    @blocks : BlockSession? = nil
    @head_plan : HeadPlan? = nil
    @head_weights = {} of String => ML::MetalBuffer
    @blocks_chain : ML::Metal::MPSCommandChain? = nil
    @head_chain : ML::Metal::MPSCommandChain? = nil
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @released = false

    def initialize(@fixture : Fixture, cache : WeightCache, *, max_named_bytes : Int64,
                   prepare_shared_gate_up : Bool = false,
                   @phase_observer : PixelUMMMemoryProbe::Recorder? = nil)
      # All shape, provenance linkage and aggregate budget decisions happen
      # before constructing the block Session, which is the first GPU uploader.
      admitted = PixelUMMDecoderProbe.admission(@fixture, max_named_bytes)
      @block_named_bytes = admitted.block_named_bytes
      @head_owned_named_bytes = admitted.head_owned_named_bytes
      @head_weight_named_bytes = admitted.head_weight_named_bytes
      @named_bytes = admitted.named_bytes
      @capture_order = @fixture.capture_order.dup

      begin
        memory_phase("before_integrated_block_session")
        @blocks = BlockSession.new(@fixture.blocks, cache, @block_named_bytes,
          ffn_f32: true, reuse_outputs: true, compare_common_ring_captures: true,
          prepare_shared_gate_up: prepare_shared_gate_up,
          phase_observer: @phase_observer)
        unless blocks.named_bytes == @block_named_bytes && blocks.capture_order.size == 38
          raise ArgumentError.new("integrated PixelUMM block inventory differs from CPU admission")
        end
        memory_phase("after_integrated_block_session", phase_bytes_kind: "block_unique_named_bytes",
          phase_bytes: blocks.named_bytes)

        fixture.head.weights.each do |name, record|
          buffer = ML::MetalBuffer.new(record.nbytes, ML::StorageMode::Shared)
          @head_weights[name] = buffer
          upload_head_weight!(fixture.head, record, buffer)
        end
        unless @head_weights.values.sum(&.size) == @head_weight_named_bytes
          raise ArgumentError.new("integrated PixelUMM uploaded head-weight extent mismatch")
        end
        memory_phase("after_integrated_head_weight_upload", phase_bytes_kind: "head_weight_named_bytes",
          phase_bytes: @head_weight_named_bytes)

        weights = PixelUMMHeadProbe.weight_shapes.keys.to_h do |name|
          {name, @head_weights[name]}
        end
        config = HeadConfig.new(temporal: 6, height: 44, width: 80)
        @head_plan = HeadPlan.new(config,
          input: blocks.plans[35].output,
          weights: weights,
          max_named_bytes: @head_owned_named_bytes,
          captures: fixture.head.capture_order)
        unless head_plan.input.handle == blocks.plans[35].output.handle &&
               @named_bytes == blocks.named_bytes + @head_weight_named_bytes + head_plan.named_bytes
          raise ArgumentError.new("integrated PixelUMM direct handoff or unique-byte inventory mismatch")
        end
        memory_phase("after_integrated_head_plan", phase_bytes_kind: "head_owned_named_bytes",
          phase_bytes: head_plan.named_bytes)
      rescue error
        rollback_construction
        raise error
      end
    end

    def blocks : BlockSession
      @blocks || raise ArgumentError.new("integrated PixelUMM block session unavailable")
    end

    def head_plan : HeadPlan
      @head_plan || raise ArgumentError.new("integrated PixelUMM head plan unavailable")
    end

    def head_input_aliases_block35_output? : Bool
      head_plan.input.handle == blocks.plans[35].output.handle
    end

    def use_shared_gate_up!(enabled : Bool) : Nil
      check_idle!
      blocks.use_shared_gate_up!(enabled)
    end

    # Enqueue all 36 native plans onto the caller's current command chain.
    def encode_blocks(chain : ML::Metal::MPSCommandChain, *, profile : Bool = false) : Nil
      check_idle!
      blocks.encode(chain, profile: profile)
      @blocks_chain = chain
      @last_chain = chain
    end

    # This is public so the negative ordering case is explicit and testable.
    # It checks all 36 block encodes completed synchronously before asking the
    # head plan for an encoder, then appends head work to that exact open chain.
    def encode_head(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("integrated PixelUMM session released") if @released
      PixelUMMDecoderProbe.validate_head_order!(
        blocks_encoded: !@blocks_chain.nil?,
        same_chain: @blocks_chain.try(&.same?(chain)) || false,
        chain_open: !chain.completed?)
      raise ArgumentError.new("integrated PixelUMM head already encoded on this chain") if @head_chain.try(&.same?(chain))
      chain.encoder
      head_plan.encode(chain)
      @head_chain = chain
      @last_chain = chain
    end

    # The intended production call: all blocks first, then the global head,
    # with no wait, host read or allocation between their graph encodes.
    def encode(chain : ML::Metal::MPSCommandChain) : Nil
      encode_blocks(chain)
      encode_head(chain)
    end

    def replay_input!(fixture : Fixture) : Nil
      check_idle!
      @fixture.compatible_replay!(fixture)
      # The block session uploads only the new x. The matching head fixture
      # contributes oracle captures but its duplicate input bytes are unused.
      blocks.replay_input!(fixture.blocks)
      @fixture = fixture
    end

    def replay_input!(block_fixture : BlockFixture,
                      head_fixture : PixelUMMHeadProbe::Fixture) : Nil
      replay_input!(Fixture.new(block_fixture, head_fixture))
    end

    def poison_transients! : Int64
      check_idle!
      block_poison_bytes = blocks.workspace.mutable_shared_buffers(include_outputs: true).values.sum(&.size)
      blocks.poison_transients!
      head_poison_bytes = head_plan.poison_transients!
      block_poison_bytes + head_poison_bytes
    end

    def compare(*, request : Int32? = nil,
                phase_observer : PixelUMMMemoryProbe::Recorder? = nil) : Array(Comparison)
      chain = @head_chain || raise ArgumentError.new("integrated PixelUMM decoder was not fully encoded")
      unless chain.completed? && chain.wait_count == 1 && @blocks_chain.try(&.same?(chain))
        raise ArgumentError.new("integrated PixelUMM diagnostics require one terminal wait after same-chain block/head encoding")
      end
      results = blocks.compare(request: request, phase_observer: phase_observer || @phase_observer)
      observer = phase_observer || @phase_observer
      fixture.head.capture_order.each do |name|
        record = fixture.head.captures[name]
        buffer = head_plan.stage(name)
        raise ArgumentError.new("integrated PixelUMM head capture extent mismatch: #{name}") unless buffer.size == record.nbytes
        comparison_bytes = record.nbytes &* 2_i64
        memory_phase("before_integrated_head_compare_buffers", request: request,
          phase_bytes_kind: "cpu_expected_and_actual_buffer_bytes", phase_bytes: comparison_bytes,
          observer: observer)
        expected = fixture.head.read(record)
        actual = Bytes.new(record.nbytes.to_i32)
        buffer.read_bytes(actual.to_unsafe, actual.size)
        memory_phase("integrated_head_compare_buffers_ready", request: request,
          phase_bytes_kind: "cpu_expected_and_actual_buffer_bytes", phase_bytes: comparison_bytes,
          observer: observer)
        results << PixelUMMHeadProbe.compare_bytes(HEAD_PREFIX + name, expected, actual)
        expected = Bytes.empty
        actual = Bytes.empty
        GC.collect
        memory_phase("integrated_head_compare_stage_done", request: request,
          phase_bytes_kind: "cpu_expected_and_actual_buffer_bytes", phase_bytes: comparison_bytes,
          observer: observer)
      end
      unless results.size == CAPTURE_COUNT && results.map { |result| result[:stage] } == @capture_order
        raise ArgumentError.new("integrated PixelUMM comparison order/count mismatch")
      end
      results
    end

    def release : Nil
      return if @released
      check_idle!
      @released = true
      first_error = nil.as(Exception?)
      begin
        @head_plan.try(&.release)
      rescue error
        first_error = error
      end
      @head_plan = nil
      @head_weights.each_value do |weight|
        begin
          weight.release
        rescue error
          first_error ||= error
        end
      end
      @head_weights.clear
      begin
        @blocks.try(&.release)
      rescue error
        first_error ||= error
      end
      @blocks = nil
      raise first_error.not_nil! if first_error
    end

    private def check_idle! : Nil
      raise ArgumentError.new("integrated PixelUMM session released") if @released
      if chain = @last_chain
        raise ArgumentError.new("integrated PixelUMM command chain is in flight") unless chain.completed?
      end
    end

    private def upload_head_weight!(fixture : PixelUMMHeadProbe::Fixture,
                                    record : Record, buffer : ML::MetalBuffer) : Nil
      raise ArgumentError.new("integrated PixelUMM head weight destination mismatch") unless buffer.valid? &&
                                                                                             buffer.storage_mode.shared? && buffer.size == record.nbytes
      destination = buffer.contents.as(Pointer(UInt8)).to_slice(record.nbytes.to_i32)
      File.open(fixture.payload) do |io|
        io.seek(record.offset)
        io.read_fully(destination)
      end
      unless Digest::SHA256.hexdigest(destination) == record.sha256
        raise ArgumentError.new("integrated PixelUMM head weight tensor hash mismatch")
      end
    end

    private def memory_phase(phase : String, *, request : Int32? = nil,
                             phase_bytes_kind : String? = nil, phase_bytes : Int64? = nil,
                             observer : PixelUMMMemoryProbe::Recorder? = @phase_observer) : Nil
      observer.try do |active|
        active.emit(phase, request: request,
          phase_bytes_kind: phase_bytes_kind, phase_bytes: phase_bytes)
      end
      nil
    end

    private def rollback_construction : Nil
      @head_plan.try(&.release)
      @head_plan = nil
      @head_weights.each_value(&.release)
      @head_weights.clear
      @blocks.try(&.release)
      @blocks = nil
    end
  end
end
