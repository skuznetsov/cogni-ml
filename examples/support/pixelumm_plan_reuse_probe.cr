require "json"
require "digest/sha256"
require "set"
require "../../src/ml/video/pixelumm_decoder_block_mps"
require "./pixelumm_connected_resource_probe"
require "./pixelumm_memory_probe"
require "./pixelumm_block_probe"

module PixelUMMPlanReuseProbe
  extend self

  SCHEMA                = "cogni_ml.pixelumm_plan_reuse_probe/v1"
  SOFTMAX_PV_SCHEMA     = "cogni_ml.pixelumm_plan_reuse_probe/softmax_pv/v1"
  SOFTMAX_PV_F32_SCHEMA = "cogni_ml.pixelumm_plan_reuse_probe/softmax_pv_f32_check/v1"
  ATOL                  = PixelUMMConnectedResourceProbe::ATOL
  RTOL                  = PixelUMMConnectedResourceProbe::RTOL
  GIB                   = PixelUMMConnectedResourceProbe::GIB
  MAX_NAMED_BYTES       = PixelUMMConnectedResourceProbe::MAX_NAMED_BYTES
  MAX_RESIDENT_PLANS    =  2
  REQUESTS              =  6
  MEMORY_FLOOR_PERCENT  = 10

  alias Budget = PixelUMMConnectedResourceProbe::Budget
  alias ScheduleEntry = NamedTuple(index: Int32, plan_index: Int32, warmup: Bool)
  alias F32TileGeometry = NamedTuple(tile_index: Int32, query_tokens: Int32,
    key_tokens: Int32, probability_elements: UInt32, attended_elements: UInt32)
  alias F32Geometry = NamedTuple(probabilities_snapshot_bytes: Int64,
    attended_snapshot_bytes: Int64, probability_elements: UInt32,
    attended_elements: UInt32, tiles: Array(F32TileGeometry))
  alias Metric = PixelUMMConnectedResourceProbe::Metric

  F32_CHECK_METRIC_SLOTS = 11
  F32_CHECK_METRIC_BYTES = F32_CHECK_METRIC_SLOTS.to_i64 * 4 * sizeof(UInt32)

  def validate_resident_plan_count!(resident_plans : Int32) : Nil
    unless 1 <= resident_plans <= MAX_RESIDENT_PLANS
      raise ArgumentError.new("PixelUMM plan reuse probe supports one or two resident native Plans")
    end
  end

  def validate_softmax_pv_configuration!(resident_plans : Int32,
                                         shared_gate_up : Bool,
                                         prepare_softmax_pv : Bool) : Nil
    validate_resident_plan_count!(resident_plans)
    return unless prepare_softmax_pv
    unless resident_plans == 2
      raise ArgumentError.new("PixelUMM softmax/PV comparison requires exactly two resident native Plans")
    end
    if shared_gate_up
      raise ArgumentError.new("PixelUMM softmax/PV comparison cannot be combined with shared gate/up")
    end
  end

  def validate_softmax_pv_f32_configuration!(prepare_softmax_pv : Bool,
                                             check_softmax_pv_f32 : Bool,
                                             resident_plans : Int32,
                                             shared_gate_up : Bool) : Nil
    return unless check_softmax_pv_f32
    unless prepare_softmax_pv
      raise ArgumentError.new("direct F32 softmax/PV checking requires --softmax-pv")
    end
    validate_softmax_pv_configuration!(resident_plans, shared_gate_up, true)
  end

  def fused_softmax_pv_for_plan!(prepare_softmax_pv : Bool,
                                 resident_plans : Int32,
                                 shared_gate_up : Bool,
                                 plan_index : Int32) : Bool
    validate_softmax_pv_configuration!(resident_plans, shared_gate_up, prepare_softmax_pv)
    unless 0 <= plan_index < resident_plans
      raise ArgumentError.new("PixelUMM attention mode selected an invalid native Plan index")
    end
    prepare_softmax_pv && plan_index == 1
  end

  # `maximum` caps named MetalBuffers. Opaque native Plan and MPS allocations
  # are outside this inventory and are guarded separately by the runner.
  def validate_budget!(tokens : Int32, resident_plans : Int32, maximum : Int64,
                       check_softmax_pv_f32 : Bool = false) : Budget
    validate_resident_plan_count!(resident_plans)
    base = PixelUMMConnectedResourceProbe.validate_budget!(tokens, maximum)
    return base unless check_softmax_pv_f32

    geometry = softmax_pv_f32_geometry(tokens)
    extra_bytes = geometry[:probabilities_snapshot_bytes] +
                  geometry[:attended_snapshot_bytes] +
                  F32_CHECK_METRIC_BYTES - base[:metric_bytes]
    total_bytes = base[:total_bytes] + extra_bytes
    if total_bytes > maximum
      raise ArgumentError.new("PixelUMM direct F32 comparator buffers exceed named-memory cap before Metal allocation")
    end
    {plan_bytes: base[:plan_bytes], borrowed_bytes: base[:borrowed_bytes],
     expected_bytes: base[:expected_bytes], metric_bytes: F32_CHECK_METRIC_BYTES,
     total_bytes: total_bytes}
  end

  # Derive direct comparator coverage from the same production tile index lists
  # and panel dimensions used by the native attention path.
  def softmax_pv_f32_geometry(tokens : Int32) : F32Geometry
    config = PixelUMMConnectedResourceProbe.config(tokens)
    layout = config.layout
    unless layout.local_attention?
      raise ArgumentError.new("direct F32 softmax/PV check requires bounded local attention")
    end

    tiles = Array(F32TileGeometry).new(layout.tile_count) do |tile_index|
      query_tokens = layout.query_indices(tile_index.to_i32).size.to_i32
      key_tokens = layout.key_indices(tile_index.to_i32).size.to_i32
      probability_elements = checked_u32!(query_tokens.to_i64 * key_tokens * config.query_heads,
        "PixelUMM probability F32 tile element count")
      attended_elements = checked_u32!(query_tokens.to_i64 * config.query_heads * config.head_dim,
        "PixelUMM attended F32 tile element count")
      {tile_index: tile_index.to_i32, query_tokens: query_tokens,
       key_tokens: key_tokens, probability_elements: probability_elements,
       attended_elements: attended_elements}
    end
    probability_total = tiles.reduce(0_u64) { |total, tile| total + tile[:probability_elements].to_u64 }
    probability_elements = checked_u32!(probability_total,
      "PixelUMM probability F32 aggregate element count")
    attended_total = tiles.reduce(0_u64) { |total, tile| total + tile[:attended_elements].to_u64 }
    attended_elements = checked_u32!(attended_total,
      "PixelUMM attended F32 aggregate element count")
    probabilities_snapshot_bytes = config.query_heads.to_i64 * layout.max_query_tokens *
                                   layout.max_key_tokens * sizeof(Float32)
    attended_snapshot_bytes = config.query_heads.to_i64 * layout.max_query_tokens *
                              config.head_dim * sizeof(Float32)
    unless probabilities_snapshot_bytes > 0 && attended_snapshot_bytes > 0
      raise ArgumentError.new("PixelUMM direct F32 snapshot panel size must be positive")
    end
    {probabilities_snapshot_bytes: probabilities_snapshot_bytes,
     attended_snapshot_bytes: attended_snapshot_bytes,
     probability_elements: probability_elements,
     attended_elements: attended_elements, tiles: tiles}
  end

  def validate_softmax_pv_f32_metrics!(metrics : Array(Metric), geometry : F32Geometry) : Nil
    unless metrics.size == F32_CHECK_METRIC_SLOTS
      raise ArgumentError.new("PixelUMM direct F32 comparator requires eleven aggregate metric slots")
    end
    validate_f32_stage_metric!(metrics[8], geometry[:probability_elements], 8)
    validate_f32_stage_metric!(metrics[9], geometry[:attended_elements], 9)
    positive_control = metrics[10]
    unless positive_control[:elements] == 1_u32 && positive_control[:failures] == 1_u32 &&
           positive_control[:different_words] == 1_u32 && positive_control[:nonfinite] == 1_u32
      raise ArgumentError.new("PixelUMM direct F32 comparator NaN positive control was not exact")
    end
  end

  def validate_f32_snapshot_sizes!(geometry : F32Geometry,
                                   probabilities_bytes : Int64,
                                   attended_bytes : Int64) : Nil
    unless probabilities_bytes == geometry[:probabilities_snapshot_bytes] &&
           attended_bytes == geometry[:attended_snapshot_bytes]
      raise ArgumentError.new("PixelUMM direct F32 snapshot buffers must exactly match scratch[4] and scratch[5]")
    end
  end

  private def checked_u32!(value : Int64, label : String) : UInt32
    unless value > 0 && value <= UInt32::MAX.to_i64
      raise ArgumentError.new("#{label} is outside the nonzero UInt32 range")
    end
    value.to_u32
  end

  private def validate_f32_stage_metric!(metric : Metric, expected_elements : UInt32,
                                         slot : Int32) : Nil
    unless metric[:elements] == expected_elements && metric[:failures] == 0_u32 &&
           metric[:nonfinite] == 0_u32
      raise ArgumentError.new("PixelUMM direct F32 comparator aggregate failed for slot #{slot}")
    end
  end

  private def checked_u32!(value : UInt64, label : String) : UInt32
    unless value > 0 && value <= UInt32::MAX.to_u64
      raise ArgumentError.new("#{label} is outside the nonzero UInt32 range")
    end
    value.to_u32
  end

  def validate_geometry!(grid : Array(Int32), tokens : Int32) : Nil
    config = PixelUMMConnectedResourceProbe.config(tokens)
    expected = [config.temporal, config.height, config.width]
    raise ArgumentError.new("connected PixelUMM fixture grid/rows mismatch") unless grid == expected
  end

  def schedule(resident_plans : Int32, *, reverse_warmups : Bool,
               reverse_first : Bool) : Array(ScheduleEntry)
    validate_resident_plan_count!(resident_plans)
    return Array(ScheduleEntry).new(REQUESTS) do |index|
      {index: index.to_i32, plan_index: 0_i32, warmup: index < 2}
    end if resident_plans == 1

    warmups = reverse_warmups ? [1_i32, 0_i32] : [0_i32, 1_i32]
    measured = reverse_first ? [1_i32, 0_i32, 0_i32, 1_i32] : [0_i32, 1_i32, 1_i32, 0_i32]
    entries = (warmups + measured).map_with_index do |plan_index, index|
      {index: index.to_i32, plan_index: plan_index, warmup: index < 2}
    end
    validate_schedule!(entries, resident_plans)
    entries
  end

  def validate_schedule!(entries : Array(ScheduleEntry), resident_plans : Int32) : Nil
    validate_resident_plan_count!(resident_plans)
    unless entries.size == REQUESTS && entries.map(&.[:index]) == (0...REQUESTS).map(&.to_i32)
      raise ArgumentError.new("PixelUMM plan reuse schedule must contain six ordered requests")
    end
    unless entries.all? { |entry| 0 <= entry[:plan_index] < resident_plans } &&
           entries.map(&.[:warmup]) == [true, true, false, false, false, false]
      raise ArgumentError.new("PixelUMM plan reuse schedule has an invalid plan selection or warmup boundary")
    end
    if resident_plans == 1
      raise ArgumentError.new("one-Plan PixelUMM schedule must select Plan zero for every request") unless entries.all? { |entry| entry[:plan_index] == 0 }
    else
      warmups = entries[0...2].map(&.[:plan_index])
      measured = entries[2..].map(&.[:plan_index])
      unless warmups == [0, 1] || warmups == [1, 0]
        raise ArgumentError.new("two-Plan PixelUMM warmups must select each distinct Plan once")
      end
      unless measured == [0, 1, 1, 0] || measured == [1, 0, 0, 1]
        raise ArgumentError.new("two-Plan PixelUMM measured requests must use a balanced ABBA order")
      end
    end
  end

  def validate_shared_bindings!(identities : Array(Hash(String, UInt64))) : Nil
    raise ArgumentError.new("PixelUMM native Plan binding audit is empty") if identities.empty?
    first = identities.first
    identities.each do |identity|
      unless identity.keys.sort == first.keys.sort && identity == first
        raise ArgumentError.new("resident PixelUMM native Plans do not share identical bindings")
      end
    end
  end

  def validate_f32_snapshot_identity!(identity : Hash(String, UInt64)) : Nil
    probability_role = "comparator.local_probabilities_f32_snapshot"
    attended_role = "comparator.local_attended_f32_snapshot"
    probability = identity[probability_role]? || raise ArgumentError.new("PixelUMM direct F32 probability snapshot binding is missing")
    attended = identity[attended_role]? || raise ArgumentError.new("PixelUMM direct F32 attended snapshot binding is missing")
    unless probability > 0_u64 && attended > 0_u64 && probability != attended
      raise ArgumentError.new("PixelUMM direct F32 snapshot bindings must be distinct and non-null")
    end
    other_handles = identity.reject { |role, _| role == probability_role || role == attended_role }.values
    if other_handles.includes?(probability) || other_handles.includes?(attended)
      raise ArgumentError.new("PixelUMM direct F32 snapshot bindings must not alias another owner buffer")
    end
  end

  def validate_distinct_plan_handles!(handles : Array(UInt64), resident_plans : Int32) : Nil
    validate_resident_plan_count!(resident_plans)
    unless handles.size == resident_plans && handles.uniq.size == resident_plans && handles.all? { |handle| handle > 0 }
      raise ArgumentError.new("resident PixelUMM native Plan handles must be distinct and non-null")
    end
  end

  def validate_native_binding_widths!(weights : Int32, core : Int32,
                                      captures : Int32, scratch : Int32) : Nil
    unless weights == 13 && core == 9 && captures == 26 && scratch == 14
      raise ArgumentError.new("PixelUMM native Plan fixed binding-array width mismatch")
    end
  end

  # A pure state machine shared by the Metal owner and CPU-only specifications.
  # A started request remains in flight after any partial submission or failed
  # wait; only its successful sole terminal wait authorizes reads and reuse.
  class SubmissionLifecycle
    @attempted = false
    @terminal_wait_succeeded = false
    @plan_encoded = false
    @comparisons_encoded = false
    @release_allowed = true
    @quarantined = false
    @request_open = false
    @chain_released = true
    @selected_plan_index : Int32? = nil
    @terminal_wait_attempted = false

    def begin_request!(plan_index : Int32, resident_plans : Int32) : Nil
      PixelUMMPlanReuseProbe.validate_resident_plan_count!(resident_plans)
      raise ArgumentError.new("PixelUMM request selected an invalid native Plan index") unless 0 <= plan_index < resident_plans
      raise ArgumentError.new("PixelUMM request chain is quarantined after an earlier failure") if @quarantined
      if @request_open && !@chain_released
        raise ArgumentError.new("prior PixelUMM plan reuse request has not released its command chain")
      end
      if @attempted && !@terminal_wait_succeeded
        raise ArgumentError.new("PixelUMM prior request has no successful sole terminal wait")
      end
      @request_open = true
      @chain_released = false
      @selected_plan_index = plan_index
      @attempted = false
      @terminal_wait_succeeded = false
      @plan_encoded = false
      @comparisons_encoded = false
      @terminal_wait_attempted = false
      @release_allowed = true
    end

    def require_selected_plan!(plan_index : Int32) : Nil
      unless @request_open && @selected_plan_index == plan_index
        raise ArgumentError.new("PixelUMM Plan selection does not match the active request")
      end
    end

    def mark_attempt! : Nil
      raise ArgumentError.new("PixelUMM submission has no active request") unless @request_open
      raise ArgumentError.new("cannot submit after PixelUMM terminal completion") if @terminal_wait_succeeded
      @attempted = true
      @release_allowed = false
    end

    def mark_plan_encoded!(plan_index : Int32) : Nil
      require_selected_plan!(plan_index)
      raise ArgumentError.new("PixelUMM native Plan was already encoded for this request") if @plan_encoded
      raise ArgumentError.new("PixelUMM Plan encode was not marked in flight") unless @attempted
      raise ArgumentError.new("PixelUMM native Plan encode completed after its terminal wait") if @terminal_wait_succeeded
      @plan_encoded = true
    end

    def mark_comparisons_encoded! : Nil
      raise ArgumentError.new("PixelUMM GPU comparisons require one completed native Plan encode") unless @plan_encoded
      raise ArgumentError.new("PixelUMM GPU comparisons were already encoded") if @comparisons_encoded
      raise ArgumentError.new("PixelUMM comparison encode was not marked in flight") unless @attempted
      raise ArgumentError.new("PixelUMM GPU comparisons completed after their terminal wait") if @terminal_wait_succeeded
      @comparisons_encoded = true
    end

    def mark_terminal_wait!(*, completed : Bool, wait_count : Int32) : Nil
      if wait_count > 0
        if @terminal_wait_attempted
          @quarantined = true
          @release_allowed = false unless @terminal_wait_succeeded
          raise ArgumentError.new("PixelUMM request already consumed its sole terminal wait")
        end
        @terminal_wait_attempted = true
      end
      unless @request_open && @attempted && completed && wait_count == 1 && !@terminal_wait_succeeded
        @quarantined = true if wait_count > 0
        raise ArgumentError.new("PixelUMM release requires one successful request terminal wait")
      end
      @terminal_wait_succeeded = true
      @release_allowed = true
    end

    def require_readable!(*, completed : Bool, wait_count : Int32) : Nil
      unless @request_open && @attempted && @terminal_wait_succeeded && completed && wait_count == 1
        raise ArgumentError.new("PixelUMM diagnostic reads require the request's sole successful terminal wait")
      end
    end

    def mark_chain_released! : Nil
      unless @request_open && @chain_released == false && (@terminal_wait_succeeded || !@attempted)
        raise ArgumentError.new("PixelUMM command chain cannot be retired while request work is in flight")
      end
      @chain_released = true
    end

    # A successful drain after a partial request may authorize teardown, but a
    # failed request must never reuse its resident bindings for another request.
    def abort! : Nil
      @quarantined = true
    end

    def quarantined? : Bool
      @quarantined
    end

    def chain_released? : Bool
      @chain_released
    end

    def request_open? : Bool
      @request_open
    end

    def plan_encoded? : Bool
      @plan_encoded
    end

    def comparisons_encoded? : Bool
      @comparisons_encoded
    end

    def attempted? : Bool
      @attempted
    end

    def terminal_wait_succeeded? : Bool
      @terminal_wait_succeeded
    end

    def release_allowed? : Bool
      !@attempted || @terminal_wait_succeeded
    end
  end

  {% unless flag?(:cpu_only) %}
    class Owner
      getter named_bytes : Int64
      getter borrowed_bytes : Int64
      getter expected_bytes : Int64
      getter metric_bytes : Int64
      getter tile_count : Int32
      getter resident_plans : Int32
      getter shared_gate_up : Bool
      getter prepare_softmax_pv : Bool
      getter check_softmax_pv_f32 : Bool
      getter f32_geometry : F32Geometry?
      getter resources : Hash(String, Int64)
      getter binding_bytes : Hash(String, Int64)
      getter binding_identity : Array(Hash(String, UInt64))
      getter memory : PixelUMMMemoryProbe::Recorder
      getter lifecycle : SubmissionLifecycle
      getter encode_named_allocation_bytes : Int64 = 0_i64

      def native_plan_handles : Array(UInt64)
        @native_plan_handles.dup
      end

      def fused_softmax_pv_for_plan(plan_index : Int32) : Bool
        PixelUMMPlanReuseProbe.fused_softmax_pv_for_plan!(
          @prepare_softmax_pv, @resident_plans, @shared_gate_up, plan_index)
      end

      CORE_ORDER    = ML::Video::PixelUMMDecoderBlockMPS::Plan::CORE_NAMES.to_a
      WEIGHT_ORDER  = ML::Video::PixelUMMDecoderBlockMPS::Plan::WEIGHT_NAMES.to_a
      CAPTURE_ORDER = ML::Video::PixelUMMDecoderBlockMPS::Plan::CAPTURE_NAMES.to_a

      @fixture : PixelUMMBlockProbe::Fixture
      @tokens : Int32
      @workspace : ML::Video::PixelUMMDecoderBlockMPS::Workspace? = nil
      @native_plans = [] of Pointer(Void)
      @native_plan_handles = [] of UInt64
      @fixture_bindings = {} of String => ML::MetalBuffer
      @capture_buffers = {} of String => ML::MetalBuffer
      @core_buffers = {} of String => ML::MetalBuffer
      @scratch_buffers = [] of ML::MetalBuffer
      @expected_buffers = {} of String => ML::MetalBuffer
      @metrics_buffer : ML::MetalBuffer? = nil
      @probabilities_snapshot_buffer : ML::MetalBuffer? = nil
      @attended_snapshot_buffer : ML::MetalBuffer? = nil
      @pipeline : ML::Metal::ComputePipeline? = nil
      @request_chain : ML::Metal::MPSCommandChain? = nil
      @seen_chains = [] of ML::Metal::MPSCommandChain
      @request_id : Int32? = nil
      @pending_request_id : Int32? = nil
      @request_plan_index : Int32? = nil
      @plan_encode_succeeded = false
      @comparisons_encode_succeeded = false
      @released = false
      @capture_names = PixelUMMConnectedResourceProbe::CAPTURE_NAMES
      @profile_markers = Slice(PixelUMMDecoderBlockMPSFFI::PhaseMarker).new(11) do
        PixelUMMDecoderBlockMPSFFI::PhaseMarker.new
      end
      @error_text = Bytes.new(1024, 0_u8)
      @owner_thread_id : UInt64

      def initialize(@fixture : PixelUMMBlockProbe::Fixture, maximum : Int64,
                     @resident_plans : Int32, @shared_gate_up : Bool = false,
                     @memory : PixelUMMMemoryProbe::Recorder = PixelUMMMemoryProbe::Recorder.system,
                     *, @prepare_softmax_pv : Bool = false,
                     @check_softmax_pv_f32 : Bool = false)
        @lifecycle = SubmissionLifecycle.new
        @owner_thread_id = Thread.current.object_id.to_u64
        PixelUMMPlanReuseProbe.validate_softmax_pv_configuration!(
          @resident_plans, @shared_gate_up, @prepare_softmax_pv)
        PixelUMMPlanReuseProbe.validate_softmax_pv_f32_configuration!(
          @prepare_softmax_pv, @check_softmax_pv_f32, @resident_plans, @shared_gate_up)
        @tokens = @fixture.grid.reduce(1) { |product, dim| product * dim }
        PixelUMMPlanReuseProbe.validate_geometry!(@fixture.grid, @tokens)
        PixelUMMConnectedResourceProbe.validate_fixture!(@fixture, @tokens)
        budget = PixelUMMPlanReuseProbe.validate_budget!(@tokens, @resident_plans, maximum,
          @check_softmax_pv_f32)
        @f32_geometry = @check_softmax_pv_f32 ? PixelUMMPlanReuseProbe.softmax_pv_f32_geometry(@tokens) : nil
        @borrowed_bytes = budget[:borrowed_bytes]
        @expected_bytes = budget[:expected_bytes]
        @metric_bytes = budget[:metric_bytes]
        @named_bytes = budget[:total_bytes]
        @tile_count = PixelUMMConnectedResourceProbe.config(@tokens).layout.tile_count
        @resources = {} of String => Int64
        @binding_bytes = {} of String => Int64
        @binding_identity = [] of Hash(String, UInt64)

        @memory.emit("prebuffers", phase_bytes_kind: "planned_named_metal_bytes",
          phase_bytes: @named_bytes)
        begin
          raise "Metal unavailable" unless ML::Metal::Device.init!
          setup_workspace!(budget[:plan_bytes])
          upload_fixture_bindings!
          upload_expected_captures!
          setup_comparator!
          prepare_native_plans!
          validate_resource_inventory!(budget)
        rescue error
          cleanup_partial_construction!
          raise error
        end
      end

      def named_resources_json : JSON::Any
        JSON.parse({named_bytes: @named_bytes, resources: @resources,
                    binding_bytes_by_role: @binding_bytes,
                    binding_identity_by_plan: @binding_identity,
                    native_plan_handles: @native_plan_handles,
                    native_plan_handles_distinct: @native_plan_handles.uniq.size == @resident_plans,
                    resident_plan_count: @resident_plans,
                    opaque_native_plan_bytes: nil,
                    opaque_native_plan_bytes_semantics: "unknown; excluded from the named MetalBuffer budget",
        }.to_json)
      end

      def preencode!(request : Int32) : Nil
        check_live!
        raise ArgumentError.new("PixelUMM preencode request index must be non-negative") if request < 0
        if @request_chain && !@lifecycle.chain_released?
          raise ArgumentError.new("prior PixelUMM command chain has not been released")
        end
        if @pending_request_id
          raise ArgumentError.new("PixelUMM preencode snapshot was repeated before request begin")
        end
        @pending_request_id = request
        @memory.emit("preencode", request: request,
          phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: @named_bytes)
      end

      def begin_request!(chain : ML::Metal::MPSCommandChain, plan_index : Int32) : Nil
        check_live!
        request = @pending_request_id || raise ArgumentError.new("PixelUMM preencode snapshot is required before a request")
        raise ArgumentError.new("PixelUMM request selected an invalid native Plan index") unless 0 <= plan_index < @resident_plans
        if @seen_chains.any? { |seen| seen.same?(chain) }
          raise ArgumentError.new("PixelUMM request cannot reuse a command chain")
        end
        @lifecycle.begin_request!(plan_index, @resident_plans)
        @seen_chains << chain
        @request_chain = chain
        @request_id = request
        @request_plan_index = plan_index
        @pending_request_id = nil
        @plan_encode_succeeded = false
        @comparisons_encode_succeeded = false
      end

      # The runner owns request-level metric validation. If any check fails,
      # quarantine this Owner: a successful drain can authorize teardown but
      # cannot authorize another request on these resident native Plans.
      def abort_request! : Nil
        check_live!
        @lifecycle.abort!
      end

      def poison_captures_and_clear_metrics!(chain : ML::Metal::MPSCommandChain) : Nil
        validate_request_chain!(chain)
        raise ArgumentError.new("PixelUMM capture poison was already submitted for this request") if @lifecycle.attempted?
        @lifecycle.mark_attempt! # A partial blit still owns every aliased buffer until its sole wait.
        paused = false
        blit = Pointer(Void).null
        begin
          chain.pause!
          paused = true
          blit = MetalDispatchFFI.create_blit_encoder(chain.root_handle)
          raise "PixelUMM probe poison blit unavailable" if blit.null?
          seen = Set(UInt64).new
          @capture_names.each do |name|
            buffer = @capture_buffers[name]
            next unless seen.add?(buffer.handle.address.to_u64)
            MetalDispatchFFI.blit_fill_buffer(blit, buffer.handle, 255_u8, 0_i64, buffer.size)
          end
          metric_buffer = metrics_buffer
          MetalDispatchFFI.blit_fill_buffer(blit, metric_buffer.handle, 0_u8, 0_i64, metric_buffer.size)
        ensure
          MetalDispatchFFI.blit_end_encoding(blit) unless blit.null?
          chain.resume! if paused
        end
      end

      def encode!(chain : ML::Metal::MPSCommandChain, plan_index : Int32) : Nil
        validate_request_chain!(chain)
        @lifecycle.require_selected_plan!(plan_index)
        raise ArgumentError.new("PixelUMM capture poison must precede native Plan encode") unless @lifecycle.attempted?
        raise ArgumentError.new("PixelUMM native Plan was already encoded for this request") if @lifecycle.plan_encoded?
        raise ArgumentError.new("PixelUMM profiled encode requires an enabled API profile") unless chain.api_profile_enabled?
        native = @native_plans[plan_index]
        before = ML::MetalBuffer.stats
        @lifecycle.mark_attempt! # Native MPS may have submitted work before returning an error.
        paused = false
        status = -1
        begin
          chain.pause!
          paused = true
          status = PixelUMMDecoderBlockMPSFFI.encode_profiled(native, chain.handle,
            chain.api_profile_handle, @profile_markers.to_unsafe,
            @profile_markers.size.to_u32, @error_text.to_unsafe, @error_text.size)
        ensure
          chain.resume! if paused
        end
        after = ML::MetalBuffer.stats
        @encode_named_allocation_bytes = after[:live_bytes] - before[:live_bytes]
        unless @encode_named_allocation_bytes == 0
          raise ArgumentError.new("PixelUMM native Plan encode changed named MetalBuffer live bytes")
        end
        raise ArgumentError.new("PixelUMM native Plan encode failed (#{status}): #{error_message}") unless status == 0
        @lifecycle.mark_plan_encoded!(plan_index)
        @plan_encode_succeeded = true
      end

      def encode_comparisons!(chain : ML::Metal::MPSCommandChain) : Nil
        validate_request_chain!(chain)
        raise ArgumentError.new("PixelUMM comparisons require one successful native Plan encode") unless @plan_encode_succeeded
        raise ArgumentError.new("PixelUMM comparisons were already encoded for this request") if @comparisons_encode_succeeded
        @lifecycle.mark_attempt! # Set before the first binding/dispatch can partially submit.
        before = ML::MetalBuffer.stats
        @capture_names.each_with_index do |name, slot|
          record = @fixture.captures[name]
          encode_comparison!(chain, actual: @capture_buffers[name], expected: @expected_buffers[name],
            slot: slot.to_i32, elements: record.nbytes // 2)
        end
        output = @fixture.captures["output"]
        encode_comparison!(chain, actual: @capture_buffers["output"],
          expected: @expected_buffers["output"], slot: 7,
          elements: output.nbytes // 2, inject_nonfinite: true)
        after = ML::MetalBuffer.stats
        @encode_named_allocation_bytes += after[:live_bytes] - before[:live_bytes]
        unless after[:live_bytes] == before[:live_bytes]
          raise ArgumentError.new("PixelUMM GPU comparator encode changed named MetalBuffer live bytes")
        end
        @lifecycle.mark_comparisons_encoded!
        @comparisons_encode_succeeded = true
      end

      def postencode!(request : Int32) : Nil
        require_request_id!(request)
        unless @plan_encode_succeeded && @comparisons_encode_succeeded
          raise ArgumentError.new("PixelUMM postencode snapshot requires native Plan and GPU comparator encodes")
        end
        raise ArgumentError.new("PixelUMM encode allocated named buffers") unless @encode_named_allocation_bytes == 0
        @memory.emit("postencode", request: request,
          phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: @named_bytes)
      end

      def postwait!(chain : ML::Metal::MPSCommandChain, request : Int32) : Nil
        validate_request_chain!(chain)
        require_request_id!(request)
        @lifecycle.mark_terminal_wait!(completed: chain.completed?, wait_count: chain.wait_count)
        raise ArgumentError.new("PixelUMM postwait requires completed GPU comparisons") unless @comparisons_encode_succeeded
        @memory.emit("postwait", request: request,
          phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: @named_bytes)
      end

      def after_chain_release!(request : Int32) : Nil
        check_live!
        require_request_id!(request)
        @lifecycle.mark_chain_released!
        @memory.emit("after_chain_release", request: request,
          phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: @named_bytes)
      end

      def metrics(chain : ML::Metal::MPSCommandChain) : Array(PixelUMMConnectedResourceProbe::Metric)
        validate_request_chain!(chain)
        @lifecycle.require_readable!(completed: chain.completed?, wait_count: chain.wait_count)
        raise ArgumentError.new("PixelUMM GPU comparator metrics are unavailable") unless @comparisons_encode_succeeded
        metric_slots = @check_softmax_pv_f32 ? F32_CHECK_METRIC_SLOTS : PixelUMMConnectedResourceProbe::METRIC_SLOTS
        words = Array(UInt32).new(metric_slots * PixelUMMConnectedResourceProbe::METRIC_WORDS, 0_u32)
        metrics_buffer.read_bytes(words.to_unsafe.as(UInt8*), @metric_bytes.to_i32)
        Array.new(metric_slots) do |index|
          {failures: words[index * 4], different_words: words[index * 4 + 1],
           nonfinite: words[index * 4 + 2], elements: words[index * 4 + 3]}
        end
      end

      def phase_profile(chain : ML::Metal::MPSCommandChain) : ML::Video::PixelUMMBlockPhaseProfile
        validate_request_chain!(chain)
        @lifecycle.require_readable!(completed: chain.completed?, wait_count: chain.wait_count)
        raise ArgumentError.new("PixelUMM phase profile requires a successful profiled Plan encode") unless @plan_encode_succeeded
        markers = @profile_markers.map_with_index do |raw, index|
          unless raw.phase == index
            raise ArgumentError.new("PixelUMM native phase inventory is incomplete")
          end
          ML::Video::PixelUMMBlockPhaseMarker.new(
            ML::Video::PixelUMMBlockPhaseProfile::MARKER_NAMES[index],
            raw.wall_ns.to_f64 / 1e9, raw.thread_cpu_ns.to_f64 / 1e9,
            raw.root_count)
        end.to_a
        plan_index = @request_plan_index || raise ArgumentError.new("PixelUMM phase profile has no selected Plan")
        ML::Video::PixelUMMBlockPhaseProfile.build(markers,
          chain.completed_root_timings, chain.gpu_elapsed_seconds,
          chain.api_profile, @tile_count, shared_gate_up: @shared_gate_up,
          fused_softmax_pv: fused_softmax_pv_for_plan(plan_index),
          check_softmax_pv_f32: @check_softmax_pv_f32)
      end

      def release_after_terminal! : Nil
        check_live!
        unless @lifecycle.attempted? && @lifecycle.terminal_wait_succeeded? && @lifecycle.release_allowed?
          raise ArgumentError.new("PixelUMM resident Plans require their successful sole terminal wait before release")
        end
        release_resources!
      end

      def release_unsubmitted! : Nil
        return if @released
        check_live!
        unless @lifecycle.release_allowed?
          raise ArgumentError.new("PixelUMM objects cannot be released after submission without terminal completion")
        end
        release_resources!
      end

      private def setup_workspace!(plan_bytes : Int64) : Nil
        config = PixelUMMConnectedResourceProbe.config(@tokens)
        expected_plan_bytes = ML::Video::PixelUMMDecoderBlockMPS::Plan.owned_bytes(
          config, config.layout, PixelUMMConnectedResourceProbe::CAPTURE_NAMES)
        raise ArgumentError.new("PixelUMM reuse named budget differs from production Plan accounting") unless expected_plan_bytes == plan_bytes
        workspace = ML::Video::PixelUMMDecoderBlockMPS::Workspace.new(config,
          captures_by_slot: [PixelUMMConnectedResourceProbe::CAPTURE_NAMES],
          max_owned_bytes: plan_bytes)
        unless workspace.slot_count == 1 && workspace.named_bytes == plan_bytes
          workspace.release
          raise ArgumentError.new("PixelUMM reuse requires one unattached production Workspace")
        end
        @workspace = workspace
        @core_buffers = workspace.core_for_slot(0)
        @capture_buffers = workspace.captures_for_slot(0)
        @scratch_buffers = workspace.scratch_buffers
        @resources.merge!(workspace.named_bytes_by_name)
        @resources.each { |role, bytes| @binding_bytes[role] = bytes }
        @memory.emit("after_shared_workspace_alloc", phase_bytes_kind: "owned_named_metal_bytes",
          phase_bytes: workspace.named_bytes)
      end

      private def upload_fixture_bindings! : Nil
        records = {"x" => @fixture.inputs["x"], "cos" => @fixture.inputs["cos"],
                   "sin" => @fixture.inputs["sin"]}
        records.each do |name, record|
          buffer = ML::MetalBuffer.new(record.nbytes)
          @fixture_bindings[name] = buffer
          upload_record!(record, buffer, "fixture input #{name}")
          add_resource!("fixture.input.#{name}", buffer)
        end
        WEIGHT_ORDER.each do |name|
          record = @fixture.weights[name]
          buffer = ML::MetalBuffer.new(record.nbytes)
          @fixture_bindings[name] = buffer
          upload_record!(record, buffer, "fixture weight #{name}")
          add_resource!("fixture.weight.#{name}", buffer)
        end
        actual_bytes = @fixture_bindings.values.sum(&.size)
        raise ArgumentError.new("PixelUMM fixture binding byte inventory mismatch") unless actual_bytes == @borrowed_bytes
        @memory.emit("after_fixture_binding_upload", phase_bytes_kind: "owned_named_metal_bytes",
          phase_bytes: @resources.values.sum)
      end

      private def upload_expected_captures! : Nil
        @capture_names.each do |name|
          record = @fixture.captures[name]
          buffer = ML::MetalBuffer.new(record.nbytes)
          @expected_buffers[name] = buffer
          upload_record!(record, buffer, "expected capture #{name}")
          add_resource!("expected.capture.#{name}", buffer)
        end
        actual_bytes = @expected_buffers.values.sum(&.size)
        raise ArgumentError.new("PixelUMM expected capture byte inventory mismatch") unless actual_bytes == @expected_bytes
        @memory.emit("after_expected_capture_upload", phase_bytes_kind: "owned_named_metal_bytes",
          phase_bytes: @resources.values.sum)
      end

      private def setup_comparator! : Nil
        @metrics_buffer = ML::MetalBuffer.new(@metric_bytes)
        add_resource!("comparator.metrics", metrics_buffer)
        if geometry = @f32_geometry
          PixelUMMPlanReuseProbe.validate_f32_snapshot_sizes!(geometry,
            @scratch_buffers[4].size, @scratch_buffers[5].size)
          @probabilities_snapshot_buffer = ML::MetalBuffer.new(geometry[:probabilities_snapshot_bytes])
          @attended_snapshot_buffer = ML::MetalBuffer.new(geometry[:attended_snapshot_bytes])
          add_resource!("comparator.local_probabilities_f32_snapshot",
            probabilities_snapshot_buffer)
          add_resource!("comparator.local_attended_f32_snapshot",
            attended_snapshot_buffer)
        end
        @pipeline = ML::Metal::ComputePipeline.new("pixelumm_ffn_compare_bf16",
          PixelUMMConnectedResourceProbe::SOURCE)
        @memory.emit("after_gpu_comparator_compile", phase_bytes_kind: "owned_named_metal_bytes",
          phase_bytes: @resources.values.sum)
      end

      private def prepare_native_plans! : Nil
        config = PixelUMMConnectedResourceProbe.config(@tokens)
        bindings = native_binding_arrays
        identity = binding_identity_for(bindings)
        @resident_plans.times do |plan_index|
          @memory.emit("preplan", slot: plan_index, phase_bytes_kind: "owned_named_metal_bytes",
            phase_bytes: @resources.values.sum)
          native = create_native_plan(config, bindings)
          if native.null?
            raise ArgumentError.new("PixelUMM native Plan preparation failed: #{error_message}")
          end
          @native_plans << native
          @native_plan_handles << native.address.to_u64
          unless @error_text[0] == 0_u8
            raise ArgumentError.new("PixelUMM native Plan preparation returned an error: #{error_message}")
          end
          @error_text.fill(0_u8)
          status = PixelUMMDecoderBlockMPSFFI.set_shared_gate_up(native,
            @shared_gate_up ? 1 : 0, @error_text.to_unsafe, @error_text.size)
          unless status == 0
            raise ArgumentError.new("PixelUMM native Plan FFN selection failed (#{status}): #{error_message}")
          end
          if @prepare_softmax_pv
            @error_text.fill(0_u8)
            fused = PixelUMMPlanReuseProbe.fused_softmax_pv_for_plan!(
              true, @resident_plans, @shared_gate_up, plan_index)
            status = PixelUMMDecoderBlockMPSFFI.plan_prepare_softmax_pv(
              native, fused ? 1 : 0, @error_text.to_unsafe, @error_text.size)
            unless status == 0
              raise ArgumentError.new("PixelUMM native softmax/PV preparation failed (#{status}): #{error_message}")
            end
          end
          if @check_softmax_pv_f32
            @error_text.fill(0_u8)
            status = PixelUMMDecoderBlockMPSFFI.plan_prepare_softmax_pv_check(
              native, probabilities_snapshot_buffer.handle,
              attended_snapshot_buffer.handle, metrics_buffer.handle,
              @error_text.to_unsafe, @error_text.size)
            unless status == 0
              raise ArgumentError.new("PixelUMM native direct F32 check preparation failed (#{status}): #{error_message}")
            end
          end
          @binding_identity << identity.dup
          PixelUMMPlanReuseProbe.validate_shared_bindings!(@binding_identity)
          PixelUMMPlanReuseProbe.validate_distinct_plan_handles!(@native_plan_handles, plan_index + 1)
          @memory.emit("after_plan_build", slot: plan_index,
            phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: @resources.values.sum)
        end
        PixelUMMPlanReuseProbe.validate_distinct_plan_handles!(@native_plan_handles, @resident_plans)
      end

      private def native_binding_arrays
        input = @fixture_bindings["x"].handle
        cosine = @fixture_bindings["cos"].handle
        sine = @fixture_bindings["sin"].handle
        weights = WEIGHT_ORDER.map { |name| @fixture_bindings[name].handle }.to_a
        cores = CORE_ORDER.map do |name|
          @core_buffers[name]?.try(&.handle) || raise ArgumentError.new("PixelUMM workspace core binding is missing: #{name}")
        end
        captures = CAPTURE_ORDER.map do |name|
          @capture_buffers[name]?.try(&.handle) || Pointer(Void).null
        end
        capture_mask = 0_u64
        CAPTURE_ORDER.each_with_index do |name, index|
          capture_mask |= 1_u64 << index if @capture_buffers.has_key?(name)
        end
        scratch = @scratch_buffers.map(&.handle).to_a
        PixelUMMPlanReuseProbe.validate_native_binding_widths!(weights.size,
          cores.size, captures.size, scratch.size)
        {input, cosine, sine, weights, cores, captures, capture_mask, scratch}
      end

      private def binding_identity_for(bindings) : Hash(String, UInt64)
        input, cosine, sine, weights, cores, captures, _, scratch = bindings
        identity = {
          "fixture.input.x"   => input.address.to_u64,
          "fixture.input.cos" => cosine.address.to_u64,
          "fixture.input.sin" => sine.address.to_u64,
        }
        WEIGHT_ORDER.each_with_index do |name, index|
          identity["fixture.weight.#{name}"] = weights[index].address.to_u64
        end
        CORE_ORDER.each_with_index do |name, index|
          identity["workspace.core.#{name}"] = cores[index].address.to_u64
        end
        CAPTURE_ORDER.each_with_index do |name, index|
          handle = captures[index]
          identity["workspace.capture.#{name}"] = handle.null? ? 0_u64 : handle.address.to_u64
        end
        scratch.each_with_index do |handle, index|
          identity["workspace.scratch.#{index}"] = handle.address.to_u64
        end
        @expected_buffers.each { |name, buffer| identity["expected.capture.#{name}"] = buffer.handle.address.to_u64 }
        identity["comparator.metrics"] = metrics_buffer.handle.address.to_u64
        if @check_softmax_pv_f32
          identity["comparator.local_probabilities_f32_snapshot"] = probabilities_snapshot_buffer.handle.address.to_u64
          identity["comparator.local_attended_f32_snapshot"] = attended_snapshot_buffer.handle.address.to_u64
          PixelUMMPlanReuseProbe.validate_f32_snapshot_identity!(identity)
        end
        @binding_bytes["native.input.x"] = @fixture_bindings["x"].size
        @binding_bytes["native.input.cos"] = @fixture_bindings["cos"].size
        @binding_bytes["native.input.sin"] = @fixture_bindings["sin"].size
        WEIGHT_ORDER.each do |name|
          @binding_bytes["native.weight.#{name}"] = @fixture_bindings[name].size
        end
        CORE_ORDER.each do |name|
          @binding_bytes["native.core.#{name}"] = @core_buffers[name].size
        end
        CAPTURE_ORDER.each do |name|
          @binding_bytes["native.capture.#{name}"] = @capture_buffers[name]?.try(&.size) || 0_i64
        end
        scratch.each_with_index do |_, index|
          @binding_bytes["native.scratch.#{index}"] = @scratch_buffers[index].size
        end
        @expected_buffers.each do |name, buffer|
          @binding_bytes["expected.capture.#{name}"] = buffer.size
        end
        @binding_bytes["comparator.metrics"] = metrics_buffer.size
        if @check_softmax_pv_f32
          @binding_bytes["comparator.local_probabilities_f32_snapshot"] = probabilities_snapshot_buffer.size
          @binding_bytes["comparator.local_attended_f32_snapshot"] = attended_snapshot_buffer.size
        end
        unless identity["workspace.core.output"] == identity["workspace.capture.output"]
          raise ArgumentError.new("PixelUMM output capture must alias the shared workspace output")
        end
        identity
      end

      private def create_native_plan(config : ML::Video::PixelUMMDecoderBlockMPS::Config,
                                     bindings) : Pointer(Void)
        input, cosine, sine, weights, cores, captures, capture_mask, scratch = bindings
        layout = config.layout
        @error_text.fill(0_u8)
        PixelUMMDecoderBlockMPSFFI.plan_create_with_shared_gate_up(
          ML::Metal::Device.instance.device_handle,
          config.temporal, config.height, config.width,
          config.tile_tokens, config.model_dim, config.ff_dim,
          config.query_heads, config.kv_heads, config.head_dim,
          config.rms_norm_eps, input, cosine, sine,
          weights.to_unsafe, cores.to_unsafe, captures.to_unsafe, capture_mask,
          scratch.to_unsafe, layout.max_query_tokens, layout.max_key_tokens,
          layout.tile_count, 1, 1, @error_text.to_unsafe, @error_text.size)
      end

      private def encode_comparison!(chain : ML::Metal::MPSCommandChain, *,
                                     actual : ML::MetalBuffer, expected : ML::MetalBuffer,
                                     slot : Int32, elements : Int64,
                                     inject_nonfinite : Bool = false) : Nil
        encoder = chain.encoder
        encoder.set_pipeline(@pipeline || raise ArgumentError.new("PixelUMM GPU comparator pipeline is unavailable"))
        encoder.set_buffer(actual, 0)
        encoder.set_buffer(expected, 1)
        encoder.set_buffer(metrics_buffer, 2, ML::Metal::BufferAccess::ReadWrite, slot.to_i64 * 16)
        encoder.set_value(elements.to_u32, 3)
        encoder.set_value(elements.to_u32, 4)
        encoder.set_value(inject_nonfinite ? 1_u32 : 0_u32, 5)
        encoder.dispatch_threadgroups({(elements.to_i32 + 255) // 256, 1, 1}, {256, 1, 1})
      end

      private def metrics_buffer : ML::MetalBuffer
        @metrics_buffer || raise ArgumentError.new("PixelUMM comparator metrics buffer is unavailable")
      end

      private def probabilities_snapshot_buffer : ML::MetalBuffer
        @probabilities_snapshot_buffer || raise ArgumentError.new("PixelUMM probability F32 snapshot buffer is unavailable")
      end

      private def attended_snapshot_buffer : ML::MetalBuffer
        @attended_snapshot_buffer || raise ArgumentError.new("PixelUMM attended F32 snapshot buffer is unavailable")
      end

      private def workspace : ML::Video::PixelUMMDecoderBlockMPS::Workspace
        @workspace || raise ArgumentError.new("PixelUMM shared Workspace is unavailable")
      end

      private def upload_record!(record : PixelUMMBlockProbe::Record,
                                 buffer : ML::MetalBuffer, label : String) : Nil
        unless buffer.valid? && buffer.storage_mode.shared? && buffer.size == record.nbytes
          raise ArgumentError.new("PixelUMM #{label} must use one exact-size shared MetalBuffer")
        end
        digest = Digest::SHA256.new
        written = 0_i64
        chunk_size = 1024 * 1024
        File.open(@fixture.payload) do |io|
          io.seek(record.offset)
          while written < record.nbytes
            amount = Math.min(chunk_size.to_i64, record.nbytes - written).to_i
            slice = Slice.new(buffer.contents.as(UInt8*) + written.to_i, amount)
            io.read_fully(slice)
            digest.update(slice)
            written += amount
          end
        end
        raise ArgumentError.new("PixelUMM #{label} fixture record hash mismatch") unless digest.hexfinal == record.sha256
      end

      private def add_resource!(role : String, buffer : ML::MetalBuffer) : Nil
        raise ArgumentError.new("PixelUMM resource role was allocated twice: #{role}") if @resources.has_key?(role)
        @resources[role] = buffer.size
        @binding_bytes[role] = buffer.size
      end

      private def validate_resource_inventory!(budget : Budget) : Nil
        unless @resources.values.sum == @named_bytes &&
               @resources.values.sum { |bytes| bytes } == budget[:total_bytes] &&
               @borrowed_bytes == budget[:borrowed_bytes] &&
               @expected_buffers.values.sum(&.size) == budget[:expected_bytes] &&
               metrics_buffer.size == budget[:metric_bytes] && workspace.named_bytes == budget[:plan_bytes]
          raise ArgumentError.new("PixelUMM role-keyed named allocation inventory does not match preflight")
        end
        if geometry = @f32_geometry
          PixelUMMPlanReuseProbe.validate_f32_snapshot_sizes!(geometry,
            probabilities_snapshot_buffer.size, attended_snapshot_buffer.size)
          unless probabilities_snapshot_buffer.valid? && probabilities_snapshot_buffer.storage_mode.shared? &&
                 attended_snapshot_buffer.valid? && attended_snapshot_buffer.storage_mode.shared?
            raise ArgumentError.new("PixelUMM direct F32 snapshots must use exact shared MetalBuffers")
          end
        elsif @probabilities_snapshot_buffer || @attended_snapshot_buffer
          raise ArgumentError.new("PixelUMM direct F32 snapshots exist outside correctness mode")
        end
        unless @fixture_bindings.size == 16 && @native_plans.size == @resident_plans
          raise ArgumentError.new("PixelUMM reuse owner does not hold the exact shared fixture/Plan inventory")
        end
      end

      private def validate_request_chain!(chain : ML::Metal::MPSCommandChain) : Nil
        check_live!
        current = @request_chain || raise ArgumentError.new("PixelUMM request has not begun")
        raise ArgumentError.new("PixelUMM operation used a different command chain") unless current.same?(chain)
      end

      private def require_request_id!(request : Int32) : Nil
        raise ArgumentError.new("PixelUMM request index does not match the active request") unless @request_id == request
      end

      private def check_live! : Nil
        current_thread_id = Thread.current.object_id.to_u64
        unless current_thread_id == @owner_thread_id
          raise ArgumentError.new("PixelUMM plan reuse Owner is confined to its creating thread (expected #{@owner_thread_id}, observed #{current_thread_id})")
        end
        raise ArgumentError.new("PixelUMM plan reuse owner has been released") if @released
      end

      private def error_message : String
        String.new(@error_text.to_unsafe)
      end

      private def cleanup_partial_construction! : Nil
        # Native plans retain handles into Workspace and borrowed fixture buffers.
        # Release every partially prepared Plan before releasing those buffers.
        @native_plans.reverse_each do |native|
          begin
            PixelUMMDecoderBlockMPSFFI.plan_release(native)
          rescue
          end
        end
        @native_plans.clear
        @workspace.try do |workspace|
          begin
            workspace.release
          rescue
          end
        end
        @fixture_bindings.each_value do |buffer|
          begin
            buffer.release
          rescue
          end
        end
        @expected_buffers.each_value do |buffer|
          begin
            buffer.release
          rescue
          end
        end
        begin
          @metrics_buffer.try(&.release)
        rescue
        end
        begin
          @probabilities_snapshot_buffer.try(&.release)
        rescue
        end
        begin
          @attended_snapshot_buffer.try(&.release)
        rescue
        end
        @fixture_bindings.clear
        @expected_buffers.clear
        @released = true
      end

      private def release_resources! : Nil
        @released = true
        cleanup = PixelUMMMemoryProbe::Cleanup.new
        cleanup.phase(@memory, "before_native_plan_release")
        @native_plans.reverse_each do |native|
          cleanup.attempt { PixelUMMDecoderBlockMPSFFI.plan_release(native) }
          cleanup.phase(@memory, "after_native_plan_release")
        end
        @native_plans.clear
        cleanup.phase(@memory, "before_shared_workspace_release")
        cleanup.attempt { workspace.release }
        cleanup.phase(@memory, "after_shared_workspace_release")
        @expected_buffers.each_value { |buffer| cleanup.attempt { buffer.release } }
        @expected_buffers.clear
        cleanup.attempt { metrics_buffer.release }
        @metrics_buffer = nil
        cleanup.attempt { probabilities_snapshot_buffer.release } if @probabilities_snapshot_buffer
        @probabilities_snapshot_buffer = nil
        cleanup.attempt { attended_snapshot_buffer.release } if @attended_snapshot_buffer
        @attended_snapshot_buffer = nil
        @fixture_bindings.each_value { |buffer| cleanup.attempt { buffer.release } }
        @fixture_bindings.clear
        cleanup.phase(@memory, "after_fixture_binding_release")
        cleanup.phase(@memory, "after_all_named_buffers_release",
          phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: 0_i64)
        cleanup.raise_first!
      end
    end
  {% end %}
end
