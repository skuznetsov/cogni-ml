require "json"
require "digest/sha256"
require "set"
require "../../src/ml/video/pixelumm_decoder_block_mps"
require "./pixelumm_block_probe"
require "./pixelumm_memory_probe"
require "./hy_nomas_probe_instrumentation"

module PixelUMMConnectedResourceProbe
  extend self

  SCHEMA          = "cogni_ml.pixelumm_connected_resource_probe/v1"
  ATOL            = 2e-5_f64
  RTOL            = 2e-5_f64
  HIDDEN          =     4096
  FFN             =    12288
  GIB             = 1024_i64 * 1024 * 1024
  MAX_EVENTS      = 4096
  METRIC_SLOTS    =    8
  METRIC_WORDS    =    4
  METRIC_BYTES    = METRIC_SLOTS.to_i64 * METRIC_WORDS * sizeof(UInt32)
  MAX_NAMED_BYTES = 8_i64 * GIB
  SOURCE          = {{ read_file("#{__DIR__}/pixelumm_ffn_dense.metal") }}

  CAPTURE_NAMES = [
    "attention", "post_attn_norm", "gate_proj", "up_proj", "product", "down_proj", "output",
  ]
  GRAPH_NAMES = {
    1_u32 => "input_norm_pre_gain", 2_u32 => "input_norm_gain",
    3_u32 => "q_projection", 4_u32 => "k_projection", 5_u32 => "v_projection",
    6_u32 => "q_norm_statistics", 7_u32 => "k_norm_statistics",
    8_u32 => "q_norm_inverse", 9_u32 => "k_norm_inverse",
    10_u32 => "attention_qk", 11_u32 => "attention_softmax", 12_u32 => "attention_pv",
    13_u32 => "o_projection", 14_u32 => "post_attn_norm_pre_gain", 15_u32 => "post_attn_norm_gain",
    16_u32 => "gate_projection", 17_u32 => "up_projection",
    18_u32 => "shared_gate_up_projection", 19_u32 => "down_projection",
  }
  PHASE_NAMES = {
    0_u32 => "before_encode", 1_u32 => "after_encode",
    2_u32 => "after_pool_drain", 3_u32 => "after_scope",
  }

  alias Budget = NamedTuple(plan_bytes: Int64, borrowed_bytes: Int64,
    expected_bytes: Int64, metric_bytes: Int64, total_bytes: Int64)
  alias Metric = NamedTuple(failures: UInt32, different_words: UInt32,
    nonfinite: UInt32, elements: UInt32)
  alias ScheduleEntry = NamedTuple(index: Int32, pool_mode: Bool,
    kind: String, warmup: Bool, shared_gate_up: Bool)

  @[Extern]
  struct NativeEvent
    property monotonic_ns : UInt64
    property device_allocated_bytes : UInt64
    property process_phys_footprint_bytes : UInt64
    property process_rss_bytes : UInt64
    property graph_id : UInt32
    property graph_index : UInt32
    property phase : UInt32
    property pool_mode : UInt32

    def initialize
      @monotonic_ns = 0_u64
      @device_allocated_bytes = 0_u64
      @process_phys_footprint_bytes = 0_u64
      @process_rss_bytes = 0_u64
      @graph_id = 0_u32
      @graph_index = 0_u32
      @phase = 0_u32
      @pool_mode = 0_u32
    end
  end

  @[Extern]
  struct NativeInfo
    property event_count : UInt32
    property overflow : UInt32
    property sample_failed : UInt32
    property unknown_graphs : UInt32
    property graph_calls : UInt32

    def initialize
      @event_count = 0_u32
      @overflow = 0_u32
      @sample_failed = 0_u32
      @unknown_graphs = 0_u32
      @graph_calls = 0_u32
    end
  end

  def capture_shape(name : String, tokens : Int32) : Array(Int32)
    raise ArgumentError.new("unsupported connected PixelUMM token count") unless {2048, 21120}.includes?(tokens)
    case name
    when "attention"                             then [1, tokens, 32, 128]
    when "post_attn_norm", "down_proj", "output" then [1, tokens, HIDDEN]
    when "gate_proj", "up_proj", "product"       then [1, tokens, FFN]
    else                                              raise ArgumentError.new("unknown connected PixelUMM capture: #{name}")
    end
  end

  def config(tokens : Int32) : ML::Video::PixelUMMDecoderBlockMPS::Config
    case tokens
    when  2048 then ML::Video::PixelUMMDecoderBlockMPS::Config.new(temporal: 2, height: 32, width: 32)
    when 21120 then ML::Video::PixelUMMDecoderBlockMPS::Config.new(temporal: 6, height: 44, width: 80)
    else            raise ArgumentError.new("unsupported connected PixelUMM token count")
    end
  end

  def budget(tokens : Int32) : Budget
    cfg = config(tokens)
    plan_bytes = ML::Video::PixelUMMDecoderBlockMPS::Plan.owned_bytes(cfg,
      cfg.layout, CAPTURE_NAMES)
    input_bytes = tokens.to_i64 * HIDDEN * 2 + 2_i64 * tokens * 128 * 4
    weights_bytes = PixelUMMBlockProbe.weight_shapes.values.sum do |shape|
      shape.reduce(1_i64) { |product, dim| product * dim } * 2
    end
    borrowed_bytes = input_bytes + weights_bytes
    expected_bytes = CAPTURE_NAMES.sum do |name|
      capture_shape(name, tokens).reduce(1_i64) { |product, dim| product * dim } * 2
    end
    metric_bytes = METRIC_BYTES
    {plan_bytes: plan_bytes, borrowed_bytes: borrowed_bytes,
     expected_bytes: expected_bytes, metric_bytes: metric_bytes,
     total_bytes: plan_bytes + borrowed_bytes + expected_bytes + metric_bytes}
  end

  def validate_budget!(tokens : Int32, maximum : Int64) : Budget
    unless maximum > 0 && maximum <= MAX_NAMED_BYTES
      raise ArgumentError.new("connected PixelUMM named-memory cap must be between one byte and 8 GiB")
    end
    required = budget(tokens)
    raise ArgumentError.new("connected PixelUMM buffers exceed cap before Metal allocation") if required[:total_bytes] > maximum
    required
  end

  def schedule(*, shared : Bool, reverse_warmups : Bool, reverse_first : Bool) : Array(ScheduleEntry)
    warmups = reverse_warmups ? [true, false] : [false, true]
    measured = reverse_first ? [true, false, false, true] : [false, true, true, false]
    (warmups + measured).map_with_index do |pool_mode, index|
      warmup = index < 2
      {index: index.to_i32, pool_mode: pool_mode,
       kind: warmup ? "warmup" : "timed", warmup: warmup,
       shared_gate_up: shared}
    end
  end

  def validate_snapshot_inventory!(*, event_count : Int32, overflow : Bool,
                                   sample_failed : Bool, unknown_graphs : Int32,
                                   expected_event_count : Int32? = nil) : Nil
    unless 0 <= event_count <= MAX_EVENTS && !overflow && !sample_failed && unknown_graphs == 0
      raise ArgumentError.new("connected PixelUMM native snapshot inventory is incomplete: events=#{event_count}, overflow=#{overflow}, sample_failed=#{sample_failed}, unknown_graphs=#{unknown_graphs}")
    end
    if expected = expected_event_count
      raise ArgumentError.new("connected PixelUMM native event count mismatch") unless event_count == expected
    end
  end

  def sample_phases(pool_mode : Bool) : Array(UInt32)
    pool_mode ? [0_u32, 1_u32, 2_u32] : [0_u32, 1_u32, 3_u32]
  end

  def expected_graph_ids(tile_count : Int32, shared_gate_up : Bool) : Array(UInt32)
    raise ArgumentError.new("connected PixelUMM attention tile count must be positive") unless tile_count > 0
    ids = [1_u32, 2_u32, 3_u32, 4_u32, 5_u32, 6_u32, 7_u32, 8_u32, 9_u32]
    tile_count.times { ids.concat([10_u32, 11_u32, 12_u32]) }
    ids.concat([13_u32, 14_u32, 15_u32])
    if shared_gate_up
      ids << 18_u32
    else
      ids.concat([16_u32, 17_u32])
    end
    ids << 19_u32
    ids
  end

  def validate_native_events!(events : Array(NativeEvent), info : NativeInfo, *,
                              tile_count : Int32, shared_gate_up : Bool,
                              pool_mode : Bool) : Nil
    expected_ids = expected_graph_ids(tile_count, shared_gate_up)
    phases = sample_phases(pool_mode)
    validate_snapshot_inventory!(event_count: info.event_count.to_i32,
      overflow: info.overflow != 0, sample_failed: info.sample_failed != 0,
      unknown_graphs: info.unknown_graphs.to_i32,
      expected_event_count: expected_ids.size * phases.size)
    unless info.graph_calls.to_i32 == expected_ids.size && events.size == info.event_count.to_i32
      raise ArgumentError.new("connected PixelUMM native graph-call count mismatch")
    end
    previous_ns = 0_u64
    events.each_with_index do |event, index|
      graph_index = index // phases.size
      phase_index = index % phases.size
      unless event.graph_index == graph_index.to_u32 && event.graph_id == expected_ids[graph_index] &&
             event.phase == phases[phase_index] && event.pool_mode == (pool_mode ? 1_u32 : 0_u32) &&
             event.monotonic_ns > 0 && event.monotonic_ns >= previous_ns &&
             event.device_allocated_bytes > 0 && event.process_phys_footprint_bytes > 0 &&
             event.process_rss_bytes > 0
        raise ArgumentError.new("connected PixelUMM native graph-event sequence mismatch")
      end
      previous_ns = event.monotonic_ns
    end
  end

  def event_json(event : NativeEvent) : JSON::Any
    graph = GRAPH_NAMES[event.graph_id]? || raise ArgumentError.new("unknown connected PixelUMM graph ID")
    phase = PHASE_NAMES[event.phase]? || raise ArgumentError.new("unknown connected PixelUMM sample phase")
    JSON.parse({monotonic_ns: event.monotonic_ns,
                timestamp_semantics: "sample-start; readings are sequential and non-atomic",
                device_current_allocated_size_bytes: event.device_allocated_bytes,
                process_phys_footprint_bytes: event.process_phys_footprint_bytes,
                process_rss_bytes: event.process_rss_bytes,
                graph_id: event.graph_id, graph: graph, graph_index: event.graph_index,
                phase_id: event.phase, phase: phase,
                pool_mode: event.pool_mode == 1}.to_json)
  end

  def validate_fixture!(fixture : PixelUMMBlockProbe::Fixture, tokens : Int32) : Nil
    cfg = config(tokens)
    expected_grid = [cfg.temporal, cfg.height, cfg.width]
    raise ArgumentError.new("connected PixelUMM fixture grid/rows mismatch") unless fixture.grid == expected_grid
    unless fixture.capture_order == CAPTURE_NAMES && fixture.captures.keys.sort == CAPTURE_NAMES.sort
      raise ArgumentError.new("connected PixelUMM fixture must contain exactly the seven authenticated captures")
    end
    CAPTURE_NAMES.each do |name|
      record = fixture.captures[name]
      unless record.dtype == "BF16" && record.shape == capture_shape(name, tokens)
        raise ArgumentError.new("connected PixelUMM capture dtype/shape mismatch: #{name}")
      end
    end
  end

  def validate_positive_control!(baseline : Metric, detected : Metric) : Nil
    unless detected[:failures] > 0 && detected[:nonfinite] > 0 &&
           detected[:elements] == baseline[:elements] &&
           detected[:different_words] == baseline[:different_words]
      raise ArgumentError.new("connected PixelUMM GPU comparator missed its injected nonfinite control")
    end
  end

  class SubmissionLifecycle
    @attempted = false
    @terminal_wait_succeeded = false

    def begin_request! : Nil
      if @attempted && !@terminal_wait_succeeded
        raise ArgumentError.new("connected PixelUMM prior request has no successful terminal wait")
      end
      @attempted = false
      @terminal_wait_succeeded = false
    end

    def mark_attempt! : Nil
      raise ArgumentError.new("cannot submit after connected PixelUMM terminal completion") if @terminal_wait_succeeded
      @attempted = true
    end

    def mark_terminal_wait!(*, completed : Bool, wait_count : Int32) : Nil
      unless @attempted && completed && wait_count == 1 && !@terminal_wait_succeeded
        raise ArgumentError.new("connected PixelUMM release requires one successful request wait")
      end
      @terminal_wait_succeeded = true
    end

    def attempted? : Bool
      @attempted
    end

    def terminal_wait_succeeded? : Bool
      @terminal_wait_succeeded
    end

    def require_readable!(*, completed : Bool, wait_count : Int32) : Nil
      unless @attempted && @terminal_wait_succeeded && completed && wait_count == 1
        raise ArgumentError.new("connected PixelUMM reads require the request's sole successful terminal wait")
      end
    end

    def release_allowed? : Bool
      !@attempted || @terminal_wait_succeeded
    end
  end

  {% unless flag?(:cpu_only) %}
    lib PixelUMMConnectedResourceFFI
      struct Info
        event_count : UInt32
        overflow : UInt32
        sample_failed : UInt32
        unknown_graphs : UInt32
        graph_calls : UInt32
      end

      struct Event
        monotonic_ns : UInt64
        device_allocated_bytes : UInt64
        process_phys_footprint_bytes : UInt64
        process_rss_bytes : UInt64
        graph_id : UInt32
        graph_index : UInt32
        phase : UInt32
        pool_mode : UInt32
      end

      fun begin = gs_pixelumm_connected_resource_begin(device : Void*, pool_mode : Int32) : Int32
      fun finish = gs_pixelumm_connected_resource_end(info : Info*) : Int32
      fun event = gs_pixelumm_connected_resource_event(index : UInt32, event : Event*) : Int32
    end

    class Owner
      getter named_bytes : Int64
      getter borrowed_bytes : Int64
      getter expected_bytes : Int64
      getter metric_bytes : Int64
      getter tile_count : Int32
      getter resources : Hash(String, Int64)
      getter memory : PixelUMMMemoryProbe::Recorder
      getter lifecycle : SubmissionLifecycle
      @session : PixelUMMBlockProbe::Session? = nil

      @expected = {} of String => ML::MetalBuffer
      @metrics : ML::MetalBuffer?
      @pipeline : ML::Metal::ComputePipeline?
      @released = false
      @native_active = false
      @tokens : Int32
      @tile_count : Int32

      def initialize(@fixture : PixelUMMBlockProbe::Fixture, maximum : Int64,
                     shared_gate_up : Bool, @memory : PixelUMMMemoryProbe::Recorder)
        @tokens = @fixture.grid.reduce(1) { |product, dim| product * dim }
        PixelUMMConnectedResourceProbe.validate_fixture!(@fixture, @tokens)
        budget = PixelUMMConnectedResourceProbe.validate_budget!(@tokens, maximum)
        @borrowed_bytes = budget[:borrowed_bytes]
        @expected_bytes = budget[:expected_bytes]
        @metric_bytes = budget[:metric_bytes]
        @named_bytes = budget[:total_bytes]
        @resources = {
          "block_plan"                             => budget[:plan_bytes],
          "borrowed_fixture_inputs_and_13_weights" => @borrowed_bytes,
          "expected_capture_buffers"               => @expected_bytes,
          "gpu_comparator_metrics"                 => @metric_bytes,
        }
        @lifecycle = SubmissionLifecycle.new
        @tile_count = PixelUMMConnectedResourceProbe.config(@tokens).layout.tile_count
        raise "Metal unavailable" unless ML::Metal::Device.init!
        unless sizeof(NativeEvent) == 48 && sizeof(NativeInfo) == 20 &&
               sizeof(PixelUMMConnectedResourceFFI::Event) == 48 && sizeof(PixelUMMConnectedResourceFFI::Info) == 20
          raise ArgumentError.new("connected PixelUMM native resource ABI size mismatch")
        end
        @memory.emit("prebuffers", phase_bytes_kind: "planned_named_metal_bytes", phase_bytes: @named_bytes)
        begin
          session_cap = maximum - @expected_bytes - @metric_bytes
          @session = PixelUMMBlockProbe::Session.new(@fixture, session_cap,
            ffn_f32: true, prepare_shared_gate_up: true)
          session.plan.use_shared_gate_up!(shared_gate_up)
          unless session.borrowed_bytes == @borrowed_bytes &&
                 session.named_bytes == @borrowed_bytes + budget[:plan_bytes]
            raise ArgumentError.new("connected PixelUMM Plan byte inventory differs from its preflight")
          end
          @memory.emit("after_fixture_upload_and_block_plan_compile",
            phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: session.named_bytes)

          CAPTURE_NAMES.each do |name|
            record = @fixture.captures[name]
            buffer = ML::MetalBuffer.new(record.nbytes)
            @expected[name] = buffer
            upload_expected!(record, buffer)
          end
          @metrics = ML::MetalBuffer.new(@metric_bytes)
          @pipeline = ML::Metal::ComputePipeline.new("pixelumm_ffn_compare_bf16", SOURCE)
          @memory.emit("after_expected_upload_and_gpu_comparator_compile",
            phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: @named_bytes)
        rescue error
          release_unsubmitted!
          raise error
        end
      end

      def preencode!(request : Int32) : Nil
        @memory.emit("preencode", request: request,
          phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: @named_bytes)
      end

      def poison_captures_and_clear_metrics!(chain : ML::Metal::MPSCommandChain) : Nil
        @lifecycle.mark_attempt!
        chain.pause!
        paused = true
        blit = Pointer(Void).null
        begin
          @lifecycle.mark_attempt!
          blit = MetalDispatchFFI.create_blit_encoder(chain.root_handle)
          raise "connected PixelUMM poison blit unavailable" if blit.null?
          seen = Set(UInt64).new
          CAPTURE_NAMES.each do |name|
            buffer = session.plan.stages[name]
            address = buffer.handle.address.to_u64
            next unless seen.add?(address)
            @lifecycle.mark_attempt!
            MetalDispatchFFI.blit_fill_buffer(blit, buffer.handle, 255_u8, 0_i64, buffer.size)
          end
          metric_buffer = metrics_buffer
          @lifecycle.mark_attempt!
          MetalDispatchFFI.blit_fill_buffer(blit, metric_buffer.handle, 0_u8, 0_i64, metric_buffer.size)
        ensure
          unless blit.null?
            @lifecycle.mark_attempt!
            MetalDispatchFFI.blit_end_encoding(blit)
          end
          if paused
            chain.resume!
            paused = false
          end
        end
      end

      def begin_graph_events!(pool_mode : Bool) : Nil
        raise ArgumentError.new("connected PixelUMM graph-event collection is already active") if @native_active
        status = PixelUMMConnectedResourceFFI.begin(ML::Metal::Device.instance.device_handle, pool_mode ? 1 : 0)
        raise ArgumentError.new("connected PixelUMM graph-event collection could not begin (#{status})") unless status == 0
        @native_active = true
      end

      def native_collection_active? : Bool
        @native_active
      end

      def finish_graph_events! : NativeInfo
        raise ArgumentError.new("connected PixelUMM graph-event collection is inactive") unless @native_active
        info = uninitialized PixelUMMConnectedResourceFFI::Info
        status = PixelUMMConnectedResourceFFI.finish(pointerof(info))
        @native_active = false
        raise ArgumentError.new("connected PixelUMM graph-event collection failed (#{status})") unless status == 0
        native_info = NativeInfo.new
        native_info.event_count = info.event_count
        native_info.overflow = info.overflow
        native_info.sample_failed = info.sample_failed
        native_info.unknown_graphs = info.unknown_graphs
        native_info.graph_calls = info.graph_calls
        native_info
      end

      def read_graph_events!(info : NativeInfo, pool_mode : Bool,
                             chain : ML::Metal::MPSCommandChain) : Array(NativeEvent)
        @lifecycle.require_readable!(completed: chain.completed?, wait_count: chain.wait_count)
        PixelUMMConnectedResourceProbe.validate_snapshot_inventory!(event_count: info.event_count.to_i32,
          overflow: info.overflow != 0, sample_failed: info.sample_failed != 0,
          unknown_graphs: info.unknown_graphs.to_i32,
          expected_event_count: PixelUMMConnectedResourceProbe.expected_graph_ids(@tile_count, session.plan.profiled_shared_gate_up).size * 3)
        events = Array(NativeEvent).new(info.event_count.to_i32) do |index|
          raw = uninitialized PixelUMMConnectedResourceFFI::Event
          status = PixelUMMConnectedResourceFFI.event(index.to_u32, pointerof(raw))
          raise ArgumentError.new("connected PixelUMM graph-event read failed (#{status})") unless status == 0
          event = NativeEvent.new
          event.monotonic_ns = raw.monotonic_ns
          event.device_allocated_bytes = raw.device_allocated_bytes
          event.process_phys_footprint_bytes = raw.process_phys_footprint_bytes
          event.process_rss_bytes = raw.process_rss_bytes
          event.graph_id = raw.graph_id
          event.graph_index = raw.graph_index
          event.phase = raw.phase
          event.pool_mode = raw.pool_mode
          event
        end
        PixelUMMConnectedResourceProbe.validate_native_events!(events, info, tile_count: @tile_count,
          shared_gate_up: session.plan.profiled_shared_gate_up, pool_mode: pool_mode)
        events
      end

      def encode_comparisons!(chain : ML::Metal::MPSCommandChain) : Nil
        CAPTURE_NAMES.each_with_index do |name, slot|
          encode_comparison!(chain, actual: session.plan.stages[name], expected: @expected[name],
            slot: slot.to_i32, elements: @fixture.captures[name].nbytes // 2)
        end
        output_record = @fixture.captures["output"]
        encode_comparison!(chain, actual: session.plan.stages["output"], expected: @expected["output"],
          slot: 7, elements: output_record.nbytes // 2, inject_nonfinite: true)
      end

      def metrics(chain : ML::Metal::MPSCommandChain) : Array(Metric)
        @lifecycle.require_readable!(completed: chain.completed?, wait_count: chain.wait_count)
        words = Array(UInt32).new(METRIC_SLOTS * METRIC_WORDS, 0_u32)
        metrics_buffer.read_bytes(words.to_unsafe.as(UInt8*), @metric_bytes.to_i32)
        Array.new(METRIC_SLOTS) do |index|
          {failures: words[index * 4], different_words: words[index * 4 + 1],
           nonfinite: words[index * 4 + 2], elements: words[index * 4 + 3]}
        end
      end

      def postencode!(request : Int32) : Nil
        @memory.emit("postencode", request: request,
          phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: @named_bytes)
      end

      def postwait!(chain : ML::Metal::MPSCommandChain, request : Int32) : Nil
        @lifecycle.mark_terminal_wait!(completed: chain.completed?, wait_count: chain.wait_count)
        @memory.emit("postwait", request: request,
          phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: @named_bytes)
      end

      def after_chain_release!(request : Int32) : Nil
        @memory.emit("after_chain_release", request: request,
          phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: @named_bytes)
      end

      def release_after_terminal! : Nil
        return if @released
        unless @lifecycle.release_allowed?
          raise ArgumentError.new("connected PixelUMM named buffers cannot be released before a successful terminal wait")
        end
        session.plan.release
        @memory.emit("after_block_plan_release",
          phase_bytes_kind: "owned_named_metal_bytes",
          phase_bytes: @borrowed_bytes + @expected_bytes + @metric_bytes)
        @expected.each_value(&.release)
        @expected.clear
        @metrics.try(&.release)
        @metrics = nil
        session.release
        @memory.emit("after_all_named_buffers_release",
          phase_bytes_kind: "owned_named_metal_bytes", phase_bytes: 0_i64)
        @released = true
      end

      def release_unsubmitted! : Nil
        return if @released
        unless @lifecycle.release_allowed?
          raise ArgumentError.new("connected PixelUMM objects cannot be released after submission without terminal completion")
        end
        @session.try(&.release)
        @expected.each_value(&.release)
        @expected.clear
        @metrics.try(&.release)
        @metrics = nil
        @released = true
      end

      def native_event_json(events : Array(NativeEvent)) : Array(JSON::Any)
        events.map { |event| PixelUMMConnectedResourceProbe.event_json(event) }
      end

      def api_profile_json(profile : ML::Video::MPSAPIProfile) : Array(JSON::Any)
        profile.phases.map do |phase|
          operations = {} of String => JSON::Any
          phase.operations.each do |name, stats|
            operations[name] = JSON.parse({count: stats.count, wall_total_ns: stats.wall_total_ns,
                                           wall_max_ns: stats.wall_max_ns, thread_cpu_total_ns: stats.thread_cpu_total_ns,
                                           thread_cpu_max_ns: stats.thread_cpu_max_ns, nested_count: stats.nested_count,
                                           nested_wall_total_ns: stats.nested_wall_total_ns,
                                           nested_thread_cpu_total_ns: stats.nested_thread_cpu_total_ns,
                                           exclusive_wall_total_ns: stats.exclusive_wall_total_ns,
                                           exclusive_thread_cpu_total_ns: stats.exclusive_thread_cpu_total_ns}.to_json)
          end
          JSON.parse({phase_id: phase.phase_id, phase: phase.name, operations: operations}.to_json)
        end
      end

      private def metrics_buffer : ML::MetalBuffer
        @metrics || raise ArgumentError.new("connected PixelUMM comparator metrics buffer is unavailable")
      end

      def session : PixelUMMBlockProbe::Session
        @session || raise ArgumentError.new("connected PixelUMM block session is unavailable")
      end

      private def upload_expected!(record : PixelUMMBlockProbe::Record,
                                   buffer : ML::MetalBuffer) : Nil
        hash = Digest::SHA256.new
        File.open(@fixture.payload) do |io|
          io.seek(record.offset)
          chunk_size = 1024 * 1024
          written = 0_i64
          while written < record.nbytes
            amount = Math.min(chunk_size.to_i64, record.nbytes - written).to_i32
            slice = Slice.new(buffer.contents.as(UInt8*) + written.to_i, amount)
            io.read_fully(slice)
            hash.update(slice)
            written += amount
          end
        end
        unless hash.hexfinal == record.sha256
          raise ArgumentError.new("connected PixelUMM expected capture content changed")
        end
      end

      private def encode_comparison!(chain : ML::Metal::MPSCommandChain, *,
                                     actual : ML::MetalBuffer, expected : ML::MetalBuffer,
                                     slot : Int32, elements : Int64,
                                     inject_nonfinite : Bool = false) : Nil
        pipeline = @pipeline || raise ArgumentError.new("connected PixelUMM GPU comparator is unavailable")
        @lifecycle.mark_attempt!
        encoder = chain.encoder
        @lifecycle.mark_attempt!
        encoder.set_pipeline(pipeline)
        @lifecycle.mark_attempt!
        encoder.set_buffer(actual, 0)
        @lifecycle.mark_attempt!
        encoder.set_buffer(expected, 1)
        @lifecycle.mark_attempt!
        encoder.set_buffer(metrics_buffer, 2, ML::Metal::BufferAccess::ReadWrite, slot.to_i64 * 16)
        @lifecycle.mark_attempt!
        encoder.set_value(elements.to_u32, 3)
        @lifecycle.mark_attempt!
        encoder.set_value(elements.to_u32, 4)
        @lifecycle.mark_attempt!
        encoder.set_value(inject_nonfinite ? 1_u32 : 0_u32, 5)
        @lifecycle.mark_attempt!
        encoder.dispatch_threadgroups({(elements.to_i32 + 255) // 256, 1, 1}, {256, 1, 1})
      end
    end
  {% end %}
end
