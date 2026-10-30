# F32 Hy-Nomas continuation attention. Each segment owns one compact workspace
# and one reusable plan shared across its layer-cache inventory. History is a
# packed, segment-major list of valid rows; no padded keys are exposed.
require "./hy_nomas_block"
require "./metal_mps_command_chain"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  @[Link(ldflags: "-framework Metal -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph")]
  lib HyNomasHistoryAttentionMPSFFI
    fun workspace_create = gs_hy_nomas_history_attention_workspace_create(
      device : Pointer(Void), history_rows : Int32, query_rows : Int32,
      current_rows : Int32,
      heads : Int32, head_dim : Int32, query_chunk : Int32, scale : Float32,
      query_stage : Pointer(Void), key_stage : Pointer(Void), value_stage : Pointer(Void),
      output_stage : Pointer(Void),
    ) : Pointer(Void)
    fun workspace_release = gs_hy_nomas_history_attention_workspace_release(
      workspace : Pointer(Void),
    ) : Void
    fun workspace_register_history = gs_hy_nomas_history_attention_workspace_register_history(
      workspace : Pointer(Void), key : Pointer(Void), value : Pointer(Void),
    ) : Int32
    fun workspace_freeze_history = gs_hy_nomas_history_attention_workspace_freeze_history(
      workspace : Pointer(Void), expected_slots : Int32,
    ) : Int32
    fun plan_create = gs_hy_nomas_history_attention_plan_create(
      workspace : Pointer(Void), query : Pointer(Void),
      current_key : Pointer(Void), current_value : Pointer(Void), output : Pointer(Void),
    ) : Pointer(Void)
    fun encode = gs_hy_nomas_history_attention_encode(
      plan : Pointer(Void), command : Pointer(Void), history_slot : Int32,
    ) : Int32
    fun plan_release = gs_hy_nomas_history_attention_plan_release(plan : Pointer(Void)) : Void
  end
{% end %}

module ML::Video
  class HyNomasHistoryAttentionWorkspace
    MAX_QUERY_CHUNK = 256

    record Geometry, history_rows : Int32, query_rows : Int32,
      current_rows : Int32, key_rows : Int32,
      heads : Int32, head_dim : Int32, query_chunk : Int32,
      graph_calls : Int32,
      query_elements : Int32, current_elements : Int32,
      history_elements : Int32, key_elements : Int32,
      max_score_elements : Int64, max_score_bytes : Int64,
      workspace_bytes : Int64, active_history_pair_bytes : Int64,
      current_io_bytes : Int64, no_history_control : Bool

    Q_STAGE_KEY      = "hy_nomas_history_attention.q"
    K_STAGE_KEY      = "hy_nomas_history_attention.k"
    V_STAGE_KEY      = "hy_nomas_history_attention.v"
    OUTPUT_STAGE_KEY = "hy_nomas_history_attention.output"

    @@lifetime_mutex = Mutex.new
    @@live_workspaces = [] of HyNomasHistoryAttentionWorkspace

    getter shape : Geometry
    @released = false
    getter history_slot_count : Int32 = 0
    getter history_inventory_bytes : Int64 = 0_i64
    getter active_history_inventory_bytes : Int64 = 0_i64
    @arena : Arena
    @handle = Pointer(Void).null
    @chain : ML::Metal::MPSCommandChain? = nil
    @plans = 0_i32
    @history_pairs = [] of Tuple(ML::MetalBuffer, ML::MetalBuffer)
    @history_frozen = false

    def released? : Bool
      @released
    end

    # Pure admission calculation: score memory is bounded by one 256-row query
    # tile, while K/V staging holds exactly the valid history plus current rows.
    def self.geometry(history_rows : Int32, query_rows : Int32,
                      heads : Int32, head_dim : Int32,
                      current_rows : Int32 = query_rows) : Geometry
      unless history_rows >= 0 && query_rows > 0 && current_rows > 0 &&
             current_rows <= query_rows && heads > 0 && head_dim > 0
        raise ArgumentError.new("Invalid Hy-Nomas history attention geometry")
      end
      key_rows_64 = history_rows.to_i64 + current_rows
      if key_rows_64 > Int32::MAX
        raise ArgumentError.new("Hy-Nomas history attention key rows exceed the native limit")
      end
      key_rows = key_rows_64.to_i32
      query_chunk = Math.min(query_rows, MAX_QUERY_CHUNK).to_i32
      graph_calls = ((query_rows.to_i64 + query_chunk - 1) // query_chunk).to_i32
      query_elements = HyNomasDualStreamBlock.checked_count(1, query_rows, heads, head_dim)
      current_elements = HyNomasDualStreamBlock.checked_count(1, current_rows, heads, head_dim)
      history_elements = if history_rows == 0
                           0
                         else
                           HyNomasDualStreamBlock.checked_count(1, history_rows, heads, head_dim)
                         end
      key_elements = HyNomasDualStreamBlock.checked_count(1, key_rows, heads, head_dim)
      tile_elements = HyNomasDualStreamBlock.checked_count(1, query_chunk, heads, head_dim)
      max_score_elements = HyNomasDualStreamBlock.checked_count(1, heads, query_chunk, key_rows).to_i64
      max_score_bytes = max_score_elements * sizeof(Float32)
      workspace_elements = 2_i64 * tile_elements + 2_i64 * key_elements
      active_history_elements = 2_i64 * history_elements
      if workspace_elements > Int64::MAX // sizeof(Float32) ||
         active_history_elements > Int64::MAX // sizeof(Float32) ||
         (2_i64 * query_elements + 2_i64 * current_elements) > Int64::MAX // sizeof(Float32)
        raise ArgumentError.new("Hy-Nomas history attention byte count overflow")
      end
      Geometry.new(history_rows, query_rows, current_rows, key_rows, heads, head_dim,
        query_chunk, graph_calls, query_elements, current_elements, history_elements, key_elements,
        max_score_elements, max_score_bytes,
        workspace_elements * sizeof(Float32),
        active_history_elements * sizeof(Float32),
        (2_i64 * query_elements + 2_i64 * current_elements) * sizeof(Float32),
        history_rows == 0 && current_rows == query_rows)
    end

    def self.retained_workspace_count : Int32
      @@lifetime_mutex.synchronize { @@live_workspaces.size.to_i32 }
    end

    def initialize(history_rows : Int32, query_rows : Int32,
                   heads : Int32, head_dim : Int32,
                   current_rows : Int32 = query_rows)
      @shape = self.class.geometry(history_rows, query_rows, heads, head_dim, current_rows)
      @arena = Arena.new
      begin
        @arena.alloc(Q_STAGE_KEY, tile_elements)
        @arena.alloc(K_STAGE_KEY, @shape.key_elements)
        @arena.alloc(V_STAGE_KEY, @shape.key_elements)
        @arena.alloc(OUTPUT_STAGE_KEY, tile_elements)
        validate_stages!
        unless @arena.resident_bytes == @shape.workspace_bytes
          raise ArgumentError.new("Hy-Nomas history attention workspace byte count changed")
        end
        {% if flag?(:darwin) && !flag?(:cpu_only) %}
          @handle = HyNomasHistoryAttentionMPSFFI.workspace_create(
            ML::Metal::Device.instance.device_handle,
            @shape.history_rows, @shape.query_rows, @shape.current_rows, @shape.heads,
            @shape.head_dim, @shape.query_chunk,
            1.0_f32 / Math.sqrt(@shape.head_dim.to_f32),
            q_stage.handle, k_stage.handle, v_stage.handle, output_stage.handle)
          raise "Failed to prepare Hy-Nomas history attention workspace" if @handle.null?
        {% else %}
          raise ArgumentError.new("Hy-Nomas history MPS attention requires macOS Metal Performance Shaders")
        {% end %}
        @@lifetime_mutex.synchronize { @@live_workspaces << self }
      rescue ex
        release_native_workspace
        @arena.release
        @released = true
        raise ex
      end
    end

    # Exact named Metal allocations owned by this workspace. MPSGraph's opaque
    # internal temporaries are not measurable from this API.
    def resident_bytes : Int64
      @arena.resident_bytes
    end

    def q_stage : ML::MetalBuffer
      @arena[Q_STAGE_KEY]
    end

    def k_stage : ML::MetalBuffer
      @arena[K_STAGE_KEY]
    end

    def v_stage : ML::MetalBuffer
      @arena[V_STAGE_KEY]
    end

    def output_stage : ML::MetalBuffer
      @arena[OUTPUT_STAGE_KEY]
    end

    def native_handle : Pointer(Void)
      raise ArgumentError.new("Hy-Nomas history attention workspace has been released") if @released || @handle.null?
      @handle
    end

    # Register one layer's persistent F32 cache. Buffer capacity may exceed P;
    # only the first P contiguous BQHD rows are read by the join kernel.
    def register_history_pair!(key : ML::MetalBuffer,
                               value : ML::MetalBuffer) : Int32
      raise ArgumentError.new("Hy-Nomas history attention workspace has been released") if @released
      raise ArgumentError.new("Freeze the Hy-Nomas history inventory before registering plans") if @history_frozen || @plans > 0
      raise ArgumentError.new("Zero-history attention does not accept cache buffers") if @shape.history_rows == 0
      {key, value}.each do |buffer|
        MetalOps.check(buffer, @shape.history_elements.to_i64)
        unless buffer.storage_mode.shared?
          raise ArgumentError.new("Hy-Nomas history cache requires shared F32 buffers")
        end
        {q_stage, k_stage, v_stage, output_stage}.each do |stage|
          if MetalOps.overlap?(buffer, stage)
            raise ArgumentError.new("Hy-Nomas history cache must not alias workspace")
          end
        end
      end
      if MetalOps.overlap?(key, value)
        raise ArgumentError.new("Hy-Nomas history K/V caches must not alias")
      end
      @history_pairs.each do |previous_key, previous_value|
        {previous_key, previous_value}.each do |previous|
          {key, value}.each do |current|
            if MetalOps.overlap?(previous, current)
              raise ArgumentError.new("Hy-Nomas registered history caches must not alias")
            end
          end
        end
      end
      if key.size > Int64::MAX - value.size ||
         @history_inventory_bytes > Int64::MAX - key.size - value.size ||
         @active_history_inventory_bytes > Int64::MAX - @shape.active_history_pair_bytes
        raise ArgumentError.new("Hy-Nomas registered history byte count overflow")
      end
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        status = HyNomasHistoryAttentionMPSFFI.workspace_register_history(
          native_handle, key.handle, value.handle)
        raise "Failed to register Hy-Nomas history K/V buffers (#{status})" unless status == 0
      {% else %}
        raise ArgumentError.new("Hy-Nomas history MPS attention requires macOS Metal Performance Shaders")
      {% end %}
      @history_pairs << {key, value}
      @history_inventory_bytes += key.size + value.size
      @active_history_inventory_bytes += @shape.active_history_pair_bytes
      @history_slot_count = @history_pairs.size.to_i32
      @history_slot_count - 1
    end

    # Freeze registration before creating the one reusable plan. Encoding only
    # selects an already retained slot; it never changes the cache inventory.
    def freeze_history_inventory! : Nil
      raise ArgumentError.new("Hy-Nomas history attention workspace has been released") if @released
      raise ArgumentError.new("Hy-Nomas history inventory already frozen") if @history_frozen
      if @shape.history_rows > 0 && @history_pairs.empty?
        raise ArgumentError.new("Positive history requires a non-empty registered cache inventory")
      end
      if @shape.history_rows == 0 && !@history_pairs.empty?
        raise ArgumentError.new("Zero-history control must have an empty cache inventory")
      end
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        status = HyNomasHistoryAttentionMPSFFI.workspace_freeze_history(
          native_handle, @history_pairs.size.to_i32)
        raise "Failed to freeze Hy-Nomas history inventory (#{status})" unless status == 0
      {% else %}
        raise ArgumentError.new("Hy-Nomas history MPS attention requires macOS Metal Performance Shaders")
      {% end %}
      @history_frozen = true
    end

    # This workspace is single-chain and serial. Multiple layer plans may use
    # it in one chain; a second chain is admitted only after the first drains.
    def claim_chain!(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("Hy-Nomas history attention workspace has been released") if @released
      if previous = @chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("Hy-Nomas history attention workspace cannot span in-flight command chains")
        end
      end
      @chain = chain
    end

    def register_plan! : Nil
      raise ArgumentError.new("Hy-Nomas history attention workspace has been released") if @released
      raise ArgumentError.new("Freeze the Hy-Nomas history inventory before creating its plan") unless @history_frozen
      raise ArgumentError.new("A history attention workspace supports one reusable plan") if @plans != 0
      @plans += 1
    end

    def unregister_plan! : Nil
      raise ArgumentError.new("Hy-Nomas history attention workspace plan count underflow") if @plans <= 0
      @plans -= 1
    end

    def release : Nil
      return if @released
      raise ArgumentError.new("Release all Hy-Nomas history attention plans first") unless @plans == 0
      if chain = @chain
        unless chain.completed?
          raise ArgumentError.new("Complete Hy-Nomas history attention workspace chain before release")
        end
      end
      release_native_workspace
      @arena.release
      @history_pairs.clear
      @history_slot_count = 0
      @history_inventory_bytes = 0_i64
      @active_history_inventory_bytes = 0_i64
      @released = true
      @@lifetime_mutex.synchronize { @@live_workspaces.reject! { |workspace| workspace.same?(self) } }
    end

    private def tile_elements : Int32
      HyNomasDualStreamBlock.checked_count(1, @shape.query_chunk, @shape.heads, @shape.head_dim)
    end

    private def validate_stages! : Nil
      expected = {
        Q_STAGE_KEY      => tile_elements,
        K_STAGE_KEY      => @shape.key_elements,
        V_STAGE_KEY      => @shape.key_elements,
        OUTPUT_STAGE_KEY => tile_elements,
      }
      expected.each do |name, elements|
        buffer = @arena[name]
        MetalOps.check(buffer, elements.to_i64)
        unless buffer.storage_mode.shared? && @arena.dtype(name).f32?
          raise ArgumentError.new("Hy-Nomas history attention workspace requires shared F32 buffers")
        end
      end
      buffers = {q_stage, k_stage, v_stage, output_stage}
      buffers.each_with_index do |left, i|
        buffers.each_with_index do |right, j|
          next if j <= i
          if MetalOps.overlap?(left, right)
            raise ArgumentError.new("Hy-Nomas history attention workspace buffers must not alias")
          end
        end
      end
    end

    private def release_native_workspace : Nil
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        HyNomasHistoryAttentionMPSFFI.workspace_release(@handle) unless @handle.null?
      {% end %}
      @handle = Pointer(Void).null
    end
  end

  # One segment plan borrowing the workspace and shared layer scratch. Reuse
  # this prepared plan across layers by selecting each setup-registered cache
  # slot. K/V remains exact F32; an optional BF16 boundary needs another mode.
  class HyNomasHistoryAttentionMPSPlan
    @@lifetime_mutex = Mutex.new
    @@live_plans = [] of HyNomasHistoryAttentionMPSPlan

    getter workspace : HyNomasHistoryAttentionWorkspace

    @query : ML::MetalBuffer
    @current_key : ML::MetalBuffer
    @current_value : ML::MetalBuffer
    @output : ML::MetalBuffer
    @handle = Pointer(Void).null
    @chain : ML::Metal::MPSCommandChain? = nil
    @registered_workspace = false
    @released = false

    def self.retained_plan_count : Int32
      @@lifetime_mutex.synchronize { @@live_plans.size.to_i32 }
    end

    def initialize(workspace : HyNomasHistoryAttentionWorkspace,
                   query : ML::MetalBuffer,
                   current_key : ML::MetalBuffer,
                   current_value : ML::MetalBuffer,
                   output : ML::MetalBuffer)
      @workspace = workspace
      @query = query
      @current_key = current_key
      @current_value = current_value
      @output = output
      begin
        validate_caller_bindings!
        @workspace.register_plan!
        @registered_workspace = true
        {% if flag?(:darwin) && !flag?(:cpu_only) %}
          @handle = HyNomasHistoryAttentionMPSFFI.plan_create(
            @workspace.native_handle,
            @query.handle, @current_key.handle, @current_value.handle, @output.handle)
          raise "Failed to prepare Hy-Nomas history MPS attention plan" if @handle.null?
        {% else %}
          raise ArgumentError.new("Hy-Nomas history MPS attention requires macOS Metal Performance Shaders")
        {% end %}
        @@lifetime_mutex.synchronize { @@live_plans << self }
      rescue ex
        release_native_plan
        if @registered_workspace
          workspace.unregister_plan!
          @registered_workspace = false
        end
        @released = true
        raise ex
      end
    end

    # Per-plan explicit buffers are caller-owned; the segment workspace reports
    # its shared, exact named allocation once, separately from Graph temporaries.
    def resident_bytes : Int64
      0_i64
    end

    def validate_bindings!(query : ML::MetalBuffer,
                           current_key : ML::MetalBuffer,
                           current_value : ML::MetalBuffer,
                           output : ML::MetalBuffer) : Nil
      raise ArgumentError.new("Hy-Nomas history attention plan has been released") if @released
      unless @query.same?(query) && @current_key.same?(current_key) && @current_value.same?(current_value) &&
             @output.same?(output)
        raise ArgumentError.new("Hy-Nomas history attention binding changed")
      end
      validate_caller_bindings!
    end

    # Appends MSL packing/gather/scatter and the prepared chunked MPSGraph into
    # the caller's existing chain. `history_slot` indexes the frozen inventory;
    # no-history square control uses the default -1 slot.
    def encode(chain : ML::Metal::MPSCommandChain,
               history_slot : Int32 = -1) : Nil
      raise ArgumentError.new("Hy-Nomas history attention plan has been released") if @released || @handle.null?
      validate_bindings!(@query, @current_key, @current_value, @output)
      if @workspace.shape.history_rows == 0
        unless history_slot == -1
          raise ArgumentError.new("Zero-history control does not select a cache slot")
        end
      elsif history_slot < 0 || history_slot >= @workspace.history_slot_count
        raise ArgumentError.new("Hy-Nomas history slot is outside the frozen cache inventory")
      end
      chain.encoder
      @workspace.claim_chain!(chain)
      if previous = @chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("Hy-Nomas history attention plan cannot span in-flight command chains")
        end
      end
      @chain = chain
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        chain.pause!
        status = -1
        begin
          status = HyNomasHistoryAttentionMPSFFI.encode(@handle, chain.handle, history_slot)
        ensure
          chain.resume!
        end
        raise "Hy-Nomas history MPS attention encode rejected (#{status})" unless status == 0
      {% else %}
        raise ArgumentError.new("Hy-Nomas history MPS attention requires macOS Metal Performance Shaders")
      {% end %}
    end

    def release : Nil
      return if @released
      if chain = @chain
        unless chain.completed?
          raise ArgumentError.new("Complete Hy-Nomas history attention chain before releasing plan")
        end
      end
      release_native_plan
      if @registered_workspace
        @workspace.unregister_plan!
        @registered_workspace = false
      end
      @released = true
      @@lifetime_mutex.synchronize { @@live_plans.reject! { |plan| plan.same?(self) } }
    end

    private def validate_caller_bindings! : Nil
      shape = @workspace.shape
      bindings = [] of Tuple(ML::MetalBuffer, Int32)
      bindings << {@query, shape.query_elements}
      bindings << {@current_key, shape.current_elements}
      bindings << {@current_value, shape.current_elements}
      bindings << {@output, shape.query_elements}
      buffers = [] of ML::MetalBuffer
      bindings.each do |buffer, elements|
        MetalOps.check(buffer, elements.to_i64)
        unless buffer.storage_mode.shared?
          raise ArgumentError.new("Hy-Nomas history attention requires shared F32 buffers")
        end
        buffers << buffer
      end
      buffers.each_with_index do |left, i|
        buffers.each_with_index do |right, j|
          next if j <= i
          if MetalOps.overlap?(left, right)
            raise ArgumentError.new("Hy-Nomas history attention caller buffers must not alias")
          end
        end
      end
      stages = {@workspace.q_stage, @workspace.k_stage,
                @workspace.v_stage, @workspace.output_stage}
      buffers.each do |buffer|
        stages.each do |stage|
          if MetalOps.overlap?(buffer, stage)
            raise ArgumentError.new("Hy-Nomas history attention inputs must not alias shared workspace")
          end
        end
      end
    end

    private def release_native_plan : Nil
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        HyNomasHistoryAttentionMPSFFI.plan_release(@handle) unless @handle.null?
      {% end %}
      @handle = Pointer(Void).null
    end
  end
end
