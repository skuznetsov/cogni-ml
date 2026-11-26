require "./metal_mps_command_chain"

{% unless flag?(:cpu_only) %}
  lib MetalStageIOFFI
    fun create = gs_stage_io_create(device : Void*) : Void*
    fun plan = gs_stage_io_plan(owner : Void*, buffers : Void**, paths : UInt8**, offsets : Int64*, sizes : Int64*, count : Int32, max_batch_bytes : Int64) : Void*
    fun plan_batch_count = gs_stage_io_plan_batch_count(plan : Void*) : Int32
    fun begin_batch = gs_stage_io_begin_batch(owner : Void*, plan : Void*, batch : Int32, root : Void*) : Int32
    fun retire = gs_stage_io_retire(owner : Void*, root : Void*) : Int32
    fun finish = gs_stage_io_finish(owner : Void*, stages : Int32*, batches : Int32*) : Int32
    fun release_plan = gs_stage_io_release_plan(plan : Void*) : Int32
    fun release = gs_stage_io_release(owner : Void*) : Int32
  end
{% end %}

module ML::Video
  # Immutable setup metadata. Source files must remain unchanged until the
  # terminal completion; the checkpoint loader owns identity/hash admission.
  struct StageLoad
    getter buffer : ML::MetalBuffer
    getter source_path : String
    getter source_offset : Int64
    getter bytes : Int64

    def initialize(@buffer, @source_path, @source_offset, @bytes)
      raise ArgumentError.new("Stage load requires a live Metal buffer") unless @buffer.valid?
      raise ArgumentError.new("Stage source must use an absolute path") unless Path[@source_path].absolute?
      raise ArgumentError.new("Stage source path contains NUL") if @source_path.includes?('\0')
      raise ArgumentError.new("Invalid stage source range") unless @source_offset >= 0 && @bytes > 0 && @bytes <= @buffer.size
    end
  end

  class StageIOPlan
    getter native : Pointer(Void)
    getter loads : Array(StageLoad)
    getter owner : StageIO
    getter batch_count : Int32

    def initialize(@owner, @native, loads : Array(StageLoad), @batch_count)
      @loads = loads.dup
    end

    def release : Nil
      return if @native.null?
      {% unless flag?(:cpu_only) %}
        raise ArgumentError.new("Stage I/O plan is still in flight") unless MetalStageIOFFI.release_plan(@native) == 0
      {% end %}
      @native = Pointer(Void).null
    end
  end

  # Serial file -> GPU-pool staging ordered against a single compute chain.
  # No host wait and no activation copy occur here. I/O completion notifications
  # publish readiness (and wake failed stages so terminal rejection can drain).
  class StageIO
    @native : Pointer(Void) = Pointer(Void).null
    @plans = [] of StageIOPlan
    @chain : ML::Metal::MPSCommandChain? = nil
    @started = false
    getter? open : Bool = false
    getter? finished : Bool = false
    getter stage_count : Int32 = 0
    getter batch_count : Int32 = 0
    getter max_batch_bytes : Int64

    def initialize(@max_batch_bytes : Int64 = 512_i64 * 1024 * 1024)
      raise ArgumentError.new("Stage I/O batch limit must be positive") unless @max_batch_bytes > 0
      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        raise "Metal not available" unless ML::Metal::Device.init!
        @native = MetalStageIOFFI.create(ML::Metal::Device.instance.device_handle)
        raise "Unable to create Metal stage I/O" if @native.null?
      {% end %}
    end

    def plan(loads : Array(StageLoad)) : StageIOPlan
      check_live!
      raise ArgumentError.new("Prepare all stage plans before beginning I/O") if @started
      raise ArgumentError.new("Stage plan requires at least one load") if loads.empty?
      handles = loads.map(&.buffer.handle)
      paths = loads.map(&.source_path.to_unsafe)
      offsets = loads.map(&.source_offset)
      sizes = loads.map(&.bytes)
      {% if flag?(:cpu_only) %}
        raise "Metal disabled (cpu_only)"
      {% else %}
        native = MetalStageIOFFI.plan(@native, handles.to_unsafe, paths.to_unsafe, offsets.to_unsafe, sizes.to_unsafe, loads.size, @max_batch_bytes)
        raise ArgumentError.new("Invalid stage I/O plan: source bounds, device, storage or destination overlap") if native.null?
        result = StageIOPlan.new(self, native, loads, MetalStageIOFFI.plan_batch_count(native))
        @plans << result
        result
      {% end %}
    end

    # Pause the chain before both event operations. Resume after begin; retire
    # after all use of that phase, then optionally continue asynchronously.
    def begin!(plan : StageIOPlan, chain : ML::Metal::MPSCommandChain) : Nil
      check_live!
      raise ArgumentError.new("A stage is already open") if @open
      raise ArgumentError.new("Stage plan belongs to another I/O owner or was released") unless plan.owner.same?(self) && !plan.native.null?
      bind_chain!(chain)
      plan.batch_count.times do |batch|
        root = chain.root_handle
        {% unless flag?(:cpu_only) %}
          status = MetalStageIOFFI.begin_batch(@native, plan.native, batch, root)
          raise ArgumentError.new("Stage I/O source changed before enqueue") if status == -4
          raise "Unable to encode stage I/O batch (#{status})" unless status == 0
        {% end %}
        @started = true
        @open = true
        unless batch == plan.batch_count - 1
          # Bound each readiness wait as well as each I/O submission. Retire
          # this load-only root before the next batch, without a host wait.
          retire!(chain)
          chain.continue_async!
        end
      end
    end

    def retire!(chain : ML::Metal::MPSCommandChain) : Nil
      check_live!
      raise ArgumentError.new("No stage is open") unless @open
      bind_chain!(chain)
      root = chain.root_handle
      {% unless flag?(:cpu_only) %}
        raise "Unable to retire stage I/O" unless MetalStageIOFFI.retire(@native, root) == 0
      {% end %}
      @open = false
    end

    def finish!(chain : ML::Metal::MPSCommandChain) : Nil
      return if @finished
      check_live!
      bind_chain!(chain)
      raise ArgumentError.new("Retire the open stage before terminal completion") if @open
      raise ArgumentError.new("Complete the compute chain before checking stage I/O") unless chain.completed?
      status = 0
      {% unless flag?(:cpu_only) %}
        status = MetalStageIOFFI.finish(@native, pointerof(@stage_count), pointerof(@batch_count))
      {% end %}
      raise "Stage I/O has not drained after compute completion" if status == -2
      @finished = true
      raise "Stage I/O failed; the generated result must be discarded (#{status})" unless status == 0
    end

    def release : Nil
      return if @native.null?
      raise ArgumentError.new("Finish stage I/O before release") if @started && !@finished
      @plans.each(&.release)
      @plans.clear
      {% unless flag?(:cpu_only) %}
        raise "Stage I/O release rejected" unless MetalStageIOFFI.release(@native) == 0
      {% end %}
      @native = Pointer(Void).null
    end

    private def bind_chain!(chain : ML::Metal::MPSCommandChain) : Nil
      if old = @chain
        raise ArgumentError.new("Stage I/O cannot switch compute chains") unless old.same?(chain)
      else
        @chain = chain
      end
    end

    private def check_live! : Nil
      raise ArgumentError.new("Stage I/O is finished or released") if @finished || @native.null?
    end
  end
end
