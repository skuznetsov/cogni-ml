require "./native"
require "./host_profile"
require "./resident_host_encoder"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
require "../../metal/compute_graph"

module ML::Audio::Drama3::Resident
  SOURCE = {{ read_file("#{__DIR__}/kernels/resident.metal") }}
  MLX_Q4_SOURCE = {{ read_file("#{__DIR__}/kernels/mlx_q4.metal") }}

  private class BufferOwner
    getter host_profile = HostProfile.new(ENV["DRAMA3_HOST_PROFILE"]? == "1")

    def view(storage : ML::MetalBuffer, count : Int32, offset : Int64 = 0_i64) : Buffer
      @host_profile.measure(HostEvent::BufferView) { Buffer.new(storage, count, self, offset) }
    end
  end

  # Views retain both allocation and queue-owner identity. The backing Metal
  # handle is internal plumbing: external writes while work is live are unsafe.
  class Buffer
    getter buffer : ML::MetalBuffer, count : Int32, offset : Int64
    getter owner : BufferOwner

    def initialize(@buffer : ML::MetalBuffer, @count : Int32, @owner : BufferOwner, @offset : Int64 = 0_i64)
      raise ArgumentError.new("resident buffer range") unless @count > 0 && @offset >= 0 && @offset + @count.to_i64 * 4 <= @buffer.size
    end

    def slice(start : Int32, count : Int32) : Buffer
      raise ArgumentError.new("resident slice range") unless start >= 0 && count > 0 && start.to_i64 + count <= @count
      @owner.view(@buffer, count, @offset + start.to_i64 * 4)
    end
  end

  # A fixed call sequence has one backing allocation per ordinal, never one
  # shared slot for simultaneously live residuals. Only the owning generator
  # may use these views: a result must be consumed before the next call starts.
  # GPU writes are ordered on Context's serial queue, including across flights.
  class Workspace
    getter owner : BufferOwner
    @slots = [] of Buffer
    @retired = [] of Buffer
    @cursor = 0
    @uses = 0_i64
    @reuses = 0_i64
    @bytes = 0_i64
    @peak_bytes = 0_i64
    @active = false
    @released = false
    @quarantined = false

    def initialize(@owner : BufferOwner)
    end

    def enter : Nil
      raise ArgumentError.new("resident workspace is released or quarantined") if @released || @quarantined
      raise ArgumentError.new("resident workspace is already active") if @active
      @active = true
      @cursor = 0
      @uses += 1
    end

    def leave : Nil
      @active = false
    end

    def alloc(count : Int32, reserve_count : Int32, &fresh : Int32 -> Buffer) : Buffer
      @owner.host_profile.measure(HostEvent::Workspace) do
        raise ArgumentError.new("resident workspace is not active") unless @active
        index = @cursor
        slot = @slots[index]?
        if slot && slot.count >= reserve_count
          @reuses += 1
        else
          replacement = fresh.call(reserve_count)
          if slot
            # Keep replaced backings until a checked terminal boundary as well.
            # A failed queue drain may clear its lease without proving completion;
            # the workspace quarantine must still retain every earlier backing.
            @retired << slot
            @slots[index] = replacement
          else
            @slots << replacement
          end
          slot = replacement
          @bytes += slot.buffer.size
          @peak_bytes = Math.max(@peak_bytes, @bytes)
        end
        @cursor += 1
        slot.count == count ? slot : slot.slice(0, count)
      end
    end

    def release : Nil
      raise ArgumentError.new("resident workspace is active or quarantined") if @active || @quarantined
      @slots.each { |slot| slot.buffer.release }
      @retired.each { |slot| slot.buffer.release }
      @slots.clear
      @retired.clear
      @bytes = 0_i64
      @released = true
    end

    def quarantine : Nil
      @quarantined = true unless @released
    end

    def stats : Hash(String, Int64)
      {"slots" => @slots.size.to_i64, "retired_slots" => @retired.size.to_i64,
       "bytes" => @bytes, "peak_bytes" => @peak_bytes,
       "uses" => @uses, "reuses" => @reuses,
       "released" => @released ? 1_i64 : 0_i64,
       "quarantined" => @quarantined ? 1_i64 : 0_i64}
    end
  end

  # CogniGraph owns this command until checked terminal completion. The wrapper
  # keeps timing/counters on the same wait path, including error-path drains.
  private class Flight
    getter command : ML::Metal::CommandBuffer
    @waited = false
    @timeline_emitted = false
    @host_before_commit_s = 0.0_f64
    @host_after_commit_s = 0.0_f64
    @encoding_resources : HostResourceSnapshot?
    @before_commit_resources : HostResourceSnapshot?
    @after_commit_resources : HostResourceSnapshot?

    def initialize(@command : ML::Metal::CommandBuffer, @index : UInt64, @timeline : Bool, @host_profile : HostProfile, @resources : Bool, &@on_wait : -> Nil)
      @encoding_resources = HostResourceSnapshot.capture if @resources
    end

    def enqueue : Nil
      @command.enqueue
    end

    def commit : Nil
      if @timeline
        @before_commit_resources = HostResourceSnapshot.capture if @resources
        commit_times = @command.commit_with_timeline
        @host_before_commit_s = commit_times[0]
        @host_after_commit_s = commit_times[1]
        @after_commit_resources = HostResourceSnapshot.capture if @resources
      else
        @command.commit
      end
    end

    def discard : Nil
      @command.discard
    end

    def wait : Nil
      unless @waited
        @waited = true
        @on_wait.call
      end
      timeline = nil.as(ML::Metal::CommandTimeline?)
      begin
        @host_profile.measure(HostEvent::CompletionWait) do
          Profile.measure("resident.submit_wait", category: "synchronization") do
            if @timeline
              @command.wait_with_timeline do |completion|
                timeline = completion
                nil
              end
            else
              @command.wait
            end
          end
        end
      ensure
        if completion = timeline
          emit_timeline(completion)
        end
      end
    end

    private def emit_timeline(timeline : ML::Metal::CommandTimeline) : Nil
      return if @timeline_emitted
      @timeline_emitted = true
      begin
        STDERR.puts({
          event: "drama3_gpu_command",
          index: @index,
          completion_status: timeline.completion_status,
          gpu_seconds: timeline.gpu_seconds?,
          gpu_timing_valid: timeline.gpu_timing_valid,
          host_before_commit_s: @host_before_commit_s,
          host_after_commit_s: @host_after_commit_s,
          host_before_wait_s: timeline.host_before_wait_s,
          host_after_wait_s: timeline.host_after_wait_s,
          gpu_start_s: timeline.gpu_timing_valid ? timeline.gpu_start_s : nil,
          gpu_end_s: timeline.gpu_timing_valid ? timeline.gpu_end_s : nil,
          kernel_start_s: timeline.kernel_timing_valid ? timeline.kernel_start_s : nil,
          kernel_end_s: timeline.kernel_timing_valid ? timeline.kernel_end_s : nil,
          kernel_timing_valid: timeline.kernel_timing_valid,
          resources: @resources ? {encoding_begin: @encoding_resources,
                                   before_commit: @before_commit_resources,
                                   after_commit: @after_commit_resources,
                                   after_wait: HostResourceSnapshot.capture} : nil,
        }.to_json)
      rescue
        # Telemetry serialization and output must never change inference result.
        nil
      end
    end
  end

  # One serial encoder preserves dependency order; pending bindings survive GC
  # until checked command completion. Only explicit input/terminal boundaries
  # touch host memory. Diagnostic reads are intentionally visible in the ledger.
  class Context
    # Failed drains do not establish that the GPU has stopped using storage.
    # As with DAC scratch, retain failed workspaces for the process lifetime.
    @@workspace_quarantine = [] of Workspace
    @@weight_bank_quarantine = [] of MetalBackend::WeightBufferBank
    @@workspace_quarantine_mutex = Mutex.new
    getter readbacks = Hash(String, Int64).new(0_i64)
    getter submissions = 0_i64
    getter dense_reduction : String
    getter expert_threads : Int32
    getter expert_packed_rows_enabled : Bool
    @command : ML::Metal::CommandBuffer?
    @encoder : HostEncoder?
    @pending = [] of Buffer
    @weights = {} of UInt64 => Buffer
    @status : Buffer?
    @timeline : Bool
    getter resources_enabled : Bool
    @graph : ML::Metal::GraphSubmissionQueue(Flight, Buffer)?
    @lease : ML::Metal::GraphSubmissionLease(Flight, Buffer)?
    @completion_waits = 0_i64
    @retained_bytes = 0_i64
    @max_retained_bytes = 0_i64
    @aborted = false
    @owner = BufferOwner.new
    @memory_profile : Bool = ENV["DRAMA3_MEMORY_PROFILE"]? == "1"
    @allocations = 0_i64
    @requested_bytes = 0_i64
    @peak_metal_allocated_bytes = 0_i64
    @allocation_profile : Bool = ENV["DRAMA3_ALLOCATION_PROFILE"]? == "1"
    @allocation_ns = 0_i64
    @max_buffer_length : Int64
    @workspace : Workspace?
    @workspaces = [] of Workspace
    @weight_wrapper_bank : MetalBackend::WeightBufferBank?

    def initialize
      raise "resident Metal requires enabled backend" unless MetalBackend.enabled?
      @max_buffer_length = MetalBackend.max_buffer_length
      @dense_reduction = validate_dense_reduction(ENV["DRAMA3_DENSE_REDUCTION"]? || "tree")
      expert_threads = ENV["DRAMA3_EXPERT_THREADS"]? || "64"
      raise ArgumentError.new("DRAMA3_EXPERT_THREADS must be 32 or 64") unless expert_threads.in?("32", "64")
      @expert_threads = expert_threads.to_i
      packed_rows = ENV["DRAMA3_EXPERT_PACKED_ROWS"]? || "0"
      raise ArgumentError.new("DRAMA3_EXPERT_PACKED_ROWS must be 0 or 1") unless packed_rows.in?("0", "1")
      @expert_packed_rows_enabled = packed_rows == "1"
      @resources_enabled = ENV["DRAMA3_METAL_RESOURCES"]? == "1"
      @timeline = ENV["DRAMA3_METAL_TIMELINE"]? == "1" || @resources_enabled
      depth = ENV["DRAMA3_COGNIGRAPH_MAX_INFLIGHT"]? || "0"
      raise ArgumentError.new("DRAMA3_COGNIGRAPH_MAX_INFLIGHT must be 0, 1, or 2") unless depth.in?("0", "1", "2")
      raise ArgumentError.new("DRAMA3_METAL_RESOURCES requires CogniGraph depth 1 or 2") if @resources_enabled && depth == "0"
      unless depth == "0"
        queue = ML::Metal::CommandQueue.new
        @graph = ML::Metal::GraphSubmissionQueue(Flight, Buffer).new(depth.to_i) do |slot, sequence|
          command = host_profile.measure(HostEvent::CommandCreate) { ML::Metal::CommandBuffer.new(queue: queue) }
          Flight.new(command, sequence, @timeline, host_profile, @resources_enabled) { @completion_waits += 1; nil }
        end
      end
    end

    def graph_enabled? : Bool
      !!@graph
    end

    # Explicit diagnostics only. Enable before any dispatch/allocation so no
    # global raw-weight aliases are introduced by this context beforehand.
    def enable_weight_wrapper_probe! : Nil
      raise ArgumentError.new("weight wrapper probe must be enabled before context use") if @weight_wrapper_bank || @submissions > 0 || @allocations > 0 || !@weights.empty?
      check_weight_wrapper_boundary!
      @weight_wrapper_bank = MetalBackend::WeightBufferBank.new
    end

    def prepare_weight_wrappers(refresh : Bool, clear_views : Bool = false) : Hash(String, Int64)
      bank = @weight_wrapper_bank
      raise ArgumentError.new("weight wrapper probe is not enabled") unless bank
      check_weight_wrapper_boundary!
      begin
        bank.prepare(refresh, clear_views)
      rescue ex
        abort
        raise ex
      end
    end

    def begin_weight_wrapper_job! : Nil
      bank = @weight_wrapper_bank
      raise ArgumentError.new("weight wrapper probe is not enabled") unless bank
      check_weight_wrapper_boundary!
      bank.start_first_command(@submissions)
    end

    def prepare_weight_extents(narrow : Bool, clear_views : Bool = false) : Hash(String, Int64)
      bank = @weight_wrapper_bank
      raise ArgumentError.new("weight wrapper probe is not enabled") unless bank
      check_weight_wrapper_boundary!
      begin
        bank.prepare_extents(narrow, clear_views)
      rescue ex
        abort
        raise ex
      end
    end

    def weight_wrapper_first_command_bindings
      bank = @weight_wrapper_bank
      raise ArgumentError.new("weight wrapper probe is not enabled") unless bank
      check_weight_wrapper_boundary!
      bank.first_command_bindings
    end

    # Host object identity only: no Metal query, weight byte or activation read.
    def converted_weight_cache_signature
      @weights.map do |key, value|
        {key, value.object_id, value.buffer.object_id, value.count, value.offset}
      end.sort_by(&.[0])
    end

    private def check_weight_wrapper_boundary! : Nil
      raise ArgumentError.new("resident context aborted") if @aborted
      if @workspace || @encoder || @command || @lease || !@pending.empty? || (@graph.try(&.pending_count) || 0) > 0
        raise ArgumentError.new("weight wrapper preparation requires a terminal queue")
      end
    end

    private def weight_buffer(value : Tensor) : MetalBackend::TensorBuffer
      if bank = @weight_wrapper_bank
        bank.tensor_buffer(value)
      else
        MetalBackend.tensor_buffer(value, true)
      end
    end

    def host_profile : HostProfile
      @owner.host_profile
    end

    def pipeline(name : String, source : String) : ML::Metal::ComputePipeline
      host_profile.measure(HostEvent::Pipeline) { ML::Metal::PipelineCache.get_or_compile(name, source) }
    end

    def new_workspace : Workspace
      raise ArgumentError.new("resident context aborted") if @aborted
      workspace = Workspace.new(@owner)
      @workspaces << workspace
      workspace
    end

    def workspace_active? : Bool
      !!@workspace
    end

    # No submit, wait, host write or read occurs at either scope boundary.
    # Nesting is rejected because it would reset an overlapping ordinal plan.
    def with_workspace(workspace : Workspace, &)
      raise ArgumentError.new("resident context aborted") if @aborted
      raise ArgumentError.new("resident workspace belongs to a different context") unless workspace.owner.same?(@owner)
      raise ArgumentError.new("nested resident workspace") if @workspace
      workspace.enter
      @workspace = workspace
      begin
        yield
      rescue ex
        abort
        raise ex
      ensure
        @workspace = nil
        workspace.leave
      end
    end

    # Caller uses an existing terminal boundary, never an implicit new drain.
    # Releasing invalidates all workspace views, so escaped results are illegal.
    def release_workspace(workspace : Workspace) : Nil
      raise ArgumentError.new("resident context aborted") if @aborted
      raise ArgumentError.new("resident workspace belongs to a different context") unless workspace.owner.same?(@owner)
      raise ArgumentError.new("resident workspace release requires a terminal queue") if @workspace || @encoder || (@graph.try(&.pending_count) || 0) > 0
      workspace.release
      @workspaces.delete(workspace)
    end

    def allocation_stats
      {allocations: @allocations, requested_bytes: @requested_bytes,
       allocation_ns: @allocation_ns, allocation_profile: @allocation_profile,
       peak_metal_allocated_bytes: @memory_profile ? @peak_metal_allocated_bytes : nil,
       memory_profile: @memory_profile}
    end

    def graph_stats : Hash(String, Int64)
      graph = @graph
      {"depth" => graph.try(&.max_in_flight.to_i64) || 0_i64,
       "submitted" => @submissions, "completed" => graph.try(&.completed_count) || @submissions,
       "pending" => graph.try(&.pending_count.to_i64) || 0_i64,
       "max_pending" => graph.try(&.max_pending_seen.to_i64) || 0_i64,
       "completion_waits" => @completion_waits,
       "retained_bytes" => @retained_bytes,
       # Conservative backing-byte sum per live lease, not physical memory or
       # scratch size. Whole mapped dense shards and shared state may count
       # twice; packed weights bound directly by native methods are excluded.
       "max_retained_bytes" => @max_retained_bytes}
    end

    def encoder : HostEncoder
      raise ArgumentError.new("resident context aborted") if @aborted
      @encoder ||= host_profile.measure(HostEvent::EncoderAcquire) do
        command = if graph = @graph
                    lease = graph.begin_submission
                    @lease = lease
                    lease.command.command
                  else
                    host_profile.measure(HostEvent::CommandCreate) { ML::Metal::CommandBuffer.new }
                  end
        @command = command
        probe = @weight_wrapper_bank
        probe = nil unless probe.try(&.first_command_capture?)
        probe.try(&.first_command_started!)
        HostEncoder.new(command, host_profile, probe)
      end
    rescue ex
      abort
      raise ex
    end

    # Submit without a CPU completion barrier when CogniGraph is enabled.
    # Bindings own fresh storage, persistent state or explicitly managed DAC
    # scratch. GPU-only cross-command writes use one tracked Metal queue;
    # scratch reuse never introduces host writes into an in-flight resource.
    def submit : Nil
      raise ArgumentError.new("resident context aborted") if @aborted
      if enc = @encoder
        @encoder = nil
        enc.end_encoding
        command = @command.not_nil!
        if graph = @graph
          lease = @lease.not_nil!
          host_profile.measure(HostEvent::Retention) do
            host_profile.add(HostCounter::PendingEntries, @pending.size.to_i64)
            unique = {} of UInt64 => Buffer
            @pending.each { |buffer| unique[buffer.buffer.handle.address.to_u64] ||= buffer }
            host_profile.add(HostCounter::UniqueRetained, unique.size.to_i64)
            host_profile.add(HostCounter::RetainCallbacks, unique.size.to_i64)
            unique.each_value do |buffer|
              @retained_bytes += buffer.buffer.size
              lease.retain(buffer) { |resource| @retained_bytes -= resource.buffer.size; nil }
            end
            @max_retained_bytes = Math.max(@max_retained_bytes, @retained_bytes)
          end
          host_profile.measure(HostEvent::CommandSubmit) { graph.submit(lease) }
        else
          @completion_waits += 1
          host_profile.measure(HostEvent::CompletionWait) do
            Profile.measure("resident.submit_wait", category: "synchronization") do
              if @timeline
                gpu_seconds = command.commit_and_wait_gpu_elapsed_seconds
                STDERR.puts({event: "drama3_gpu_command", index: @submissions, gpu_seconds: gpu_seconds}.to_json)
              else
                command.commit_and_wait
              end
            end
          end
        end
        @command = nil
        @lease = nil
        @submissions += 1
        @weight_wrapper_bank.try(&.first_command_submitted!)
        @pending.clear
      end
    rescue ex
      abort
      raise ex
    end

    def finish : Nil
      submit
      @graph.try(&.drain)
    rescue ex
      abort
      raise ex
    end

    # A failed inference cannot be resumed: earlier GPU writes are not undone.
    # Preserve the original exception while making every known command terminal.
    def abort : Nil
      @weight_wrapper_bank.try(&.discard_first_command!)
      unless @aborted
        @@workspace_quarantine_mutex.synchronize do
          # A failed drain is not a lifetime certificate for packed weights.
          @weight_wrapper_bank.try { |bank| @@weight_bank_quarantine << bank }
          @workspaces.each do |workspace|
            workspace.quarantine
            @@workspace_quarantine << workspace
          end
        end
      end
      @aborted = true
      if enc = @encoder
        begin
          enc.end_encoding
        rescue
        end
      end
      @encoder = nil
      if graph = @graph
        graph.abort
      elsif command = @command
        begin
          command.committed? ? command.wait : command.discard
        rescue
        end
      end
      @command = nil
      @lease = nil
      @pending.clear
    end

    def alloc(count : Int32, *, reserve_count : Int32 = count) : Buffer
      raise ArgumentError.new("resident context aborted") if @aborted
      raise ArgumentError.new("resident allocation count/capacity") unless count > 0 && reserve_count >= count
      raise ArgumentError.new("resident allocation exceeds device buffer limit") if @max_buffer_length > 0 && reserve_count.to_i64 * 4 > @max_buffer_length
      if workspace = @workspace
        result = workspace.alloc(count, reserve_count) { |capacity| alloc_fresh(capacity) }
        # Direct encoder paths also need an owning reference in every flight.
        @pending << result
        result
      else
        alloc_fresh(count)
      end
    end

    private def alloc_fresh(count : Int32) : Buffer
      raise ArgumentError.new("resident context aborted") if @aborted
      raise ArgumentError.new("resident allocation count must be positive") unless count > 0
      storage = host_profile.measure(HostEvent::Backing) do
        if @allocation_profile
          started = Time.instant
          allocated = ML::MetalBuffer.new(count.to_i64 * 4)
          @allocation_ns += (Time.instant - started).total_nanoseconds.to_i64
          allocated
        else
          ML::MetalBuffer.new(count.to_i64 * 4)
        end
      end
      result = @owner.view(storage, count)
      @pending << result
      @allocations += 1
      @requested_bytes += count.to_i64 * 4
      if @memory_profile
        @peak_metal_allocated_bytes = Math.max(@peak_metal_allocated_bytes, ML::Metal::Device.instance.current_allocated_size)
      end
      result
    end

    def upload(values : Vector) : Buffer
      # Host writes and persistent weight conversions must never reuse a slot
      # that an earlier command may still read on the GPU.
      result = alloc_fresh(values.size)
      result.buffer.write(values)
      result
    end

    def zeros(count : Int32) : Buffer
      result = alloc(count)
      dispatch("dr3_fill", SOURCE, [result], [count.to_u64, 0_u64], count)
      result
    end

    def read(value : Buffer, purpose : String) : Vector
      check_owner!(value)
      raise ArgumentError.new("readback purpose must be terminal, control, or diagnostic") unless purpose.in?("terminal", "control", "diagnostic")
      finish
      if status = @status
        @readbacks["control"] += 4_i64
        raise ArgumentError.new("non-finite resident intermediate") unless status.buffer.contents.as(UInt32*).value == 0_u32
      end
      @readbacks[purpose] += value.count.to_i64 * 4
      result = Array(Float32).new(value.count, 0_f32)
      # Offset views cannot use MetalBuffer#read, which always begins at zero.
      (value.buffer.contents.as(Float32*) + value.offset // 4).copy_to(result.to_unsafe, value.count)
      Ops.finite!(result, "resident #{purpose}")
    end

    def guard(x : Buffer) : Nil
      status = @status ||= begin
        value = alloc_fresh(1)
        dispatch("dr3_fill", SOURCE, [value], [1_u64, 0_u64], 1)
        value
      end
      dispatch("dr3_finite", SOURCE, [x, status], [x.count.to_u64], x.count)
    end

    def dispatch(name : String, source : String, bindings : Array(Buffer), params : Array(UInt64), count : Int32, group : Int32 = 64) : Nil
      raise ArgumentError.new("invalid resident dispatch") unless count > 0 && group > 0
      host_profile.add(HostCounter::GenericDispatches)
      host_profile.add(HostCounter::BindingEntries, bindings.size.to_i64)
      host_profile.add(HostCounter::ParameterElements, params.size.to_i64)
      bindings.each { |binding| check_owner!(binding) }
      enc = encoder
      enc.set_pipeline(pipeline(name, source))
      bindings.each_with_index { |b, i| enc.set_buffer(b.buffer, i, offset: b.offset); @pending << b }
      enc.set_bytes(params.to_unsafe.as(Void*), (params.size * 8).to_i32, bindings.size) unless params.empty?
      enc.dispatch_1d(count, group)
    end

    def tensor(value : Tensor) : Buffer
      @weights[value.object_id] ||= begin
        source = weight_buffer(value)
        raw = @owner.view(source.buffer, (source.buffer.size // 4).clamp(1_i64, Int32::MAX.to_i64).to_i32)
        result = alloc_fresh(value.shape.reduce(1) { |a, b| a * b })
        dispatch("dr3_convert", SOURCE, [raw, result], [result.count.to_u64, source.offset.to_u64, encoding(value)], result.count)
        result
      end
    end

    def encoding(value : Tensor) : UInt64
      case value.dtype
      when "F32" then 0_u64
      when "BF16" then 1_u64
      when "F16" then 2_u64
      else raise ArgumentError.new("resident floating tensor dtype #{value.dtype}")
      end
    end

    def copy(source : Buffer, target : Buffer) : Nil
      raise ArgumentError.new("resident copy size") unless source.count == target.count
      dispatch("dr3_copy", SOURCE, [source, target], [source.count.to_u64], source.count)
    end

    private def validate_dense_reduction(value : String) : String
      raise ArgumentError.new("DRAMA3_DENSE_REDUCTION must be tree, simd, or simd32") unless value.in?("tree", "simd", "simd32")
      value
    end

    private def validate_expert_reduction(value : String) : String
      raise ArgumentError.new("expert reduction must be native or mlx_q4") unless value.in?("native", "mlx_q4")
      value
    end

    private def validate_mlx_q4!(matrix : Matrix) : Nil
      raise ArgumentError.new("MLX Q4 reduction requires affine scales and biases") unless matrix.scales && matrix.biases
      raise ArgumentError.new("MLX Q4 reduction requires 4-bit weights") unless matrix.bits == 4
      raise ArgumentError.new("MLX Q4 reduction requires group size 64") unless matrix.group_size == 64
      raise ArgumentError.new("MLX Q4 reduction requires K divisible by 512") unless matrix.cols % 512 == 0
      raise ArgumentError.new("MLX Q4 reduction requires N divisible by 8") unless matrix.rows % 8 == 0
    end

    private def dispatch_mlx_q4(matrix : Matrix, x : Buffer, output : Buffer, ids : Buffer?, rank : Int32,
                                batched : Bool, per_rank_input : Bool) : Nil
      weight = weight_buffer(matrix.tensor)
      scales_tensor, biases_tensor = matrix.scales.not_nil!, matrix.biases.not_nil!
      scales = weight_buffer(scales_tensor)
      biases = weight_buffer(biases_tensor)
      selection = ids || x
      experts = matrix.tensor.shape.size == 3 ? matrix.tensor.shape[0] : 1
      params = [matrix.rows.to_u64, matrix.cols.to_u64, experts.to_u64,
                weight.offset.to_u64, scales.offset.to_u64, biases.offset.to_u64,
                encoding(scales_tensor), encoding(biases_tensor), ids ? 1_u64 : 0_u64,
                rank.to_u64, matrix.expert.to_u64, batched ? 1_u64 : 0_u64,
                per_rank_input ? 1_u64 : 0_u64]

      enc = encoder
      enc.set_pipeline(pipeline("dr3_expert_mlx_q4", MLX_Q4_SOURCE))
      enc.set_buffer(weight.buffer, 0)
      enc.set_buffer(scales.buffer, 1)
      enc.set_buffer(biases.buffer, 2)
      enc.set_buffer(x.buffer, 3, offset: x.offset)
      enc.set_buffer(output.buffer, 4, offset: output.offset)
      enc.set_buffer(selection.buffer, 5, offset: selection.offset)
      enc.set_bytes(params.to_unsafe.as(Void*), (params.size * 8).to_i32, 6)
      ranks = batched ? ids.not_nil!.count : 1
      enc.dispatch_threadgroups({matrix.rows // 8, ranks, 1}, {64, 1, 1})
      @pending.concat([x, output, selection])
    end

    def add(a : Buffer, b : Buffer) : Buffer
      binary(a, b, 0)
    end

    def mul(a : Buffer, b : Buffer) : Buffer
      binary(a, b, 1)
    end

    private def binary(a : Buffer, b : Buffer, op : Int32) : Buffer
      raise ArgumentError.new("resident binary shape") unless a.count == b.count
      output = alloc(a.count)
      dispatch("dr3_binary", SOURCE, [a, b, output], [a.count.to_u64, op.to_u64], a.count)
      output
    end

    def silu(x : Buffer) : Buffer
      output = alloc(x.count)
      dispatch("dr3_silu", SOURCE, [x, output], [x.count.to_u64], x.count)
      output
    end

    def rms(x : Buffer, w : Buffer, eps : Float64, zero_centered : Bool = false, heads : Int32 = 1) : Buffer
      raise ArgumentError.new("resident RMS shape") unless heads > 0 && x.count % heads == 0 && w.count == x.count // heads && eps.finite? && eps > 0
      output = alloc(x.count)
      dispatch("dr3_rms", SOURCE, [x, w, output], [x.count.to_u64, w.count.to_u64, eps.to_f32.unsafe_as(UInt32).to_u64, zero_centered ? 1_u64 : 0_u64], heads)
      output
    end

    def rope!(x : Buffer, heads : Int32, dim : Int32, rotary : Int32, pos : Int32, base : Float64, interleaved : Bool) : Nil
      raise ArgumentError.new("resident RoPE shape") unless x.count == heads * dim && rotary > 0 && rotary.even? && rotary <= dim && pos >= 0
      dispatch("dr3_rope", SOURCE, [x], [heads.to_u64, dim.to_u64, rotary.to_u64, pos.to_u64, base.to_f32.unsafe_as(UInt32).to_u64, interleaved ? 1_u64 : 0_u64], heads * rotary // 2)
    end

    def matvec(matrix : Matrix, x : Buffer, dense_reduction : String? = nil) : Buffer
      check_owner!(x)
      raise ArgumentError.new("resident matvec width") unless matrix.cols == x.count
      return expert(matrix, x) if matrix.scales
      reduction = dense_reduction ? validate_dense_reduction(dense_reduction) : @dense_reduction
      w = weight_buffer(matrix.tensor)
      y = alloc(matrix.rows)
      threadgroup_width = 64
      selected_pipeline = if reduction == "simd32"
                            candidate = self.pipeline("dr3_dense_simd32", SOURCE)
                            if candidate.thread_execution_width == 32
                              threadgroup_width = 32
                              candidate
                            else
                              self.pipeline("dr3_dense", SOURCE)
                            end
                          else
                            name = reduction == "simd" ? "dr3_dense_simd" : "dr3_dense"
                            self.pipeline(name, SOURCE)
                          end
      enc = encoder
      enc.set_pipeline(selected_pipeline)
      enc.set_buffer(w.buffer, 0)
      enc.set_buffer(x.buffer, 1, offset: x.offset)
      enc.set_buffer(y.buffer, 2, offset: y.offset)
      p = [matrix.rows.to_u64, matrix.cols.to_u64, w.offset.to_u64, encoding(matrix.tensor), matrix.expert.to_u64]
      enc.set_bytes(p.to_unsafe.as(Void*), (p.size * 8).to_i32, 3)
      enc.dispatch_threadgroups({matrix.rows, 1, 1}, {threadgroup_width, 1, 1})
      @pending.concat([x, y])
      y
    end

    def expert(matrix : Matrix, x : Buffer, ids : Buffer? = nil, rank : Int32 = 0, reduction : String = "native") : Buffer
      check_owner!(x)
      check_owner!(ids) if ids
      raise ArgumentError.new("resident expert rank") unless rank >= 0 && (!ids || rank < ids.count)
      selected_reduction = validate_expert_reduction(reduction)
      validate_mlx_q4!(matrix) if selected_reduction == "mlx_q4"
      unless matrix.scales
        return matvec(matrix, x) unless ids
        raise ArgumentError.new("resident expert rank") unless rank >= 0 && rank < ids.count && x.count == matrix.cols
        w = weight_buffer(matrix.tensor)
        raw = @owner.view(w.buffer, (w.buffer.size // 4).clamp(1_i64, Int32::MAX.to_i64).to_i32)
        output = alloc(matrix.rows)
        experts = matrix.tensor.shape.size == 3 ? matrix.tensor.shape[0] : 1
        dispatch("dr3_dense_selected", SOURCE, [raw, x, ids, output], [matrix.rows.to_u64, matrix.cols.to_u64, w.offset.to_u64, encoding(matrix.tensor), rank.to_u64, experts.to_u64], matrix.rows)
        return output
      end
      raise ArgumentError.new("resident affine requires packed weights") unless matrix.scales && matrix.biases && x.count == matrix.cols
      if selected_reduction == "mlx_q4"
        output = alloc(matrix.rows)
        dispatch_mlx_q4(matrix, x, output, ids, rank, false, false)
        return output
      end
      raise ArgumentError.new("resident affine groups") if matrix.cols // matrix.group_size > 8192
      weight = weight_buffer(matrix.tensor)
      st, bt = matrix.scales.not_nil!, matrix.biases.not_nil!
      scales, biases = weight_buffer(st), weight_buffer(bt)
      output = alloc(matrix.rows)
      selection = ids || zeros(1)
      p = [matrix.rows.to_u64, matrix.cols.to_u64, matrix.bits.to_u64, matrix.group_size.to_u64, matrix.expert.to_u64,
           weight.offset.to_u64, scales.offset.to_u64, biases.offset.to_u64, encoding(st), encoding(bt),
           ids ? 1_u64 : 0_u64, rank.to_u64, (matrix.tensor.shape.size == 3 ? matrix.tensor.shape[0] : 1).to_u64]
      enc = encoder
      enc.set_pipeline(pipeline("dr3_expert", SOURCE))
      enc.set_buffer(weight.buffer, 0)
      enc.set_buffer(scales.buffer, 1)
      enc.set_buffer(biases.buffer, 2)
      enc.set_buffer(x.buffer, 3, offset: x.offset)
      enc.set_buffer(output.buffer, 4, offset: output.offset)
      enc.set_buffer(selection.buffer, 5, offset: selection.offset)
      enc.set_bytes(p.to_unsafe.as(Void*), (p.size * 8).to_i32, 6)
      enc.set_threadgroup_memory(matrix.cols // matrix.group_size * 4, 0)
      enc.dispatch_threadgroups({matrix.rows, 1, 1}, {native_expert_threads(matrix), 1, 1})
      @pending.concat([x, output, selection])
      output
    end

    # Sum a contiguous range of codebook embeddings. The fused route reads the
    # table in its checkpoint dtype and preserves the scalar FP32 addition
    # order of the legacy row/embedding/add dispatch chain.
    def codebook_sum(matrix : Matrix, frame : Buffer, first_codebook : Int32, end_codebook : Int32,
                     vocab_size : Int32, semantic_begin : Int32, semantic_end : Int32,
                     source : String, fused : Bool = false) : Buffer
      check_owner!(frame)
      raise ArgumentError.new("resident codebook vocabulary") unless vocab_size > 0 && matrix.rows % vocab_size == 0
      raise ArgumentError.new("resident codebook range") unless first_codebook >= 0 && first_codebook <= end_codebook &&
        end_codebook <= matrix.rows // vocab_size && end_codebook < frame.count
      raise ArgumentError.new("resident codebook semantic range") unless semantic_begin <= semantic_end

      if fused && matrix.scales.nil? && matrix.tensor.dtype.in?("F32", "BF16", "F16")
        return fused_codebook_sum(matrix, frame, first_codebook, end_codebook,
          vocab_size, semantic_begin, semantic_end, source)
      end

      total = zeros(matrix.cols)
      return total if first_codebook == end_codebook
      first_codebook.upto(end_codebook - 1) do |codebook|
        row = alloc(1)
        dispatch(
          "drama3_tower_codebook_row", source,
          [frame, row],
          [vocab_size.to_u64, codebook.to_u64, (codebook + 1).to_u64,
           semantic_begin.to_i64.to_u64, semantic_end.to_i64.to_u64],
          1,
        )
        embedding = embedding(matrix, row)
        total = add(total, embedding)
      end
      masked = alloc(matrix.cols)
      dispatch(
        "drama3_tower_mask_semantic", source,
        [total, frame.slice(0, 1), masked],
        [matrix.cols.to_u64, semantic_begin.to_i64.to_u64, semantic_end.to_i64.to_u64],
        matrix.cols,
      )
      masked
    end

    private def fused_codebook_sum(matrix : Matrix, frame : Buffer, first_codebook : Int32, end_codebook : Int32,
                                   vocab_size : Int32, semantic_begin : Int32, semantic_end : Int32,
                                   source : String) : Buffer
      weight = weight_buffer(matrix.tensor)
      # Retain the mapped backing handle with the command lease. `count` is
      # only a checked view size here; the kernel uses the tensor's byte offset.
      retained_weight = @owner.view(weight.buffer,
        (weight.buffer.size // 4).clamp(1_i64, Int32::MAX.to_i64).to_i32)
      output = alloc(matrix.cols)
      enc = encoder
      enc.set_pipeline(pipeline("drama3_tower_unseen_embedding", source))
      enc.set_buffer(weight.buffer, 0)
      enc.set_buffer(frame.buffer, 1, offset: frame.offset)
      enc.set_buffer(output.buffer, 2, offset: output.offset)
      params = [vocab_size.to_u64, matrix.cols.to_u64, first_codebook.to_u64, end_codebook.to_u64,
                semantic_begin.to_i64.to_u64, semantic_end.to_i64.to_u64,
                weight.offset.to_u64, encoding(matrix.tensor)]
      enc.set_bytes(params.to_unsafe.as(Void*), (params.size * 8).to_i32, 3)
      enc.dispatch_1d(matrix.cols)
      @pending.concat([retained_weight, frame, output])
      output
    end

    def embedding(matrix : Matrix, token : Buffer) : Buffer
      check_owner!(token)
      raise ArgumentError.new("resident embedding scalar") unless token.count == 1
      output = alloc(matrix.cols)
      w = weight_buffer(matrix.tensor)
      if matrix.scales
        st, bt = matrix.scales.not_nil!, matrix.biases.not_nil!
        s, b = weight_buffer(st), weight_buffer(bt)
        enc = encoder
        enc.set_pipeline(pipeline("dr3_embedding_quant", SOURCE))
        enc.set_buffer(w.buffer, 0); enc.set_buffer(s.buffer, 1); enc.set_buffer(b.buffer, 2)
        enc.set_buffer(token.buffer, 3, offset: token.offset); enc.set_buffer(output.buffer, 4)
        p = [matrix.rows.to_u64, matrix.cols.to_u64, matrix.bits.to_u64, matrix.group_size.to_u64, w.offset.to_u64, s.offset.to_u64, b.offset.to_u64, encoding(st), encoding(bt)]
        enc.set_bytes(p.to_unsafe.as(Void*), (p.size * 8).to_i32, 5)
        enc.dispatch_1d(matrix.cols)
        @pending.concat([token, output])
      else
        raw = @owner.view(w.buffer, (w.buffer.size // 4).clamp(1_i64, Int32::MAX.to_i64).to_i32)
        dispatch("dr3_embedding", SOURCE, [raw, token, output], [matrix.rows.to_u64, matrix.cols.to_u64, w.offset.to_u64, encoding(matrix.tensor)], matrix.cols)
      end
      output
    end

    # Independent ranks share one dispatch, retaining the scalar affine-group
    # arithmetic of #expert. Down projections read a different input per rank.
    def experts(matrix : Matrix, x : Buffer, ids : Buffer, per_rank_input : Bool = false, reduction : String = "native") : Buffer
      check_owner!(x)
      check_owner!(ids)
      selected_reduction = validate_expert_reduction(reduction)
      raise ArgumentError.new("resident batched experts require affine weights") unless matrix.scales && matrix.biases
      expected = matrix.cols.to_i64 * (per_rank_input ? ids.count : 1)
      raise ArgumentError.new("resident batched expert input width") unless x.count == expected
      validate_mlx_q4!(matrix) if selected_reduction == "mlx_q4"
      raise ArgumentError.new("resident affine groups") if matrix.cols // matrix.group_size > 8192
      count = matrix.rows.to_i64 * ids.count
      raise ArgumentError.new("resident batched expert output too large") if count > Int32::MAX
      if selected_reduction == "mlx_q4"
        output = alloc(count.to_i32)
        dispatch_mlx_q4(matrix, x, output, ids, 0, true, per_rank_input)
        return output
      end
      packed_groups = matrix.cols // matrix.group_size
      packed_rows = @expert_packed_rows_enabled && matrix.group_size == 64 && matrix.bits.in?(4, 8) &&
                    native_expert_threads(matrix) == 64 && packed_groups.in?(1..32) && 64 % packed_groups == 0
      weight = weight_buffer(matrix.tensor)
      st, bt = matrix.scales.not_nil!, matrix.biases.not_nil!
      scales, biases = weight_buffer(st), weight_buffer(bt)
      output = alloc(count.to_i32)
      p = [matrix.rows.to_u64, matrix.cols.to_u64, matrix.bits.to_u64, matrix.group_size.to_u64,
           weight.offset.to_u64, scales.offset.to_u64, biases.offset.to_u64, encoding(st), encoding(bt),
           (matrix.tensor.shape.size == 3 ? matrix.tensor.shape[0] : 1).to_u64, per_rank_input ? 1_u64 : 0_u64]
      enc = encoder
      enc.set_pipeline(pipeline(packed_rows ? "dr3_experts_packed_rows" : "dr3_experts", SOURCE))
      enc.set_buffer(weight.buffer, 0)
      enc.set_buffer(scales.buffer, 1)
      enc.set_buffer(biases.buffer, 2)
      [x, output, ids].each_with_index { |v, i| enc.set_buffer(v.buffer, i + 3, offset: v.offset) }
      enc.set_bytes(p.to_unsafe.as(Void*), (p.size * 8).to_i32, 6)
      if packed_rows
        rows_per_threadgroup = 64 // packed_groups
        enc.set_threadgroup_memory(64 * 4, 0)
        enc.dispatch_threadgroups({(matrix.rows + rows_per_threadgroup - 1) // rows_per_threadgroup, ids.count, 1}, {64, 1, 1})
      else
        enc.set_threadgroup_memory(matrix.cols // matrix.group_size * 4, 0)
        enc.dispatch_threadgroups({matrix.rows, ids.count, 1}, {native_expert_threads(matrix), 1, 1})
      end
      @pending.concat([x, output, ids])
      output
    end

    private def native_expert_threads(matrix : Matrix) : Int32
      # Both affine shaders partition groups with g += 64. Fewer lanes are
      # complete only when every group fits in the first 32; group 33 must
      # retain the original launch and scalar group-sum order.
      @expert_threads == 32 && matrix.cols // matrix.group_size <= 32 ? 32 : 64
    end

    def argmax(x : Buffer) : Buffer
      output = alloc(1)
      dispatch("dr3_argmax", SOURCE, [x, output], [x.count.to_u64], 1)
      output
    end

    def sample(x : Buffer, draw : Buffer, temperature : Float64, top_k : Int32, top_p : Float64) : Buffer
      raise ArgumentError.new("invalid sampling parameters") unless temperature.finite? && temperature >= 0 && top_k.in?(1..1024) && top_p.finite? && top_p > 0 && top_p <= 1 && draw.count == 1
      guard(x)
      output = alloc(1)
      dispatch("dr3_sample", SOURCE, [x, draw, output], [x.count.to_u64, Math.min(top_k, x.count).to_u64, Math.max(temperature, 1e-5).to_f32.unsafe_as(UInt32).to_u64, top_p.to_f32.unsafe_as(UInt32).to_u64], 1)
      output
    end

    private def check_owner!(value : Buffer) : Nil
      raise ArgumentError.new("buffer belongs to a different resident context") unless value.owner.same?(@owner)
    end
  end
end
{% end %}
