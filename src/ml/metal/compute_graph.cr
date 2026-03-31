{% if flag?(:cpu_only) %}
module ML
  module Metal
    class ComputeGraph
      def add_op(pipeline : ComputePipeline, &block : OpBuilder ->) : Int32
        raise "Metal disabled (cpu_only)"
      end

      def compile! : Nil
        raise "Metal disabled (cpu_only)"
      end

      def encode(cmd : CommandBuffer) : Nil
        raise "Metal disabled (cpu_only)"
      end

      def clear : Nil
      end

      struct OpBuilder
        def buffer(buf : MetalBuffer, index : Int32, access : BufferAccess = :read, offset : Int64 = 0) : OpBuilder
          raise "Metal disabled (cpu_only)"
        end

        def value(val, index : Int32) : OpBuilder
          raise "Metal disabled (cpu_only)"
        end

        def dispatch_threadgroups(grid : {Int32, Int32, Int32}, size : {Int32, Int32, Int32}) : Nil
          raise "Metal disabled (cpu_only)"
        end

        def dispatch_1d(count : Int32, tg_size : Int32 = 256) : Nil
          raise "Metal disabled (cpu_only)"
        end

        def dispatch_indirect(buf : MetalBuffer, offset : Int64, size : {Int32, Int32, Int32}) : Nil
          raise "Metal disabled (cpu_only)"
        end

        def threadgroup_memory(bytes : Int32) : OpBuilder
          raise "Metal disabled (cpu_only)"
        end
      end
    end

    enum BufferAccess
      Read
      Write
      ReadWrite
    end
  end
end
{% else %}
require "./device"
require "./dispatch"

# Metal Compute Graph — automatic barrier optimization via dependency analysis
#
# Instead of manually inserting memory barriers between every dispatch,
# the graph tracks buffer read/write access per operation and compiles
# them into "waves" of independent operations. Only one barrier per wave
# boundary, and all ops within a wave execute concurrently.
#
# Usage:
#   graph = ComputeGraph.new
#   graph.add_op(pipeline) do |op|
#     op.buffer(input, 0, :read)
#     op.buffer(output, 1, :write)
#     op.value(dim, 2)
#     op.dispatch_threadgroups({grid_x, grid_y, 1}, {32, 2, 1})
#   end
#   graph.compile!
#   graph.encode(cmd)  # encodes all ops with minimal barriers
#
module ML
  module Metal
    # BufferAccess enum defined in dispatch.cr

    # Shared types for ComputeGraph and GraphEncoder
    record BufBinding,
      buffer : MetalBuffer,
      index : Int32,
      offset : Int64 = 0_i64,
      length : Int64 = -1_i64,   # -1 = entire buffer (unknown range)
      access : BufferAccess = BufferAccess::Read,
      partition : Int32 = -1 do   # -1 = full buffer. ≥0 = partition key (Block Integrity)
      def buffer_id : UInt64
        @buffer.handle.address
      end

      # Two bindings conflict if same buffer AND overlapping access
      # Block Integrity (LTP/WBA §3.3): different partition keys = no overlap guaranteed
      def conflicts?(other : BufBinding) : Bool
        return false if buffer_id != other.buffer_id
        # Different partitions on same buffer = structurally non-overlapping (MoE Block Integrity)
        if partition >= 0 && other.partition >= 0 && partition != other.partition
          return false
        end
        # Unknown range conflicts with everything
        return true if length < 0 || other.length < 0
        # Check offset range overlap
        a_end = offset + length
        b_end = other.offset + other.length
        offset < b_end && other.offset < a_end
      end
    end

    record ValBinding, data : Bytes, index : Int32

    enum DispatchMode
      Threadgroups
      Threads
      Indirect
    end

    record DispatchInfo,
      mode : DispatchMode,
      grid : Tuple(Int32, Int32, Int32),
      tg_size : Tuple(Int32, Int32, Int32),
      indirect_buf : MetalBuffer? = nil,
      indirect_off : Int64 = 0_i64

    class ComputeGraph
      # Internal operation record
      class Op
        property pipeline : ComputePipeline
        property buffers : Array(BufBinding)
        property values : Array(ValBinding)
        property dispatch : DispatchInfo?
        property tg_memory : Int32

        def initialize(@pipeline)
          @buffers = [] of BufBinding
          @values = [] of ValBinding
          @dispatch = nil
          @tg_memory = 0
        end
      end

      # Builder passed to the add_op block for fluent op construction
      struct OpBuilder
        @op : Op

        protected def initialize(@op)
        end

        def buffer(buf : MetalBuffer, index : Int32, access : BufferAccess = BufferAccess::Read, offset : Int64 = 0) : OpBuilder
          @op.buffers << BufBinding.new(buf, index, offset, access)
          self
        end

        def value(val : UInt32, index : Int32) : OpBuilder
          bytes = Bytes.new(4)
          IO::ByteFormat::LittleEndian.encode(val, bytes)
          @op.values << ValBinding.new(bytes, index)
          self
        end

        def value(val : Float32, index : Int32) : OpBuilder
          bytes = Bytes.new(4)
          IO::ByteFormat::LittleEndian.encode(val, bytes)
          @op.values << ValBinding.new(bytes, index)
          self
        end

        def threadgroup_memory(bytes : Int32) : OpBuilder
          @op.tg_memory = bytes
          self
        end

        def dispatch_threadgroups(grid : {Int32, Int32, Int32}, size : {Int32, Int32, Int32}) : Nil
          @op.dispatch = DispatchInfo.new(DispatchMode::Threadgroups, grid, size)
        end

        def dispatch_1d(count : Int32, tg_size : Int32 = 256) : Nil
          @op.dispatch = DispatchInfo.new(DispatchMode::Threads, {count, 1, 1}, {tg_size, 1, 1})
        end

        def dispatch_indirect(buf : MetalBuffer, offset : Int64, size : {Int32, Int32, Int32}) : Nil
          @op.dispatch = DispatchInfo.new(DispatchMode::Indirect, {0, 0, 0}, size, buf, offset)
        end
      end

      @ops : Array(Op)
      @waves : Array(Array(Int32))?
      @stats : Stats?

      struct Stats
        property n_ops : Int32 = 0
        property n_waves : Int32 = 0
        property n_barriers : Int32 = 0
        property max_wave_width : Int32 = 0
      end

      def initialize
        @ops = [] of Op
      end

      # Number of operations in the graph
      def size : Int32
        @ops.size
      end

      # Compilation stats (available after compile!)
      def stats : Stats
        @stats || Stats.new
      end

      # Add a pre-built Op directly (used by GraphEncoder)
      def add_op_direct(op : Op) : Int32
        idx = @ops.size
        @ops << op
        @waves = nil
        idx
      end

      # Add an operation to the graph. Returns its index.
      def add_op(pipeline : ComputePipeline, &block : OpBuilder ->) : Int32
        op = Op.new(pipeline)
        builder = OpBuilder.new(op)
        yield builder
        idx = @ops.size
        @ops << op
        @waves = nil  # invalidate
        idx
      end

      # Clear all operations (reuse the graph object)
      def clear : Nil
        @ops.clear
        @waves = nil
        @stats = nil
      end

      private record AccessRecord, op_index : Int32, binding : BufBinding

      # Compile: analyze dependencies, group into waves
      def compile! : Nil
        n = @ops.size
        return if n == 0

        # Offset-aware dependency analysis: two ops on the same buffer only conflict
        # if their accessed offset ranges overlap. This allows concurrent writes to
        # different regions of the same buffer (e.g., 8 MoE expert outputs).

        # Track all active writers/readers per buffer_id
        writers = {} of UInt64 => Array(AccessRecord)
        readers = {} of UInt64 => Array(AccessRecord)

        wave_of = Array(Int32).new(n, 0)

        n.times do |i|
          op = @ops[i]
          max_dep_wave = -1  # Track max WAVE of dependencies (not max op index!)

          # Collect explicit + implicit (indirect dispatch) buffer accesses
          all_accesses = op.buffers.dup
          if (d = op.dispatch) && d.mode.indirect? && (ibuf = d.indirect_buf)
            all_accesses << BufBinding.new(ibuf, -1, d.indirect_off, access: BufferAccess::Read)
          end

          all_accesses.each do |binding|
            bid = binding.buffer_id

            case binding.access
            when .read?
              if ws = writers[bid]?
                ws.each do |wr|
                  if binding.conflicts?(wr.binding)
                    w = wave_of[wr.op_index]
                    max_dep_wave = w if w > max_dep_wave
                  end
                end
              end
              (readers[bid] ||= [] of AccessRecord) << AccessRecord.new(i, binding)

            when .write?
              if rs = readers[bid]?
                rs.each do |rd|
                  if binding.conflicts?(rd.binding)
                    w = wave_of[rd.op_index]
                    max_dep_wave = w if w > max_dep_wave
                  end
                end
              end
              if ws = writers[bid]?
                ws.each do |wr|
                  if binding.conflicts?(wr.binding)
                    w = wave_of[wr.op_index]
                    max_dep_wave = w if w > max_dep_wave
                  end
                end
              end
              (writers[bid] ||= [] of AccessRecord) << AccessRecord.new(i, binding)

            when .read_write?
              if ws = writers[bid]?
                ws.each do |wr|
                  if binding.conflicts?(wr.binding)
                    w = wave_of[wr.op_index]; max_dep_wave = w if w > max_dep_wave
                  end
                end
              end
              if rs = readers[bid]?
                rs.each do |rd|
                  if binding.conflicts?(rd.binding)
                    w = wave_of[rd.op_index]; max_dep_wave = w if w > max_dep_wave
                  end
                end
              end
              rec = AccessRecord.new(i, binding)
              (writers[bid] ||= [] of AccessRecord) << rec
              (readers[bid] ||= [] of AccessRecord) << rec
            end
          end

          wave_of[i] = max_dep_wave + 1
        end

        # Group ops by wave
        max_wave = wave_of.max? || 0
        @waves = (0..max_wave).map do |w|
          (0...n).select { |i| wave_of[i] == w }.to_a
        end

        # Compute stats
        waves = @waves.not_nil!
        st = Stats.new
        st.n_ops = n
        st.n_waves = waves.size
        st.n_barriers = {waves.size - 1, 0}.max
        st.max_wave_width = waves.max_of(&.size)
        @stats = st

        if @@debug_encode
          widest = waves.max_by(&.size)
          STDERR.puts "  widest wave (#{widest.size} ops):"
          widest.each do |oi|
            op = @ops[oi]
            bufs = op.buffers.map { |b| "#{b.index}:#{b.access}#{b.partition >= 0 ? "/p#{b.partition}" : ""}" }.join(" ")
            STDERR.puts "    op[#{oi}] #{op.pipeline.name} bufs=[#{bufs}]"
          end
        end
      end

      # Encode sequentially (debug: barrier between every op, no wave optimization)
      def encode_sequential(cmd : CommandBuffer) : Nil
        enc = ComputeEncoder.new(cmd, concurrent: true)
        @ops.each_with_index do |op, i|
          encode_op(enc, op)
          enc.memory_barrier if i < @ops.size - 1
        end
        enc.end_encoding
      end

      # Encode the compiled graph into a command buffer
      def encode(cmd : CommandBuffer) : Nil
        waves = @waves
        unless waves
          compile!
          waves = @waves.not_nil!
        end

        enc = ComputeEncoder.new(cmd, concurrent: true)

        waves.each_with_index do |wave, wi|
          wave.each do |op_idx|
            op = @ops[op_idx]
            encode_op(enc, op)
          end

          # Barrier between waves (not after the last one)
          enc.memory_barrier if wi < waves.size - 1
        end

        enc.end_encoding
      end

      @@debug_encode = false
      class_property debug_encode : Bool

      private def encode_op(enc : ComputeEncoder, op : Op) : Nil
        enc.set_pipeline(op.pipeline)

        op.buffers.each do |b|
          enc.set_buffer(b.buffer, b.index, offset: b.offset)
        end

        op.values.each do |v|
          if @@debug_encode
            val_u32 = v.data.to_unsafe.as(Pointer(UInt32)).value
            STDERR.puts "    val[#{v.index}] = 0x#{val_u32.to_s(16)} (#{val_u32})"
          end
          enc.set_bytes(v.data.to_unsafe.as(Pointer(Void)), v.data.size, v.index)
        end

        enc.set_threadgroup_memory(op.tg_memory, 0) if op.tg_memory > 0

        if d = op.dispatch
          case d.mode
          when .threadgroups?
            enc.dispatch_threadgroups(d.grid, d.tg_size)
          when .threads?
            enc.dispatch(d.grid, d.tg_size)
          when .indirect?
            enc.dispatch_threadgroups_indirect(d.indirect_buf.not_nil!, d.indirect_off, d.tg_size)
          end
        end
      end
    end

    # GraphEncoder — drop-in replacement for ComputeEncoder that builds a ComputeGraph.
    # Same API (set_pipeline, set_buffer, set_value, dispatch_*, memory_barrier).
    # memory_barrier is a no-op — the graph computes optimal barriers automatically.
    #
    # Usage:
    #   graph = ComputeGraph.new
    #   enc = GraphEncoder.new(graph)
    #   # ... same calls as ComputeEncoder ...
    #   graph.compile!
    #   graph.encode(cmd)
    #
    struct GraphEncoder
      @graph : ComputeGraph
      @pipeline : ComputePipeline?
      @buffers : Array(BufBinding)
      @values : Array(ValBinding)
      @tg_mem : Int32

      def initialize(@graph)
        @pipeline = nil
        @buffers = [] of BufBinding
        @values = [] of ValBinding
        @tg_mem = 0
      end

      def set_pipeline(p : ComputePipeline) : self
        @pipeline = p
        self
      end

      # Buffer binding with access mode (default: :read)
      # length: -1 = entire buffer. partition: ≥0 = Block Integrity key (different partitions don't conflict)
      def set_buffer(buf : MetalBuffer, index : Int32, access : BufferAccess = BufferAccess::Read,
                     offset : Int64 = 0, length : Int64 = -1, partition : Int32 = -1) : self
        @buffers << BufBinding.new(buf, index, offset, length, access, partition)
        self
      end

      def set_value(val : UInt32, index : Int32) : self
        bytes = Bytes.new(4)
        IO::ByteFormat::LittleEndian.encode(val, bytes)
        @values << ValBinding.new(bytes, index)
        self
      end

      def set_value(val : Float32, index : Int32) : self
        bytes = Bytes.new(4)
        IO::ByteFormat::LittleEndian.encode(val, bytes)
        @values << ValBinding.new(bytes, index)
        self
      end

      def set_threadgroup_memory(length : Int32, index : Int32) : self
        @tg_mem = length
        self
      end

      # memory_barrier is a NO-OP — the graph handles barriers automatically
      def memory_barrier : self
        self
      end

      def dispatch_threadgroups(grid : {Int32, Int32, Int32}, size : {Int32, Int32, Int32}) : self
        flush!(DispatchInfo.new(DispatchMode::Threadgroups, grid, size))
        self
      end

      def dispatch(grid : {Int32, Int32, Int32}, size : {Int32, Int32, Int32}) : self
        flush!(DispatchInfo.new(DispatchMode::Threads, grid, size))
        self
      end

      def dispatch_1d(count : Int32, tg_size : Int32 = 256) : self
        flush!(DispatchInfo.new(DispatchMode::Threads, {count, 1, 1}, {tg_size, 1, 1}))
        self
      end

      def dispatch_threadgroups_indirect(buf : MetalBuffer, offset : Int64, size : {Int32, Int32, Int32}) : self
        flush!(DispatchInfo.new(DispatchMode::Indirect, {0, 0, 0}, size, buf, offset))
        self
      end

      def end_encoding : Nil
        # No-op for graph encoder — encoding happens via graph.encode(cmd)
      end

      private def flush!(dispatch : DispatchInfo) : Nil
        p = @pipeline.not_nil!

        # Directly construct the Op without going through OpBuilder
        op = ComputeGraph::Op.new(p)
        op.buffers = @buffers.dup
        op.values = @values.dup
        op.tg_memory = @tg_mem
        op.dispatch = dispatch

        @graph.add_op_direct(op)

        @buffers.clear
        @values.clear
        @tg_mem = 0
      end
    end
  end
end
{% end %}
