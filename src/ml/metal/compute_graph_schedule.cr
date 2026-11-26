# Metal-independent dependency compiler shared by ComputeGraph and native
# operation planners. Keep this file free of device/dispatch initialization.
module ML
  module Metal
    module CogniGraphSchedule
      MAX_OPS      =  256_u32
      MAX_ACCESSES = 2048_u32

      enum AccessMode : Int32
        Read      = 0
        Write     = 1
        ReadWrite = 2
      end

      record Access,
        buffer_id : UInt64,
        offset : Int64,
        length : Int64,
        access : AccessMode,
        partition : Int32 do
        def conflicts?(other : Access) : Bool
          return false if buffer_id != other.buffer_id

          # Different partitions on the same buffer are structurally disjoint.
          if partition >= 0 && other.partition >= 0 && partition != other.partition
            return false
          end

          # An unknown range conflicts with every range on the buffer.
          return true if length < 0 || other.length < 0

          a_end = offset + length
          b_end = other.offset + other.length
          offset < b_end && other.offset < a_end
        end
      end

      record Schedule, wave_of : Array(Int32), wave_count : UInt32

      # C-compatible layout matching cognigraph_schedule.h.
      @[Extern]
      struct NativeAccess
        @buffer_id : UInt64
        @offset : Int64
        @length : Int64
        @access : Int32
        @partition : Int32

        def initialize(@buffer_id : UInt64, @offset : Int64, @length : Int64,
                       @access : Int32, @partition : Int32)
        end

        def buffer_id : UInt64
          @buffer_id
        end

        def offset : Int64
          @offset
        end

        def length : Int64
          @length
        end

        def access : Int32
          @access
        end

        def partition : Int32
          @partition
        end
      end

      alias Callback = Proc(Pointer(NativeAccess), Pointer(UInt32), UInt32, UInt32,
                            Pointer(Int32), Pointer(UInt32), Int32)

      private record PriorAccess, op_index : Int32, access : Access

      # Compile flattened operation accesses. offsets must contain op_count + 1
      # entries; each pair defines the half-open access slice for one operation.
      def self.compile(op_offsets : Array(UInt32), accesses : Array(Access)) : Schedule
        unless op_offsets.size > 0 && op_offsets.first == 0_u32 && op_offsets.last == accesses.size
          raise ArgumentError.new("invalid CogniGraph operation access offsets")
        end

        previous_offset = 0_u32
        op_offsets.each do |offset|
          unless offset >= previous_offset && offset <= accesses.size
            raise ArgumentError.new("non-monotonic CogniGraph operation access offsets")
          end
          previous_offset = offset
        end

        n = op_offsets.size - 1
        return Schedule.new([] of Int32, 0_u32) if n == 0

        writers = {} of UInt64 => Array(PriorAccess)
        readers = {} of UInt64 => Array(PriorAccess)
        wave_of = Array(Int32).new(n, 0)

        n.times do |op_index|
          max_dep_wave = -1
          start = op_offsets[op_index].to_i
          finish = op_offsets[op_index + 1].to_i

          (start...finish).each do |access_index|
            binding = accesses[access_index]
            buffer_id = binding.buffer_id

            case binding.access
            when .read?
              if prior_writers = writers[buffer_id]?
                prior_writers.each do |prior|
                  if binding.conflicts?(prior.access)
                    prior_wave = wave_of[prior.op_index]
                    max_dep_wave = prior_wave if prior_wave > max_dep_wave
                  end
                end
              end
              (readers[buffer_id] ||= [] of PriorAccess) << PriorAccess.new(op_index, binding)
            when .write?
              if prior_readers = readers[buffer_id]?
                prior_readers.each do |prior|
                  if binding.conflicts?(prior.access)
                    prior_wave = wave_of[prior.op_index]
                    max_dep_wave = prior_wave if prior_wave > max_dep_wave
                  end
                end
              end
              if prior_writers = writers[buffer_id]?
                prior_writers.each do |prior|
                  if binding.conflicts?(prior.access)
                    prior_wave = wave_of[prior.op_index]
                    max_dep_wave = prior_wave if prior_wave > max_dep_wave
                  end
                end
              end
              (writers[buffer_id] ||= [] of PriorAccess) << PriorAccess.new(op_index, binding)
            when .read_write?
              if prior_writers = writers[buffer_id]?
                prior_writers.each do |prior|
                  if binding.conflicts?(prior.access)
                    prior_wave = wave_of[prior.op_index]
                    max_dep_wave = prior_wave if prior_wave > max_dep_wave
                  end
                end
              end
              if prior_readers = readers[buffer_id]?
                prior_readers.each do |prior|
                  if binding.conflicts?(prior.access)
                    prior_wave = wave_of[prior.op_index]
                    max_dep_wave = prior_wave if prior_wave > max_dep_wave
                  end
                end
              end
              record = PriorAccess.new(op_index, binding)
              (writers[buffer_id] ||= [] of PriorAccess) << record
              (readers[buffer_id] ||= [] of PriorAccess) << record
            end
          end

          wave_of[op_index] = max_dep_wave + 1
        end

        max_wave = wave_of.max? || 0
        Schedule.new(wave_of, (max_wave + 1).to_u32)
      end

      # Stable, non-capturing callback pointer to pass to native preparation.
      # The callback runs once while preparing a graph, never per request.
      @@callback : Callback = ->(accesses : Pointer(NativeAccess), op_offsets : Pointer(UInt32), op_count : UInt32, access_count : UInt32, wave_of : Pointer(Int32), wave_count : Pointer(UInt32)) {
        compile_callback(accesses, op_offsets, op_count, access_count, wave_of, wave_count)
      }

      def self.callback : Callback
        @@callback
      end

      # Bounded C ABI adapter. All malformed inputs and Crystal exceptions are
      # translated to a negative result so nothing unwinds across the ABI.
      def self.compile_callback(accesses : Pointer(NativeAccess),
                                op_offsets : Pointer(UInt32),
                                op_count : UInt32,
                                access_count : UInt32,
                                wave_of : Pointer(Int32),
                                wave_count : Pointer(UInt32)) : Int32
        return -1 if wave_count.null?
        wave_count.value = 0_u32

        begin
          return -2 if op_count > MAX_OPS || access_count > MAX_ACCESSES
          return -3 if op_offsets.null?
          return -3 if access_count > 0 && accesses.null?
          return -3 if op_count > 0 && wave_of.null?

          offsets = Array(UInt32).new(op_count.to_i + 1)
          op_count.to_i.times do |index|
            offsets << (op_offsets + index).value
          end
          offsets << (op_offsets + op_count.to_i).value

          unless offsets.first == 0_u32 && offsets.last == access_count
            return -4
          end
          offsets.each_cons_pair do |left, right|
            return -4 if right < left || right > access_count
          end

          native_accesses = Array(Access).new(access_count.to_i)
          access_count.to_i.times do |index|
            native = (accesses + index).value
            mode = case native.access
                   when 0 then AccessMode::Read
                   when 1 then AccessMode::Write
                   when 2 then AccessMode::ReadWrite
                   else
                     return -5
                   end

            # Native buffers use nonnegative byte offsets. -1 is the only
            # accepted unknown-length sentinel.
            if native.offset < 0 || native.length < -1
              return -5
            end
            if native.length >= 0 && native.offset > Int64::MAX - native.length
              return -5
            end

            native_accesses << Access.new(native.buffer_id, native.offset,
              native.length, mode, native.partition)
          end

          schedule = compile(offsets, native_accesses)
          schedule.wave_of.each_with_index do |wave, index|
            (wave_of + index).value = wave
          end
          wave_count.value = schedule.wave_count
          0
        rescue
          -6
        end
      end
    end
  end
end
