require "json"
require "../../src/ml/video/hy_nomas_block"
require "../../src/ml/video/metal_mps_command_chain"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  @[Link(ldflags: "-framework Metal -framework Foundation -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph")]
  lib HyNomasMPSReductionFFI
    fun plan_create = gs_hy_nomas_mps_reduction_plan_create(
      device : Pointer(Void), mode : Int32, batches : Int32, query_rows : Int32,
      key_rows : Int32, heads : Int32, head_dim : Int32, scale : Float32,
      input_a : Pointer(Void), input_b : Pointer(Void), input_c : Pointer(Void),
      packed_a : Pointer(Void), packed_b : Pointer(Void), packed_c : Pointer(Void),
      packed_output : Pointer(Void), output : Pointer(Void),
    ) : Pointer(Void)
    fun encode = gs_hy_nomas_mps_reduction_encode(plan : Pointer(Void), command : Pointer(Void)) : Int32
    fun plan_release = gs_hy_nomas_mps_reduction_plan_release(plan : Pointer(Void)) : Void
  end
{% end %}

module HyAttentionProbe
  # Prepared, fixed-buffer MPSGraph controls for isolated attention reductions.
  # Inputs and outputs remain BQHD/BQHK in caller-owned shared F32 buffers.
  # GPU transpose kernels populate setup-allocated contiguous BHQD/BHQK
  # buffers before MPSGraph runs, matching PyTorch's contiguous operands.
  # Keep this object and its Arena alive through the chain's terminal wait.
  class MPSReductionControls
    private struct Entry
      getter index : Int32
      getter name : String
      getter kind : String
      getter scale : Float32
      getter input_names : Array(String)
      getter inputs : Array(ML::MetalBuffer)
      getter packed_inputs : Array(ML::MetalBuffer)
      getter packed_output : ML::MetalBuffer
      getter output : ML::MetalBuffer
      getter handle : Pointer(Void)

      def initialize(@index, @name, @kind, @scale, @input_names, @inputs,
                     @packed_inputs, @packed_output, @output, @handle)
      end
    end

    @config : ML::Video::HyNomasConfig
    @arena : ML::Video::Arena
    @entries = {} of Int32 => Entry
    @released = false

    def initialize(@config : ML::Video::HyNomasConfig, ops : Array(JSON::Any),
                   @arena : ML::Video::Arena)
      @config.validate!
      raise ArgumentError.new("MPS reduction controls require operations") if ops.empty?
      seen = Set(String).new
      begin
        ops.each_with_index do |op, raw_index|
          kind = op["kind"].as_s
          next unless {"qk", "softmax", "pv", "attention"}.includes?(kind)
          name = op["name"].as_s
          raise ArgumentError.new("Duplicate MPS reduction operation") unless seen.add?(name)
          entry = prepare_entry!(op, raw_index.to_i32)
          @entries[entry.index] = entry
        end
      rescue ex
        release
        raise ex
      end
    end

    def plan_count : Int32
      @entries.size.to_i32
    end

    def has_plan?(index : Int32) : Bool
      !@released && @entries.has_key?(index)
    end

    # Append one already-prepared graph to the caller's serial chain. The
    # chain wrapper is borrowed only while paused; resume! obtains its current
    # root after MPSGraph has had a chance to continue the command buffer.
    # This path does not allocate MetalBuffers, wait, or read back data.
    def encode(chain : ML::Metal::MPSCommandChain, op : JSON::Any,
               index : Int32) : Nil
      raise "MPS reduction controls have been released" if @released
      entry = @entries[index]?
      raise ArgumentError.new("No MPS reduction plan for operation index #{index}") unless entry
      validate_operation_identity!(op, entry)
      validate_bindings!(entry)
      chain.encoder
      chain.pause!
      status = -1
      begin
        {% if flag?(:darwin) && !flag?(:cpu_only) %}
          status = HyNomasMPSReductionFFI.encode(entry.handle, chain.handle)
        {% else %}
          raise ArgumentError.new("MPS reduction controls require macOS Metal Performance Shaders")
        {% end %}
      ensure
        chain.resume!
      end
      raise "MPS reduction graph encode rejected (#{status})" unless status == 0
    end

    # Call after the caller's terminal wait. Plans own Graph descriptors and
    # retain fixed bindings, but never own or release Arena buffers.
    def release : Nil
      return if @released
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        @entries.each_value do |entry|
          HyNomasMPSReductionFFI.plan_release(entry.handle) unless entry.handle.null?
        end
      {% end %}
      @entries.clear
      @released = true
    end

    private def prepare_entry!(op : JSON::Any, index : Int32) : Entry
      kind = op["kind"].as_s
      name = op["name"].as_s
      input_names = case kind
                    when "qk"        then ["q", "k"]
                    when "softmax"   then ["scores"]
                    when "pv"        then ["probabilities", "v"]
                    when "attention" then ["q", "k", "v"]
                    else                  raise ArgumentError.new("Unsupported MPS reduction kind: #{kind}")
                    end
      inputs_record = op["inputs"].as_h
      unless inputs_record.keys.to_set == input_names.to_set
        raise ArgumentError.new("MPS reduction input inventory mismatch: #{name}")
      end

      batches = @config.batches
      query_rows = @config.joint_tokens
      key_rows = @config.joint_tokens
      heads = @config.heads
      head_dim = @config.head_dim
      q_shape = [batches, query_rows, heads, head_dim]
      score_shape = [batches, query_rows, heads, key_rows]
      output_shape = {"qk", "softmax"}.includes?(kind) ? score_shape : q_shape
      input_shapes = case kind
                     when "qk", "attention" then input_names.map { q_shape }
                     when "softmax"         then [score_shape]
                     when "pv"              then [score_shape, q_shape]
                     else                        raise ArgumentError.new("Unsupported MPS reduction kind: #{kind}")
                     end
      input_names.each_with_index do |input_name, input_index|
        require_shape!(inputs_record[input_name], input_shapes[input_index], "#{name}/#{input_name}")
      end
      require_shape!(op["expected"], output_shape, "#{name}/output")

      scale = if {"qk", "attention"}.includes?(kind)
                HyAttentionProbe.scale(op, @config)
              else
                1.0_f32
              end
      q_count = ML::Video::HyNomasDualStreamBlock.checked_count(
        batches, query_rows, heads, head_dim,
      )
      score_count = ML::Video::HyNomasDualStreamBlock.checked_count(
        batches, query_rows, heads, key_rows,
      )
      output_count = if {"qk", "softmax"}.includes?(kind)
                       score_count
                     else
                       q_count
                     end
      input_counts = case kind
                     when "qk", "attention" then Array(Int32).new(input_names.size, q_count)
                     when "softmax"         then [score_count]
                     when "pv"              then [score_count, q_count]
                     else                        raise ArgumentError.new("Unsupported MPS reduction kind: #{kind}")
                     end
      inputs = input_names.map_with_index do |input_name, input_index|
        key = "#{index}.input.#{input_name}"
        buffer = @arena[key]
        raise ArgumentError.new("MPS reductions require shared F32 inputs") unless @arena.dtype(key).f32? && buffer.storage_mode.shared?
        ML::Video::MetalOps.check(buffer, input_counts[input_index].to_i64)
        buffer
      end
      packed_inputs = input_names.map_with_index do |input_name, input_index|
        key = "#{index}.mps_reduction.packed.#{input_name}"
        buffer = @arena.alloc(key, input_counts[input_index])
        buffer
      end
      packed_output = @arena.alloc("#{index}.mps_reduction.packed_output", output_count)
      output = @arena["#{index}.output"]
      unless @arena.dtype("#{index}.output").f32? && output.storage_mode.shared?
        raise ArgumentError.new("MPS reductions require a shared F32 output")
      end
      {packed_output, output}.each { |buffer| ML::Video::MetalOps.check(buffer, output_count.to_i64) }
      validate_distinct!(inputs + packed_inputs + [packed_output, output], name)

      handle = Pointer(Void).null
      {% if flag?(:darwin) && !flag?(:cpu_only) %}
        pointers = Array(Pointer(Void)).new(3, Pointer(Void).null)
        inputs.each_with_index { |buffer, input_index| pointers[input_index] = buffer.handle }
        packed_pointers = Array(Pointer(Void)).new(3, Pointer(Void).null)
        packed_inputs.each_with_index { |buffer, input_index| packed_pointers[input_index] = buffer.handle }
        mode = case kind
               when "qk"        then 0
               when "softmax"   then 1
               when "pv"        then 2
               when "attention" then 3
               else                  raise ArgumentError.new("Unsupported MPS reduction kind: #{kind}")
               end
        handle = HyNomasMPSReductionFFI.plan_create(
          ML::Metal::Device.instance.device_handle, mode, batches, query_rows,
          key_rows, heads, head_dim, scale, pointers[0], pointers[1], pointers[2],
          packed_pointers[0], packed_pointers[1], packed_pointers[2],
          packed_output.handle, output.handle,
        )
        raise "Failed to prepare MPS reduction graph for #{name}" if handle.null?
      {% else %}
        raise ArgumentError.new("MPS reduction controls require macOS Metal Performance Shaders")
      {% end %}
      Entry.new(index, name, kind, scale, input_names, inputs, packed_inputs,
        packed_output, output, handle)
    end

    private def validate_operation_identity!(op : JSON::Any, entry : Entry) : Nil
      unless op["name"].as_s == entry.name && op["kind"].as_s == entry.kind
        raise ArgumentError.new("MPS reduction operation identity changed")
      end
      scale = if {"qk", "attention"}.includes?(entry.kind)
                HyAttentionProbe.scale(op, @config)
              else
                1.0_f32
              end
      raise ArgumentError.new("MPS reduction operation scale changed") unless scale == entry.scale
    end

    private def validate_bindings!(entry : Entry) : Nil
      raise "MPS reduction plan has been released" if entry.handle.null?
      entry.input_names.each_with_index do |name, input_index|
        buffer_key = "#{entry.index}.input.#{name}"
        current = @arena[buffer_key]
        packed_key = "#{entry.index}.mps_reduction.packed.#{name}"
        packed = @arena[packed_key]
        unless current.valid? && current.same?(entry.inputs[input_index]) &&
               @arena.dtype(buffer_key).f32? && current.storage_mode.shared? &&
               packed.valid? && packed.same?(entry.packed_inputs[input_index]) &&
               @arena.dtype(packed_key).f32? && packed.storage_mode.shared?
          raise ArgumentError.new("MPS reduction input binding changed: #{entry.name}/#{name}")
        end
      end
      packed_output = @arena["#{entry.index}.mps_reduction.packed_output"]
      current_output = @arena["#{entry.index}.output"]
      unless packed_output.valid? && packed_output.same?(entry.packed_output) &&
             packed_output.storage_mode.shared? && current_output.valid? &&
             current_output.same?(entry.output) && current_output.storage_mode.shared?
        raise ArgumentError.new("MPS reduction output binding changed: #{entry.name}")
      end
    end

    private def validate_distinct!(buffers : Array(ML::MetalBuffer), label : String) : Nil
      buffers.each_with_index do |left, left_index|
        buffers.each_with_index do |right, right_index|
          next if right_index <= left_index
          if ML::Video::MetalOps.overlap?(left, right)
            raise ArgumentError.new("MPS reduction buffers must be distinct: #{label}")
          end
        end
      end
    end

    private def require_shape!(record : JSON::Any, expected : Array(Int32), label : String) : Nil
      actual = record["shape"].as_a.map(&.as_i)
      unless actual == expected
        raise ArgumentError.new("MPS reduction layout mismatch: #{label}")
      end
    end
  end
end
