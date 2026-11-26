require "../weights"
require "./scratch"

{% if flag?(:darwin) && !flag?(:cpu_only) %}
  module ML::GGUF
    # Encode-only elementwise and depth-residual operations for the resident
    # Inkling 2 path. The caller owns command-buffer lifetime and scratch views.
    module Inkling2ResidentMath
      THREADS = 256_i32
      SOURCE = {{ read_file("#{__DIR__}/../../../metal/kernels/inkling2/inkling2_resident_math.metal") }}
      private TOP1_TILE = 4 * THREADS

    # CPU RMSNorm accumulates squared values in Float64 before converting the
    # reciprocal scale to Float32. This row-parallel Metal implementation
    # keeps the same epsilon, feature weights, and final multiply order.
    def self.rms(
      encoder : ML::Metal::ComputeEncoder, input : Inkling2View, output : Inkling2View,
      weights : Inkling2View?, width : Int32, rows : Int32, eps : Float32,
    ) : Nil
      validate_rows!(width, rows)
      weight = weights || input
      encoder.set_pipeline(pipeline("inkling2_resident_rms"))
      encoder.set_buffer(input.buffer, 0, offset: input.offset)
      encoder.set_buffer(weight.buffer, 1, offset: weight.offset)
      encoder.set_buffer(output.buffer, 2, ML::Metal::BufferAccess::Write, offset: output.offset)
      encoder.set_value(width.to_u32, 3)
      encoder.set_value(rows.to_u32, 4)
      encoder.set_value(eps, 5)
      encoder.set_value(weights ? 1_u32 : 0_u32, 6)
      encoder.dispatch_threadgroups({rows, 1, 1}, {THREADS, 1, 1})
    end

    # Scalar Q, K, and V use separate weights and write consecutive rows.
    def self.rms_triplet(
      encoder : ML::Metal::ComputeEncoder,
      inputs : {Inkling2View, Inkling2View, Inkling2View},
      weights : {Inkling2View, Inkling2View, Inkling2View},
      output : Inkling2View, width : Int32, eps : Float32,
    ) : Nil
      output_elements = triplet_elements(width, 3)
      validate_view_extent!(output, output_elements, "RMS triplet output")
      (inputs + weights).each do |view|
        validate_view_extent!(view, width, "RMS triplet source")
        if output.buffer.handle == view.buffer.handle &&
           output.offset < view.offset + width.to_i64 * sizeof(Float32) &&
           view.offset < output.offset + output_elements.to_i64 * sizeof(Float32)
          raise ArgumentError.new("RMS triplet output overlaps a source")
        end
      end
      encoder.set_pipeline(pipeline("inkling2_resident_rms_triplet"))
      inputs.each_with_index do |input, index|
        weight = weights[index]
        encoder.set_buffer(input.buffer, index, offset: input.offset)
        encoder.set_buffer(weight.buffer, index + 3, offset: weight.offset)
      end
      encoder.set_buffer(output.buffer, 6, ML::Metal::BufferAccess::Write, offset: output.offset)
      encoder.set_value(width.to_u32, 7)
      encoder.set_value(eps, 8)
      encoder.dispatch_threadgroups({3, 1, 1}, {THREADS, 1, 1})
    end

    # Controller input is the concatenation of independently normalized
    # input and message vectors (Crystal Array#+ concatenates these arrays).
    # Fuse both reductions and the join into one dispatch.
    def self.rms_concat(
      encoder : ML::Metal::ComputeEncoder, x : Inkling2View, y : Inkling2View, output : Inkling2View,
      width : Int32, rows : Int32, eps : Float32,
    ) : Nil
      validate_rows!(width, rows)
      encoder.set_pipeline(pipeline("inkling2_resident_rms_concat"))
      encoder.set_buffer(x.buffer, 0, offset: x.offset)
      encoder.set_buffer(y.buffer, 1, offset: y.offset)
      encoder.set_buffer(output.buffer, 2, ML::Metal::BufferAccess::Write, offset: output.offset)
      encoder.set_value(width.to_u32, 3)
      encoder.set_value(rows.to_u32, 4)
      encoder.set_value(eps, 5)
      encoder.dispatch_threadgroups({rows, 1, 1}, {THREADS, 1, 1})
    end

    # Scalar decode: preserve three independent RMS reductions but encode their
    # work as three threadgroups in one dispatch. `messages` and
    # `output` are consecutive triplets; input views may be distinct slices or
    # aliases of the same vector.
    def self.rms_concat_triplet(
      encoder : ML::Metal::ComputeEncoder,
      inputs : {Inkling2View, Inkling2View, Inkling2View},
      messages : Inkling2View, output : Inkling2View, width : Int32, eps : Float32,
    ) : Nil
      validate_rows!(width, 1_i32)
      message_elements = triplet_elements(width, 3)
      output_elements = rms_concat_triplet_elements(width)
      inputs.each_with_index do |input, index|
        validate_view_extent!(input, width, "RMS-concat triplet input #{index}")
      end
      validate_view_extent!(messages, message_elements, "RMS-concat triplet messages")
      validate_view_extent!(output, output_elements, "RMS-concat triplet output")

      encoder.set_pipeline(pipeline("inkling2_resident_rms_concat_triplet"))
      inputs.each_with_index do |input, index|
        encoder.set_buffer(input.buffer, index, offset: input.offset)
      end
      encoder.set_buffer(messages.buffer, 3, offset: messages.offset)
      encoder.set_buffer(output.buffer, 4, ML::Metal::BufferAccess::Write, offset: output.offset)
      encoder.set_value(width.to_u32, 5)
      encoder.set_value(eps, 6)
      encoder.dispatch_threadgroups({3, 1, 1}, {THREADS, 1, 1})
    end

    # Shared checked arithmetic for triplet scratch allocations and the
    # triplet encoder's packed message/output layouts.
    def self.triplet_elements(elements_per_item : Int32, items : Int32) : Int32
      raise ArgumentError.new("Inkling 2 triplet element width must be positive") unless elements_per_item > 0
      raise ArgumentError.new("Inkling 2 triplet item count must be positive") unless items > 0
      elements = elements_per_item.to_i64 * items.to_i64
      raise ArgumentError.new("Inkling 2 triplet element count exceeds Int32") if elements > Int32::MAX
      elements.to_i32
    end

    def self.rms_concat_triplet_elements(width : Int32) : Int32
      triplet_elements(width, 6)
    end

    # `bank` is normalized and laid out [depth][row][width]. The optional
    # normalized current state is appended after all bank entries, matching
    # Inkling2Math.depth_mix's [blocks + current] ordering.
    def self.depth_mix(
      encoder : ML::Metal::ComputeEncoder, bank : Inkling2View, count : Int32,
      current_norm : Inkling2View?, queries : Inkling2View, output : Inkling2View,
      width : Int32, rows : Int32, query_count : Int32,
    ) : Nil
      validate_rows!(width, rows)
      raise ArgumentError.new("Inkling 2 depth mix requires at least one key") if count < 0 || (count == 0 && current_norm.nil?)
      raise ArgumentError.new("Inkling 2 depth mix bank is too large") if count > 512
      raise ArgumentError.new("Inkling 2 depth mix requires a query") unless query_count > 0

      # Eight SIMD groups each write one partial dot product per key. One
      # additional value per key holds its completed score, then its weight.
      keys = count + (current_norm ? 1 : 0)
      payload_bytes = keys.to_i64 * 9_i64 * sizeof(Float32)
      # Metal requires a 16-byte-aligned threadgroup memory length.
      scratch_bytes = (payload_bytes + 15) // 16 * 16
      raise ArgumentError.new("Inkling 2 depth mix scratch exceeds Int32") if scratch_bytes > Int32::MAX

      current = current_norm || bank
      encoder.set_pipeline(pipeline("inkling2_resident_depth_mix"))
      encoder.set_buffer(bank.buffer, 0, offset: bank.offset)
      encoder.set_buffer(current.buffer, 1, offset: current.offset)
      encoder.set_buffer(queries.buffer, 2, offset: queries.offset)
      encoder.set_buffer(output.buffer, 3, ML::Metal::BufferAccess::Write, offset: output.offset)
      encoder.set_value(width.to_u32, 4)
      encoder.set_value(rows.to_u32, 5)
      encoder.set_value(count.to_u32, 6)
      encoder.set_value(query_count.to_u32, 7)
      encoder.set_value(current_norm ? 1_u32 : 0_u32, 8)
      encoder.set_threadgroup_memory(scratch_bytes.to_i32, 0)
      encoder.dispatch_threadgroups({rows, query_count, 1}, {THREADS, 1, 1})
    end

    # Elementwise add permits output to alias either input.
    def self.add(
      encoder : ML::Metal::ComputeEncoder, x : Inkling2View, y : Inkling2View,
      output : Inkling2View, count : Int32,
    ) : Nil
      validate_count!(count)
      encoder.set_pipeline(pipeline("inkling2_resident_add"))
      encoder.set_buffer(x.buffer, 0, offset: x.offset)
      encoder.set_buffer(y.buffer, 1, offset: y.offset)
      encoder.set_buffer(output.buffer, 2, ML::Metal::BufferAccess::Write, offset: output.offset)
      encoder.set_value(count.to_u32, 3)
      encoder.dispatch_1d(count, THREADS)
    end

    def self.copy(
      encoder : ML::Metal::ComputeEncoder, input : Inkling2View, output : Inkling2View,
      count : Int32,
    ) : Nil
      validate_count!(count)
      encoder.set_pipeline(pipeline("inkling2_resident_copy"))
      encoder.set_buffer(input.buffer, 0, offset: input.offset)
      encoder.set_buffer(output.buffer, 1, ML::Metal::BufferAccess::Write, offset: output.offset)
      encoder.set_value(count.to_u32, 2)
      encoder.dispatch_1d(count, THREADS)
    end

    # Applies the CPU controller equation in its original operation order:
    # input - tanh(forget)*input + tanh(gain)*message.
    def self.controls(
      encoder : ML::Metal::ComputeEncoder, input : Inkling2View, message : Inkling2View,
      gain : Inkling2View, forget : Inkling2View, output : Inkling2View, count : Int32,
    ) : Nil
      validate_count!(count)
      encoder.set_pipeline(pipeline("inkling2_resident_controls"))
      encoder.set_buffer(input.buffer, 0, offset: input.offset)
      encoder.set_buffer(message.buffer, 1, offset: message.offset)
      encoder.set_buffer(gain.buffer, 2, offset: gain.offset)
      encoder.set_buffer(forget.buffer, 3, offset: forget.offset)
      encoder.set_buffer(output.buffer, 4, ML::Metal::BufferAccess::Write, offset: output.offset)
      encoder.set_value(count.to_u32, 5)
      encoder.dispatch_1d(count, THREADS)
    end

    # Returns false when the embedding encoding is not admitted; callers can
    # stage those loader-supported formats through their explicit fallback.
    # The current resident path handles the checkpoint's F32 and Q8_0 rows,
    # plus IQ4_NL for compact local fixtures.
    def self.embed(
      encoder : ML::Metal::ComputeEncoder, weight : Inkling2View, weight_type : TensorType,
      token_ids : Inkling2View, output : Inkling2View, width : Int32, rows : Int32,
    ) : Bool
      return false unless weight_type.f32? || weight_type.q8_0? || weight_type.iq4_nl?
      validate_rows!(width, rows)
      if weight_type.q8_0? || weight_type.iq4_nl?
        return false unless width % weight_type.block_elements == 0
      end

      row_bytes = (width // weight_type.block_elements) * weight_type.block_bytes
      encoder.set_pipeline(pipeline("inkling2_resident_embed"))
      encoder.set_buffer(weight.buffer, 0, offset: weight.offset)
      encoder.set_buffer(token_ids.buffer, 1, offset: token_ids.offset)
      encoder.set_buffer(output.buffer, 2, ML::Metal::BufferAccess::Write, offset: output.offset)
      encoder.set_value(width.to_u32, 3)
      encoder.set_value(rows.to_u32, 4)
      encoder.set_value(row_bytes.to_u32, 5)
      encoder.set_value(weight_type.value.to_u32, 6)
      encoder.dispatch_1d(width * rows, THREADS)
      true
    end

    # Cap is zero when disabled. Input and output may be the same view.
    def self.softcap(
      encoder : ML::Metal::ComputeEncoder, input : Inkling2View, output : Inkling2View,
      cap : Float32, vocab : Int32,
    ) : Nil
      validate_count!(vocab)
      encoder.set_pipeline(pipeline("inkling2_resident_softcap"))
      encoder.set_buffer(input.buffer, 0, offset: input.offset)
      encoder.set_buffer(output.buffer, 1, ML::Metal::BufferAccess::Write, offset: output.offset)
      encoder.set_value(vocab.to_u32, 2)
      encoder.set_value(cap, 3)
      encoder.dispatch_1d(vocab, THREADS)
    end

    # The reference is already in the target's score space. Output may alias target.
    def self.extrapolate(
      encoder : ML::Metal::ComputeEncoder, target : Inkling2View, reference : Inkling2View, output : Inkling2View,
      count : Int32, strength : Float32, *, cap : Float32 = 0_f32,
    ) : Nil
      validate_count!(count)
      validate_view_extent!(target, count, "LoopCD target")
      validate_view_extent!(reference, count, "LoopCD reference")
      validate_view_extent!(output, count, "LoopCD output")
      encoder.set_pipeline(pipeline("inkling2_resident_extrapolate"))
      encoder.set_buffer(target.buffer, 0, offset: target.offset)
      encoder.set_buffer(reference.buffer, 1, offset: reference.offset)
      encoder.set_buffer(output.buffer, 2, ML::Metal::BufferAccess::Write, offset: output.offset)
      encoder.set_value(count.to_u32, 3)
      encoder.set_value(strength, 4)
      encoder.set_value(cap, 5)
      encoder.dispatch_1d(count, THREADS)
    end

    # Reduce only the unpadded vocabulary, applying the same softcap as the
    # full-logit path before selecting the first finite maximum. A row without
    # a finite score returns -1, which the greedy caller handles when consumed.
    def self.greedy_ids(
      encoder : ML::Metal::ComputeEncoder, scratch : Inkling2ResidentScratch,
      input : Inkling2View, output : Inkling2View, cap : Float32,
      vocab : Int32, stride : Int32, rows : Int32,
    ) : Nil
      validate_rows!(stride, rows)
      raise ArgumentError.new("Inkling 2 greedy vocabulary must fit in one row") unless 0 < vocab <= stride
      tiles = (vocab - 1) // TOP1_TILE + 1
      partials = scratch.take(2 * tiles * rows)
      encoder.set_pipeline(pipeline("inkling2_resident_greedy_tiles"))
      encoder.set_buffer(input.buffer, 0, offset: input.offset)
      encoder.set_buffer(partials.buffer, 1, ML::Metal::BufferAccess::Write, offset: partials.offset)
      encoder.set_value(vocab.to_u32, 2)
      encoder.set_value(stride.to_u32, 3)
      encoder.set_value(tiles.to_u32, 4)
      encoder.set_value(cap, 5)
      encoder.dispatch_threadgroups({tiles, rows, 1}, {THREADS, 1, 1})

      encoder.set_pipeline(pipeline("inkling2_resident_greedy_reduce"))
      encoder.set_buffer(partials.buffer, 0, offset: partials.offset)
      encoder.set_buffer(output.buffer, 1, ML::Metal::BufferAccess::Write, offset: output.offset)
      encoder.set_value(tiles.to_u32, 2)
      encoder.dispatch_threadgroups({rows, 1, 1}, {THREADS, 1, 1})
    end

    private def self.validate_rows!(width : Int32, rows : Int32) : Nil
      raise ArgumentError.new("Inkling 2 resident math width must be positive") unless width > 0
      raise ArgumentError.new("Inkling 2 resident math rows must be positive") unless rows > 0
      raise ArgumentError.new("Inkling 2 resident math element count exceeds Int32") if width.to_i64 * rows > Int32::MAX
    end

    private def self.validate_count!(count : Int32) : Nil
      raise ArgumentError.new("Inkling 2 resident math count must be positive") unless count > 0
    end

    private def self.validate_view_extent!(view : Inkling2View, elements : Int32, label : String) : Nil
      bytes = elements.to_i64 * sizeof(Float32)
      offset = view.offset
      unless offset >= 0 && offset <= view.buffer.size && bytes <= view.buffer.size - offset
        raise ArgumentError.new("#{label} view exceeds its Metal buffer")
      end
    end

    private def self.pipeline(name : String) : ML::Metal::ComputePipeline
      ML::Metal::PipelineCache.get(name) do
        ML::Metal::ComputePipeline.new(name, SOURCE)
      end
    end
    end
  end
{% end %}
