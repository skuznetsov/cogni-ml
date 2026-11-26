require "./prompt"
require "./resident"

{% if flag?(:drama3_metal) && flag?(:darwin) && !flag?(:cpu_only) %}
  module ML::Audio::Drama3
    RESIDENT_PROMPT_SOURCE = {{ read_file("#{__DIR__}/kernels/prompt_resident.metal") }}

    # The flat row-major prompt buffer stays in the same resident context as its
    # encoder codes and the model. Audio rows occupy the plan's marked span.
    class ResidentPrompt
      getter rows : Resident::Buffer
      getter row_count : Int32
      getter row_width : Int32
      getter audio_row_start : Int32
      getter audio_row_count : Int32
      getter codebooks : Int32
      getter semantic_begin : Int32
      getter codebook_size : Int32
      getter context : Resident::Context

      def initialize(@rows : Resident::Buffer, @row_count : Int32, @row_width : Int32,
                     @audio_row_start : Int32, @audio_row_count : Int32, @codebooks : Int32,
                     @semantic_begin : Int32, @codebook_size : Int32, @context : Resident::Context)
        values = @row_count.to_i64 * @row_width
        raise ArgumentError.new("invalid resident prompt rows") unless @row_count > 0 &&
                                                                       @row_count <= 8192 && @codebooks > 0 && @row_width == @codebooks + 1 &&
                                                                       @codebook_size > 0 && @semantic_begin >= 0 && @audio_row_count > 0 &&
                                                                       @audio_row_start >= 0 && @audio_row_start.to_i64 + @audio_row_count <= @row_count &&
                                                                       values == @rows.count
      end

      def row(index : Int32) : Resident::Buffer
        raise IndexError.new("resident prompt row") unless index >= 0 && index < @row_count
        @rows.slice(index * @row_width, @row_width)
      end
    end

    module PromptResident
      extend self

      # `codes` is codebook-major [codebooks,frames] F32 integer IDs from the
      # encoder. The GPU kernel writes `[c0 + semantic_begin,c0,...]` directly
      # into the prompt; invalid IDs become safe zeros and poison the existing
      # finite guard, which fails at the generator's next control read.
      def build(plan : Prompt::ReferencePlan, codes : Resident::Buffer,
                context : Resident::Context) : ResidentPrompt
        raise ArgumentError.new("invalid resident prompt plan") unless plan.row_count > 0 &&
                                                                       plan.row_count <= 8192 && plan.codebooks > 0 && plan.codebooks < Int32::MAX &&
                                                                       plan.codebook_size > 0 && plan.semantic_begin >= 0 &&
                                                                       plan.semantic_begin.to_i64 + plan.codebook_size - 1 <= 16_777_215_i64 &&
                                                                       plan.audio_row_start == plan.prefix_rows.size && plan.audio_row_count > 0 &&
                                                                       plan.audio_row_start + plan.audio_row_count + plan.suffix_rows.size == plan.row_count &&
                                                                       plan.row_width == plan.codebooks + 1
        raise ArgumentError.new("resident prompt row width overflow") unless plan.row_width > 1
        total_values = plan.row_count.to_i64 * plan.row_width
        raise ArgumentError.new("resident prompt is too large") unless total_values > 0 && total_values <= Int32::MAX
        expected_codes = plan.codebooks.to_i64 * plan.audio_row_count
        raise ArgumentError.new("reference code buffer shape mismatch") unless expected_codes == codes.count

        values = Array(Float32).new(total_values.to_i32, 0_f32)
        write_text_rows(values, plan.prefix_rows, 0, plan.row_width)
        write_text_rows(values, plan.suffix_rows,
          plan.audio_row_start + plan.audio_row_count, plan.row_width)
        rows = context.upload(values)
        validity = context.upload([1_f32])
        context.dispatch(
          "dr3_prompt_reference_rows",
          RESIDENT_PROMPT_SOURCE,
          [codes, rows, validity],
          [plan.audio_row_count.to_u64, plan.codebooks.to_u64, plan.row_width.to_u64,
           plan.audio_row_start.to_u64, plan.semantic_begin.to_u64, plan.codebook_size.to_u64],
          codes.count,
        )
        context.guard(validity)

        ResidentPrompt.new(rows, plan.row_count, plan.row_width,
          plan.audio_row_start, plan.audio_row_count, plan.codebooks,
          plan.semantic_begin, plan.codebook_size, context)
      end

      private def write_text_rows(output : Array(Float32), rows : Array(Array(Int32)),
                                  start_row : Int32, width : Int32) : Nil
        rows.each_with_index do |row, row_index|
          raise ArgumentError.new("resident prompt text row width mismatch") unless row.size == width
          row.each_with_index do |token, column|
            raise ArgumentError.new("resident prompt token is not exactly representable") unless token >= 0 && token <= 16_777_216
            output[(start_row + row_index) * width + column] = token.to_f32
          end
        end
      end
    end
  end
{% end %}
