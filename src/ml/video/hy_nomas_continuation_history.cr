# F32 persistent K/V history for sequential compact-text Hy-Nomas segments.
# The provider is setup-bound: cache buffers, graph workspaces, plans, and the
# append pipeline are prepared before serial encode. Encode only appends to the
# caller's chain; no Metal allocations, host copies, reads, or waits occur.
require "./hy_nomas_block_mps"
require "./hy_nomas_history_attention"
require "./metal_ops"

module ML::Video
  module HyNomasContinuationHistoryKernels
    APPEND_KERNEL = "hy_nomas_continuation_history_append_f32"
    SOURCE        = MetalOps::SOURCE + "\n" + {{ read_file("#{__DIR__}/kernels/hy_nomas_continuation_history.metal") }}

    {% if flag?(:hy_nomas_history_probe) %}
      FILL_KERNEL  = "hy_nomas_continuation_history_fill_f32"
      FILL_SOURCE  = MetalOps::SOURCE + "\n" + {{ read_file("#{__DIR__}/kernels/hy_nomas_continuation_history_probe.metal") }}
    {% end %}

    class AppendPlan
      @pipeline : ML::Metal::ComputePipeline

      def initialize
        @pipeline = ML::Metal::PipelineCache.get(APPEND_KERNEL) do
          ML::Metal::ComputePipeline.new(APPEND_KERNEL, SOURCE)
        end
      end

      # Copy `count` active row-major F32 elements into `destination` at a flat
      # element offset. `count` and the offset are fixed by CPU geometry before
      # the serial encode begins.
      def encode(encoder : ML::Metal::ComputeEncoder,
                 source : ML::MetalBuffer, destination : ML::MetalBuffer,
                 count : Int32, destination_offset : Int32) : Nil
        unless count > 0 && destination_offset >= 0 &&
               destination_offset.to_i64 + count <= Int32::MAX
          raise ArgumentError.new("Hy-Nomas continuation append extent exceeds the native limit")
        end
        MetalOps.check(source, count.to_i64)
        MetalOps.check(destination, destination_offset.to_i64 + count)
        unless source.storage_mode.shared? && destination.storage_mode.shared?
          raise ArgumentError.new("Hy-Nomas continuation cache append requires shared F32 buffers")
        end
        MetalOps.distinct(destination, source)

        encoder.set_pipeline(@pipeline)
          .set_buffer(source, 0)
          .set_buffer(destination, 1, ML::Metal::BufferAccess::Write)
          .set_value(count.to_u32, 2)
          .set_value(destination_offset.to_u32, 3)
          .dispatch_1d(count)
      end
    end

    {% if flag?(:hy_nomas_history_probe) %}
      class FillPlan
        @pipeline : ML::Metal::ComputePipeline

        def initialize
          @pipeline = ML::Metal::PipelineCache.get(FILL_KERNEL) do
            ML::Metal::ComputePipeline.new(FILL_KERNEL, FILL_SOURCE)
          end
        end

        # Diagnostic setup hook: initialize persistent cache contents on the
        # caller's already-open command chain, without a host write or wait.
        def encode(encoder : ML::Metal::ComputeEncoder,
                   destination : ML::MetalBuffer, count : Int32,
                   value : Float32) : Nil
          unless count > 0 && value.finite?
            raise ArgumentError.new("Invalid Hy-Nomas continuation diagnostic fill")
          end
          MetalOps.check(destination, count.to_i64)
          unless destination.storage_mode.shared?
            raise ArgumentError.new("Hy-Nomas continuation diagnostic fill requires shared F32 buffers")
          end
          encoder.set_pipeline(@pipeline)
            .set_buffer(destination, 0, ML::Metal::BufferAccess::Write)
            .set_value(value, 1)
            .set_value(count.to_u32, 2)
            .dispatch_1d(count)
        end
      end
    {% end %}
  end

  # CPU-only forward cursor. Each segment consumes a fixed number of
  # read-only denoise forwards. A non-final segment then admits one separate
  # writer forward; the final segment advances after its last denoise pass.
  class HyNomasContinuationHistorySchedule
    getter segment_count : Int32
    getter layers : Int32
    getter denoise_steps : Int32
    getter next_segment : Int32 = 0_i32
    getter denoise_passes : Int32 = 0_i32
    getter? complete : Bool = false

    @selected_segment : Int32? = nil
    @selected_writer = false
    @next_layer = 0_i32

    def initialize(@segment_count : Int32, @layers : Int32,
                   @denoise_steps : Int32 = 4)
      unless (2..HyNomasContinuationHistory::MAX_SEGMENTS).includes?(@segment_count) &&
             @layers > 0 && @denoise_steps > 0
        raise ArgumentError.new("Invalid Hy-Nomas continuation forward schedule")
      end
    end

    def select_segment!(index : Int32, writer : Bool) : Nil
      if selected = @selected_segment
        if selected == index && @selected_writer == writer && @next_layer == 0
          return
        end
        raise ArgumentError.new("Finish the selected Hy-Nomas continuation forward first")
      end
      unless !@complete && index == @next_segment
        raise ArgumentError.new("Hy-Nomas continuation segments must be selected once in order")
      end

      if writer
        unless index < @segment_count - 1 && @denoise_passes == @denoise_steps
          raise ArgumentError.new("A continuation cache writer requires all denoise forwards first")
        end
      elsif @denoise_passes >= @denoise_steps
        raise ArgumentError.new("The denoise-forward limit is reached; a cache writer is required")
      end

      @selected_segment = index
      @selected_writer = writer
      @next_layer = 0_i32
    end

    # Restoring a committed post-writer boundary skips no in-flight work.
    # Cache bytes must be authenticated and installed by the owner first.
    def validate_restore_boundary!(next_segment : Int32) : Nil
      unless 1 <= next_segment < @segment_count && idle? &&
             @next_segment == 0 && @denoise_passes == 0 && !@complete
        raise ArgumentError.new("Continuation restore requires a fresh schedule and an extendable boundary")
      end
    end

    def restore_boundary!(next_segment : Int32) : Nil
      validate_restore_boundary!(next_segment)
      @next_segment = next_segment
    end

    def selected_segment : Int32
      @selected_segment || raise ArgumentError.new("Select a continuation segment before encoding")
    end

    def selected_writer? : Bool
      raise ArgumentError.new("Select a continuation segment before encoding") unless @selected_segment
      @selected_writer
    end

    def validate_layer!(block_index : Int32) : Nil
      raise ArgumentError.new("Select a continuation segment before encoding") unless @selected_segment
      unless block_index == @next_layer && block_index < @layers
        raise ArgumentError.new("Hy-Nomas continuation layers must encode once in order")
      end
    end

    # Advance only after the provider appended all operations for this layer.
    def finish_layer!(block_index : Int32) : Nil
      validate_layer!(block_index)
      @next_layer += 1
      return unless @next_layer == @layers

      segment = @selected_segment.not_nil!
      writer = @selected_writer
      @selected_segment = nil
      @selected_writer = false
      @next_layer = 0_i32

      if writer
        @next_segment += 1
        @denoise_passes = 0_i32
      else
        @denoise_passes += 1
        if segment == @segment_count - 1 && @denoise_passes == @denoise_steps
          @next_segment += 1
          @denoise_passes = 0_i32
        end
      end
      @complete = @next_segment == @segment_count
    end

    def awaiting_writer? : Bool
      @next_segment < @segment_count - 1 && @denoise_passes == @denoise_steps &&
        @selected_segment.nil?
    end

    def idle? : Bool
      @selected_segment.nil? && @next_layer == 0
    end
  end

  # Prepared rectangular attention workspaces and per-layer caches for one
  # bounded sequence. `select_segment!` is called before the serial owner
  # encodes a segment. Segments append their active K/V prefix except the final
  # segment, which cannot have a successor within this admitted cache extent.
  class HyNomasContinuationHistory < HyNomasMPSHistoryProvider
    MAX_SEGMENTS =  6
    MAX_LAYERS   = 54

    # PerChunk preserves the original experimental provider contract. A caller
    # that reuses one semantic scene must opt into OncePerScene, matching
    # upstream chunk_ar's cache_txt flag. Text queries remain active in both.
    enum TextCachePolicy
      PerChunk
      OncePerScene
    end

    # Immutable view of the prompt row counts and cache-text policy admitted
    # for one segment sequence. Inputs and returned arrays are copied so a
    # caller cannot change append geometry after workspace admission.
    class SegmentTextProfile
      @text_tokens : Array(Int32)
      @cache_text : Array(Bool)

      def initialize(text_tokens : Array(Int32), cache_text : Array(Bool))
        @text_tokens = text_tokens.dup
        @cache_text = cache_text.dup
      end

      def text_tokens : Array(Int32)
        @text_tokens.dup
      end

      def cache_text : Array(Bool)
        @cache_text.dup
      end

      def text_tokens_at(segment : Int32) : Int32
        @text_tokens[segment]
      end

      def cache_text_at?(segment : Int32) : Bool
        @cache_text[segment]
      end
    end

    record Geometry,
      segment_count : Int32,
      video_tokens : Int32,
      valid_text_tokens : Int32,
      storage_text_tokens : Int32,
      denoise_steps : Int32,
      layers : Int32,
      heads : Int32,
      head_dim : Int32,
      query_rows : Int32,
      storage_rows : Int32,
      query_elements : Int32,
      storage_elements : Int32,
      max_history_rows : Int32,
      cache_elements_per_buffer : Int32,
      cache_bytes : Int64,
      workspace_geometries : Array(HyNomasHistoryAttentionWorkspace::Geometry),
      writer_workspace_geometries : Array(HyNomasHistoryAttentionWorkspace::Geometry),
      text_cache_policy : TextCachePolicy,
      segment_text_profile : SegmentTextProfile,
      workspace_bytes : Int64,
      named_bytes : Int64 do
      def segment_text_tokens : Array(Int32)
        segment_text_profile.text_tokens
      end

      def segment_cache_text : Array(Bool)
        segment_text_profile.cache_text
      end

      def append_elements(segment : Int32) : Int32
        raise ArgumentError.new("Final or invalid chunk cannot append history") unless 0 <= segment < segment_count - 1
        rows = video_tokens.to_i64 + if segment_text_profile.cache_text_at?(segment)
                                       segment_text_profile.text_tokens_at(segment)
                                     else
                                       0_i64
                                     end
        (rows.to_i64 * heads * head_dim).to_i32
      end

      def append_offset(segment : Int32) : Int32
        raise ArgumentError.new("Final or invalid chunk cannot append history") unless 0 <= segment < segment_count - 1
        (workspace_geometries[segment].history_rows.to_i64 * heads * head_dim).to_i32
      end

      def boundary_cache_elements(next_segment : Int32) : Int32
        unless 1 <= next_segment < segment_count
          raise ArgumentError.new("Continuation checkpoint requires an extendable post-writer boundary")
        end
        (workspace_geometries[next_segment].history_rows.to_i64 * heads * head_dim).to_i32
      end
    end

    getter shape : Geometry
    @query : ML::MetalBuffer
    @key : ML::MetalBuffer
    @value : ML::MetalBuffer
    @output : ML::MetalBuffer
    @schedule : HyNomasContinuationHistorySchedule
    @cache_arena : Arena
    @cache_pairs = [] of Tuple(ML::MetalBuffer, ML::MetalBuffer)
    @history_workspaces = [] of HyNomasHistoryAttentionWorkspace
    @history_plans = [] of HyNomasHistoryAttentionMPSPlan
    @writer_plans = [] of HyNomasHistoryAttentionMPSPlan
    @append_plan : HyNomasContinuationHistoryKernels::AppendPlan? = nil
    {% if flag?(:hy_nomas_history_probe) %}
      @fill_plan : HyNomasContinuationHistoryKernels::FillPlan? = nil
      @probe_cache_filled = false
      @probe_pre_writer_snapshot = false
      @probe_post_writer_snapshot = false
      @probe_final_snapshot = false
    {% end %}
    @selected_segment : Int32? = nil
    @selected_writer = false
    @last_chain : ML::Metal::MPSCommandChain? = nil
    @poisoned = false
    @released = false

    def self.geometry(segment_count : Int32, video_tokens : Int32,
                      valid_text_tokens : Int32, storage_text_tokens : Int32,
                      layers : Int32, heads : Int32, head_dim : Int32,
                      denoise_steps : Int32 = 4, *,
                      text_cache_policy : TextCachePolicy = TextCachePolicy::PerChunk,
                      segment_text_tokens : Array(Int32)? = nil,
                      segment_cache_text : Array(Bool)? = nil) : Geometry
      unless (2..MAX_SEGMENTS).includes?(segment_count) &&
             video_tokens > 0 && valid_text_tokens > 0 &&
             storage_text_tokens >= valid_text_tokens &&
             layers == MAX_LAYERS && heads > 0 && head_dim > 0 && denoise_steps > 0
        raise ArgumentError.new("Invalid Hy-Nomas continuation history geometry")
      end

      profile = validated_segment_text_profile(segment_count, valid_text_tokens,
        text_cache_policy, segment_text_tokens, segment_cache_text)

      query_rows_64 = video_tokens.to_i64 + valid_text_tokens
      storage_rows_64 = video_tokens.to_i64 + storage_text_tokens
      append_rows = Array(Int32).new(segment_count) do |segment|
        rows = video_tokens.to_i64 + if profile.cache_text_at?(segment)
                                       profile.text_tokens_at(segment)
                                     else
                                       0_i64
                                     end
        unless rows <= Int32::MAX
          raise ArgumentError.new("Hy-Nomas continuation append rows exceed the native limit")
        end
        rows.to_i32
      end
      prefix_rows = Array(Int32).new(segment_count, 0_i32)
      prefix = 0_i64
      segment_count.times do |segment|
        unless prefix <= Int32::MAX
          raise ArgumentError.new("Hy-Nomas continuation history rows exceed the native limit")
        end
        prefix_rows[segment] = prefix.to_i32
        prefix = checked_sum(prefix, append_rows[segment].to_i64,
          "Hy-Nomas continuation history rows")
      end
      max_history_rows_64 = prefix_rows[segment_count - 1].to_i64
      unless query_rows_64 <= Int32::MAX && storage_rows_64 <= Int32::MAX &&
             max_history_rows_64 <= Int32::MAX
        raise ArgumentError.new("Hy-Nomas continuation history rows exceed the native limit")
      end
      query_rows = query_rows_64.to_i32
      storage_rows = storage_rows_64.to_i32
      max_history_rows = max_history_rows_64.to_i32

      query_elements_64 = checked_product(
        checked_product(query_rows_64, heads.to_i64, "Hy-Nomas continuation query elements"),
        head_dim.to_i64, "Hy-Nomas continuation query elements")
      storage_elements_64 = checked_product(
        checked_product(storage_rows_64, heads.to_i64, "Hy-Nomas continuation storage elements"),
        head_dim.to_i64, "Hy-Nomas continuation storage elements")
      cache_elements_64 = checked_product(
        checked_product(max_history_rows_64, heads.to_i64, "Hy-Nomas continuation cache elements"),
        head_dim.to_i64, "Hy-Nomas continuation cache elements")
      unless query_elements_64 <= Int32::MAX && storage_elements_64 <= Int32::MAX &&
             cache_elements_64 > 0 && cache_elements_64 <= Int32::MAX
        raise ArgumentError.new("Hy-Nomas continuation history tensor exceeds the native limit")
      end

      pair_bytes = checked_product(cache_elements_64, 2_i64 * sizeof(Float32),
        "Hy-Nomas continuation K/V cache bytes")
      cache_bytes = checked_product(pair_bytes, layers.to_i64,
        "Hy-Nomas continuation cache inventory bytes")
      workspace_geometries = Array(HyNomasHistoryAttentionWorkspace::Geometry).new(segment_count)
      writer_workspace_geometries = Array(HyNomasHistoryAttentionWorkspace::Geometry).new(segment_count - 1)
      workspace_bytes = 0_i64
      segment_count.times do |segment|
        history_rows = prefix_rows[segment].to_i64
        unless history_rows <= Int32::MAX
          raise ArgumentError.new("Hy-Nomas continuation segment history exceeds the native limit")
        end
        current_rows = append_rows[segment]
        workspace = HyNomasHistoryAttentionWorkspace.geometry(
          history_rows.to_i32, query_rows, heads, head_dim, current_rows: current_rows)
        workspace_geometries << workspace
        workspace_bytes = checked_sum(workspace_bytes, workspace.workspace_bytes,
          "Hy-Nomas continuation workspace bytes")
        if segment < segment_count - 1
          # The upstream cache writer attends the current prompt transiently
          # even when persistent history already contains the same text.
          writer_current_rows = video_tokens.to_i64 + profile.text_tokens_at(segment)
          unless writer_current_rows <= Int32::MAX
            raise ArgumentError.new("Hy-Nomas continuation writer rows exceed the native limit")
          end
          writer_workspace = if writer_current_rows == current_rows
                               workspace
                             else
                               HyNomasHistoryAttentionWorkspace.geometry(
                                 history_rows.to_i32, query_rows, heads, head_dim,
                                 current_rows: writer_current_rows.to_i32)
                             end
          writer_workspace_geometries << writer_workspace
          unless writer_workspace == workspace
            workspace_bytes = checked_sum(workspace_bytes, writer_workspace.workspace_bytes,
              "Hy-Nomas continuation writer workspace bytes")
          end
        end
      end
      named_bytes = checked_sum(cache_bytes, workspace_bytes,
        "Hy-Nomas continuation named bytes")

      Geometry.new(segment_count, video_tokens, valid_text_tokens,
        storage_text_tokens, denoise_steps, layers, heads, head_dim, query_rows, storage_rows,
        query_elements_64.to_i32, storage_elements_64.to_i32,
        max_history_rows, cache_elements_64.to_i32, cache_bytes,
        workspace_geometries, writer_workspace_geometries, text_cache_policy, profile,
        workspace_bytes, named_bytes)
    end

    private def self.validated_segment_text_profile(segment_count : Int32,
                                                    valid_text_tokens : Int32,
                                                    text_cache_policy : TextCachePolicy,
                                                    segment_text_tokens : Array(Int32)?,
                                                    segment_cache_text : Array(Bool)?) : SegmentTextProfile
      if segment_text_tokens.nil? != segment_cache_text.nil?
        raise ArgumentError.new("Per-segment prompt rows and cache flags must be supplied together")
      end
      tokens = segment_text_tokens || Array(Int32).new(segment_count, valid_text_tokens)
      cache_text = segment_cache_text || Array(Bool).new(segment_count) do |segment|
        !text_cache_policy.once_per_scene? || segment == 0
      end
      unless tokens.size == segment_count && cache_text.size == segment_count &&
             tokens.all? { |count| count > 0 && count <= valid_text_tokens } &&
             tokens.max? == valid_text_tokens
        raise ArgumentError.new("Invalid Hy-Nomas per-segment prompt token counts")
      end
      unless cache_text.first
        raise ArgumentError.new("The first Hy-Nomas segment must cache its prompt text")
      end

      last_cached_text_tokens = tokens.first
      (1...segment_count).each do |segment|
        if cache_text[segment]
          last_cached_text_tokens = tokens[segment]
        elsif tokens[segment] != last_cached_text_tokens
          raise ArgumentError.new("Uncached Hy-Nomas prompt rows must match the last cached prompt")
        end
      end
      SegmentTextProfile.new(tokens, cache_text)
    end

    def self.named_bytes(segment_count : Int32, video_tokens : Int32,
                         valid_text_tokens : Int32, storage_text_tokens : Int32,
                         layers : Int32, heads : Int32, head_dim : Int32,
                         denoise_steps : Int32 = 4, *,
                         text_cache_policy : TextCachePolicy = TextCachePolicy::PerChunk,
                         segment_text_tokens : Array(Int32)? = nil,
                         segment_cache_text : Array(Bool)? = nil) : Int64
      geometry(segment_count, video_tokens, valid_text_tokens, storage_text_tokens,
        layers, heads, head_dim, denoise_steps, text_cache_policy: text_cache_policy,
        segment_text_tokens: segment_text_tokens,
        segment_cache_text: segment_cache_text).named_bytes
    end

    def self.named_bytes(*, segment_count : Int32, video_tokens : Int32,
                         valid_text_tokens : Int32, storage_text_tokens : Int32,
                         layers : Int32, heads : Int32, head_dim : Int32,
                         denoise_steps : Int32 = 4,
                         text_cache_policy : TextCachePolicy = TextCachePolicy::PerChunk,
                         segment_text_tokens : Array(Int32)? = nil,
                         segment_cache_text : Array(Bool)? = nil) : Int64
      geometry(segment_count, video_tokens, valid_text_tokens, storage_text_tokens,
        layers, heads, head_dim, denoise_steps, text_cache_policy: text_cache_policy,
        segment_text_tokens: segment_text_tokens,
        segment_cache_text: segment_cache_text).named_bytes
    end

    def initialize(*, segment_count : Int32, video_tokens : Int32,
                   valid_text_tokens : Int32, storage_text_tokens : Int32,
                   layers : Int32, heads : Int32, head_dim : Int32,
                   denoise_steps : Int32 = 4,
                   text_cache_policy : TextCachePolicy = TextCachePolicy::PerChunk,
                   segment_text_tokens : Array(Int32)? = nil,
                   segment_cache_text : Array(Bool)? = nil,
                   query : ML::MetalBuffer, key : ML::MetalBuffer,
                   value : ML::MetalBuffer, output : ML::MetalBuffer)
      @shape = self.class.geometry(segment_count, video_tokens,
        valid_text_tokens, storage_text_tokens, layers, heads, head_dim,
        denoise_steps, text_cache_policy: text_cache_policy,
        segment_text_tokens: segment_text_tokens,
        segment_cache_text: segment_cache_text)
      @schedule = HyNomasContinuationHistorySchedule.new(segment_count, layers,
        denoise_steps)
      @query = query
      @key = key
      @value = value
      @output = output
      validate_current_bindings!
      @cache_arena = Arena.new

      begin
        allocate_layer_caches!
        @shape.workspace_geometries.each_with_index do |geometry, segment|
          @history_plans << prepare_attention_plan!(geometry)
          if writer_geometry = @shape.writer_workspace_geometries[segment]?
            @writer_plans << (writer_geometry == geometry ? @history_plans.last : prepare_attention_plan!(writer_geometry))
          end
        end
        @append_plan = HyNomasContinuationHistoryKernels::AppendPlan.new
        {% if flag?(:hy_nomas_history_probe) %}
          @fill_plan = HyNomasContinuationHistoryKernels::FillPlan.new
        {% end %}
        unless resident_bytes == @shape.named_bytes
          raise ArgumentError.new("Hy-Nomas continuation resident bytes differ from pure admission")
        end
      rescue ex
        release_prepared!
        @cache_arena.release
        @released = true
        raise ex
      end
    end

    # Exact named cache and MPS staging allocations owned by this provider.
    # MPSGraph's opaque internal temporaries are not included.
    def resident_bytes : Int64
      @cache_arena.resident_bytes + @history_workspaces.sum(&.resident_bytes)
    end

    # Select each forward. For every segment, call this once per read-only
    # denoise pass, then once with writer:true for non-final cache capture.
    def select_segment!(index : Int32, *, writer : Bool) : Nil
      raise ArgumentError.new("Hy-Nomas continuation history has been released") if @released
      raise ArgumentError.new("Hy-Nomas continuation history is unusable after an encode failure") if @poisoned
      @schedule.select_segment!(index, writer)
    end

    def checkpoint_inventory(next_segment : Int32) : Hash(String, Int64)
      bytes = @shape.boundary_cache_elements(next_segment).to_i64 * sizeof(Float32)
      inventory = {} of String => Int64
      @shape.layers.times do |layer|
        inventory["history.layer#{layer}.key"] = bytes
        inventory["history.layer#{layer}.value"] = bytes
      end
      inventory
    end

    # Setup-only bindings let callers prepare boundary kernels before encode.
    # Counts describe the future valid prefix, never the allocated tail.
    def prepare_checkpoint_bindings(next_segment : Int32,
                                    & : String, ML::MetalBuffer, Int32 ->) : Nil
      unless !@released && !@poisoned && @last_chain.nil?
        raise ArgumentError.new("Prepare continuation bindings before encoding")
      end
      @schedule.validate_restore_boundary!(next_segment)
      count = @shape.boundary_cache_elements(next_segment)
      @cache_pairs.each_with_index do |pair, layer|
        yield "history.layer#{layer}.key", pair[0], count
        yield "history.layer#{layer}.value", pair[1], count
      end
    end

    # Terminal file I/O only. Expose valid prefixes, never uninitialized cache
    # capacity. No duplicate GPU bank or CPU tensor array is needed.
    def each_checkpoint_cache(next_segment : Int32,
                              & : String, ML::MetalBuffer, Int32 ->) : Nil
      count = @shape.boundary_cache_elements(next_segment)
      chain = @last_chain
      unless !@released && !@poisoned && chain && chain.completed? &&
             @schedule.idle? && @schedule.next_segment == next_segment &&
             @schedule.denoise_passes == 0
        raise ArgumentError.new("Drain the post-writer boundary before exporting continuation state")
      end
      @cache_pairs.each_with_index do |pair, layer|
        yield "history.layer#{layer}.key", pair[0], count
        yield "history.layer#{layer}.value", pair[1], count
      end
    end

    # Setup file I/O streams directly into final shared Metal storage. An I/O
    # failure poisons the owner; a partially restored cache cannot be encoded.
    def restore_boundary!(next_segment : Int32,
                          & : String, ML::MetalBuffer, Int32 ->) : Nil
      count = @shape.boundary_cache_elements(next_segment)
      unless !@released && !@poisoned && @last_chain.nil?
        raise ArgumentError.new("Continuation state can only be restored into a fresh live provider")
      end
      @schedule.validate_restore_boundary!(next_segment)
      begin
        @cache_pairs.each_with_index do |pair, layer|
          yield "history.layer#{layer}.key", pair[0], count
          yield "history.layer#{layer}.value", pair[1], count
        end
        @schedule.restore_boundary!(next_segment)
      rescue ex
        @poisoned = true
        raise ex
      end
    end

    def validate_bindings!(tokens : Int32, heads : Int32, head_dim : Int32,
                           query : ML::MetalBuffer, key : ML::MetalBuffer,
                           value : ML::MetalBuffer, output : ML::MetalBuffer) : Nil
      raise ArgumentError.new("Hy-Nomas continuation history has been released") if @released
      raise ArgumentError.new("Hy-Nomas continuation history is unusable after an encode failure") if @poisoned
      unless tokens == @shape.query_rows && heads == @shape.heads &&
             head_dim == @shape.head_dim && @query.same?(query) && @key.same?(key) &&
             @value.same?(value) && @output.same?(output)
        raise ArgumentError.new("Hy-Nomas continuation active Q/K/V geometry or binding changed")
      end
      validate_current_bindings!
    end

    def validate_chain!(chain : ML::Metal::MPSCommandChain) : Nil
      raise ArgumentError.new("Hy-Nomas continuation history has been released") if @released
      raise ArgumentError.new("Hy-Nomas continuation history is unusable after an encode failure") if @poisoned
      chain.encoder
      if previous = @last_chain
        unless previous.same?(chain) || previous.completed?
          raise ArgumentError.new("Hy-Nomas continuation history cannot span in-flight command chains")
        end
      end
    end

    {% if flag?(:hy_nomas_history_probe) %}
      # Diagnostic-only cache initialization used by the provider qualification
      # probe. It queues fills on the caller's chain without CPU writes, buffer
      # allocations, readbacks, or waits.
      def encode_probe_fill_cache!(chain : ML::Metal::MPSCommandChain,
                                   value : Float32) : Nil
        raise ArgumentError.new("Hy-Nomas continuation history has been released") if @released
        raise ArgumentError.new("Hy-Nomas continuation history is unusable after an encode failure") if @poisoned
        unless !@probe_cache_filled && @last_chain.nil? && @schedule.idle? &&
               @schedule.next_segment == 0 && @schedule.denoise_passes == 0
          raise ArgumentError.new("Diagnostic cache fill must precede all continuation forwards")
        end
        validate_chain!(chain)
        @last_chain = chain
        begin
          encoder = chain.encoder
          @cache_pairs.each do |key, cache_value|
            fill_plan.encode(encoder, key, @shape.cache_elements_per_buffer, value)
            fill_plan.encode(encoder, cache_value, @shape.cache_elements_per_buffer, value)
          end
          @probe_cache_filled = true
        rescue ex
          @poisoned = true
          raise ex
        end
      end

      # Diagnostic-only cache capture for qualification. `:before_writer`
      # checks read-only denoise passes; `:after_writer` checks the append.
      # Caller owns exact packed F32 snapshot buffers allocated before encoding.
      def encode_probe_snapshot_cache!(chain : ML::Metal::MPSCommandChain,
                                       key_destination : ML::MetalBuffer,
                                       value_destination : ML::MetalBuffer,
                                       phase : Symbol) : Nil
        raise ArgumentError.new("Hy-Nomas continuation history has been released") if @released
        raise ArgumentError.new("Hy-Nomas continuation history is unusable after an encode failure") if @poisoned
        raise ArgumentError.new("Diagnostic cache snapshot requires prior sentinel fill") unless @probe_cache_filled
        unless @schedule.idle?
          raise ArgumentError.new("Diagnostic cache snapshot requires a completed forward")
        end
        case phase
        when :before_writer
          unless @schedule.awaiting_writer? && !@probe_pre_writer_snapshot
            raise ArgumentError.new("Pre-writer cache snapshot must follow all denoise forwards")
          end
        when :after_writer
          unless @probe_pre_writer_snapshot &&
                 @schedule.next_segment == 1 && @schedule.denoise_passes == 0 &&
                 !@probe_post_writer_snapshot
            raise ArgumentError.new("Post-writer cache snapshot must follow the first noisy-cache forward")
          end
        when :complete
          unless @schedule.complete? && @probe_post_writer_snapshot && !@probe_final_snapshot
            raise ArgumentError.new("Final cache snapshot must follow every continuation forward")
          end
        else
          raise ArgumentError.new("Unknown Hy-Nomas diagnostic cache snapshot phase")
        end

        validate_chain!(chain)
        validate_probe_snapshot_buffer!(key_destination)
        validate_probe_snapshot_buffer!(value_destination)
        unless !key_destination.same?(value_destination) &&
               !MetalOps.overlap?(key_destination, value_destination)
          raise ArgumentError.new("Hy-Nomas diagnostic K/V snapshots must not alias")
        end
        {@query, @key, @value, @output}.each do |current|
          if MetalOps.overlap?(current, key_destination) || MetalOps.overlap?(current, value_destination)
            raise ArgumentError.new("Hy-Nomas diagnostic snapshots must not alias current buffers")
          end
        end

        encoder = chain.encoder
        @cache_pairs.each_with_index do |pair, layer|
          offset = layer.to_i64 * @shape.cache_elements_per_buffer
          unless offset <= Int32::MAX
            raise ArgumentError.new("Hy-Nomas diagnostic snapshot offset exceeds the native limit")
          end
          append_plan.encode(encoder, pair[0], key_destination,
            @shape.cache_elements_per_buffer, offset.to_i32)
          append_plan.encode(encoder, pair[1], value_destination,
            @shape.cache_elements_per_buffer, offset.to_i32)
        end
        @probe_pre_writer_snapshot = true if phase == :before_writer
        @probe_post_writer_snapshot = true if phase == :after_writer
        @probe_final_snapshot = true if phase == :complete
      rescue ex
        @poisoned = true
        raise ex
      end
    {% end %}

    # Called by the serial owner after its Q/K/V projections and RoPE have
    # produced the active prefix. Attention consumes the prior cache first;
    # optional cache appends then follow it on the same current chain encoder.
    def encode(chain : ML::Metal::MPSCommandChain, block_index : Int32) : Nil
      raise ArgumentError.new("Hy-Nomas continuation history has been released") if @released
      segment = @schedule.selected_segment
      @schedule.validate_layer!(block_index)
      validate_chain!(chain)
      @last_chain = chain
      begin
        plan = @schedule.selected_writer? ? @writer_plans[segment] : @history_plans[segment]
        history_slot = if plan.workspace.shape.history_rows == 0
                         -1_i32
                       else
                         block_index
                       end
        plan.encode(chain, history_slot)
        if @schedule.selected_writer?
          cache_key, cache_value = @cache_pairs[block_index]
          destination_offset = @shape.append_offset(segment)
          count = @shape.append_elements(segment)
          encoder = chain.encoder
          append_plan.encode(encoder, @key, cache_key,
            count, destination_offset)
          append_plan.encode(encoder, @value, cache_value,
            count, destination_offset)
        end

        @schedule.finish_layer!(block_index)
      rescue ex
        @poisoned = true
        raise ex
      end
    end

    # Owner calls release only after the command chain's terminal wait. Plans
    # are released before their graph workspaces and those before cache memory.
    def release : Nil
      return if @released
      if chain = @last_chain
        raise ArgumentError.new("Complete Hy-Nomas continuation command chain before release") unless chain.completed?
      end
      release_prepared!
      @cache_arena.release
      @released = true
    end

    private def allocate_layer_caches! : Nil
      @shape.layers.times do |layer|
        key = @cache_arena.alloc("hy_nomas_history.layer#{layer}.key",
          @shape.cache_elements_per_buffer)
        value = @cache_arena.alloc("hy_nomas_history.layer#{layer}.value",
          @shape.cache_elements_per_buffer)
        validate_cache_pair!(key, value)
        @cache_pairs << {key, value}
      end
    end

    private def prepare_attention_plan!(geometry : HyNomasHistoryAttentionWorkspace::Geometry) : HyNomasHistoryAttentionMPSPlan
      workspace = HyNomasHistoryAttentionWorkspace.new(
        geometry.history_rows, @shape.query_rows, @shape.heads, @shape.head_dim,
        current_rows: geometry.current_rows)
      @history_workspaces << workspace
      if geometry.history_rows > 0
        @cache_pairs.each do |cache_key, cache_value|
          workspace.register_history_pair!(cache_key, cache_value)
        end
      end
      workspace.freeze_history_inventory!
      HyNomasHistoryAttentionMPSPlan.new(workspace, @query, @key, @value, @output)
    end

    private def validate_cache_pair!(key : ML::MetalBuffer,
                                     value : ML::MetalBuffer) : Nil
      expected_bytes = @shape.cache_elements_per_buffer.to_i64 * sizeof(Float32)
      unless key.size == expected_bytes && value.size == expected_bytes &&
             key.storage_mode.shared? && value.storage_mode.shared?
        raise ArgumentError.new("Hy-Nomas continuation cache requires exact-size shared F32 K/V buffers")
      end
      if MetalOps.overlap?(key, value)
        raise ArgumentError.new("Hy-Nomas continuation K/V cache buffers must not alias")
      end
      {key, value}.each do |cache|
        {@query, @key, @value, @output}.each do |current|
          if MetalOps.overlap?(cache, current)
            raise ArgumentError.new("Hy-Nomas continuation caches must not alias current Q/K/V/output")
          end
        end
        @cache_pairs.each do |previous_key, previous_value|
          {previous_key, previous_value}.each do |previous|
            if MetalOps.overlap?(cache, previous)
              raise ArgumentError.new("Hy-Nomas continuation cache inventory must not alias")
            end
          end
        end
      end
    end

    private def validate_current_bindings! : Nil
      expected_bytes = @shape.storage_elements.to_i64 * sizeof(Float32)
      buffers = {@query, @key, @value, @output}
      buffers.each do |buffer|
        unless buffer.valid? && buffer.size == expected_bytes && buffer.storage_mode.shared?
          raise ArgumentError.new("Hy-Nomas continuation current buffers require exact storage-row shared F32 geometry")
        end
      end
      buffers.each_with_index do |left, left_index|
        buffers.each_with_index do |right, right_index|
          next if right_index <= left_index
          if MetalOps.overlap?(left, right)
            raise ArgumentError.new("Hy-Nomas continuation current Q/K/V/output buffers must not alias")
          end
        end
      end
    end

    {% if flag?(:hy_nomas_history_probe) %}
      private def validate_probe_snapshot_buffer!(buffer : ML::MetalBuffer) : Nil
        bytes = @shape.layers.to_i64 * @shape.cache_elements_per_buffer * sizeof(Float32)
        unless buffer.valid? && buffer.size == bytes && buffer.storage_mode.shared?
          raise ArgumentError.new("Hy-Nomas diagnostic snapshot requires exact-size shared F32 storage")
        end
        @cache_pairs.each do |key, value|
          if MetalOps.overlap?(buffer, key) || MetalOps.overlap?(buffer, value)
            raise ArgumentError.new("Hy-Nomas diagnostic snapshot must not alias continuation caches")
          end
        end
      end
    {% end %}

    private def release_prepared! : Nil
      @writer_plans.each(&.release)
      @writer_plans.clear
      @history_plans.each(&.release)
      @history_plans.clear
      @history_workspaces.each(&.release)
      @history_workspaces.clear
      @cache_pairs.clear
      @append_plan = nil
      {% if flag?(:hy_nomas_history_probe) %}
        @fill_plan = nil
      {% end %}
    end

    private def append_plan : HyNomasContinuationHistoryKernels::AppendPlan
      @append_plan || raise ArgumentError.new("Hy-Nomas continuation append plan is unavailable")
    end

    {% if flag?(:hy_nomas_history_probe) %}
      private def fill_plan : HyNomasContinuationHistoryKernels::FillPlan
        @fill_plan || raise ArgumentError.new("Hy-Nomas continuation fill plan is unavailable")
      end
    {% end %}

    private def self.checked_product(left : Int64, right : Int64,
                                     label : String) : Int64
      unless left >= 0 && right >= 0 && (right == 0 || left <= Int64::MAX // right)
        raise ArgumentError.new("#{label} overflow")
      end
      left * right
    end

    private def self.checked_sum(left : Int64, right : Int64,
                                 label : String) : Int64
      unless left >= 0 && right >= 0 && left <= Int64::MAX - right
        raise ArgumentError.new("#{label} overflow")
      end
      left + right
    end
  end
end
