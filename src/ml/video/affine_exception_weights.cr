require "../metal/dispatch"
require "./metal_ops"

module ML::Video::AffineExceptionWeights
  SOURCE = {{ read_file("#{__DIR__}/kernels/affine_exception_weights.metal") }}

  STATUS_WORDS                 =    8
  STATUS_ERROR_WORDS           =    4
  STATUS_PACK_ERROR            =    0
  STATUS_DECODE_ERROR          =    1
  STATUS_NONFINITE             =    2
  STATUS_LAYOUT_ERROR          =    3
  STATUS_DECODE_SCRATCH        =    4
  MODE_SPARSE_WEIGHT           =    0
  MODE_SPARSE_ACTIVATION       =    1
  MODE_K_CHANNEL               =    2
  MODE_N_ROW                   =    3
  MODE_JOINT_QUOTA_WEIGHT      =    4
  MODE_JOINT_GLOBAL_WEIGHT     =    5
  MODE_JOINT_GLOBAL_ACTIVATION =    6
  RADIX_CHUNK                  = 4096
  RECORD_VALUES                = 1024
  GROUP_VALUES                 =  256
  GROUP_BYTES                  =  136
  RECORD_BASE_BYTES            =  544
  Q4_K_BITS_PER_WEIGHT         =  144

  # Per-run geometry and exact byte budget. The payload size intentionally
  # matches Q4_K's complete storage price; no host readback determines a mode.
  class Geometry
    getter n : Int32
    getter k : Int32
    getter weights : Int64
    getter payload_bytes : Int64
    getter records_1024 : Int32
    getter groups_256 : Int32
    getter patch_count : Int32
    getter k_channel_count : Int32
    getter n_row_count : Int32
    getter joint_upgrade_count : Int32
    getter fit_bytes : Int64
    getter workspace_bytes : Int64

    getter base_bytes : Int64
    getter sparse_counts_offset : Int64
    getter sparse_offsets_offset : Int64
    getter sparse_patches_offset : Int64
    getter k_map_offset : Int64
    getter k_ids_offset : Int64
    getter k_sidecar_offset : Int64
    getter k_padding_offset : Int64
    getter n_directory_bytes : Int64
    getter n_row_base_bytes : Int64
    getter n_padding_bytes : Int64
    getter candidate_capacity : Int32
    getter radix_groups : Int32
    getter candidate_groups : Int32

    def status_words : Int32
      STATUS_WORDS
    end

    # Word 4 is transient validation state; these four words remain sticky.
    def status_error_words : Array(Int32)
      [STATUS_PACK_ERROR, STATUS_DECODE_ERROR, STATUS_NONFINITE, STATUS_LAYOUT_ERROR]
    end

    def initialize(@n : Int32, @k : Int32)
      unless @n > 0 && @k >= GROUP_VALUES && @k % GROUP_VALUES == 0
        raise ArgumentError.new("Affine exception geometry requires N>0 and K divisible by 256")
      end
      @weights = @n.to_i64 * @k
      unless @weights > 0 && @weights <= Int32::MAX && @weights % RECORD_VALUES == 0
        raise ArgumentError.new("Affine exception geometry requires an Int32 weight count divisible by 1024")
      end
      @payload_bytes = @weights * Q4_K_BITS_PER_WEIGHT // 256
      raise ArgumentError.new("Affine exception payload exceeds supported dispatch/offset range") if @payload_bytes > Int32::MAX
      raise ArgumentError.new("Affine exception K-channel ids exceed UInt16 range") if @k > UInt16::MAX.to_i32 + 1
      @records_1024 = (@weights // RECORD_VALUES).to_i32
      @groups_256 = (@weights // GROUP_VALUES).to_i32
      @base_bytes = @groups_256.to_i64 * GROUP_BYTES
      @fit_bytes = @groups_256.to_i64 * 48

      sparse_header = @base_bytes + @records_1024.to_i64 * 4 + (@records_1024.to_i64 + 1) * 4
      sparse_remaining = @payload_bytes - sparse_header
      raise ArgumentError.new("Q4_K budget cannot hold affine records and sparse directories") if sparse_remaining < 0
      @patch_count = (sparse_remaining // 4).to_i32
      @joint_upgrade_count = @patch_count - @groups_256
      raise ArgumentError.new("Q4_K budget cannot hold one joint exception per group") if @joint_upgrade_count < 0
      @sparse_counts_offset = @base_bytes
      @sparse_offsets_offset = @sparse_counts_offset + @records_1024.to_i64 * 4
      @sparse_patches_offset = @sparse_offsets_offset + (@records_1024.to_i64 + 1) * 4
      raise ArgumentError.new("Sparse patch count exceeds position/index range") if @patch_count > Int32::MAX

      @k_map_offset = @base_bytes
      @k_ids_offset = @k_map_offset + @k.to_i64 * 2
      @k_channel_count = 0
      while @k_channel_count < @k && k_layout_end(@k_channel_count + 1) <= @payload_bytes
        @k_channel_count += 1
      end
      raise ArgumentError.new("Q4_K budget cannot hold affine K-channel map") if @k_channel_count == 0
      @k_sidecar_offset = align4(@k_ids_offset + align4(@k_channel_count.to_i64 * 2))
      @k_padding_offset = @k_sidecar_offset + @n.to_i64 * @k_channel_count * 2

      @n_directory_bytes = (@n.to_i64 + 1) * 4
      @n_row_base_bytes = @k.to_i64 // GROUP_VALUES * GROUP_BYTES
      delta = @k.to_i64 * 2 - @n_row_base_bytes
      raise ArgumentError.new("Raw BF16 N-row must cost more than its affine base") unless delta > 0
      @n_row_count = 0
      while @n_row_count < @n && n_layout_end(@n_row_count + 1, delta) <= @payload_bytes
        @n_row_count += 1
      end
      raise ArgumentError.new("Q4_K budget cannot hold N-row directory and affine base") if @n_row_count == 0
      @n_padding_bytes = @payload_bytes - n_layout_end(@n_row_count, delta)

      @candidate_capacity = Math.max(Math.max(@n, @k), @groups_256)
      @radix_groups = ((@weights + RADIX_CHUNK - 1) // RADIX_CHUNK).to_i32
      @candidate_groups = ((@candidate_capacity.to_i64 + GROUP_VALUES - 1) // GROUP_VALUES).to_i32
      @workspace_bytes = calculate_workspace_bytes
      unless @radix_groups > 0 && @candidate_groups > 0 && @workspace_bytes > 0
        raise ArgumentError.new("Affine exception workspace geometry overflow")
      end
    end

    def compatible?(n : Int32, k : Int32) : Bool
      @n == n && @k == k
    end

    def target_count(mode : Int32) : Int32
      case mode
      when MODE_SPARSE_WEIGHT, MODE_SPARSE_ACTIVATION then @patch_count
      when MODE_K_CHANNEL                             then @k_channel_count
      when MODE_N_ROW                                 then @n_row_count
      when MODE_JOINT_QUOTA_WEIGHT, MODE_JOINT_GLOBAL_WEIGHT,
           MODE_JOINT_GLOBAL_ACTIVATION then @patch_count
      else raise ArgumentError.new("Unknown affine exception mode")
      end
    end

    private def k_layout_end(count : Int32) : Int64
      @k_ids_offset + align4(count.to_i64 * 2) + @n.to_i64 * count * 2
    end

    private def n_layout_end(count : Int32, delta : Int64) : Int64
      @n_directory_bytes + @base_bytes + count.to_i64 * delta
    end

    private def align4(value : Int64) : Int64
      (value + 3_i64) & ~3_i64
    end

    private def calculate_workspace_bytes : Int64
      max_candidates = @candidate_capacity.to_i64
      max_groups = @candidate_groups.to_i64
      @base_bytes + @fit_bytes + @weights * 4 + max_candidates * 4 +
        @radix_groups.to_i64 * 256 * 4 + 256 * 4 + 16 +
        max_candidates * 4 + max_candidates * 4 + max_groups * 8
    end
  end

  # Scratch is setup-owned and reusable. Do not share one workspace between
  # overlapping encoders; the caller releases it after pack/qualification.
  class Workspace
    getter geometry : Geometry
    getter affine : ML::MetalBuffer
    getter joint_fit : ML::MetalBuffer
    getter scores : ML::MetalBuffer
    getter candidate_scores : ML::MetalBuffer
    getter histograms : ML::MetalBuffer
    getter digit_histogram : ML::MetalBuffer
    getter rank_state : ML::MetalBuffer
    getter candidate_flags : ML::MetalBuffer
    getter candidate_prefix : ML::MetalBuffer
    getter candidate_group_sums : ML::MetalBuffer
    getter candidate_group_offsets : ML::MetalBuffer
    getter buffers : Array(ML::MetalBuffer)
    @released = false

    def initialize(@geometry : Geometry)
      candidates = @geometry.candidate_capacity.to_i64
      groups = @geometry.radix_groups.to_i64
      candidate_groups = @geometry.candidate_groups.to_i64
      created = [] of ML::MetalBuffer
      begin
        sizes = [@geometry.base_bytes, @geometry.fit_bytes, @geometry.weights * 4, candidates * 4,
                 groups * 256 * 4, 256_i64 * 4, 16_i64, candidates * 4,
                 candidates * 4, candidate_groups * 4, candidate_groups * 4]
        sizes.each { |bytes| created << ML::MetalBuffer.new(bytes) }
      rescue ex
        created.each(&.release)
        raise ex
      end
      @affine = created[0]
      @joint_fit = created[1]
      @scores = created[2]
      @candidate_scores = created[3]
      @histograms = created[4]
      @digit_histogram = created[5]
      @rank_state = created[6]
      @candidate_flags = created[7]
      @candidate_prefix = created[8]
      @candidate_group_sums = created[9]
      @candidate_group_offsets = created[10]
      @buffers = created
    end

    def resident_bytes : Int64
      @geometry.workspace_bytes
    end

    def valid_for?(n : Int32, k : Int32) : Bool
      !@released && @geometry.compatible?(n, k) && @buffers.all?(&.valid?)
    end

    def release : Nil
      return if @released
      @buffers.each(&.release)
      @released = true
    end
  end

  @@base_score : ML::Metal::ComputePipeline?
  @@joint_fit : ML::Metal::ComputePipeline?
  @@joint_materialize : ML::Metal::ComputePipeline?
  @@joint_sparse_counts : ML::Metal::ComputePipeline?
  @@joint_sparse_pack : ML::Metal::ComputePipeline?
  @@candidate_score : ML::Metal::ComputePipeline?
  @@rank_init : ML::Metal::ComputePipeline?
  @@rank_hist : ML::Metal::ComputePipeline?
  @@rank_reduce : ML::Metal::ComputePipeline?
  @@rank_digit : ML::Metal::ComputePipeline?
  @@candidate_select : ML::Metal::ComputePipeline?
  @@candidate_scan : ML::Metal::ComputePipeline?
  @@candidate_scan_groups : ML::Metal::ComputePipeline?
  @@candidate_scan_finish : ML::Metal::ComputePipeline?
  @@zero_payload : ML::Metal::ComputePipeline?
  @@copy_base : ML::Metal::ComputePipeline?
  @@sparse_counts : ML::Metal::ComputePipeline?
  @@sparse_offsets : ML::Metal::ComputePipeline?
  @@sparse_pack : ML::Metal::ComputePipeline?
  @@k_layout : ML::Metal::ComputePipeline?
  @@k_sidecar : ML::Metal::ComputePipeline?
  @@n_directory : ML::Metal::ComputePipeline?
  @@n_rows : ML::Metal::ComputePipeline?
  @@validate_init : ML::Metal::ComputePipeline?
  @@validate_sparse : ML::Metal::ComputePipeline?
  @@validate_k : ML::Metal::ComputePipeline?
  @@validate_n : ML::Metal::ComputePipeline?
  @@decode_sparse_base : ML::Metal::ComputePipeline?
  @@decode_sparse_patch : ML::Metal::ComputePipeline?
  @@decode_k : ML::Metal::ComputePipeline?
  @@decode_n : ML::Metal::ComputePipeline?

  def self.prepare! : Nil
    @@base_score ||= pipeline("affine_exception_base_score")
    @@joint_fit ||= pipeline("affine_exception_joint_fit")
    @@joint_materialize ||= pipeline("affine_exception_joint_materialize")
    @@joint_sparse_counts ||= pipeline("affine_exception_joint_sparse_counts")
    @@joint_sparse_pack ||= pipeline("affine_exception_joint_sparse_pack")
    @@candidate_score ||= pipeline("affine_exception_candidate_score")
    @@rank_init ||= pipeline("affine_exception_rank_init")
    @@rank_hist ||= pipeline("affine_exception_rank_hist")
    @@rank_reduce ||= pipeline("affine_exception_rank_reduce")
    @@rank_digit ||= pipeline("affine_exception_rank_digit")
    @@candidate_select ||= pipeline("affine_exception_candidate_select")
    @@candidate_scan ||= pipeline("affine_exception_candidate_scan")
    @@candidate_scan_groups ||= pipeline("affine_exception_candidate_scan_groups")
    @@candidate_scan_finish ||= pipeline("affine_exception_candidate_scan_finish")
    @@zero_payload ||= pipeline("affine_exception_zero_payload")
    @@copy_base ||= pipeline("affine_exception_copy_base")
    @@sparse_counts ||= pipeline("affine_exception_sparse_counts")
    @@sparse_offsets ||= pipeline("affine_exception_sparse_offsets")
    @@sparse_pack ||= pipeline("affine_exception_sparse_pack")
    @@k_layout ||= pipeline("affine_exception_k_layout")
    @@k_sidecar ||= pipeline("affine_exception_k_sidecar")
    @@n_directory ||= pipeline("affine_exception_n_directory")
    @@n_rows ||= pipeline("affine_exception_n_rows")
    @@validate_init ||= pipeline("affine_exception_validate_init")
    @@validate_sparse ||= pipeline("affine_exception_validate_sparse")
    @@validate_k ||= pipeline("affine_exception_validate_k")
    @@validate_n ||= pipeline("affine_exception_validate_n")
    @@decode_sparse_base ||= pipeline("affine_exception_decode_sparse_base")
    @@decode_sparse_patch ||= pipeline("affine_exception_decode_sparse_patch")
    @@decode_k ||= pipeline("affine_exception_decode_k")
    @@decode_n ||= pipeline("affine_exception_decode_n")
  end

  # Append-only encoder API. The caller initializes status once; this method
  # never clears it, waits, allocates Metal storage, or reads buffer contents.
  def self.pack(e : ML::Metal::ComputeEncoder,
                original_bf16 : ML::MetalBuffer,
                calibration_f32 : ML::MetalBuffer,
                payload : ML::MetalBuffer,
                workspace : Workspace,
                status : ML::MetalBuffer,
                n : Int32, k : Int32,
                mode : Int32, fold : Int32 = 0) : Nil
    validate_pack!(original_bf16, calibration_f32, payload, workspace, status, n, k, mode, fold)
    if joint_mode?(mode)
      pack_joint(e, original_bf16, calibration_f32, payload, workspace, status, n, k, mode, fold)
      return
    end
    base_score = @@base_score || raise ArgumentError.new("Call AffineExceptionWeights.prepare! before encode")
    groups = workspace.geometry.groups_256
    e.set_pipeline(base_score).set_buffer(original_bf16, 0).set_buffer(calibration_f32, 1)
      .set_buffer(workspace.affine, 2, ML::Metal::BufferAccess::Write)
      .set_buffer(workspace.scores, 3, ML::Metal::BufferAccess::Write)
      .set_buffer(status, 4, ML::Metal::BufferAccess::ReadWrite)
      .set_value(n.to_u32, 5).set_value(k.to_u32, 6).set_value(mode.to_u32, 7)
      .set_value(fold.to_u32, 8).dispatch_threadgroups({groups, 1, 1}, {GROUP_VALUES, 1, 1})

    rank_source = workspace.scores
    rank_count = workspace.geometry.weights.to_i32
    if mode == MODE_K_CHANNEL || mode == MODE_N_ROW
      candidate = @@candidate_score.not_nil!
      rank_count = mode == MODE_K_CHANNEL ? k : n
      e.set_pipeline(candidate).set_buffer(workspace.scores, 0)
        .set_buffer(workspace.candidate_scores, 1, ML::Metal::BufferAccess::Write)
        .set_buffer(status, 2, ML::Metal::BufferAccess::ReadWrite)
        .set_value(n.to_u32, 3).set_value(k.to_u32, 4).set_value(mode.to_u32, 5)
        .dispatch_1d(rank_count)
      rank_source = workspace.candidate_scores
    end

    count = workspace.geometry.target_count(mode)
    e.set_pipeline(@@rank_init.not_nil!).set_buffer(workspace.rank_state, 0, ML::Metal::BufferAccess::Write)
      .set_value(count.to_u32, 1).dispatch_1d(1)
    radix_groups = ((rank_count.to_i64 + RADIX_CHUNK - 1) // RADIX_CHUNK).to_i32
    {56, 48, 40, 32, 24, 16, 8, 0}.each do |shift|
      e.set_pipeline(@@rank_hist.not_nil!).set_buffer(rank_source, 0)
        .set_buffer(workspace.histograms, 1, ML::Metal::BufferAccess::Write)
        .set_buffer(workspace.rank_state, 2).set_value(rank_count.to_u32, 3)
        .set_value(shift.to_u32, 4).dispatch_threadgroups({radix_groups, 1, 1}, {GROUP_VALUES, 1, 1})
      e.set_pipeline(@@rank_reduce.not_nil!).set_buffer(workspace.histograms, 0)
        .set_buffer(workspace.digit_histogram, 1, ML::Metal::BufferAccess::Write)
        .set_value(radix_groups.to_u32, 2).dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
      e.set_pipeline(@@rank_digit.not_nil!).set_buffer(workspace.digit_histogram, 0)
        .set_buffer(workspace.rank_state, 1, ML::Metal::BufferAccess::ReadWrite)
        .set_value(shift.to_u32, 2).dispatch_1d(1)
    end

    e.set_pipeline(@@zero_payload.not_nil!).set_buffer(payload, 0, ML::Metal::BufferAccess::Write)
      .set_value(workspace.geometry.payload_bytes.to_u32, 1)
      .dispatch_1d(workspace.geometry.payload_bytes.to_i32)

    if mode == MODE_SPARSE_WEIGHT || mode == MODE_SPARSE_ACTIVATION
      encode_sparse(e, original_bf16, payload, workspace, status, n, k)
    elsif mode == MODE_K_CHANNEL
      encode_candidates(e, original_bf16, payload, workspace, status, n, k, mode, rank_count)
    else
      encode_candidates(e, original_bf16, payload, workspace, status, n, k, mode, rank_count)
    end
  end

  # Fail-closed GPU decode. Validation runs before any variable section is
  # consumed; a malformed payload poisons the complete output matrix with NaN.
  def self.decode(e : ML::Metal::ComputeEncoder,
                  payload : ML::MetalBuffer, stage : ML::MetalBuffer,
                  status : ML::MetalBuffer, n : Int32, k : Int32,
                  mode : Int32) : Nil
    validate_decode!(payload, stage, status, n, k, mode)
    geometry = Geometry.new(n, k)
    e.set_pipeline(@@validate_init.not_nil!).set_buffer(status, 0, ML::Metal::BufferAccess::ReadWrite).dispatch_1d(1)
    case mode
    when MODE_SPARSE_WEIGHT, MODE_SPARSE_ACTIVATION,
         MODE_JOINT_QUOTA_WEIGHT, MODE_JOINT_GLOBAL_WEIGHT, MODE_JOINT_GLOBAL_ACTIVATION
      e.set_pipeline(@@validate_sparse.not_nil!).set_buffer(payload, 0).set_buffer(status, 1, ML::Metal::BufferAccess::ReadWrite)
        .set_value(geometry.records_1024.to_u32, 2).set_value(geometry.patch_count.to_u32, 3)
        .dispatch_threadgroups({geometry.records_1024, 1, 1}, {GROUP_VALUES, 1, 1})
      e.set_pipeline(@@decode_sparse_base.not_nil!).set_buffer(payload, 0).set_buffer(stage, 1, ML::Metal::BufferAccess::Write)
        .set_buffer(status, 2).set_value(geometry.groups_256.to_u32, 3).set_value(n.to_u32, 4).set_value(k.to_u32, 5)
        .dispatch_threadgroups({geometry.groups_256, 1, 1}, {GROUP_VALUES, 1, 1})
      e.set_pipeline(@@decode_sparse_patch.not_nil!).set_buffer(payload, 0).set_buffer(stage, 1, ML::Metal::BufferAccess::ReadWrite)
        .set_buffer(status, 2).set_value(geometry.records_1024.to_u32, 3)
        .dispatch_threadgroups({geometry.records_1024, 1, 1}, {GROUP_VALUES, 1, 1})
    when MODE_K_CHANNEL
      e.set_pipeline(@@validate_k.not_nil!).set_buffer(payload, 0).set_buffer(status, 1, ML::Metal::BufferAccess::ReadWrite)
        .set_value(n.to_u32, 2).set_value(k.to_u32, 3).set_value(geometry.k_channel_count.to_u32, 4)
        .set_value(geometry.base_bytes.to_u32, 5)
        .dispatch_1d(Math.max(Math.max(geometry.weights.to_i32, n * geometry.k_channel_count), geometry.groups_256))
      e.set_pipeline(@@decode_k.not_nil!).set_buffer(payload, 0).set_buffer(stage, 1, ML::Metal::BufferAccess::Write)
        .set_buffer(status, 2).set_value(n.to_u32, 3).set_value(k.to_u32, 4)
        .set_value(geometry.k_channel_count.to_u32, 5).set_value(geometry.base_bytes.to_u32, 6)
        .set_value(geometry.k_map_offset.to_u32, 7).set_value(geometry.k_ids_offset.to_u32, 8)
        .set_value(geometry.k_sidecar_offset.to_u32, 9).dispatch_1d(geometry.weights.to_i32)
    when MODE_N_ROW
      e.set_pipeline(@@validate_n.not_nil!).set_buffer(payload, 0).set_buffer(status, 1, ML::Metal::BufferAccess::ReadWrite)
        .set_value(n.to_u32, 2).set_value(k.to_u32, 3).set_value(geometry.n_row_base_bytes.to_u32, 4)
        .set_value(geometry.n_directory_bytes.to_u32, 5)
        .set_value((geometry.payload_bytes - geometry.n_padding_bytes).to_u32, 6)
        .dispatch_threadgroups({n, 1, 1}, {GROUP_VALUES, 1, 1})
      e.set_pipeline(@@decode_n.not_nil!).set_buffer(payload, 0).set_buffer(stage, 1, ML::Metal::BufferAccess::Write)
        .set_buffer(status, 2).set_value(n.to_u32, 3).set_value(k.to_u32, 4)
        .set_value(geometry.n_row_base_bytes.to_u32, 5).set_value(geometry.n_directory_bytes.to_u32, 6)
        .dispatch_threadgroups({n, 1, 1}, {GROUP_VALUES, 1, 1})
    end
  end

  def self.validate_decode!(payload : ML::MetalBuffer, stage : ML::MetalBuffer,
                            status : ML::MetalBuffer, n : Int32, k : Int32,
                            mode : Int32) : Nil
    geometry = Geometry.new(n, k)
    validate_mode!(mode)
    bytes!(payload, geometry.payload_bytes)
    bytes!(stage, geometry.weights * 4)
    bytes!(status, STATUS_WORDS.to_i64 * 4)
    MetalOps.distinct(stage, payload, status)
    MetalOps.distinct(status, payload)
    raise ArgumentError.new("Call AffineExceptionWeights.prepare! before decode") unless @@validate_init
  end

  private def self.encode_sparse(e : ML::Metal::ComputeEncoder, original : ML::MetalBuffer,
                                 payload : ML::MetalBuffer, workspace : Workspace,
                                 status : ML::MetalBuffer, n : Int32, k : Int32) : Nil
    g = workspace.geometry
    e.set_pipeline(@@copy_base.not_nil!).set_buffer(workspace.affine, 0)
      .set_buffer(payload, 1, ML::Metal::BufferAccess::Write).set_value(g.base_bytes.to_u32, 2)
      .dispatch_1d(g.base_bytes.to_i32)
    e.set_pipeline(@@sparse_counts.not_nil!).set_buffer(workspace.scores, 0)
      .set_buffer(workspace.rank_state, 1).set_buffer(payload, 2, ML::Metal::BufferAccess::ReadWrite)
      .set_buffer(status, 3, ML::Metal::BufferAccess::ReadWrite).set_value(g.records_1024.to_u32, 4)
      .set_value(g.patch_count.to_u32, 5).set_value(g.sparse_counts_offset.to_u32, 6)
      .dispatch_threadgroups({g.records_1024, 1, 1}, {GROUP_VALUES, 1, 1})
    e.set_pipeline(@@sparse_offsets.not_nil!).set_buffer(payload, 0, ML::Metal::BufferAccess::ReadWrite)
      .set_buffer(status, 1, ML::Metal::BufferAccess::ReadWrite).set_value(g.records_1024.to_u32, 2)
      .set_value(g.patch_count.to_u32, 3).set_value(g.sparse_offsets_offset.to_u32, 4).dispatch_1d(1)
    e.set_pipeline(@@sparse_pack.not_nil!).set_buffer(original, 0).set_buffer(workspace.scores, 1)
      .set_buffer(workspace.rank_state, 2).set_buffer(payload, 3, ML::Metal::BufferAccess::ReadWrite)
      .set_buffer(status, 4, ML::Metal::BufferAccess::ReadWrite).set_value(g.records_1024.to_u32, 5)
      .set_value(g.patch_count.to_u32, 6).set_value(g.sparse_offsets_offset.to_u32, 7)
      .set_value(g.sparse_patches_offset.to_u32, 8).dispatch_threadgroups({g.records_1024, 1, 1}, {GROUP_VALUES, 1, 1})
  end

  private def self.pack_joint(e : ML::Metal::ComputeEncoder, original : ML::MetalBuffer,
                              calibration : ML::MetalBuffer, payload : ML::MetalBuffer,
                              workspace : Workspace, status : ML::MetalBuffer,
                              n : Int32, k : Int32, mode : Int32, fold : Int32) : Nil
    g = workspace.geometry
    e.set_pipeline(@@joint_fit || raise ArgumentError.new("Call AffineExceptionWeights.prepare! before encode"))
      .set_buffer(original, 0).set_buffer(calibration, 1)
      .set_buffer(workspace.joint_fit, 2, ML::Metal::BufferAccess::Write)
      .set_buffer(workspace.candidate_scores, 3, ML::Metal::BufferAccess::Write)
      .set_buffer(status, 4, ML::Metal::BufferAccess::ReadWrite)
      .set_value(g.groups_256.to_u32, 5).set_value(k.to_u32, 6)
      .set_value(mode.to_u32, 7).set_value(fold.to_u32, 8)
      .dispatch_threadgroups({g.groups_256, 1, 1}, {GROUP_VALUES, 1, 1})

    if mode == MODE_JOINT_QUOTA_WEIGHT
      rank_scores(e, workspace.candidate_scores, g.groups_256, g.joint_upgrade_count, workspace)
      select_candidates(e, workspace.candidate_scores, g.groups_256, g.joint_upgrade_count, workspace, status)
      e.set_pipeline(@@joint_materialize.not_nil!).set_buffer(original, 0)
        .set_buffer(calibration, 1).set_buffer(workspace.joint_fit, 2, ML::Metal::BufferAccess::ReadWrite)
        .set_buffer(workspace.affine, 3, ML::Metal::BufferAccess::Write)
        .set_buffer(workspace.scores, 4, ML::Metal::BufferAccess::Write)
        .set_buffer(workspace.candidate_flags, 5).set_buffer(status, 6, ML::Metal::BufferAccess::ReadWrite)
        .set_value(g.groups_256.to_u32, 7).set_value(k.to_u32, 8).set_value(mode.to_u32, 9).set_value(fold.to_u32, 10)
        .dispatch_threadgroups({g.groups_256, 1, 1}, {GROUP_VALUES, 1, 1})
      zero_payload(e, payload, g)
      encode_joint_sparse(e, original, payload, workspace, status)
    else
      e.set_pipeline(@@joint_materialize.not_nil!).set_buffer(original, 0).set_buffer(calibration, 1)
        .set_buffer(workspace.joint_fit, 2, ML::Metal::BufferAccess::ReadWrite)
        .set_buffer(workspace.affine, 3, ML::Metal::BufferAccess::Write)
        .set_buffer(workspace.scores, 4, ML::Metal::BufferAccess::Write)
        .set_buffer(workspace.candidate_flags, 5).set_buffer(status, 6, ML::Metal::BufferAccess::ReadWrite)
        .set_value(g.groups_256.to_u32, 7).set_value(k.to_u32, 8)
        .set_value(mode.to_u32, 9).set_value(fold.to_u32, 10)
        .dispatch_threadgroups({g.groups_256, 1, 1}, {GROUP_VALUES, 1, 1})
      rank_scores(e, workspace.scores, g.weights.to_i32, g.patch_count, workspace)
      zero_payload(e, payload, g)
      encode_sparse(e, original, payload, workspace, status, n, k)
    end
  end

  private def self.rank_scores(e : ML::Metal::ComputeEncoder, source : ML::MetalBuffer,
                               count : Int32, target : Int32, workspace : Workspace) : Nil
    e.set_pipeline(@@rank_init.not_nil!).set_buffer(workspace.rank_state, 0, ML::Metal::BufferAccess::Write)
      .set_value(target.to_u32, 1).dispatch_1d(1)
    radix_groups = ((count.to_i64 + RADIX_CHUNK - 1) // RADIX_CHUNK).to_i32
    {56, 48, 40, 32, 24, 16, 8, 0}.each do |shift|
      e.set_pipeline(@@rank_hist.not_nil!).set_buffer(source, 0)
        .set_buffer(workspace.histograms, 1, ML::Metal::BufferAccess::Write)
        .set_buffer(workspace.rank_state, 2).set_value(count.to_u32, 3)
        .set_value(shift.to_u32, 4).dispatch_threadgroups({radix_groups, 1, 1}, {GROUP_VALUES, 1, 1})
      e.set_pipeline(@@rank_reduce.not_nil!).set_buffer(workspace.histograms, 0)
        .set_buffer(workspace.digit_histogram, 1, ML::Metal::BufferAccess::Write)
        .set_value(radix_groups.to_u32, 2).dispatch_threadgroups({1, 1, 1}, {256, 1, 1})
      e.set_pipeline(@@rank_digit.not_nil!).set_buffer(workspace.digit_histogram, 0)
        .set_buffer(workspace.rank_state, 1, ML::Metal::BufferAccess::ReadWrite)
        .set_value(shift.to_u32, 2).dispatch_1d(1)
    end
  end

  private def self.select_candidates(e : ML::Metal::ComputeEncoder, source : ML::MetalBuffer,
                                     count : Int32, target : Int32, workspace : Workspace,
                                     status : ML::MetalBuffer) : Nil
    e.set_pipeline(@@candidate_select.not_nil!).set_buffer(source, 0)
      .set_buffer(workspace.rank_state, 1).set_buffer(workspace.candidate_flags, 2, ML::Metal::BufferAccess::Write)
      .set_value(count.to_u32, 3).dispatch_1d(count)
    groups = (count + GROUP_VALUES - 1) // GROUP_VALUES
    e.set_pipeline(@@candidate_scan.not_nil!).set_buffer(workspace.candidate_flags, 0)
      .set_buffer(workspace.candidate_prefix, 1, ML::Metal::BufferAccess::Write)
      .set_buffer(workspace.candidate_group_sums, 2, ML::Metal::BufferAccess::Write)
      .set_value(count.to_u32, 3).dispatch_threadgroups({groups, 1, 1}, {GROUP_VALUES, 1, 1})
    e.set_pipeline(@@candidate_scan_groups.not_nil!).set_buffer(workspace.candidate_group_sums, 0)
      .set_buffer(workspace.candidate_group_offsets, 1, ML::Metal::BufferAccess::Write)
      .set_buffer(status, 3, ML::Metal::BufferAccess::ReadWrite)
      .set_value(groups.to_u32, 2).set_value(target.to_u32, 4).dispatch_1d(1)
    e.set_pipeline(@@candidate_scan_finish.not_nil!).set_buffer(workspace.candidate_prefix, 0, ML::Metal::BufferAccess::ReadWrite)
      .set_buffer(workspace.candidate_group_offsets, 1).set_value(count.to_u32, 2)
      .dispatch_1d(count)
  end

  private def self.zero_payload(e : ML::Metal::ComputeEncoder, payload : ML::MetalBuffer,
                                geometry : Geometry) : Nil
    e.set_pipeline(@@zero_payload.not_nil!).set_buffer(payload, 0, ML::Metal::BufferAccess::Write)
      .set_value(geometry.payload_bytes.to_u32, 1).dispatch_1d(geometry.payload_bytes.to_i32)
  end

  private def self.encode_joint_sparse(e : ML::Metal::ComputeEncoder, original : ML::MetalBuffer,
                                       payload : ML::MetalBuffer, workspace : Workspace,
                                       status : ML::MetalBuffer) : Nil
    g = workspace.geometry
    e.set_pipeline(@@copy_base.not_nil!).set_buffer(workspace.affine, 0)
      .set_buffer(payload, 1, ML::Metal::BufferAccess::Write).set_value(g.base_bytes.to_u32, 2)
      .dispatch_1d(g.base_bytes.to_i32)
    e.set_pipeline(@@joint_sparse_counts.not_nil!).set_buffer(workspace.joint_fit, 0)
      .set_buffer(payload, 1, ML::Metal::BufferAccess::ReadWrite)
      .set_buffer(status, 2, ML::Metal::BufferAccess::ReadWrite)
      .set_value(g.records_1024.to_u32, 3).set_value(g.sparse_counts_offset.to_u32, 4)
      .dispatch_threadgroups({g.records_1024, 1, 1}, {GROUP_VALUES, 1, 1})
    e.set_pipeline(@@sparse_offsets.not_nil!).set_buffer(payload, 0, ML::Metal::BufferAccess::ReadWrite)
      .set_buffer(status, 1, ML::Metal::BufferAccess::ReadWrite).set_value(g.records_1024.to_u32, 2)
      .set_value(g.patch_count.to_u32, 3).set_value(g.sparse_offsets_offset.to_u32, 4).dispatch_1d(1)
    e.set_pipeline(@@joint_sparse_pack.not_nil!).set_buffer(original, 0).set_buffer(workspace.joint_fit, 1)
      .set_buffer(payload, 2, ML::Metal::BufferAccess::ReadWrite).set_buffer(status, 3, ML::Metal::BufferAccess::ReadWrite)
      .set_value(g.groups_256.to_u32, 4).set_value(g.records_1024.to_u32, 5)
      .set_value(g.patch_count.to_u32, 6).set_value(g.sparse_offsets_offset.to_u32, 7)
      .set_value(g.sparse_patches_offset.to_u32, 8)
      .dispatch_threadgroups({g.groups_256, 1, 1}, {GROUP_VALUES, 1, 1})
  end

  private def self.encode_candidates(e : ML::Metal::ComputeEncoder, original : ML::MetalBuffer,
                                     payload : ML::MetalBuffer, workspace : Workspace,
                                     status : ML::MetalBuffer, n : Int32, k : Int32,
                                     mode : Int32, count : Int32) : Nil
    g = workspace.geometry
    e.set_pipeline(@@candidate_select.not_nil!).set_buffer(workspace.candidate_scores, 0)
      .set_buffer(workspace.rank_state, 1).set_buffer(workspace.candidate_flags, 2, ML::Metal::BufferAccess::Write)
      .set_value(count.to_u32, 3).dispatch_1d(count)
    groups = (count + GROUP_VALUES - 1) // GROUP_VALUES
    e.set_pipeline(@@candidate_scan.not_nil!).set_buffer(workspace.candidate_flags, 0)
      .set_buffer(workspace.candidate_prefix, 1, ML::Metal::BufferAccess::Write)
      .set_buffer(workspace.candidate_group_sums, 2, ML::Metal::BufferAccess::Write)
      .set_value(count.to_u32, 3).dispatch_threadgroups({groups, 1, 1}, {GROUP_VALUES, 1, 1})
    e.set_pipeline(@@candidate_scan_groups.not_nil!).set_buffer(workspace.candidate_group_sums, 0)
      .set_buffer(workspace.candidate_group_offsets, 1, ML::Metal::BufferAccess::Write)
      .set_buffer(status, 3, ML::Metal::BufferAccess::ReadWrite)
      .set_value(groups.to_u32, 2).set_value(g.target_count(mode).to_u32, 4).dispatch_1d(1)
    e.set_pipeline(@@candidate_scan_finish.not_nil!).set_buffer(workspace.candidate_prefix, 0, ML::Metal::BufferAccess::ReadWrite)
      .set_buffer(workspace.candidate_group_offsets, 1).set_value(count.to_u32, 2)
      .dispatch_1d(count)

    if mode == MODE_K_CHANNEL
      e.set_pipeline(@@copy_base.not_nil!).set_buffer(workspace.affine, 0)
        .set_buffer(payload, 1, ML::Metal::BufferAccess::Write).set_value(g.base_bytes.to_u32, 2)
        .dispatch_1d(g.base_bytes.to_i32)
      e.set_pipeline(@@k_layout.not_nil!).set_buffer(workspace.candidate_flags, 0)
        .set_buffer(workspace.candidate_prefix, 1).set_buffer(payload, 2, ML::Metal::BufferAccess::ReadWrite)
        .set_buffer(status, 3, ML::Metal::BufferAccess::ReadWrite).set_value(k.to_u32, 4)
        .set_value(g.k_channel_count.to_u32, 5).set_value(g.k_map_offset.to_u32, 6)
        .set_value(g.k_ids_offset.to_u32, 7).set_value(g.k_sidecar_offset.to_u32, 8).dispatch_1d(k)
      e.set_pipeline(@@k_sidecar.not_nil!).set_buffer(original, 0).set_buffer(payload, 1, ML::Metal::BufferAccess::ReadWrite)
        .set_buffer(workspace.candidate_flags, 2).set_buffer(workspace.candidate_prefix, 3)
        .set_value(n.to_u32, 4).set_value(k.to_u32, 5).set_value(g.k_channel_count.to_u32, 6)
        .set_value(g.k_map_offset.to_u32, 7).set_value(g.k_sidecar_offset.to_u32, 8)
        .dispatch_1d(g.weights.to_i32)
    else
      e.set_pipeline(@@n_directory.not_nil!).set_buffer(workspace.candidate_flags, 0)
        .set_buffer(workspace.candidate_prefix, 1).set_buffer(payload, 2, ML::Metal::BufferAccess::Write)
        .set_buffer(status, 3, ML::Metal::BufferAccess::ReadWrite).set_value(n.to_u32, 4)
        .set_value(g.n_row_count.to_u32, 5).set_value(g.n_directory_bytes.to_u32, 6)
        .set_value(g.n_row_base_bytes.to_u32, 7).set_value(k.to_u32, 8)
        .dispatch_1d(n + 1)
      e.set_pipeline(@@n_rows.not_nil!).set_buffer(original, 0).set_buffer(workspace.affine, 1)
        .set_buffer(workspace.candidate_flags, 2).set_buffer(payload, 3, ML::Metal::BufferAccess::ReadWrite)
        .set_buffer(status, 4, ML::Metal::BufferAccess::ReadWrite).set_value(n.to_u32, 5)
        .set_value(k.to_u32, 6).set_value(g.n_directory_bytes.to_u32, 7)
        .set_value(g.n_row_base_bytes.to_u32, 8).dispatch_threadgroups({n, 1, 1}, {GROUP_VALUES, 1, 1})
    end
  end

  private def self.validate_pack!(original : ML::MetalBuffer, calibration : ML::MetalBuffer,
                                  payload : ML::MetalBuffer, workspace : Workspace,
                                  status : ML::MetalBuffer, n : Int32, k : Int32,
                                  mode : Int32, fold : Int32) : Nil
    validate_mode!(mode)
    unless fold == 0 || fold == 1
      raise ArgumentError.new("Calibration fold must be 0 or 1")
    end
    unless workspace.valid_for?(n, k)
      raise ArgumentError.new("Affine exception workspace is released or has mismatched geometry")
    end
    g = workspace.geometry
    bytes!(original, g.weights * 2)
    bytes!(calibration, k.to_i64 * 4 * 4)
    bytes!(payload, g.payload_bytes)
    bytes!(status, STATUS_WORDS.to_i64 * 4)
    MetalOps.distinct(payload, original, calibration, status)
    MetalOps.distinct(status, original, calibration)
    workspace.buffers.each_with_index do |buffer, i|
      MetalOps.distinct(buffer, original, calibration, payload, status)
      workspace.buffers.each_with_index do |other, j|
        MetalOps.distinct(buffer, other) if j > i
      end
    end
    raise ArgumentError.new("Call AffineExceptionWeights.prepare! before encode") unless @@base_score
  end

  private def self.validate_mode!(mode : Int32) : Nil
    unless {MODE_SPARSE_WEIGHT, MODE_SPARSE_ACTIVATION, MODE_K_CHANNEL, MODE_N_ROW,
            MODE_JOINT_QUOTA_WEIGHT, MODE_JOINT_GLOBAL_WEIGHT, MODE_JOINT_GLOBAL_ACTIVATION}.includes?(mode)
      raise ArgumentError.new("Unknown affine exception mode")
    end
  end

  private def self.joint_mode?(mode : Int32) : Bool
    mode == MODE_JOINT_QUOTA_WEIGHT || mode == MODE_JOINT_GLOBAL_WEIGHT || mode == MODE_JOINT_GLOBAL_ACTIVATION
  end

  private def self.bytes!(buffer : ML::MetalBuffer, minimum : Int64) : Nil
    unless buffer.valid? && minimum > 0 && buffer.size >= minimum
      raise ArgumentError.new("Invalid or undersized affine exception buffer")
    end
  end

  private def self.pipeline(name : String) : ML::Metal::ComputePipeline
    ML::Metal::PipelineCache.get(name) { ML::Metal::ComputePipeline.new(name, SOURCE) }
  end
end
