require "../metal/dispatch"
require "./metal_ops"

module ML::Video::Q4KDeltaLinear
  enum GemmVariant
    Legacy
    B128N32S4K32
    B64N32S4K32BT
    B64N32S4K64BT
    B64N32S4DeviceF32
    B64N32S4DeviceQ4K
  end

  BLOCK_ELEMENTS               =   256
  Q4_K_BLOCK_BYTES             =   144
  STATUS_WORDS                 =     1
  STATUS_ERROR                 = 1_u32
  MAX_STATIC_THREADGROUP_BYTES = 32 * 1024
  LEGACY_TILE_M                =  64
  LEGACY_TILE_N                =  32
  LEGACY_TILE_K                =  32
  CANDIDATE_TILE_M             = 128
  CANDIDATE_TILE_N             =  32
  CANDIDATE_TILE_K             =  32
  BT_CANDIDATE_TILE_M          =  64
  BT_CANDIDATE_TILE_N          =  32
  BT_CANDIDATE_TILE_K          =  32
  BT_K64_CANDIDATE_TILE_M      =  64
  BT_K64_CANDIDATE_TILE_N      =  32
  BT_K64_CANDIDATE_TILE_K      =  64
  DEVICE_F32_CANDIDATE_TILE_M  =  64
  DEVICE_F32_CANDIDATE_TILE_N  =  32
  DEVICE_F32_CANDIDATE_TILE_K  =  32
  DEVICE_Q4K_CANDIDATE_TILE_M  =  64
  DEVICE_Q4K_CANDIDATE_TILE_N  =  32
  DEVICE_Q4K_CANDIDATE_TILE_K  =  32
  GEMM_THREADS                 = 128

  LEGACY_LINEAR_KERNEL           = "video_q4k_delta_linear_f32"
  LEGACY_DENSE_KERNEL            = "video_q4k_delta_dense_linear_f32"
  CANDIDATE_LINEAR_KERNEL        = "video_q4k_delta_linear_f32_candidate_b128n32_sg4_tk32"
  CANDIDATE_DENSE_KERNEL         = "video_q4k_delta_dense_linear_f32_candidate_b128n32_sg4_tk32"
  BT_CANDIDATE_LINEAR_KERNEL     = "video_q4k_delta_linear_f32_candidate_b64n32_sg4_tk32_bt"
  BT_CANDIDATE_DENSE_KERNEL      = "video_q4k_delta_dense_linear_f32_candidate_b64n32_sg4_tk32_bt"
  BT_K64_CANDIDATE_LINEAR_KERNEL = "video_q4k_delta_linear_f32_candidate_b64n32_sg4_tk64_bt"
  BT_K64_CANDIDATE_DENSE_KERNEL  = "video_q4k_delta_dense_linear_f32_candidate_b64n32_sg4_tk64_bt"
  DEVICE_F32_DENSE_KERNEL        = "video_q4k_delta_dense_linear_f32_candidate_b64n32_sg4_device_f32"
  DEVICE_F32_TAIL_KERNEL         = "video_q4k_delta_dense_linear_f32_candidate_b64n32_sg4_device_f32_tail"
  DEVICE_Q4K_LINEAR_KERNEL       = "video_q4k_delta_linear_f32_candidate_b64n32_sg4_device_q4k"
  DEVICE_Q4K_TAIL_KERNEL         = "video_q4k_delta_linear_f32_candidate_b64n32_sg4_device_q4k_tail"
  DEVICE_Q4K_COORDINATE_PROBE    = "video_q4k_delta_coordinate_mapping_probe"
  DEVICE_Q4K_FRAGMENT_PROBE      = "video_q4k_delta_q4_fragment_probe"

  SOURCE = {{ read_file("#{__DIR__}/kernels/q4k_delta_linear.metal") }}

  @@linear_pipeline : ML::Metal::ComputePipeline?
  @@dense_pipeline : ML::Metal::ComputePipeline?
  @@candidate_linear_pipeline : ML::Metal::ComputePipeline?
  @@candidate_dense_pipeline : ML::Metal::ComputePipeline?
  @@bt_candidate_linear_pipeline : ML::Metal::ComputePipeline?
  @@bt_candidate_dense_pipeline : ML::Metal::ComputePipeline?
  @@bt_k64_candidate_linear_pipeline : ML::Metal::ComputePipeline?
  @@bt_k64_candidate_dense_pipeline : ML::Metal::ComputePipeline?
  @@device_f32_candidate_dense_pipeline : ML::Metal::ComputePipeline?
  @@device_f32_candidate_tail_pipeline : ML::Metal::ComputePipeline?
  @@device_q4k_candidate_linear_pipeline : ML::Metal::ComputePipeline?
  @@device_q4k_candidate_tail_pipeline : ML::Metal::ComputePipeline?
  @@coordinate_mapping_probe_pipeline : ML::Metal::ComputePipeline?
  @@q4k_fragment_probe_pipeline : ML::Metal::ComputePipeline?
  @@decode_pipeline : ML::Metal::ComputePipeline?

  # Compile and validate reusable kernels during setup. Encode methods only
  # validate caller-owned storage, bind it, and append work to the open encoder.
  def self.prepare! : Nil
    @@linear_pipeline ||= ML::Metal::PipelineCache.get(LEGACY_LINEAR_KERNEL) do
      ML::Metal::ComputePipeline.new(LEGACY_LINEAR_KERNEL, SOURCE)
    end
    @@dense_pipeline ||= ML::Metal::PipelineCache.get(LEGACY_DENSE_KERNEL) do
      ML::Metal::ComputePipeline.new(LEGACY_DENSE_KERNEL, SOURCE)
    end
    @@candidate_linear_pipeline ||= ML::Metal::PipelineCache.get(CANDIDATE_LINEAR_KERNEL) do
      ML::Metal::ComputePipeline.new(CANDIDATE_LINEAR_KERNEL, SOURCE)
    end
    @@candidate_dense_pipeline ||= ML::Metal::PipelineCache.get(CANDIDATE_DENSE_KERNEL) do
      ML::Metal::ComputePipeline.new(CANDIDATE_DENSE_KERNEL, SOURCE)
    end
    @@bt_candidate_linear_pipeline ||= ML::Metal::PipelineCache.get(BT_CANDIDATE_LINEAR_KERNEL) do
      ML::Metal::ComputePipeline.new(BT_CANDIDATE_LINEAR_KERNEL, SOURCE)
    end
    @@bt_candidate_dense_pipeline ||= ML::Metal::PipelineCache.get(BT_CANDIDATE_DENSE_KERNEL) do
      ML::Metal::ComputePipeline.new(BT_CANDIDATE_DENSE_KERNEL, SOURCE)
    end
    @@bt_k64_candidate_linear_pipeline ||= ML::Metal::PipelineCache.get(BT_K64_CANDIDATE_LINEAR_KERNEL) do
      ML::Metal::ComputePipeline.new(BT_K64_CANDIDATE_LINEAR_KERNEL, SOURCE)
    end
    @@bt_k64_candidate_dense_pipeline ||= ML::Metal::PipelineCache.get(BT_K64_CANDIDATE_DENSE_KERNEL) do
      ML::Metal::ComputePipeline.new(BT_K64_CANDIDATE_DENSE_KERNEL, SOURCE)
    end
    @@device_f32_candidate_dense_pipeline ||= ML::Metal::PipelineCache.get(DEVICE_F32_DENSE_KERNEL) do
      ML::Metal::ComputePipeline.new(DEVICE_F32_DENSE_KERNEL, SOURCE)
    end
    @@device_f32_candidate_tail_pipeline ||= ML::Metal::PipelineCache.get(DEVICE_F32_TAIL_KERNEL) do
      ML::Metal::ComputePipeline.new(DEVICE_F32_TAIL_KERNEL, SOURCE)
    end
    @@device_q4k_candidate_linear_pipeline ||= ML::Metal::PipelineCache.get(DEVICE_Q4K_LINEAR_KERNEL) do
      ML::Metal::ComputePipeline.new(DEVICE_Q4K_LINEAR_KERNEL, SOURCE)
    end
    @@device_q4k_candidate_tail_pipeline ||= ML::Metal::PipelineCache.get(DEVICE_Q4K_TAIL_KERNEL) do
      ML::Metal::ComputePipeline.new(DEVICE_Q4K_TAIL_KERNEL, SOURCE)
    end
    @@coordinate_mapping_probe_pipeline ||= ML::Metal::PipelineCache.get(DEVICE_Q4K_COORDINATE_PROBE) do
      ML::Metal::ComputePipeline.new(DEVICE_Q4K_COORDINATE_PROBE, SOURCE)
    end
    @@q4k_fragment_probe_pipeline ||= ML::Metal::PipelineCache.get(DEVICE_Q4K_FRAGMENT_PROBE) do
      ML::Metal::ComputePipeline.new(DEVICE_Q4K_FRAGMENT_PROBE, SOURCE)
    end
    @@decode_pipeline ||= ML::Metal::PipelineCache.get("video_q4k_delta_decode_probe") do
      ML::Metal::ComputePipeline.new("video_q4k_delta_decode_probe", SOURCE)
    end
    validate_gemm_pipeline!(@@linear_pipeline.not_nil!, "fused Q4_K")
    validate_gemm_pipeline!(@@dense_pipeline.not_nil!, "dense control")
    validate_gemm_pipeline!(@@candidate_linear_pipeline.not_nil!, "candidate fused Q4_K")
    validate_gemm_pipeline!(@@candidate_dense_pipeline.not_nil!, "candidate dense control")
    validate_gemm_pipeline!(@@bt_candidate_linear_pipeline.not_nil!, "transposed-B candidate fused Q4_K")
    validate_gemm_pipeline!(@@bt_candidate_dense_pipeline.not_nil!, "transposed-B candidate dense control")
    validate_gemm_pipeline!(@@bt_k64_candidate_linear_pipeline.not_nil!, "TK64 transposed-B candidate fused Q4_K")
    validate_gemm_pipeline!(@@bt_k64_candidate_dense_pipeline.not_nil!, "TK64 transposed-B candidate dense control")
    validate_gemm_pipeline!(@@device_f32_candidate_dense_pipeline.not_nil!, "device-load candidate dense core")
    validate_gemm_pipeline!(@@device_f32_candidate_tail_pipeline.not_nil!, "device-load candidate staged tail")
    validate_gemm_pipeline!(@@device_q4k_candidate_linear_pipeline.not_nil!, "device-load Q4_K candidate core")
    validate_gemm_pipeline!(@@device_q4k_candidate_tail_pipeline.not_nil!, "device-load Q4_K candidate staged tail")
  end

  # Expose setup-time pipeline metadata so harnesses can record the actual
  # compiled threadgroup footprint rather than a source-level estimate.
  def self.linear_pipeline(variant : GemmVariant = GemmVariant::Legacy) : ML::Metal::ComputePipeline
    selected = case variant
               when GemmVariant::Legacy
                 @@linear_pipeline
               when GemmVariant::B128N32S4K32
                 @@candidate_linear_pipeline
               when GemmVariant::B64N32S4K32BT
                 @@bt_candidate_linear_pipeline
               when GemmVariant::B64N32S4K64BT
                 @@bt_k64_candidate_linear_pipeline
               when GemmVariant::B64N32S4DeviceF32
                 raise ArgumentError.new("The device-load GEMM variant is dense-only")
               when GemmVariant::B64N32S4DeviceQ4K
                 @@device_q4k_candidate_linear_pipeline
               else
                 raise ArgumentError.new("Unsupported GEMM variant")
               end
    selected || raise ArgumentError.new("Call Q4KDeltaLinear.prepare! before querying the pipeline")
  end

  def self.dense_pipeline(variant : GemmVariant = GemmVariant::Legacy) : ML::Metal::ComputePipeline
    selected = case variant
               when GemmVariant::Legacy
                 @@dense_pipeline
               when GemmVariant::B128N32S4K32
                 @@candidate_dense_pipeline
               when GemmVariant::B64N32S4K32BT
                 @@bt_candidate_dense_pipeline
               when GemmVariant::B64N32S4K64BT
                 @@bt_k64_candidate_dense_pipeline
               when GemmVariant::B64N32S4DeviceF32
                 @@device_f32_candidate_dense_pipeline
               when GemmVariant::B64N32S4DeviceQ4K
                 @@device_f32_candidate_dense_pipeline
               else
                 raise ArgumentError.new("Unsupported GEMM variant")
               end
    selected || raise ArgumentError.new("Call Q4KDeltaLinear.prepare! before querying the pipeline")
  end

  # The direct-load candidate is dense-only. Its safe fringe kernel stages the
  # same 64x32x32 tile used by the transposed-B control, with caller-supplied
  # group origins so right and bottom fringes remain disjoint.
  def self.dense_staged_tail_pipeline(variant : GemmVariant) : ML::Metal::ComputePipeline
    unless variant == GemmVariant::B64N32S4DeviceF32 || variant == GemmVariant::B64N32S4DeviceQ4K
      raise ArgumentError.new("Only the device-load GEMM variant has a staged tail pipeline")
    end
    @@device_f32_candidate_tail_pipeline || raise ArgumentError.new("Call Q4KDeltaLinear.prepare! before querying the pipeline")
  end

  def self.linear_staged_tail_pipeline(variant : GemmVariant) : ML::Metal::ComputePipeline
    unless variant == GemmVariant::B64N32S4DeviceQ4K
      raise ArgumentError.new("Only the device-load Q4_K variant has a staged tail pipeline")
    end
    @@device_q4k_candidate_tail_pipeline || raise ArgumentError.new("Call Q4KDeltaLinear.prepare! before querying the pipeline")
  end

  def self.variant_name(variant : GemmVariant = GemmVariant::Legacy) : String
    case variant
    when GemmVariant::Legacy
      "legacy"
    when GemmVariant::B128N32S4K32
      "candidate-b128n32-sg4-tk32"
    when GemmVariant::B64N32S4K32BT
      "candidate-b64n32-sg4-tk32-bt"
    when GemmVariant::B64N32S4K64BT
      "candidate-b64n32-sg4-tk64-bt"
    when GemmVariant::B64N32S4DeviceF32
      "candidate-b64n32-sg4-device-f32"
    when GemmVariant::B64N32S4DeviceQ4K
      "candidate-b64n32-sg4-device-q4k"
    else
      raise ArgumentError.new("Unsupported GEMM variant")
    end
  end

  def self.tile_m(variant : GemmVariant = GemmVariant::Legacy) : Int32
    case variant
    when GemmVariant::Legacy
      LEGACY_TILE_M
    when GemmVariant::B128N32S4K32
      CANDIDATE_TILE_M
    when GemmVariant::B64N32S4K32BT
      BT_CANDIDATE_TILE_M
    when GemmVariant::B64N32S4K64BT
      BT_K64_CANDIDATE_TILE_M
    when GemmVariant::B64N32S4DeviceF32
      DEVICE_F32_CANDIDATE_TILE_M
    when GemmVariant::B64N32S4DeviceQ4K
      DEVICE_Q4K_CANDIDATE_TILE_M
    else
      raise ArgumentError.new("Unsupported GEMM variant")
    end
  end

  def self.tile_n(variant : GemmVariant = GemmVariant::Legacy) : Int32
    case variant
    when GemmVariant::Legacy
      LEGACY_TILE_N
    when GemmVariant::B128N32S4K32
      CANDIDATE_TILE_N
    when GemmVariant::B64N32S4K32BT
      BT_CANDIDATE_TILE_N
    when GemmVariant::B64N32S4K64BT
      BT_K64_CANDIDATE_TILE_N
    when GemmVariant::B64N32S4DeviceF32
      DEVICE_F32_CANDIDATE_TILE_N
    when GemmVariant::B64N32S4DeviceQ4K
      DEVICE_Q4K_CANDIDATE_TILE_N
    else
      raise ArgumentError.new("Unsupported GEMM variant")
    end
  end

  def self.tile_k(variant : GemmVariant = GemmVariant::Legacy) : Int32
    case variant
    when GemmVariant::Legacy
      LEGACY_TILE_K
    when GemmVariant::B128N32S4K32
      CANDIDATE_TILE_K
    when GemmVariant::B64N32S4K32BT
      BT_CANDIDATE_TILE_K
    when GemmVariant::B64N32S4K64BT
      BT_K64_CANDIDATE_TILE_K
    when GemmVariant::B64N32S4DeviceF32
      DEVICE_F32_CANDIDATE_TILE_K
    when GemmVariant::B64N32S4DeviceQ4K
      DEVICE_Q4K_CANDIDATE_TILE_K
    else
      raise ArgumentError.new("Unsupported GEMM variant")
    end
  end

  def self.threadgroup_threads(variant : GemmVariant = GemmVariant::Legacy) : Int32
    GEMM_THREADS
  end

  def self.static_threadgroup_memory_bytes(variant : GemmVariant = GemmVariant::Legacy) : Int64
    linear_pipeline(variant).static_threadgroup_memory_length
  end

  def self.dense_static_threadgroup_memory_bytes(variant : GemmVariant = GemmVariant::Legacy) : Int64
    dense_pipeline(variant).static_threadgroup_memory_length
  end

  def self.dense_staged_tail_static_threadgroup_memory_bytes(variant : GemmVariant) : Int64
    dense_staged_tail_pipeline(variant).static_threadgroup_memory_length
  end

  def self.linear_staged_tail_static_threadgroup_memory_bytes(variant : GemmVariant) : Int64
    linear_staged_tail_pipeline(variant).static_threadgroup_memory_length
  end

  # Describe the direct full-tile core and the two disjoint staged fringes.
  # The direct kernel only receives complete 64x32 tiles; the right fringe
  # covers core rows, and the bottom fringe covers every column for tail rows.
  def self.dense_dispatch_partition(rows : Int32, in_dim : Int32, out_dim : Int32,
                                    variant : GemmVariant = GemmVariant::B64N32S4DeviceF32)
    unless variant == GemmVariant::B64N32S4DeviceF32 || variant == GemmVariant::B64N32S4DeviceQ4K
      raise ArgumentError.new("Dispatch partition is only defined for device-load GEMM variants")
    end
    checked_geometry(rows, in_dim, out_dim)
    grid_x, grid_y = checked_dispatch_grid!(rows, out_dim, variant)
    core_x = out_dim // tile_n(variant)
    core_y = rows // tile_m(variant)
    right_tail_groups = (grid_x - core_x) * core_y
    bottom_tail_groups = grid_x * (grid_y - core_y)
    direct_dispatches = core_x > 0 && core_y > 0 ? 1 : 0
    staged_tail_dispatches = (right_tail_groups > 0 ? 1 : 0) +
                             (bottom_tail_groups > 0 ? 1 : 0)
    {
      direct_rows:            core_y * tile_m(variant),
      direct_columns:         core_x * tile_n(variant),
      direct_groups_x:        core_x,
      direct_groups_y:        core_y,
      right_tail_groups:      right_tail_groups,
      bottom_tail_groups:     bottom_tail_groups,
      direct_dispatches:      direct_dispatches,
      staged_tail_dispatches: staged_tail_dispatches,
    }
  end

  # Packed direct tiles use the same 64x32 geometry as the dense DeviceF32
  # control, but dispatch their own Q4_K core and staged fringe pipeline.
  def self.linear_dispatch_partition(rows : Int32, in_dim : Int32, out_dim : Int32,
                                     variant : GemmVariant)
    unless variant == GemmVariant::B64N32S4DeviceQ4K
      raise ArgumentError.new("Dispatch partition is only defined for the device-load Q4_K variant")
    end
    checked_geometry(rows, in_dim, out_dim)
    grid_x, grid_y = checked_dispatch_grid!(rows, out_dim, variant)
    core_x = out_dim // tile_n(variant)
    core_y = rows // tile_m(variant)
    right_tail_groups = (grid_x - core_x) * core_y
    bottom_tail_groups = grid_x * (grid_y - core_y)
    direct_dispatches = core_x > 0 && core_y > 0 ? 1 : 0
    staged_tail_dispatches = (right_tail_groups > 0 ? 1 : 0) +
                             (bottom_tail_groups > 0 ? 1 : 0)
    {
      direct_rows:            core_y * tile_m(variant),
      direct_columns:         core_x * tile_n(variant),
      direct_groups_x:        core_x,
      direct_groups_y:        core_y,
      right_tail_groups:      right_tail_groups,
      bottom_tail_groups:     bottom_tail_groups,
      direct_dispatches:      direct_dispatches,
      staged_tail_dispatches: staged_tail_dispatches,
    }
  end

  # Validate the positive-M/N, 256-multiple-K geometry and every element count
  # used by bindings and dispatches. The tuple is useful for setup probes.
  def self.checked_geometry(rows : Int32, in_dim : Int32,
                            out_dim : Int32) : {Int32, Int32, Int32}
    ML::Video::MetalOps.dims(rows, in_dim, out_dim)
    unless in_dim % BLOCK_ELEMENTS == 0
      raise ArgumentError.new("Q4_K fused linear input dimension must be a multiple of 256")
    end

    input_count = checked_product!(rows.to_i64, in_dim.to_i64, "input elements")
    weight_count = checked_product!(out_dim.to_i64, in_dim.to_i64, "weight elements")
    output_count = checked_product!(rows.to_i64, out_dim.to_i64, "output elements")
    {input_count, weight_count, output_count}.each do |count|
      unless count > 0 && count <= Int32::MAX
        raise ArgumentError.new("Q4_K fused linear element count exceeds Int32")
      end
    end
    {rows, in_dim, out_dim}
  end

  # The zero-patch mode still receives a one-byte caller-owned placeholder;
  # its contents are never read by the GPU.
  def self.sidecar_bytes(count : Int32, patches : Int32) : Int64
    validate_count!(count)
    validate_patches!(patches)
    return 1_i64 if patches == 0
    checked_product!(count.to_i64 // BLOCK_ELEMENTS,
      patches.to_i64 * 3_i64, "sidecar bytes")
  end

  def self.linear(e : ML::Metal::ComputeEncoder,
                  input : ML::MetalBuffer, payload : ML::MetalBuffer,
                  sidecar : ML::MetalBuffer, bias : ML::MetalBuffer,
                  output : ML::MetalBuffer, status : ML::MetalBuffer,
                  rows : Int32, in_dim : Int32, out_dim : Int32,
                  patches : Int32,
                  variant : GemmVariant = GemmVariant::Legacy) : Nil
    checked_geometry(rows, in_dim, out_dim)
    validate_patches!(patches)
    input_count = rows.to_i64 * in_dim
    weight_count = out_dim.to_i64 * in_dim
    output_count = rows.to_i64 * out_dim
    payload_bytes = payload_bytes!(weight_count)
    deltas_bytes = sidecar_bytes(weight_count.to_i32, patches)
    grid_x, grid_y = checked_dispatch_grid!(rows, out_dim, variant)

    ML::Video::MetalOps.check(input, input_count)
    check_bytes!(payload, payload_bytes, "Q4_K payload")
    check_bytes!(sidecar, deltas_bytes, "Q4_K delta sidecar")
    ML::Video::MetalOps.check(bias, out_dim.to_i64)
    ML::Video::MetalOps.check(output, output_count)
    check_status!(status)
    assert_all_disjoint!(input, payload, sidecar, bias, output, status)

    if variant == GemmVariant::B64N32S4DeviceF32
      raise ArgumentError.new("The device-load F32 GEMM variant is dense-only")
    end

    if variant == GemmVariant::B64N32S4DeviceQ4K
      partition = linear_dispatch_partition(rows, in_dim, out_dim, variant)
      grid_x, grid_y = checked_dispatch_grid!(rows, out_dim, variant)
      selected = linear_pipeline(variant)
      tail_pipeline = nil.as(ML::Metal::ComputePipeline?)
      if partition[:staged_tail_dispatches] > 0
        tail_pipeline = linear_staged_tail_pipeline(variant)
      end

      if partition[:direct_dispatches] > 0
        e.set_pipeline(selected)
        e.set_buffer(input, 0).set_buffer(payload, 1).set_buffer(sidecar, 2)
        e.set_buffer(bias, 3).set_buffer(output, 4, ML::Metal::BufferAccess::Write)
        e.set_buffer(status, 5, ML::Metal::BufferAccess::ReadWrite)
        e.set_value(rows.to_u32, 6).set_value(in_dim.to_u32, 7)
        e.set_value(out_dim.to_u32, 8).set_value(patches.to_u32, 9)
        e.dispatch_threadgroups(
          {partition[:direct_groups_x], partition[:direct_groups_y], 1},
          {threadgroup_threads(variant), 1, 1})
      end

      if partition[:right_tail_groups] > 0
        e.set_pipeline(tail_pipeline.not_nil!)
        e.set_buffer(input, 0).set_buffer(payload, 1).set_buffer(sidecar, 2)
        e.set_buffer(bias, 3).set_buffer(output, 4, ML::Metal::BufferAccess::Write)
        e.set_buffer(status, 5, ML::Metal::BufferAccess::ReadWrite)
        e.set_value(rows.to_u32, 6).set_value(in_dim.to_u32, 7)
        e.set_value(out_dim.to_u32, 8).set_value(patches.to_u32, 9)
        e.set_value(partition[:direct_groups_x].to_u32, 10).set_value(0_u32, 11)
        e.dispatch_threadgroups(
          {grid_x - partition[:direct_groups_x], partition[:direct_groups_y], 1},
          {threadgroup_threads(variant), 1, 1})
      end

      bottom_tail_groups = grid_y - partition[:direct_groups_y]
      if bottom_tail_groups > 0
        e.set_pipeline(tail_pipeline.not_nil!)
        e.set_buffer(input, 0).set_buffer(payload, 1).set_buffer(sidecar, 2)
        e.set_buffer(bias, 3).set_buffer(output, 4, ML::Metal::BufferAccess::Write)
        e.set_buffer(status, 5, ML::Metal::BufferAccess::ReadWrite)
        e.set_value(rows.to_u32, 6).set_value(in_dim.to_u32, 7)
        e.set_value(out_dim.to_u32, 8).set_value(patches.to_u32, 9)
        e.set_value(0_u32, 10).set_value(partition[:direct_groups_y].to_u32, 11)
        e.dispatch_threadgroups(
          {grid_x, bottom_tail_groups, 1},
          {threadgroup_threads(variant), 1, 1})
      end
      return
    end

    selected = linear_pipeline(variant)
    e.set_pipeline(selected)
    e.set_buffer(input, 0).set_buffer(payload, 1).set_buffer(sidecar, 2)
    e.set_buffer(bias, 3).set_buffer(output, 4, ML::Metal::BufferAccess::Write)
    e.set_buffer(status, 5, ML::Metal::BufferAccess::ReadWrite)
    e.set_value(rows.to_u32, 6).set_value(in_dim.to_u32, 7)
    e.set_value(out_dim.to_u32, 8).set_value(patches.to_u32, 9)
    e.dispatch_threadgroups(
      {grid_x, grid_y, 1}, {threadgroup_threads(variant), 1, 1})
  end

  # Same shared-panel F32 GEMM and bias schedule as the fused path, reading a
  # caller-provided row-major [N,K] matrix directly without an MPS stage.
  def self.dense_linear(e : ML::Metal::ComputeEncoder,
                        input : ML::MetalBuffer, weight : ML::MetalBuffer,
                        bias : ML::MetalBuffer, output : ML::MetalBuffer,
                        rows : Int32, in_dim : Int32, out_dim : Int32,
                        variant : GemmVariant = GemmVariant::Legacy) : Nil
    checked_geometry(rows, in_dim, out_dim)
    input_count = rows.to_i64 * in_dim
    weight_count = out_dim.to_i64 * in_dim
    output_count = rows.to_i64 * out_dim
    grid_x, grid_y = checked_dispatch_grid!(rows, out_dim, variant)
    ML::Video::MetalOps.check(input, input_count)
    ML::Video::MetalOps.check(weight, weight_count)
    ML::Video::MetalOps.check(bias, out_dim.to_i64)
    ML::Video::MetalOps.check(output, output_count)
    assert_all_disjoint!(input, weight, bias, output)

    selected = dense_pipeline(variant)
    if variant == GemmVariant::B64N32S4DeviceF32 || variant == GemmVariant::B64N32S4DeviceQ4K
      dense_variant = GemmVariant::B64N32S4DeviceF32
      partition = dense_dispatch_partition(rows, in_dim, out_dim, dense_variant)
      tail_pipeline = nil.as(ML::Metal::ComputePipeline?)
      if partition[:staged_tail_dispatches] > 0
        tail_pipeline = dense_staged_tail_pipeline(dense_variant)
      end

      if partition[:direct_dispatches] > 0
        e.set_pipeline(selected)
        e.set_buffer(input, 0).set_buffer(weight, 1).set_buffer(bias, 2)
        e.set_buffer(output, 3, ML::Metal::BufferAccess::Write)
        e.set_value(rows.to_u32, 4).set_value(in_dim.to_u32, 5)
        e.set_value(out_dim.to_u32, 6)
        e.dispatch_threadgroups(
          {partition[:direct_groups_x], partition[:direct_groups_y], 1},
          {threadgroup_threads(variant), 1, 1})
      end

      if partition[:right_tail_groups] > 0
        e.set_pipeline(tail_pipeline.not_nil!)
        e.set_buffer(input, 0).set_buffer(weight, 1).set_buffer(bias, 2)
        e.set_buffer(output, 3, ML::Metal::BufferAccess::Write)
        e.set_value(rows.to_u32, 4).set_value(in_dim.to_u32, 5)
        e.set_value(out_dim.to_u32, 6)
        e.set_value(partition[:direct_groups_x].to_u32, 7).set_value(0_u32, 8)
        e.dispatch_threadgroups(
          {grid_x - partition[:direct_groups_x], partition[:direct_groups_y], 1},
          {threadgroup_threads(variant), 1, 1})
      end

      bottom_tail_groups = grid_y - partition[:direct_groups_y]
      if bottom_tail_groups > 0
        e.set_pipeline(tail_pipeline.not_nil!)
        e.set_buffer(input, 0).set_buffer(weight, 1).set_buffer(bias, 2)
        e.set_buffer(output, 3, ML::Metal::BufferAccess::Write)
        e.set_value(rows.to_u32, 4).set_value(in_dim.to_u32, 5)
        e.set_value(out_dim.to_u32, 6)
        e.set_value(0_u32, 7).set_value(partition[:direct_groups_y].to_u32, 8)
        e.dispatch_threadgroups(
          {grid_x, bottom_tail_groups, 1},
          {threadgroup_threads(variant), 1, 1})
      end
      return
    end

    e.set_pipeline(selected)
    e.set_buffer(input, 0).set_buffer(weight, 1).set_buffer(bias, 2)
    e.set_buffer(output, 3, ML::Metal::BufferAccess::Write)
    e.set_value(rows.to_u32, 4).set_value(in_dim.to_u32, 5)
    e.set_value(out_dim.to_u32, 6)
    e.dispatch_threadgroups(
      {grid_x, grid_y, 1}, {threadgroup_threads(variant), 1, 1})
  end

  # Diagnostic decode only. This entry point and the fused GEMM call the same
  # coefficient helper in Metal so their Q4_K/F16 rounding and error policy
  # remain directly comparable.
  def self.decode_probe(e : ML::Metal::ComputeEncoder,
                        payload : ML::MetalBuffer, sidecar : ML::MetalBuffer,
                        destination : ML::MetalBuffer, status : ML::MetalBuffer,
                        count : Int32, patches : Int32) : Nil
    validate_count!(count)
    validate_patches!(patches)
    payload_bytes = payload_bytes!(count.to_i64)
    deltas_bytes = sidecar_bytes(count, patches)
    check_bytes!(payload, payload_bytes, "Q4_K payload")
    check_bytes!(sidecar, deltas_bytes, "Q4_K delta sidecar")
    ML::Video::MetalOps.check(destination, count.to_i64)
    check_status!(status)
    assert_all_disjoint!(payload, sidecar, destination, status)

    selected = @@decode_pipeline || raise ArgumentError.new("Call Q4KDeltaLinear.prepare! before decode_probe")
    e.set_pipeline(selected)
    e.set_buffer(payload, 0).set_buffer(sidecar, 1)
    e.set_buffer(destination, 2, ML::Metal::BufferAccess::Write)
    e.set_buffer(status, 3, ML::Metal::BufferAccess::ReadWrite)
    e.set_value(count.to_u32, 4).set_value(patches.to_u32, 5)
    e.dispatch_1d(count)
  end

  # Qualification probe for the backend's simdgroup matrix-element mapping.
  # It round-trips one shared 8x8 coordinate tile through each of four SIMD
  # groups; callers compare all 256 group/row/column identities.
  def self.coordinate_mapping_probe(e : ML::Metal::ComputeEncoder,
                                    destination : ML::MetalBuffer) : Nil
    ML::Video::MetalOps.check(destination, 256_i64)
    selected = @@coordinate_mapping_probe_pipeline || raise ArgumentError.new("Call Q4KDeltaLinear.prepare! before coordinate_mapping_probe")
    e.set_pipeline(selected)
    e.set_buffer(destination, 0, ML::Metal::BufferAccess::Write)
    e.dispatch_threadgroups({1, 1, 1}, {threadgroup_threads, 1, 1})
  end

  # Materialize K=248 and K=256 fragments for both N halves, all four SIMD
  # groups, and both 8-channel subtiles. This uses the exact coordinate-derived
  # fragment loader used by the direct core.
  def self.q4_fragment_probe(e : ML::Metal::ComputeEncoder,
                             payload : ML::MetalBuffer, sidecar : ML::MetalBuffer,
                             destination : ML::MetalBuffer, status : ML::MetalBuffer,
                             patches : Int32) : Nil
    validate_patches!(patches)
    count = 32_i32 * 512_i32
    check_bytes!(payload, payload_bytes!(count.to_i64), "Q4_K payload")
    check_bytes!(sidecar, sidecar_bytes(count, patches), "Q4_K delta sidecar")
    ML::Video::MetalOps.check(destination, 1024_i64)
    check_status!(status)
    assert_all_disjoint!(payload, sidecar, destination, status)

    selected = @@q4k_fragment_probe_pipeline || raise ArgumentError.new("Call Q4KDeltaLinear.prepare! before q4_fragment_probe")
    e.set_pipeline(selected)
    e.set_buffer(payload, 0).set_buffer(sidecar, 1)
    e.set_buffer(destination, 2, ML::Metal::BufferAccess::Write)
    e.set_buffer(status, 3, ML::Metal::BufferAccess::ReadWrite)
    e.set_value(patches.to_u32, 4)
    e.dispatch_threadgroups({1, 2, 1}, {threadgroup_threads, 1, 1})
  end

  private def self.validate_gemm_pipeline!(pipeline : ML::Metal::ComputePipeline,
                                           label : String) : Nil
    unless pipeline.thread_execution_width == 32
      raise ArgumentError.new("#{label} requires a 32-wide SIMD group")
    end
    unless pipeline.max_total_threads_per_threadgroup >= 128
      raise ArgumentError.new("#{label} requires at least 128 threads per threadgroup")
    end
    unless pipeline.static_threadgroup_memory_length <= MAX_STATIC_THREADGROUP_BYTES
      raise ArgumentError.new("#{label} static threadgroup memory exceeds 32 KiB")
    end
  end

  private def self.validate_count!(count : Int32) : Nil
    unless count > 0 && count % BLOCK_ELEMENTS == 0
      raise ArgumentError.new("Q4_K value count must be a positive multiple of 256")
    end
  end

  private def self.validate_patches!(patches : Int32) : Nil
    unless patches == 0 || patches == 1 || patches == 4 || patches == 8
      raise ArgumentError.new("Q4_K delta patch count must be 0, 1, 4, or 8")
    end
  end

  private def self.payload_bytes!(count : Int64) : Int64
    unless count > 0 && count % BLOCK_ELEMENTS == 0
      raise ArgumentError.new("Q4_K payload count must contain whole 256-value blocks")
    end
    checked_product!(count // BLOCK_ELEMENTS, Q4_K_BLOCK_BYTES.to_i64,
      "Q4_K payload bytes")
  end

  private def self.check_bytes!(buffer : ML::MetalBuffer, required_bytes : Int64,
                                label : String) : Nil
    unless buffer.valid? && required_bytes > 0 && buffer.size >= required_bytes
      raise ArgumentError.new("Invalid or undersized #{label} buffer")
    end
  end

  private def self.check_status!(status : ML::MetalBuffer) : Nil
    check_bytes!(status, STATUS_WORDS.to_i64 * sizeof(UInt32), "Q4_K status")
  end

  private def self.assert_all_disjoint!(a : ML::MetalBuffer,
                                        b : ML::MetalBuffer,
                                        c : ML::MetalBuffer,
                                        d : ML::MetalBuffer,
                                        e : ML::MetalBuffer? = nil,
                                        f : ML::MetalBuffer? = nil) : Nil
    unless !ML::Video::MetalOps.overlap?(a, b) &&
           !ML::Video::MetalOps.overlap?(a, c) &&
           !ML::Video::MetalOps.overlap?(a, d) &&
           !ML::Video::MetalOps.overlap?(b, c) &&
           !ML::Video::MetalOps.overlap?(b, d) &&
           !ML::Video::MetalOps.overlap?(c, d)
      raise ArgumentError.new("Q4_K fused linear buffers must be disjoint")
    end
    if fifth = e
      unless !ML::Video::MetalOps.overlap?(a, fifth) &&
             !ML::Video::MetalOps.overlap?(b, fifth) &&
             !ML::Video::MetalOps.overlap?(c, fifth) &&
             !ML::Video::MetalOps.overlap?(d, fifth)
        raise ArgumentError.new("Q4_K fused linear buffers must be disjoint")
      end
    end
    if sixth = f
      unless fifth = e
        raise ArgumentError.new("Q4_K disjointness check requires the fifth buffer")
      end
      unless !ML::Video::MetalOps.overlap?(a, sixth) &&
             !ML::Video::MetalOps.overlap?(b, sixth) &&
             !ML::Video::MetalOps.overlap?(c, sixth) &&
             !ML::Video::MetalOps.overlap?(d, sixth) &&
             !ML::Video::MetalOps.overlap?(fifth, sixth)
        raise ArgumentError.new("Q4_K fused linear buffers must be disjoint")
      end
    end
  end

  private def self.checked_product!(left : Int64, right : Int64,
                                    label : String) : Int64
    if left < 0 || right < 0 || (right != 0 && left > Int64::MAX // right)
      raise ArgumentError.new("#{label} exceeds Int64")
    end
    left * right
  end

  private def self.checked_dispatch_grid!(rows : Int32,
                                          out_dim : Int32,
                                          variant : GemmVariant) : {Int32, Int32}
    columns = tile_n(variant).to_i64
    row_tile = tile_m(variant).to_i64
    grid_x = (out_dim.to_i64 + columns - 1_i64) // columns
    grid_y = (rows.to_i64 + row_tile - 1_i64) // row_tile
    grid_count = checked_product!(grid_x, grid_y, "GEMM dispatch groups")
    unless grid_x > 0 && grid_y > 0 &&
           grid_x <= Int32::MAX && grid_y <= Int32::MAX &&
           grid_count <= Int32::MAX && grid_count <= UInt32::MAX
      raise ArgumentError.new("Q4_K GEMM dispatch grid exceeds supported bounds")
    end
    {grid_x.to_i32, grid_y.to_i32}
  end
end
