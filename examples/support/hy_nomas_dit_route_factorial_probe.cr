require "digest/sha256"
require "file/tempfile"
require "json"
require "path"

# CPU-only admission and source-reference reader for the direct text-route
# factorial. This file deliberately owns no Metal resources and performs no
# inference. The caller must finish this admission before creating a device,
# plan, or command chain.
module HyNomasDiTRouteFactorialProbe
  extend self

  SCHEMA                      = "hy-nomas-dit-route-factorial-reference-v1"
  UPSTREAM_REVISION           = "911cea6458a89c2663513ff7099561ec190dd491"
  REFERENCE_DEFAULT_PATH      = "/opt/mldata_ssd/cogni-ml/hy-nomas-direct-routes-20261105.klzuhg0r/source.json"
  FROZEN_INPUT_SHA256         = "71abfe81a63cc3042063444d76a494e960e704386408a81a45a16f280398c3cf"
  ROI_MANIFEST_SHA256         = "ee7a830eb287cffd5d4ee2297d1ce5bd99e7f26a3dcc36442358c12f6ba72dd8"
  ROI_PAYLOAD_SHA256          = "f59abfea419c9f2fe5bddc6c8595665ce55a96038f431f0273f88475a9f6388c"
  ROI_CAPTURE_SHA256          = "1b7f1a09a1d71954cf93f1439e1b77fbc2aa1fa6f3c4a33be4174da76c7129da"
  CONTINUATION_SHA256         = "f23608e9b93bb1a2da0b90e9d48f9edf1506fb65bd2b2ae783d9c1440a1c8cec"
  CONTINUATION_PAYLOAD_SHA256 = "b1e10540ccf3dbf58c62511f37d4f00e32b0dec4b60348ef0dff85fd584be8f9"
  PARENT_COMPARISON_SHA256    = "2bea5367da4dea43783243b6aefa0daa06e29eadf7879160e88eba23cbc51bd2"
  SOURCE_RANGE_BUILDER_SHA256 = "acf827bb7feb84c9046618737146544843a3f50c3500a3b16925dadeb6c5bdee"
  CACHE_INDEX_SHA256          = "a235ca592b47161472c7fd94cd3d886fce164480cd0571179c8860002063cbb0"
  CHECKPOINT_SHA256           = "73819bcf3b82b090116dd1a1748795de95eef703c304e7443963d00ea8aa4d70"

  VIDEO_ROWS             = 4_320
  T0_TEXT_ROWS           =   263
  T1_TEXT_ROWS           =   224
  TEXT_STORAGE_ROWS      =   320
  PHYSICAL_CACHE_ROWS    = 4_640
  HISTORY_ROWS           = VIDEO_ROWS + T0_TEXT_ROWS
  QUERY_ROWS             = VIDEO_ROWS + T1_TEXT_ROWS
  KEY_ROWS               = HISTORY_ROWS + QUERY_ROWS
  HIDDEN                 =  2_048
  HEADS                  =     16
  HEAD_DIM               =    128
  BLOCKS                 =     54
  ATOL                   = 2.0e-5
  RTOL                   = 2.0e-5
  BANK_ELEMENTS          = VIDEO_ROWS * HEAD_DIM
  BANK_BYTES             = BANK_ELEMENTS.to_i64 * sizeof(Float32)
  ROPE_COS_RECORD_OFFSET = 12_492_800_i64
  ROPE_SIN_RECORD_OFFSET = 25_763_840_i64
  ROPE_RECORD_BYTES      = 13_271_040_i64

  CASES                = ["M11", "M00", "M10", "M01", "M11_repeat"]
  EXPECTED_MASK_SHA256 = {
    "M11"        => "e4cf416593cb0adea6ee0de61806413384ba3f024b679974462fa2c9e69791ce",
    "M00"        => "35f63fb0e5dcc8e0d38fed8c9125ad07c20d2ecffb025ea7f19651182f280e2c",
    "M10"        => "6503be9d8153246c0da49195274a306eb82b98b4a542982207a0993bb98b8fb9",
    "M01"        => "400e5f3e3180e1c48ceb95ce124432626fb01fe12c583ba0a7a724fc59fe4e30",
    "M11_repeat" => "e4cf416593cb0adea6ee0de61806413384ba3f024b679974462fa2c9e69791ce",
  }
  CHECKPOINTS      = ["velocity", "block0.image", "block0.text_valid", "block53.image", "block53.text_valid"]
  T0_CAPTURE_NAMES = ["t0.patch", "t0.text_valid", "t0.vector"]
  T1_CAPTURE_NAMES = ["t1.step1.patch", "t1.step1.text_valid", "t1.step1.vector"]
  INPUT_NAMES      = T0_CAPTURE_NAMES + T1_CAPTURE_NAMES + ["t1.initial_latent", "t1.step1.velocity"]
  INPUT_SHAPES     = {
    "t0.patch"            => [1, VIDEO_ROWS, HIDDEN],
    "t0.text_valid"       => [1, T0_TEXT_ROWS, HIDDEN],
    "t0.vector"           => [1, HIDDEN],
    "t1.step1.patch"      => [1, VIDEO_ROWS, HIDDEN],
    "t1.step1.text_valid" => [1, T1_TEXT_ROWS, HIDDEN],
    "t1.step1.vector"     => [1, HIDDEN],
    "t1.initial_latent"   => [1, 32, 30, 9, 16],
    "t1.step1.velocity"   => [1, 32, 30, 9, 16],
  }
  INPUT_BINDINGS = {
    "t0.patch"            => {3_268_608_i64, 35_389_440_i64, "bbcc0e5fdfb37c3099bcef32bde4b599910e6f84262818939c3a834a7b09ba1f"},
    "t0.text_valid"       => {1_114_112_i64, 2_154_496_i64, "c51b363ce28586b2589eb801790f0d73a73289f808813723f0dadf01f946d79f"},
    "t0.vector"           => {1_105_920_i64, 8_192_i64, "e9f579a4363097651df785dda4837e468949319afab4ad47763920b90f217697"},
    "t1.step1.patch"      => {153_686_016_i64, 35_389_440_i64, "d6925473556e9fb99adc1cc9ec1d4160b89eafb7da025073cd0c519f8d73d4bc"},
    "t1.step1.text_valid" => {151_851_008_i64, 1_835_008_i64, "082473f87ae24b6104afa8a8a69e4a9e7ba57eef46437aa61dd3df7a732dc089"},
    "t1.step1.vector"     => {151_842_816_i64, 8_192_i64, "68c388122813f799812431a52d7af5f5b863eda2a923ef0365fe079f6971081d"},
    "t1.initial_latent"   => {151_289_856_i64, 552_960_i64, "503615dc8920afe80d441fa025380c4b764b1e81f07d3a44525127010ddb2e5a"},
    "t1.step1.velocity"   => {263_524_352_i64, 552_960_i64, "bce56894f9a6dacd12e78e7171c7770d98206aa3f8a2dff3c5bdd30a110d2222"},
  }
  ROPE_RECORDS = {
    "input.rope_cos" => {ROPE_COS_RECORD_OFFSET, "7ddb17895b9bcd14ed840a8b80eb399c0470ccf6933b0777903cda1536d0afb8",
                         {"T0" => "441ecf64926cacbd19f7b2690c6b13caebc6ff11ab4aa691e674350ddd04e6f6",
                          "T1" => "82202e2b10995313e532e514a9a216e31919d23d4a12aaa38ba77e98283f3069"}},
    "input.rope_sin" => {ROPE_SIN_RECORD_OFFSET, "272ac7251cb39de8a442c7221792e4bef8440c140cff639496a2b82f90c50306",
                         {"T0" => "1180aa76085af34dd7d66b277fea29699ccb8689ebca0359b59b4738b6e9645d",
                          "T1" => "70a852688a745cc6fb040ee18fc3795c1070b738321bcbc8992df387875ae929"}},
  }

  FACTORS = {
    "M11"        => {action1_video_to_text: true, action2_video_to_text: true},
    "M00"        => {action1_video_to_text: false, action2_video_to_text: false},
    "M10"        => {action1_video_to_text: true, action2_video_to_text: false},
    "M01"        => {action1_video_to_text: false, action2_video_to_text: true},
    "M11_repeat" => {action1_video_to_text: true, action2_video_to_text: true},
  }
  OUTPUT_SHAPES = {
    "velocity"           => [1, 32, 30, 9, 16],
    "block0.image"       => [1, VIDEO_ROWS, HIDDEN],
    "block0.text_valid"  => [1, T1_TEXT_ROWS, HIDDEN],
    "block53.image"      => [1, VIDEO_ROWS, HIDDEN],
    "block53.text_valid" => [1, T1_TEXT_ROWS, HIDDEN],
  }
  MAX_MANIFEST_BYTES = 16_i64 * 1024 * 1024
  HASH_CHUNK_BYTES   = 8 * 1024 * 1024

  record TensorRecord, name : String, shape : Array(Int32),
    dtype : String, source_dtype : String, offset_bytes : Int64,
    nbytes : Int64, sha256 : String

  record Comparison, elements : Int64, failures : Int64,
    max_abs : Float64, rms_abs : Float64, exact_bit_differences : Int64

  def expected_geometry : Hash(String, JSON::Any)
    JSON.parse({
      "video_rows" => VIDEO_ROWS, "t0_valid_text_rows" => T0_TEXT_ROWS,
      "t1_valid_text_rows" => T1_TEXT_ROWS, "text_storage_rows" => TEXT_STORAGE_ROWS,
      "physical_cache_rows" => PHYSICAL_CACHE_ROWS, "history_rows" => HISTORY_ROWS,
      "query_rows" => QUERY_ROWS, "key_rows" => KEY_ROWS, "hidden" => HIDDEN,
      "heads" => HEADS, "head_dim" => HEAD_DIM, "blocks" => BLOCKS,
      "attention_layout" => "compact",
    }.to_json).as_h
  end

  def query_key_labels : Tuple(Array(Int32), Array(Int32))
    q = Array(Int32).new(VIDEO_ROWS, 0)
    # Preserve the pinned source range labels for fixture authentication.
    # Here -1 is an active video query outside an action window, with access
    # to global keys. It is not native padding; translate it before binding.
    [[0, 144, -1], [144, 1_440, 1], [1_440, 1_728, -1],
     [1_728, 4_032, 2], [4_032, 4_320, -1]].each do |window|
      start = window[0].to_i32
      stop = window[1].to_i32
      label = window[2].to_i32
      q[start...stop] = Array(Int32).new(stop - start, label)
    end
    q.concat(Array(Int32).new(49, 0))
    q.concat(Array(Int32).new(83, 1))
    q.concat(Array(Int32).new(92, 2))

    # All history and current video keys stay globally visible. The positive
    # labels are only on text key spans, so q=0 disables only direct text routes.
    k = Array(Int32).new(HISTORY_ROWS + VIDEO_ROWS, 0)
    k.concat(Array(Int32).new(49, 0))
    k.concat(Array(Int32).new(83, 1))
    k.concat(Array(Int32).new(92, 2))
    raise ArgumentError.new("DiT route label geometry mismatch") unless q.size == QUERY_ROWS && k.size == KEY_ROWS
    {q, k}
  end

  def query_labels_for(case_name : String) : Array(Int32)
    factors = FACTORS[case_name]? || raise ArgumentError.new("Unadmitted DiT route case: #{case_name}")
    q, _ = query_key_labels
    VIDEO_ROWS.times do |row|
      label = q[row]
      next unless label == 1 || label == 2
      action = label == 1 ? 1 : 2
      enabled = action == 1 ? factors[:action1_video_to_text] : factors[:action2_video_to_text]
      # The accepted M11 ROI mask already clears the matching action-text
      # edges outside its authored ROI. The factorial can additionally clear
      # the positive in-ROI query labels for an action whose route is off.
      next if enabled && roi_contains?(row, action)
      q[row] = 0
    end
    q
  end

  def native_query_labels_for(case_name : String) : Array(Int32)
    labels = query_labels_for(case_name)
    VIDEO_ROWS.times do |row|
      # Native negative labels reserve a finite dummy softmax input and zero
      # the scattered context. All compact video rows are active, including
      # no-action windows, whose global-only native label must instead be 0.
      labels[row] = 0 if labels[row] < 0
    end
    unless labels.size == QUERY_ROWS && labels.all? { |label| label >= 0 }
      raise ArgumentError.new("compact route queries must not contain native padding")
    end
    labels
  end

  def factorial_matrix_sha256(case_name : String) : Tuple(String, Int64, Int64, Int64)
    CASES.includes?(case_name) || raise ArgumentError.new("Unadmitted route case: #{case_name}")
    raw_queries, _ = query_key_labels
    q = query_labels_for(case_name)
    _, keys = query_key_labels
    templates = {} of Int32 => Bytes
    row_counts = {} of Int32 => Int64
    [-1, 0, 1, 2].each do |label|
      row = Bytes.new(KEY_ROWS, 0_u8)
      (0...HISTORY_ROWS + VIDEO_ROWS + 49).each { |index| row[index] = 1_u8 }
      if label == 1
        (HISTORY_ROWS + VIDEO_ROWS + 49...HISTORY_ROWS + VIDEO_ROWS + 132).each { |index| row[index] = 1_u8 }
      elsif label == 2
        (HISTORY_ROWS + VIDEO_ROWS + 132...KEY_ROWS).each { |index| row[index] = 1_u8 }
      end
      # Check the template against explicit key labels so the compact row
      # optimization cannot silently diverge from the declared label rule.
      row.each_index do |index|
        allowed = keys[index] == 0 || (label >= 0 && keys[index] == label)
        raise ArgumentError.new("route row template mismatch") unless row[index] == (allowed ? 1_u8 : 0_u8)
      end
      row_counts[label] = row.count(1_u8).to_i64
      templates[label] = row
    end
    digest = Digest::SHA256.new
    allowed = 0_i64
    removed_action1 = 0_i64
    removed_action2 = 0_i64
    q.each_with_index do |label, query|
      digest.update(templates[label])
      allowed += row_counts[label]
      if query < VIDEO_ROWS
        original_label = raw_queries[query]
        if original_label == 1 && roi_contains?(query, 1) && !FACTORS[case_name][:action1_video_to_text]
          removed_action1 += 83
        elsif original_label == 2 && roi_contains?(query, 2) && !FACTORS[case_name][:action2_video_to_text]
          removed_action2 += 92
        end
      end
    end
    {digest.final.hexstring, allowed, removed_action1, removed_action2}
  end

  def roi_contains?(query_index : Int32, action : Int32) : Bool
    frame, spatial = query_index.divmod(144)
    h, w = spatial.divmod(16)
    case action
    when 1
      1 <= frame < 10 && 0 <= h < 9 && 1 <= w < 8
    when 2
      12 <= frame < 28 && 2 <= h < 8 && 10 <= w < 15
    else
      false
    end
  end

  def sha256_file(path : String) : String
    digest = Digest::SHA256.new
    File.open(path) do |io|
      buffer = Bytes.new(HASH_CHUNK_BYTES)
      while (count = io.read(buffer)) > 0
        digest.update(buffer[0, count])
      end
    end
    digest.final.hexstring
  end

  def sha256_range(path : String, offset : Int64, count : Int64) : String
    raise ArgumentError.new("Invalid file range") if offset < 0 || count < 0
    info = File.info(path)
    raise ArgumentError.new("File range exceeds payload: #{path}") if offset + count > info.size
    digest = Digest::SHA256.new
    File.open(path) do |io|
      io.seek(offset)
      buffer = Bytes.new(Math.min(HASH_CHUNK_BYTES.to_i64, Math.max(count, 1_i64)).to_i)
      remaining = count
      while remaining > 0
        take = Math.min(buffer.size.to_i64, remaining).to_i
        io.read_fully(buffer[0, take])
        digest.update(buffer[0, take])
        remaining -= take
      end
    end
    digest.final.hexstring
  end

  def canonical_regular_file(path : String, label : String) : String
    info = File.info?(path, false)
    unless info && info.file? && !info.symlink?
      raise ArgumentError.new("Expected a non-symlink regular file for #{label}: #{path}")
    end
    File.realpath(path)
  end

  def require_digest!(digest : String, label : String) : Nil
    unless digest.matches?(/\A[0-9a-f]{64}\z/)
      raise ArgumentError.new("Malformed #{label} SHA-256")
    end
  end

  def exact_keys!(value : JSON::Any, keys : Array(String), label : String) : Nil
    actual = value.as_h.keys.sort
    expected = keys.sort
    unless actual == expected
      raise ArgumentError.new("#{label} keys mismatch; expected #{expected.join(',')}, got #{actual.join(',')}")
    end
  end

  def read_json(path : String, expected_sha256 : String, label : String) : JSON::Any
    canonical = canonical_regular_file(path, label)
    actual = sha256_file(canonical)
    unless actual == expected_sha256
      raise ArgumentError.new("#{label} digest mismatch")
    end
    JSON.parse(File.read(canonical))
  rescue ex : JSON::ParseException
    raise ArgumentError.new("Malformed #{label} JSON: #{ex.message}")
  end

  def tensor_elements(shape : Array(Int32)) : Int64
    shape.reduce(1_i64) { |product, dimension| product * dimension }
  end

  def compare_values(expected : Array(Float32), actual : Array(Float32),
                     atol : Float64 = ATOL, rtol : Float64 = RTOL) : Comparison
    unless expected.size == actual.size && atol == ATOL && rtol == RTOL
      raise ArgumentError.new("DiT route comparator shape or tolerance mismatch")
    end
    failures = 0_i64
    different = 0_i64
    maximum = 0.0_f64
    square_sum = 0.0_f64
    expected.each_index do |index|
      reference = expected[index]
      value = actual[index]
      unless reference.finite? && value.finite?
        failures += 1
        next
      end
      different += 1 if reference.unsafe_as(UInt32) != value.unsafe_as(UInt32)
      error = (reference.to_f64 - value.to_f64).abs
      maximum = Math.max(maximum, error)
      square_sum += error * error
      failures += 1 if error > atol + rtol * reference.to_f64.abs
    end
    rms = expected.empty? ? 0.0_f64 : Math.sqrt(square_sum / expected.size)
    Comparison.new(expected.size.to_i64, failures, maximum, rms, different)
  end

  # Count bitwise changes only on protected rows. A true entry in
  # permitted_change_rows allows that row to differ; false requires an exact
  # Float32-bit match for every element in the row.
  def exact_protected_row_bit_differences(expected : Array(Float32),
                                          actual : Array(Float32),
                                          row_width : Int32,
                                          permitted_change_rows : Array(Bool)) : Int64
    unless row_width > 0 && expected.size == actual.size && expected.size % row_width == 0
      raise ArgumentError.new("protected-row comparison shape mismatch")
    end
    rows = expected.size // row_width
    unless permitted_change_rows.size == rows
      raise ArgumentError.new("protected-row permission mask shape mismatch")
    end

    differences = 0_i64
    expected.each_index do |index|
      row = index // row_width
      next if permitted_change_rows[row]
      differences += 1 if expected[index].unsafe_as(UInt32) != actual[index].unsafe_as(UInt32)
    end
    differences
  end

  # Count nonzero elementwise Float64 interactions for v11-v10-v01+v00.
  # Inputs must have the same extent and contain finite Float32 values.
  def four_arm_interaction_nonzero_count(v11 : Array(Float32), v00 : Array(Float32),
                                         v10 : Array(Float32), v01 : Array(Float32)) : Int64
    unless v11.size == v00.size && v11.size == v10.size && v11.size == v01.size
      raise ArgumentError.new("four-arm interaction shape mismatch")
    end

    nonzero = 0_i64
    v11.each_index do |index|
      unless v11[index].finite? && v10[index].finite? &&
             v01[index].finite? && v00[index].finite?
        raise ArgumentError.new("four-arm interaction contains a non-finite value")
      end
      interaction = v11[index].to_f64 - v10[index].to_f64 - v01[index].to_f64 + v00[index].to_f64
      nonzero += 1 unless interaction == 0.0_f64
    end
    nonzero
  end

  # Crystal's File mode parser does not support "wx". O_EXCL preserves fresh
  # output semantics even if another file or a symlink appears after admission.
  # Stream payloads directly instead of making another full host copy.
  def with_fresh_output(path : String, &) : Nil
    fd = LibC.open(path, LibC::O_WRONLY | LibC::O_CREAT | LibC::O_EXCL, 0o600)
    raise IO::Error.new("native output must be fresh and writable: #{path}") if fd < 0
    io = IO::FileDescriptor.new(fd)
    begin
      yield io
    ensure
      io.close
    end
  end

  def fresh_output_self_test! : Nil
    path = File.tempname("hy-nomas-route-output-", ".bin")
    link = path + ".symlink"
    created = false
    link_created = false
    expected = Bytes[0_u8, 10_u8, 255_u8]
    begin
      with_fresh_output(path) do |io|
        created = true
        io.write(expected)
      end
      raise IO::Error.new("fresh output byte roundtrip failed") unless File.read(path).to_slice == expected

      overwrite_rejected = false
      begin
        with_fresh_output(path) { |io| io << "overwrite" }
      rescue IO::Error
        overwrite_rejected = true
      end
      unless overwrite_rejected && File.read(path).to_slice == expected
        raise IO::Error.new("fresh output overwrite guard failed")
      end

      File.symlink(path, link)
      link_created = true
      symlink_rejected = false
      begin
        with_fresh_output(link) { |io| io << "overwrite through symlink" }
      rescue IO::Error
        symlink_rejected = true
      end
      unless symlink_rejected && File.symlink?(link) && File.read(path).to_slice == expected
        raise IO::Error.new("fresh output symlink guard failed")
      end
    ensure
      File.delete(link) if link_created
      File.delete(path) if created
    end
  end

  # A small deterministic CPU falsifier for the direct-route-only label change
  # and unchanged tolerance comparator. It exercises the same route-label and
  # comparison methods used by Reference admission/runner code.
  def self_test! : Nil
    fresh_output_self_test!
    q11 = query_labels_for("M11")
    q00 = query_labels_for("M00")
    q10 = query_labels_for("M10")
    q01 = query_labels_for("M01")
    _, keys = query_key_labels
    raise ArgumentError.new("route self-test label extents failed") unless q11.size == QUERY_ROWS && keys.size == KEY_ROWS
    raise ArgumentError.new("route self-test altered a text query") unless q11[VIDEO_ROWS, T1_TEXT_ROWS] == q00[VIDEO_ROWS, T1_TEXT_ROWS]
    raise ArgumentError.new("route self-test altered a text query") unless q11[VIDEO_ROWS, T1_TEXT_ROWS] == q10[VIDEO_ROWS, T1_TEXT_ROWS]
    raise ArgumentError.new("route self-test altered a text query") unless q11[VIDEO_ROWS, T1_TEXT_ROWS] == q01[VIDEO_ROWS, T1_TEXT_ROWS]
    raise ArgumentError.new("route self-test altered a video key") unless keys[0, HISTORY_ROWS + VIDEO_ROWS].all?(&.zero?)
    raise ArgumentError.new("route self-test failed to disable A1") unless q00.count(1_i32) < q11.count(1_i32) && q00.count(2_i32) < q11.count(2_i32)
    raise ArgumentError.new("route self-test did not isolate A2") unless q10.count(1_i32) == q11.count(1_i32) && q10.count(2_i32) < q11.count(2_i32)
    raise ArgumentError.new("route self-test did not isolate A1") unless q01.count(1_i32) < q11.count(1_i32) && q01.count(2_i32) == q11.count(2_i32)
    exact = compare_values([1.0_f32, -0.5_f32], [1.0_f32, -0.5_f32])
    bad = compare_values([1.0_f32], [1.001_f32])
    raise ArgumentError.new("route self-test exact comparator failed") unless exact.failures == 0 && exact.exact_bit_differences == 0
    raise ArgumentError.new("route self-test tolerance falsifier failed") unless bad.failures == 1

    # A single BF16 subnormal is far inside the absolute tolerance at zero,
    # but exact protected-row checks must still catch it.
    tiny_bf16 = 0x0001_0000_u32.unsafe_as(Float32)
    under_tolerance = compare_values([0.0_f32], [tiny_bf16])
    protected_change = exact_protected_row_bit_differences(
      [0.0_f32, 1.0_f32], [tiny_bf16, 1.0_f32], 1, [false, true])
    permitted_change = exact_protected_row_bit_differences(
      [0.0_f32, 1.0_f32], [tiny_bf16, 1.0_f32], 1, [true, false])
    unless under_tolerance.failures == 0 && under_tolerance.exact_bit_differences == 1 &&
           protected_change == 1 && permitted_change == 0
      raise ArgumentError.new("protected-row exactness falsifier failed")
    end

    zero_arm = [0.0_f32, 0.0_f32]
    interaction_count = four_arm_interaction_nonzero_count(
      [tiny_bf16, 1.0_f32], zero_arm, zero_arm, zero_arm)
    unless interaction_count == 2
      raise ArgumentError.new("four-arm interaction falsifier failed")
    end
    cancelled_interaction = four_arm_interaction_nonzero_count(
      [1.0_f32], [0.0_f32], [0.5_f32], [0.5_f32])
    raise ArgumentError.new("four-arm interaction cancellation failed") unless cancelled_interaction == 0

    invalid_row_shape = false
    begin
      exact_protected_row_bit_differences([0.0_f32], [0.0_f32], 0, [] of Bool)
    rescue ArgumentError
      invalid_row_shape = true
    end
    invalid_permission_mask = false
    begin
      exact_protected_row_bit_differences([0.0_f32], [0.0_f32], 1, [false, true])
    rescue ArgumentError
      invalid_permission_mask = true
    end
    invalid_interaction_shape = false
    begin
      four_arm_interaction_nonzero_count([0.0_f32], [] of Float32,
        [0.0_f32], [0.0_f32])
    rescue ArgumentError
      invalid_interaction_shape = true
    end
    unless invalid_row_shape && invalid_permission_mask && invalid_interaction_shape
      raise ArgumentError.new("protected-row or interaction shape guard falsifier failed")
    end

    CASES.each do |name|
      source_labels = query_labels_for(name)
      native_labels = native_query_labels_for(name)
      unless source_labels.count(-1) == 720 && native_labels.none? { |label| label < 0 }
        raise ArgumentError.new("active global-video query translation failed: #{name}")
      end
      source_labels.each_with_index do |label, row|
        expected = row < VIDEO_ROWS && label < 0 ? 0 : label
        unless native_labels[row] == expected
          raise ArgumentError.new("native query translation changed an action or text row: #{name}")
        end
      end
      digest, _edges, removed_a1, removed_a2 = factorial_matrix_sha256(name)
      factors = FACTORS[name]
      expected_a1 = factors[:action1_video_to_text] ? 0_i64 : 47_061_i64
      expected_a2 = factors[:action2_video_to_text] ? 0_i64 : 44_160_i64
      unless digest == EXPECTED_MASK_SHA256[name] && removed_a1 == expected_a1 && removed_a2 == expected_a2
        raise ArgumentError.new("route self-test matrix digest or edge-removal count mismatch: #{name}")
      end
    end
  end

  class Reference
    include HyNomasDiTRouteFactorialProbe

    getter path : String
    getter sha256 : String
    getter document : JSON::Any
    getter source_root : String
    getter cache_index_path : String
    getter cache_index_sha256 : String
    getter frozen_input_paths : Hash(String, String)
    getter inputs : Hash(String, Array(Float32))
    getter rope_cos : Array(Float32)
    getter rope_sin : Array(Float32)
    getter query_labels : Array(Int32)
    getter key_labels : Array(Int32)
    getter output_payload_path : String
    getter output_payload_sha256 : String
    getter output_tensors : Hash(String, TensorRecord)

    @output_info : File::Info

    def initialize(path : String, expected_sha256 : String)
      @path = HyNomasDiTRouteFactorialProbe.canonical_regular_file(path, "route source reference")
      raise ArgumentError.new("route reference exceeds the manifest size limit") if File.size(@path) > MAX_MANIFEST_BYTES
      raw = File.read(@path)
      @sha256 = Digest::SHA256.hexdigest(raw)
      HyNomasDiTRouteFactorialProbe.require_digest!(expected_sha256, "expected reference")
      raise ArgumentError.new("route reference digest mismatch") unless @sha256 == expected_sha256
      @document = JSON.parse(raw)
      validate_document!

      source = @document["source"]
      @source_root = HyNomasDiTRouteFactorialProbe.canonical_regular_file(
        File.join(source["upstream_path"].as_s, "configs/constants.py"), "pinned upstream root anchor",
      ).sub(/\/configs\/constants\.py\z/, "")
      @cache_index_path = HyNomasDiTRouteFactorialProbe.canonical_regular_file(
        source["cache_index_path"].as_s, "DiT cache index",
      )
      @cache_index_sha256 = source["cache_index_sha256"].as_s

      @frozen_input_paths = {} of String => String
      @inputs = {} of String => Array(Float32)
      @rope_cos = [] of Float32
      @rope_sin = [] of Float32
      @query_labels, @key_labels = HyNomasDiTRouteFactorialProbe.query_key_labels
      @frozen_document = nil.as(JSON::Any?)
      validate_source_inputs!
      validate_mask_proof!
      @output_payload_path, @output_payload_sha256 = read_output_payload!
      @output_info = File.info(@output_payload_path)
      @output_tensors = read_output_tensors!
      validate_source_output_invariants!
    end

    def query_labels_for(case_name : String) : Array(Int32)
      HyNomasDiTRouteFactorialProbe.query_labels_for(case_name)
    end

    def native_query_labels_for(case_name : String) : Array(Int32)
      HyNomasDiTRouteFactorialProbe.native_query_labels_for(case_name)
    end

    def expected_values(key : String) : Array(Float32)
      record = @output_tensors[key]? || raise ArgumentError.new("Unknown route reference tensor: #{key}")
      current = File.info(@output_payload_path)
      unless current.same_file?(@output_info) && current.size == @output_info.size &&
             current.modification_time == @output_info.modification_time
        raise ArgumentError.new("route output payload changed after admission")
      end
      raw = Bytes.new(record.nbytes.to_i)
      File.open(@output_payload_path) do |io|
        io.seek(record.offset_bytes)
        io.read_fully(raw)
      end
      unless Digest::SHA256.hexdigest(raw) == record.sha256
        raise ArgumentError.new("route tensor changed after admission: #{key}")
      end
      count = (record.nbytes // sizeof(Float32)).to_i
      values = Array(Float32).new(count) do |index|
        IO::ByteFormat::LittleEndian.decode(Float32, raw[index * 4, 4])
      end
      values.each do |value|
        unless value.finite? && value.unsafe_as(UInt32) & 0xffff_u32 == 0_u32
          raise ArgumentError.new("route tensor is not finite lossless BF16 expansion: #{key}")
        end
      end
      values
    end

    private def validate_document! : Nil
      exact_keys!(@document,
        %w(schema schema_version upstream_revision source source_inputs geometry case_factors mask_proof precision execution payload tensors),
        "route source reference")
      unless @document["schema"].as_s == SCHEMA && @document["schema_version"].as_i == 1 &&
             @document["upstream_revision"].as_s == UPSTREAM_REVISION
        raise ArgumentError.new("route source schema or upstream revision mismatch")
      end
      validate_geometry!
      validate_case_factors!
      validate_precision!
      validate_execution!
      validate_source!
    end

    private def validate_geometry! : Nil
      expected = HyNomasDiTRouteFactorialProbe.expected_geometry
      actual = @document["geometry"].as_h
      raise ArgumentError.new("route geometry key inventory mismatch") unless actual.keys.sort == expected.keys.sort
      expected.each do |key, value|
        raise ArgumentError.new("route geometry mismatch: #{key}") unless actual[key] == value
      end
    end

    private def validate_case_factors! : Nil
      factors = @document["case_factors"]
      exact_keys!(factors, CASES, "route case factors")
      CASES.each do |name|
        record = factors[name]
        exact_keys!(record, %w(action1_video_to_text action2_video_to_text), "factor #{name}")
        expected = FACTORS[name]
        unless record["action1_video_to_text"].as_bool == expected[:action1_video_to_text] &&
               record["action2_video_to_text"].as_bool == expected[:action2_video_to_text]
          raise ArgumentError.new("route factor semantics mismatch: #{name}")
        end
      end
    end

    private def validate_precision! : Nil
      precision = @document["precision"]
      exact_keys!(precision,
        %w(resident_weights input_storage activation_dtype attention_adapter cache_dtype all_54_block_shards_authenticated),
        "route precision")
      unless precision["resident_weights"].as_s == "BF16" &&
             precision["input_storage"].as_s == "F32 lossless promotion of BF16 saved states" &&
             precision["activation_dtype"].as_s == "BF16" &&
             precision["attention_adapter"].as_s == "pinned Hy-Nomas source cache body with compact boolean MPS SDPA chunks of 256; BF16 Q/K/V" &&
             precision["cache_dtype"].as_s == "BF16" &&
             precision["all_54_block_shards_authenticated"].as_bool
        raise ArgumentError.new("route source precision mismatch")
      end
    end

    private def validate_execution! : Nil
      execution = @document["execution"]
      exact_keys!(execution,
        %w(case_order scope schedule full_blocks_per_arm t0_writer_block_calls t1_arm_block_calls
          source_range_builder_invocations source_mask_cases terminal_export_count
          explicit_terminal_mps_synchronize_count block0_text_equal_all_cases parent_replay source_aa),
        "route execution")
      unless execution["case_order"].as_a.map(&.as_s) == CASES &&
             execution["scope"].as_s == "saved-state direct DiT only; no encoder/refiner/Euler/DCAE" &&
             execution["schedule"].as_s == "layer-outer T0 writer then five independent T1 arms per layer; one resident BF16 block and one T0 K/V cache" &&
             execution["full_blocks_per_arm"].as_i == BLOCKS &&
             execution["t0_writer_block_calls"].as_i == BLOCKS &&
             execution["t1_arm_block_calls"].as_i == BLOCKS * (CASES.size) &&
             execution["source_range_builder_invocations"].as_i == 1 &&
             execution["source_mask_cases"].as_i == CASES.size &&
             execution["terminal_export_count"].as_i == 1 &&
             execution["explicit_terminal_mps_synchronize_count"].as_i == 1 &&
             execution["block0_text_equal_all_cases"].as_bool
        raise ArgumentError.new("route execution order or scope mismatch")
      end
      replay = execution["parent_replay"]
      exact_keys!(replay, ["M11"], "accepted parent replay")
      validate_pass_records!(replay["M11"], "parent.M11")
      validate_pass_records!(execution["source_aa"], "M11/M11_repeat")
    end

    private def validate_pass_records!(records : JSON::Any, prefix : String) : Nil
      exact_keys!(records, CHECKPOINTS, "route comparison checkpoint inventory")
      CHECKPOINTS.each do |checkpoint|
        record = records[checkpoint]
        exact_keys!(record,
          %w(checkpoint elements max_abs mean_abs rms_abs outside_atol_rtol all_finite passed atol rtol),
          "route comparison")
        label = prefix == "parent.M11" ? "parent.M11.#{checkpoint}" : "M11/M11_repeat.#{checkpoint}"
        unless record["checkpoint"].as_s == checkpoint && record["elements"].as_i64 == tensor_elements(OUTPUT_SHAPES[checkpoint]) &&
               record["outside_atol_rtol"].as_i == 0 && record["all_finite"].as_bool &&
               record["passed"].as_bool && record["atol"].as_f == ATOL && record["rtol"].as_f == RTOL
          raise ArgumentError.new("route source comparison gate failed: #{label}")
        end
        %w(max_abs mean_abs rms_abs).each do |statistic|
          value = record[statistic].as_f
          raise ArgumentError.new("invalid route source comparison statistic: #{label}.#{statistic}") unless value.finite? && value >= 0.0
          if prefix == "M11/M11_repeat" && value != 0.0
            raise ArgumentError.new("source repeat summary is nonzero despite exact A/A requirement: #{label}")
          end
        end
      end
    end

    private def validate_source! : Nil
      source = @document["source"]
      exact_keys!(source,
        %w(upstream_path upstream_revision source_hashes cache_index_path cache_index_sha256 block_shards global_shard total_weight_bytes),
        "route source")
      raise ArgumentError.new("upstream root must be absolute") unless source["upstream_path"].as_s.starts_with?('/')
      raise ArgumentError.new("pinned source revision mismatch") unless source["upstream_revision"].as_s == UPSTREAM_REVISION
      files = source["source_hashes"].as_h
      raise ArgumentError.new("pinned source file map must not be empty") if files.empty?
      files.each do |relative, digest|
        unless !relative.starts_with?('/') && !relative.split('/').includes?("..")
          raise ArgumentError.new("unsafe pinned source path: #{relative}")
        end
        require_digest!(digest.as_s, "pinned source file")
        path = File.join(source["upstream_path"].as_s, relative)
        raise ArgumentError.new("pinned upstream source hash mismatch: #{relative}") unless sha256_file(canonical_regular_file(path, relative)) == digest.as_s
      end
      cache_index_sha256 = source["cache_index_sha256"].as_s
      unless cache_index_sha256 == CACHE_INDEX_SHA256 &&
             sha256_file(canonical_regular_file(source["cache_index_path"].as_s, "DiT cache index")) == cache_index_sha256
        raise ArgumentError.new("DiT cache index pin mismatch")
      end
      shards = source["block_shards"].as_a
      raise ArgumentError.new("DiT source must bind 54 block shards") unless shards.size == BLOCKS
      bytes = 0_i64
      shards.each_with_index do |shard, index|
        exact_keys!(shard, %w(block path bytes sha256 tensor_count), "DiT block shard")
        raise ArgumentError.new("DiT block shard order mismatch") unless shard["block"].as_i == index
        require_digest!(shard["sha256"].as_s, "DiT block shard")
        raise ArgumentError.new("DiT block shard has no tensors") unless shard["tensor_count"].as_i > 0
        shard_path = canonical_regular_file(shard["path"].as_s, "DiT block shard #{index}")
        info = File.info(shard_path)
        unless info.size == shard["bytes"].as_i64
          raise ArgumentError.new("DiT block shard byte size mismatch: #{index}")
        end
        bytes += info.size
      end
      global = source["global_shard"]
      exact_keys!(global, %w(path bytes sha256 tensor_count), "DiT global shard")
      require_digest!(global["sha256"].as_s, "DiT global shard")
      raise ArgumentError.new("DiT global shard has no tensors") unless global["tensor_count"].as_i > 0
      global_path = canonical_regular_file(global["path"].as_s, "DiT global shard")
      global_info = File.info(global_path)
      unless global_info.size == global["bytes"].as_i64
        raise ArgumentError.new("DiT global shard byte size mismatch")
      end
      unless bytes + global_info.size == source["total_weight_bytes"].as_i64
        raise ArgumentError.new("DiT block shard total byte count mismatch")
      end
    end

    private def validate_source_inputs! : Nil
      source_inputs = @document["source_inputs"]
      exact_keys!(source_inputs,
        %w(frozen_inputs roi continuation accepted_parent_comparison_sha256 tensors),
        "route source input provenance")
      unless source_inputs["accepted_parent_comparison_sha256"].as_s == PARENT_COMPARISON_SHA256
        raise ArgumentError.new("accepted ROI comparison digest mismatch")
      end
      validate_frozen_manifest!(source_inputs["frozen_inputs"])
      roi_payload_path = validate_roi_binding!(source_inputs["roi"])
      continuation_payload_path = validate_continuation_binding!(source_inputs["continuation"])
      validate_input_tensors!(source_inputs["tensors"], roi_payload_path)
      cosine, sine = load_rope_banks(source_inputs["continuation"], continuation_payload_path)
      @rope_cos = cosine
      @rope_sin = sine
    end

    private def validate_frozen_manifest!(binding : JSON::Any) : Nil
      exact_keys!(binding, %w(path sha256), "frozen input manifest binding")
      raise ArgumentError.new("frozen manifest digest mismatch") unless binding["sha256"].as_s == FROZEN_INPUT_SHA256
      path = canonical_regular_file(binding["path"].as_s, "frozen input manifest")
      raise ArgumentError.new("frozen input manifest content mismatch") unless sha256_file(path) == FROZEN_INPUT_SHA256
      frozen = JSON.parse(File.read(path))
      unless frozen["schema"].as_s == "cogni_ml.direct_route_frozen_input_binding/v1" &&
             frozen["parent_source_sha256"].as_s == ROI_MANIFEST_SHA256 &&
             frozen["accepted_parent_comparison_sha256"].as_s == PARENT_COMPARISON_SHA256
        raise ArgumentError.new("frozen input manifest is not bound to the accepted ROI parent")
      end
      @frozen_document = frozen
    end

    private def validate_roi_binding!(binding : JSON::Any) : String
      exact_keys!(binding,
        %w(manifest_path manifest_sha256 payload_path payload_sha256 capture_script_sha256),
        "accepted ROI source binding")
      raise ArgumentError.new("ROI manifest digest mismatch") unless binding["manifest_sha256"].as_s == ROI_MANIFEST_SHA256
      raise ArgumentError.new("ROI payload or capture script pin mismatch") unless binding["payload_sha256"].as_s == ROI_PAYLOAD_SHA256 &&
                                                                                   binding["capture_script_sha256"].as_s == ROI_CAPTURE_SHA256
      manifest_path = canonical_regular_file(binding["manifest_path"].as_s, "accepted ROI manifest")
      payload_path = canonical_regular_file(binding["payload_path"].as_s, "accepted ROI payload")
      raise ArgumentError.new("ROI manifest bytes changed") unless sha256_file(manifest_path) == ROI_MANIFEST_SHA256
      raise ArgumentError.new("ROI payload bytes changed") unless sha256_file(payload_path) == ROI_PAYLOAD_SHA256
      roi = JSON.parse(File.read(manifest_path))
      unless roi["schema"].as_s == "cogni_ml.hy_nomas_structured_reference_control/v2" &&
             roi["source"]["upstream_revision"].as_s == UPSTREAM_REVISION &&
             roi["capture_payload"]["sha256"].as_s == ROI_PAYLOAD_SHA256 &&
             roi["capture_payload"]["capture_script_sha256"].as_s == ROI_CAPTURE_SHA256
        raise ArgumentError.new("accepted ROI parent manifest provenance mismatch")
      end
      if frozen = @frozen_document
        unless File.realpath(frozen["parent_source_manifest"].as_s) == manifest_path
          raise ArgumentError.new("frozen input parent manifest path mismatch")
        end
      else
        raise ArgumentError.new("frozen input manifest was not admitted")
      end
      payload_basename = roi["capture_payload"]["path"].as_s
      unless File.realpath(File.join(File.dirname(manifest_path), payload_basename)) == payload_path
        raise ArgumentError.new("ROI manifest/payload path binding mismatch")
      end
      payload_path
    end

    private def validate_continuation_binding!(binding : JSON::Any) : String
      exact_keys!(binding,
        %w(manifest_path manifest_sha256 payload_path payload_sha256 rope_cos rope_sin),
        "continuation source binding")
      unless binding["manifest_sha256"].as_s == CONTINUATION_SHA256 &&
             binding["payload_sha256"].as_s == CONTINUATION_PAYLOAD_SHA256
        raise ArgumentError.new("continuation manifest or payload digest mismatch")
      end
      manifest_path = canonical_regular_file(binding["manifest_path"].as_s, "continuation manifest")
      payload_path = canonical_regular_file(binding["payload_path"].as_s, "continuation payload")
      raise ArgumentError.new("continuation manifest bytes changed") unless sha256_file(manifest_path) == CONTINUATION_SHA256
      raise ArgumentError.new("continuation payload bytes changed") unless sha256_file(payload_path) == CONTINUATION_PAYLOAD_SHA256
      continuation = JSON.parse(File.read(manifest_path))
      unless continuation["schema"].as_s == "cogni_ml.hy_nomas_continuation_inputs/v1" &&
             continuation["upstream_revision"].as_s == UPSTREAM_REVISION &&
             continuation["payload_sha256"].as_s == CONTINUATION_PAYLOAD_SHA256 &&
             File.realpath(File.join(File.dirname(manifest_path), continuation["payload"].as_s)) == payload_path
        raise ArgumentError.new("continuation parent manifest binding mismatch")
      end
      payload_path
    end

    private def validate_input_tensors!(records : JSON::Any, roi_payload_path : String) : Nil
      exact_keys!(records, INPUT_NAMES, "frozen input tensor inventory")
      roi_manifest = JSON.parse(File.read(@document["source_inputs"]["roi"]["manifest_path"].as_s))
      source_records = roi_manifest["capture_payload"]["records"].as_a
      frozen = @frozen_document || raise ArgumentError.new("frozen input manifest is unavailable")
      frozen_records = frozen["inputs"].as_a
      raise ArgumentError.new("frozen source record inventory mismatch") unless frozen_records.size == INPUT_NAMES.size
      INPUT_NAMES.each do |name|
        record = records[name]
        exact_keys!(record, %w(path bytes sha256 dtype shape source_range_offset source_range_bytes), "frozen input #{name}")
        offset, nbytes, digest = INPUT_BINDINGS[name]
        unless record["dtype"].as_s == "F32" && record["shape"].as_a.map(&.as_i.to_i32) == INPUT_SHAPES[name] &&
               record["source_range_offset"].as_i64 == offset &&
               record["source_range_bytes"].as_i64 == nbytes &&
               record["bytes"].as_i64 == nbytes && record["sha256"].as_s == digest
          raise ArgumentError.new("frozen input record mismatch: #{name}")
        end
        origins = frozen_records.select { |item| item["name"].as_s == name }
        raise ArgumentError.new("frozen input source record missing or duplicated: #{name}") unless origins.size == 1
        origin = origins[0]
        origin_path = canonical_regular_file(origin["native_file"].as_s, "frozen source input #{name}")
        source_file = canonical_regular_file(record["path"].as_s, "frozen capture #{name}")
        unless source_file == origin_path && origin["bytes"].as_i64 == nbytes &&
               origin["source_range_offset"].as_i64 == offset && origin["source_range_bytes"].as_i64 == nbytes &&
               origin["sha256"].as_s == digest && origin["matches_accepted_source_record"].as_bool
          raise ArgumentError.new("frozen input is not bound to its admitted source record: #{name}")
        end
        matches = source_records.select { |item| item["name"].as_s == name }
        raise ArgumentError.new("ROI source capture record missing or duplicated: #{name}") unless matches.size == 1
        source = matches[0]
        unless source["offset"].as_i64 == offset && source["nbytes"].as_i64 == nbytes && source["sha256"].as_s == digest
          raise ArgumentError.new("ROI source tensor range mismatch: #{name}")
        end
        raise ArgumentError.new("ROI source tensor bytes mismatch: #{name}") unless sha256_range(roi_payload_path, offset, nbytes) == digest

        file_path = source_file
        unless File.size(file_path) == nbytes && sha256_file(file_path) == digest
          raise ArgumentError.new("frozen capture digest mismatch: #{name}")
        end
        @frozen_input_paths[name] = file_path
        @inputs[name] = read_f32(file_path, tensor_elements(INPUT_SHAPES[name]).to_i, name)
      end
    end

    private def load_rope_banks(continuation_binding : JSON::Any,
                                continuation_payload_path : String) : Tuple(Array(Float32), Array(Float32))
      cosine = [] of Float32
      sine = [] of Float32
      ROPE_RECORDS.each do |name, record_info|
        offset, record_sha, bank_shas = record_info
        binding_key = name == "input.rope_cos" ? "rope_cos" : "rope_sin"
        rope_binding = continuation_binding[binding_key]
        exact_keys!(rope_binding, %w(record banks), "#{name} source record")
        parent = rope_binding["record"]
        exact_keys!(parent, %w(dtype shape offset nbytes sha256), "#{name} parent tensor")
        unless parent["dtype"].as_s == "F32" && parent["shape"].as_a.map(&.as_i) == [6, VIDEO_ROWS, HEAD_DIM] &&
               parent["offset"].as_i64 == offset && parent["nbytes"].as_i64 == ROPE_RECORD_BYTES &&
               parent["sha256"].as_s == record_sha && sha256_range(continuation_payload_path, offset, ROPE_RECORD_BYTES) == record_sha
          raise ArgumentError.new("#{name} full continuation range mismatch")
        end
        banks = rope_binding["banks"]
        exact_keys!(banks, %w(T0 T1), "#{name} selected RoPE banks")
        {"T0" => 0, "T1" => 1}.each do |bank_name, bank_index|
          bank = banks[bank_name]
          exact_keys!(bank, %w(selected_bank selected_range_offset selected_range_bytes selected_range_sha256), "#{name} #{bank_name}")
          selected_offset = offset + bank_index.to_i64 * BANK_BYTES
          selected_sha = bank_shas[bank_name]
          unless bank["selected_bank"].as_i == bank_index &&
                 bank["selected_range_offset"].as_i64 == selected_offset &&
                 bank["selected_range_bytes"].as_i64 == BANK_BYTES &&
                 bank["selected_range_sha256"].as_s == selected_sha &&
                 sha256_range(continuation_payload_path, selected_offset, BANK_BYTES) == selected_sha
            raise ArgumentError.new("#{name} #{bank_name} bank digest mismatch")
          end
          values = read_f32_range(continuation_payload_path, selected_offset, BANK_BYTES, BANK_ELEMENTS)
          (name == "input.rope_cos" ? cosine : sine).concat(values)
        end
      end
      unless cosine.size == BANK_ELEMENTS * 2 && sine.size == BANK_ELEMENTS * 2
        raise ArgumentError.new("T0/T1 RoPE bank count mismatch")
      end
      {cosine, sine}
    end

    private def validate_mask_proof! : Nil
      proof = @document["mask_proof"]
      exact_keys!(proof,
        %w(shape encoding query_labels key_labels label_rule roi source_range_builder_sha256 cases),
        "route mask proof")
      q, k = HyNomasDiTRouteFactorialProbe.query_key_labels
      unless proof["shape"].as_a.map(&.as_i) == [QUERY_ROWS, KEY_ROWS] &&
             proof["query_labels"].as_a.map(&.as_i) == q &&
             proof["key_labels"].as_a.map(&.as_i) == k &&
             proof["encoding"].as_s == "row-major uint8; 0=masked, 1=allowed" &&
             proof["label_rule"].as_s == "global plus same-action labels; interventions clear only matching action-text edges from positive video queries" &&
             proof["source_range_builder_sha256"].as_s == SOURCE_RANGE_BUILDER_SHA256
        raise ArgumentError.new("route mask geometry/label proof mismatch")
      end
      expected_roi = JSON.parse(%({"action1":{"h":[0,9],"w":[1,8]},"action2":{"h":[2,8],"w":[10,15]}}))
      raise ArgumentError.new("route ROI control mismatch") unless proof["roi"] == expected_roi
      cases = proof["cases"]
      exact_keys!(cases, CASES, "route mask cases")
      CASES.each do |name|
        record = cases[name]
        exact_keys!(record, %w(matrix_sha256 allowed_edges removed_edges_from_M11 route_bits), "route mask #{name}")
        expected_sha, expected_edges, removed_a1, removed_a2 = HyNomasDiTRouteFactorialProbe.factorial_matrix_sha256(name)
        expected_factors = FACTORS[name]
        bits = record["route_bits"]
        exact_keys!(bits, %w(action1_video_to_text action2_video_to_text), "route bits #{name}")
        removals = record["removed_edges_from_M11"]
        exact_keys!(removals, %w(action1 action2), "route edge removals #{name}")
        unless record["matrix_sha256"].as_s == expected_sha && record["allowed_edges"].as_i64 == expected_edges &&
               removals["action1"].as_i64 == removed_a1 && removals["action2"].as_i64 == removed_a2 &&
               bits["action1_video_to_text"].as_bool == expected_factors[:action1_video_to_text] &&
               bits["action2_video_to_text"].as_bool == expected_factors[:action2_video_to_text]
          raise ArgumentError.new("route mask matrix/counter proof mismatch: #{name}")
        end
      end
    end

    private def read_output_payload! : Tuple(String, String)
      payload = @document["payload"]
      exact_keys!(payload, %w(basename bytes sha256 encoding), "source output payload")
      basename = payload["basename"].as_s
      raise ArgumentError.new("unsafe route source payload basename") unless File.basename(basename) == basename
      raise ArgumentError.new("route source payload encoding mismatch") unless payload["encoding"].as_s == "f32le_bf16_expansion"
      digest = payload["sha256"].as_s
      require_digest!(digest, "source output payload")
      file = canonical_regular_file(File.join(File.dirname(@path), basename), "source output payload")
      unless File.size(file) == payload["bytes"].as_i64 && sha256_file(file) == digest
        raise ArgumentError.new("route source output payload size or digest mismatch")
      end
      {file, digest}
    end

    private def read_output_tensors! : Hash(String, TensorRecord)
      tensors = @document["tensors"]
      names = CASES.flat_map { |name| CHECKPOINTS.map { |checkpoint| "#{name}.#{checkpoint}" } }
      exact_keys!(tensors, names, "route output tensors")
      cursor = 0_i64
      result = {} of String => TensorRecord
      CASES.each do |case_name|
        CHECKPOINTS.each do |checkpoint|
          name = "#{case_name}.#{checkpoint}"
          record = tensors[name]
          exact_keys!(record, %w(dtype source_dtype shape offset_bytes nbytes sha256 checkpoint_boundary), name)
          shape = OUTPUT_SHAPES[checkpoint]
          nbytes = tensor_elements(shape) * sizeof(Float32)
          digest = record["sha256"].as_s
          require_digest!(digest, "#{name}")
          unless record["dtype"].as_s == "F32" && record["source_dtype"].as_s == "BF16" &&
                 record["shape"].as_a.map(&.as_i.to_i32) == shape &&
                 record["offset_bytes"].as_i64 == cursor && record["nbytes"].as_i64 == nbytes &&
                 record["checkpoint_boundary"].as_s == checkpoint &&
                 sha256_range(@output_payload_path, cursor, nbytes) == digest
            raise ArgumentError.new("route output tensor metadata or range digest mismatch: #{name}")
          end
          result[name] = TensorRecord.new(name, shape, "F32", "BF16", cursor, nbytes, digest)
          cursor += nbytes
        end
      end
      unless cursor == File.size(@output_payload_path)
        raise ArgumentError.new("route output tensor ranges do not cover source payload")
      end
      result
    end

    private def validate_source_output_invariants! : Nil
      digests = CASES.map do |case_name|
        @output_tensors["#{case_name}.block0.text_valid"].sha256
      end
      unless digests.uniq.size == 1
        raise ArgumentError.new("source block0 valid-text outputs differ across direct video-route masks")
      end
      CHECKPOINTS.each do |checkpoint|
        first = @output_tensors["M11.#{checkpoint}"].sha256
        repeated = @output_tensors["M11_repeat.#{checkpoint}"].sha256
        unless first == repeated
          raise ArgumentError.new("source M11 repeat bytes differ from M11: #{checkpoint}")
        end
      end
      accepted = @inputs["t1.step1.velocity"]
      replayed = expected_values("M11.velocity")
      comparison = HyNomasDiTRouteFactorialProbe.compare_values(accepted, replayed)
      unless comparison.failures == 0
        raise ArgumentError.new("source M11 velocity bytes fail the accepted-parent replay tolerance")
      end
    end

    private def read_f32(path : String, count : Int, label : String) : Array(Float32)
      expected_bytes = count.to_i64 * sizeof(Float32)
      raise ArgumentError.new("#{label} input size mismatch") unless File.size(path) == expected_bytes
      raw = File.read(path).to_slice
      values = Array(Float32).new(count) do |index|
        IO::ByteFormat::LittleEndian.decode(Float32, raw[index * 4, 4])
      end
      values.each do |value|
        raise ArgumentError.new("#{label} contains non-finite input") unless value.finite?
      end
      values
    end

    private def read_f32_range(path : String, offset : Int64, bytes : Int64,
                               count : Int) : Array(Float32)
      raise ArgumentError.new("F32 range size mismatch") unless bytes == count.to_i64 * 4
      raw = Bytes.new(bytes.to_i)
      File.open(path) do |io|
        io.seek(offset)
        io.read_fully(raw)
      end
      Array(Float32).new(count) do |index|
        IO::ByteFormat::LittleEndian.decode(Float32, raw[index * 4, 4])
      end
    end
  end
end
