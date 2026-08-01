# Pure-CPU resource and specialization contract for future TRELLIS.2 device
# execution. This file does not compile kernels, allocate device buffers, or
# admit Metal execution.

module ML::ThreeD::Trellis2
  record DenseActivationRequest,
    profile_id : String,
    batch : Int32,
    voxel_tokens : Int32,
    context_tokens : Int32,
    dtype : ML::DType

  # Explicit semantic identity for one future compiled-kernel family. These
  # fields make collisions inspectable; actual Metal integration must populate
  # them from its source, compiler, and device instead of trusting a nickname.
  class DenseKernelABI
    getter source_digest : String
    getter device_family : String
    getter compiler_abi : String
    getter weight_format : String
    getter accumulation_dtype : ML::DType
    getter activation_mode : String
    getter normalization_mode : String
    getter attention_mode : String
    getter rope_mode : String
    getter mask_mode : String
    getter layout_mode : String

    def initialize(
      @source_digest : String,
      @device_family : String,
      @compiler_abi : String,
      @weight_format : String,
      @accumulation_dtype : ML::DType,
      @activation_mode : String,
      @normalization_mode : String,
      @attention_mode : String,
      @rope_mode : String,
      @mask_mode : String,
      @layout_mode : String,
    )
      unless @source_digest.matches?(/\A[0-9a-f]{64}\z/)
        raise ArgumentError.new("kernel source_digest must be 64 lowercase hexadecimal characters")
      end
      {
        "device_family"      => @device_family,
        "compiler_abi"       => @compiler_abi,
        "weight_format"      => @weight_format,
        "activation_mode"    => @activation_mode,
        "normalization_mode" => @normalization_mode,
        "attention_mode"     => @attention_mode,
        "rope_mode"          => @rope_mode,
        "mask_mode"          => @mask_mode,
        "layout_mode"        => @layout_mode,
      }.each do |name, value|
        unless value.matches?(/\A[a-z0-9][a-z0-9-]{0,63}\z/)
          raise ArgumentError.new("#{name} must be a lowercase ASCII cache token")
        end
      end
      unless @accumulation_dtype.floating?
        raise ArgumentError.new("kernel accumulation dtype must be floating")
      end
    end

    def canonical : String
      "src-#{@source_digest}/dev-#{@device_family}/compiler-#{@compiler_abi}/" \
      "weight-#{@weight_format}/acc-#{@accumulation_dtype.to_s.downcase}/" \
      "act-#{@activation_mode}/norm-#{@normalization_mode}/" \
      "attn-#{@attention_mode}/rope-#{@rope_mode}/mask-#{@mask_mode}/" \
      "layout-#{@layout_mode}"
    end
  end

  # Owner-scoped shape-specialization key for observe-before-compile admission.
  # This value is not a process-global Metal PipelineCache admission by itself.
  record KernelSpecializationKey,
    cache_owner : String,
    kernel_abi : String,
    profile_id : String,
    profile_signature : String,
    kernel_variant : String,
    dtype : ML::DType,
    padding_policy : String,
    padded_batch : Int32,
    padded_voxel_tokens : Int32,
    padded_context_tokens : Int32 do
    def canonical : String
      "trellis2/resource-v1/owner-#{@cache_owner}/#{@kernel_abi}/" \
      "#{@profile_id}/#{@profile_signature}/" \
      "#{@kernel_variant}/#{@dtype.to_s.downcase}/#{@padding_policy}/" \
      "b#{@padded_batch}/n#{@padded_voxel_tokens}/s#{@padded_context_tokens}"
    end
  end

  # One immutable dense-stage geometry profile. Bucket arrays are copied so a
  # caller cannot expand the admitted specialization space after validation.
  class DenseStageResourceProfile
    MAX_BUCKETS_PER_AXIS = 16

    getter id : String
    getter in_channels : Int32
    getter model_channels : Int32
    getter context_channels : Int32
    getter out_channels : Int32
    getter num_heads : Int32
    getter mlp_hidden_channels : Int32
    getter frequency_dim : Int32

    @batch_buckets : Array(Int32)
    @voxel_buckets : Array(Int32)
    @context_buckets : Array(Int32)

    def initialize(
      @id : String,
      @in_channels : Int32,
      @model_channels : Int32,
      @context_channels : Int32,
      @out_channels : Int32,
      @num_heads : Int32,
      @mlp_hidden_channels : Int32,
      @frequency_dim : Int32,
      batch_buckets : Array(Int32),
      voxel_buckets : Array(Int32),
      context_buckets : Array(Int32),
    )
      validate_token!(@id, "profile id")
      {
        "in_channels"         => @in_channels,
        "model_channels"      => @model_channels,
        "context_channels"    => @context_channels,
        "out_channels"        => @out_channels,
        "num_heads"           => @num_heads,
        "mlp_hidden_channels" => @mlp_hidden_channels,
        "frequency_dim"       => @frequency_dim,
      }.each do |name, value|
        raise ArgumentError.new("#{name} must be positive") unless value > 0
      end
      unless @model_channels % @num_heads == 0
        raise ArgumentError.new("num_heads must divide model_channels")
      end
      if @model_channels > Int32::MAX // 6
        raise ArgumentError.new("model_channels makes the 6C modulation width overflow Int32")
      end
      head_dim = @model_channels // @num_heads
      unless head_dim.even? && head_dim // 2 // 3 > 0
        raise ArgumentError.new(
          "head dimension must be even and provide a 3D RoPE frequency per spatial axis"
        )
      end
      unless @frequency_dim >= 2
        raise ArgumentError.new("frequency_dim must be at least 2")
      end

      @batch_buckets = validate_buckets!(batch_buckets, "batch")
      @voxel_buckets = validate_buckets!(voxel_buckets, "voxel token")
      @voxel_buckets.each do |value|
        unless perfect_cube?(value)
          raise ArgumentError.new("voxel token buckets must be exact positive cubes for NCDHW")
        end
      end
      @context_buckets = validate_buckets!(context_buckets, "context token")
    end

    def batch_buckets : Array(Int32)
      @batch_buckets.dup
    end

    def voxel_buckets : Array(Int32)
      @voxel_buckets.dup
    end

    def context_buckets : Array(Int32)
      @context_buckets.dup
    end

    def signature : String
      "ncdhw-cubic-i#{@in_channels}-c#{@model_channels}-k#{@context_channels}-" \
      "o#{@out_channels}-h#{@num_heads}-m#{@mlp_hidden_channels}-f#{@frequency_dim}"
    end

    # Batch values are an exact finite set in T2N2d0. Dummy-batch padding has no
    # admitted masking/lifetime semantics yet.
    def padded_batch(logical : Int32) : Int32
      raise ArgumentError.new("logical batch length must be positive") unless logical > 0
      return logical if @batch_buckets.includes?(logical)
      raise ArgumentError.new("no declared exact batch value matches logical length #{logical}")
    end

    def padded_voxel_tokens(logical : Int32) : Int32
      unless perfect_cube?(logical)
        raise ArgumentError.new("logical voxel token length must be an exact positive cube for NCDHW")
      end
      bucket_for(@voxel_buckets, logical, "voxel token")
    end

    def padded_context_tokens(logical : Int32) : Int32
      bucket_for(@context_buckets, logical, "context token")
    end

    private def validate_token!(value : String, label : String) : Nil
      unless value.matches?(/\A[a-z0-9][a-z0-9-]{0,63}\z/)
        raise ArgumentError.new("#{label} must be a lowercase ASCII cache token")
      end
    end

    private def validate_buckets!(values : Array(Int32), label : String) : Array(Int32)
      unless 1 <= values.size <= MAX_BUCKETS_PER_AXIS
        raise ArgumentError.new(
          "#{label} buckets must contain 1..#{MAX_BUCKETS_PER_AXIS} entries"
        )
      end
      previous = 0_i32
      values.each do |value|
        unless value > previous
          raise ArgumentError.new("#{label} buckets must be positive and strictly increasing")
        end
        previous = value
      end
      values.dup
    end

    private def bucket_for(values : Array(Int32), logical : Int32, label : String) : Int32
      raise ArgumentError.new("logical #{label} length must be positive") unless logical > 0
      values.each { |value| return value if value >= logical }
      raise ArgumentError.new("no declared #{label} bucket can contain logical length #{logical}")
    end

    private def perfect_cube?(value : Int32) : Bool
      return false unless value > 0
      target = value.to_i64
      low = 1_i64
      high = 1291_i64
      while low <= high
        root = low + (high - low) // 2_i64
        cube = root * root * root
        return true if cube == target
        if cube < target
          low = root + 1_i64
        else
          high = root - 1_i64
        end
      end
      false
    end
  end

  # A metadata-only plan. `declared_tensor_bytes` is a deterministic inventory
  # upper sum for the tensors named by this contract, not a measured allocator,
  # command-buffer, trace-retention, or Metal peak-memory value.
  class DenseActivationPlan
    getter padding_policy : String
    getter logical_batch : Int32
    getter logical_voxel_tokens : Int32
    getter logical_context_tokens : Int32
    getter padded_batch : Int32
    getter padded_voxel_tokens : Int32
    getter padded_context_tokens : Int32
    getter max_single_tensor_bytes : Int64
    getter declared_activation_bytes : Int64

    @kernel_keys : Array(KernelSpecializationKey)

    def initialize(
      @logical_batch : Int32,
      @logical_voxel_tokens : Int32,
      @logical_context_tokens : Int32,
      @padded_batch : Int32,
      @padded_voxel_tokens : Int32,
      @padded_context_tokens : Int32,
      @padding_policy : String,
      kernel_keys : Array(KernelSpecializationKey),
      declared_tensor_bytes : Hash(String, Int64),
    )
      @kernel_keys = kernel_keys.dup
      @declared_tensor_bytes = declared_tensor_bytes.dup
      @max_single_tensor_bytes = @declared_tensor_bytes.values.max
      @declared_activation_bytes = @declared_tensor_bytes.values.sum
    end

    def kernel_keys : Array(KernelSpecializationKey)
      @kernel_keys.dup
    end

    def declared_tensor_bytes : Hash(String, Int64)
      @declared_tensor_bytes.dup
    end

    def requires_attention_mask? : Bool
      @logical_voxel_tokens != @padded_voxel_tokens ||
        @logical_context_tokens != @padded_context_tokens
    end

    def requires_output_trim? : Bool
      @logical_voxel_tokens != @padded_voxel_tokens
    end

    def rope_padding_phase : {Float64, Float64}
      {1.0, 0.0}
    end
  end

  class DenseDeviceResourceContract
    # This string is an execution obligation, not evidence that current
    # attention kernels implement masking or trimming.
    PADDING_POLICY              = "right-zero-mask-trim-v1"
    REQUIRED_MASK_MODE          = "right-valid-trim"
    REQUIRED_ROPE_MODE          = "realpair-3d"
    REQUIRED_LAYOUT_MODE        = "ncdhw-cubic"
    MAX_PROFILES                =   16
    MAX_KERNEL_VARIANTS         =   64
    MAX_DTYPES                  =    3
    MAX_THEORETICAL_KERNEL_KEYS = 4096
    MAX_AXIS_PADDING_RATIO      = 16.0

    getter max_axis_padding_ratio : Float64
    getter cache_owner : String
    getter kernel_abi : DenseKernelABI
    getter max_single_tensor_bytes : Int64
    getter max_declared_activation_bytes : Int64
    getter theoretical_kernel_key_count : Int32

    @profiles : Array(DenseStageResourceProfile)
    @kernel_variants : Array(String)
    @dtypes : Array(ML::DType)
    @kernel_abi : DenseKernelABI
    @kernel_abi_signature : String

    def initialize(
      profiles : Array(DenseStageResourceProfile),
      kernel_variants : Array(String),
      dtypes : Array(ML::DType),
      @cache_owner : String,
      @kernel_abi : DenseKernelABI,
      @max_axis_padding_ratio : Float64,
      @max_single_tensor_bytes : Int64,
      @max_declared_activation_bytes : Int64,
    )
      unless 1 <= profiles.size <= MAX_PROFILES
        raise ArgumentError.new("dense resource contract must declare 1..#{MAX_PROFILES} profiles")
      end
      unless @cache_owner.matches?(/\A[a-z0-9][a-z0-9-]{0,63}\z/)
        raise ArgumentError.new("cache_owner must be a lowercase ASCII cache token")
      end
      unless @kernel_abi.mask_mode == REQUIRED_MASK_MODE
        raise ArgumentError.new("kernel ABI mask_mode must be #{REQUIRED_MASK_MODE}")
      end
      unless @kernel_abi.rope_mode == REQUIRED_ROPE_MODE
        raise ArgumentError.new("kernel ABI rope_mode must be #{REQUIRED_ROPE_MODE}")
      end
      unless @kernel_abi.layout_mode == REQUIRED_LAYOUT_MODE
        raise ArgumentError.new("kernel ABI layout_mode must be #{REQUIRED_LAYOUT_MODE}")
      end
      @kernel_abi_signature = @kernel_abi.canonical
      unless 1 <= kernel_variants.size <= MAX_KERNEL_VARIANTS
        raise ArgumentError.new(
          "dense resource contract must declare 1..#{MAX_KERNEL_VARIANTS} kernel variants"
        )
      end
      unless 1 <= dtypes.size <= MAX_DTYPES
        raise ArgumentError.new("dense resource contract must declare 1..#{MAX_DTYPES} dtypes")
      end
      unless @max_axis_padding_ratio.finite? &&
             1.0 <= @max_axis_padding_ratio <= MAX_AXIS_PADDING_RATIO
        raise ArgumentError.new(
          "max_axis_padding_ratio must be finite and within 1..#{MAX_AXIS_PADDING_RATIO}"
        )
      end
      unless @max_single_tensor_bytes > 0
        raise ArgumentError.new("max_single_tensor_bytes must be positive")
      end
      unless @max_declared_activation_bytes >= @max_single_tensor_bytes
        raise ArgumentError.new(
          "max_declared_activation_bytes must be at least max_single_tensor_bytes"
        )
      end

      @profiles = profiles.dup
      profile_ids = @profiles.map(&.id)
      if profile_ids.uniq.size != profile_ids.size
        raise ArgumentError.new("duplicate profile id in dense resource contract")
      end

      @kernel_variants = kernel_variants.dup
      @kernel_variants.each do |variant|
        unless variant.matches?(/\A[a-z0-9][a-z0-9-]{0,63}\z/)
          raise ArgumentError.new("kernel variant must be a lowercase ASCII cache token")
        end
      end
      if @kernel_variants.uniq.size != @kernel_variants.size
        raise ArgumentError.new("duplicate kernel variant in dense resource contract")
      end

      @dtypes = dtypes.dup
      unless @dtypes.all?(&.floating?)
        raise ArgumentError.new("dense resource contract dtypes must be floating")
      end
      if @dtypes.uniq.size != @dtypes.size
        raise ArgumentError.new("duplicate dtype in dense resource contract")
      end

      count = 0_i64
      @profiles.each do |profile|
        profile_shapes = checked_product!(
          [
            profile.batch_buckets.size.to_i64,
            profile.voxel_buckets.size.to_i64,
            profile.context_buckets.size.to_i64,
            @kernel_variants.size.to_i64,
            @dtypes.size.to_i64,
          ],
          MAX_THEORETICAL_KERNEL_KEYS.to_i64,
          "theoretical kernel key count"
        )
        if count > MAX_THEORETICAL_KERNEL_KEYS - profile_shapes
          raise ArgumentError.new(
            "theoretical kernel key count exceeds #{MAX_THEORETICAL_KERNEL_KEYS}"
          )
        end
        count += profile_shapes
      end
      @theoretical_kernel_key_count = count.to_i32
    end

    def plan(request : DenseActivationRequest) : DenseActivationPlan
      profile = @profiles.find { |candidate| candidate.id == request.profile_id }
      unless profile
        raise ArgumentError.new("unknown dense resource profile #{request.profile_id.inspect}")
      end
      unless @dtypes.includes?(request.dtype)
        raise ArgumentError.new("unsupported dtype #{request.dtype} for dense resource contract")
      end

      padded_batch = profile.padded_batch(request.batch)
      padded_voxels = profile.padded_voxel_tokens(request.voxel_tokens)
      padded_context = profile.padded_context_tokens(request.context_tokens)
      check_padding_ratio!(request.voxel_tokens, padded_voxels, "voxel token")
      check_padding_ratio!(request.context_tokens, padded_context, "context token")

      keys = @kernel_variants.map do |variant|
        KernelSpecializationKey.new(
          @cache_owner,
          @kernel_abi_signature,
          profile.id,
          profile.signature,
          variant,
          request.dtype,
          PADDING_POLICY,
          padded_batch,
          padded_voxels,
          padded_context
        )
      end
      inventory = declared_tensor_inventory(
        profile,
        request.dtype,
        padded_batch,
        padded_voxels,
        padded_context
      )
      DenseActivationPlan.new(
        request.batch,
        request.voxel_tokens,
        request.context_tokens,
        padded_batch,
        padded_voxels,
        padded_context,
        PADDING_POLICY,
        keys,
        inventory
      )
    end

    def all_kernel_keys : Array(KernelSpecializationKey)
      result = Array(KernelSpecializationKey).new(@theoretical_kernel_key_count)
      @profiles.each do |profile|
        @kernel_variants.each do |variant|
          @dtypes.each do |dtype|
            profile.batch_buckets.each do |batch|
              profile.voxel_buckets.each do |voxels|
                profile.context_buckets.each do |context|
                  result << KernelSpecializationKey.new(
                    @cache_owner,
                    @kernel_abi_signature,
                    profile.id,
                    profile.signature,
                    variant,
                    dtype,
                    PADDING_POLICY,
                    batch,
                    voxels,
                    context
                  )
                end
              end
            end
          end
        end
      end
      result
    end

    def allows?(key : KernelSpecializationKey) : Bool
      profile = @profiles.find { |candidate| candidate.id == key.profile_id }
      return false unless profile
      declared = key.cache_owner == @cache_owner &&
                 key.kernel_abi == @kernel_abi_signature &&
                 key.profile_signature == profile.signature &&
                 key.padding_policy == PADDING_POLICY &&
                 @kernel_variants.includes?(key.kernel_variant) &&
                 @dtypes.includes?(key.dtype) &&
                 profile.batch_buckets.includes?(key.padded_batch) &&
                 profile.voxel_buckets.includes?(key.padded_voxel_tokens) &&
                 profile.context_buckets.includes?(key.padded_context_tokens)
      return false unless declared

      begin
        declared_tensor_inventory(
          profile,
          key.dtype,
          key.padded_batch,
          key.padded_voxel_tokens,
          key.padded_context_tokens
        )
        true
      rescue ArgumentError
        false
      end
    end

    private def check_padding_ratio!(logical : Int32, padded : Int32, label : String) : Nil
      ratio = padded.to_f64 / logical.to_f64
      if ratio > @max_axis_padding_ratio
        raise ArgumentError.new(
          "#{label} padding ratio #{ratio} exceeds #{@max_axis_padding_ratio}"
        )
      end
    end

    private def declared_tensor_inventory(
      profile : DenseStageResourceProfile,
      dtype : ML::DType,
      batch : Int32,
      voxels : Int32,
      context : Int32,
    ) : Hash(String, Int64)
      b = batch.to_i64
      n = voxels.to_i64
      s = context.to_i64
      i = profile.in_channels.to_i64
      c = profile.model_channels.to_i64
      k = profile.context_channels.to_i64
      o = profile.out_channels.to_i64
      h = profile.num_heads.to_i64
      m = profile.mlp_hidden_channels.to_i64
      f = profile.frequency_dim.to_i64
      head_dim = c // h

      factors = {
        "voxel input"            => [b, i, n],
        "flattened input"        => [b, n, i],
        "projected tokens"       => [b, n, c],
        "context"                => [b, s, k],
        "timestep frequency"     => [b, f],
        "timestep embedding"     => [b, c],
        "shared modulation"      => [b, 6_i64, c],
        "coordinates"            => [n, 3_i64],
        "rotary phases"          => [n, head_dim],
        "self attention qkv"     => [b, n, 3_i64, c],
        "self attention scores"  => [b, h, n, n],
        "cross attention kv"     => [b, s, 2_i64, c],
        "cross attention scores" => [b, h, n, s],
        "mlp hidden"             => [b, n, m],
        "output tokens"          => [b, n, o],
        "output"                 => [b, o, n],
      }
      result = Hash(String, Int64).new
      total = 0_i64
      factors.each do |name, dimensions|
        elements = checked_product!(dimensions, Int64::MAX, name)
        # Metadata plans conservatively size every declared tensor with the
        # wider of activation storage and accumulation precision. Per-kernel
        # workspace liveness remains outside this no-execution contract.
        byte_size = Math.max(dtype.byte_size, @kernel_abi.accumulation_dtype.byte_size).to_i64
        if elements > @max_single_tensor_bytes // byte_size
          raise ArgumentError.new(
            "#{name} exceeds single tensor budget #{@max_single_tensor_bytes} bytes"
          )
        end
        bytes = elements * byte_size
        if total > @max_declared_activation_bytes - bytes
          raise ArgumentError.new(
            "declared activation budget #{@max_declared_activation_bytes} bytes exceeded at #{name}"
          )
        end
        result[name] = bytes
        total += bytes
      end
      result
    end

    private def checked_product!(factors : Array(Int64), limit : Int64, label : String) : Int64
      product = 1_i64
      factors.each do |factor|
        raise ArgumentError.new("#{label} factor must be positive") unless factor > 0
        if product > limit // factor
          raise ArgumentError.new("#{label} exceeds #{limit}")
        end
        product *= factor
      end
      product
    end
  end

  class KernelCacheCapacityError < Exception
  end

  # CPU-only admission ledger for a future owner-scoped compiled-kernel cache.
  # It deliberately refuses instead of evicting: no lifetime, in-flight command,
  # or device allocation semantics are admitted by T2N2d0. All ledgers for one
  # process owner share the synchronized entry below, so a second ledger cannot
  # reset or bypass the first ledger's key capacity.
  class BoundedKernelKeyLedger
    MAX_PROCESS_OWNERS      =   64
    MAX_PROCESS_KERNEL_KEYS = 4096

    getter capacity : Int32

    private class OwnerEntry
      getter capacity : Int32
      getter keys : Array(KernelSpecializationKey)

      def initialize(@capacity : Int32)
        @keys = [] of KernelSpecializationKey
      end
    end

    # Crystal's Mutex#synchronize is the repository's existing process-wide
    # coordination primitive (Qwen35NativeRuntime, Qwen35Metal caches, and
    # ML::LLM backend all use this shape). The entry map is intentionally keyed
    # only by cache_owner: ABI/profile/shape remain structured fields of each
    # KernelSpecializationKey and therefore count toward one owner-wide bound.
    @@owner_mutex = Mutex.new
    @@owner_entries = {} of String => OwnerEntry
    @@process_kernel_key_count = 0_i32

    def initialize(
      @contract : DenseDeviceResourceContract,
      @capacity : Int32,
    )
      unless 0 < @capacity <= @contract.theoretical_kernel_key_count
        raise ArgumentError.new(
          "kernel ledger capacity must be within 1..#{@contract.theoretical_kernel_key_count}"
        )
      end

      # Retain the immutable, GC-owned owner value for the ledger lifetime. The
      # contract remains the authority for each structured key at admission.
      @owner = @contract.cache_owner.dup
      # This cache is a contract-local admission certificate, not a second
      # capacity authority. Entries are added only after this ledger's contract
      # validates the key and the owner-wide registry admits it.
      @contract_keys = {} of KernelSpecializationKey => KernelSpecializationKey
      @@owner_mutex.synchronize do
        if entry = @@owner_entries[@owner]?
          unless entry.capacity == @capacity
            raise KernelCacheCapacityError.new(
              "kernel cache owner #{@owner} already fixed capacity #{entry.capacity}; " \
              "requested #{@capacity}"
            )
          end
        else
          if @@owner_entries.size >= MAX_PROCESS_OWNERS
            raise KernelCacheCapacityError.new(
              "process kernel owner capacity #{MAX_PROCESS_OWNERS} would be exceeded"
            )
          end
          @@owner_entries[@owner] = OwnerEntry.new(@capacity)
        end
      end
    end

    # Admit one key and return the stable, GC-owned record retained by the
    # ledger. Normal Crystal Strings are immutable references, so retaining the
    # record is safer and cheaper than retaining non-owning byte slices.
    def admit!(key : KernelSpecializationKey) : KernelSpecializationKey
      if admitted = @@owner_mutex.synchronize { @contract_keys[key]? }
        return admitted
      end

      snapshot = snapshot_key(key)
      unless @contract.allows?(snapshot)
        raise ArgumentError.new(
          "kernel key #{snapshot.canonical} is not declared by this contract"
        )
      end

      # Validation may allocate and is deliberately outside the process-wide
      # lock. The second local lookup closes the concurrent-miss race.
      @@owner_mutex.synchronize do
        if admitted = @contract_keys[snapshot]?
          admitted
        else
          entry = @@owner_entries[@owner]
          unless snapshot.cache_owner == @owner
            raise ArgumentError.new(
              "kernel key #{snapshot.canonical} is not owned by ledger #{@owner}"
            )
          end

          canonical = entry.keys.find { |candidate| candidate == snapshot }
          unless canonical
            if @@process_kernel_key_count >= MAX_PROCESS_KERNEL_KEYS
              raise KernelCacheCapacityError.new(
                "process kernel key capacity #{MAX_PROCESS_KERNEL_KEYS} would be exceeded; " \
                "no keys admitted"
              )
            end
            if entry.keys.size >= entry.capacity
              raise KernelCacheCapacityError.new(
                "kernel cache capacity #{entry.capacity} for owner #{@owner} would be exceeded; " \
                "no keys admitted"
              )
            end
            entry.keys << snapshot
            @@process_kernel_key_count += 1
            canonical = snapshot
          end

          @contract_keys[canonical] = canonical
          canonical
        end
      end
    end

    def admit!(keys : Array(KernelSpecializationKey)) : Nil
      # Snapshot the value record before validation. Its String references are
      # immutable and GC-owned; no non-owning byte view crosses this boundary.
      snapshots = keys.map { |key| snapshot_key(key) }
      snapshots.each do |key|
        unless @contract.allows?(key)
          raise ArgumentError.new("kernel key #{key.canonical} is not declared by this contract")
        end
      end

      # Validation above has no shared-state side effects. The dedupe, capacity
      # check, and append are one critical section so concurrent ledgers either
      # admit the whole batch or admit nothing.
      unique = snapshots.uniq
      @@owner_mutex.synchronize do
        entry = @@owner_entries[@owner]
        unique.each do |key|
          unless key.cache_owner == @owner
            raise ArgumentError.new(
              "kernel key #{key.canonical} is not owned by ledger #{@owner}"
            )
          end
        end
        unseen = unique.reject { |key| entry.keys.includes?(key) }
        if @@process_kernel_key_count > MAX_PROCESS_KERNEL_KEYS - unseen.size
          raise KernelCacheCapacityError.new(
            "process kernel key capacity #{MAX_PROCESS_KERNEL_KEYS} would be exceeded; " \
            "no keys admitted"
          )
        end
        if entry.keys.size + unseen.size > entry.capacity
          raise KernelCacheCapacityError.new(
            "kernel cache capacity #{entry.capacity} for owner #{@owner} would be exceeded; " \
            "no keys admitted"
          )
        end
        entry.keys.concat(unseen)
        @@process_kernel_key_count += unseen.size
      end
    end

    def size : Int32
      @@owner_mutex.synchronize do
        @@owner_entries[@owner].keys.size.to_i32
      end
    end

    def keys : Array(KernelSpecializationKey)
      @@owner_mutex.synchronize { @@owner_entries[@owner].keys.dup }
    end

    private def snapshot_key(key : KernelSpecializationKey) : KernelSpecializationKey
      KernelSpecializationKey.new(
        key.cache_owner.dup,
        key.kernel_abi.dup,
        key.profile_id.dup,
        key.profile_signature.dup,
        key.kernel_variant.dup,
        key.dtype,
        key.padding_policy.dup,
        key.padded_batch,
        key.padded_voxel_tokens,
        key.padded_context_tokens
      )
    end
  end
end
