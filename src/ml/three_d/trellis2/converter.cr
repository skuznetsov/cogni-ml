require "digest/sha256"
require "file_utils"
require "json"
require "set"

require "./conversion_plan"
require "./inventory"
require "./layout"
require "./manifest"

{% if flag?(:darwin) %}
  lib LibC
    fun renameatx_np(
      from_fd : Int32,
      from_path : UInt8*,
      to_fd : Int32,
      to_path : UInt8*,
      flags : UInt32,
    ) : Int32
  end
{% end %}

module ML::ThreeD::Trellis2
  class ConverterError < Exception
  end

  struct ConvertedTensor
    getter tensor : RawTensor
    getter layout : String

    def initialize(@tensor, @layout)
    end
  end

  class Converter
    MAX_SOURCE_FILE_BYTES = 64_i64 * 1024_i64 * 1024_i64
    CONVERTER_CONFIG_PATH = "conversion/plan.json"
    STAGE_CONFIG_VERSION  =          1_i32
    RENAME_EXCL           = 0x00000004_u32

    def self.convert!(
      source_root : String,
      output_root : String,
      plan_relative_path : String = "conversion.json",
      before_publish : Proc(String, Nil)? = nil,
    ) : Manifest
      source = validated_directory(source_root, "source root")
      plan_path = safe_child_path(
        source,
        safe_relative_path(plan_relative_path, "plan path"),
        "conversion plan"
      )
      plan = ConversionPlan.load(plan_path)
      requested_output = validated_output_path(output_root)
      parent = validated_directory(
        File.dirname(requested_output),
        "output parent"
      )
      output = File.join(parent, File.basename(requested_output))
      if output == source || output.starts_with?("#{source}/")
        raise ConverterError.new(
          "output path must be outside the immutable source root"
        )
      end
      if File.info?(output, follow_symlinks: false)
        raise ConverterError.new(
          "output path #{output.inspect} already exists"
        )
      end

      temporary = File.tempname(
        ".#{File.basename(output)}.tmp-",
        nil,
        dir: parent
      )
      Dir.mkdir(temporary)
      published = false
      begin
        build_pack!(source, temporary, plan)
        Manifest.load(temporary)
        before_publish.try(&.call(temporary))
        Manifest.load(temporary)
        verify_source_file!(source, plan.source_file)
        publish_exclusive!(temporary, output)
        published = true
        Manifest.load(output)
      ensure
        if !published && File.exists?(temporary)
          FileUtils.rm_rf(temporary)
        end
      end
    rescue ex : ConverterError
      raise ex
    rescue ex : ConversionPlanError | LayoutError | InventoryError | ManifestError
      raise ConverterError.new(ex.message)
    rescue ex : File::Error | IO::Error
      raise ConverterError.new("conversion failed: #{ex.message}")
    end

    private def self.build_pack!(
      source_root : String,
      pack_root : String,
      plan : ConversionPlan,
    ) : Nil
      if {plan.output_file, plan.config_path}.includes?(CONVERTER_CONFIG_PATH)
        raise ConverterError.new(
          "pack paths must not collide with #{CONVERTER_CONFIG_PATH.inspect}"
        )
      end

      source_path = verify_source_file!(source_root, plan.source_file)
      inventory = SafetensorsInventory.read(source_path)
      if inventory.data_length > MAX_SOURCE_FILE_BYTES
        raise ConverterError.new(
          "source tensor payload exceeds T2N1 limit #{MAX_SOURCE_FILE_BYTES}"
        )
      end

      expected_names = plan.operations.flat_map(&.sources).to_set
      actual_names = inventory.tensors.map(&.name).to_set
      missing = (expected_names - actual_names).to_a.sort
      unconsumed = (actual_names - expected_names).to_a.sort
      unless missing.empty? && unconsumed.empty?
        messages = [] of String
        messages << "missing source tensors #{missing.join(", ")}" unless missing.empty?
        messages << "unconsumed source tensors #{unconsumed.join(", ")}" unless unconsumed.empty?
        raise ConverterError.new(messages.join("; "))
      end

      sources = read_source_tensors(source_path, inventory)
      converted = execute_operations(plan.operations, sources)
      if converted.sum { |entry| entry.tensor.bytes.size.to_i64 } >
           RawTensor::MAX_BYTES
        raise ConverterError.new(
          "converted payload exceeds T2N1 limit #{RawTensor::MAX_BYTES}"
        )
      end

      weights_path = pack_path(pack_root, plan.output_file)
      offsets = write_safetensors(weights_path, converted)
      converter_config_path = pack_path(
        pack_root,
        CONVERTER_CONFIG_PATH
      )
      write_text_file(converter_config_path, plan.canonical_json)
      stage_config = canonical_stage_config(plan.stage_id)
      stage_config_path = pack_path(pack_root, plan.config_path)
      write_text_file(stage_config_path, stage_config)

      weights_size = File.size(weights_path)
      converter_config_size = File.size(converter_config_path)
      stage_config_size = File.size(stage_config_path)
      weights_sha = file_sha256(weights_path)
      converter_config_sha = file_sha256(converter_config_path)
      stage_config_sha = file_sha256(stage_config_path)
      manifest = build_manifest_json(
        plan,
        converted,
        offsets,
        weights_size,
        converter_config_size,
        stage_config_size,
        weights_sha,
        converter_config_sha,
        stage_config_sha,
        "0" * 64
      )
      pack_id = Manifest.compute_pack_id_for_draft(manifest)
      sealed = build_manifest_json(
        plan,
        converted,
        offsets,
        weights_size,
        converter_config_size,
        stage_config_size,
        weights_sha,
        converter_config_sha,
        stage_config_sha,
        pack_id
      )
      write_text_file(File.join(pack_root, "manifest.json"), sealed)
      verify_source_file!(source_root, plan.source_file)
    end

    private def self.execute_operations(
      operations : Array(ConversionOperation),
      sources : Hash(String, RawTensor),
    ) : Array(ConvertedTensor)
      result = [] of ConvertedTensor
      operations.sort_by(&.id).each do |operation|
        case operation.kind
        when "identity"
          result << ConvertedTensor.new(
            Layout.identity(
              sources[operation.sources.first],
              operation.destinations.first
            ),
            operation.kind
          )
        when "transpose"
          result << ConvertedTensor.new(
            Layout.transpose(
              sources[operation.sources.first],
              operation.destinations.first,
              operation.axes[0],
              operation.axes[1]
            ),
            operation.kind
          )
        when "permute"
          result << ConvertedTensor.new(
            Layout.permute(
              sources[operation.sources.first],
              operation.destinations.first,
              operation.axes
            ),
            operation.kind
          )
        when "split"
          Layout.split(
            sources[operation.sources.first],
            operation.destinations,
            operation.axis.not_nil!,
            operation.sizes
          ).each do |tensor|
            result << ConvertedTensor.new(tensor, operation.kind)
          end
        when "concat"
          result << ConvertedTensor.new(
            Layout.concat(
              operation.sources.map { |name| sources[name] },
              operation.destinations.first,
              operation.axis.not_nil!
            ),
            operation.kind
          )
        else
          raise ConverterError.new(
            "unsupported operation #{operation.kind.inspect}"
          )
        end
      end
      result.sort_by!(&.tensor.name)
      result
    end

    private def self.read_source_tensors(
      path : String,
      inventory : SafetensorsInventory,
    ) : Hash(String, RawTensor)
      result = {} of String => RawTensor
      File.open(path, "rb") do |io|
        inventory.tensors.each do |tensor|
          if tensor.data_bytes > RawTensor::MAX_BYTES
            raise ConverterError.new(
              "source tensor #{tensor.name.inspect} exceeds T2N1 byte limit"
            )
          end
          bytes = Bytes.new(tensor.data_bytes.to_i)
          io.seek(inventory.data_offset + tensor.data_start)
          io.read_fully(bytes)
          result[tensor.name] = RawTensor.new(
            tensor.name,
            tensor.dtype,
            tensor.shape,
            bytes
          )
        end
      end
      result
    end

    private def self.write_safetensors(
      path : String,
      tensors : Array(ConvertedTensor),
    ) : Hash(String, {Int64, Int64})
      ordered = tensors.sort_by(&.tensor.name)
      offsets = {} of String => {Int64, Int64}
      offset = 0_i64
      ordered.each do |entry|
        next_offset = offset + entry.tensor.bytes.size
        offsets[entry.tensor.name] = {offset, next_offset}
        offset = next_offset
      end

      header = JSON.build do |json|
        json.object do
          ordered.each do |entry|
            tensor = entry.tensor
            start_offset, end_offset = offsets[tensor.name]
            json.field tensor.name do
              json.object do
                json.field "dtype", tensor.dtype
                json.field "shape" do
                  json.array do
                    tensor.shape.each { |dimension| json.number(dimension) }
                  end
                end
                json.field "data_offsets" do
                  json.array do
                    json.number start_offset
                    json.number end_offset
                  end
                end
              end
            end
          end
        end
      end
      padding = (8 - header.bytesize % 8) % 8
      padded_header = header + " " * padding

      Dir.mkdir_p(File.dirname(path))
      File.open(path, "wb") do |io|
        io.write_bytes(
          padded_header.bytesize.to_u64,
          IO::ByteFormat::LittleEndian
        )
        io.write(padded_header.to_slice)
        ordered.each { |entry| io.write(entry.tensor.bytes) }
        io.flush
      end
      offsets
    end

    private def self.canonical_stage_config(stage_id : String) : String
      JSON.build do |json|
        json.object do
          json.field "schema_version", STAGE_CONFIG_VERSION
          json.field "stage_id", stage_id
          json.field "runtime_admitted", false
        end
      end
    end

    private def self.build_manifest_json(
      plan : ConversionPlan,
      tensors : Array(ConvertedTensor),
      offsets : Hash(String, {Int64, Int64}),
      weights_size : Int64,
      converter_config_size : Int64,
      stage_config_size : Int64,
      weights_sha : String,
      converter_config_sha : String,
      stage_config_sha : String,
      pack_id : String,
    ) : String
      weights_path = plan.output_file
      converter_config_path = CONVERTER_CONFIG_PATH
      JSON.build do |json|
        json.object do
          json.field "schema_version", 2
          json.field "converter_version", plan.converter_version
          json.field "converter_config_sha256", converter_config_sha
          json.field "converter_config_path", converter_config_path
          write_identity(json, "source", plan.source)
          write_identity(json, "model", plan.model)
          json.field "external_dependencies" do
            json.array { }
          end
          json.field "execution_order" do
            json.array { json.string(plan.stage_id) }
          end
          json.field "files" do
            json.array do
              write_file_record(
                json,
                weights_path,
                "weights",
                weights_size,
                weights_sha,
                plan.source_file.license
              )
              write_file_record(
                json,
                plan.config_path,
                "config",
                stage_config_size,
                stage_config_sha,
                plan.converter_license
              )
              write_file_record(
                json,
                converter_config_path,
                "converter_config",
                converter_config_size,
                converter_config_sha,
                plan.converter_license
              )
            end
          end
          json.field "stages" do
            json.array do
              json.object do
                json.field "id", plan.stage_id
                json.field "order", 0
                json.field "config_path", plan.config_path
                json.field "config_sha256", stage_config_sha
                json.field "tensors" do
                  json.array do
                    tensors.sort_by(&.tensor.name).each do |entry|
                      tensor = entry.tensor
                      start_offset, end_offset = offsets[tensor.name]
                      json.object do
                        json.field "source_name", tensor.name
                        json.field(
                          "destination_name",
                          "#{plan.stage_id}.#{tensor.name}"
                        )
                        json.field "file", weights_path
                        json.field "dtype", tensor.dtype
                        json.field "shape" do
                          json.array do
                            tensor.shape.each do |dimension|
                              json.number(dimension)
                            end
                          end
                        end
                        json.field "data_offsets" do
                          json.array do
                            json.number(start_offset)
                            json.number(end_offset)
                          end
                        end
                        json.field "byte_order", "little"
                        json.field "layout", entry.layout
                      end
                    end
                  end
                end
              end
            end
          end
          json.field "pack_id", pack_id
        end
      end
    end

    private def self.write_identity(
      json : JSON::Builder,
      name : String,
      identity : ConversionIdentity,
    ) : Nil
      json.field name do
        json.object do
          json.field "repository", identity.repository
          json.field "revision", identity.revision
          json.field "license", identity.license
        end
      end
    end

    private def self.write_file_record(
      json : JSON::Builder,
      path : String,
      role : String,
      byte_length : Int64,
      sha256 : String,
      license : String,
    ) : Nil
      json.object do
        json.field "path", path
        json.field "role", role
        json.field "byte_length", byte_length
        json.field "sha256", sha256
        json.field "license", license
      end
    end

    private def self.verify_source_file!(
      source_root : String,
      source_file : ConversionSourceFile,
    ) : String
      path = safe_child_path(
        source_root,
        source_file.path,
        "source file"
      )
      info = File.info(path, follow_symlinks: false)
      unless info.file?
        raise ConverterError.new("source file must be a regular file")
      end
      if info.size > MAX_SOURCE_FILE_BYTES
        raise ConverterError.new(
          "source file exceeds T2N1 limit #{MAX_SOURCE_FILE_BYTES}"
        )
      end
      unless info.size == source_file.byte_length
        raise ConverterError.new(
          "source file byte length mismatch"
        )
      end
      unless file_sha256(path) == source_file.sha256
        raise ConverterError.new("source file SHA-256 mismatch")
      end
      path
    rescue ex : File::Error
      raise ConverterError.new("cannot inspect source file: #{ex.message}")
    end

    private def self.validated_output_path(output_root : String) : String
      output = Path.new(File.expand_path(output_root)).normalize.to_s
      basename = File.basename(output)
      if basename.empty? || basename == "." || basename == ".."
        raise ConverterError.new("output path is invalid")
      end
      output
    end

    private def self.validated_directory(
      path : String,
      context : String,
    ) : String
      normalized = Path.new(File.expand_path(path)).normalize.to_s
      reject_symlink_components!(normalized, context)
      info = File.info(normalized, follow_symlinks: false)
      unless info.directory?
        raise ConverterError.new("#{context} must be a directory")
      end
      File.realpath(normalized)
    rescue ex : ConverterError
      raise ex
    rescue ex : File::Error
      raise ConverterError.new("cannot inspect #{context}: #{ex.message}")
    end

    private def self.safe_child_path(
      root : String,
      relative : String,
      context : String,
    ) : String
      path = root
      relative.split('/').each do |component|
        path = File.join(path, component)
        reject_symlink_components!(path, context)
      end
      path
    end

    private def self.pack_path(root : String, relative : String) : String
      File.join(root, relative)
    end

    private def self.safe_relative_path(
      value : String,
      context : String,
    ) : String
      components = value.split('/', remove_empty: false)
      unsafe = value.empty? ||
               value.starts_with?('/') ||
               value.includes?('\\') ||
               value.includes?('\0') ||
               value.includes?(':') ||
               components.any? do |component|
                 component.empty? || component == "." || component == ".."
               end
      if unsafe
        raise ConverterError.new(
          "unsafe #{context} #{value.inspect}"
        )
      end
      value
    end

    private def self.reject_symlink_components!(
      path : String,
      context : String,
    ) : Nil
      normalized = Path.new(File.expand_path(path)).normalize.to_s
      current = "/"
      normalized.split('/', remove_empty: true).each do |component|
        current = File.join(current, component)
        if info = File.info?(current, follow_symlinks: false)
          if info.symlink?
            raise ConverterError.new(
              "#{context} must not contain symlink component #{current.inspect}"
            )
          end
        end
      end
    end

    private def self.write_text_file(path : String, content : String) : Nil
      Dir.mkdir_p(File.dirname(path))
      File.write(path, content)
    end

    private def self.file_sha256(path : String) : String
      Digest::SHA256.new.file(path).hexfinal
    end

    private def self.publish_exclusive!(
      temporary : String,
      output : String,
    ) : Nil
      {% if flag?(:darwin) %}
        result = LibC.renameatx_np(
          LibC::AT_FDCWD,
          temporary,
          LibC::AT_FDCWD,
          output,
          RENAME_EXCL
        )
        unless result == 0
          errno = Errno.value
          if errno == Errno::EEXIST
            raise ConverterError.new(
              "output path #{output.inspect} already exists"
            )
          end
          raise ConverterError.new(
            "atomic no-overwrite publish failed: #{errno}"
          )
        end
      {% else %}
        raise ConverterError.new(
          "atomic no-overwrite publication is not supported on this platform"
        )
      {% end %}
    end
  end
end
