require "digest/sha256"
require "json"
require "set"

require "./conversion_plan"
require "./inventory"
require "./strict_json"

module ML::ThreeD::Trellis2
  class ManifestError < Exception
  end

  struct SourceRef
    getter repository : String
    getter revision : String
    getter license : String

    def initialize(@repository, @revision, @license)
    end
  end

  struct ExternalDependency
    getter id : String
    getter repository : String
    getter revision : String
    getter path : String
    getter byte_length : Int64
    getter sha256 : String
    getter license : String

    def initialize(
      @id,
      @repository,
      @revision,
      @path,
      @byte_length,
      @sha256,
      @license,
    )
    end
  end

  struct PackFile
    getter path : String
    getter role : String
    getter byte_length : Int64
    getter sha256 : String
    getter license : String

    def initialize(@path, @role, @byte_length, @sha256, @license)
    end
  end

  struct ManifestTensor
    getter source_name : String
    getter destination_name : String
    getter file : String
    getter dtype : String
    getter shape : Array(Int64)
    getter data_offsets : Array(Int64)
    getter byte_order : String
    getter layout : String

    def initialize(
      @source_name,
      @destination_name,
      @file,
      @dtype,
      @shape,
      @data_offsets,
      @byte_order,
      @layout,
    )
    end
  end

  struct ManifestStage
    getter id : String
    getter order : Int32
    getter config_path : String
    getter config_sha256 : String
    getter tensors : Array(ManifestTensor)

    def initialize(@id, @order, @config_path, @config_sha256, @tensors)
    end
  end

  class Manifest
    LATEST_SCHEMA_VERSION     = 2_i32
    SUPPORTED_SCHEMA_VERSIONS = Set{1_i32, 2_i32}
    PACK_ID_DOMAIN            = "cogni-ml/trellis2/pack-id/v1"
    MAX_MANIFEST_BYTES        = 16_i64 * 1024_i64 * 1024_i64
    MAX_CONFIG_BYTES          = 16_i64 * 1024_i64 * 1024_i64
    ALLOWED_STAGES            = Set{
      "dino_v3",
      "sparse_structure_flow",
      "sparse_structure_decoder",
      "shape_flow_512",
      "shape_flow_1024",
      "shape_decoder",
      "texture_flow",
      "texture_decoder",
    }
    ALLOWED_DTYPES  = Set{"BF16", "F16", "F32"}
    ALLOWED_LAYOUTS = Set{
      "identity",
      "transpose",
      "permute",
      "split",
      "concat",
    }

    getter schema_version : Int32
    getter converter_version : String
    getter converter_config_sha256 : String
    getter converter_config_path : String?
    getter source : SourceRef
    getter model : SourceRef
    getter external_dependencies : Array(ExternalDependency)
    getter execution_order : Array(String)
    getter files : Array(PackFile)
    getter stages : Array(ManifestStage)
    getter pack_id : String

    private def initialize(
      @schema_version,
      @converter_version,
      @converter_config_sha256,
      @converter_config_path,
      @source,
      @model,
      @external_dependencies,
      @execution_order,
      @files,
      @stages,
      @pack_id,
    )
    end

    def self.load(pack_root : String) : Manifest
      root = validated_pack_root(pack_root)
      path = File.join(root, "manifest.json")
      reject_symlink_components!(path, "manifest")
      info = File.info(path, follow_symlinks: false)
      unless info.file?
        raise ManifestError.new("manifest must be a regular file")
      end
      if info.size > MAX_MANIFEST_BYTES
        raise ManifestError.new(
          "manifest size #{info.size} exceeds #{MAX_MANIFEST_BYTES}"
        )
      end
      manifest = parse(File.read(path))
      manifest.validate_pack!(root)
      manifest
    rescue ex : ManifestError
      raise ex
    rescue ex : File::Error
      raise ManifestError.new("cannot load manifest: #{ex.message}")
    end

    def self.parse(source : String) : Manifest
      manifest = parse_identity(source)
      if manifest.pack_id != manifest.computed_pack_id
        raise ManifestError.new(
          "pack_id mismatch: expected #{manifest.computed_pack_id}, got #{manifest.pack_id}"
        )
      end
      manifest
    end

    def self.compute_pack_id_for_draft(source : String) : String
      parse_identity(source).computed_pack_id
    end

    private def self.parse_identity(source : String) : Manifest
      root = strict_object(source, "manifest")
      unless root.has_key?("schema_version")
        raise ManifestError.new("missing manifest key \"schema_version\"")
      end
      schema_version = int32(root["schema_version"], "schema_version")
      unless SUPPORTED_SCHEMA_VERSIONS.includes?(schema_version)
        raise ManifestError.new("unsupported schema_version #{schema_version}")
      end

      allowed_keys = [
        "schema_version",
        "converter_version",
        "converter_config_sha256",
        "source",
        "model",
        "external_dependencies",
        "execution_order",
        "files",
        "stages",
        "pack_id",
      ]
      if schema_version >= 2
        allowed_keys << "converter_config_path"
      end
      expect_exact_keys!(
        root,
        allowed_keys,
        "manifest"
      )

      converter_version = string(root["converter_version"], "converter_version")
      unless converter_version.matches?(/\A[0-9]+\.[0-9]+\.[0-9]+\z/)
        raise ManifestError.new("converter_version must be semantic x.y.z")
      end
      converter_config_sha256 = sha256(
        string(root["converter_config_sha256"], "converter_config_sha256"),
        "converter_config_sha256"
      )
      converter_config_path = if schema_version >= 2
                                safe_relative_path(
                                  string(
                                    root["converter_config_path"],
                                    "converter_config_path"
                                  )
                                )
                              end
      source_ref = parse_source_ref(root["source"], "source")
      model_ref = parse_source_ref(root["model"], "model")
      external_dependencies = parse_external_dependencies(
        root["external_dependencies"]
      )
      execution_order = string_array(root["execution_order"], "execution_order")
      files = parse_files(root["files"], schema_version)
      stages = parse_stages(root["stages"])
      pack_id = sha256(string(root["pack_id"], "pack_id"), "pack_id")

      manifest = new(
        schema_version,
        converter_version,
        converter_config_sha256,
        converter_config_path,
        source_ref,
        model_ref,
        external_dependencies,
        execution_order,
        files,
        stages,
        pack_id
      )
      manifest.validate_relations!
      manifest
    rescue ex : ManifestError
      raise ex
    rescue ex : StrictJSONError
      raise ManifestError.new(ex.message)
    rescue ex : JSON::ParseException
      raise ManifestError.new("invalid manifest JSON: #{ex.message}")
    end

    def computed_pack_id : String
      bytes = IO::Memory.new
      write_pack_id_frame(bytes, PACK_ID_DOMAIN)
      write_pack_id_frame(bytes, canonical_identity_json)
      @files.sort_by(&.path).each do |file|
        write_pack_id_frame(bytes, file.path)
        bytes.write_bytes(
          file.byte_length.to_u64,
          IO::ByteFormat::LittleEndian
        )
        write_pack_id_frame(bytes, file.sha256)
      end
      Digest::SHA256.hexdigest(bytes.to_slice)
    end

    def canonical_identity_json : String
      JSON.build do |json|
        json.object do
          json.field "schema_version", @schema_version
          json.field "converter_version", @converter_version
          json.field "converter_config_sha256", @converter_config_sha256
          if path = @converter_config_path
            json.field "converter_config_path", path
          end
          write_source_ref(json, "source", @source)
          write_source_ref(json, "model", @model)
          json.field "external_dependencies" do
            json.array do
              @external_dependencies.sort_by(&.id).each do |dependency|
                json.object do
                  json.field "id", dependency.id
                  json.field "repository", dependency.repository
                  json.field "revision", dependency.revision
                  json.field "path", dependency.path
                  json.field "byte_length", dependency.byte_length
                  json.field "sha256", dependency.sha256
                  json.field "license", dependency.license
                end
              end
            end
          end
          json.field "execution_order" do
            json.array { @execution_order.each { |stage_id| json.string(stage_id) } }
          end
          json.field "files" do
            json.array do
              @files.sort_by(&.path).each do |file|
                json.object do
                  json.field "path", file.path
                  json.field "role", file.role
                  json.field "byte_length", file.byte_length
                  json.field "sha256", file.sha256
                  json.field "license", file.license
                end
              end
            end
          end
          json.field "stages" do
            json.array do
              @stages.sort_by(&.order).each do |stage|
                json.object do
                  json.field "id", stage.id
                  json.field "order", stage.order
                  json.field "config_path", stage.config_path
                  json.field "config_sha256", stage.config_sha256
                  json.field "tensors" do
                    json.array do
                      stage.tensors.sort_by(&.destination_name).each do |tensor|
                        json.object do
                          json.field "source_name", tensor.source_name
                          json.field "destination_name", tensor.destination_name
                          json.field "file", tensor.file
                          json.field "dtype", tensor.dtype
                          json.field "shape" do
                            json.array { tensor.shape.each { |dim| json.number(dim) } }
                          end
                          json.field "data_offsets" do
                            json.array do
                              tensor.data_offsets.each { |offset| json.number(offset) }
                            end
                          end
                          json.field "byte_order", tensor.byte_order
                          json.field "layout", tensor.layout
                        end
                      end
                    end
                  end
                end
              end
            end
          end
        end
      end
    end

    def validate_pack!(pack_root : String) : Nil
      root = self.class.validated_pack_root(pack_root)
      validate_root_manifest_binding!(root)
      declared_files = @files.to_h { |file| {file.path, file} }

      @files.each do |file|
        path = safe_pack_path(root, file.path)
        unless File.file?(path)
          raise ManifestError.new("missing pack file #{file.path.inspect}")
        end
        actual_size = File.size(path)
        unless actual_size == file.byte_length
          raise ManifestError.new(
            "byte length mismatch for #{file.path.inspect}: " \
            "expected #{file.byte_length}, got #{actual_size}"
          )
        end
        actual_sha = Digest::SHA256.new.file(path).hexfinal
        unless actual_sha == file.sha256
          raise ManifestError.new("SHA-256 mismatch for #{file.path.inspect}")
        end
      end

      @stages.each do |stage|
        config = declared_files[stage.config_path]
        unless config && config.sha256 == stage.config_sha256
          raise ManifestError.new(
            "stage #{stage.id.inspect} config digest does not match declared file"
          )
        end
        path = safe_pack_path(root, stage.config_path)
        begin
          StrictJSON.parse(File.read(path)).as_h
        rescue ex : StrictJSONError
          raise ManifestError.new(
            "stage #{stage.id.inspect} config is invalid JSON: #{ex.message}"
          )
        rescue
          raise ManifestError.new(
            "stage #{stage.id.inspect} config must be a JSON object"
          )
        end
      end

      if converter_config_path = @converter_config_path
        path = safe_pack_path(root, converter_config_path)
        validate_converter_plan!(path, declared_files)
      end

      expected_by_file = Hash(String, Array(ManifestTensor)).new do |hash, key|
        hash[key] = [] of ManifestTensor
      end
      @stages.each do |stage|
        stage.tensors.each { |tensor| expected_by_file[tensor.file] << tensor }
      end

      expected_by_file.each do |relative_path, expected|
        inventory = SafetensorsInventory.read(safe_pack_path(root, relative_path))
        actual_by_name = inventory.tensors.to_h { |tensor| {tensor.name, tensor} }
        expected_names = expected.map(&.source_name).to_set
        actual_names = actual_by_name.keys.to_set
        missing = (expected_names - actual_names).to_a.sort
        unexpected = (actual_names - expected_names).to_a.sort
        unless missing.empty? && unexpected.empty?
          messages = [] of String
          messages << "missing tensor #{missing.join(", ")}" unless missing.empty?
          messages << "unexpected tensor #{unexpected.join(", ")}" unless unexpected.empty?
          raise ManifestError.new(messages.join("; "))
        end

        expected.each do |tensor|
          actual = actual_by_name[tensor.source_name]
          unless actual.dtype == tensor.dtype &&
                 actual.shape == tensor.shape &&
                 [actual.data_start, actual.data_end] == tensor.data_offsets
            raise ManifestError.new(
              "tensor metadata mismatch for #{tensor.source_name.inspect}"
            )
          end
        end
      end
    rescue ex : InventoryError
      raise ManifestError.new(ex.message)
    end

    protected def validate_relations! : Nil
      raise ManifestError.new("files must not be empty") if @files.empty?
      raise ManifestError.new("stages must not be empty") if @stages.empty?
      if @converter_config_path.nil? &&
         @stages.any? { |stage| stage.tensors.any? { |tensor| tensor.layout != "identity" } }
        raise ManifestError.new(
          "transformed layouts require schema v2 with a bound converter config"
        )
      end

      duplicate_file = duplicate(@files.map(&.path))
      if duplicate_file
        raise ManifestError.new("duplicate pack file #{duplicate_file.inspect}")
      end
      duplicate_stage = duplicate(@stages.map(&.id))
      if duplicate_stage
        raise ManifestError.new("duplicate stage #{duplicate_stage.inspect}")
      end
      duplicate_dependency = duplicate(@external_dependencies.map(&.id))
      if duplicate_dependency
        raise ManifestError.new(
          "duplicate external dependency #{duplicate_dependency.inspect}"
        )
      end
      duplicate_destination = duplicate(
        @stages.flat_map { |stage| stage.tensors.map(&.destination_name) }
      )
      if duplicate_destination
        raise ManifestError.new(
          "duplicate destination tensor #{duplicate_destination.inspect}"
        )
      end
      duplicate_source = duplicate(
        @stages.flat_map do |stage|
          stage.tensors.map { |tensor| "#{tensor.file}\0#{tensor.source_name}" }
        end
      )
      if duplicate_source
        raise ManifestError.new("duplicate source tensor in one file")
      end

      declared_files = @files.to_h { |file| {file.path, file} }
      used_configs = Set(String).new
      used_weights = Set(String).new
      used_converter_configs = Set(String).new
      if converter_config_path = @converter_config_path
        converter_config = declared_files[converter_config_path]?
        unless converter_config &&
               converter_config.role == "converter_config" &&
               converter_config.sha256 == @converter_config_sha256
          raise ManifestError.new(
            "converter_config_path must reference its declared role and digest"
          )
        end
        used_converter_configs << converter_config_path
      elsif @files.any? { |file| file.role == "converter_config" }
        raise ManifestError.new(
          "schema v1 must not declare a converter_config-role file"
        )
      end
      @stages.each do |stage|
        config = declared_files[stage.config_path]?
        unless config
          raise ManifestError.new(
            "stage #{stage.id.inspect} config file is not declared"
          )
        end
        unless config.role == "config"
          raise ManifestError.new(
            "stage #{stage.id.inspect} config must reference a config-role file"
          )
        end
        used_configs << stage.config_path
        stage.tensors.each do |tensor|
          file = declared_files[tensor.file]?
          unless file
            raise ManifestError.new(
              "tensor #{tensor.source_name.inspect} file is not declared"
            )
          end
          unless file.role == "weights"
            raise ManifestError.new(
              "tensor #{tensor.source_name.inspect} must reference a weights-role file"
            )
          end
          used_weights << tensor.file
        end
      end
      @files.each do |file|
        used = case file.role
               when "config"           then used_configs
               when "weights"          then used_weights
               when "converter_config" then used_converter_configs
               else
                 raise ManifestError.new(
                   "unsupported declared file role #{file.role.inspect}"
                 )
               end
        unless used.includes?(file.path)
          raise ManifestError.new(
            "declared #{file.role}-role file #{file.path.inspect} is not referenced"
          )
        end
      end

      ordered = @stages.sort_by(&.order)
      ordered.each_with_index do |stage, index|
        unless stage.order == index
          raise ManifestError.new("stage orders must be contiguous from zero")
        end
      end
      expected_order = ordered.map(&.id)
      unless @execution_order == expected_order
        raise ManifestError.new(
          "execution_order must exactly match stage order #{expected_order}"
        )
      end
    end

    private def safe_pack_path(pack_root : String, relative_path : String) : String
      path = pack_root
      relative_path.split('/').each do |component|
        path = File.join(path, component)
        self.class.reject_symlink_components!(
          path,
          "pack path #{relative_path.inspect}"
        )
      end
      path
    end

    private def validate_converter_plan!(
      path : String,
      declared_files : Hash(String, PackFile),
    ) : Nil
      source = File.read(path)
      plan = ConversionPlan.parse(source)
      unless source == plan.canonical_json
        raise ManifestError.new(
          "converter plan must use canonical JSON serialization"
        )
      end
      unless @external_dependencies.empty?
        raise ManifestError.new(
          "T2N1 converter plans do not admit external pack dependencies"
        )
      end
      unless plan.converter_version == @converter_version &&
             same_identity?(plan.source, @source) &&
             same_identity?(plan.model, @model)
        raise ManifestError.new(
          "converter plan identity does not match manifest identity"
        )
      end
      unless @stages.size == 1 && @stages.first.id == plan.stage_id
        raise ManifestError.new(
          "converter plan stage does not match manifest stage"
        )
      end

      stage = @stages.first
      weight_files = @files
        .select { |file| file.role == "weights" }
        .map(&.path)
        .sort
      unless weight_files == [plan.output_file] &&
             stage.config_path == plan.config_path
        raise ManifestError.new(
          "converter plan pack paths do not match manifest files"
        )
      end
      unless declared_files[plan.output_file].license ==
               plan.source_file.license &&
             declared_files[plan.config_path].license ==
               plan.converter_license &&
             declared_files[@converter_config_path.not_nil!].license ==
               plan.converter_license
        raise ManifestError.new(
          "converter plan licenses do not match manifest files"
        )
      end

      expected_tensors = plan.operations.flat_map do |operation|
        operation.destinations.map do |destination|
          {destination, operation.kind}
        end
      end.sort
      actual_tensors = stage.tensors.map do |tensor|
        unless tensor.destination_name == "#{stage.id}.#{tensor.source_name}"
          raise ManifestError.new(
            "converter plan tensor destination is not stage-qualified"
          )
        end
        {tensor.source_name, tensor.layout}
      end.sort
      unless actual_tensors == expected_tensors
        raise ManifestError.new(
          "converter plan tensors do not match manifest tensors"
        )
      end
    rescue ex : ConversionPlanError
      raise ManifestError.new("converter plan is invalid: #{ex.message}")
    end

    private def same_identity?(
      plan : ConversionIdentity,
      manifest : SourceRef,
    ) : Bool
      plan.repository == manifest.repository &&
        plan.revision == manifest.revision &&
        plan.license == manifest.license
    end

    private def validate_root_manifest_binding!(root : String) : Nil
      path = safe_pack_path(root, "manifest.json")
      info = File.info?(path, follow_symlinks: false)
      unless info && info.file?
        raise ManifestError.new(
          "cannot load manifest: manifest must be a regular file"
        )
      end
      if info.size > MAX_MANIFEST_BYTES
        raise ManifestError.new(
          "manifest size #{info.size} exceeds #{MAX_MANIFEST_BYTES}"
        )
      end

      on_disk = self.class.parse(File.read(path))
      unless on_disk.pack_id == @pack_id &&
             on_disk.canonical_identity_json == canonical_identity_json
        raise ManifestError.new(
          "root manifest does not match the manifest being validated"
        )
      end
    rescue ex : ManifestError
      raise ex
    rescue ex : File::Error
      raise ManifestError.new("cannot load manifest: #{ex.message}")
    end

    private def duplicate(values : Array(String)) : String?
      seen = Set(String).new
      values.each do |value|
        return value unless seen.add?(value)
      end
      nil
    end

    private def write_source_ref(
      json : JSON::Builder,
      name : String,
      ref : SourceRef,
    ) : Nil
      json.field name do
        json.object do
          json.field "repository", ref.repository
          json.field "revision", ref.revision
          json.field "license", ref.license
        end
      end
    end

    private def write_pack_id_frame(io : IO, value : String) : Nil
      io.write_bytes(value.bytesize.to_u64, IO::ByteFormat::LittleEndian)
      io.write(value.to_slice)
    end

    private def self.strict_object(
      source : String,
      context : String,
    ) : Hash(String, JSON::Any)
      StrictJSON.parse(source).as_h
    rescue ex : StrictJSONError
      raise ManifestError.new(ex.message)
    rescue
      raise ManifestError.new("#{context} must be a JSON object")
    end

    private def self.parse_source_ref(
      value : JSON::Any,
      context : String,
    ) : SourceRef
      object = object(value, context)
      expect_exact_keys!(object, ["repository", "revision", "license"], context)
      repository = string(object["repository"], "#{context}.repository")
      revision = string(object["revision"], "#{context}.revision")
      license = string(object["license"], "#{context}.license")
      raise ManifestError.new("#{context}.repository must not be empty") if repository.empty?
      raise ManifestError.new("#{context}.license must not be empty") if license.empty?
      unless revision.matches?(/\A(?:[0-9a-f]{40}|[0-9a-f]{64})\z/)
        raise ManifestError.new("#{context} must use an immutable revision")
      end
      SourceRef.new(repository, revision, license)
    end

    private def self.parse_external_dependencies(
      value : JSON::Any,
    ) : Array(ExternalDependency)
      array(value, "external_dependencies").map_with_index do |entry, index|
        context = "external_dependency[#{index}]"
        object = object(entry, context)
        expect_exact_keys!(
          object,
          [
            "id",
            "repository",
            "revision",
            "path",
            "byte_length",
            "sha256",
            "license",
          ],
          "external_dependency"
        )

        id = nonempty_string(object["id"], "#{context}.id")
        unless id.matches?(/\A[a-z0-9][a-z0-9_.-]*\z/)
          raise ManifestError.new("#{context}.id is invalid")
        end
        repository = nonempty_string(
          object["repository"],
          "#{context}.repository"
        )
        revision = string(object["revision"], "#{context}.revision")
        unless revision.matches?(/\A(?:[0-9a-f]{40}|[0-9a-f]{64})\z/)
          raise ManifestError.new(
            "#{context} must use an immutable revision"
          )
        end
        path = safe_relative_path(string(object["path"], "#{context}.path"))
        byte_length = int64(object["byte_length"], "#{context}.byte_length")
        unless byte_length > 0
          raise ManifestError.new("#{context}.byte_length must be positive")
        end
        digest = sha256(string(object["sha256"], "#{context}.sha256"), context)
        license = nonempty_string(object["license"], "#{context}.license")
        ExternalDependency.new(
          id,
          repository,
          revision,
          path,
          byte_length,
          digest,
          license
        )
      end
    end

    private def self.parse_files(
      value : JSON::Any,
      schema_version : Int32,
    ) : Array(PackFile)
      array(value, "files").map_with_index do |entry, index|
        context = "file[#{index}]"
        object = object(entry, context)
        expect_exact_keys!(
          object,
          ["path", "role", "byte_length", "sha256", "license"],
          "file"
        )
        path = safe_relative_path(string(object["path"], "#{context}.path"))
        role = string(object["role"], "#{context}.role")
        allowed_roles = schema_version >= 2 ? {"config", "weights", "converter_config"} : {"config", "weights"}
        unless allowed_roles.includes?(role)
          raise ManifestError.new("#{context}.role is unsupported")
        end
        byte_length = int64(object["byte_length"], "#{context}.byte_length")
        unless byte_length >= 0
          raise ManifestError.new("#{context}.byte_length must be non-negative")
        end
        if {"config", "converter_config"}.includes?(role) &&
           byte_length > MAX_CONFIG_BYTES
          raise ManifestError.new(
            "#{context}.byte_length exceeds config limit #{MAX_CONFIG_BYTES}"
          )
        end
        digest = sha256(string(object["sha256"], "#{context}.sha256"), context)
        license = string(object["license"], "#{context}.license")
        raise ManifestError.new("#{context}.license must not be empty") if license.empty?
        PackFile.new(path, role, byte_length, digest, license)
      end
    end

    private def self.parse_stages(value : JSON::Any) : Array(ManifestStage)
      array(value, "stages").map_with_index do |entry, index|
        context = "stage[#{index}]"
        object = object(entry, context)
        expect_exact_keys!(
          object,
          ["id", "order", "config_path", "config_sha256", "tensors"],
          "stage"
        )
        id = string(object["id"], "#{context}.id")
        unless ALLOWED_STAGES.includes?(id)
          raise ManifestError.new("unsupported stage #{id.inspect}")
        end
        order = int32(object["order"], "#{context}.order")
        if order < 0
          raise ManifestError.new("#{context}.order must be non-negative")
        end
        config_path = safe_relative_path(
          string(object["config_path"], "#{context}.config_path")
        )
        config_sha256 = sha256(
          string(object["config_sha256"], "#{context}.config_sha256"),
          "#{context}.config_sha256"
        )
        tensors = parse_tensors(object["tensors"], context)
        raise ManifestError.new("#{context}.tensors must not be empty") if tensors.empty?
        ManifestStage.new(id, order, config_path, config_sha256, tensors)
      end
    end

    private def self.parse_tensors(
      value : JSON::Any,
      stage_context : String,
    ) : Array(ManifestTensor)
      array(value, "#{stage_context}.tensors").map_with_index do |entry, index|
        context = "#{stage_context}.tensor[#{index}]"
        object = object(entry, context)
        expect_exact_keys!(
          object,
          [
            "source_name",
            "destination_name",
            "file",
            "dtype",
            "shape",
            "data_offsets",
            "byte_order",
            "layout",
          ],
          "tensor"
        )
        source_name = nonempty_string(object["source_name"], "#{context}.source_name")
        destination_name = nonempty_string(
          object["destination_name"],
          "#{context}.destination_name"
        )
        file = safe_relative_path(string(object["file"], "#{context}.file"))
        dtype = string(object["dtype"], "#{context}.dtype")
        unless ALLOWED_DTYPES.includes?(dtype)
          raise ManifestError.new("unsupported dtype #{dtype.inspect}")
        end
        shape = int64_array(object["shape"], "#{context}.shape")
        elements = checked_elements(shape, context)
        offsets = int64_array(object["data_offsets"], "#{context}.data_offsets")
        unless offsets.size == 2 && offsets[0] >= 0 && offsets[1] >= offsets[0]
          raise ManifestError.new("#{context}.data_offsets are invalid")
        end
        bytes_per_element = dtype == "F32" ? 4_i64 : 2_i64
        if elements > Int64::MAX // bytes_per_element
          raise ManifestError.new("#{context}.shape byte size overflow")
        end
        expected_bytes = elements * bytes_per_element
        unless offsets[1] - offsets[0] == expected_bytes
          raise ManifestError.new("#{context}.data_offsets do not match shape/dtype")
        end
        byte_order = string(object["byte_order"], "#{context}.byte_order")
        unless byte_order == "little"
          raise ManifestError.new("unsupported byte_order #{byte_order.inspect}")
        end
        layout = string(object["layout"], "#{context}.layout")
        unless ALLOWED_LAYOUTS.includes?(layout)
          raise ManifestError.new("unsupported layout #{layout.inspect}")
        end
        ManifestTensor.new(
          source_name,
          destination_name,
          file,
          dtype,
          shape,
          offsets,
          byte_order,
          layout
        )
      end
    end

    private def self.expect_exact_keys!(
      object : Hash(String, JSON::Any),
      allowed : Array(String),
      context : String,
    ) : Nil
      object.each_key do |key|
        unless allowed.includes?(key)
          raise ManifestError.new("unknown #{context} key #{key.inspect}")
        end
      end
      allowed.each do |key|
        unless object.has_key?(key)
          raise ManifestError.new("missing #{context} key #{key.inspect}")
        end
      end
    end

    private def self.object(
      value : JSON::Any,
      context : String,
    ) : Hash(String, JSON::Any)
      value.as_h
    rescue
      raise ManifestError.new("#{context} must be an object")
    end

    private def self.array(
      value : JSON::Any,
      context : String,
    ) : Array(JSON::Any)
      value.as_a
    rescue
      raise ManifestError.new("#{context} must be an array")
    end

    private def self.string(value : JSON::Any, context : String) : String
      value.as_s
    rescue
      raise ManifestError.new("#{context} must be a string")
    end

    private def self.nonempty_string(value : JSON::Any, context : String) : String
      result = string(value, context)
      raise ManifestError.new("#{context} must not be empty") if result.empty?
      result
    end

    private def self.int64(value : JSON::Any, context : String) : Int64
      value.as_i64
    rescue
      raise ManifestError.new("#{context} must be an integer")
    end

    private def self.int32(value : JSON::Any, context : String) : Int32
      raw = int64(value, context)
      unless Int32::MIN <= raw <= Int32::MAX
        raise ManifestError.new("#{context} is outside Int32 range")
      end
      raw.to_i32
    end

    private def self.string_array(value : JSON::Any, context : String) : Array(String)
      array(value, context).map_with_index do |entry, index|
        nonempty_string(entry, "#{context}[#{index}]")
      end
    end

    private def self.int64_array(value : JSON::Any, context : String) : Array(Int64)
      array(value, context).map_with_index do |entry, index|
        int64(entry, "#{context}[#{index}]")
      end
    end

    private def self.checked_elements(shape : Array(Int64), context : String) : Int64
      shape.each do |dimension|
        if dimension < 0
          raise ManifestError.new("#{context}.shape dimensions must be non-negative")
        end
      end
      return 0_i64 if shape.any?(&.zero?)

      shape.reduce(1_i64) do |product, dimension|
        if product > Int64::MAX // dimension
          raise ManifestError.new("#{context}.shape overflow")
        end
        product * dimension
      end
    end

    private def self.sha256(value : String, context : String) : String
      unless value.matches?(/\A[0-9a-f]{64}\z/)
        raise ManifestError.new("#{context} must be a lowercase SHA-256")
      end
      value
    end

    private def self.safe_relative_path(value : String) : String
      components = value.split('/', remove_empty: false)
      unsafe = value.empty? ||
               value.starts_with?('/') ||
               value.includes?('\\') ||
               value.includes?('\0') ||
               value.includes?(':') ||
               components.any? { |component| component.empty? || component == "." || component == ".." }
      if unsafe
        raise ManifestError.new("unsafe relative path #{value.inspect}")
      end
      value
    end

    protected def self.validated_pack_root(
      path : String,
      context : String = "pack root",
    ) : String
      normalized = Path.new(File.expand_path(path)).normalize.to_s
      reject_symlink_components!(normalized, context)
      info = File.info(normalized, follow_symlinks: false)
      unless info.directory?
        raise ManifestError.new("#{context} must be a directory")
      end
      normalized
    rescue ex : ManifestError
      raise ex
    rescue ex : File::Error
      raise ManifestError.new("cannot inspect #{context}: #{ex.message}")
    end

    protected def self.reject_symlink_components!(
      path : String,
      context : String,
    ) : Nil
      normalized = Path.new(File.expand_path(path)).normalize.to_s
      current = "/"
      normalized.split('/', remove_empty: true).each do |component|
        current = File.join(current, component)
        if info = File.info?(current, follow_symlinks: false)
          if info.symlink?
            raise ManifestError.new(
              "#{context} must not contain symlink component #{current.inspect}"
            )
          end
        end
      end
    end
  end
end
