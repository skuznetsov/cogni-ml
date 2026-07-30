require "json"
require "set"

require "./strict_json"

module ML::ThreeD::Trellis2
  class ConversionPlanError < Exception
  end

  struct ConversionIdentity
    getter repository : String
    getter revision : String
    getter license : String

    def initialize(@repository, @revision, @license)
    end
  end

  struct ConversionSourceFile
    getter path : String
    getter byte_length : Int64
    getter sha256 : String
    getter license : String

    def initialize(@path, @byte_length, @sha256, @license)
    end
  end

  struct ConversionOperation
    getter id : String
    getter kind : String
    getter sources : Array(String)
    getter destinations : Array(String)
    getter axes : Array(Int32)
    getter axis : Int32?
    getter sizes : Array(Int64)

    def initialize(
      @id,
      @kind,
      @sources,
      @destinations,
      @axes = [] of Int32,
      @axis = nil,
      @sizes = [] of Int64,
    )
    end

    def write_canonical(json : JSON::Builder) : Nil
      json.object do
        json.field "id", @id
        json.field "kind", @kind
        case @kind
        when "identity"
          json.field "source", @sources.first
          json.field "destination", @destinations.first
        when "transpose", "permute"
          json.field "source", @sources.first
          json.field "destination", @destinations.first
          json.field "axes" do
            json.array { @axes.each { |axis| json.number(axis) } }
          end
        when "split"
          json.field "source", @sources.first
          json.field "destinations" do
            json.array do
              @destinations.each { |destination| json.string(destination) }
            end
          end
          json.field "axis", @axis.not_nil!
          json.field "sizes" do
            json.array { @sizes.each { |size| json.number(size) } }
          end
        when "concat"
          json.field "sources" do
            json.array { @sources.each { |source| json.string(source) } }
          end
          json.field "destination", @destinations.first
          json.field "axis", @axis.not_nil!
        else
          raise ConversionPlanError.new(
            "unsupported conversion operation #{@kind.inspect}"
          )
        end
      end
    end
  end

  class ConversionPlan
    SCHEMA_VERSION = 1_i32
    MAX_PLAN_BYTES = 16_i64 * 1024_i64 * 1024_i64
    ALLOWED_STAGES = Set{
      "dino_v3",
      "sparse_structure_flow",
      "sparse_structure_decoder",
      "shape_flow_512",
      "shape_flow_1024",
      "shape_decoder",
      "texture_flow",
      "texture_decoder",
    }

    getter schema_version : Int32
    getter converter_version : String
    getter converter_license : String
    getter source : ConversionIdentity
    getter model : ConversionIdentity
    getter stage_id : String
    getter source_file : ConversionSourceFile
    getter output_file : String
    getter config_path : String
    getter operations : Array(ConversionOperation)

    private def initialize(
      @schema_version,
      @converter_version,
      @converter_license,
      @source,
      @model,
      @stage_id,
      @source_file,
      @output_file,
      @config_path,
      @operations,
    )
    end

    def self.load(path : String) : ConversionPlan
      normalized = validated_file_path(path)
      size = File.size(normalized)
      if size > MAX_PLAN_BYTES
        raise ConversionPlanError.new(
          "conversion plan size #{size} exceeds #{MAX_PLAN_BYTES}"
        )
      end
      parse(File.read(normalized))
    rescue ex : ConversionPlanError
      raise ex
    rescue ex : File::Error
      raise ConversionPlanError.new(
        "cannot load conversion plan #{path.inspect}: #{ex.message}"
      )
    end

    def self.parse(source : String) : ConversionPlan
      root = strict_object(source, "conversion plan")
      expect_exact_keys!(
        root,
        [
          "schema_version",
          "converter_version",
          "converter_license",
          "source",
          "model",
          "stage_id",
          "source_file",
          "output_file",
          "config_path",
          "operations",
        ],
        "conversion plan"
      )

      schema_version = int32(root["schema_version"], "schema_version")
      unless schema_version == SCHEMA_VERSION
        raise ConversionPlanError.new(
          "unsupported conversion plan schema_version #{schema_version}"
        )
      end
      converter_version = string(
        root["converter_version"],
        "converter_version"
      )
      unless converter_version.matches?(/\A[0-9]+\.[0-9]+\.[0-9]+\z/)
        raise ConversionPlanError.new(
          "converter_version must be semantic x.y.z"
        )
      end
      converter_license = nonempty_string(
        root["converter_license"],
        "converter_license"
      )
      source_identity = parse_identity(root["source"], "source")
      model_identity = parse_identity(root["model"], "model")
      stage_id = nonempty_string(root["stage_id"], "stage_id")
      unless ALLOWED_STAGES.includes?(stage_id)
        raise ConversionPlanError.new("unsupported stage #{stage_id.inspect}")
      end
      source_file = parse_source_file(root["source_file"])
      output_file = safe_relative_path(
        string(root["output_file"], "output_file")
      )
      unless output_file.ends_with?(".safetensors")
        raise ConversionPlanError.new(
          "output_file must use the .safetensors extension"
        )
      end
      config_path = safe_relative_path(
        string(root["config_path"], "config_path")
      )
      unless config_path.ends_with?(".json")
        raise ConversionPlanError.new(
          "config_path must use the .json extension"
        )
      end
      if output_file == config_path
        raise ConversionPlanError.new(
          "output_file and config_path must be distinct"
        )
      end
      operations = parse_operations(root["operations"])
      validate_operation_graph!(operations)

      new(
        schema_version,
        converter_version,
        converter_license,
        source_identity,
        model_identity,
        stage_id,
        source_file,
        output_file,
        config_path,
        operations
      )
    rescue ex : ConversionPlanError
      raise ex
    rescue ex : StrictJSONError
      raise ConversionPlanError.new(ex.message)
    end

    def canonical_json : String
      JSON.build do |json|
        json.object do
          json.field "schema_version", @schema_version
          json.field "converter_version", @converter_version
          json.field "converter_license", @converter_license
          write_identity(json, "source", @source)
          write_identity(json, "model", @model)
          json.field "stage_id", @stage_id
          json.field "source_file" do
            json.object do
              json.field "path", @source_file.path
              json.field "byte_length", @source_file.byte_length
              json.field "sha256", @source_file.sha256
              json.field "license", @source_file.license
            end
          end
          json.field "output_file", @output_file
          json.field "config_path", @config_path
          json.field "operations" do
            json.array do
              @operations.sort_by(&.id).each(&.write_canonical(json))
            end
          end
        end
      end
    end

    private def write_identity(
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

    private def self.parse_identity(
      value : JSON::Any,
      context : String,
    ) : ConversionIdentity
      object = object(value, context)
      expect_exact_keys!(
        object,
        ["repository", "revision", "license"],
        context
      )
      repository = nonempty_string(
        object["repository"],
        "#{context}.repository"
      )
      revision = string(object["revision"], "#{context}.revision")
      unless revision.matches?(/\A(?:[0-9a-f]{40}|[0-9a-f]{64})\z/)
        raise ConversionPlanError.new(
          "#{context} must use an immutable revision"
        )
      end
      license = nonempty_string(object["license"], "#{context}.license")
      ConversionIdentity.new(repository, revision, license)
    end

    private def self.parse_source_file(
      value : JSON::Any,
    ) : ConversionSourceFile
      object = object(value, "source_file")
      expect_exact_keys!(
        object,
        ["path", "byte_length", "sha256", "license"],
        "source_file"
      )
      path = safe_relative_path(string(object["path"], "source_file.path"))
      byte_length = int64(
        object["byte_length"],
        "source_file.byte_length"
      )
      unless byte_length > 0
        raise ConversionPlanError.new(
          "source_file.byte_length must be positive"
        )
      end
      digest = sha256(
        string(object["sha256"], "source_file.sha256"),
        "source_file.sha256"
      )
      license = nonempty_string(
        object["license"],
        "source_file.license"
      )
      ConversionSourceFile.new(path, byte_length, digest, license)
    end

    private def self.parse_operations(
      value : JSON::Any,
    ) : Array(ConversionOperation)
      entries = array(value, "operations")
      if entries.empty?
        raise ConversionPlanError.new("operations must not be empty")
      end
      entries.map_with_index do |entry, index|
        context = "operation[#{index}]"
        object = object(entry, context)
        id = nonempty_string(
          object["id"]? || raise(
            ConversionPlanError.new("missing operation key \"id\"")
          ),
          "#{context}.id"
        )
        unless id.matches?(/\A[a-z0-9][a-z0-9_.-]*\z/)
          raise ConversionPlanError.new("#{context}.id is invalid")
        end
        kind = nonempty_string(
          object["kind"]? || raise(
            ConversionPlanError.new("missing operation key \"kind\"")
          ),
          "#{context}.kind"
        )
        parse_operation(object, id, kind, context)
      end
    end

    private def self.parse_operation(
      object : Hash(String, JSON::Any),
      id : String,
      kind : String,
      context : String,
    ) : ConversionOperation
      case kind
      when "identity"
        expect_exact_keys!(
          object,
          ["id", "kind", "source", "destination"],
          "operation"
        )
        ConversionOperation.new(
          id,
          kind,
          [nonempty_string(object["source"], "#{context}.source")],
          [
            nonempty_string(
              object["destination"],
              "#{context}.destination"
            ),
          ]
        )
      when "transpose", "permute"
        expect_exact_keys!(
          object,
          ["id", "kind", "source", "destination", "axes"],
          "operation"
        )
        axes = int32_array(object["axes"], "#{context}.axes")
        if kind == "transpose"
          unless axes.size == 2 &&
                 axes[0] >= 0 &&
                 axes[1] >= 0 &&
                 axes[0] != axes[1]
            raise ConversionPlanError.new(
              "#{context} transpose axes must be two distinct non-negative axes"
            )
          end
        elsif axes.empty? ||
              axes.sort != (0...axes.size).map(&.to_i32)
          raise ConversionPlanError.new(
            "#{context} permutation axes must contain every axis exactly once"
          )
        end
        ConversionOperation.new(
          id,
          kind,
          [nonempty_string(object["source"], "#{context}.source")],
          [
            nonempty_string(
              object["destination"],
              "#{context}.destination"
            ),
          ],
          axes
        )
      when "split"
        expect_exact_keys!(
          object,
          [
            "id",
            "kind",
            "source",
            "destinations",
            "axis",
            "sizes",
          ],
          "operation"
        )
        destinations = string_array(
          object["destinations"],
          "#{context}.destinations"
        )
        sizes = int64_array(object["sizes"], "#{context}.sizes")
        if destinations.empty? ||
           destinations.size != sizes.size ||
           sizes.any? { |size| size < 0 }
          raise ConversionPlanError.new(
            "#{context} split destinations/sizes are invalid"
          )
        end
        axis = int32(object["axis"], "#{context}.axis")
        if axis < 0
          raise ConversionPlanError.new(
            "#{context}.axis must be non-negative"
          )
        end
        ConversionOperation.new(
          id,
          kind,
          [nonempty_string(object["source"], "#{context}.source")],
          destinations,
          axis: axis,
          sizes: sizes
        )
      when "concat"
        expect_exact_keys!(
          object,
          ["id", "kind", "sources", "destination", "axis"],
          "operation"
        )
        sources = string_array(object["sources"], "#{context}.sources")
        if sources.empty?
          raise ConversionPlanError.new(
            "#{context}.sources must not be empty"
          )
        end
        axis = int32(object["axis"], "#{context}.axis")
        if axis < 0
          raise ConversionPlanError.new(
            "#{context}.axis must be non-negative"
          )
        end
        ConversionOperation.new(
          id,
          kind,
          sources,
          [
            nonempty_string(
              object["destination"],
              "#{context}.destination"
            ),
          ],
          axis: axis
        )
      else
        raise ConversionPlanError.new(
          "unsupported conversion operation #{kind.inspect}"
        )
      end
    end

    private def self.validate_operation_graph!(
      operations : Array(ConversionOperation),
    ) : Nil
      duplicate_id = duplicate(operations.map(&.id))
      if duplicate_id
        raise ConversionPlanError.new(
          "duplicate conversion operation id #{duplicate_id.inspect}"
        )
      end
      destinations = operations.flat_map(&.destinations)
      duplicate_destination = duplicate(destinations)
      if duplicate_destination
        raise ConversionPlanError.new(
          "duplicate conversion destination #{duplicate_destination.inspect}"
        )
      end
      sources = operations.flat_map(&.sources)
      duplicate_source = duplicate(sources)
      if duplicate_source
        raise ConversionPlanError.new(
          "source tensor #{duplicate_source.inspect} is consumed more than once"
        )
      end
      overlap = sources.to_set & destinations.to_set
      unless overlap.empty?
        raise ConversionPlanError.new(
          "conversion operations must not depend on generated outputs"
        )
      end
    end

    private def self.validated_file_path(path : String) : String
      normalized = Path.new(File.expand_path(path)).normalize.to_s
      reject_symlink_components!(normalized, "conversion plan")
      info = File.info(normalized, follow_symlinks: false)
      unless info.file?
        raise ConversionPlanError.new(
          "conversion plan must be a regular file"
        )
      end
      normalized
    rescue ex : ConversionPlanError
      raise ex
    rescue ex : File::Error
      raise ConversionPlanError.new(
        "cannot inspect conversion plan #{path.inspect}: #{ex.message}"
      )
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
            raise ConversionPlanError.new(
              "#{context} must not contain symlink component #{current.inspect}"
            )
          end
        end
      end
    end

    private def self.strict_object(
      source : String,
      context : String,
    ) : Hash(String, JSON::Any)
      StrictJSON.parse(source).as_h
    rescue ex : StrictJSONError
      raise ConversionPlanError.new(ex.message)
    rescue
      raise ConversionPlanError.new("#{context} must be a JSON object")
    end

    private def self.expect_exact_keys!(
      object : Hash(String, JSON::Any),
      allowed : Array(String),
      context : String,
    ) : Nil
      object.each_key do |key|
        unless allowed.includes?(key)
          raise ConversionPlanError.new(
            "unknown #{context} key #{key.inspect}"
          )
        end
      end
      allowed.each do |key|
        unless object.has_key?(key)
          raise ConversionPlanError.new(
            "missing #{context} key #{key.inspect}"
          )
        end
      end
    end

    private def self.object(
      value : JSON::Any,
      context : String,
    ) : Hash(String, JSON::Any)
      value.as_h
    rescue
      raise ConversionPlanError.new("#{context} must be an object")
    end

    private def self.array(
      value : JSON::Any,
      context : String,
    ) : Array(JSON::Any)
      value.as_a
    rescue
      raise ConversionPlanError.new("#{context} must be an array")
    end

    private def self.string(value : JSON::Any, context : String) : String
      value.as_s
    rescue
      raise ConversionPlanError.new("#{context} must be a string")
    end

    private def self.nonempty_string(
      value : JSON::Any,
      context : String,
    ) : String
      result = string(value, context)
      if result.empty?
        raise ConversionPlanError.new("#{context} must not be empty")
      end
      result
    end

    private def self.int64(value : JSON::Any, context : String) : Int64
      value.as_i64
    rescue
      raise ConversionPlanError.new("#{context} must be an integer")
    end

    private def self.int32(value : JSON::Any, context : String) : Int32
      raw = int64(value, context)
      unless Int32::MIN <= raw <= Int32::MAX
        raise ConversionPlanError.new("#{context} is outside Int32 range")
      end
      raw.to_i32
    end

    private def self.string_array(
      value : JSON::Any,
      context : String,
    ) : Array(String)
      array(value, context).map_with_index do |entry, index|
        nonempty_string(entry, "#{context}[#{index}]")
      end
    end

    private def self.int64_array(
      value : JSON::Any,
      context : String,
    ) : Array(Int64)
      array(value, context).map_with_index do |entry, index|
        int64(entry, "#{context}[#{index}]")
      end
    end

    private def self.int32_array(
      value : JSON::Any,
      context : String,
    ) : Array(Int32)
      array(value, context).map_with_index do |entry, index|
        int32(entry, "#{context}[#{index}]")
      end
    end

    private def self.sha256(value : String, context : String) : String
      unless value.matches?(/\A[0-9a-f]{64}\z/)
        raise ConversionPlanError.new(
          "#{context} must be a lowercase SHA-256"
        )
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
               components.any? do |component|
                 component.empty? || component == "." || component == ".."
               end
      if unsafe
        raise ConversionPlanError.new(
          "unsafe relative path #{value.inspect}"
        )
      end
      value
    end

    private def self.duplicate(values : Array(String)) : String?
      seen = Set(String).new
      values.each do |value|
        return value unless seen.add?(value)
      end
      nil
    end
  end
end
