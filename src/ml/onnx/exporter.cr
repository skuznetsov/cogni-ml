# Walks a Variable autograd graph and emits an ONNX ModelProto.
# Leaves that aren't pre-named as inputs become TensorProto initializers.

require "./protobuf_writer"
require "./constants"
require "../autograd/variable"

module ML
  module ONNX
    private record NodeSpec, op_type : String, inputs : Array(String), outputs : Array(String)
    private record TensorSpec, name : String, dims : Array(Int32), data : Array(Float32)
    private record ValueSpec, name : String, dims : Array(Int32)

    class Exporter
      def initialize
        @next_id = 0
        @var_names = {} of UInt64 => String
        @nodes = [] of NodeSpec
        @initializers = [] of TensorSpec
        @inputs = [] of ValueSpec
        @outputs = [] of ValueSpec
        @visited = Set(UInt64).new
      end

      def input(var : Autograd::Variable, name : String = "input") : self
        @var_names[var.object_id] = name
        @inputs << ValueSpec.new(name, var.shape.to_a)
        self
      end

      def output(var : Autograd::Variable, name : String = "output") : self
        @var_names[var.object_id] = name
        trace(var)
        @outputs << ValueSpec.new(name, var.shape.to_a)
        self
      end

      def to_bytes(model_name : String = "cogni-ml") : Bytes
        root = PBWriter.new
        root.int64(1, IR_VERSION)
        root.string(2, model_name)
        root.message(8) do |os|
          os.int64(2, OPSET_VERSION)
        end
        root.message(7) do |g|
          g.string(2, model_name)
          @nodes.each_with_index { |n, idx| write_node(g, 1, n, idx) }
          @initializers.each { |t| write_tensor(g, 5, t) }
          @inputs.each { |i| write_value_info(g, 11, i) }
          @outputs.each { |o| write_value_info(g, 12, o) }
        end
        root.to_slice
      end

      def write(path : String, model_name : String = "cogni-ml") : Nil
        b = to_bytes(model_name)
        File.open(path, "wb") { |io| io.write(b) }
      end

      private def trace(var : Autograd::Variable) : Nil
        id = var.object_id
        return if @visited.includes?(id)
        @visited.add(id)

        gf = var.grad_fn
        if gf.nil?
          unless @var_names.has_key?(id)
            nm = "const_#{@initializers.size}"
            @var_names[id] = nm
            data_cpu = var.data.on_cpu? ? var.data : var.data.to_cpu
            floats = data_cpu.cpu_data.not_nil!.dup
            @initializers << TensorSpec.new(nm, var.shape.to_a, floats)
          end
          return
        end

        gf.inputs.each { |inp| trace(inp) }
        in_names = gf.inputs.map { |inp| name_of(inp) }
        out_name = name_of(var)
        @nodes << NodeSpec.new(op_type_for(gf), in_names, [out_name])
      end

      private def name_of(var : Autograd::Variable) : String
        existing = @var_names[var.object_id]?
        return existing if existing
        nm = "v_#{@next_id}"
        @next_id += 1
        @var_names[var.object_id] = nm
        nm
      end

      private def op_type_for(gf : Autograd::GradFn) : String
        case gf.name
        when "AddBackward"       then "Add"
        when "SubBackward"       then "Sub"
        when "MulBackward"       then "Mul"
        when "DivBackward"       then "Div"
        when "MatmulBackward"    then "MatMul"
        when "ReluBackward"      then "Relu"
        when "SigmoidBackward"   then "Sigmoid"
        when "TransposeBackward" then "Transpose"
        when "ReshapeBackward"   then "Reshape"
        else
          raise "ONNX export: unsupported op '#{gf.name}'"
        end
      end

      private def write_value_info(w : PBWriter, field : Int32, v : ValueSpec) : Nil
        w.message(field) do |vi|
          vi.string(1, v.name)
          vi.message(2) do |tp|
            tp.message(1) do |tt|
              tt.int32(1, ELEM_FLOAT)
              tt.message(2) do |sh|
                v.dims.each do |d|
                  sh.message(1) do |dim|
                    dim.int64(1, d.to_i64)
                  end
                end
              end
            end
          end
        end
      end

      private def write_tensor(w : PBWriter, field : Int32, t : TensorSpec) : Nil
        w.message(field) do |tp|
          tp.packed_int64(1, t.dims.map(&.to_i64))
          tp.int32(2, ELEM_FLOAT)
          tp.string(8, t.name)
          tp.raw_floats(9, t.data)
        end
      end

      private def write_node(w : PBWriter, field : Int32, n : NodeSpec, idx : Int32) : Nil
        w.message(field) do |np|
          n.inputs.each { |i| np.string(1, i) }
          n.outputs.each { |o| np.string(2, o) }
          np.string(3, "#{n.op_type}_#{idx}")
          np.string(4, n.op_type)
        end
      end
    end
  end
end
