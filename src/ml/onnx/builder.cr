# Direct ONNX graph builder for handcrafted networks.
# Unlike Exporter (which traces a Variable autograd graph),
# Builder lets you compose nodes/initializers/IO directly — mirroring
# onnx.helper.make_node / make_graph / make_model.
#
# Typical use:
#   b = Builder.new
#   b.input("input", [1, 10, 30, 30])
#   b.output("output", [1, 10, 30, 30])
#   b.initializer("W", [10, 10, 3, 3], weights_flat)
#   b.conv("input", "W", "output", kernel_shape: [3,3], pads: [1,1,1,1])
#   b.write("task001.onnx")

require "./protobuf_writer"
require "./constants"

module ML
  module ONNX
    class Builder
      # AttributeType enum from onnx.proto (subset)
      ATTR_FLOAT  = 1
      ATTR_INT    = 2
      ATTR_STRING = 3
      ATTR_TENSOR = 4
      ATTR_FLOATS = 6
      ATTR_INTS   = 7

      alias AttrValue = Float32 | Int32 | Int64 | String | Array(Int64) | Array(Float32)

      private record Node,
        op_type : String,
        inputs : Array(String),
        outputs : Array(String),
        attrs : Hash(String, AttrValue),
        name : String

      private record Init, name : String, dims : Array(Int32), data : Array(Float32)
      private record InitI64, name : String, dims : Array(Int32), data : Array(Int64)
      private record VInfo, name : String, dims : Array(Int32)

      def initialize
        @nodes = [] of Node
        @inits = [] of Init
        @inits_i64 = [] of InitI64
        @inputs = [] of VInfo
        @outputs = [] of VInfo
        @auto_id = 0
      end

      def input(name : String, shape : Array(Int32)) : String
        @inputs << VInfo.new(name, shape)
        name
      end

      def output(name : String, shape : Array(Int32)) : String
        @outputs << VInfo.new(name, shape)
        name
      end

      def initializer(name : String, shape : Array(Int32), data : Array(Float32)) : String
        raise ArgumentError.new("size mismatch: shape=#{shape.product} data=#{data.size}") if shape.product != data.size
        @inits << Init.new(name, shape, data)
        name
      end

      def initializer_int64(name : String, shape : Array(Int32), data : Array(Int64)) : String
        raise ArgumentError.new("size mismatch: shape=#{shape.product} data=#{data.size}") if shape.product != data.size
        @inits_i64 << InitI64.new(name, shape, data)
        name
      end

      def node(op_type : String, inputs : Array(String), outputs : Array(String),
               attrs : Hash(String, AttrValue) = {} of String => AttrValue,
               name : String? = nil) : Nil
        @auto_id += 1
        @nodes << Node.new(op_type, inputs, outputs, attrs, name || "#{op_type}_#{@auto_id}")
      end

      # Conv2d convenience: assumes input [N,Cin,H,W] and weight [Cout,Cin,kH,kW]
      def conv(input_name : String, weight_name : String, output_name : String,
               kernel_shape : Array(Int32),
               pads : Array(Int32)? = nil,
               strides : Array(Int32)? = nil,
               dilations : Array(Int32)? = nil,
               group : Int32 = 1,
               bias_name : String? = nil) : Nil
        attrs = {} of String => AttrValue
        attrs["kernel_shape"] = kernel_shape.map(&.to_i64)
        attrs["pads"] = pads.map(&.to_i64) if pads
        attrs["strides"] = strides.map(&.to_i64) if strides
        attrs["dilations"] = dilations.map(&.to_i64) if dilations
        attrs["group"] = group.to_i64 if group != 1
        inputs = bias_name ? [input_name, weight_name, bias_name] : [input_name, weight_name]
        node("Conv", inputs, [output_name], attrs)
      end

      def to_bytes(model_name : String = "cogni-ml") : Bytes
        root = PBWriter.new
        root.int64(1, IR_VERSION)
        root.string(2, model_name)
        root.message(8) { |os| os.int64(2, OPSET_VERSION) }
        root.message(7) do |g|
          g.string(2, model_name)
          @nodes.each { |n| write_node(g, 1, n) }
          @inits.each { |t| write_tensor(g, 5, t) }
          @inits_i64.each { |t| write_tensor_i64(g, 5, t) }
          @inputs.each { |i| write_value_info(g, 11, i) }
          @outputs.each { |o| write_value_info(g, 12, o) }
        end
        root.to_slice
      end

      def write(path : String, model_name : String = "cogni-ml") : Nil
        b = to_bytes(model_name)
        File.open(path, "wb") { |io| io.write(b) }
      end

      private def write_value_info(w : PBWriter, field : Int32, v : VInfo) : Nil
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

      private def write_tensor(w : PBWriter, field : Int32, t : Init) : Nil
        w.message(field) do |tp|
          tp.packed_int64(1, t.dims.map(&.to_i64))
          tp.int32(2, ELEM_FLOAT)
          tp.string(8, t.name)
          tp.raw_floats(9, t.data)
        end
      end

      private def write_tensor_i64(w : PBWriter, field : Int32, t : InitI64) : Nil
        w.message(field) do |tp|
          tp.packed_int64(1, t.dims.map(&.to_i64))
          tp.int32(2, ELEM_INT64)
          tp.string(8, t.name)
          tp.packed_int64(7, t.data)
        end
      end

      private def write_node(w : PBWriter, field : Int32, n : Node) : Nil
        w.message(field) do |np|
          n.inputs.each { |i| np.string(1, i) }
          n.outputs.each { |o| np.string(2, o) }
          np.string(3, n.name)
          np.string(4, n.op_type)
          n.attrs.each do |k, v|
            np.message(5) { |ap| write_attribute(ap, k, v) }
          end
        end
      end

      private def write_attribute(ap : PBWriter, name : String, value : AttrValue) : Nil
        ap.string(1, name)
        case value
        when Float32
          ap.float(2, value)
          ap.int32(20, ATTR_FLOAT)
        when Int32
          ap.int64(3, value.to_i64)
          ap.int32(20, ATTR_INT)
        when Int64
          ap.int64(3, value)
          ap.int32(20, ATTR_INT)
        when String
          ap.bytes_field(4, value.to_slice)
          ap.int32(20, ATTR_STRING)
        when Array(Int64)
          value.each { |i| ap.int64(8, i) }
          ap.int32(20, ATTR_INTS)
        when Array(Float32)
          value.each { |f| ap.float(7, f) }
          ap.int32(20, ATTR_FLOATS)
        end
      end
    end
  end
end
