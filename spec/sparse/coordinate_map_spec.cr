require "json"
require "digest/sha256"
require "../../src/ml/sparse/coordinate_map"
require "../spec_helper"

private def sparse_map_fixture : JSON::Any
  JSON.parse(File.read(File.join(
    __DIR__,
    "../fixtures/trellis2/sparse_tensor_cpu_v1.json"
  )))
end

private def sparse_map_coordinates(payload : JSON::Any) : Array(Int32)
  payload.as_a.flat_map do |row|
    row.as_a.map { |value| value.as_i.to_i32 }
  end
end

private def sparse_map_i32le_sha256(values : Indexable(Int32)) : String
  bytes = Bytes.new(values.size * 4, 0_u8)
  values.each_with_index do |value, index|
    IO::ByteFormat::LittleEndian.encode(value, bytes[index * 4, 4])
  end
  Digest::SHA256.hexdigest(bytes)
end

describe ML::Sparse::CoordinateMap3D do
  it "matches pinned upstream layout metadata while preserving row order" do
    fixture = sparse_map_fixture
    fixture["schema"].as_s.should eq(
      "cogni-ml/trellis2/sparse-tensor-structural-oracle/v1"
    )
    provenance = fixture["provenance"]
    provenance["commit"].as_s.should eq(
      "75fbf0183001ed9876c8dbb35de6b68552ee08bd"
    )
    provenance["source_sha256"].as_s.should eq(
      "99dbcb7298238fdb6b6d47918ec068f618c68067653489d22b32c136c1ae0e78"
    )
    provenance["torch_version"].as_s.should eq("2.9.0")
    provenance["network"].as_s.should eq("none")
    provenance["device"].as_s.should eq("cpu")
    provenance["weights"].as_s.should eq("none")
    fixture["contract"]["input_order"].as_s.should eq(
      "preserved within nondecreasing batch groups"
    )
    fixture["contract"]["dense_materialization"].as_s.should eq("not admitted")
    fixture["upstream"]["to_dense_failure"]["type"].as_s.should eq("TypeError")
    fixture["upstream"]["to_dense_failure"]["message"].as_s.should contain("list")
    fixture["upstream"]["to_dense_failure"]["message"].as_s.should contain("tuple")

    input = fixture["input"]
    coordinates = sparse_map_coordinates(input["coordinates"])
    sparse_map_i32le_sha256(coordinates).should eq(
      input["coordinates_i32le_sha256"].as_s
    )
    spatial = input["spatial_shape"].as_a.map { |value| value.as_i.to_i32 }
    map = ML::Sparse::CoordinateMap3D.new(
      coordinates,
      input["batch_size"].as_i.to_i32,
      {spatial[0], spatial[1], spatial[2]}
    )

    map.batch_size.should eq(3)
    map.point_count.should eq(6)
    map.spatial_shape.should eq({4, 3, 4})
    map.occupied_spatial_shape.should eq({4, 3, 4})
    map.coordinates_copy.should eq(coordinates)
    map.layout.map { |part| [part.start, part.stop] }.should eq(
      fixture["upstream"]["layout"].as_a.map do |part|
        part.as_a.map { |value| value.as_i.to_i32 }
      end
    )
    map.sequence_lengths.should eq([2, 0, 4])
    map.cumulative_sequence_lengths.should eq([0, 2, 2, 6])
    map.batch_broadcast_map.should eq([0, 0, 2, 2, 2, 2])
    map.coordinate(0, 1).should eq(2)
    map.coordinate(1, 1).should eq(0)

    expected_tuple_queries = [
      {[2, 3, 1, 2], 4_i32},
      {[0, 0, 1, 0], 1_i32},
      {[2, 0, 0, 3], 3_i32},
    ]
    fixture["independent_tuple_lookup_queries"].as_a.first(3).each_with_index do |entry, index|
      coordinate = entry["coordinate"].as_a.map { |value| value.as_i.to_i32 }
      coordinate.should eq(expected_tuple_queries[index][0])
      map.index_of(
        coordinate[0], coordinate[1], coordinate[2], coordinate[3]
      ).should eq(expected_tuple_queries[index][1])
    end
    missing_query = fixture["independent_tuple_lookup_queries"].as_a.last
    missing_query["coordinate"].as_a.map(&.as_i).should eq([1, 0, 0, 0])
    missing_query["row"].raw.should be_nil
    map.index_of(1, 0, 0, 0).should be_nil

    same = ML::Sparse::CoordinateMap3D.new(coordinates, 3, {4, 3, 4})
    same.digest.should eq(map.digest)
    reordered = coordinates.dup
    first = reordered[0, 4].dup
    reordered[0, 4] = reordered[4, 4]
    reordered[4, 4] = first
    ML::Sparse::CoordinateMap3D.new(reordered, 3, {4, 3, 4}).digest
      .should_not eq(map.digest)

    coordinates[1] = 3
    map.coordinate(0, 1).should eq(2)
    copy = map.coordinates_copy
    copy[1] = 1
    map.coordinate(0, 1).should eq(2)
  end

  it "represents empty batches with explicit extents" do
    map = ML::Sparse::CoordinateMap3D.new([] of Int32, 3, {8, 7, 6})
    map.point_count.should eq(0)
    map.occupied_spatial_shape.should eq({0, 0, 0})
    map.layout.map { |part| [part.start, part.stop] }.should eq(
      [[0, 0], [0, 0], [0, 0]]
    )
    map.sequence_lengths.should eq([0, 0, 0])
    map.cumulative_sequence_lengths.should eq([0, 0, 0, 0])
    map.batch_broadcast_map.should be_empty
    map.index_of(0, 0, 0, 0).should be_nil

    boundary = ML::Sparse::CoordinateMap3D.new(
      [62, 1_023, 1_023, 1_023, 63, 1_023, 1_023, 1_023] of Int32,
      64,
      {1_024, 1_024, 1_024}
    )
    boundary.index_of(62, 1_023, 1_023, 1_023).should eq(0)
    boundary.index_of(63, 1_023, 1_023, 1_023).should eq(1)
    boundary.digest.should_not eq(map.digest)

    trailing_empty = ML::Sparse::CoordinateMap3D.new(
      [0, 2, 0, 0, 1, 1, 1, 1] of Int32,
      3,
      {4, 4, 4}
    )
    trailing_empty.layout.map { |part| [part.start, part.stop] }.should eq(
      [[0, 1], [1, 2], [2, 2]]
    )
    trailing_empty.sequence_lengths.should eq([1, 1, 0])
  end

  it "fails closed on ambiguous or malformed coordinate maps" do
    expect_raises(ML::Sparse::CoordinateMapError, /multiple of 4/) do
      ML::Sparse::CoordinateMap3D.new([0, 1, 2] of Int32, 1, {4, 4, 4})
    end
    expect_raises(ML::Sparse::CoordinateMapError, /duplicate/) do
      ML::Sparse::CoordinateMap3D.new(
        [0, 1, 1, 1, 0, 1, 1, 1] of Int32, 1, {4, 4, 4}
      )
    end
    expect_raises(ML::Sparse::CoordinateMapError, /nondecreasing batch/) do
      ML::Sparse::CoordinateMap3D.new(
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0] of Int32,
        2,
        {4, 4, 4}
      )
    end
    expect_raises(ML::Sparse::CoordinateMapError, /batch.*range/) do
      ML::Sparse::CoordinateMap3D.new([2, 0, 0, 0] of Int32, 2, {4, 4, 4})
    end
    expect_raises(ML::Sparse::CoordinateMapError, /x.*non-negative/) do
      ML::Sparse::CoordinateMap3D.new([0, -1, 0, 0] of Int32, 1, {4, 4, 4})
    end
    expect_raises(ML::Sparse::CoordinateMapError, /y.*declared extent/) do
      ML::Sparse::CoordinateMap3D.new([0, 0, 4, 0] of Int32, 1, {4, 4, 4})
    end
    expect_raises(ML::Sparse::CoordinateMapError, /batch size/) do
      ML::Sparse::CoordinateMap3D.new([] of Int32, 0, {4, 4, 4})
    end
    expect_raises(ML::Sparse::CoordinateMapError, /batch size/) do
      ML::Sparse::CoordinateMap3D.new([] of Int32, 65, {4, 4, 4})
    end
    expect_raises(ML::Sparse::CoordinateMapError, /spatial extent/) do
      ML::Sparse::CoordinateMap3D.new([] of Int32, 1, {4, 0, 4})
    end
    expect_raises(ML::Sparse::CoordinateMapError, /10-bit/) do
      ML::Sparse::CoordinateMap3D.new([] of Int32, 1, {1025, 4, 4})
    end

    too_many = Array(Int32).new(
      (ML::Sparse::CoordinateMap3D::MAX_POINTS + 1) * 4,
      0_i32
    )
    expect_raises(ML::Sparse::CoordinateMapBudgetError, /point count/) do
      ML::Sparse::CoordinateMap3D.new(too_many, 1, {1, 1, 1})
    end
  end
end
