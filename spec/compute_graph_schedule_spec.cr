require "spec"
require "../src/ml/metal/compute_graph_schedule"

private alias ScheduleAccess = ML::Metal::CogniGraphSchedule::Access
private alias AccessMode = ML::Metal::CogniGraphSchedule::AccessMode

private def access(buffer_id : UInt64, mode : AccessMode,
                   offset : Int64 = 0_i64, length : Int64 = -1_i64,
                   partition : Int32 = -1_i32) : ScheduleAccess
  ScheduleAccess.new(buffer_id, offset, length, mode, partition)
end

describe ML::Metal::CogniGraphSchedule do
  it "groups independent accesses and delays conflicting readers and writers" do
    accesses = [
      access(1_u64, AccessMode::Read),
      access(2_u64, AccessMode::Write),
      access(1_u64, AccessMode::Read),
      access(3_u64, AccessMode::Write),
      access(2_u64, AccessMode::Read),
      access(3_u64, AccessMode::Read),
      access(1_u64, AccessMode::Write),
    ]

    schedule = ML::Metal::CogniGraphSchedule.compile([0_u32, 2_u32, 4_u32, 7_u32], accesses)

    schedule.wave_of.should eq([0, 0, 1])
    schedule.wave_count.should eq(2_u32)
  end

  it "uses byte ranges and partition keys when finding dependencies" do
    accesses = [
      access(7_u64, AccessMode::Write, offset: 0_i64, length: 4_i64),
      access(7_u64, AccessMode::Read, offset: 4_i64, length: 4_i64),
      access(8_u64, AccessMode::Write, partition: 0_i32),
      access(8_u64, AccessMode::Write, partition: 1_i32),
    ]

    schedule = ML::Metal::CogniGraphSchedule.compile([0_u32, 1_u32, 2_u32, 3_u32, 4_u32], accesses)

    schedule.wave_of.should eq([0, 0, 0, 0])
    schedule.wave_count.should eq(1_u32)
  end

  it "treats read-write bindings as both dependencies and writers" do
    accesses = [
      access(4_u64, AccessMode::ReadWrite),
      access(4_u64, AccessMode::Read),
      access(4_u64, AccessMode::Write),
    ]

    schedule = ML::Metal::CogniGraphSchedule.compile([0_u32, 1_u32, 2_u32, 3_u32], accesses)

    schedule.wave_of.should eq([0, 1, 2])
    schedule.wave_count.should eq(3_u32)
  end

  it "exposes a stable callback pointer that accepts the native ABI" do
    sizeof(ML::Metal::CogniGraphSchedule::NativeAccess).should eq(32)

    native_accesses = [
      ML::Metal::CogniGraphSchedule::NativeAccess.new(10_u64, 0_i64, -1_i64, 0_i32, -1_i32),
      ML::Metal::CogniGraphSchedule::NativeAccess.new(11_u64, 0_i64, -1_i64, 1_i32, -1_i32),
      ML::Metal::CogniGraphSchedule::NativeAccess.new(11_u64, 0_i64, -1_i64, 0_i32, -1_i32),
    ]
    offsets = [0_u32, 2_u32, 3_u32]
    wave_of = Slice(Int32).new(2, -1_i32)
    wave_count = 99_u32

    callback = ML::Metal::CogniGraphSchedule.callback
    callback.pointer.null?.should be_false
    callback.pointer.should eq(ML::Metal::CogniGraphSchedule.callback.pointer)
    result = callback.call(native_accesses.to_unsafe, offsets.to_unsafe,
      2_u32, 3_u32, wave_of.to_unsafe, pointerof(wave_count))

    result.should eq(0)
    wave_of.to_a.should eq([0, 1])
    wave_count.should eq(2_u32)
  end

  it "rejects malformed arrays and unknown native access codes without raising" do
    native_accesses = [
      ML::Metal::CogniGraphSchedule::NativeAccess.new(10_u64, 0_i64, -1_i64, 7_i32, -1_i32),
    ]
    malformed_offsets = [0_u32, 0_u32]
    wave_of = Slice(Int32).new(1, -1_i32)
    wave_count = 99_u32

    result = ML::Metal::CogniGraphSchedule.compile_callback(
      native_accesses.to_unsafe, malformed_offsets.to_unsafe,
      1_u32, 1_u32, wave_of.to_unsafe, pointerof(wave_count))

    result.should be < 0
    wave_count.should eq(0_u32)
  end

  it "rejects callback inputs above the bounded native limits" do
    offsets = Slice(UInt32).new(258) { |index| index.to_u32 }
    wave_of = Slice(Int32).new(257, -1_i32)
    wave_count = 99_u32

    result = ML::Metal::CogniGraphSchedule.compile_callback(
      Pointer(ML::Metal::CogniGraphSchedule::NativeAccess).null,
      offsets.to_unsafe, 257_u32, 0_u32, wave_of.to_unsafe, pointerof(wave_count))

    result.should be < 0
    wave_count.should eq(0_u32)
  end
end
