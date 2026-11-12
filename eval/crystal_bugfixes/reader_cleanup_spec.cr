require "spec"
require "../src/ml/gguf/reader"

private def open_fd_count : Int32?
  path = if Dir.exists?("/proc/self/fd")
           "/proc/self/fd"
         elsif Dir.exists?("/dev/fd")
           "/dev/fd"
         end
  path ? Dir.children(path).size : nil
end

describe ML::GGUF::GGUFFile do
  it "closes its file when malformed metadata fails during construction" do
    before = open_fd_count
    pending!("open file descriptor inventory unavailable") unless before
    path = File.tempname("malformed-gguf", ".gguf")
    File.write(path, "not a GGUF file")

    begin
      32.times do
        expect_raises(Exception, /GGUF/) do
          ML::GGUF::GGUFFile.new(path)
        end
      end
      (open_fd_count.not_nil! - before).should be <= 2
    ensure
      File.delete(path) if File.exists?(path)
    end
  end
end
