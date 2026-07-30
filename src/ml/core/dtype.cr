# Element data types shared by tensor and storage primitives.

module ML
  enum DType
    # Existing numeric values are explicit because backend or artifact code may
    # persist enum values. New entries must not renumber the admitted surface.
    F32  = 0
    F16  = 1
    I32  = 2
    I64  = 3
    U8   = 4 # For Bool
    BF16 = 5

    def byte_size : Int32
      case self
      in .f32?  then 4
      in .f16?  then 2
      in .bf16? then 2
      in .i32?  then 4
      in .i64?  then 8
      in .u8?   then 1
      end
    end

    def floating? : Bool
      f32? || f16? || bf16?
    end
  end
end
