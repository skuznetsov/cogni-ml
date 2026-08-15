require "./spec_helper"
require "../src/ml/gguf/qwen_qbit_session_checkpoint"

private def checkpoint_anchor(tokens : Array(Int32), checkpoint_id : String = "a" * 64)
  ML::GGUF::QwenQBitSessionCheckpoint.build_anchor(
    session_id: "session-a",
    checkpoint_id: checkpoint_id,
    parent_checkpoint_id: nil,
    anchor_cache_id: 42_u64,
    anchor_lookup_key: "b" * 64,
    anchor_generation_id: "c" * 64,
    anchor_certificate_id: "d" * 64,
    token_ids: tokens,
    boundary_text: "rendered-boundary-a",
    expires_at_unix: 10_000_i64,
    created_at_unix: 1_000_i64,
  )
end

describe ML::GGUF::QwenQBitSessionCheckpoint do
  checkpoints = ML::GGUF::QwenQBitSessionCheckpoint

  it "binds a full anchor without storing a raw session identity" do
    tokens = [11_i32, 22_i32, 33_i32]
    entry = checkpoint_anchor(tokens)

    entry.session_hash.should eq(checkpoints.session_hash("session-a"))
    entry.to_json.should_not contain("session-a")
    entry.depth.should eq(0)
    entry.cumulative_token_ids.should be_empty
    entry.anchor_token_ids.should eq(tokens)
    checkpoints.validate_boundary!(entry, "session-a", "rendered-boundary-a<|im_end|>")
    checkpoints.validate!(entry, "session-a", tokens, checkpoint_id: entry.checkpoint_id)

    expect_raises(ArgumentError, /session/) do
      checkpoints.validate!(entry, "session-b", tokens, checkpoint_id: entry.checkpoint_id)
    end
    expect_raises(ArgumentError, /text boundary/) do
      checkpoints.validate_boundary!(entry, "session-a", "rendered-boundary-b<|im_end|>")
    end
  end

  it "stores an exact cumulative token delta rooted at one immutable anchor" do
    anchor_tokens = [11_i32, 22_i32, 33_i32]
    anchor = checkpoint_anchor(anchor_tokens)
    child_tokens = anchor_tokens + [44_i32, 55_i32]
    child = checkpoints.build_delta(
      session_id: "session-a",
      checkpoint_id: "e" * 64,
      parent: anchor,
      token_ids: child_tokens,
      boundary_text: "rendered-boundary-a<|im_end|>child",
      created_at_unix: 1_100_i64,
    )

    child.parent_checkpoint_id.should eq(anchor.checkpoint_id)
    child.anchor_lookup_key.should eq(anchor.anchor_lookup_key)
    child.anchor_generation_id.should eq(anchor.anchor_generation_id)
    child.depth.should eq(1)
    child.cumulative_token_ids.should eq([44_i32, 55_i32])
    child.expires_at_unix.should eq(anchor.expires_at_unix)
    checkpoints.validate!(child, "session-a", child_tokens, checkpoint_id: child.checkpoint_id)

    changed = child_tokens.dup
    changed[4] = 56
    expect_raises(ArgumentError, /child token hash|delta tokens/) do
      checkpoints.validate!(child, "session-a", changed, checkpoint_id: child.checkpoint_id)
    end
  end

  it "supports branches while bounding replay depth and token count" do
    anchor_tokens = [11_i32, 22_i32, 33_i32]
    anchor = checkpoint_anchor(anchor_tokens)
    left = checkpoints.build_delta(
      session_id: "session-a",
      checkpoint_id: "1" * 64,
      parent: anchor,
      token_ids: anchor_tokens + [44_i32],
      boundary_text: "rendered-boundary-a<|im_end|>left",
      created_at_unix: 1_100_i64,
    )
    right = checkpoints.build_delta(
      session_id: "session-a",
      checkpoint_id: "2" * 64,
      parent: anchor,
      token_ids: anchor_tokens + [45_i32],
      boundary_text: "rendered-boundary-a<|im_end|>right",
      created_at_unix: 1_101_i64,
    )

    left.parent_checkpoint_id.should eq(right.parent_checkpoint_id)
    left.child_token_hash.should_not eq(right.child_token_hash)
    checkpoints.delta_admissible?(left, anchor_tokens + [44_i32, 66_i32]).should be_true
    checkpoints.delta_admissible?(
      left,
      anchor_tokens + Array(Int32).new(ML::GGUF::QwenQBitSessionCheckpoint::MAX_REPLAY_TOKENS + 1, 66_i32),
    ).should be_false

    chain = anchor
    chain_tokens = anchor_tokens
    1.upto(ML::GGUF::QwenQBitSessionCheckpoint::MAX_DELTA_DEPTH) do |depth|
      chain_tokens += [100_i32 + depth]
      chain = checkpoints.build_delta(
        session_id: "session-a",
        checkpoint_id: depth.to_s(16) * 64,
        parent: chain,
        token_ids: chain_tokens,
        boundary_text: "rendered-boundary-a<|im_end|>depth-#{depth}",
        created_at_unix: 1_200_i64 + depth,
      )
    end
    chain.depth.should eq(ML::GGUF::QwenQBitSessionCheckpoint::MAX_DELTA_DEPTH)
    checkpoints.delta_admissible?(chain, chain_tokens + [999_i32]).should be_false

    tampered = ML::GGUF::QwenQBitSessionCheckpoint::Entry.from_json(left.to_json)
    tampered.parent_checkpoint_id = "3" * 64
    expect_raises(ArgumentError, /certificate/) do
      checkpoints.validate!(tampered, "session-a", anchor_tokens + [44_i32])
    end
  end
end
