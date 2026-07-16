require "./spec_helper"
require "../src/ml/gguf/qwen35_wba_phi_atlas"

private def qwen35_wba_profile_sample(ab_default_p50 : Float64? = nil,
                                      ab_other_p50 : Float64? = nil,
                                      wins : String = "0/0") : String
  ab = if ab_default_p50 && ab_other_p50
         <<-LOG

         A/B QWEN35_Q8_B64_GEMM: default vs "1" (paired interleaved)
           default: avg=52.35 ms p50=#{ab_default_p50} ms 4951.76 tok/s
           other:   avg=53.81 ms p50=#{ab_other_p50} ms 4875.44 tok/s
           default-other: -1.45 ms  wins=#{wins}
         LOG
       else
         ""
       end

  <<-LOG
  Qwen35 prefill attribution

  -- Qwen35Metal.Profile report --
    grouped command buffers:
      prefill.append_cmd    1 calls  encode 0.00 ms  wait 47.90 ms  read 0.00 ms  upload 0.00 MiB  readback 0.00 MiB  gpu 47.59 ms
      rec0-2.boundary       0 calls  encode 0.00 ms  wait 0.00 ms  read 0.00 ms  upload 1.00 MiB  readback 0.00 MiB  gpu 0.00 ms
    matmul shapes:
      prefill.rec.ffn_upgate q8_gemm Q8_0 1024x3584 b256   36 calls  133.88 MiB logical weights  27.51%
      prefill.rec.proj q8_gemm Q8_0 1024x6144 b256   18 calls  114.75 MiB logical weights  23.58%
      prefill.rec.proj q8_gemm Q8_0 1024x2048 b256   18 calls  38.25 MiB logical weights  7.86%
      matmul                                  total  486.69 MiB logical weights
    logical traffic mix: matmul 100.00%  conversion 0.00%
    cpu_fallback matvecs: 0
    total metal syncs: 1
    profiled wall: 51.42 ms  4978.75 tok/s
  #{ab}
  LOG
end

describe ML::GGUF::Qwen35WbaPhiAtlas do
  it "classifies a slower paired move as refuted after recomputing wall potential" do
    analysis = ML::GGUF::Qwen35WbaPhiAtlas.analyze(
      qwen35_wba_profile_sample(ab_default_p50: 51.70, ab_other_p50: 52.51, wins: "7/7"),
      source: "q8-b64-slower.log"
    )

    analysis.verdict.should eq(ML::GGUF::Qwen35WbaPhiAtlas::Verdict::Refuted)
    analysis.ab_result.not_nil!.env.should eq("QWEN35_Q8_B64_GEMM")
    analysis.phi.active_window.should eq("prefill.append_cmd")
    analysis.phi.bad_corners.should contain("paired_move_regresses_wall")
    analysis.render.should contain("Verdict: REFUTED")
  end

  it "marks a faster paired move as a candidate instead of verified promotion" do
    analysis = ML::GGUF::Qwen35WbaPhiAtlas.analyze(
      qwen35_wba_profile_sample(ab_default_p50: 52.51, ab_other_p50: 51.70, wins: "0/7"),
      source: "candidate.log"
    )

    analysis.verdict.should eq(ML::GGUF::Qwen35WbaPhiAtlas::Verdict::CandidateDescent)
    analysis.phi.bad_corners.should contain("promotion_needs_correctness_certificate")
    analysis.render.should contain("Verdict: CANDIDATE_DESCENT")
  end

  it "keeps profile-only logs downgraded when no paired certificate exists" do
    analysis = ML::GGUF::Qwen35WbaPhiAtlas.analyze(
      qwen35_wba_profile_sample,
      source: "profile-only.log"
    )

    analysis.verdict.should eq(ML::GGUF::Qwen35WbaPhiAtlas::Verdict::ProfileOnly)
    analysis.phi.bad_corners.should contain("missing_paired_timing_certificate")
    analysis.render.should contain("Dual frame: exact baseline/profile-only")
  end

  it "keeps multiple log files as separate local certificates by default" do
    first = File.tempname("qwen35-wba-first", ".log")
    second = File.tempname("qwen35-wba-second", ".log")
    begin
      File.write(first, qwen35_wba_profile_sample(ab_default_p50: 51.70, ab_other_p50: 52.51, wins: "7/7"))
      File.write(second, qwen35_wba_profile_sample)

      analyses = ML::GGUF::Qwen35WbaPhiAtlas.analyze_paths([first, second])
      analyses.size.should eq(2)
      analyses[0].verdict.should eq(ML::GGUF::Qwen35WbaPhiAtlas::Verdict::Refuted)
      analyses[1].verdict.should eq(ML::GGUF::Qwen35WbaPhiAtlas::Verdict::ProfileOnly)

      combined = ML::GGUF::Qwen35WbaPhiAtlas.analyze_paths([first, second], combine: true)
      combined.size.should eq(1)
      combined[0].source.should contain(",")
    ensure
      File.delete(first) if File.exists?(first)
      File.delete(second) if File.exists?(second)
    end
  end

  it "falls back to logical area instead of choosing a zero-signal boundary row as active window" do
    sample = qwen35_wba_profile_sample.lines.reject(&.includes?("prefill.append_cmd")).join("\n")
    analysis = ML::GGUF::Qwen35WbaPhiAtlas.analyze(sample, source: "zero-group.log")

    analysis.groups.map(&.name).should contain("rec0-2.boundary")
    analysis.phi.active_window.should eq("work:prefill.rec.proj")
    analysis.phi.active_signal_ms.should eq(153.0)
    analysis.work_families.find { |family| family.name == "prefill.rec.proj" }.not_nil!.mib.should eq(153.0)
  end

  it "aggregates full-detail checkpoint buckets into phase families before choosing a window" do
    sample = <<-LOG
    Qwen35 prefill attribution

    -- Qwen35Metal.Profile report --
      grouped command buffers:
        rec0-2.rec0.proj      1 calls  encode 0.04 ms  wait 1.17 ms  read 0.00 ms  upload 0.00 MiB  readback 0.00 MiB  gpu 0.00 ms
        rec0-2.rec1.proj      1 calls  encode 0.02 ms  wait 0.73 ms  read 0.00 ms  upload 0.00 MiB  readback 0.00 MiB  gpu 0.00 ms
        full3+rec4-6.rec0.proj    1 calls  encode 0.02 ms  wait 0.72 ms  read 0.00 ms  upload 0.00 MiB  readback 0.00 MiB  gpu 0.00 ms
        full3+rec4-6.full.attn_rows    1 calls  encode 0.01 ms  wait 0.97 ms  read 0.00 ms  upload 0.00 MiB  readback 0.00 MiB  gpu 0.00 ms
        full7+rec8-10.full.attn_rows    1 calls  encode 0.01 ms  wait 0.89 ms  read 0.00 ms  upload 0.00 MiB  readback 0.00 MiB  gpu 0.00 ms
        rec0-2.rec0.ffn_upgate    1 calls  encode 0.01 ms  wait 0.62 ms  read 0.00 ms  upload 0.00 MiB  readback 0.00 MiB  gpu 0.00 ms
        full3+rec4-6.rec1.ffn_upgate    1 calls  encode 0.01 ms  wait 0.60 ms  read 0.00 ms  upload 0.00 MiB  readback 0.00 MiB  gpu 0.00 ms
        rec0-2.boundary       0 calls  encode 0.00 ms  wait 0.00 ms  read 0.00 ms  upload 1.00 MiB  readback 0.00 MiB  gpu 0.00 ms
      matmul shapes:
        prefill.rec.ffn_upgate q8_gemm Q8_0 1024x3584 b256   36 calls  133.88 MiB logical weights  27.51%
        matmul                                  total  486.69 MiB logical weights
      cpu_fallback matvecs: 0
      total metal syncs: 161
    LOG

    analysis = ML::GGUF::Qwen35WbaPhiAtlas.analyze(sample, source: "full-detail.log")

    analysis.phase_families.map(&.name).should contain("rec.proj")
    analysis.phi.active_window.should eq("phase:rec.proj")
    analysis.phi.active_signal_ms.should eq(2.62)
    analysis.phi.bad_corners.should contain("diagnostic_phase_profile_sync_inflation=161")
    analysis.render.should contain("Top diagnostic phase families")
  end
end
