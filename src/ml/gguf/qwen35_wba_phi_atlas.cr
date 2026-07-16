module ML::GGUF::Qwen35WbaPhiAtlas
  enum Verdict
    Refuted
    CandidateDescent
    ProfileOnly
    InsufficientEvidence
  end

  record GroupRow,
    name : String,
    calls : Int32,
    encode_ms : Float64,
    wait_ms : Float64,
    read_ms : Float64,
    upload_mib : Float64,
    readback_mib : Float64,
    gpu_ms : Float64 do
    def signal_ms : Float64
      @gpu_ms > 0 ? @gpu_ms : @wait_ms
    end

    def boundary_transfer? : Bool
      @upload_mib > 0 || @readback_mib > 0
    end
  end

  record TrafficRow,
    name : String,
    calls : Int32,
    mib : Float64,
    pct : Float64,
    kind : String

  record PhaseFamily,
    name : String,
    calls : Int32,
    wait_ms : Float64,
    gpu_ms : Float64,
    signal_ms : Float64

  record WorkFamily,
    name : String,
    calls : Int32,
    mib : Float64,
    kind : String

  record AbResult,
    env : String,
    other_label : String,
    default_p50_ms : Float64,
    other_p50_ms : Float64,
    default_wins : Int32,
    pairs : Int32 do
    def other_minus_default_ms : Float64
      @other_p50_ms - @default_p50_ms
    end

    def other_faster? : Bool
      @other_p50_ms < @default_p50_ms
    end

    def default_won_majority? : Bool
      @pairs > 0 && @default_wins.to_f > (@pairs.to_f / 2.0)
    end
  end

  record Phi,
    active_window : String,
    active_signal_ms : Float64,
    tied_dominant_routes : Int32,
    bad_corners : Array(String),
    remaining_work_mib : Float64 do
    def tuple : String
      "#{@active_window}:#{Qwen35WbaPhiAtlas.fmt(@active_signal_ms)}ms, #{@tied_dominant_routes}, #{@bad_corners.size}, #{Qwen35WbaPhiAtlas.fmt(@remaining_work_mib)}MiB"
    end
  end

  class Analysis
    getter source : String
    getter groups : Array(GroupRow)
    getter phase_families : Array(PhaseFamily)
    getter work_families : Array(WorkFamily)
    getter matmuls : Array(TrafficRow)
    getter conversions : Array(TrafficRow)
    getter total_syncs : Int32?
    getter cpu_fallback : Int32
    getter ab_result : AbResult?
    getter phi : Phi
    getter verdict : Verdict

    def initialize(@source : String,
                   @groups : Array(GroupRow),
                   @phase_families : Array(PhaseFamily),
                   @work_families : Array(WorkFamily),
                   @matmuls : Array(TrafficRow),
                   @conversions : Array(TrafficRow),
                   @total_syncs : Int32?,
                   @cpu_fallback : Int32,
                   @ab_result : AbResult?,
                   @phi : Phi,
                   @verdict : Verdict)
    end

    def verdict_label : String
      case @verdict
      when Verdict::Refuted              then "REFUTED"
      when Verdict::CandidateDescent     then "CANDIDATE_DESCENT"
      when Verdict::ProfileOnly          then "PROFILE_ONLY"
      when Verdict::InsufficientEvidence then "INSUFFICIENT_EVIDENCE"
      else
        @verdict.to_s.upcase
      end
    end

    def render : String
      String.build do |io|
        io << "Qwen35 WBA Phi Atlas\n"
        io << "  source: " << @source << "\n"
        io << "  Verdict: " << verdict_label << "\n"
        io << "  Phi=(I, M, P, Area)=(" << @phi.tuple << ")\n"
        io << "  active_window: " << @phi.active_window << "\n"
        io << "  remaining_work: matmul=" << Qwen35WbaPhiAtlas.fmt(total_matmul_mib) << "MiB conversion=" << Qwen35WbaPhiAtlas.fmt(total_conversion_mib) << "MiB\n"
        io << "  syncs: " << (@total_syncs || 0).to_s << " cpu_fallback=" << @cpu_fallback << "\n"

        if ab = @ab_result
          io << "  paired_certificate: " << ab.env << " default_p50=" << Qwen35WbaPhiAtlas.fmt(ab.default_p50_ms)
          io << "ms other(" << ab.other_label << ")_p50=" << Qwen35WbaPhiAtlas.fmt(ab.other_p50_ms)
          io << "ms other_minus_default=" << Qwen35WbaPhiAtlas.signed_fmt(ab.other_minus_default_ms)
          io << "ms default_wins=" << ab.default_wins << "/" << ab.pairs << "\n"
        else
          io << "  paired_certificate: missing\n"
        end

        io << "\n"
        io << "WBA card\n"
        io << "  Window or trigger: " << window_trigger << "\n"
        io << "  Transport corridor: " << transport_corridor << "\n"
        io << "  Legal move: " << legal_move << "\n"
        io << "  Boundary safety: exact parity/spec gate plus rollback/default route is required before promotion\n"
        io << "  Lexicographic potential: Phi=(active_window_signal, tied_dominant_routes, bad_corner_count, remaining_work_mib)\n"
        io << "  Recompute safety: " << recompute_safety << "\n"
        io << "  Dual frame: exact baseline/profile-only fallback; do not stack the move if paired timing does not descend\n"
        io << "  Local certificate: " << local_certificate << "\n"

        unless @phi.bad_corners.empty?
          io << "\n"
          io << "Bad corners\n"
          @phi.bad_corners.each { |corner| io << "  - " << corner << "\n" }
        end

        unless @groups.empty?
          io << "\n"
          io << "Top timed groups\n"
          @groups.sort_by { |g| {-g.signal_ms, g.name} }.first(5).each do |g|
            io << "  - " << g.name << ": signal=" << Qwen35WbaPhiAtlas.fmt(g.signal_ms)
            io << "ms wait=" << Qwen35WbaPhiAtlas.fmt(g.wait_ms) << "ms gpu=" << Qwen35WbaPhiAtlas.fmt(g.gpu_ms)
            io << "ms upload=" << Qwen35WbaPhiAtlas.fmt(g.upload_mib) << "MiB readback=" << Qwen35WbaPhiAtlas.fmt(g.readback_mib) << "MiB\n"
          end
        end

        unless @phase_families.empty?
          io << "\n"
          io << "Top diagnostic phase families\n"
          @phase_families.sort_by { |f| {-f.signal_ms, f.name} }.first(5).each do |f|
            io << "  - " << f.name << ": signal=" << Qwen35WbaPhiAtlas.fmt(f.signal_ms)
            io << "ms wait=" << Qwen35WbaPhiAtlas.fmt(f.wait_ms) << "ms gpu=" << Qwen35WbaPhiAtlas.fmt(f.gpu_ms)
            io << "ms calls=" << f.calls << "\n"
          end
        end

        unless @work_families.empty?
          io << "\n"
          io << "Top logical work families\n"
          @work_families.sort_by { |f| {-f.mib, f.name} }.first(5).each do |f|
            io << "  - " << f.name << ": " << Qwen35WbaPhiAtlas.fmt(f.mib)
            io << "MiB calls=" << f.calls << " kind=" << f.kind << "\n"
          end
        end

        unless @matmuls.empty?
          io << "\n"
          io << "Top logical matmul rows\n"
          @matmuls.sort_by { |m| {-m.mib, m.name} }.first(5).each do |m|
            io << "  - " << m.name << ": " << Qwen35WbaPhiAtlas.fmt(m.mib) << "MiB " << Qwen35WbaPhiAtlas.fmt(m.pct) << "%\n"
          end
        end
      end
    end

    def summary_tsv : String
      ab = @ab_result
      [
        @source,
        verdict_label,
        @phi.active_window,
        Qwen35WbaPhiAtlas.fmt(@phi.active_signal_ms),
        @phi.tied_dominant_routes,
        @phi.bad_corners.size,
        Qwen35WbaPhiAtlas.fmt(@phi.remaining_work_mib),
        ab.try(&.env) || "",
        ab ? Qwen35WbaPhiAtlas.fmt(ab.default_p50_ms) : "",
        ab ? Qwen35WbaPhiAtlas.fmt(ab.other_p50_ms) : "",
        ab ? "#{ab.default_wins}/#{ab.pairs}" : "",
      ].join('\t')
    end

    def total_matmul_mib : Float64
      @matmuls.sum(&.mib)
    end

    def total_conversion_mib : Float64
      @conversions.sum(&.mib)
    end

    private def window_trigger : String
      if @phi.active_window == "none"
        "no timed/logical active window parsed"
      elsif @phi.active_window.starts_with?("phase:")
        "dominant diagnostic phase family '#{@phi.active_window.lchop("phase:")}'"
      elsif @phi.active_window.starts_with?("work:")
        "dominant logical work family '#{@phi.active_window.lchop("work:")}'"
      elsif @groups.any? { |g| g.name == @phi.active_window }
        "dominant timed command-buffer bucket '#{@phi.active_window}'"
      else
        "dominant logical work row '#{@phi.active_window}'"
      end
    end

    private def transport_corridor : String
      name = @phi.active_window
      if name.includes?("prefill") && name.includes?("append")
        "prompt hidden rows through the appended prefill command-buffer corridor"
      elsif name.starts_with?("phase:rec.") || name.starts_with?("phase:full.")
        "diagnostic non-appended prefill phase corridor; use only to choose the next production-safe probe"
      elsif name.starts_with?("work:prefill.")
        "logical prefill operation family through quantized weights into exact layer state"
      elsif name.includes?("q8_gemm") || dominant_matmul_name.includes?("q8_gemm")
        "Q8_0 prompt rows through quantized weight tiles into exact layer state"
      elsif name.includes?("conversion") || total_conversion_mib > 0
        "activation dtype/staging corridor between producer and consumer kernels"
      else
        "profiled route band; requires a more specific source/consumer certificate before optimization"
      end
    end

    private def legal_move : String
      case @verdict
      when Verdict::Refuted
        "no promotion; paired recomputation refuted descent"
      when Verdict::CandidateDescent
        "candidate only; add correctness/adversary certificate before promotion"
      when Verdict::ProfileOnly
        "measurement window only; run paired A/B before claiming a legal move"
      when Verdict::InsufficientEvidence
        "none; profile lacks enough evidence to choose a move"
      else
        "none; unrecognized verdict"
      end
    end

    private def recompute_safety : String
      if ab = @ab_result
        if ab.other_faster?
          "paired wall p50 descended by #{Qwen35WbaPhiAtlas.fmt(-ab.other_minus_default_ms)}ms; still needs parity/adversary evidence"
        else
          "paired wall p50 did not descend; other-default=#{Qwen35WbaPhiAtlas.signed_fmt(ab.other_minus_default_ms)}ms"
        end
      else
        "not established; no paired timing certificate"
      end
    end

    private def local_certificate : String
      if ab = @ab_result
        "#{@source}; #{ab.env} default_wins=#{ab.default_wins}/#{ab.pairs}"
      else
        @source
      end
    end

    private def dominant_matmul_name : String
      @matmuls.max_by?(&.mib).try(&.name) || ""
    end
  end

  def self.analyze(text : String, source : String = "stdin") : Analysis
    groups = [] of GroupRow
    matmuls = [] of TrafficRow
    conversions = [] of TrafficRow
    total_syncs = nil.as(Int32?)
    cpu_fallback = 0
    section = nil.as(Symbol?)

    ab_env = nil.as(String?)
    ab_other_label = nil.as(String?)
    ab_default_p50 = nil.as(Float64?)
    ab_other_p50 = nil.as(Float64?)
    ab_default_wins = nil.as(Int32?)
    ab_pairs = nil.as(Int32?)

    text.each_line do |line|
      stripped = line.strip
      case stripped
      when "grouped command buffers:"
        section = :groups
        next
      when "matmul shapes:"
        section = :matmuls
        next
      when "conversion kernels:"
        section = :conversions
        next
      end

      if m = stripped.match(/^cpu_fallback matvecs:\s+(\d+)/)
        cpu_fallback = m[1].to_i
        next
      end

      if m = stripped.match(/^total metal syncs:\s+(\d+)/)
        total_syncs = m[1].to_i
        next
      end

      if m = stripped.match(/^A\/B\s+([A-Za-z0-9_]+):\s+default vs\s+(.+?)\s+\(paired/)
        ab_env = m[1]
        ab_other_label = m[2].strip
        next
      end

      if m = stripped.match(/^default:\s+avg=[0-9.]+ ms p50=([0-9.]+) ms/)
        ab_default_p50 = m[1].to_f
        next
      end

      if m = stripped.match(/^other:\s+avg=[0-9.]+ ms p50=([0-9.]+) ms/)
        ab_other_p50 = m[1].to_f
        next
      end

      if m = stripped.match(/^default-other:\s+[-+0-9.]+ ms\s+wins=(\d+)\/(\d+)/)
        ab_default_wins = m[1].to_i
        ab_pairs = m[2].to_i
        next
      end

      case section
      when :groups
        if m = stripped.match(/^(.+?)\s+(\d+) calls\s+encode\s+([0-9.]+) ms\s+wait\s+([0-9.]+) ms\s+read\s+([0-9.]+) ms\s+upload\s+([0-9.]+) MiB\s+readback\s+([0-9.]+) MiB\s+gpu\s+([0-9.]+) ms/)
          groups << GroupRow.new(m[1].strip, m[2].to_i, m[3].to_f, m[4].to_f, m[5].to_f, m[6].to_f, m[7].to_f, m[8].to_f)
        elsif m = stripped.match(/^(.+?)\s+(\d+) calls\s+encode\s+([0-9.]+) ms\s+wait\s+([0-9.]+) ms\s+read\s+([0-9.]+) ms\s+upload\s+([0-9.]+) MiB\s+readback\s+([0-9.]+) MiB/)
          groups << GroupRow.new(m[1].strip, m[2].to_i, m[3].to_f, m[4].to_f, m[5].to_f, m[6].to_f, m[7].to_f, 0.0)
        elsif m = stripped.match(/^(.+?)\s+(\d+) calls\s+encode\s+([0-9.]+) ms\s+wait\s+([0-9.]+) ms\s+read\s+([0-9.]+) ms/)
          groups << GroupRow.new(m[1].strip, m[2].to_i, m[3].to_f, m[4].to_f, m[5].to_f, 0.0, 0.0, 0.0)
        elsif !stripped.empty? && !stripped.includes?("calls")
          section = nil
        end
      when :matmuls
        if m = stripped.match(/^(.+?)\s+(\d+) calls\s+([0-9.]+) MiB logical weights\s+([0-9.]+)%/)
          matmuls << TrafficRow.new(m[1].strip, m[2].to_i, m[3].to_f, m[4].to_f, "matmul")
        elsif stripped.starts_with?("matmul") || stripped.starts_with?("logical traffic")
          section = nil
        end
      when :conversions
        if m = stripped.match(/^(.+?)\s+(\d+) calls\s+([0-9.]+) MiB logical traffic\s+([0-9.]+)%/)
          conversions << TrafficRow.new(m[1].strip, m[2].to_i, m[3].to_f, m[4].to_f, "conversion")
        elsif stripped.starts_with?("conversion") || stripped.starts_with?("logical traffic")
          section = nil
        end
      end
    end

    ab_result = if ab_env && ab_other_label && ab_default_p50 && ab_other_p50
                  AbResult.new(
                    ab_env.not_nil!,
                    ab_other_label.not_nil!,
                    ab_default_p50.not_nil!,
                    ab_other_p50.not_nil!,
                    ab_default_wins || 0,
                    ab_pairs || 0
                  )
                end

    phase_families = build_phase_families(groups)
    work_families = build_work_families(matmuls, conversions)
    phi = build_phi(groups, phase_families, work_families, matmuls, conversions, total_syncs, cpu_fallback, ab_result)
    verdict = classify(groups, matmuls, conversions, ab_result)
    Analysis.new(source, groups, phase_families, work_families, matmuls, conversions, total_syncs, cpu_fallback, ab_result, phi, verdict)
  end

  def self.analyze_paths(paths : Array(String), combine : Bool = false) : Array(Analysis)
    if paths.empty?
      [analyze(STDIN.gets_to_end, source: "stdin")]
    elsif combine
      [analyze(paths.map { |path| File.read(path) }.join("\n"), source: paths.join(","))]
    else
      paths.map { |path| analyze(File.read(path), source: path) }
    end
  end

  def self.build_phi(groups : Array(GroupRow),
                     phase_families : Array(PhaseFamily),
                     work_families : Array(WorkFamily),
                     matmuls : Array(TrafficRow),
                     conversions : Array(TrafficRow),
                     total_syncs : Int32?,
                     cpu_fallback : Int32,
                     ab_result : AbResult?) : Phi
    positive_groups = groups.select { |group| group.signal_ms > 0 }
    positive_families = phase_families.select { |family| family.signal_ms > 0 }
    dominant_group = positive_groups.max_by?(&.signal_ms)
    dominant_family = positive_families.max_by?(&.signal_ms)
    dominant_work_family = work_families.max_by?(&.mib)
    dominant_matmul = matmuls.max_by?(&.mib)
    active_window = if dominant_family
                      "phase:#{dominant_family.not_nil!.name}"
                    elsif dominant_group
                      dominant_group.not_nil!.name
                    elsif dominant_work_family
                      "work:#{dominant_work_family.not_nil!.name}"
                    else
                      dominant_matmul.try(&.name) || "none"
                    end
    active_signal = dominant_family.try(&.signal_ms) || dominant_group.try(&.signal_ms) || dominant_work_family.try(&.mib) || dominant_matmul.try(&.mib) || 0.0
    tied = if dominant_family && dominant_family.not_nil!.signal_ms > 0
             threshold = dominant_family.not_nil!.signal_ms * 0.80
             positive_families.count { |f| f.signal_ms >= threshold }
           elsif dominant_group && dominant_group.not_nil!.signal_ms > 0
             threshold = dominant_group.not_nil!.signal_ms * 0.80
             positive_groups.count { |g| g.signal_ms >= threshold }
           elsif dominant_work_family && dominant_work_family.not_nil!.mib > 0
             threshold = dominant_work_family.not_nil!.mib * 0.80
             work_families.count { |f| f.mib >= threshold }
           elsif dominant_matmul && dominant_matmul.not_nil!.mib > 0
             threshold = dominant_matmul.not_nil!.mib * 0.80
             matmuls.count { |m| m.mib >= threshold }
           else
             0
           end
    bad = [] of String

    if ab = ab_result
      if ab.other_faster?
        bad << "promotion_needs_correctness_certificate"
      else
        bad << "paired_move_regresses_wall"
      end
    else
      bad << "missing_paired_timing_certificate"
    end

    bad << "missing_timed_group_certificate" if groups.empty?
    bad << "missing_area_metric" if matmuls.empty? && conversions.empty?
    bad << "cpu_fallback=#{cpu_fallback}" if cpu_fallback > 0
    bad << "sync_count=#{total_syncs}" if total_syncs && total_syncs.not_nil! > 1
    bad << "diagnostic_phase_profile_sync_inflation=#{total_syncs}" if !phase_families.empty? && total_syncs && total_syncs.not_nil! > 16
    boundary_groups = groups.count(&.boundary_transfer?)
    bad << "boundary_transfer_groups=#{boundary_groups}" if boundary_groups > 0
    conversion_mib = conversions.sum(&.mib)
    bad << "conversion_mib=#{Qwen35WbaPhiAtlas.fmt(conversion_mib)}" if conversion_mib > 0
    bad << "tied_dominant_routes=#{tied}" if tied > 1

    Phi.new(active_window, active_signal, tied, bad, matmuls.sum(&.mib) + conversion_mib)
  end

  def self.build_phase_families(groups : Array(GroupRow)) : Array(PhaseFamily)
    calls = Hash(String, Int32).new(0)
    waits = Hash(String, Float64).new(0.0)
    gpus = Hash(String, Float64).new(0.0)
    signals = Hash(String, Float64).new(0.0)

    groups.each do |group|
      next unless group.signal_ms > 0
      family = phase_family_name(group.name)
      next unless family
      key = family.not_nil!
      calls[key] += group.calls
      waits[key] += group.wait_ms
      gpus[key] += group.gpu_ms
      signals[key] += group.signal_ms
    end

    calls.keys.sort.map do |name|
      PhaseFamily.new(name, calls[name], waits[name], gpus[name], signals[name])
    end
  end

  def self.build_work_families(matmuls : Array(TrafficRow), conversions : Array(TrafficRow)) : Array(WorkFamily)
    calls = Hash(String, Int32).new(0)
    mib = Hash(String, Float64).new(0.0)
    kinds = Hash(String, String).new

    (matmuls + conversions).each do |row|
      family = work_family_name(row)
      calls[family] += row.calls
      mib[family] += row.mib
      kinds[family] = row.kind
    end

    calls.keys.sort.map do |name|
      WorkFamily.new(name, calls[name], mib[name], kinds[name])
    end
  end

  def self.phase_family_name(name : String) : String?
    return nil if name.ends_with?(".boundary") || name.ends_with?(".read")
    return "full.kv_only" if name == "full_kv_only"

    if m = name.match(/(?:^|\.)(full|rec\d+)\.([A-Za-z0-9_]+)$/)
      prefix = m[1].starts_with?("rec") ? "rec" : "full"
      "#{prefix}.#{m[2]}"
    end
  end

  def self.work_family_name(row : TrafficRow) : String
    first = row.name.split(/\s+/, 2).first? || row.name
    if first.starts_with?("prefill.")
      first
    else
      "#{row.kind}:#{first}"
    end
  end

  def self.classify(groups : Array(GroupRow),
                    matmuls : Array(TrafficRow),
                    conversions : Array(TrafficRow),
                    ab_result : AbResult?) : Verdict
    if ab = ab_result
      ab.other_faster? ? Verdict::CandidateDescent : Verdict::Refuted
    elsif !groups.empty? || !matmuls.empty? || !conversions.empty?
      Verdict::ProfileOnly
    else
      Verdict::InsufficientEvidence
    end
  end

  def self.fmt(value : Float64) : String
    "%.2f" % value
  end

  def self.signed_fmt(value : Float64) : String
    "%+.2f" % value
  end
end
