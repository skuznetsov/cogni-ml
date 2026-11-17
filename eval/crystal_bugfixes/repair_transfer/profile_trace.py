"""Build renamed and independently rewritten Crystal repair-transfer cases."""

from __future__ import annotations

import re


METAL_RENAMES = {
    "graph_lookup_ms": "plan_lookup_ms",
    "cmd_setup_ms": "command_prep_ms",
    "token_write_ms": "input_upload_ms",
    "lengths_write_ms": "shape_upload_ms",
    "prepass_encode_ms": "warmup_encode_ms",
    "graph_encode_ms": "model_encode_ms",
    "submit_wait_ms": "queue_wait_ms",
    "readback_ms": "copyback_ms",
}


def _rename_tokens(text: str, renames: dict[str, str]) -> str:
    for old, new in renames.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return text


def _rename_record_fields(source: str, record: str, renames: dict[str, str]) -> str:
    start = source.index(f"  record {record},")
    end_match = re.search(r"(?m)^  end(?=\n|$)", source[start:])
    if end_match is None:
        raise AssertionError(f"Could not find the end of {record}")
    end = start + end_match.end()
    return source[:start] + _rename_tokens(source[start:end], renames) + source[end:]


def _line_hole(source: str, line_number: int) -> tuple[str, str]:
    if "<HOLE>" in source:
        raise AssertionError("Input source already contains a hole")
    lines = source.splitlines(keepends=True)
    if not 1 <= line_number <= len(lines):
        raise AssertionError((line_number, len(lines)))
    line = lines[line_number - 1]
    content = line.rstrip("\r\n")
    indent = content[: len(content) - len(content.lstrip(" \t"))]
    replacement = content[len(indent) :]
    newline = line[len(content) :]
    lines[line_number - 1] = f"{indent}<HOLE>{newline}"
    return "".join(lines), replacement


def _hole_line(source: str) -> int:
    lines = source.splitlines()
    matches = [index + 1 for index, line in enumerate(lines) if "<HOLE>" in line]
    if len(matches) != 1:
        raise AssertionError(f"Expected one hole, found {len(matches)}")
    return matches[0]


def _excerpt(source: str, row: dict, *, method: str | None = None) -> str:
    lines = source.splitlines()
    hole_line = _hole_line(source)
    if method is not None:
        try:
            start = next(i for i, line in enumerate(lines) if line == method)
        except StopIteration as error:
            raise AssertionError(f"Could not find excerpt method {method!r}") from error
        end = next(
            (i for i in range(start + 1, len(lines)) if lines[i] == "    end"),
            None,
        )
        if end is None or not start + 1 <= hole_line - 1 <= end:
            raise AssertionError((method, hole_line, end))
        return "\n".join(lines[start : end + 1])

    slot_lines = row["case"]["slot"]["lines"]
    first = slot_lines[0] - 1
    last = max(slot_lines[1], hole_line + 3)
    last = min(last, len(lines))
    if not first <= hole_line - 1 < last:
        raise AssertionError((slot_lines, hole_line, len(lines)))
    return "\n".join(lines[first:last])


def _assert_result(template: str, excerpt: str, fixture: str, baseline: str, teacher: str) -> None:
    if template.count("<HOLE>") != 1 or excerpt.count("<HOLE>") != 1:
        raise AssertionError("Template and excerpt must each contain exactly one hole")
    if "<HOLE>" in fixture or "<HOLE>" in baseline or "<HOLE>" in teacher:
        raise AssertionError("A hole leaked into a completed replacement or fixture")


def _metal_guard(renames: dict[str, str]) -> str:
    test = '''it "adds fractional timings with a nonzero readback stage" do
    profile = ML::GGUF::MetalEncodeProfile.new(
      graph_lookup_ms: 0.5, cmd_setup_ms: 1.25, token_write_ms: 2.5,
      lengths_write_ms: 4.0, prepass_encode_ms: 8.0, graph_encode_ms: 16.0,
      submit_wait_ms: 32.0, readback_ms: 64.0,
    )

    profile.total_ms.should eq(128.25)
  end'''
    return _rename_tokens(test, renames)


def _trace_guard() -> str:
    return '''it "keeps a true result on the completed phase" do
    io = IO::Memory.new
    trace = ML::GGUF::QwenPrefillCommandTrace.new(io)

    trace.observe_full_layer(2, 1, 4) { true }.should be_true
    io.to_s.should contain("phase=call_end")
    io.to_s.should_not contain("phase=call_declined")
  end'''


def _metal_rename(row: dict, source: str, fixture: str, drafts: dict[str, str]) -> dict:
    case = row["case"]
    renames = METAL_RENAMES.copy()
    renamed_source = _rename_record_fields(source, "MetalEncodeProfile", renames)
    renamed_fixture = _rename_tokens(fixture, renames)
    template, baseline = _line_hole(renamed_source, case["slot"]["line"])
    teacher = _rename_tokens(case["teacher"], renames)
    renamed_drafts = {arm: _rename_tokens(answer.strip(), renames) for arm, answer in drafts.items()}
    guards = _metal_guard(renames)
    excerpt = _excerpt(template, row)
    _assert_result(template, excerpt, renamed_fixture, baseline, teacher)
    return {
        "template": template,
        "excerpt": excerpt,
        "baseline": baseline,
        "teacher": teacher,
        "drafts": renamed_drafts,
        "fixture": renamed_fixture,
        "guards": guards,
        "description": case["problem"],
        "mechanism": "The ordered stage sum ends before the readback field, so that duration is absent from the profile total.",
        "renames": renames,
    }


def _metal_rewrite(source: str, fixture: str, drafts: dict[str, str], row: dict) -> dict:
    marker = "  record MetalEncodeProfile,"
    record_start = source.index(marker)
    method_start = source.index("    def total_ms : Float64\n", record_start)
    method_end = source.index("\n    end", method_start) + len("\n    end")
    method = '''    def total_ms : Float64
      [
        @graph_lookup_ms,
        @cmd_setup_ms,
        @token_write_ms,
        @lengths_write_ms,
        @prepass_encode_ms,
        @graph_encode_ms,
        @submit_wait_ms,
        <HOLE>,
      ].sum
    end'''
    template = source[:method_start] + method + source[method_end:]
    baseline = "0.0"
    teacher = "@readback_ms"
    adapted_drafts = {arm: answer.strip() for arm, answer in drafts.items()}
    excerpt = _excerpt(template, row, method="    def total_ms : Float64")
    guards = _metal_guard({})
    _assert_result(template, excerpt, fixture, baseline, teacher)
    return {
        "template": template,
        "excerpt": excerpt,
        "baseline": baseline,
        "teacher": teacher,
        "drafts": adapted_drafts,
        "fixture": fixture,
        "guards": guards,
        "description": row["case"]["problem"],
        "mechanism": "The total is reduced from an ordered stage array whose final slot contributes zero instead of the last recorded duration.",
        "renames": {},
    }


def _trace_rename(row: dict, source: str, fixture: str, drafts: dict[str, str]) -> dict:
    case = row["case"]
    renames = {"result": "outcome"}
    renamed_source = _rename_tokens(source, renames)
    template, baseline = _line_hole(renamed_source, case["slot"]["line"])
    teacher = _rename_tokens(case["teacher"], renames)
    adapted_drafts = {arm: _rename_tokens(answer.strip(), renames) for arm, answer in drafts.items()}
    excerpt = _excerpt(template, row)
    guards = _trace_guard()
    _assert_result(template, excerpt, fixture, baseline, teacher)
    return {
        "template": template,
        "excerpt": excerpt,
        "baseline": baseline,
        "teacher": teacher,
        "drafts": adapted_drafts,
        "fixture": fixture,
        "guards": guards,
        "description": case["problem"],
        "mechanism": "The phase uses Boolean truthiness, so false and nil select the same declined event even though the yielded value is returned.",
        "renames": renames,
    }


def _trace_rewrite(source: str, fixture: str, drafts: dict[str, str], row: dict) -> dict:
    lines = source.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if line.startswith("    def observe_full_layer("))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].rstrip("\r\n") == "    end")
    block = "".join(lines[start : end + 1])
    if block.count("        result = yield\n") != 1 or block.count("      result\n") != 1:
        raise AssertionError("Unexpected observe_full_layer body")
    if block.count('      phase = result ? "call_end" : "call_declined"\n') != 1:
        raise AssertionError("Unexpected phase expression in observe_full_layer")
    block = block.replace("        result = yield\n", "        call_return = yield\n", 1)
    block = block.replace(
        '      phase = result ? "call_end" : "call_declined"\n',
        '      phase_table = {true => "call_end", false => "call_declined"}\n'
        '      completed = <HOLE>\n'
        '      phase = phase_table[completed]\n',
        1,
    )
    block = block.replace("      result\n", "      call_return\n", 1)
    template = "".join(lines[:start]) + block + "".join(lines[end + 1 :])
    baseline = "!!call_return"
    teacher = "!call_return.nil?"
    raw_drafts = {arm: answer.strip() for arm, answer in drafts.items()}
    if raw_drafts != {"own": "return false", "peer": "return result"}:
        raise AssertionError(f"Unexpected historical trace drafts: {raw_drafts}")
    adapted_drafts = {"own": "false", "peer": "!!call_return"}
    excerpt = _excerpt(template, row)
    guards = _trace_guard()
    _assert_result(template, excerpt, fixture, baseline, teacher)
    return {
        "template": template,
        "excerpt": excerpt,
        "baseline": baseline,
        "teacher": teacher,
        "drafts": adapted_drafts,
        "fixture": fixture,
        "guards": guards,
        "description": row["case"]["problem"],
        "mechanism": "A Boolean-keyed phase table uses a completion flag derived from the yielded value's truthiness, combining false and nil under one event.",
        "renames": {"result": "call_return"},
    }


def build(case_id: str, mode: str, row: dict, source: str, fixture: str, drafts: dict[str, str]) -> dict:
    """Build one transformed case in pure-alpha-rename or rewrite mode."""
    if row.get("id") != case_id:
        raise ValueError(f"Row {row.get('id')!r} does not match {case_id!r}")
    if mode not in {"rename", "rewrite"}:
        raise ValueError(f"Unsupported transformation mode: {mode!r}")
    if set(drafts) != {"own", "peer"}:
        raise ValueError("Expected exactly the historical own and peer drafts")

    if case_id == "metal-encode-profile-total-stages":
        if mode == "rename":
            return _metal_rename(row, source, fixture, drafts)
        return _metal_rewrite(source, fixture, drafts, row)
    if case_id == "trace-declined-sentinel":
        if mode == "rename":
            return _trace_rename(row, source, fixture, drafts)
        return _trace_rewrite(source, fixture, drafts, row)
    raise ValueError(f"Unsupported case: {case_id!r}")
