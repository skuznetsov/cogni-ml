"""Fresh paired caller-transfer fixtures for binary Crystal flags."""

from __future__ import annotations


_HELPER = '''module FlagSupport
  def self.decode(value : Int32) : Bool
    unless value == 0 || value == 1
      raise ArgumentError.new("flag value must be 0 or 1")
    end
    <HOLE>
  end
end
'''


def _method(signature: str, body: str) -> str:
    return f"def self.{signature}\n{body}\nend"


def _indent(source: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line for line in source.splitlines())


def _family(
    *,
    family: str,
    module_name: str,
    external_label: str,
    caller_name: str,
    caller_input: str,
    public_zero: bool,
    before: tuple[tuple[str, str], ...],
    after: tuple[tuple[str, str], ...],
    regressions: tuple[tuple[str, str], ...],
) -> list[dict]:
    """Return direct and inverted callers for one public contract."""
    expected = [public_zero, not public_zero]  # Public results for inputs 0, 1.
    rows = []

    for variant, caller_expr, helper_truth in (
        ("direct", "FlagSupport.decode(value)", expected),
        ("inverted", "!FlagSupport.decode(value)", [not item for item in expected]),
    ):
        caller_source = _method(
            f"{caller_name}({caller_input} : Int32) : Bool",
            f"  {caller_expr.replace('value', caller_input)}",
        )
        methods = dict(before)
        methods[caller_name] = caller_source
        methods.update(after)
        source = _HELPER + f"\nmodule {module_name}\n"
        source += "\n\n".join(_indent(method, 2) for method in methods.values())
        source += "\nend\n"
        if source.count("<HOLE>") != 1:
            raise AssertionError("template must contain exactly one helper hole")

        teacher = "value == 1" if helper_truth[1] else "value == 0"
        opposite = "value == 0" if helper_truth[1] else "value == 1"
        zero_result, one_result = expected
        zero_desc = f"returns {'true' if zero_result else 'false'} for {external_label} 0"
        one_desc = f"returns {'true' if one_result else 'false'} for {external_label} 1"
        sequence = [0, 1, 0, 0, 1, 0]
        sequence_result = [expected[value] for value in sequence]
        expected_literal = "[" + ", ".join(str(item).lower() for item in sequence_result) + "]"
        invocation = f"{module_name}.{caller_name}"
        regression_specs = "\n\n".join(
            f'  it "{description}" do\n{_indent(body, 4)}\n  end'
            for description, body in regressions
        )
        fixture = f'''require "spec"
require "./subject"

describe {module_name} do
  it "{zero_desc}" do
    {invocation}(0).should eq({str(zero_result).lower()})
  end

  it "{one_desc}" do
    {invocation}(1).should eq({str(one_result).lower()})
  end

  it "keeps decisions stable across an interleaved sequence" do
    [0, 1, 0, 0, 1, 0].map {{ |{caller_input}| {invocation}({caller_input}) }}.should eq({expected_literal})
  end

  it "rejects values outside the binary domain" do
    expect_raises(ArgumentError) {{ {invocation}(-1) }}
    expect_raises(ArgumentError) {{ {invocation}(2) }}
  end

{regression_specs}
end
'''
        issue_verb = "dispatch" if public_zero is False else "retain"
        issue = (
            f"For {module_name}, the external {issue_verb} decision must be "
            f"{'true' if zero_result else 'false'} for {external_label} 0 and "
            f"{'true' if one_result else 'false'} for {external_label} 1."
        )
        rows.append(
            {
                "family": family,
                "variant": variant,
                "helper": _HELPER,
                "methods": methods,
                "caller": caller_name,
                "issue": issue,
                "template": source,
                "teacher": teacher,
                "opposite": opposite,
                "broken": "false",
                "fixture": fixture,
                "truth": list(helper_truth),
                "target_tests": [zero_desc, one_desc],
            }
        )
    return rows


def rows() -> list[dict]:
    """Return two fresh matched families with opposite public flag conventions."""
    queue_methods = (
        (
            "at_batch_boundary?",
            _method(
                "at_batch_boundary?(offset : Int32) : Bool",
                "  offset > 0 && offset % 16 == 0",
            ),
        ),
        (
            "within_capacity?",
            _method(
                "within_capacity?(queued : Int32, capacity : Int32) : Bool",
                "  queued < capacity",
            ),
        ),
    )
    span_methods_before = (
        (
            "overlaps?",
            _method(
                "overlaps?(left_start : Int32, left_end : Int32, right_start : Int32, right_end : Int32) : Bool",
                "  left_start < right_end && right_start < left_end",
            ),
        ),
        (
            "near_segment_edge?",
            _method(
                "near_segment_edge?(position : Int32) : Bool",
                "  position >= 0 && position % 64 == 0",
            ),
        ),
    )
    span_methods_after = (
        (
            "bounded_end",
            _method(
                "bounded_end(requested : Int32, extent : Int32) : Int32",
                "  if requested < extent\n    requested\n  else\n    extent\n  end",
            ),
        ),
    )
    return _family(
        family="qcf-9c71",
        module_name="ChunkIngress",
        external_label="marker",
        caller_name="should_dispatch?",
        caller_input="marker",
        public_zero=False,
        before=((queue_methods[0]),),
        after=((queue_methods[1]),),
        regressions=(
            (
                "keeps batch boundary checks independent",
                "ChunkIngress.at_batch_boundary?(16).should be_true\n"
                "ChunkIngress.at_batch_boundary?(17).should be_false",
            ),
            (
                "keeps capacity checks independent",
                "ChunkIngress.within_capacity?(3, 5).should be_true\n"
                "ChunkIngress.within_capacity?(5, 5).should be_false",
            ),
        ),
    ) + _family(
        family="srf-43d2",
        module_name="SpanLedger",
        external_label="edge code",
        caller_name="should_retain?",
        caller_input="edge_code",
        public_zero=True,
        before=span_methods_before,
        after=span_methods_after,
        regressions=(
            (
                "detects overlapping spans",
                "SpanLedger.overlaps?(2, 8, 6, 10).should be_true\n"
                "SpanLedger.overlaps?(2, 5, 6, 10).should be_false",
            ),
            (
                "keeps edge checks independent",
                "SpanLedger.near_segment_edge?(64).should be_true\n"
                "SpanLedger.near_segment_edge?(65).should be_false",
            ),
            (
                "caps a reported end at the extent",
                "SpanLedger.bounded_end(7, 9).should eq(7)\n"
                "SpanLedger.bounded_end(12, 9).should eq(9)",
            ),
        ),
    )
