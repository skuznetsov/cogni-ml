"""Build renamed and rewritten transfer fixtures for autograd and Qwen masks."""

from __future__ import annotations

import re


_AUTOGRAD = "autograd-mean-gradient-element-count"
_QWEN_MASK = "qwen-image21-conditioning-mask-binary"

_AUTOGRAD_GUARDS = '''
describe ML::Autograd::MeanBackward do
  it "scales a non-unit upstream value over a three-dimensional input" do
    upstream = ML::Tensor.from_array([48.0_f32], ML::Shape.new(1))
    gradient = ML::Autograd::MeanBackward.new(ML::Shape.new(2, 3, 4)).backward(upstream).first.not_nil!

    gradient.shape.should eq(ML::Shape.new(2, 3, 4))
    gradient.to_a.should eq(Array.new(24, 2.0_f32))
  end
end
'''

_QWEN_MASK_GUARDS = '''
describe ML::GGUF::QwenImage21ConditioningBundle do
  it "decodes the reverse image mask ordering" do
    manifest, payload = mask_fixture(valid_mask: [0_u8, 1_u8])
    bundle = ML::GGUF::QwenImage21ConditioningBundle.parse(manifest, payload)

    bundle.encoder_hidden_states_mask.should eq([false, true])
    bundle.encoder_img_mask.should eq([false, false])
  end

  it "rejects nonbinary bytes in the image mask too" do
    manifest, payload = mask_fixture(image_mask: [0_u8, 2_u8])

    expect_raises(ArgumentError, /conditioning mask is not binary/) do
      ML::GGUF::QwenImage21ConditioningBundle.parse(manifest, payload)
    end
  end
end
'''


def _replace_identifiers(text: str, renames: dict[str, str]) -> str:
    for old, new in sorted(renames.items(), key=lambda item: -len(item[0])):
        text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])", new, text)
    return text


def _replace_region(source: str, start: str, end: str, transform) -> str:
    first = source.find(start)
    if first < 0 or source.find(start, first + len(start)) >= 0:
        raise ValueError(f"Expected one source region starting with {start!r}")
    last = source.find(end, first + len(start))
    if last < 0:
        raise ValueError(f"Missing end marker {end!r} for {start!r}")
    return source[:first] + transform(source[first:last]) + source[last:]


def _hole(source: str, line_number: int, expected: str) -> str:
    lines = source.splitlines(keepends=True)
    original = lines[line_number - 1]
    if original.strip() != expected:
        raise ValueError(f"Expected transformed slot {expected!r}, got {original.strip()!r}")
    newline = "\r\n" if original.endswith("\r\n") else "\n" if original.endswith("\n") else ""
    indent = original[: len(original) - len(original.lstrip(" \t"))]
    lines[line_number - 1] = indent + "<HOLE>" + newline
    template = "".join(lines)
    if template.count("<HOLE>") != 1:
        raise ValueError("Transformed source must contain exactly one hole")
    return template


def _line_number(source: str, expected: str) -> int:
    matches = [index for index, line in enumerate(source.splitlines(), start=1) if line.strip() == expected]
    if len(matches) != 1:
        raise ValueError(f"Expected one source line {expected!r}, found {len(matches)}")
    return matches[0]


def _excerpt(template: str, slot: dict) -> str:
    lines = template.splitlines()
    first, last = slot["lines"]
    line = slot["line"]
    if not (1 <= first <= line <= last <= len(lines)):
        raise ValueError(f"Invalid prompt excerpt bounds: {slot}")
    excerpt = "\n".join(lines[first - 1 : last])
    if excerpt.count("<HOLE>") != 1:
        raise ValueError("Prompt excerpt must contain exactly one hole")
    return excerpt


def _build_autograd(mode: str, case: dict, source: str, fixture: str, drafts: dict) -> dict:
    slot = case["slot"]
    line = slot["line"]
    renames: dict[str, str] = {}

    if mode == "rename":
        renames = {
            "input_shape": "operand_shape",
            "grad_val": "upstream_scalar",
            "n": "element_count",
            "result": "gradient_tensor",
        }

        def rename_class(region: str) -> str:
            return _replace_identifiers(region, renames)

        source = _replace_region(source, "class MeanBackward < GradFn", "# Accumulate gradient", rename_class)
        baseline = "element_count = @operand_shape.dims.first.to_f32"
        teacher = "element_count = @operand_shape.numel.to_f32"
        adapted_drafts = {name: _replace_identifiers(text, renames) for name, text in drafts.items()}
        mechanism = "Alpha-renamed the shape and divisor locals while retaining the first-axis divisor."
    elif mode == "rewrite":
        def rewrite_class(region: str) -> str:
            if region.count("n = @input_shape.dims.first.to_f32") != 1:
                raise ValueError("Autograd source does not contain the seeded first-axis divisor")
            if region.count("result.fill!(grad_val / n)") != 1:
                raise ValueError("Autograd fill expression differs from the archived fixture")
            region = region.replace(
                "n = @input_shape.dims.first.to_f32",
                "scale = 1.0_f32 / @input_shape.dims.first.to_f32",
                1,
            )
            return region.replace("result.fill!(grad_val / n)", "result.fill!(grad_val * scale)", 1)

        source = _replace_region(source, "class MeanBackward < GradFn", "# Accumulate gradient", rewrite_class)
        baseline = "scale = 1.0_f32 / @input_shape.dims.first.to_f32"
        teacher = "scale = 1.0_f32 / @input_shape.numel.to_f32"
        adapted_drafts = {
            name: _replace_identifiers(text, {"n": "scale"}) for name, text in drafts.items()
        }
        mechanism = "Re-expressed the divisor as a reciprocal scale multiplied into the upstream gradient."
    else:
        raise ValueError(f"Unsupported mode {mode!r}")

    template = _hole(source, line, baseline)
    transformed_slot = {**slot, "old": baseline}
    return {
        "template": template,
        "excerpt": _excerpt(template, transformed_slot),
        "slot": transformed_slot,
        "baseline": baseline,
        "teacher": teacher,
        "drafts": adapted_drafts,
        "renames": renames,
        "fixture": fixture,
        "guards": _AUTOGRAD_GUARDS,
        "description": case["problem"],
        "mechanism": mechanism,
    }


def _build_qwen_mask(mode: str, case: dict, source: str, fixture: str, drafts: dict) -> dict:
    slot = case["slot"]
    line = slot["line"]
    renames = {"value": "mask_byte"}

    def rename_reader(region: str) -> str:
        if "value = data[index]" not in region:
            raise ValueError("Mask reader does not contain the expected byte binding")
        return _replace_identifiers(region, renames)

    if mode == "rename":
        source = _replace_region(source, "private def self.read_mask", "private def self.expect", rename_reader)
        baseline = "mask_byte == 0"
        teacher = "mask_byte == 1"
        adapted_drafts = {name: _replace_identifiers(text, renames) for name, text in drafts.items()}
        mechanism = "Alpha-renamed the mask byte local while retaining the inverted equality decoder."
    elif mode == "rewrite":
        def rewrite_reader(region: str) -> str:
            if "expect(value <= 1, \"conditioning mask is not binary\")" not in region:
                raise ValueError("Mask reader lost its binary-domain check")
            return (
                "private def self.read_mask(data : Bytes) : Array(Bool)\n"
                "      read_mask_off(data).map { |is_off| !is_off }\n"
                "    end\n\n"
                "    private def self.read_mask_off(data : Bytes) : Array(Bool)\n"
                "      Array(Bool).new(data.size) do |index|\n"
                "        mask_byte = data[index]\n"
                "        expect(mask_byte <= 1, \"conditioning mask is not binary\")\n"
                "        mask_byte == 1\n"
                "      end\n"
                "    end\n\n"
                "    "
            )

        source = _replace_region(source, "private def self.read_mask", "private def self.expect", rewrite_reader)
        line = _line_number(source, "mask_byte == 1")
        first = _line_number(source, "private def self.read_mask(data : Bytes) : Array(Bool)")
        last = _line_number(source, "private def self.expect(condition : Bool, message : String) : Nil")
        slot = {**slot, "line": line, "lines": [first, last]}
        baseline = "mask_byte == 1"
        teacher = "mask_byte == 0"
        adapted_drafts = {name: _replace_identifiers(text, renames) for name, text in drafts.items()}
        mechanism = "Builds an off-entry mask, then inverts it into the public valid-token mask."
    else:
        raise ValueError(f"Unsupported mode {mode!r}")

    template = _hole(source, line, baseline)
    transformed_slot = {**slot, "old": baseline}
    return {
        "template": template,
        "excerpt": _excerpt(template, transformed_slot),
        "slot": transformed_slot,
        "baseline": baseline,
        "teacher": teacher,
        "drafts": adapted_drafts,
        "renames": renames,
        "fixture": fixture,
        "guards": _QWEN_MASK_GUARDS,
        "description": case["problem"],
        "mechanism": mechanism,
    }


def build(case_id: str, mode: str, row: dict, source: str, fixture: str, drafts: dict) -> dict:
    """Build one transformed task while preserving its original behavior checks."""
    case = row.get("case", row)
    if case.get("id", row.get("id")) != case_id:
        raise ValueError(f"Row id {case.get('id', row.get('id'))!r} does not match {case_id!r}")
    if source.count("<HOLE>"):
        raise ValueError("Input source unexpectedly already contains a hole")
    if case_id == _AUTOGRAD:
        return _build_autograd(mode, case, source, fixture, drafts)
    if case_id == _QWEN_MASK:
        return _build_qwen_mask(mode, case, source, fixture, drafts)
    raise ValueError(f"Unsupported Crystal transfer case {case_id!r}")
