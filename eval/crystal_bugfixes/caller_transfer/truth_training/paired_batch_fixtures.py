"""Matched executable pairs for real-versus-shuffled caller transfer."""

from __future__ import annotations


def _indent(source: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line for line in source.splitlines())


def _template(
    helper_module: str,
    helper_signature: str,
    consumer_module: str,
    consumer_signature: str,
    consumer_body: str,
) -> str:
    source = (
        f"module {helper_module}\n"
        f"  def self.{helper_signature}\n"
        "    <HOLE>\n"
        "  end\n"
        "end\n\n"
        f"module {consumer_module}\n"
        f"  def self.{consumer_signature}\n"
        f"{_indent(consumer_body, 4)}\n"
        "  end\n"
        "end\n"
    )
    if source.count("<HOLE>") != 1:
        raise AssertionError("template must contain exactly one helper hole")
    return source


def _spec(description: str, body: str) -> str:
    return (
        f'  it "{description}" do\n'
        f"{_indent(body, 4)}\n"
        "  end"
    )


def _fixture(
    suite: str,
    target_specs: tuple[tuple[str, str], ...],
    regression_specs: tuple[tuple[str, str], ...],
) -> str:
    specs = target_specs + regression_specs
    rendered = "\n\n".join(_spec(description, body) for description, body in specs)
    return (
        'require "spec"\n'
        'require "./subject"\n\n'
        f'describe "{suite}" do\n'
        f"{rendered}\n"
        "end\n"
    )


def _pair(
    *,
    family: str,
    kind: str,
    issue: str,
    helper_module: str,
    helper_signature: str,
    consumer_module: str,
    consumer_signature: str,
    consumers: tuple[str, str],
    fixture: str,
    target_tests: tuple[str, ...],
    teachers: tuple[str, str],
    broken: str,
    regression_traps: tuple[str, str],
) -> list[dict[str, object]]:
    rows = []
    for variant, consumer, teacher, trap in zip(
        ("a", "b"), consumers, teachers, regression_traps, strict=True
    ):
        row = {
            "family": family,
            "variant": variant,
            "kind": kind,
            "issue": issue,
            "template": _template(
                helper_module,
                helper_signature,
                consumer_module,
                consumer_signature,
                consumer,
            ),
            "fixture": fixture,
            "target_tests": list(target_tests),
            "teacher": teacher,
            "broken": broken,
            "regression_trap": trap,
        }
        rows.append(row)
    return rows


def rows() -> list[dict[str, object]]:
    """Return four fresh caller pairs with executable Crystal specifications."""
    cursor_targets = (
        (
            "retries when the candidate cursor trails the checkpoint",
            "ScanRecovery.should_retry?(4, 9).should be_true",
        ),
        (
            "does not retry when the candidate cursor advances past the checkpoint",
            "ScanRecovery.should_retry?(11, 9).should be_false",
        ),
    )
    cursor_regressions = (
        (
            "leaves an unchanged checkpoint alone",
            "ScanRecovery.should_retry?(9, 9).should be_false",
        ),
        (
            "rejects invalid negative cursor positions",
            "ScanRecovery.should_retry?(-1, 9).should be_false\n"
            "ScanRecovery.should_retry?(4, -1).should be_false",
        ),
    )
    cursor_fixture = _fixture("ScanRecovery", cursor_targets, cursor_regressions)

    slot_targets = (
        (
            "accepts an ordinary slot below capacity",
            "SlotReader.readable?(2, 5).should be_true",
        ),
        (
            "rejects a slot beyond capacity",
            "SlotReader.readable?(7, 5).should be_false",
        ),
    )
    slot_regressions = (
        (
            "excludes an index equal to the slot count",
            "SlotReader.readable?(5, 5).should be_false",
        ),
        (
            "keeps the last allocated slot readable",
            "SlotReader.readable?(4, 5).should be_true",
        ),
        (
            "does not read from an empty slot table",
            "SlotReader.readable?(0, 0).should be_false",
        ),
        (
            "rejects negative positions and capacities",
            "SlotReader.readable?(-1, 5).should be_false\n"
            "SlotReader.readable?(0, -1).should be_false",
        ),
    )
    slot_fixture = _fixture("SlotReader", slot_targets, slot_regressions)

    batch_targets = (
        (
            "flushes a multi-record batch",
            'BatchFlusher.flush?(["one", "two", "three"]).should be_true',
        ),
        (
            "does not flush an empty batch",
            "BatchFlusher.flush?([] of String).should be_false",
        ),
    )
    batch_regressions = (
        (
            "flushes a single-record batch",
            'BatchFlusher.flush?(["only"]).should be_true',
        ),
        (
            "handles consecutive batches without carrying state",
            "BatchFlusher.flush?([] of String).should be_false\n"
            'BatchFlusher.flush?(["first"]).should be_true\n'
            "BatchFlusher.flush?([] of String).should be_false\n"
            'BatchFlusher.flush?(["second", "third"]).should be_true',
        ),
    )
    batch_fixture = _fixture("BatchFlusher", batch_targets, batch_regressions)

    backup_targets = (
        (
            "finds a backup suffix in a mixed list",
            'BackupScanner.contains_backup?([".tmp", ".BAK"]).should be_true',
        ),
        (
            "ignores a list with no backup suffix",
            'BackupScanner.contains_backup?([".txt", ".log"]).should be_false',
        ),
    )
    backup_regressions = (
        (
            "matches lowercase backup suffixes",
            'BackupScanner.contains_backup?([".bak"]).should be_true',
        ),
        (
            "does not match longer suffix text",
            'BackupScanner.contains_backup?([".bak.old"]).should be_false',
        ),
        (
            "returns false for an empty suffix list",
            "BackupScanner.contains_backup?([] of String).should be_false",
        ),
    )
    backup_fixture = _fixture("BackupScanner", backup_targets, backup_regressions)

    result = []
    result += _pair(
        family="cursor_checkpoint_order",
        kind="flip",
        issue=(
            "Recovery must retry only when the candidate cursor is strictly "
            "behind the saved checkpoint. Equal or later positions must not "
            "retry, and negative positions are invalid."
        ),
        helper_module="CursorCompare",
        helper_signature="precedes?(left : Int32, right : Int32) : Bool",
        consumer_module="ScanRecovery",
        consumer_signature="should_retry?(candidate : Int32, checkpoint : Int32) : Bool",
        consumers=(
            "return false if candidate < 0 || checkpoint < 0\n"
            "CursorCompare.precedes?(candidate, checkpoint)",
            "return false if candidate < 0 || checkpoint < 0\n"
            "CursorCompare.precedes?(checkpoint, candidate)",
        ),
        fixture=cursor_fixture,
        target_tests=tuple(description for description, _ in cursor_targets),
        teachers=("left < right", "left > right"),
        broken="left == right",
        regression_traps=("left <= right", "left >= right"),
    )
    result += _pair(
        family="zero_based_slot_limit",
        kind="flip",
        issue=(
            "The page reader must accept exactly the zero-based slot indices "
            "from 0 through slot_count minus one. An empty table has no "
            "readable slots."
        ),
        helper_module="SlotBounds",
        helper_signature="before_limit?(index : Int32, limit : Int32) : Bool",
        consumer_module="SlotReader",
        consumer_signature="readable?(index : Int32, slot_count : Int32) : Bool",
        consumers=(
            "return false if index < 0 || slot_count < 0\n"
            "SlotBounds.before_limit?(index, slot_count)",
            "return false if index < 0 || slot_count < 0\n"
            "last_slot = slot_count - 1\n"
            "SlotBounds.before_limit?(index, last_slot)",
        ),
        fixture=slot_fixture,
        target_tests=tuple(description for description, _ in slot_targets),
        teachers=("index < limit", "index <= limit"),
        broken="index == limit",
        regression_traps=("index <= limit", "index < limit"),
    )
    result += _pair(
        family="outbox_pending_count",
        kind="invariant",
        issue=(
            "The outbox dispatcher must flush every nonempty batch, including "
            "a single record. It must skip empty batches."
        ),
        helper_module="BatchCount",
        helper_signature="has_pending?(pending_count : Int32) : Bool",
        consumer_module="BatchFlusher",
        consumer_signature="flush?(records : Array(String)) : Bool",
        consumers=(
            "BatchCount.has_pending?(records.size)",
            "pending_count = records.size\n"
            "BatchCount.has_pending?(pending_count)",
        ),
        fixture=batch_fixture,
        target_tests=tuple(description for description, _ in batch_targets),
        teachers=("pending_count > 0", "pending_count > 0"),
        broken="pending_count < 0",
        regression_traps=("pending_count > 1", "pending_count > 1"),
    )
    result += _pair(
        family="backup_suffix_scan",
        kind="invariant",
        issue=(
            "The cleanup scan should report whether the list contains the "
            "exact suffix .bak, ignoring case. A longer suffix such as "
            ".bak.old does not qualify."
        ),
        helper_module="SuffixMatch",
        helper_signature="backup?(suffix : String) : Bool",
        consumer_module="BackupScanner",
        consumer_signature="contains_backup?(suffixes : Array(String)) : Bool",
        consumers=(
            "suffixes.each do |suffix|\n"
            "  return true if SuffixMatch.backup?(suffix)\n"
            "end\n"
            "false",
            "suffixes.any? { |suffix| SuffixMatch.backup?(suffix) }",
        ),
        fixture=backup_fixture,
        target_tests=tuple(description for description, _ in backup_targets),
        teachers=('suffix.downcase == ".bak"', 'suffix.downcase == ".bak"'),
        broken='suffix == ".bak"',
        regression_traps=(
            'suffix.downcase.includes?("bak")',
            'suffix.downcase.includes?("bak")',
        ),
    )
    return result
