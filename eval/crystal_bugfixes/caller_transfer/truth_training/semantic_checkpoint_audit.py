"""Match a native QLoRA checkpoint against a revalidated archived run."""

from contextlib import redirect_stdout
import math
import os
from pathlib import Path

import fit


def _factor_updates(initial, adapter):
    before, after = fit._load_factors(initial), fit._load_factors(adapter)
    offset, updated = 0, []
    for layer in fit.LAYERS:
        for name, width, height in fit.PROJECTIONS:
            a_end = offset + fit.RANK * width
            b_end = a_end + fit.RANK * height
            changed_a = sum(a != b for a, b in zip(before[offset:a_end], after[offset:a_end]))
            changed_b = sum(a != b for a, b in zip(before[a_end:b_end], after[a_end:b_end]))
            if not changed_a or not changed_b:
                raise ValueError(f"LoRA factors did not both update: layer {layer} {name}")
            updated.append(dict(layer=layer, projection=name, rank=fit.RANK,
                                a_elements=a_end - offset, a_changed_elements=changed_a,
                                b_elements=b_end - a_end, b_changed_elements=changed_b))
            offset = b_end
    if offset != len(before) or offset != len(after):
        raise ValueError("checkpoint adapter factor layout has trailing or missing values")
    return updated


def _timing_free(event):
    return {key: value for key, value in event.items() if key != "seconds"}


def audit_prefix(attempt: Path, reference: dict, steps: int) -> dict:
    """Compare the full prefill and update event prefix after the parent fit audit."""
    if reference["status"] != "complete" or reference["config"] != fit.CONFIG:
        raise ValueError("expected a revalidated full-SFT training audit")
    if steps not in (10, 25):
        raise ValueError("expected a 10- or 25-step replay")

    inputs = reference["inputs"]
    archive = Path(inputs["train"]["path"]).parent / "training/train.log"
    if fit._digest(archive) != reference["train_log_sha256"]:
        raise ValueError("archived training log changed")
    archived_events = fit._events(archive)
    start_index, = [i for i, event in enumerate(archived_events) if event["event"] == "start"]
    expected = archived_events[:start_index + steps + 1]
    expected[start_index] = dict(expected[start_index], updates=steps)

    log, adapter = attempt / "train.log", attempt / "adapter.bin"
    events = fit._events(log)
    if len(events) != len(expected) + 1 or events[-1]["event"] != "saved":
        raise ValueError("checkpoint log is missing its complete prefix or save")
    for event, archived in zip(events[:-1], expected, strict=True):
        if _timing_free(event) != _timing_free(archived):
            raise ValueError(f"checkpoint {event['event']} row {event.get('row')} differs from the archive")
    saved = events[-1]
    if saved["adapter"] != str(adapter.resolve()):
        raise ValueError("trainer saved to an unexpected adapter path")
    if not math.isfinite(saved["validation_loss"]) or not 0 <= saved["seconds"] < math.inf:
        raise ValueError("trainer reported invalid validation loss or elapsed time")

    start = events[start_index]
    updates = events[start_index + 1:-1]
    visits = [event["row"] for event in updates]
    losses = [event["loss"] for event in updates]
    if visits != reference["visits"][:steps] or losses != reference["losses"]["updates"][:steps]:
        raise ValueError("checkpoint optimizer visits or loss history disagree with the fit audit")
    targets = sum(reference["target_tokens_by_train_row"][str(row)] for row in visits)
    if targets != steps * 4:
        raise ValueError("checkpoint used an unexpected completion-token budget")

    checker_path, checker = fit._checker(Path(fit.__file__).resolve().parents[4])
    record = inputs["adapter_checker"]
    if str(checker_path.resolve()) != record["path"] or fit._digest(checker_path) != record["sha256"]:
        raise ValueError("archived adapter checker changed")
    with open(os.devnull, "w") as sink, redirect_stdout(sink):
        adapter_sha = checker.check_adapter(adapter)
    factors = _factor_updates(Path(inputs["initial"]["path"]), adapter)
    return dict(
        status="complete", steps=steps, adapter_sha256=adapter_sha, train_log_sha256=fit._digest(log),
        visits=visits, losses=dict(initial_validation=start["validation_loss"], updates=losses,
                                   final_validation=saved["validation_loss"]),
        supervised_target_tokens=targets, seconds=saved["seconds"],
        updated_projections=len(factors), updated_factors=2 * len(factors), factor_updates=factors,
        guards=dict(archive_prefills_match=True, archive_recoveries_match=True, archive_start_match=True,
                    archive_update_prefix_match=True, adapter_checked=True, warm_start_factors_updated=True),
    )
