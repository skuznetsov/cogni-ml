"""Reverse the adjacent label pairs of the frozen alternating full-SFT experiment."""

import argparse
from collections import Counter
import json
from pathlib import Path

import alternate_checkpoints as forward

prior = forward.prior
fit, semantic = forward.fit, forward.semantic
digest, read, keep = forward.digest, forward.read, forward.keep
OUT = semantic.OUT / "reversed-order-v1"
DRIVER = forward.OUT / "native-checkpoints"
ARCHIVE_PINS = {
    "protocol.json": "13f2ec22ecc0a101064ad8e9ba6ebf6de520b15ec807f9699f6a7710695362b9",
    "results.json": "b5dc11bf8440ed9c14369c8894a6a49781e83b6ba14d8feb172af27bc6e3002b",
}


def reverse_visits(visits, labels):
    assert len(visits) == fit.UPDATES
    pairs = zip(visits[::2], visits[1::2], strict=True)
    reverse = [row for true_row, false_row in pairs for row in (false_row, true_row)]
    assert [labels[row - 1] for row in reverse] == ["false", "true"] * (fit.UPDATES // 2)
    return reverse


def side(entry, last):
    metrics = entry["score"]["summary"]["all"]
    true_leads = metrics["true_leaders"]
    false_leads = metrics["false_leaders"]
    odds = metrics["mean_log_odds_true"]
    majority = "true" if true_leads > false_leads else "false" if false_leads > true_leads else "tie"
    margin_leader = "true" if odds > 0 else "false" if odds < 0 else "tie"
    return dict(last_target=last, true_leads=true_leads, false_leads=false_leads, mean_log_odds_true=odds,
                majority=majority, margin_leader=margin_leader, majority_follows_last=majority == last,
                margin_follows_last=margin_leader == last, complete_pairs=metrics["both_correct"],
                first_token_pairs=metrics["both_first_correct"])


def main(out):
    assert out != forward.OUT
    for file, expected in ARCHIVE_PINS.items():
        assert digest(forward.OUT / file) == expected, file
    archive = read(forward.OUT / "results.json")
    protocol = read(forward.OUT / "protocol.json")
    assert archive["protocol_sha256"] == ARCHIVE_PINS["protocol.json"]
    for file, expected in protocol["files"].items():
        assert digest(Path(file)) == expected, file
    cases, requests, _, audits = prior.freeze(prior.OUT)
    labels = forward.labels_for(audits["original"])
    assert labels == forward.labels_for(audits["diverse"])
    decks, visits = forward.make_schedule(DRIVER, labels, audits)
    assert decks == protocol["seed_decks"] and visits == protocol["visits"]
    reverse = reverse_visits(visits, labels)
    assert sorted(reverse[:32]) == list(range(1, 33))
    budget = {}
    for step in forward.CHECKPOINTS:
        first, second = visits[:step], reverse[:step]
        replacements = sum((Counter(second) - Counter(first)).values())
        assert replacements == step % 2
        budget[str(step)] = dict(replaced_visits=replacements, last_forward=labels[first[-1] - 1],
                                 last_reverse=labels[second[-1] - 1],
                                 reverse_counts=dict(Counter(labels[row - 1] for row in second)))

    control_root = forward.OUT / "unchanged-order-control"
    control_folder, = control_root.glob("attempt-*/step-10")
    assert digest(control_folder / "adapter.bin") == archive["control"]["adapter_sha256"]
    assert digest(control_folder.parent / "train.log") == archive["control"]["train_log_sha256"]
    control = forward.qualify(forward.OUT, DRIVER, audits["original"], labels,
                              archive["shuffled"]["original"]["10"], ARCHIVE_PINS["protocol.json"])
    assert control == archive["control"]

    keep(out / "protocol.json", dict(
        archive_pins=ARCHIVE_PINS, training_config=protocol["training_config"],
        sampling=protocol["sampling"], context=protocol["context"], max_new=protocol["max_new"],
        requests=protocol["requests"], archive_audits=protocol["archive_audits"],
        steps=list(forward.CHECKPOINTS), forward_visits=visits, reverse_visits=reverse,
        targets_by_step=[labels[row - 1] for row in reverse], checkpoints=budget,
        files={**protocol["files"], str(Path(__file__).resolve()): digest(Path(__file__))},
        recency_metrics="last visited first literal versus first-token majority and mean log-odds sign",
    ))
    protocol_sha = digest(out / "protocol.json")
    scores, comparisons = {}, {}
    for arm, reference in audits.items():
        assert digest(DRIVER) == protocol["files"][str(DRIVER)]
        _, checkpoints = forward.train(out / arm, DRIVER, reference, reverse, labels,
                                       forward.CHECKPOINTS, protocol_sha)
        scores[arm], comparisons[arm] = {}, {}
        for step, (folder, audit) in checkpoints.items():
            entry = prior.evaluate(folder, cases, requests, audit)
            key = str(step)
            scores[arm][key] = entry
            comparison = dict(
                forward=side(archive["alternating"][arm][key], budget[key]["last_forward"]),
                reverse=side(entry, budget[key]["last_reverse"]),
            )
            comparisons[arm][key] = comparison
            print(arm, step, json.dumps(comparison), flush=True)
    keep(out / "results.json", dict(protocol_sha256=protocol_sha, tool_four=archive["tool_four"],
                                   reversed=scores, comparison=comparisons))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=OUT)
    main(cli.parse_args().output.resolve())
