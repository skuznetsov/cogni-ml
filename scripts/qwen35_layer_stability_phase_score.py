#!/usr/bin/env python3
import csv
import sys

if len(sys.argv) != 2:
    print("usage: qwen35_layer_stability_phase_score.py ATLAS.tsv", file=sys.stderr)
    sys.exit(2)

path = sys.argv[1]
rows = []
with open(path, newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if row.get("prompt_id") in (None, "", "=== STDERR ==="):
            continue
        try:
            row["stable_from_layer"] = int(row["stable_from_layer"])
            for k in ("prompt_tokens", "unique_rate", "repeat_rate", "bigram_repeat_rate", "adjacent_repeat_rate"):
                row[k] = float(row[k])
        except (KeyError, ValueError):
            continue
        rows.append(row)

if not rows:
    print("no rows", file=sys.stderr)
    sys.exit(1)

early_cutoff = 27
features = ["unique_rate", "repeat_rate", "bigram_repeat_rate", "adjacent_repeat_rate", "prompt_tokens"]
print("kind\tfeature\top\tthreshold\tselected\tearly_selected\tprecision\trecall\tavg_stable_selected\tavg_stable_all")
early_total = sum(1 for r in rows if r["stable_from_layer"] <= early_cutoff)
avg_all = sum(r["stable_from_layer"] for r in rows) / len(rows)

for feature in features:
    values = sorted({r[feature] for r in rows})
    for threshold in values:
        for op in ("<=", ">="):
            if op == "<=":
                selected = [r for r in rows if r[feature] <= threshold]
            else:
                selected = [r for r in rows if r[feature] >= threshold]
            if not selected:
                continue
            early_selected = sum(1 for r in selected if r["stable_from_layer"] <= early_cutoff)
            precision = early_selected / len(selected)
            recall = early_selected / early_total if early_total else 0.0
            avg_sel = sum(r["stable_from_layer"] for r in selected) / len(selected)
            print("gate\t{}\t{}\t{:.6f}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.2f}\t{:.2f}".format(
                feature, op, threshold, len(selected), early_selected, precision, recall, avg_sel, avg_all
            ))
