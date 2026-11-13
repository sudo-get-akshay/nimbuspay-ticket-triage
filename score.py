"""Score triage predictions against labelled data.

    python score.py --gold data/dev.jsonl --pred my_dev_predictions.jsonl

Prediction file: one JSON object per line
    {"id": "dv-00001", "output": "<the raw text your model generated>"}

Scoring is strict:
  * "output" must be ONLY a JSON object (leading/trailing whitespace is ignored).
  * Every field must match exactly, including type (true is not 1, 129900 is not "129900").
  * A missing id or invalid JSON counts as wrong on every field.
"""
import argparse, json
from collections import defaultdict

KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def same(a, b):
    return type(a) is type(b) and a == b


def parse(text):
    try:
        obj = json.loads(text.strip())
    except (json.JSONDecodeError, AttributeError):
        return None
    return obj if isinstance(obj, dict) else None


def score(gold_rows, pred_rows):
    preds = {p["id"]: p.get("output") for p in pred_rows}
    groups = defaultdict(lambda: {"n": 0, "valid": 0, "exact": 0, "field": defaultdict(int)})
    for g in gold_rows:
        gold = json.loads(g["messages"][-1]["content"])
        pred = parse(preds.get(g["id"]))
        names = ["all"] + ([f"slice={g['slice']}"] if "slice" in g else []) + [f"category={gold['category']}"]
        hits = {k: pred is not None and k in pred and same(pred[k], gold[k]) for k in KEYS}
        exact = pred is not None and all(hits.values()) and set(pred) == set(KEYS)
        for name in names:
            s = groups[name]
            s["n"] += 1
            s["valid"] += pred is not None
            s["exact"] += exact
            for k in KEYS:
                s["field"][k] += hits[k]
    out = {}
    for name, s in groups.items():
        n = s["n"]
        fields = {k: s["field"][k] / n for k in KEYS}
        out[name] = {"n": n, "json_valid": s["valid"] / n, "exact_match": s["exact"] / n,
                     "mean_field_acc": sum(fields.values()) / len(KEYS), "fields": fields}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    a = ap.parse_args()
    res = score(load(a.gold), load(a.pred))
    if a.json:
        print(json.dumps(res, indent=1))
        return
    print(f"{'group':28} {'n':>5} {'valid':>7} {'exact':>7} {'fields':>7}")
    for name in sorted(res, key=lambda x: (x != "all", x)):
        r = res[name]
        print(f"{name:28} {r['n']:5d} {r['json_valid']:7.1%} {r['exact_match']:7.1%} {r['mean_field_acc']:7.1%}")
    print("\nper-field accuracy (all rows)")
    for k, v in res["all"]["fields"].items():
        print(f"  {k:14} {v:6.1%}")


if __name__ == "__main__":
    main()
