#!/usr/bin/env python3
import argparse, json, statistics
from pathlib import Path

def load(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    return {str(r["id"]): r for r in rows}

def summarize(method_path, base_path, cap):
    a, b = load(method_path), load(base_path)
    ids = sorted(set(a) & set(b))
    items = []
    for i in ids:
        x, y = a[i], b[i]
        gd = (x["grounded_edges"] / x["gold_edges"] if x["gold_edges"] else 0) - (y["grounded_edges"] / y["gold_edges"] if y["gold_edges"] else 0)
        items.append({"id": i, "token_delta": x["tokens"]-y["tokens"], "grounded_delta": gd,
                      "method_tokens": x["tokens"], "base_tokens": y["tokens"],
                      "same_output": x["output"] == y["output"]})
    ds = [z["token_delta"] for z in items]
    top = sorted(items, key=lambda z: z["token_delta"], reverse=True)
    bottom = sorted(items, key=lambda z: z["token_delta"])
    return {
        "n": len(items),
        "same_output_rate": sum(z["same_output"] for z in items)/len(items),
        "token_delta": {
            "mean": statistics.mean(ds), "median": statistics.median(ds),
            "method_shorter": sum(d < 0 for d in ds), "equal": sum(d == 0 for d in ds), "method_longer": sum(d > 0 for d in ds),
            "mean_excluding_top_5": statistics.mean(sorted(ds)[:-5]),
            "mean_excluding_top_10": statistics.mean(sorted(ds)[:-10]),
        },
        "cap_transitions": {
            "both": sum(z["method_tokens"] >= cap and z["base_tokens"] >= cap for z in items),
            "method_only": sum(z["method_tokens"] >= cap and z["base_tokens"] < cap for z in items),
            "base_only": sum(z["method_tokens"] < cap and z["base_tokens"] >= cap for z in items),
            "neither": sum(z["method_tokens"] < cap and z["base_tokens"] < cap for z in items),
        },
        "largest_increases": top[:10],
        "largest_savings": bottom[:10],
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method-test', required=True); p.add_argument('--base-test', required=True)
    p.add_argument('--method-dev', required=True); p.add_argument('--base-dev', required=True)
    p.add_argument('--output', required=True); p.add_argument('--cap', type=int, default=256)
    a=p.parse_args()
    out={"dev": summarize(a.method_dev,a.base_dev,a.cap), "test": summarize(a.method_test,a.base_test,a.cap)}
    Path(a.output).write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out,indent=2))
if __name__=='__main__': main()
