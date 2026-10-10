#!/usr/bin/env python3
"""Create a reproducible blinded qualitative packet from six frozen test runs."""
from __future__ import annotations
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models

import argparse, hashlib, json, random, re
from pathlib import Path

ROOT = Path(str(_release_root))
RUNS = [
 ("mistral",20260930,"runs/qampari_cap256_seed20260930/aequal_kl/eval_results.jsonl","runs/qampari_trajectory_onpolicy_repair_seed20260930/interp05/regression_test/eval_results.jsonl"),
 ("mistral",20261001,"runs/qampari_onpolicy_multiseed/seed20261001/baseline_test/eval_results.jsonl","runs/qampari_onpolicy_multiseed/seed20261001/interp05/test/eval_results.jsonl"),
 ("mistral",20261002,"runs/qampari_onpolicy_multiseed/seed20261002/baseline_test/eval_results.jsonl","runs/qampari_onpolicy_multiseed/seed20261002/interp05/test/eval_results.jsonl"),
 ("qwen3",20260930,"runs/qwen3_14b_onpolicy_repair_seed20260930/baseline_test/eval_results.jsonl","runs/qwen3_14b_onpolicy_repair_seed20260930/interp05/test/eval_results.jsonl"),
 ("qwen3",20261001,"runs/qwen3_14b_onpolicy_multiseed/seed20261001/baseline_test/eval_results.jsonl","runs/qwen3_14b_onpolicy_multiseed/seed20261001/interp05/test/eval_results.jsonl"),
 ("qwen3",20260929,"runs/qwen3_14b_onpolicy_multiseed/seed20260929/baseline_test/eval_results.jsonl","runs/qwen3_14b_onpolicy_multiseed/seed20260929/interp05/test/eval_results.jsonl"),
]

def read(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def norm(x):
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", re.sub(r"\[\d+\]", "", x.lower())).split())

def duplicate_stats(text):
    units=[norm(x) for x in text.split(",") if norm(x)]
    return {"units":len(units),"duplicate_units":len(units)-len(set(units))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out-dir',required=True); ap.add_argument('--per-stratum',type=int,default=12); a=ap.parse_args()
    data={x['id']:x for x in read(ROOT/'data/qampari_evidence_full/test.jsonl')}
    pools={"repair_success":[],"unchanged":[],"regression":[]}
    for family,seed,bp,mp in RUNS:
        base={x['id']:x for x in read(ROOT/bp)}; method={x['id']:x for x in read(ROOT/mp)}
        for qid in sorted(base.keys() & method.keys()):
            b,m=base[qid],method[qid]; edges=max(1,b['gold_edges'])
            dq=(m['grounded_edges']-b['grounded_edges'])/edges; dt=m['tokens']-b['tokens']
            bd=duplicate_stats(b['output']); md=duplicate_stats(m['output'])
            item={"family":family,"seed":seed,"id":qid,"baseline":b,"method":m,"dq":dq,"dt":dt,"base_dup":bd,"method_dup":md}
            if b['tokens']>=256 and m['tokens']<=b['tokens']-10 and dq>=-0.01: pools['repair_success'].append(item)
            elif abs(dt)<=2 and abs(dq)<=0.01: pools['unchanged'].append(item)
            elif dq < -0.01 or dt>=10: pools['regression'].append(item)
    rng=random.Random(20261006); out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    packet=[]; key=[]
    for stratum,items in pools.items():
        rng.shuffle(items)
        for item in items[:a.per_stratum]:
            swap=bool(rng.getrandbits(1)); first=item['method'] if swap else item['baseline']; second=item['baseline'] if swap else item['method']
            blind_id=hashlib.sha256(f"{item['family']}:{item['seed']}:{item['id']}".encode()).hexdigest()[:12]
            packet.append({
              "blind_id":blind_id,"sampling_stratum":stratum,"question":data[item['id']]['question'],
              "output_A":first['output'],"output_B":second['output'],
              "annotation": {"preferred_for_grounded_completeness":"", "preferred_for_conciseness":"", "repetition_removed":"", "answer_loss":"", "citation_loss":"", "semantic_duplication":"", "notes":""}
            })
            key.append({"blind_id":blind_id,"family":item['family'],"seed":item['seed'],"id":item['id'],
                        "A":("method" if swap else "baseline"),"B":("baseline" if swap else "method"),
                        "quality_delta":item['dq'],"token_delta":item['dt'],
                        "baseline_duplicate_units":item['base_dup']['duplicate_units'],"method_duplicate_units":item['method_dup']['duplicate_units']})
    for name,rows in (("blind_packet.jsonl",packet),("identity_key.jsonl",key)):
        with (out/name).open('w',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    summary={"seed":20261006,"requested_per_stratum":a.per_stratum,"available":{k:len(v) for k,v in pools.items()},"selected":{k:sum(x['sampling_stratum']==k for x in packet) for k in pools},"n":len(packet)}
    (out/'manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
