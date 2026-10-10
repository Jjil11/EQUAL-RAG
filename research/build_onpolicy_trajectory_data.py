#!/usr/bin/env python3
import argparse, json
from pathlib import Path

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train', required=True)
    p.add_argument('--mining-dir', required=True)
    p.add_argument('--output', required=True)
    a=p.parse_args()
    trajectories={}
    for path in sorted(Path(a.mining_dir).glob('shard_*.jsonl')):
        for row in read_jsonl(path): trajectories[str(row['id'])]=row
    rows=read_jsonl(a.train)
    if len(trajectories)!=len(rows):
        raise ValueError(f'trajectories={len(trajectories)} train={len(rows)}')
    out=[]
    for row in rows:
        mined=trajectories[str(row['id'])]
        x=dict(row)
        x['gold_short']=x['short']
        x['short']=mined['output']
        x['anchor_output_tokens']=mined['tokens']
        x['anchor_cap_hit']=mined['cap_hit']
        out.append(x)
    out_path=Path(a.output); out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w',encoding='utf-8') as f:
        for row in out: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(json.dumps({'rows':len(out),'trajectory_rows':sum(bool(x.get('has_trajectory')) for x in out),
                      'cap_hits':sum(bool(x['anchor_cap_hit']) for x in out)},indent=2))
if __name__=='__main__': main()
