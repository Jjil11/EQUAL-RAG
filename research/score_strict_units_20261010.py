"""Strict unit-local citation proxy and official-style QAMPARI list F1.

Reports lexical citation support, not semantic entailment. Preserve the legacy
window metric separately; exact unit matching prevents neighboring-citation credit.
"""
import argparse
import json
import re
import string
from pathlib import Path
import numpy as np


def norm(s):
    s=re.sub(r'\[\d+\]','',s)
    s=''.join(c for c in s.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())


def score(row, pred):
    units=[x.strip() for x in pred['output'].rstrip().rstrip('.').rstrip(',').split(',') if norm(x)]
    values=[norm(x) for x in units]
    groups=row.get('all_gold_answers',[e['aliases'] for e in row['edges']])
    groups=[set(norm(a) for a in aliases if norm(a)) for aliases in groups]
    flat=set().union(*groups) if groups else set()
    correct=sum(v in flat for v in values)
    matched=sum(any(v in g for v in values) for g in groups)
    precision=correct/max(1,len(values))
    recall=matched/max(1,len(groups))
    recall5=min(5,matched)/max(1,min(5,len(groups)))
    grounded=0
    for edge in row['edges']:
        aliases={norm(a) for a in edge['aliases'] if norm(a)}
        supports=set(edge.get('supporting_doc_ids',[edge['doc_id']]))
        if any(v in aliases and supports.intersection(int(r) for r in re.findall(r'\[(\d+)\]',u))
               for v,u in zip(values,units)):
            grounded+=1
    return {'id':row['id'],'tokens':pred['tokens'],'output':pred['output'],
            'gold_edges':len(row['edges']),'grounded_edges':grounded,
            'legacy_grounded_edges':pred['grounded_edges'],
            'answer_precision':precision,'answer_recall':recall,
            'answer_f1':2*precision*recall/max(1e-12,precision+recall),
            'answer_f1_top5':2*precision*recall5/max(1e-12,precision+recall5),
            'repeated_units':len(values)-len(set(values)),
            'cap_hit_proxy':pred['tokens']>=256}


def summarize(rows):
    edges=sum(x['gold_edges'] for x in rows)
    q=sum(x['grounded_edges'] for x in rows)/max(1,edges)
    tokens=float(np.mean([x['tokens'] for x in rows]))
    return {'n':len(rows),'gold_edges':edges,'grounded_quality':q,'mean_tokens':tokens,
            'utility':q/max(1,tokens),
            **{k:float(np.mean([x[k] for x in rows])) for k in ['answer_precision','answer_recall','answer_f1','answer_f1_top5']},
            'repetition_rate':float(np.mean([x['repeated_units']>0 for x in rows])),
            'cap_hit_rate_proxy':float(np.mean([x['cap_hit_proxy'] for x in rows])),
            'metric_warning':'Citation proxy uses exact comma-list units and lexical support; no entailment claim.'}


def paired(method,base,samples=20000):
    b={x['id']:x for x in base};m={x['id']:x for x in method}
    if b.keys()!=m.keys() or len(b)!=len(base) or len(m)!=len(method):
        raise ValueError('Pairing mismatch or duplicate IDs')
    keys=sorted(b)
    fields=['grounded_edges','gold_edges','tokens','answer_f1','answer_precision','answer_recall','repeated_units','cap_hit_proxy']
    a=np.array([[m[k][f] for f in fields] for k in keys],dtype=float)
    z=np.array([[b[k][f] for f in fields] for k in keys],dtype=float)
    def calc(v):
        q=v[:,0].sum()/max(1,v[:,1].sum()); t=v[:,2].mean()
        return np.array([q,t,q/max(1,t),*v[:,3:6].mean(0),np.mean(v[:,6]>0),v[:,7].mean()])
    names=['grounded_quality','mean_tokens','utility','answer_f1','answer_precision','answer_recall','repetition_rate','cap_hit_rate_proxy']
    observed=calc(a)-calc(z); rng=np.random.default_rng(20261010)
    boot=[]
    for _ in range(samples):
        ix=rng.integers(0,len(keys),len(keys));boot.append(calc(a[ix])-calc(z[ix]))
    boot=np.array(boot)
    result={name:{'delta':float(observed[i]),'ci95':np.quantile(boot[:,i],[.025,.975]).tolist()} for i,name in enumerate(names)}
    lower=float(np.quantile(boot[:,0],.05))
    result['quality_noninferiority']={'margin':-.01,'one_sided_95_lower':lower,'supported':lower>-.01,
                                     'warning':'Margin fixed for this new analysis; not a retrospective preregistration of old experiments.'}
    result['method']=summarize(method);result['baseline']=summarize(base)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--predictions',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--baseline');args=ap.parse_args()
    data={x['id']:x for x in [json.loads(l) for l in Path(args.data).read_text().splitlines()]}
    preds=[json.loads(l) for l in Path(args.predictions).read_text().splitlines()]
    if len({x['id'] for x in preds})!=len(preds):raise ValueError('Duplicate prediction IDs')
    rows=[score(data[x['id']],x) for x in preds]
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.with_suffix('.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    result=summarize(rows)
    if args.baseline:
        base=[score(data[x['id']],x) for x in [json.loads(l) for l in Path(args.baseline).read_text().splitlines()]]
        result=paired(rows,base)
    out.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
