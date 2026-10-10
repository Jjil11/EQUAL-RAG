"""Evidence-aware blind packet. Baseline-only sampling; no fabricated ratings."""
import hashlib
import json
import random
from pathlib import Path
from build_blinded_taxonomy import ROOT,RUNS,read


def main():
    data={x['id']:x for x in read(ROOT/'data/qampari_evidence_full/test.jsonl')}
    # Mistral legacy has a real EOT target. Qwen is excluded until rerun fixed.
    pools={'baseline_cap_hit':[],'baseline_noncap':[]}
    rng=random.Random(20261010)
    for family,seed,bp,mp in RUNS:
        if family!='mistral':continue
        base={x['id']:x for x in read(ROOT/bp)};method={x['id']:x for x in read(ROOT/mp)}
        for qid in sorted(base):
            stratum='baseline_cap_hit' if base[qid]['tokens']>=256 else 'baseline_noncap'
            pools[stratum].append((family,seed,qid,base[qid],method[qid]))
    chosen=[];used=set()
    for stratum,pool in pools.items():
        rng.shuffle(pool)
        for item in pool:
            if item[2] in used:continue
            used.add(item[2]);chosen.append((stratum,item))
            if sum(s==stratum for s,_ in chosen)>=30:break
    rng.shuffle(chosen)
    packet=[];key=[]
    for stratum,(family,seed,qid,base,method) in chosen:
        swap=rng.getrandbits(1);a,b=(method,base) if swap else (base,method)
        blind=hashlib.sha256(f'evidence-blind-20261010:{family}:{seed}:{qid}'.encode()).hexdigest()[:12]
        packet.append({'blind_id':blind,'question':data[qid]['question'],'retrieved_docs':data[qid]['docs'],
                       'output_A':a['output'],'output_B':b['output'],
                       'annotation':{'grounded_completeness_preference':'','citation_correctness_preference':'',
                                     'conciseness_preference':'','unsupported_claim_A':'','unsupported_claim_B':'',
                                     'answer_loss_A':'','answer_loss_B':'','repetition_A':'','repetition_B':'','notes':''}})
        key.append({'blind_id':blind,'family':family,'seed':seed,'id':qid,'stratum':stratum,
                    'A':'method' if swap else 'baseline','B':'baseline' if swap else 'method'})
    out=ROOT/'runs/submission_20261010/blind_evidence'
    if out.exists():raise FileExistsError('Never overwrite blind packet')
    out.mkdir()
    for name,rows in [('packet.jsonl',packet),('identity_key_PRIVATE.jsonl',key)]:
        (out/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    manifest={'n':len(packet),'unique_questions':len(used),'seed':20261010,
              'sampling':'30 baseline cap-hit + 30 baseline non-cap; no method outcome used',
              'identity_key_must_not_be_given_to_annotators':True,
              'independent_human_annotators_required':2,'human_annotations_completed':0,
              'qwen_excluded_due_to_boundary_bug':True}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
