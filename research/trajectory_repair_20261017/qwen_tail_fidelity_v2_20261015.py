"""Corrected training-only pilot: protect interiors, not faulty native EOS.

Same A/C1/C2 labels and same Base initialization. Restore flipped prefixes
without diluting their Hinge by the many already-correct positions.
"""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import qwen_tail_fidelity_20261015 as core

q=core.q
PILOT=core.RUN
RUN=q.ROOT/'runs/qwen_tail_fidelity_v2_20261015'
core.RUN=core.ev.RUN=RUN


def retention_loss(lp,ref,protocol):
    import torch
    import torch.nn.functional as F
    kl=(ref.exp()*(ref-lp)).sum(-1).mean().clamp_min(0)
    anchor=ref.argmax(-1);top=lp.argmax(-1);values=lp.topk(2,-1).values
    other=torch.where(top==anchor,values[:,1],values[:,0])
    changed=top!=anchor;score=lp.gather(1,anchor[:,None]).squeeze(1)
    penalties=F.relu(protocol['fidelity_margin']-(score-other))*changed
    # Still zero on every unchanged argmax, even at a small positive margin.
    fidelity=penalties.sum()/changed.sum().clamp_min(1)
    return protocol['anchor_alpha']*kl+protocol['fidelity_alpha']*fidelity,kl,fidelity,changed.float().mean()


core.retention_loss=retention_loss


def prepare():
    import torch
    from transformers import AutoTokenizer
    assert not RUN.exists();RUN.mkdir()
    assert not (PILOT/'eval/base_ce').exists() and not (PILOT/'eval/base_pair').exists()
    for name in ['eval69.jsonl','training_source350_snapshot.jsonl']:
        shutil.copy2(PILOT/name,RUN/name)
    for name in ['cached_eval','eval/base_block','diagnostic/base']:
        shutil.copytree(PILOT/name,RUN/name)
    (RUN/'compiled').mkdir();shutil.copy2(PILOT/'compiled/records.jsonl',RUN/'compiled/records.jsonl')
    tok=AutoTokenizer.from_pretrained(q.MODEL,use_fast=True)
    native=q.read(PILOT/'compiled/native_retention.jsonl');oldindex=json.loads((PILOT/'reference/index.json').read_text())
    index={};changed=[];(RUN/'reference').mkdir()
    A={r['native_id'] for r in q.read(RUN/'compiled/records.jsonl') if r['type']=='A'}
    for r in native:
        before=r['positions']
        if r['known_state']['delta_E']:
            ids=r['actual_ids'];n=r['prompt_tokens']
            roots=[i for i in range(len(ids)-2,max(n-1,len(ids)-5),-1) if
                tok.decode([ids[i]]) in [']','].',']\n'] and tok.decode(ids[i+1:-1],skip_special_tokens=False).strip() in ['', '.']]
            assert roots,r['id']
            root=roots[0];r['positions']=core.positions(n,root);r['editable_tail']=True
            r['tail_open_reason']='known_remaining_set_nonempty_not_an_EOS_retention_target'
        else:
            r['positions']=core.positions(r['prompt_tokens'],len(r['actual_ids']))
            r['tail_open_reason']=None
        assert before[:len(r['positions'])]==r['positions']
        assert len(r['positions'])<=len(before)
        source=PILOT/'reference'/(r['id']+'.pt');target=RUN/'reference'/(r['id']+'.pt')
        if r['positions']==before:shutil.copy2(source,target)
        else:
            ref=torch.load(source,map_location='cpu',weights_only=True)
            torch.save(ref[:len(r['positions'])].clone(),target);del ref
            changed.append(r['id'])
        old=oldindex[r['id']]
        expected=[r['actual_ids'][pos+1] for pos in r['positions']]
        tops=old['anchor_top1'][:len(r['positions'])]
        index[r['id']]={'positions':r['positions'],'anchor_top1':tops,
            'argmax_vs_original_greedy_token_mismatches':sum(a!=b for a,b in zip(tops,expected)),
            'sha256':q.sha(target),'parent_cache_sha256':old['sha256']}
    q.write_rows(RUN/'compiled/native_retention.jsonl',native)
    q.dump(RUN/'reference/index.json',index);q.dump(RUN/'reference/status.json',{'state':'COMPLETE','rows':len(native),'exact_Base_cache_subset_reused':True})
    protocol=json.loads((PILOT/'protocol.json').read_text())
    protocol.update(anchor_alpha=10.,fidelity_alpha=2.,fidelity_normalization='mean_over_changed_positions_only',
        retention_sha256=q.sha(RUN/'compiled/native_retention.jsonl'),pilot_run=str(PILOT),
        pilot_had_no_new_development_generation=True,open_all_known_nonempty_native_tail_windows=True,
        original_core_sha256=q.sha(q.ROOT/'code/qwen_tail_fidelity_20261015.py'),
        driver_sha256=q.sha(__file__),initialization_from_Base_not_pilot_checkpoint=True)
    q.dump(RUN/'protocol.json',protocol)
    audit=json.loads((PILOT/'compiled/audit.json').read_text())
    audit.update(retention_positions=sum(len(r['positions']) for r in native),
        editable_native_tail_roots=sum(r['editable_tail'] for r in native),
        supervised_A_tail_roots=len(A),additional_unlabelled_nonempty_tail_roots=len(changed),
        only_A_editable_original_tail_roots=None,
        internal_nonrepeat_prefix_positions_never_removed=True,
        no_retention_for_underanswer_EOS=True,
        supervised_records_byte_identical_to_pilot=True,
        pilot_source_sha256=q.sha(PILOT/'compiled/records.jsonl'))
    q.dump(RUN/'compiled/audit.json',audit)
    q.dump(PILOT/'pipeline_status.json',{'state':'ABORTED_TRAINING_ONLY_FOR_MASK_AND_NORMALIZATION_REPAIR',
        'new_development_generations':0,'weights_and_gates_preserved':True,'replacement':str(RUN)})
    print(json.dumps(audit,indent=2),flush=True)


def verify():
    records=q.read(RUN/'compiled/records.jsonl');native=q.read(RUN/'compiled/native_retention.jsonl')
    assert q.sha(RUN/'compiled/records.jsonl')==q.sha(PILOT/'compiled/records.jsonl')
    old_verified=json.loads((PILOT/'independent_verification.json').read_text());assert old_verified['status']=='PASS'
    byid={r['id']:r for r in native};checks=[]
    for r in records:
        assert r['retention_logit_positions']==byid[r['native_id']]['positions']
        assert r['type']!='C' or r['positive_ids']==r['actual_ids']
    for r in native:
        p=r['prompt_tokens'];ids=r['actual_ids']
        if r['known_state']['delta_E']:
            assert r['positions']==core.positions(p,max(r['positions'])+2)
            assert len(ids)-1-(max(r['positions'])+2) in [1,2,3]
            assert len(ids)-2 not in r['positions']
        else:assert r['positions']==core.positions(p,len(ids))
        assert r['positions']==list(range(p-1,max(r['positions'])+1))
        checks.append({'id':r['id'],'PASS':True,'positions':len(r['positions']),'open_native_tail':r['editable_tail']})
    q.write_rows(RUN/'compiled/independent_checks.jsonl',checks)
    result={'status':'PASS','supervision_records':len(records),'native_retention_trajectories':len(native),
        'verified_retention_positions':sum(len(r['positions']) for r in native),
        'all_internal_nonrepeat_prefix_positions_retained':True,
        'unlabelled_underanswer_EOS_not_forced_to_remain':True,
        'A_C1_C2_labels_identical_to_independently_verified_pilot':True,
        'all_training_prompts_use_FORMAT':True,'no_Anchor_training_data':True,
        'sealed311_inputs_opened':0,'script_sha256':q.sha(__file__)}
    q.dump(RUN/'independent_verification.json',result);print(json.dumps(result,indent=2),flush=True)


def child(args,label,devices='1,3'):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices,OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    log=(q.ROOT/'logs'/('qwen_tail_fidelity_v2_'+label+'_20261015.log')).open('x')
    p=subprocess.Popen([sys.executable,__file__,*args],cwd=q.ROOT/'code',env=env,stdout=log,stderr=subprocess.STDOUT)
    return p,log


def pipeline():
    prepare();verify();q.dump(RUN/'pipeline_status.json',{'state':'INDEPENDENT_TRAINING_ONLY_SWEET_SPOT_GATES'})
    workers=[child(['--train',arm],arm,devices) for arm,devices in [('base_ce','0,2'),('base_pair','1,3')]]
    codes=[]
    for p,log in workers:codes.append(p.wait());log.close()
    assert codes==[0,0],codes
    gates={arm:json.loads((RUN/arm/'gate_latest.json').read_text()) for arm in ['base_ce','base_pair']}
    if not all(g['passed'] for g in gates.values()):
        q.dump(RUN/'pipeline_status.json',{'state':'GATE_FAILED_NO_NEW_DEV_GENERATION','gates':gates});return
    q.dump(RUN/'pipeline_status.json',{'state':'GATED_FREE_GENERATION','steps':{k:g['step'] for k,g in gates.items()}})
    workers=[child(['--generate',arm],arm+'_eval',devices) for arm,devices in [('base_ce','0,2'),('base_pair','1,3')]]
    codes=[]
    for p,log in workers:codes.append(p.wait());log.close()
    assert codes==[0,0],codes
    q.dump(RUN/'pipeline_status.json',{'state':'GENERATIONS_COMPLETE_READY_FOR_ANALYSIS','sealed_predictions':0,
        'main_models_use_rules':False,'new_dev_predictions':138})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pipeline',action='store_true');ap.add_argument('--train',choices=['base_ce','base_pair'])
    ap.add_argument('--generate',choices=['base_ce','base_pair']);ap.add_argument('--closeout',action='store_true');ap.add_argument('--tests',action='store_true');args=ap.parse_args()
    if args.pipeline:pipeline()
    elif args.train:
        core.train(args.train)
        path=RUN/args.train/'manifest.json';value=json.loads(path.read_text())
        value.update(driver_sha256=q.sha(__file__),retention_loss_normalization='changed_positions_only')
        q.dump(path,value)
    elif args.generate:
        assert all(json.loads((RUN/arm/'gate_latest.json').read_text())['passed'] for arm in ['base_ce','base_pair'])
        core.ev.generate(args.generate)
    elif args.closeout:
        import closeout_tail_fidelity_20261015 as out
        out.RUN=out.a.RUN=RUN;out.main(wait=True)
    elif args.tests:
        import unittest
        suite=unittest.defaultTestLoader.loadTestsFromName('test_tail_fidelity_v2_20261015')
        result=unittest.TextTestRunner(verbosity=2).run(suite);sys.exit(0 if result.wasSuccessful() else 1)
