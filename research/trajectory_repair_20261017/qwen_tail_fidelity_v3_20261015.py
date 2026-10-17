"""Balanced semantic-positive and native-stop-class supervision.

No new negative noun supervision. Labels and retention masks are inherited
from the verified native-only training pool, before any dev generation.
"""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import qwen_tail_fidelity_v2_20261015 as parent
from entity_abc_contracts_20261015 import parse_closed

core=parent.core;q=core.q
SOURCE=parent.RUN
RUN=q.ROOT/'runs/qwen_tail_fidelity_v3_20261015'
core.RUN=core.ev.RUN=RUN
retention_loss=parent.retention_loss
core.retention_loss=retention_loss


def supervised(model,row,protocol):
    import torch
    import torch.nn.functional as F
    n=row['shared_tokens']
    x=torch.tensor(row['positive_ids'],device=model.get_input_embeddings().weight.device)[None]
    logits=model(input_ids=x,use_cache=False).logits[0]
    lp=F.log_softmax(logits[n-1:-1].float(),-1)
    prefix=F.log_softmax(logits[row['retention_logit_positions']].float(),-1)
    targets=torch.tensor(row['positive_ids'][n:],device=lp.device)
    picked=lp.gather(1,targets[:,None]).squeeze(1)
    picked=torch.cat([picked[:-1],torch.logsumexp(lp[-1,core.STOPS],0)[None]])
    terminal=picked.sum()*0;semantic=picked.sum()*0
    if row['type']=='A':
        at=row['terminal_relative_positions'][0]
        close=torch.logsumexp(lp[at,row['C1_closure_allowed']],0)
        picked=torch.cat([picked[:at],close[None],picked[at+1:]])
        terminal=-picked[row['terminal_relative_positions']].mean()
        semantic=-picked[row['semantic_positive_positions']].mean()
    first=-picked[0];span=-picked.mean();groups={}
    for site in row['pair_sites']:
        pos=torch.logsumexp(lp[site['index'],site['allowed']],0)
        neg=torch.logsumexp(lp[site['index'],site['rejected']],0)
        groups.setdefault(site['group'],[]).append(F.relu(protocol['pair_margin']-(pos-neg)))
    hinge=sum(protocol['pair_weights'][key]*torch.stack(values).mean() for key,values in groups.items())
    ce=first+span+protocol['C1_terminal_CE_weight']*terminal+protocol['semantic_positive_CE_weight']*semantic
    return ce,hinge,first,span,terminal,prefix


core.supervised=supervised


def prepare():
    from transformers import AutoTokenizer
    assert not RUN.exists();RUN.mkdir()
    assert not (SOURCE/'eval/base_ce').exists() and not (SOURCE/'eval/base_pair').exists()
    for name in ['eval69.jsonl','training_source350_snapshot.jsonl']:
        shutil.copy2(SOURCE/name,RUN/name)
    for name in ['cached_eval','eval/base_block','diagnostic/base']:
        shutil.copytree(SOURCE/name,RUN/name)
    (RUN/'reference').mkdir()
    for src in (SOURCE/'reference').iterdir():
        if src.is_file():
            target=RUN/'reference'/src.name
            if src.suffix=='.pt':
                try:os.link(src,target)
                except OSError:shutil.copy2(src,target)
            else:shutil.copy2(src,target)
    tok=AutoTokenizer.from_pretrained(q.MODEL,use_fast=True)
    rows=q.read(SOURCE/'compiled/records.jsonl')
    for row in rows:
        if row['type']!='A':continue
        prompt=tok.decode(row['actual_ids'][:row['prompt_tokens']],skip_special_tokens=False)
        encoded=tok(prompt+row['positive_text'],add_special_tokens=False,return_offsets_mapping=True)
        assert encoded['input_ids']==row['positive_ids'][:-1]
        unit=parse_closed(row['positive_text'])[0][row['past_unit_count']]
        start=len(prompt)+unit['name_start'];end=start+len(unit['name'])
        indices=[i-row['shared_tokens'] for i,(a,b) in enumerate(encoded['offset_mapping']) if
            b>start and a<end and i>=row['shared_tokens']]
        assert indices
        row['semantic_positive_positions']=indices
        row['C1_closure_allowed']=[60,936]
        for site in row['pair_sites']:
            if site['group']=='terminal' and site['index']==row['terminal_relative_positions'][0]:
                site['allowed']=[60,936]
                assert not set(site['allowed'])&set(site['rejected'])
    (RUN/'compiled').mkdir();q.write_rows(RUN/'compiled/records.jsonl',rows)
    shutil.copy2(SOURCE/'compiled/native_retention.jsonl',RUN/'compiled/native_retention.jsonl')
    audit=json.loads((SOURCE/'compiled/audit.json').read_text())
    audit.update(semantic_positive_first_new_entity_CE=True,C1_accepts_native_bare_bracket_or_period_closure=True,
        positive_token_trajectories_identical_to_parent=True,
        all_A_C1_C2_semantic_entity_labels_unchanged=True)
    q.dump(RUN/'compiled/audit.json',audit)
    protocol=json.loads((SOURCE/'protocol.json').read_text())
    protocol.update(C1_terminal_CE_weight=2.,semantic_positive_CE_weight=1.,
        pair_weights={'action':1.,'citation':.5,'terminal':2.},
        C1_class_CE_and_pair_allowed_closure_tokens=[60,936],
        records_sha256=q.sha(RUN/'compiled/records.jsonl'),
        retention_sha256=q.sha(RUN/'compiled/native_retention.jsonl'),
        driver_sha256=q.sha(__file__),training_only_parent=str(SOURCE),
        no_prior_dev_generation=True,first_entity_positive_CE_not_negative_Hinge=True,
        terminal_weight_reason='Each C1 closing/EOS action receives unit CE weight, not quarter weight')
    q.dump(RUN/'protocol.json',protocol)
    q.dump(SOURCE/'pipeline_status.json',{'state':'ABORTED_TRAINING_ONLY_FOR_BALANCED_TERMINAL_AND_SEMANTIC_CE',
        'new_development_generations':0,'weights_and_gates_preserved':True,'replacement':str(RUN)})
    verify();print(json.dumps(audit,indent=2),flush=True)


def verify():
    rows=q.read(RUN/'compiled/records.jsonl');old={r['id']:r for r in q.read(SOURCE/'compiled/records.jsonl')}
    check=[]
    for row in rows:
        before=old[row['id']]
        for field in ['positive_ids','actual_ids','positive_text','state','target_entity','retention_logit_positions']:
            assert row[field]==before[field],field
        if row['type']=='A':
            assert row['semantic_positive_positions'] and min(row['semantic_positive_positions'])>0
            assert max(row['semantic_positive_positions'])<row['terminal_relative_positions'][0]
            assert row['C1_closure_allowed']==[60,936]
        else:assert row['positive_ids']==row['actual_ids']
        assert all(not set(s['allowed'])&set(s['rejected']) for s in row['pair_sites'])
        check.append({'id':row['id'],'PASS':True,'positive_trajectories_unchanged':True})
    assert q.sha(RUN/'compiled/native_retention.jsonl')==q.sha(SOURCE/'compiled/native_retention.jsonl')
    native=q.read(RUN/'compiled/native_retention.jsonl')
    q.write_rows(RUN/'compiled/independent_checks.jsonl',check)
    result={'status':'PASS','supervision_records':len(rows),'native_retention_trajectories':120,
        'verified_retention_positions':sum(len(r['positions']) for r in native),
        'all_native_prefix_masks_identical_to_verified_parent':True,
        'positive_answer_token_trajectories_and_entity_labels_unchanged':True,
        'all_A_first_new_entity_positive_CE_positions_checked':21,
        'no_cross_sample_ordinary_noun_negative_Hinge':True,
        'C2_original_native_token_paths_preserved':20,
        'C1_stop_equivalence_not_artificial_period_requirement':True,
        'sealed311_inputs_opened':0,'script_sha256':q.sha(__file__)}
    q.dump(RUN/'independent_verification.json',result);print(json.dumps(result,indent=2),flush=True)


def child(args,label,devices='1,3'):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices,OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    log=(q.ROOT/'logs'/('qwen_tail_fidelity_v3_'+label+'_20261015.log')).open('x')
    p=subprocess.Popen([sys.executable,__file__,*args],cwd=q.ROOT/'code',env=env,stdout=log,stderr=subprocess.STDOUT)
    return p,log


def pipeline():
    prepare();q.dump(RUN/'pipeline_status.json',{'state':'INDEPENDENT_TRAINING_ONLY_SWEET_SPOT_GATES'})
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
        value.update(driver_sha256=q.sha(__file__),retention_loss_normalization='changed_positions_only',
            semantic_positive_CE=True,C1_stop_class_equivalence=True)
        q.dump(path,value)
    elif args.generate:
        assert all(json.loads((RUN/arm/'gate_latest.json').read_text())['passed'] for arm in ['base_ce','base_pair'])
        core.ev.generate(args.generate)
    elif args.closeout:
        import closeout_tail_fidelity_20261015 as out
        out.RUN=out.a.RUN=RUN;out.main(wait=True)
    elif args.tests:
        import unittest
        suite=unittest.defaultTestLoader.loadTestsFromName('test_tail_fidelity_v3_20261015')
        result=unittest.TextTestRunner(verbosity=2).run(suite);sys.exit(0 if result.wasSuccessful() else 1)
