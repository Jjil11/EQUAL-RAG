"""Corrected short-span CE/Pair at the actual original Native suspect states.

Reuse frozen tested loss functions, but move supervision start to audited roots.
For continue states the paths initially agree; do not demand a false stop action.
"""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import json
import shutil
import subprocess
import sys
import time
import qwen_equal_graph_20261014 as q
import qwen_gemini_span_20261014 as m
import eval_qwen_gemini_span_20261014 as transfer
import eval_qwen_equal_curriculum_transfer_20261015 as curriculum_transfer

ORIGINAL=m.RUN
RUN=q.ROOT/'runs/qwen_gemini_native_span_20261015'


def prepare():
    assert not RUN.exists();RUN.mkdir()
    records=q.read(ORIGINAL/'native_prefix_records_20261015.jsonl');assert len(records)==13
    candidates=[]
    for r in records:
        n=len(r['native_context_ids'])
        assert all(p[:n]==r['native_context_ids'] for p in r['candidate_paths'])
        assert r['actual_ids'][:n]==r['native_context_ids']
        x=dict(r,shared_tokens=n,retention_logit_positions=m.buffer_positions(r['prompt_tokens'],n,2),
            positive_suffix_lengths=[len(p)-n for p in r['candidate_paths']],negative_suffix_length=len(r['actual_ids'])-n)
        for key in ['positive_ids','reference_variant_scores','selected_variant']:x.pop(key,None)
        assert max(x['positive_suffix_lengths'])<=32 and x['negative_suffix_length']<=160
        assert all(p[-1] in m.STOPS for p in x['candidate_paths']) and x['actual_ids'][-1] in m.STOPS
        candidates.append(x)
    q.write_rows(RUN/'candidates.jsonl',candidates)
    q.write_rows(RUN/'validation.jsonl',q.read(ORIGINAL/'validation.jsonl'))
    q.dump(RUN/'compile_audit.json',{'gate':True,'rows':13,'stop':8,'continue':5,
        'native_roots_verified_in_all_positive_and_actual_paths':True,'invented_cap_EOS':0,
        'candidate_positive_lengths':{r['id']:r['positive_suffix_lengths'] for r in candidates},
        'code_sha256':q.sha(__file__)})
    q.dump(RUN/'protocol.json',{'arms':m.ARMS,'rows':13,'epochs':2,'lr':1e-5,'updates_each':8,
        'starting_adapter_sha256':q.sha(q.ANCHOR/'adapter_model.safetensors'),
        'supervision_start':'Actual audited original Native prefix, rewind determined by tokenizer. NOT later longest edit divergence.',
        'continue_first_positive_and_negative_can_agree':True,'false_continue_stop_labels':0,
        'kl_alpha':1,'kl_target':.005,'buffer':2,'margin':2,'EOS_body_same_per_token_CE':True,
        'loss_helper_sha256':q.sha(q.ROOT/'code/qwen_gemini_span_20261014.py'),
        'code_sha256':q.sha(__file__),'candidates_sha256':q.sha(RUN/'candidates.jsonl'),
        'validation_sha256':q.sha(RUN/'validation.jsonl'),'full_prompt_rows':24,
        'new_full_prompt_predictions':48,'cached_full_prompt_controls':48,
        'cached_anchor_native_prefix_diagnostics':13,'new_native_prefix_diagnostics':26,
        'paired_comparison_family':7,'zero_known_edge_questions':sum(not r['edges'] for r in q.read(RUN/'validation.jsonl')),
        'point_screen':'Q/F1 within.02anchor and cost<=.9anchor, within.02anchor repeat rule and cost<=rule.',
        'same_labels_initialization_order_updates_compute_branches':True,'same_gradient_or_label_budget':False,
        'no_blind_confirmation':True,'sealed_predictions':0,'seed_count':1,'no_hyperparameter_or_checkpoint_search':True})
    q.dump(RUN/'status.json',{'state':'WAITING_FOR_OWN_GPU_WORKERS'})


def pipeline():
    prepare();until=time.time()+14400
    while True:
        state=json.loads((ORIGINAL/'native_prefix_status_20261015.json').read_text())['state']
        assert state!='FAILED' and time.time()<until
        if state=='COMPLETE':break
        time.sleep(15)
    logroot=q.ROOT/'logs/qwen_gemini_native_span_20261015';logroot.mkdir()
    tasks=[('reference',['--cache'])]
    for arm in m.ARMS:tasks += [(arm+'_train',['--train',arm]),(arm+'_native_diagnostic',['--diagnose',arm]),(arm+'_full_prompt',['--generate',arm])]
    for name,args in tasks:
        q.dump(RUN/'status.json',{'state':'RUNNING','stage':name})
        with (logroot/(name+'.log')).open('x') as log:
            subprocess.run([sys.executable,__file__,*args],stdout=log,stderr=subprocess.STDOUT,check=True)
    for which in ['base','anchor']:
        root=RUN/'eval'/(which+'__greedy');shutil.copytree(ORIGINAL/'eval'/(which+'__greedy'),root)
        q.dump(root/'reuse_provenance.json',{'source':str(ORIGINAL/'eval'/(which+'__greedy')),
            'source_predictions_sha256':q.sha(root/'predictions.jsonl'),'not_a_new_generation':True})
    q.dump(RUN/'status.json',{'state':'COMPLETE','new_full_prompt_predictions':48,'new_native_prefix_diagnostics':26})


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--cache',action='store_true');ap.add_argument('--train',choices=m.ARMS)
    ap.add_argument('--diagnose',choices=m.ARMS);ap.add_argument('--generate',choices=m.ARMS);a=ap.parse_args()
    if a.cache or a.train or a.diagnose or a.generate:
        m.RUN=RUN
        if a.cache:m.cache()
        elif a.train:m.train(a.train)
        elif a.diagnose:m.diagnose(a.diagnose)
        else:transfer.generate(a.generate)
    else:
        try:pipeline()
        except Exception:q.dump(RUN/'status.json',{'state':'FAILED'});raise
