"""Frozen 60-step, 240-visit whole-answer CE and standard sequence DPO."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse
import gc
import json
import time
from supp_runtime import *

def dpo_loss(chosen, rejected, ref_chosen, ref_rejected):
    import torch.nn.functional as F
    return -F.logsigmoid(.1 * ((chosen-rejected)-(ref_chosen-ref_rejected)))

def train(arm):
    import torch
    from qwen_base_entity_abc_train_20261015 import adapter_init_hash
    rr = qwen_records()
    folder = SUITE / 'package2' / arm
    folder.mkdir(parents=True, exist_ok=False)
    model, tok, mapping = load(train=True)
    init = adapter_init_hash(model)
    assert init == js(SOURCE / 'base_ce/manifest.json')['initial_trainable_weights_sha256']
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-5)
    refs = {}
    if arm == 'matched_dpo':
        model.eval()
        with model.disable_adapter(), torch.no_grad():
            for r in rr:
                if r['type'] != 'A':
                    continue
                pos, _ = sequence_logp(model, r['positive_ids'], r['prompt_tokens'])
                neg, _ = sequence_logp(model, r['actual_ids'], r['prompt_tokens'])
                refs[r['id']] = [float(pos), float(neg)]
        dump(folder / 'reference_sequence_logps.json', refs)
    visits = order(rr)
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    with safe(folder/'visits.jsonl').open('x', encoding='utf-8') as f:
        for step in range(60):
            model.train()
            for r in visits[step*4:step*4+4]:
                pos, count = sequence_logp(model, r['positive_ids'], r['prompt_tokens'])
                ce = -pos/count
                pref = ce*0
                if arm == 'matched_dpo' and r['type'] == 'A':
                    neg, _ = sequence_logp(model, r['actual_ids'], r['prompt_tokens'])
                    pref = dpo_loss(pos, neg, *refs[r['id']])
                # C2 chosen==rejected: no meaningful preference. C2 receives .1 CE.
                loss = ce if arm == 'matched_sft' else pref+.1*ce
                assert torch.isfinite(loss)
                (loss/4).backward()
                f.write(json.dumps({'step':step+1,'id':r['id'],'type':r['type'],
                    'whole_answer_CE':float(ce.detach()),'DPO':float(pref.detach()),
                    'supervised_response_tokens':count})+'\n')
                f.flush()
                del pos, ce, pref, loss
                if 'neg' in locals():
                    del neg
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            assert torch.isfinite(torch.tensor(norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            dump(folder/'training_status.json', {'step':step+1,'seconds':time.time()-started,
                 'grad_norm':norm}, replace=True)
            print(json.dumps({'arm':arm,'step':step+1,'seconds':time.time()-started,'grad_norm':norm}), flush=True)
    model.save_pretrained(safe(folder/'adapter'))
    dump(folder/'manifest.json', {'arm':arm,'steps':60,'record_visits':240,'seed':SEED,
         'lr':2e-5,'grad_accum':4,'max_grad_norm':1.,'LoRA':{'r':8,'alpha':16,'dropout':0},
         'records_sha256':RECORD_SHA,'order_sha256':order_sha(visits),
         'initial_trainable_weights_sha256':init,'adapter_sha256':sha(folder/'adapter/adapter_model.safetensors'),
         'reference':'Official frozen Base Qwen3-14B','DPO_beta':.1 if arm=='matched_dpo' else None,
         'DPO_sequence_score':'sum of response-token log probabilities; no length normalization',
         'DPO_C2_policy':'No preference gradient for identical pairs; weight .1 whole-answer CE stabilization',
         'SFT_weight':.1 if arm=='matched_dpo' else 1.,'prefix_masking':False,'retention_guard':False,
         'auxiliary_retention_visits':0,'EQUAL_auxiliary_retention_visits':240,
         'equal_updates_and_supervised_visits_not_equal_FLOPs':True,'training_seconds':time.time()-started,
         'placement':mapping,'script_sha256':sha(__file__)})
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()

def evaluate(arm):
    folder = SUITE / 'package2' / arm
    if arm == 'exhaustive_prompt':
        folder.mkdir(parents=True, exist_ok=False)
    model, tok, mapping = load(adapter=None if arm=='exhaustive_prompt' else folder/'adapter')
    if arm == 'exhaustive_prompt':
        dump(folder/'manifest.json', {'training':False,'prompt_appendix':EXHAUSTIVE,
             'placement':mapping,'script_sha256':sha(__file__)})
    baseline = {}
    for p in read(SOURCE/'cached_eval/base.jsonl') + read(HELD/'predictions/base/predictions.jsonl'):
        baseline[p['id']] = p['prompt_token_ids']
    inference(model, tok, eval_rows(), folder/'predictions380.jsonl', exhaustive=arm=='exhaustive_prompt',
              expected_prompt_ids=None if arm=='exhaustive_prompt' else baseline)
    if arm == 'exhaustive_prompt':
        # A prompt intervention has no defined teacher-forced native-prefix Ret./120.
        dump(folder/'retention_summary.json', {'exact':None,'trajectories':120,
             'reason':'N/A: modified conditioning prompt; unmodified-prompt parameter identity is 120/120 by construction.'})
    else:
        fidelity(model, qwen_native(), folder)
    dump(folder/'COMPLETE.json', {'state':'COMPLETE','predictions_sha256':sha(folder/'predictions380.jsonl')})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('arm', choices=['exhaustive_prompt','matched_sft','matched_dpo'])
    ap.add_argument('--eval-only', action='store_true')
    a = ap.parse_args()
    if a.arm != 'exhaustive_prompt' and not a.eval_only:
        train(a.arm)
    evaluate(a.arm)

if __name__ == '__main__':
    main()
