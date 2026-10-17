"""Fixed-60 A-action/citation Hinge ablation; never merge, never gate."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import qwen_tail_fidelity_v3_20261015 as v3
from qwen_tail_guard_scan_20261015 import free_generate
from qwen_tail_open_eval_20261015 import link_tree

c = v3.core
q = c.q
SOURCE = q.ROOT / 'runs/qwen_tail_fidelity_v3_20261015'
OPEN = q.ROOT / 'runs/qwen_tail_open_suite_20261015'
RUN = q.ROOT / 'runs/qwen_pair_aonly_20261015'
BETAS = {'beta010': .10, 'beta025': .25, 'beta050': .50}


def aonly_protocol(original):
    return dict(original, pair_weights=dict(original['pair_weights'], terminal=0.))


def prepare():
    RUN.mkdir(exist_ok=False)
    for name in ['compiled', 'cached_eval']:
        shutil.copytree(SOURCE / name, RUN / name)
    for name in ['eval69.jsonl', 'training_source350_snapshot.jsonl']:
        shutil.copy2(SOURCE / name, RUN / name)
    link_tree(SOURCE / 'reference', RUN / 'reference')
    p = aonly_protocol(json.loads((SOURCE / 'protocol.json').read_text()))
    p.update(maximum_steps=60, no_training_gate=True, no_early_stopping=True,
        independent_first_gate_stopping_both_arms=False, no_prior_dev_generation=False,
        eval_checkpoint='fixed_final_60_all_three_betas', betas=BETAS,
        retention_loss='10KL+2changed_only_mean_margin_0.5', max_guard=False,
        merge_called=False, inference_rules=False, sealed311_reads=0,
        starting_checkpoint='official_Base_fresh_LoRA', source_protocol_sha256=q.sha(SOURCE/'protocol.json'),
        note='Historical compile gate fields inactive; reused development views, not blind tests.',
        script_sha256=q.sha(__file__))
    assert p['anchor_alpha'] == 10 and p['fidelity_alpha'] == 2 and p['fidelity_margin'] == .5
    assert p['anchor_kl_target'] == 0
    q.dump(RUN / 'protocol.json', p)
    q.dump(RUN / 'status.json', {'state': 'PREPARED', 'sealed311_reads': 0})


def train(task):
    import torch
    c.RUN = c.ev.RUN = RUN
    p = json.loads((RUN / 'protocol.json').read_text())
    rows = q.read(RUN / 'compiled/records.jsonl')
    native = q.read(RUN / 'compiled/native_retention.jsonl')
    bynative = {r['id']: r for r in native}
    folder = RUN / task
    folder.mkdir(exist_ok=False)
    model, tok, _ = c.ev.load('base_pair', train=True)
    init = c.adapter_init_hash(model)
    old = json.loads((SOURCE / 'base_ce/manifest.json').read_text())
    assert init == old['initial_trainable_weights_sha256']
    optimizer = torch.optim.AdamW((v for v in model.parameters() if v.requires_grad), lr=p['lr'])
    sup, aux = [], []
    maximum, accum = p['maximum_steps'], p['grad_accum']
    for target, data, salt in [(sup, rows, ''), (aux, native, 'retention')]:
        epoch = 0
        while len(target) < maximum*accum:
            target += sorted(data, key=lambda r: c.digest(str(c.SEED+epoch)+salt+r['id']))
            epoch += 1
    assert c.digest(json.dumps([r['id'] for r in sup[:maximum*accum]])) == old['order_sha256']
    assert c.digest(json.dumps([r['id'] for r in aux[:maximum*accum]])) == old['aux_order_sha256']
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    with (folder / 'visits.jsonl').open('x') as stream:
        for step in range(1, maximum+1):
            model.train()
            for j in range(accum):
                r = sup[(step-1)*accum+j]
                ce, hinge, first, span, terminal, lp = v3.supervised(model, r, p)
                own = bynative[r['native_id']]
                assert own['positions'] == r['retention_logit_positions']
                assert all(r['positive_ids'][:k+1] == own['actual_ids'][:k+1] for k in own['positions'])
                ref = torch.load(RUN/'reference'/(own['id']+'.pt'), map_location=lp.device, weights_only=True)
                penalty, kl, fid, flip = v3.retention_loss(lp, ref, p)
                loss = ce + BETAS[task]*hinge + penalty
                assert torch.isfinite(loss)
                (loss/accum).backward()
                item = {'step': step, 'id': r['id'], 'type': r['type'], 'CE': float(ce.detach()),
                    'hinge_unscaled': float(hinge.detach()), 'beta': BETAS[task],
                    'KL': float(kl.detach()), 'fidelity_penalty': float(fid.detach()), 'flip_fraction': float(flip.detach())}
                del ce, hinge, first, span, terminal, lp, ref, penalty, kl, fid, flip, loss
                extra = aux[(step-1)*accum+j]
                lp = c.get_lp(model, extra['actual_ids'], extra['positions'])
                ref = torch.load(RUN/'reference'/(extra['id']+'.pt'), map_location=lp.device, weights_only=True)
                penalty, kl, fid, flip = v3.retention_loss(lp, ref, p)
                (p['aux_retention_weight']*penalty/accum).backward()
                item.update(aux_id=extra['id'], aux_KL=float(kl.detach()))
                stream.write(json.dumps(item)+'\n'); stream.flush()
                del lp, ref, penalty, kl, fid, flip
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            assert torch.isfinite(torch.tensor(norm))
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            q.dump(folder/'status.json', {'state': 'TRAINING_FIXED_60', 'step': step, 'seconds': time.time()-started})
            print(json.dumps(dict(item, grad_norm=norm)), flush=True)
    model.save_pretrained(folder/'adapter')
    q.dump(folder/'manifest.json', {'updates': maximum, 'supervised_visits': maximum*accum,
        'aux_retention_visits': maximum*accum, 'initial_trainable_weights_sha256': init,
        'final_trainable_weights_sha256': c.adapter_init_hash(model),
        'adapter_sha256': q.sha(folder/'adapter/adapter_model.safetensors'), 'beta': BETAS[task],
        'data_sha256': p['records_sha256'], 'retention_sha256': p['retention_sha256'],
        'order_sha256': old['order_sha256'], 'aux_order_sha256': old['aux_order_sha256'],
        'C1_C2_Hinge_weight': 0, 'A_action_weight': BETAS[task], 'A_citation_weight': .5*BETAS[task],
        'retention_loss': p['retention_loss'], 'max_guard': False, 'merge_called': False,
        'old_Anchor_loaded': False, 'evaluated_regardless_of_training_metrics': True, 'script_sha256': q.sha(__file__)})
    del optimizer, model
    gc.collect(); torch.cuda.empty_cache()


def profile(task):
    from peft import PeftModel
    from peft.tuners.lora.layer import LoraLayer
    c.RUN = c.ev.RUN = RUN
    folder = RUN/task/'profile'
    folder.mkdir(exist_ok=False)
    model, tok, placement = q.load_model('base')
    adapter = RUN/task/'adapter'
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    layers = [m for m in model.modules() if isinstance(m, LoraLayer)]
    assert layers and not any(m.merged for m in layers)
    q.dump(folder/'manifest.json', {'task': task, 'placement': placement, 'merge_called': False,
        'LoRA_layers': len(layers), 'all_layers_unmerged': True, 'inference_rules': False,
        'adapter_sha256': q.sha(adapter/'adapter_model.safetensors'), 'max_new_tokens': 256,
        'do_sample': False, 'script_sha256': q.sha(__file__)})
    fidelity = c.fidelity_probe(model, json.loads((RUN/'reference/index.json').read_text()), folder/'fidelity')
    diagnostic = c.diagnose(model, tok, task, folder/'diagnostic')
    records = {r['question_id']: r for r in q.read(RUN/'compiled/records.jsonl') if r['type'] == 'A'}
    ti = {r['id']: r for r in q.read(RUN/'training_source350_snapshot.jsonl')}
    ar = [{'id': k, 'prompt_token_ids': r['actual_ids'][:r['prompt_tokens']], 'record': r, 'input': ti[k]} for k,r in sorted(records.items())]
    out = free_generate(model, tok, ar, folder/'training_A14', training=True)
    ts = {'n': 14, 'full_prefix_preserved': sum(r['old_repair_root_reached_exactly'] for r in out),
        'mean_tokens': sum(r['tokens'] for r in out)/14, 'true_cap': sum(r['hit_cap_without_eos'] for r in out)}
    for mode in ['name_only', 'locally_cited']:
        ts[mode] = {k: sum(len(r[mode][k]) for r in out) for k in ['added', 'lost']}
    q.dump(folder/'training_A14/summary.json', ts)
    base = {r['id']: r for r in q.read(RUN/'cached_eval/base.jsonl')}
    er = [{'id': r['id'], 'prompt_token_ids': base[r['id']]['prompt_token_ids']} for r in q.read(RUN/'eval69.jsonl')]
    assert len(er) == 69
    free_generate(model, tok, er, folder/'eval69')
    q.dump(folder/'status.json', {'state': 'COMPLETE', 'fidelity': fidelity, 'diagnostic': diagnostic, 'training_A14': ts, 'eval_predictions': 69})
    q.dump(RUN/task/'status.json', {'state': 'COMPLETE', 'step': 60, 'merge_called': False})


def launch(task, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
    log = (q.ROOT/'logs'/('qwen_pair_aonly_'+task+'_20261015.log')).open('x')
    return subprocess.Popen([sys.executable, __file__, '--job', task], cwd=q.ROOT/'code', env=env, stdout=log, stderr=subprocess.STDOUT), log


def wait(worker):
    p, log = worker
    rc = p.wait(); log.close()
    assert rc == 0, rc


def pipeline():
    prepare()
    slow = launch('beta010', '0,2')
    wait(launch('beta025', '1,3'))
    fast = launch('beta050', '1,3')
    wait(slow); wait(fast)
    q.dump(RUN/'status.json', {'state': 'ALL_GENERATIONS_COMPLETE', 'new_predictions': 207, 'sealed311_reads': 0})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--pipeline', action='store_true'); ap.add_argument('--job', choices=BETAS)
    args = ap.parse_args()
    if args.pipeline: pipeline()
    elif args.job: train(args.job); profile(args.job)
