"""Fixed-budget max-prefix-violation training and static merged-weight scan.

No training metric can suppress evaluation. All declared configurations run.
"""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import qwen_tail_fidelity_v3_20261015 as v3
from entity_abc_contracts_20261015 import inventory, normalize, common_prefix
from qwen_tail_open_eval_20261015 import link_tree

c = v3.core
q = c.q
SOURCE = q.ROOT / 'runs/qwen_tail_fidelity_v3_20261015'
ROOT = q.ROOT / 'runs/qwen_tail_open_suite_20261015'
GUARD = ROOT / 'guard60'
TASKS = {
    'scale070': {'arm': 'base_pair', 'lambda': .7, 'origin': 'frozen60'},
    'scale085': {'arm': 'base_pair', 'lambda': .85, 'origin': 'frozen60'},
    'scale100': {'arm': 'base_pair', 'lambda': 1., 'origin': 'frozen60'},
    'scale115': {'arm': 'base_pair', 'lambda': 1.15, 'origin': 'frozen60'},
    'ce60_merged': {'arm': 'base_ce', 'lambda': 1., 'origin': 'frozen60'},
    'guard_ce': {'arm': 'base_ce', 'lambda': 1., 'origin': 'guard60'},
    'guard_pair': {'arm': 'base_pair', 'lambda': 1., 'origin': 'guard60'},
}


def guard_loss(lp, ref, protocol):
    import torch
    import torch.nn.functional as F
    kl = (ref.exp() * (ref-lp)).sum(-1).mean().clamp_min(0)
    anchor = ref.argmax(-1)
    top = lp.argmax(-1)
    values = lp.topk(2, -1).values
    other = torch.where(top == anchor, values[:, 1], values[:, 0])
    margin = lp.gather(1, anchor[:, None]).squeeze(1)-other
    changed = top != anchor
    old_mean = (F.relu(protocol['fidelity_margin']-margin)*changed).sum()/changed.sum().clamp_min(1)
    worst = F.relu(protocol['guard_margin']-margin).max()
    penalty = protocol['anchor_alpha']*kl + protocol['fidelity_alpha']*old_mean + protocol['guard_alpha']*worst
    return penalty, kl, old_mean, worst, changed.float().mean()


def prepare():
    assert json.loads((ROOT / 'status.json').read_text())['state'] == 'STAGE1_GENERATIONS_COMPLETE'
    assert not GUARD.exists()
    GUARD.mkdir()
    for name in ['compiled', 'cached_eval']:
        shutil.copytree(SOURCE / name, GUARD / name)
    for name in ['eval69.jsonl', 'training_source350_snapshot.jsonl']:
        shutil.copy2(SOURCE / name, GUARD / name)
    link_tree(SOURCE / 'reference', GUARD / 'reference')
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    protocol.update(maximum_steps=60, no_training_gate=True, no_early_stopping=True,
        guard_margin=.3, guard_alpha=2., guard_in_addition_to_existing_changed_only_mean=True,
        eval_checkpoint='fixed_final_step_60_regardless_of_metrics',
        guard_applies_to_all_retained_positions_including_positive_small_margins=True,
        starting_checkpoint='official_Base_fresh_LoRA_not_60_step_adapter',
        driver_sha256=q.sha(__file__), sealed311_reads=0)
    q.dump(GUARD / 'protocol.json', protocol)
    q.dump(ROOT / 'stage2_protocol.json', {'tasks': TASKS, 'all_configs_evaluated': True,
        'scaled_delta_formula': 'W_merged = W_Base + lambda * (alpha/r) * B @ A',
        'merge_once_in_memory_not_per_token': True, 'full_merged_weights_not_exported': True,
        'training_guard': protocol, 'source_protocol_sha256': q.sha(SOURCE / 'protocol.json'),
        'script_sha256': q.sha(__file__), 'no_sealed_data': True})


def train(arm):
    import torch
    c.RUN = c.ev.RUN = GUARD
    protocol = json.loads((GUARD / 'protocol.json').read_text())
    rows = q.read(GUARD / 'compiled/records.jsonl')
    native = q.read(GUARD / 'compiled/native_retention.jsonl')
    bynative = {r['id']: r for r in native}
    folder = GUARD / arm
    folder.mkdir()
    model, tok, _ = c.ev.load(arm, train=True)
    init = c.adapter_init_hash(model)
    old = json.loads((SOURCE / arm / 'manifest.json').read_text())
    assert init == old['initial_trainable_weights_sha256']
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=protocol['lr'])
    sup, aux = [], []
    maximum, accum = protocol['maximum_steps'], protocol['grad_accum']
    epoch = 0
    while len(sup) < maximum*accum:
        sup += sorted(rows, key=lambda r: c.digest(str(c.SEED+epoch)+r['id']))
        epoch += 1
    epoch = 0
    while len(aux) < maximum*accum:
        aux += sorted(native, key=lambda r: c.digest(str(c.SEED+epoch)+'retention'+r['id']))
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
                ce, hinge, first, span, terminal, lp = v3.supervised(model, r, protocol)
                own = bynative[r['native_id']]
                assert own['positions'] == r['retention_logit_positions']
                assert all(r['positive_ids'][:p+1] == own['actual_ids'][:p+1] for p in own['positions'])
                ref = torch.load(GUARD / 'reference' / (own['id']+'.pt'), map_location=lp.device, weights_only=True)
                penalty, kl, mean, worst, flip = guard_loss(lp, ref, protocol)
                loss = ce + (hinge if arm == 'base_pair' else 0) + penalty
                assert torch.isfinite(loss)
                (loss/accum).backward()
                item = {'step': step, 'id': r['id'], 'type': r['type'], 'CE': float(ce.detach()),
                    'hinge': float(hinge.detach()), 'hinge_used': arm == 'base_pair',
                    'KL': float(kl.detach()), 'old_mean_guard': float(mean.detach()),
                    'max_guard': float(worst.detach()), 'flip_fraction': float(flip.detach())}
                del ce, hinge, first, span, terminal, lp, ref, penalty, kl, mean, worst, flip, loss
                extra = aux[(step-1)*accum+j]
                lp = c.get_lp(model, extra['actual_ids'], extra['positions'])
                ref = torch.load(GUARD / 'reference' / (extra['id']+'.pt'), map_location=lp.device, weights_only=True)
                penalty, kl, mean, worst, flip = guard_loss(lp, ref, protocol)
                (protocol['aux_retention_weight']*penalty/accum).backward()
                item.update(aux_id=extra['id'], aux_KL=float(kl.detach()), aux_max_guard=float(worst.detach()))
                stream.write(json.dumps(item)+'\n')
                stream.flush()
                del lp, ref, penalty, kl, mean, worst, flip
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            assert torch.isfinite(torch.tensor(norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            q.dump(folder / 'status.json', {'state': 'TRAINING_FIXED_BUDGET', 'step': step,
                'max_steps': maximum, 'seconds': time.time()-started, 'grad_norm': norm})
            print(json.dumps(dict(item, grad_norm=norm)), flush=True)
    model.save_pretrained(folder / 'adapter')
    manifest = {'updates': maximum, 'supervised_visits': maximum*accum, 'aux_retention_visits': maximum*accum,
        'initial_trainable_weights_sha256': init, 'final_trainable_weights_sha256': c.adapter_init_hash(model),
        'adapter_sha256': q.sha(folder / 'adapter/adapter_model.safetensors'),
        'data_sha256': protocol['records_sha256'], 'retention_sha256': protocol['retention_sha256'],
        'order_sha256': old['order_sha256'], 'aux_order_sha256': old['aux_order_sha256'],
        'old_Anchor_loaded': False, 'starting_model': q.MODEL, 'guard_loss': '10KL+2changed_mean+2max_margin_0.3',
        'script_sha256': q.sha(__file__), 'evaluated_regardless_of_training_metrics': True}
    q.dump(folder / 'manifest.json', manifest)
    q.dump(folder / 'status.json', {'state': 'TRAINED_FIXED_60_READY_FOR_UNCONDITIONAL_PROFILE', 'steps': maximum})
    del optimizer, model
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def free_generate(model, tok, rows, folder, training=False):
    import re
    import torch
    folder.mkdir(parents=True, exist_ok=False)
    output = []
    model.eval()
    model.config.use_cache = True
    with torch.inference_mode(), (folder / 'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for i, row in enumerate(rows):
            prompt = row['prompt_token_ids']
            x = torch.tensor(prompt, device=model.get_input_embeddings().weight.device)[None]
            ids = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=256,
                do_sample=False, pad_token_id=tok.eos_token_id)[0, len(prompt):].tolist()
            p = {'id': row['id'], 'tokens': len(ids), 'output': tok.decode(ids, skip_special_tokens=True).strip(),
                'prompt_token_ids': prompt, 'generated_token_ids': ids, 'blocking_events': [],
                'ended_by_eos': bool(ids and ids[-1] in c.STOPS),
                'hit_cap_without_eos': bool(len(ids) == 256 and ids[-1] not in c.STOPS)}
            if training:
                source = row['record']
                base_ids = source['actual_ids'][source['prompt_tokens']:]
                base_text = tok.decode(base_ids, skip_special_tokens=True).strip()
                p.update(base_tokens=len(base_ids), base_output=base_text,
                    old_repair_root_reached_exactly=(prompt+ids)[:source['repair_root_tokens']] == source['positive_ids'][:source['repair_root_tokens']])
                groups = inventory(row['input'])
                for mode, cited in [('name_only', False), ('locally_cited', True)]:
                    def hits(text):
                        units = [u.strip() for u in text.rstrip().rstrip('.').rstrip(',').split(',') if normalize(u)]
                        return {g['key'] for g in groups if g['supporting_doc_ids'] and any(normalize(u) in g['values'] and
                            (not cited or set(g['supporting_doc_ids']) & {int(v) for v in re.findall(r'\[(\d+)\]', u)}) for u in units)}
                    a, b = hits(p['output']), hits(base_text)
                    p[mode] = {'base': sorted(b), 'current': sorted(a), 'added': sorted(a-b), 'lost': sorted(b-a),
                        'known_delta_now_hit': sorted(a & set(source['state']['delta_E']))}
            stream.write(json.dumps(p, ensure_ascii=False)+'\n')
            stream.flush()
            output.append(p)
            q.dump(folder / 'status.json', {'state': 'GENERATING', 'completed': i+1, 'total': len(rows)})
    q.dump(folder / 'status.json', {'state': 'COMPLETE', 'completed': len(rows), 'total': len(rows)})
    return output


def profile(task):
    import torch
    from peft import PeftModel
    from peft.tuners.lora.layer import LoraLayer
    spec = TASKS[task]
    folder = ROOT / 'profiles' / task
    folder.mkdir(parents=True, exist_ok=False)
    q.dump(folder / 'status.json', {'state': 'LOADING_AND_STATIC_MERGING'})
    origin = SOURCE if spec['origin'] == 'frozen60' else GUARD
    adapter = origin / spec['arm'] / 'adapter'
    model, tok, placement = q.load_model('base')
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    layers = [m for m in model.modules() if isinstance(m, LoraLayer)]
    assert layers
    before = [dict(m.scaling) for m in layers]
    for m in layers:
        m.scale_layer(spec['lambda'])
    for m, old in zip(layers, before):
        assert all(abs(m.scaling[key]-value*spec['lambda']) < 1e-9 for key, value in old.items())
    merged = model.merge_and_unload(safe_merge=True)
    assert not any(isinstance(m, LoraLayer) for m in merged.modules())
    merged.eval()
    model = merged
    c.RUN = c.ev.RUN = GUARD
    index = json.loads((GUARD / 'reference/index.json').read_text())
    q.dump(folder / 'manifest.json', {'task': task, 'spec': spec, 'adapter': str(adapter),
        'adapter_sha256': q.sha(adapter / 'adapter_model.safetensors'), 'placement': placement,
        'merged_once': True, 'LoRA_layers_before_merge': len(layers), 'LoRA_layers_after_merge': 0,
        'weight_dtype': str(next(model.parameters()).dtype), 'safe_merge': True,
        'per_token_LoRA_or_rule_overhead': False, 'script_sha256': q.sha(__file__)})
    del layers, before
    fidelity = c.fidelity_probe(model, index, folder / 'fidelity')
    diagnostic = c.diagnose(model, tok, task, folder / 'diagnostic')
    records = {r['question_id']: r for r in q.read(GUARD / 'compiled/records.jsonl') if r['type'] == 'A'}
    train_inputs = {r['id']: r for r in q.read(GUARD / 'training_source350_snapshot.jsonl')}
    arows = [{'id': key, 'prompt_token_ids': r['actual_ids'][:r['prompt_tokens']], 'record': r, 'input': train_inputs[key]} for key, r in sorted(records.items())]
    aout = free_generate(model, tok, arows, folder / 'training_A14', training=True)
    training_summary = {'n': 14, 'training_only': True,
        'full_prefix_preserved': sum(p['old_repair_root_reached_exactly'] for p in aout),
        'mean_tokens': sum(p['tokens'] for p in aout)/14,
        'true_cap': sum(p['hit_cap_without_eos'] for p in aout)}
    for mode in ['name_only', 'locally_cited']:
        training_summary[mode] = {key: sum(len(p[mode][key]) for p in aout) for key in ['added', 'lost']}
        training_summary[mode]['questions_missing_entity_recovered'] = sum(bool(p[mode]['known_delta_now_hit']) for p in aout)
    q.dump(folder / 'training_A14/summary.json', training_summary)
    base = {p['id']: p for p in q.read(SOURCE / 'cached_eval/base.jsonl')}
    eval_rows = q.read(GUARD / 'eval69.jsonl')
    erows = [{'id': row['id'], 'prompt_token_ids': base[row['id']]['prompt_token_ids']} for row in eval_rows]
    assert len(erows) == 69
    free_generate(model, tok, erows, folder / 'eval69')
    q.dump(folder / 'status.json', {'state': 'COMPLETE', 'fidelity': fidelity, 'diagnostic': diagnostic,
        'training_A14': training_summary, 'eval_predictions': 69, 'no_training_gate': True})


def launch(args, label, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
    log = (q.ROOT / 'logs' / ('qwen_tail_open_' + label + '_20261015.log')).open('x')
    p = subprocess.Popen([sys.executable, __file__, *args], cwd=q.ROOT / 'code', env=env, stdout=log, stderr=subprocess.STDOUT)
    return p, log


def wait_all(workers):
    codes = []
    for p, log in workers:
        codes.append(p.wait())
        log.close()
    assert all(code == 0 for code in codes), codes


def guard_job(arm):
    train(arm)
    profile('guard_pair' if arm == 'base_pair' else 'guard_ce')


def pipeline():
    # The first phase must finish, but its quality metrics never block this one.
    for _ in range(720):
        path = ROOT / 'status.json'
        if path.exists() and json.loads(path.read_text())['state'] == 'STAGE1_GENERATIONS_COMPLETE':
            break
        time.sleep(10)
    else:
        raise TimeoutError('Stage1 generation did not finish')
    import analyze_tail_open_suite_20261015 as analysis
    analysis.analyze(final=False)
    prepare()
    q.dump(ROOT / 'status.json', {'state': 'STAGE2_STATIC_SCAN'})
    wait_all([launch(['--profile', task], task, gpu) for task, gpu in [('scale070', '0,2'), ('scale100', '1,3')]])
    wait_all([launch(['--profile', task], task, gpu) for task, gpu in [('scale085', '0,2'), ('scale115', '1,3')]])
    q.dump(ROOT / 'status.json', {'state': 'STAGE2_MAX_GUARD_TRAINING_AND_MERGE_CONTROL'})
    slow = launch(['--guard-job', 'base_ce'], 'guard_ce', '0,2')
    wait_all([launch(['--profile', 'ce60_merged'], 'ce60_merged', '1,3')])
    fast = launch(['--guard-job', 'base_pair'], 'guard_pair', '1,3')
    wait_all([slow, fast])
    q.dump(ROOT / 'status.json', {'state': 'ALL_GENERATIONS_COMPLETE', 'new_eval_predictions': 621, 'sealed_reads': 0})
    analysis.analyze(final=True)
    import closeout_tail_open_suite_20261015 as out
    out.main()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pipeline', action='store_true')
    ap.add_argument('--profile', choices=list(TASKS))
    ap.add_argument('--guard-job', choices=['base_ce', 'base_pair'])
    args = ap.parse_args()
    if args.pipeline:
        pipeline()
    elif args.profile:
        profile(args.profile)
    elif args.guard_job:
        guard_job(args.guard_job)
