"""Training-prefix mechanism diagnostic; read-only weights, not new test evidence."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from train_verified_boundary_20261010 import aligned_trajectory

ROOT = Path(str(_release_root))
RUN = ROOT / 'runs/submission_20261010'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=['qwen3','mistral'], required=True)
    args = parser.parse_args()
    family = args.family
    if family == 'qwen3':
        base = str(_release_model)
        anchor = ROOT / 'runs/qwen3_14b_compression_calibration/pref_0p4/dev/adapter'
        data = ROOT / 'data/qampari_evidence_full/train_trajectory_qwen3_14b_onpolicy_seed20260930.jsonl'
        arms = ['baseline','semantic_eos','unit_ul','wrong_boundary','kl_only']
    else:
        base = '/common_data/model/Mistral-7B-Instruct-v0.3'
        anchor = ROOT / 'runs/aequal_kl_dev/k0_conservative/adapter'
        data = ROOT / 'data/qampari_evidence_full/train_trajectory_onpolicy_seed20260930.jsonl'
        arms = ['baseline','semantic_eos','unit_ul']
    rows = [json.loads(x) for x in data.read_text().splitlines() if x.strip()]
    rows = sorted([x for x in rows if x.get('has_trajectory')], key=lambda x: x['id'])
    assert len(rows) == (39 if family == 'qwen3' else 135)
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True)
    tokenizer._equal_rag_disable_thinking = family == 'qwen3'
    contexts = [aligned_trajectory(tokenizer, row, 2048) for row in rows]
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(anchor), adapter_name='baseline')
    # Assemble every adapter on CPU, as in the main experiment, before CUDA.
    for arm in arms[1:]:
        model.load_adapter(str(RUN / f'{family}_{arm}' / 'interp05/adapter'), adapter_name=arm, is_trainable=False)
    model = model.cuda()
    model.config.use_cache = False
    records, arrays = [], {}
    with torch.inference_mode():
        for arm in arms:
            model.set_adapter(arm)
            model.eval()
            values = []
            for row, context in zip(rows, contexts):
                ids, _, _, canonical, stops, _ = context
                lp = torch.log_softmax(model(input_ids=ids[None].cuda(), use_cache=False).logits[0,-1].float(),-1)
                competing = lp.clone()
                competing[stops] = -torch.inf
                stop_mass = torch.logsumexp(lp[stops],0)
                value = {'id':row['id'],'arm':arm,'canonical_eot_id':canonical,
                         'eos_ids':stops,'context_tokens':int(ids.numel()),
                         'canonical_eot_logp':float(lp[canonical]),
                         'total_stop_probability':float(stop_mass.exp()),
                         'canonical_margin':float(lp[canonical]-competing.max()),
                         'stop_class_margin':float(stop_mass-competing.max())}
                records.append(value)
                values.append([value[k] for k in ['canonical_eot_logp','total_stop_probability','canonical_margin','stop_class_margin']])
            arrays[arm] = np.array(values)
            print(f'{family} {arm} {len(rows)} prefixes measured',flush=True)
    names = ['canonical_eot_logp','total_stop_probability','canonical_margin','stop_class_margin']
    ix = np.random.default_rng(20261010).integers(0,len(rows),(20000,len(rows)))
    comparisons = {}
    for arm in arms[1:]:
        diff = arrays[arm]-arrays['baseline']
        boot = diff[ix].mean(1)
        comparisons[f'{arm}_minus_baseline'] = {name:{'delta':float(diff[:,i].mean()),
            'ci95':np.quantile(boot[:,i],[.025,.975]).tolist()} for i,name in enumerate(names)}
    diff = arrays['semantic_eos']-arrays['unit_ul']
    boot = diff[ix].mean(1)
    comparisons['semantic_eos_minus_unit_ul'] = {name:{'delta':float(diff[:,i].mean()),
        'ci95':np.quantile(boot[:,i],[.025,.975]).tolist()} for i,name in enumerate(names)}
    report = {'status':'COMPLETE','family':family,'n_training_prefixes':len(rows),
              'scope':'All active original training prefixes; same fixed token-aligned contexts for every arm.',
              'warning':'In-sample teacher-forced mechanism diagnostic, not independent effectiveness or heldout quality evidence. Total EOS mass is only measured here; main training does not use EOS-class aggregation.',
              'weights_updated':False,'new_generation':False,'holdout_opened':False,
              'training_data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
              'means':{arm:dict(zip(names,arrays[arm].mean(0).tolist())) for arm in arms},
              'comparisons':comparisons,'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30}
    folder = RUN / 'stop_probability_diagnostic'
    folder.mkdir(exist_ok=True)
    (folder / f'{family}_prefixes.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in records))
    (folder / f'{family}_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
