"""Supplementary-only runtime; frozen scorer and legacy sources are read-only."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True
SUITE = _release_root / 'artifacts/supplementary_20261017'
PROJECT = SUITE.parents[1]
sys.path.insert(0, str(PROJECT / 'code'))
sys.path.insert(0, str(PROJECT / 'equal_rag'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
SEED = 20261015
SOURCE = PROJECT / 'runs/qwen_tail_fidelity_v3_20261015'
HELD = PROJECT / 'runs/qwen_heldout311_preregistered_20261015'
AONLY = PROJECT / 'runs/qwen_pair_aonly_20261015'
QWEN = str(_release_model)
MISTRAL = str(_release_mistral)
EXHAUSTIVE = 'Important: Exhaustively list ALL distinct entities supported by the provided passages, attach the accurate supporting passage citation [i] to each entity, do not repeat any entity name, and stop immediately once all supported entities have been listed.'
RECORD_SHA = '0d91e2e47fc2089183ea3cfa7e45ba9ae79e78de9fd26c95cfbc18dcbba85ceb'

def read(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]

def js(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def safe(path):
    p = Path(path).resolve()
    if not p.is_relative_to(SUITE):
        raise ValueError('Write outside supplementary directory: ' + str(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def dump(path, obj, replace=False):
    p = safe(path)
    data = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)
    if replace:
        tmp = p.with_name(p.name + '.tmp.' + str(os.getpid()))
        with tmp.open('x', encoding='utf-8') as f:
            f.write(data)
        os.replace(tmp, p)
    else:
        with p.open('x', encoding='utf-8') as f:
            f.write(data)

def rows(path, values):
    with safe(path).open('x', encoding='utf-8') as f:
        for row in values:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

def seed():
    import numpy as np
    import torch
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def order(records, salt=''):
    result = []
    epoch = 0
    while len(result) < 240:
        result += sorted(records, key=lambda r: hashlib.sha256((str(SEED + epoch) + salt + r['id']).encode()).hexdigest())
        epoch += 1
    return result[:240]

def order_sha(rr):
    return hashlib.sha256(json.dumps([r['id'] for r in rr]).encode()).hexdigest()

def qwen_records():
    p = SOURCE / 'compiled/records.jsonl'
    assert sha(p) == RECORD_SHA
    rr = read(p)
    assert len(rr) == 41 and sum(r['type'] == 'A' for r in rr) == 21
    assert sum(r['type'] == 'C' for r in rr) == 20
    assert order_sha(order(rr)) == js(SOURCE / 'base_ce/manifest.json')['order_sha256']
    return rr

def eval_rows():
    dev = read(SOURCE / 'eval69.jsonl')
    held = read(HELD / 'data/eval311.jsonl')
    assert len(dev) == 69 and len(held) == 311
    rr = dev + held
    assert len({r['id'] for r in rr}) == 380
    return rr

def load(backbone='qwen', train=False, adapter=None):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if backbone == 'qwen':
        import qwen_equal_graph_20261014 as q
        model, tok, mapping = q.load_model('base')
    else:
        seed()
        tok = AutoTokenizer.from_pretrained(MISTRAL, use_fast=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(MISTRAL, torch_dtype=torch.bfloat16,
                    attn_implementation='sdpa', device_map={'': 0}, local_files_only=True)
        mapping = {'': 0}
    if train:
        seed()
        model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0., bias='none',
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'], task_type='CAUSAL_LM'))
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    elif adapter:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    model.config.use_cache = not train
    return model, tok, mapping

def stops(model, tok):
    ids = model.generation_config.eos_token_id
    return [ids] if isinstance(ids, int) else list(ids or [tok.eos_token_id])

def prompt_ids(tok, prompt):
    return tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
                                  add_generation_prompt=True, enable_thinking=False)

def sequence_logp(model, ids, prompt_n):
    import torch
    import torch.nn.functional as F
    x = torch.tensor(ids, device=model.get_input_embeddings().weight.device)[None]
    logits = model(input_ids=x, use_cache=False).logits[0, prompt_n-1:-1].float()
    target = x[0, prompt_n:].to(logits.device)
    return -F.cross_entropy(logits, target, reduction='sum'), len(target)

def inference(model, tok, records, target, exhaustive=False, expected_prompt_ids=None):
    import torch
    target = safe(target)
    assert not target.exists(), 'Never overwrite predictions'
    eos = stops(model, tok)
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = True
    layers = [m for m in model.modules() if hasattr(m, 'lora_A')]
    assert all(not m.merged for m in layers)
    started = time.time()
    with target.open('x', encoding='utf-8') as f, torch.inference_mode():
        for i, row in enumerate(records):
            prompt = row['prompt'] + ('\n\n' + EXHAUSTIVE if exhaustive else '')
            inp = prompt_ids(tok, prompt)
            if expected_prompt_ids is not None:
                assert inp == expected_prompt_ids[row['id']]
            x = torch.tensor(inp, device=model.get_input_embeddings().weight.device)[None]
            begin = time.time()
            gen = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False,
                num_beams=1, max_new_tokens=256, pad_token_id=tok.eos_token_id,
                use_cache=True)[0, len(inp):].tolist()
            p = {'id':row['id'], 'tokens':len(gen), 'output':tok.decode(gen, skip_special_tokens=True).strip(),
                 'prompt_token_ids':inp, 'generated_token_ids':gen, 'ended_by_eos':bool(gen and gen[-1] in eos),
                 'hit_cap_without_eos':bool(len(gen)==256 and gen[-1] not in eos), 'blocking_events':[],
                 'generation_seconds':time.time()-begin}
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
            f.flush()
            dump(target.parent / 'generation_status.json', {'completed':i+1, 'total':len(records),
                 'seconds':time.time()-started}, replace=True)
            if (i+1) % 10 == 0:
                print(json.dumps({'phase':'generate', 'completed':i+1, 'total':len(records),
                                  'seconds':time.time()-started}), flush=True)
    assert len(read(target)) == len(records)
    dump(target.parent / 'generation_complete.json', {'rows':len(records), 'sha256':sha(target),
         'seconds':time.time()-started, 'unmerged':True, 'eos_ids':eos})

def fidelity(model, native, target):
    import torch
    model.eval()
    out = []
    with torch.inference_mode():
        for r in native:
            pos = r['positions']
            if not pos:
                out.append({'id':r['id'],'positions':0,'changed':0,'exact':True})
                continue
            # Only retained prefix tokens are needed, reducing unused suffix work.
            ids = r['actual_ids'][:max(pos)+1]
            x = torch.tensor(ids, device=model.get_input_embeddings().weight.device)[None]
            top = model(input_ids=x, use_cache=False).logits[0, pos].argmax(-1).tolist()
            base_top = r['base_top1']
            changed = sum(a != b for a,b in zip(top, base_top))
            out.append({'id':r['id'], 'positions':len(pos), 'changed':changed, 'exact':changed==0})
    rows(target / 'trajectory_agreement.jsonl', out)
    result = {'trajectories':len(out), 'exact':sum(r['exact'] for r in out),
              'positions':sum(r['positions'] for r in out), 'changed':sum(r['changed'] for r in out)}
    result['top1_fidelity'] = 1-result['changed']/max(1,result['positions'])
    dump(target / 'retention_summary.json', result)
    return result

def qwen_native():
    import torch
    rr = read(SOURCE / 'compiled/native_retention.jsonl')
    assert len(rr) == 120
    for r in rr:
        ref = torch.load(SOURCE / 'reference' / (r['id']+'.pt'), map_location='cpu', weights_only=True)
        r['base_top1'] = ref.argmax(-1).tolist()
    return rr
