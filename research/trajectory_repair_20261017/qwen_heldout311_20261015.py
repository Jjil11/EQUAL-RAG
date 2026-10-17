"""One-shot preregistered inference of twelve EXISTING systems. No training."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(str(_release_root))
RUN = ROOT/'runs/qwen_heldout311_preregistered_20261015'
SOURCE = ROOT/'runs/qwen_tail_fidelity_v3_20261015'
SEEDRUN = ROOT/'runs/qwen_seed_ablation_v2_20261015'
AONLY = ROOT/'runs/qwen_pair_aonly_20261015'
HELD = ROOT/'data/prospective_qampari_gtr5_v2_20261010/holdout.jsonl'
HELD_SHA = '0197773c65a745b316b25f58a70e3f67b5b5f43e7f71a9818aab9f19e6433097'
MODEL = Path(str(_release_model))
ANCHOR = ROOT/'runs/qwen3_14b_compression_calibration/pref_0p4/dev/adapter'
SEEDS = [20261015, 20261118, 20261229]
CONFIGS = {'ce': 0., 'a025': .25, 'a050': .50}
VIEWS = ['full311', 'multi', 'sparse']
FORMAT = 'Each answer item must use square-bracket citations immediately after that item. Syntax example only (not answers to the question): Answer A [1], Answer B [2]. Do not put all citations in a separate list at the end.\n\n'
STOPS = [151643, 151645]


def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return [json.loads(v) for v in Path(p).read_text(encoding='utf-8').splitlines() if v.strip()]
def js(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
def dump(p, obj, exclusive=False):
    p = Path(p)
    if exclusive:
        with p.open('x', encoding='utf-8') as f: json.dump(obj, f, ensure_ascii=False, indent=2)
    else:
        tmp = p.with_name(p.name+'.tmp.'+str(os.getpid()))
        with tmp.open('w', encoding='utf-8') as f: json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
def rows(p, values):
    with Path(p).open('x', encoding='utf-8') as f:
        for v in values: f.write(json.dumps(v, ensure_ascii=False)+'\n')


def systems():
    out = {'base': {'adapter': None}, 'base_block': {'adapter': None}, 'anchor': {'adapter': str(ANCHOR)}}
    for cfg, beta in CONFIGS.items():
        for seed in SEEDS:
            if seed == SEEDS[0]:
                folder = SOURCE/'base_ce' if cfg == 'ce' else AONLY/('beta025' if cfg == 'a025' else 'beta050')
            else: folder = SEEDRUN/f'{cfg}_seed{seed}'
            out[f'{cfg}_s{seed}'] = {'adapter': str(folder/'adapter'), 'training_manifest': str(folder/'manifest.json'),
                                      'config': cfg, 'beta': beta, 'training_seed': seed}
    return out


def single_pairs():
    baselines = ['base', 'base_block', 'anchor']
    out = [(baselines[j], baselines[i]) for i in range(3) for j in range(i+1, 3)]
    for cfg in CONFIGS:
        for seed in SEEDS: out.extend((f'{cfg}_s{seed}', b) for b in baselines)
    for seed in SEEDS:
        out += [(f'a025_s{seed}', f'ce_s{seed}'), (f'a050_s{seed}', f'ce_s{seed}'), (f'a050_s{seed}', f'a025_s{seed}')]
    assert len(out) == len(set(out)) == 39
    return out


def mean_pairs():
    names = ['base', 'base_block', 'anchor', 'ce', 'a025', 'a050']
    return [(names[j], names[i]) for i in range(6) for j in range(i+1, 6)]


def file_record(p):
    p = Path(p); st = p.stat()
    return {'path': str(p), 'sha256': sha(p), 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}


def prepare():
    assert not RUN.exists(), 'Never overwrite or rename a prior held-out attempt'
    RUN.mkdir(); (RUN/'code').mkdir(); (RUN/'data').mkdir(); (RUN/'predictions').mkdir(); (RUN/'logs').mkdir()
    prereg = ROOT/'code/EQUAL_HELDOUT311_PREREGISTRATION_20261015.md'
    shutil.copy2(prereg, RUN/prereg.name)
    # Load only existing functions; none of their CLI/main entry points execute.
    import qwen_equal_graph_20261014 as q
    import qwen_base_entity_abc_eval_20261015
    from audit_qwen_gemini_next_20261015 import lexical_groups
    from eval_qwen_recovery_controls_20261012 import FORMAT as original_format
    from score_strict_units_20261010 import norm
    from entity_abc_contracts_20261015 import normalize
    from transformers import AutoTokenizer
    assert FORMAT == original_format and q.MODEL == str(MODEL) and q.ANCHOR == ANCHOR
    assert js(SEEDRUN/'status.json')['state'] == 'COMPLETE'
    registry = systems()
    for key, spec in registry.items():
        if spec['adapter']:
            adapter = Path(spec['adapter'])
            spec['files'] = [file_record(p) for p in sorted(adapter.iterdir()) if p.is_file()]
            weight = next(v for v in spec['files'] if Path(v['path']).name == 'adapter_model.safetensors')
            if 'training_manifest' in spec:
                m = js(spec['training_manifest']); assert m['updates'] == 60 and m['adapter_sha256'] == weight['sha256']
                assert not m.get('merge_called', False) and not m.get('old_Anchor_loaded', False)
                assert m['data_sha256'] == '0d91e2e47fc2089183ea3cfa7e45ba9ae79e78de9fd26c95cfbc18dcbba85ceb'
                spec['manifest_sha256'] = sha(spec['training_manifest'])
                shutil.copy2(spec['training_manifest'], RUN/'data'/(key+'_training_manifest.json'))
        else: spec['files'] = []
    print(json.dumps({'state':'HASHING_FROZEN_BASE','utc':now()}), flush=True)
    base_files = [file_record(p) for p in sorted(MODEL.iterdir()) if p.is_file()]
    code_files = {Path(__file__), ROOT/'code/analyze_heldout311_20261015.py', ROOT/'code/test_heldout311_20261015.py'}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename and Path(filename).parent == ROOT/'code' and filename.endswith('.py'): code_files.add(Path(filename))
    code_hashes = {}
    for p in sorted(code_files):
        shutil.copy2(p, RUN/'code'/p.name); code_hashes[str(p)] = sha(p)
    versions = {name: importlib.metadata.version(name) for name in ['torch','transformers','peft','accelerate','numpy','safetensors']}
    protocol = {'state':'PREREGISTERED_BEFORE_HELDOUT_CONTENT_AND_GENERATION', 'utc':now(), 'internal_registration_only':True,
        'authorization':'User explicitly authorized the existing held-out311 once, with zero training or selection.',
        'preregistration_sha256':sha(RUN/prereg.name), 'heldout_source':str(HELD), 'heldout_expected_sha256':HELD_SHA,
        'systems':registry,'base_files':base_files,'code_hashes':code_hashes,'versions':versions,'seeds':SEEDS,'configs':CONFIGS,
        'views':{'full311':'all311','multi':'lexical_groups >= 2','sparse':'lexical_groups <= 1'},
        'single_pairs':single_pairs(),'mean_pairs':mean_pairs(),'bootstrap_samples':20000,'question_rng_seed':20261010,'seed_rng_seed':20261011,
        'decoding':{'do_sample':False,'num_beams':1,'max_new_tokens':256,'enable_thinking':False,'no_truncation':True,
                    'dtype':'BF16','attention':'SDPA','unmerged_LoRA':True,'only_rule_system':'base_block'},
        'no_new_training':True,'no_checkpoint_selection':True,'no_hyperparameter_search':True,'no_quality_gate':True,
        'no_quality_scores_until_all_generations_complete':True,'one_attempt_per_model_question':True,
        'frozen_format':FORMAT,'historical_raw24_not_pure_sparse':True,'multiple_comparison_adjustment':False,
        'no_exclusive_primary_configuration_or_endpoint':True}
    dump(RUN/'protocol.json', protocol, True)
    # Content is opened only AFTER the analysis plan, weights and implementation freeze.
    assert sha(HELD) == HELD_SHA
    original = read(HELD); assert len(original) == 311
    assert len({r['id'] for r in original}) == len({norm(r['question']) for r in original}) == 311
    train = read(SOURCE/'training_source350_snapshot.jsonl')
    dev160 = read(ROOT/'runs/qwen_stratified_raw_dev_20261015/known_raw_pool160.jsonl')
    dev69 = read(SOURCE/'eval69.jsonl')
    overlap = {}
    for name, source_rows in [('training350',train),('known_dev160',dev160),('recent_dev69',dev69)]:
        overlap[name] = {'id':len({r['id'] for r in original}&{r['id'] for r in source_rows}),
            'question_scorer_normalization':len({norm(r['question']) for r in original}&{norm(r['question']) for r in source_rows}),
            'question_entity_normalization':len({normalize(r['question']) for r in original}&{normalize(r['question']) for r in source_rows})}
    dump(RUN/'data/overlap_audit.json', overlap, True)
    assert all(v == 0 for record in overlap.values() for v in record.values()), 'Do not remove overlapping held-out rows to manufacture independence'
    assert all(lexical_groups(r) == r['edges'] for r in dev69), 'Scoring-label contract must match all69 frozen dev rows'
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    compiled = []; lengths = []
    for r in original:
        assert r['retrieval_setting'] == 'unaltered_gtr_top5' and len(r['docs']) == 5
        edges = lexical_groups(r); split = 'multi' if len(edges) >= 2 else 'sparse'
        prompt = FORMAT+r['prompt']
        text = tok.apply_chat_template([{'role':'user','content':prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prompt_ids = tok(text, add_special_tokens=True)['input_ids']
        assert prompt_ids == tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        lengths.append(len(prompt_ids)); assert lengths[-1] <= 2048
        compiled.append(dict(r, original_edges=r['edges'], edges=edges, prompt=prompt, eval_split=split, prompt_token_ids=prompt_ids))
    shutil.copy2(HELD, RUN/'data/original_holdout311.jsonl')
    shutil.copy2(SOURCE/'training_source350_snapshot.jsonl',RUN/'data/training350_snapshot.jsonl')
    rows(RUN/'data/eval311.jsonl', compiled)
    # Generation workers read a gold-free projection only.
    rows(RUN/'data/inference311.jsonl', [{'id':r['id'],'prompt':r['prompt'],'prompt_token_ids':r['prompt_token_ids']} for r in compiled])
    strata = {}
    for view in VIEWS:
        subset = compiled if view == 'full311' else [r for r in compiled if r['eval_split'] == view]
        rows(RUN/'data'/('ids_'+view+'.jsonl'),[{'id':r['id']} for r in subset])
        strata[view] = {'questions':len(subset),'Q_effective_questions':sum(bool(r['edges']) for r in subset),
            'supported_group_denominator':sum(len(r['edges']) for r in subset),'zero_support_questions':sum(not r['edges'] for r in subset),
            'largest_question_groups':max((len(r['edges']) for r in subset),default=0),'ids_sha256':sha(RUN/'data'/('ids_'+view+'.jsonl'))}
    preflight = {'utc':now(),'protocol_sha256':sha(RUN/'protocol.json'),'source_sha256':sha(HELD),'eval311_sha256':sha(RUN/'data/eval311.jsonl'),
        'inference311_sha256':sha(RUN/'data/inference311.jsonl'),'strata':strata,'overlaps':overlap,
        'prompt_tokens_min':min(lengths),'prompt_tokens_max':max(lengths),'all69_lexical_labels_match':True,'generations_so_far':0,
        'training350_sha256':sha(SOURCE/'training_source350_snapshot.jsonl'),'known_dev160_sha256':sha(ROOT/'runs/qwen_stratified_raw_dev_20261015/known_raw_pool160.jsonl')}
    dump(RUN/'preflight.json', preflight, True)
    tests = subprocess.run([sys.executable,'-m','unittest','test_heldout311_20261015','-v'],cwd=ROOT/'code',capture_output=True,text=True)
    dump(RUN/'unit_tests.json',{'returncode':tests.returncode,'stdout':tests.stdout,'stderr':tests.stderr},True)
    assert tests.returncode == 0
    freeze = {'utc':now(),'protocol_sha256':sha(RUN/'protocol.json'),'preflight_sha256':sha(RUN/'preflight.json'),
        'preregistration_sha256':sha(RUN/prereg.name),'unit_tests_sha256':sha(RUN/'unit_tests.json'),
        'all_pre_generation_files':{str(p.relative_to(RUN)):sha(p) for p in sorted(RUN.rglob('*')) if p.is_file()}}
    dump(RUN/'freeze.json',freeze,True)
    for p in RUN.rglob('*'):
        if p.is_file():p.chmod(0o444)
    dump(RUN/'status.json',{'state':'FROZEN_READY_NO_GENERATIONS','utc':now(),'strata':strata},True)
    print(json.dumps(preflight,ensure_ascii=False),flush=True)


def check_freeze():
    freeze = js(RUN/'freeze.json')
    for name, h in freeze['all_pre_generation_files'].items(): assert sha(RUN/name) == h, name
    protocol = js(RUN/'protocol.json')
    for path, h in protocol['code_hashes'].items(): assert sha(path) == h, path
    return protocol


def infer(which):
    import torch
    from peft import PeftModel
    from transformers import LogitsProcessorList
    import qwen_equal_graph_20261014 as q
    from qwen_base_entity_abc_eval_20261015 import EntityFirstTokenBlocker
    protocol = check_freeze(); spec = protocol['systems'][which]
    folder = RUN/'predictions'/which; folder.mkdir(exist_ok=False)
    dump(folder/'status.json',{'state':'LOADING','utc':now()},True)
    try:
        for item in spec['files']:assert sha(item['path']) == item['sha256']
        model,tok,mapping = q.load_model('base')
        if spec['adapter']:model = PeftModel.from_pretrained(model,spec['adapter'],is_trainable=False)
        model.requires_grad_(False);model.eval();model.config.use_cache=True
        layers = [m for m in model.modules() if hasattr(m,'lora_A')]
        assert (len(layers)>0) == bool(spec['adapter'])
        assert all(not m.merged and not m.disable_adapters for m in layers)
        stops = model.generation_config.eos_token_id;stops = [stops] if isinstance(stops,int) else list(stops)
        assert sorted(stops) == STOPS
        assert model.generation_config.repetition_penalty == 1 and model.generation_config.no_repeat_ngram_size == 0
        dump(folder/'manifest.json',{'system':which,'utc':now(),'protocol_sha256':sha(RUN/'protocol.json'),
            'freeze_sha256':sha(RUN/'freeze.json'),'adapter_files':spec['files'],'LoRA_layers':len(layers),'all_unmerged':True,
            'merge_called':False,'training_called':False,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'inference_rules':which=='base_block','placement':mapping,'eos_ids':stops,'decoding':protocol['decoding'],
            'generation_config':model.generation_config.to_dict(),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES')},True)
        inputs = read(RUN/'data/inference311.jsonl');assert len(inputs)==311
        started=time.time()
        with (folder/'attempts.jsonl').open('x',encoding='utf-8') as attempts, (folder/'predictions.jsonl').open('x',encoding='utf-8') as out, torch.inference_mode():
            for i,row in enumerate(inputs):
                assert set(row)=={'id','prompt','prompt_token_ids'}
                text=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                enc=tok(text,return_tensors='pt').to(model.get_input_embeddings().weight.device)
                assert enc['input_ids'][0].tolist()==row['prompt_token_ids']
                n=enc['input_ids'].shape[1];assert n<=2048
                blocker=EntityFirstTokenBlocker(tok,n)
                extra={'logits_processor':LogitsProcessorList([blocker])} if which=='base_block' else {}
                attempts.write(json.dumps({'id':row['id'],'utc':now(),'attempt':1})+'\n');attempts.flush();os.fsync(attempts.fileno())
                begin=time.time()
                ids=model.generate(**enc,do_sample=False,num_beams=1,max_new_tokens=256,pad_token_id=tok.eos_token_id,**extra)[0,n:].tolist()
                p={'id':row['id'],'tokens':len(ids),'output':tok.decode(ids,skip_special_tokens=True).strip(),
                    'prompt_token_ids':row['prompt_token_ids'],'generated_token_ids':ids,
                    'ended_by_eos':bool(ids and ids[-1] in STOPS),'hit_cap_without_eos':bool(len(ids)==256 and ids[-1] not in STOPS),
                    'blocking_events':blocker.events,'generation_seconds':time.time()-begin}
                out.write(json.dumps(p,ensure_ascii=False)+'\n');out.flush();os.fsync(out.fileno())
                dump(folder/'status.json',{'state':'GENERATING','completed':i+1,'of':311,'seconds':time.time()-started,'utc':now()})
                if (i+1)%10==0:print(json.dumps({'system':which,'completed':i+1,'of':311,'seconds':time.time()-started}),flush=True)
        assert len(read(folder/'attempts.jsonl'))==len(read(folder/'predictions.jsonl'))==311
        dump(folder/'status.json',{'state':'COMPLETE','rows':311,'seconds':time.time()-started,'utc':now(),
            'predictions_sha256':sha(folder/'predictions.jsonl'),'attempts_sha256':sha(folder/'attempts.jsonl')})
    except Exception as exc:
        dump(folder/'status.json',{'state':'FAILED_NO_AUTOMATIC_RETRY','utc':now(),'error':repr(exc)})
        raise


def pipeline():
    check_freeze(); assert js(RUN/'status.json')['state']=='FROZEN_READY_NO_GENERATIONS'
    # A lock prevents two coordinators ever starting the same generation queue.
    dump(RUN/'coordinator_lock.json',{'pid':os.getpid(),'utc':now()},True)
    pending=list(systems());workers={};completed=[];failed=[]
    while pending or workers:
        if not failed:
            for gpu in ['0,2','1,3']:
                if pending and gpu not in workers:
                    which=pending.pop(0);log=(RUN/'logs'/(which+'.log')).open('x')
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
                    proc=subprocess.Popen([sys.executable,__file__,'--infer',which],cwd=ROOT/'code',env=env,stdout=log,stderr=subprocess.STDOUT)
                    workers[gpu]=(which,proc,log)
        for gpu,(which,proc,log) in list(workers.items()):
            code=proc.poll()
            if code is not None:
                log.close();del workers[gpu]
                if code==0 and js(RUN/'predictions'/which/'status.json')['state']=='COMPLETE':completed.append(which)
                else:failed.append({'system':which,'exit_code':code})
        dump(RUN/'status.json',{'state':'INFRASTRUCTURE_FAILURE_NO_RETRY' if failed else 'GENERATING_ALL_FROZEN_SYSTEMS',
            'completed':completed,'failed':failed,'pending':pending,'running':{g:w[0] for g,w in workers.items()},'utc':now()})
        if failed and not workers:raise RuntimeError(failed)
        if pending or workers:time.sleep(10)
    assert len(completed)==12
    dump(RUN/'status.json',{'state':'ALL_GENERATIONS_COMPLETE','completed':completed,'predictions':3732,'utc':now()})
    # No quality statistics were computed until all frozen systems finished.
    from analyze_heldout311_20261015 import finish
    finish()


def status():
    out=js(RUN/'status.json')
    out['progress']={}
    for which in systems():
        p=RUN/'predictions'/which/'status.json'
        if p.exists():out['progress'][which]=js(p)
    print(json.dumps(out,ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--pipeline',action='store_true');g.add_argument('--infer',choices=list(systems()));g.add_argument('--status',action='store_true')
    a=ap.parse_args()
    if a.prepare:prepare()
    elif a.pipeline:pipeline()
    elif a.infer:infer(a.infer)
    else:status()
