"""User-authorized unconditional evaluation of the frozen 60-step adapters.

Historical failed training gates remain immutable. No metric-based exclusion.
"""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import qwen_tail_fidelity_20261015 as c

q = c.q
SOURCE = q.ROOT / 'runs/qwen_tail_fidelity_v3_20261015'
ROOT = q.ROOT / 'runs/qwen_tail_open_suite_20261015'
RUN = ROOT / 'stage1'
c.RUN = c.ev.RUN = RUN


def link_tree(src, dest):
    dest.mkdir(parents=True, exist_ok=False)
    for p in src.iterdir():
        if p.is_file():
            try:
                os.link(p, dest / p.name)
            except OSError:
                shutil.copy2(p, dest / p.name)


def prepare():
    assert not ROOT.exists(), 'Refusing to overwrite a previous experiment'
    RUN.mkdir(parents=True)
    for name in ['eval69.jsonl', 'training_source350_snapshot.jsonl']:
        shutil.copy2(SOURCE / name, RUN / name)
    for name in ['cached_eval', 'eval/base_block']:
        shutil.copytree(SOURCE / name, RUN / name)
    lineage = {}
    for arm in ['base_ce', 'base_pair']:
        manifest = json.loads((SOURCE / arm / 'manifest.json').read_text())
        assert manifest['updates'] == 60
        assert not manifest['old_Anchor_loaded']
        link_tree(SOURCE / arm / 'adapter', RUN / arm / 'adapter')
        shutil.copy2(SOURCE / arm / 'manifest.json', RUN / arm / 'source_manifest.json')
        lineage[arm] = {'source': str(SOURCE / arm / 'adapter'),
            'sha256': q.sha(RUN / arm / 'adapter/adapter_model.safetensors'),
            'original_training_gate': json.loads((SOURCE / arm / 'gate_latest.json').read_text())}
    protocol = {'user_authorized_no_training_gate': True, 'no_early_stop_or_metric_filter': True,
        'source': str(SOURCE), 'lineage': lineage, 'eval69_sha256': q.sha(RUN / 'eval69.jsonl'),
        'main_models_decoding_rules': False, 'sealed311_reads': 0, 'script_sha256': q.sha(__file__)}
    q.dump(ROOT / 'protocol.json', protocol)
    q.dump(ROOT / 'status.json', {'state': 'STAGE1_FREE_GENERATION'})


def launch(args, label, devices):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices, OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
    log = (q.ROOT / 'logs' / ('qwen_tail_open_' + label + '_20261015.log')).open('x')
    child = subprocess.Popen([sys.executable, __file__, *args], cwd=q.ROOT / 'code', env=env,
        stdout=log, stderr=subprocess.STDOUT)
    return child, log


def pipeline():
    prepare()
    workers = [launch(['--model', arm], arm + '_60_eval', gpu) for arm, gpu in [('base_ce', '0,2'), ('base_pair', '1,3')]]
    codes = []
    for child, log in workers:
        codes.append(child.wait())
        log.close()
    assert codes == [0, 0], codes
    q.dump(ROOT / 'status.json', {'state': 'STAGE1_GENERATIONS_COMPLETE', 'new_predictions': 138,
        'training_gate_used': False, 'sealed311_reads': 0})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pipeline', action='store_true')
    ap.add_argument('--model', choices=['base_ce', 'base_pair'])
    args = ap.parse_args()
    if args.pipeline:
        pipeline()
    elif args.model:
        c.ev.generate(args.model)
