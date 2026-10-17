"""Supplement-only paths and immutable-source contracts; no GPU imports."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import hashlib
import json
import os
from pathlib import Path
import sys

SUITE = _release_root / 'artifacts/supplementary_20261017'
PROJECT = SUITE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(SUITE / 'dependencies'))
sys.path.insert(0, str(PROJECT / 'equal_rag'))
sys.path.insert(0, str(PROJECT / 'code'))
os.environ.setdefault('HF_HOME', str(SUITE / 'tokenizers' / 'hf_home'))
os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(SUITE / 'tokenizers' / 'tiktoken_cache'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '12')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '30')
SEED = 20261015
BOOT_SEED = 20261010
RECORD_SHA = '0d91e2e47fc2089183ea3cfa7e45ba9ae79e78de9fd26c95cfbc18dcbba85ceb'
FROZEN = PROJECT / 'artifacts/qwen_pair_aonly_20261015/verified/run'
HELD = PROJECT / 'artifacts/qwen_heldout311_20261015/verified/run'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]

def safe(path):
    p = Path(path).resolve()
    if not p.is_relative_to(SUITE):
        raise ValueError('Write outside supplementary directory: ' + str(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def dump(path, obj):
    with safe(path).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)

def rows(path, rr):
    with safe(path).open('x', encoding='utf-8') as f:
        for r in rr:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')

def records():
    p = FROZEN / 'compiled/records.jsonl'
    assert sha(p) == RECORD_SHA
    rr = read(p)
    assert len(rr) == 41 and sum(r['type'] == 'A' for r in rr) == 21
    assert sum(r['type'] == 'C' for r in rr) == 20
    return rr

def schedule(rr, count=240):
    order = []
    epoch = 0
    while len(order) < count:
        order += sorted(rr, key=lambda r: hashlib.sha256((str(SEED + epoch) + r['id']).encode()).hexdigest())
        epoch += 1
    return order[:count]

def protect_snapshot():
    paths = [PROJECT / 'EQUAL_HELDOUT311_FINAL_RESULTS_20261015.md']
    for folder in [FROZEN, HELD, PROJECT / 'artifacts/qwen_seed_ablation_20261015/verified/run']:
        paths += [p for p in folder.rglob('*') if p.is_file()]
    return {str(p.relative_to(PROJECT)): {'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in sorted(set(paths))}

def verify_protected(before):
    for name, info in before.items():
        assert sha(PROJECT / name) == info['sha256'], 'Frozen source changed: ' + name
    return {'status': 'PASS', 'unchanged_frozen_files': len(before)}
