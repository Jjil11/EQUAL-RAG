"""Recompute from already downloaded lightweight snapshots; no network dependency."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import suite_common as c
from run_tokenizer_boundary_audit import audit, text_pairs

def main():
    from transformers import AutoTokenizer
    pairs = text_pairs()
    output = c.SUITE / 'analysis/cached_revalidation_v2'
    results, failures = [], []
    cache = c.SUITE / 'tokenizers/hf_cache'
    candidates = []
    for folder in sorted(cache.glob('models--*/snapshots/*')):
        repo = folder.parent.parent.name[len('models--'):].replace('--','/')
        if (folder/'tokenizer.json').exists() or (folder/'tokenizer.model').exists():
            candidates.append((repo, folder))
    local = c.PROJECT/'artifacts/equal_rag_cequal_dev_calibration_20261002/runs/cequal_dev_calibration/c1_quality_mid/adapter'
    if (local/'tokenizer.json').exists():
        candidates.append(('local:Mistral-7B-Instruct-v0.3',local))
    for name, folder in candidates:
        try:
            tok = AutoTokenizer.from_pretrained(str(folder), use_fast=True, local_files_only=True, trust_remote_code=False)
            backend = json.loads(tok.backend_tokenizer.to_str())['model']
            provenance = {'source':str(folder), 'snapshot_revision':folder.name,
                'files':{p.name:c.sha(p) for p in folder.iterdir() if p.is_file()},
                'tokenizer_class':tok.__class__.__name__, 'algorithm':backend['type'],
                'vocab_merges_sha256':hashlib.sha256(json.dumps(backend,sort_keys=True).encode()).hexdigest()}
            family = ('Mistral SentencePiece BPE' if 'Mistral' in name else
                      'Qwen byte-level BPE' if 'Qwen' in name else
                      'Llama-2/SentencePiece' if 'Phi-3' in name else name)
            results.append(audit(name,family,lambda s:tok.encode(s,add_special_tokens=False),
                lambda ids:tok.decode(ids,skip_special_tokens=False),provenance,pairs))
        except Exception as e:
            failures.append({'name':name,'error':repr(e)})
    try:
        import tiktoken
        for name in ['cl100k_base','o200k_base']:
            try:
                enc=tiktoken.get_encoding(name)
                results.append(audit(name,'tiktoken '+name,enc.encode,enc.decode,{'library':'tiktoken'},pairs))
            except Exception as e:
                failures.append({'name':name,'error':repr(e)})
    except ImportError as e:
        failures.append({'name':'tiktoken','error':repr(e)})
    assert len(results)>=6
    c.dump(output/'audit.json',{'status':'COMPLETE','results':results,'failures':failures,
        'original_audit_sha256':c.sha(c.SUITE/'analysis/tokenizer_boundary_audit.json'),
        'all_results_recomputed':True,'interpretation':'Segmentation mismatch, not a universal model deadlock proof.'})
    lines=['# S1 cached tokenizer revalidation','','| Tokenizer | ] | ], | ], space | Mismatch |','|---|---|---|---|---|']
    for r in results:
        ids=r['literal_ids']
        lines.append(f"| {r['tokenizer']} | {ids[']']} | {ids['],']} | {ids['], ']} | {r['mismatches']}/21 |")
    with c.safe(output/'TABLE_S1.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')

if __name__=='__main__':main()
