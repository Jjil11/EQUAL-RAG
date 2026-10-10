"""Read-only template, split-exposure, and CUDA audit; never evaluate holdouts."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, GenerationConfig

ROOT = Path(str(_release_root))
sys.path.insert(0, str(ROOT / 'code'))
import train_trajectory_equal as trainer


def norm(text):
    return ' '.join(str(text).lower().split())


def main():
    report = {'cuda': [], 'templates': {}, 'exposure': {}}
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            free, total = torch.cuda.mem_get_info()
            report['cuda'].append({'device': i, 'name': torch.cuda.get_device_name(i),
                                   'free_gib': free / 2**30, 'total_gib': total / 2**30})
    families = {
        'mistral': '/common_data/model/Mistral-7B-Instruct-v0.3',
        'qwen3': str(_release_model),
    }
    for family, model in families.items():
        if not Path(model).exists():
            report['templates'][family] = {'missing': model}
            continue
        tok = AutoTokenizer.from_pretrained(model)
        tok._equal_rag_disable_thinking = family == 'qwen3'
        cfg = GenerationConfig.from_pretrained(model)
        eos = cfg.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        entries = []
        for response in ['Paris [1].', 'Paris [1], London [2]']:
            ids, mask, labels = trainer.encode_example(tok, 'Give a cited list.', response, 2048)
            pos = labels.ne(-100).nonzero().flatten()
            end = int(pos[-1])
            entries.append({'response': response, 'legacy_boundary_id': int(ids[end]),
                            'legacy_boundary_is_generation_eos': int(ids[end]) in eos,
                            'suffix_ids': ids[-12:].tolist(),
                            'suffix_tokens': tok.convert_ids_to_tokens(ids[-12:].tolist()),
                            'suffix_text': tok.decode(ids[-12:])})
        report['templates'][family] = {'model': model, 'eos_ids': eos,
                                      'tokenizer_eos_id': tok.eos_token_id, 'examples': entries}
    seen_ids, seen_questions = set(), set()
    data_files = []
    for path in sorted((ROOT / 'data').rglob('*.jsonl')):
        if 'qampari' not in str(path).lower():
            continue
        n = 0
        try:
            for line in path.open():
                if not line.strip():
                    continue
                row = json.loads(line)
                seen_ids.add(str(row.get('id', '')))
                if row.get('question'):
                    seen_questions.add(norm(row['question']))
                n += 1
            data_files.append({'path': str(path), 'rows': n})
        except Exception as exc:
            data_files.append({'path': str(path), 'error': str(exc)})
    result_ids = set()
    for path in (ROOT / 'runs').rglob('eval_results.jsonl'):
        try:
            for line in path.open():
                row = json.loads(line)
                result_ids.add(str(row.get('id', '')))
        except Exception:
            pass
    seen_ids |= result_ids
    sources = []
    for path in sorted((ROOT / 'ALCE/data').glob('qampari*.json')):
        raw = json.loads(path.read_text())
        if isinstance(raw, dict):
            raw = raw.get('data', [])
        remaining = [r for i, r in enumerate(raw)
                     if str(r.get('id', r.get('sample_id', i))) not in seen_ids
                     and norm(r.get('question', '')) not in seen_questions]
        sources.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'rows': len(raw), 'unexposed_candidates': len(remaining)})
    report['exposure'] = {'unique_seen_ids': len(seen_ids),
                          'unique_seen_questions': len(seen_questions),
                          'data_files': data_files, 'sources': sources,
                          'conservative_rule': 'Exclude all locally materialized QAMPARI rows, plus all result IDs.'}
    out = ROOT / 'runs/submission_20261010'
    out.mkdir(exist_ok=True)
    (out / 'audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
