"""Finalize only after all eight fixed evaluations; never opens sealed holdout."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import hashlib
import json
import time
from pathlib import Path
from score_strict_units_20261010 import score,paired

ROOT=Path(str(_release_root))
ROUND=ROOT/'runs/submission_20261010'
ARMS={'qwen3':['baseline','semantic_eos','unit_ul','wrong_boundary','kl_only'],
      'mistral':['baseline','semantic_eos','unit_ul']}


def read(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    while True:
        states={f'{family}_{arm}':(ROUND/f'{family}_frozen_comparison/{arm}_status.txt')
                for family,arms in ARMS.items() for arm in arms}
        loaded={k:p.read_text().strip() if p.exists() else 'QUEUED' for k,p in states.items()}
        if any(s=='FAILED' for s in loaded.values()):
            (ROUND/'finalization_status.json').write_text(json.dumps({'status':'FAILED','evaluations':loaded},indent=2))
            raise RuntimeError('One fixed evaluation failed; do not silently drop it')
        if all(s=='COMPLETE' for s in loaded.values()):break
        time.sleep(15)
    report={'status':'DEVELOPMENT_ROUND_COMPLETE','seed':20260930,
            'holdout_status':'311 prospective rows sealed: no inference',
            'legacy_test_status':'No new legacy test inference; previously exposed regression collection',
            'families':{},'human_annotations_completed':0}
    for family,arms in ARMS.items():
        folder=ROUND/f'{family}_frozen_comparison'
        report['families'][family]={}
        for setting in ['regression_dev','prospective_dev']:
            path=ROOT/'data/qampari_evidence_full/dev.jsonl' if setting=='regression_dev' else ROOT/'data/prospective_qampari_gtr5_v2_20261010/dev.jsonl'
            data={x['id']:x for x in read(path)}
            runs={arm:[score(data[x['id']],x) for x in read(folder/setting/arm/'eval_results.jsonl')] for arm in arms}
            records={}
            for arm in arms:
                if arm=='baseline':continue
                item=paired(runs[arm],runs['baseline'])
                item['termination']=json.loads((folder/setting/arm/'termination_metrics.json').read_text())
                item['baseline_termination']=json.loads((folder/setting/'baseline/termination_metrics.json').read_text())
                (folder/setting/f'{arm}_paired_strict.json').write_text(json.dumps(item,indent=2))
                records[arm]=item
            comparison=paired(runs['semantic_eos'],runs['unit_ul'])
            (folder/setting/'semantic_eos_vs_ul.json').write_text(json.dumps(comparison,indent=2))
            records['semantic_eos_vs_ul']=comparison
            report['families'][family][setting]=records
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in ROUND.rglob('*.json') if 'adapter' not in p.parts}
    (ROUND/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2))
    (ROUND/'development_round_summary.json').write_text(json.dumps(report,indent=2))
    (ROUND/'finalization_status.json').write_text(json.dumps({'status':'COMPLETE','evaluations':loaded},indent=2))
    print('Development round finalized; sealed holdout untouched',flush=True)


if __name__=='__main__':main()
