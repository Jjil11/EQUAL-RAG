"""Complete action-pair vs baseline analysis while other arms run."""
import json
from qwen_clean_labels_20261010 import ROOT,RUN
from finalize_qwen_priority_20261010 import read,paired_fast
from score_strict_units_20261010 import score

target=RUN/'completed_action_pair_comparisons.json'
if target.exists():raise FileExistsError('Frozen interim comparison exists')
official={str(r.get('id',r.get('sample_id',i))):r['answers'] for i,r in
          enumerate(json.loads((ROOT/'ALCE/data/qampari_eval_gtr_top100.json').read_text()))}
report={'status':'COMPLETE_SETS_ONLY','arm':'action_pair','settings':{},
        'warning':'Development-only interim comparison; not selected on partial questions or independent final confirmation.'}
assert (RUN/'action_pair/status.txt').read_text().strip()=='COMPLETE'
for setting in ['regression_dev','prospective_dev']:
    p=ROOT/'data/qampari_evidence_full/dev.jsonl' if setting=='regression_dev' else ROOT/'data/prospective_qampari_gtr5_v2_20261010/dev.jsonl'
    data={r['id']:r for r in read(p)}
    if setting=='regression_dev':
        for k,r in data.items():r['all_gold_answers']=official[k]
    values={};tel={}
    for a in ['action_pair','baseline','unit_ul']:
        folder=RUN/a/setting if a=='action_pair' else ROOT/'runs/submission_20261010/qwen3_frozen_comparison'/setting/a
        preds=read(folder/'eval_results.jsonl');t=read(folder/'termination_telemetry.jsonl')
        assert len(preds)==len(t)==80 and {r['id'] for r in preds}==set(data)
        tel[a]={r['id']:r for r in t};values[a]=[score(data[r['id']],r) for r in preds]
    report['settings'][setting]={b:paired_fast(values['action_pair'],values[b],tel['action_pair'],tel[b]) for b in ['baseline','unit_ul']}
target.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
