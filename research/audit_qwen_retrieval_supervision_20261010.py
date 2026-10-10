"""Train-only inspection of the actual supervised action, not rollout success."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from build_qwen_retrieval_align_20261010 import ROOT,RUN

OUT=ROOT/'runs/qwen_retrieval_supervision_audit_20261010'
def main():
    if OUT.exists():raise FileExistsError('Refusing audit overwrite')
    path=RUN/'compiled/train.jsonl';rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    assert len(rows)==350
    active=[r for r in rows if r['has_trajectory']];stops={151645,151643};records=[]
    for r in active:
        d=r['action_decision'];assert d['rejected_id'] not in d['allowed_ids']
        records.append({'id':r['id'],'target_action':r['completion_pair']['action'],
            'decision_allows_eos':bool(stops.intersection(d['allowed_ids'])),
            'allowed_token_count':len(d['allowed_ids']),'candidate_count':r['candidate_count'],
            'steps_after_shared':d['steps_after_shared'],
            'decision_context_tokens':len(d['context_ids'])})
    report={'status':'COMPLETE_TRAIN_ONLY_SUPERVISION_AUDIT','rows':350,'active_actions':len(active),
        'compiled_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'model_forwards':0,'development_or_holdout_rows_read':0,
        'target_actions':dict(Counter(r['target_action'] for r in records)),
        'decision_allows_eos':sum(r['decision_allows_eos'] for r in records),
        'eos_allowed_by_target_action':{a:sum(r['decision_allows_eos'] for r in records if r['target_action']==a)
            for a in ['stop','complete_then_stop']},
        'allowed_token_count_histogram':dict(Counter(r['allowed_token_count'] for r in records)),
        'steps_after_shared_histogram':dict(Counter(r['steps_after_shared'] for r in records)),
        'warnings':['A completed repair ending in EOS does not imply the supervised earliest action is EOS.',
                    'The current implementation has one supervised action per active row, not all completion tokens.',
                    'These are training labels, not held-out quality or generation success.'],'per_row':records}
    OUT.mkdir();(OUT/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='per_row'},indent=2),flush=True)

if __name__=='__main__':main()
