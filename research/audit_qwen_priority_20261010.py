"""Read-only in-sample legal-action greedy diagnostic; no test generation."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from qwen_priority_compile_20261010 import ROOT,RUN


def main():
    rows=sorted([json.loads(x) for x in (RUN/'compiled/train.jsonl').read_text().splitlines()
                 if json.loads(x).get('has_trajectory')],key=lambda r:r['id'])
    arms=['baseline','seq_full','seq_local','action_pair','action_ce']
    target=RUN/'action_diagnostic_summary.json'
    if target.exists():raise FileExistsError('Existing diagnostic')
    model=AutoModelForCausalLM.from_pretrained(str(_release_model),torch_dtype=torch.bfloat16)
    anchor=ROOT/'runs/qwen3_14b_compression_calibration/pref_0p4/dev/adapter'
    model=PeftModel.from_pretrained(model,str(anchor),adapter_name='baseline')
    for arm in arms[1:]:
        model.load_adapter(str(RUN/arm/'interp05/adapter'),adapter_name=arm,is_trainable=False)
    model=model.cuda();model.config.use_cache=False
    records=[];summary={}
    with torch.inference_mode():
        for arm in arms:
            model.set_adapter(arm);model.eval();values=[]
            for row in rows:
                d=row['action_decision'];ids=torch.tensor(d['context_ids'],device='cuda')[None]
                logits=model(input_ids=ids,use_cache=False).logits[0,-1].float()
                lp=torch.log_softmax(logits,0);other=logits.clone();other[d['allowed_ids']]=-torch.inf
                legal=logits[d['allowed_ids']].max()
                greedy=int(logits.argmax());record={'id':row['id'],'arm':arm,
                    'action':row['completion_pair']['action'],'allowed_count':len(d['allowed_ids']),
                    'legal_probability':float(torch.logsumexp(lp[d['allowed_ids']],0).exp()),
                    'legal_minus_original_rejected_margin':float(legal-logits[d['rejected_id']]),
                    'legal_minus_all_other_margin':float(legal-other.max()),
                    'greedy_allowed':greedy in d['allowed_ids'],
                    'greedy_original_rejected':greedy==d['rejected_id']}
                records.append(record);values.append(record)
            summary[arm]={'n':len(values),'greedy_legal_wins':sum(v['greedy_allowed'] for v in values),
                'greedy_original_rejected':sum(v['greedy_original_rejected'] for v in values),
                'margin1_satisfied':sum(v['legal_minus_original_rejected_margin']>=1 for v in values),
                'mean_legal_probability':sum(v['legal_probability'] for v in values)/len(values)}
            print(json.dumps({'arm':arm,**summary[arm]}),flush=True)
    (RUN/'action_diagnostic_states.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in records))
    target.write_text(json.dumps({'status':'COMPLETE','means':summary,'training_states':39,
        'teacher_forced_forward_calls':195,'weights_updated':False,'heldout_inputs':0,
        'warning':'In-sample rejected-prefix decision diagnostic, not rollout success or independent generalization.'},indent=2))


if __name__=='__main__':main()
