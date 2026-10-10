"""Seven-adapter, train-state action diagnostic. No weights updated."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from qwen_clean_labels_20261010 import ROOT,RUN


def main():
    target=RUN/'action_diagnostic_summary.json'
    if target.exists():raise FileExistsError('Existing diagnostic')
    rows=sorted([json.loads(x) for x in (RUN/'compiled/train.jsonl').read_text().splitlines()
                 if json.loads(x).get('has_trajectory')],key=lambda r:r['id'])
    folders={'baseline':ROOT/'runs/qwen3_14b_compression_calibration/pref_0p4/dev/adapter'}
    folders.update({a:RUN/a/'interp05/adapter' for a in ['seq_pair','seq_sft','action_pair','action_ce']})
    folders.update({a:ROOT/'runs/qwen_clean_controls_20261010'/a/'interp05/adapter' for a in ['action_ul','strong_ce']})
    assert len(rows)==39 and all((p/'adapter_config.json').exists() for p in folders.values())
    model=AutoModelForCausalLM.from_pretrained(str(_release_model),torch_dtype=torch.bfloat16)
    model=PeftModel.from_pretrained(model,str(folders['baseline']),adapter_name='baseline')
    for a,p in folders.items():
        if a!='baseline':model.load_adapter(str(p),adapter_name=a,is_trainable=False)
    model=model.cuda();model.config.use_cache=False;records=[];summary={}
    with torch.inference_mode():
        for arm in folders:
            model.set_adapter(arm);model.eval();values=[]
            for row in rows:
                d=row['action_decision'];ids=torch.tensor(d['context_ids'],device='cuda')[None]
                logits=model(input_ids=ids,use_cache=False).logits[0,-1].float()
                lp=torch.log_softmax(logits,0);other=logits.clone();other[d['allowed_ids']]=-torch.inf
                legal=logits[d['allowed_ids']].max();greedy=int(logits.argmax())
                r={'id':row['id'],'arm':arm,'action':row['completion_pair']['action'],
                   'greedy_allowed':greedy in d['allowed_ids'],'greedy_rejected':greedy==d['rejected_id'],
                   'legal_probability':float(torch.logsumexp(lp[d['allowed_ids']],0).exp()),
                   'legal_minus_rejected':float(legal-logits[d['rejected_id']]),
                   'legal_minus_all_other':float(legal-other.max())}
                records.append(r);values.append(r)
            summary[arm]={'n':39,'greedy_legal_wins':sum(r['greedy_allowed'] for r in values),
                'greedy_original_rejected':sum(r['greedy_rejected'] for r in values),
                'mean_legal_probability':sum(r['legal_probability'] for r in values)/39}
            print(json.dumps({'arm':arm,**summary[arm]}),flush=True)
    (RUN/'action_diagnostic_states.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    target.write_text(json.dumps({'status':'COMPLETE','means':summary,'teacher_forced_calls':273,
        'heldout_inputs':0,'weights_updated':False,'warning':'Training prefixes; not independent rollout success. Four citation forks differ from the first round, so counts must be compared within this round.'},indent=2))


if __name__=='__main__':main()
