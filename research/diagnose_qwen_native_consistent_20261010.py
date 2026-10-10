"""Teacher-forced training-site diagnostic; not independent generation evidence."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import gc,hashlib,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM
from compile_qwen_native_consistent_20261010 import ROOT,RUN
OUT=ROOT/'runs/qwen_native_consistent_diagnostic_20261010'

@torch.inference_mode()
def main():
    if (OUT/'summary.json').exists():raise FileExistsError('Refusing diagnostic overwrite')
    source=RUN/'compiled/train.jsonl';rows=[json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    rows=[r for r in rows if r['has_trajectory']];assert len(rows)==150
    summary={'status':'COMPLETE_TRAINING_SITE_DIAGNOSTIC','n_training_rows':150,'model_forwards':1200,
        'development_or_holdout_rows_read':0,'compiled_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'warnings':['Teacher-forced training sites, not held-out rollout success or independent statistical evidence.',
                    '72 original train stop prefixes, 78 teacher-forced completions; no independent generalization evidence.',
                    'Probability/class mass alone does not guarantee greedy termination. Native first and terminal share 72 contexts: 288 redundant forwards across four models.'],'arms':{}}
    for arm in ['baseline','action_pair','action_ul','action_ce']:
        if arm!='baseline':
            while True:
                state=(RUN/arm/'status.txt').read_text().strip()
                if state=='FAILED':raise RuntimeError(arm+' failed')
                if state.startswith('EVALUATING_') or state=='COMPLETE':break
                print('WAIT_ADAPTER',arm,state,flush=True);time.sleep(15)
        target=OUT/(arm+'.jsonl')
        if target.exists():raise FileExistsError('Refusing arm diagnostic overwrite')
        assert torch.cuda.mem_get_info()[0]/2**30>32
        adapter=ROOT/'runs/qwen3_14b_compression_calibration/pref_0p4/dev/adapter' if arm=='baseline' else RUN/arm/'interp05/adapter'
        model=PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(str(_release_model),torch_dtype=torch.bfloat16),str(adapter)).cuda().eval()
        model.config.use_cache=False;values=[]
        with target.open('w') as handle:
            for row in rows:
                record={'id':row['id'],'target_action':row['completion_pair']['action'],'native_stop':row['terminal_decision']['origin']=='recorded_onpolicy_prefix'}
                for label,key in [('first','action_decision'),('terminal','terminal_decision')]:
                    d=row[key];ids=torch.tensor(d['context_ids'],device='cuda')[None]
                    logits=model(input_ids=ids,use_cache=False).logits[0,-1].float();lp=F.log_softmax(logits,-1)
                    allowed=d['allowed_ids'];best=int(logits.argmax());legal=float(logits[allowed].max())
                    wrong=logits.clone();wrong[allowed]=-torch.inf;best_wrong=float(wrong.max())
                    record[label]={'legal_argmax':best in allowed,'allowed_probability':float(lp[allowed].exp().sum()),
                        'best_legal_minus_best_illegal_logit':legal-best_wrong,'argmax_id':best}
                    del ids,logits,lp,wrong
                values.append(record);handle.write(json.dumps(record)+'\n');handle.flush()
                if len(values)%30==0:print('DIAGNOSTIC',arm,len(values),150,flush=True)
        summary['arms'][arm]={'n':150,'first_legal_argmax_count':sum(r['first']['legal_argmax'] for r in values),
            'terminal_eos_argmax_count':sum(r['terminal']['legal_argmax'] for r in values),
            'by_target_action':{a:{'n':sum(r['target_action']==a for r in values),
                'first_legal_argmax_count':sum(r['first']['legal_argmax'] for r in values if r['target_action']==a),
                'terminal_eos_argmax_count':sum(r['terminal']['legal_argmax'] for r in values if r['target_action']==a)}
                for a in ['stop','complete_then_stop']},
            'native_stop':{'n':72,'eos_argmax_count':sum(r['terminal']['legal_argmax'] for r in values if r['native_stop']),
                'mean_eos_probability':sum(r['terminal']['allowed_probability'] for r in values if r['native_stop'])/72},
            'mean_terminal_eos_probability':sum(r['terminal']['allowed_probability'] for r in values)/150,
            'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
        print(json.dumps({'arm':arm,**summary['arms'][arm]}),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
