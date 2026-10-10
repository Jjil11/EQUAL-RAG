"""Train-only audit of recorded-token versus retokenized terminal contexts."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import hashlib,json
from collections import Counter
from transformers import AutoTokenizer
from compile_qwen_terminal_bridge_20261010 import ROOT,RUN
from completion_pairs_20261010 import units,norm,covered,true_repeat
from score_strict_units_20261010 import score
OUT=ROOT/'runs/qwen_terminal_bridge_reachability_20261010'

def main():
    if OUT.exists():raise FileExistsError('Refusing audit overwrite')
    source=RUN/'compiled/train.jsonl';rows=[json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    tok=AutoTokenizer.from_pretrained(str(_release_model),use_fast=True);records=[]
    for row in rows:
        if not row['has_trajectory'] or row['completion_pair']['action']!='stop':continue
        text=row['short'];seen=set();boundary=None
        for lo,hi,unit in units(text):
            v=norm(unit)
            if v in seen:boundary=lo;break
            seen.add(v)
        assert boundary is not None
        native_text=text[:boundary].rstrip();ids=row['generation_token_ids']
        assert tok.decode(ids,skip_special_tokens=True).strip()==text
        matches=[k for k in range(1,len(ids)+1) if tok.decode(ids[:k],skip_special_tokens=True).strip()==native_text]
        reliable=len(covered(native_text,row['edges']))==len(row['edges']) and not true_repeat({'short':native_text})
        rec={'id':row['id'],'original_prefix_already_covers_training_edges':reliable,
            'exact_recorded_token_boundary_found':bool(matches)}
        if matches:
            k=max(matches)
            prompt=tok.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
            native=prompt+ids[:k];canonical=row['terminal_decision']['context_ids']
            rec.update(native_last_token=ids[k-1],native_last_token_text=tok.decode([ids[k-1]]),
                native_context_equals_current_terminal_context=native==canonical,
                native_context_is_recorded_onpolicy_prefix=True)
            a=score(row,{'id':row['id'],'tokens':1,'output':native_text,'grounded_edges':0})
            b=score(row,{'id':row['id'],'tokens':1,'output':native_text.rstrip(',').rstrip(),'grounded_edges':0})
            rec['trailing_comma_does_not_change_current_metrics']=all(a[x]==b[x] for x in ['grounded_edges','answer_f1','repeated_units'])
        records.append(rec)
    assert len(records)==75
    valid=[r for r in records if r['original_prefix_already_covers_training_edges'] and r['exact_recorded_token_boundary_found']]
    report={'status':'COMPLETE_TRAIN_ONLY_REACHABILITY_AUDIT','stop_targets':75,'reliable_native_stop_sites':len(valid),
        'native_context_differs_from_current_terminal':sum(not r['native_context_equals_current_terminal_context'] for r in valid),
        'native_last_token_histogram':dict(Counter(r['native_last_token_text'] for r in valid)),
        'trailing_comma_metric_invariance_all_valid':all(r['trailing_comma_does_not_change_current_metrics'] for r in valid),
        'compiled_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'model_forwards':0,'model_generations':0,
        'development_or_holdout_rows_read':0,'warnings':['Training-only syntax and support proxy audit, not a model improvement.',
            'Known lexical coverage does not certify all world facts or semantic support.',
            'Native-context training is not implemented or evaluated by this audit.'],'per_row':records}
    OUT.mkdir();(OUT/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='per_row'},indent=2),flush=True)

if __name__=='__main__':main()
