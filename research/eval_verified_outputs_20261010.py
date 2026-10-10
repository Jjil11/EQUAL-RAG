"""Keep legacy decoding unchanged, record exact IDs/EOT/cap telemetry."""
import json
from pathlib import Path
from collections import Counter
import train_trajectory_equal as legacy

original_evaluate=legacy.evaluate


def evaluate(args,model,tokenizer,rows,out_dir):
    generate=model.generate
    telemetry=[]
    stop=model.generation_config.eos_token_id
    stops=[stop] if isinstance(stop,int) else list(stop)
    for row in rows:
        text=tokenizer.apply_chat_template([{'role':'user','content':row['prompt']}],tokenize=False,
                                          add_generation_prompt=True,enable_thinking=not args.disable_thinking)
        if len(tokenizer(text)['input_ids'])>args.max_length:
            raise ValueError(f"Prompt truncation would affect {row['id']}")

    def captured(*pos,**kw):
        output=generate(*pos,**kw)
        prefix=kw['input_ids'].shape[-1]
        ids=output[0,prefix:].tolist()
        last=ids[-1] if ids else None
        telemetry.append({'id':rows[len(telemetry)]['id'],'generated_token_ids':ids,
                          'eos_ids':stops,'last_token_id':last,'ended_by_eos':last in stops,
                          'hit_cap_without_eos':len(ids)>=args.max_new_tokens and last not in stops})
        return output
    model.generate=captured
    try:
        result=original_evaluate(args,model,tokenizer,rows,out_dir)
    finally:
        model.generate=generate
    assert len(telemetry)==len(rows)
    out_dir=Path(out_dir)
    (out_dir/'termination_telemetry.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in telemetry))
    report={'n':len(rows),'configured_eos_ids':stops,
            'eos_endings':dict(Counter(str(x['last_token_id']) for x in telemetry if x['ended_by_eos'])),
            'true_cap_hit_rate':sum(x['hit_cap_without_eos'] for x in telemetry)/len(rows),
            'same_decoding_as_legacy':True,'no_prompt_truncation':True}
    (out_dir/'termination_metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    return result


legacy.evaluate=evaluate
if __name__=='__main__':legacy.main()
