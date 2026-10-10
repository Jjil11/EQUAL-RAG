"""Fixed train IDs with rank-only top5; training labels never select documents."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import hashlib
import json
from pathlib import Path
from build_qampari_pilot import clean,prompt_for
from audit_qwen_training_retrieval_v2_20261010 import edges_for,norm
from completion_pairs_20261010 import valid_training_edges

ROOT=Path(str(_release_root))
RUN=ROOT/'runs/qwen_retrieval_align_20261010'
CACHE=Path(str(_release_cache / 'qwen_retrieval_align_20261010'))


def convert(item,ident):
    docs=item['docs'][:5]
    edges=valid_training_edges(edges_for(item))
    return {'id':ident,'question':clean(item['question']),'prompt':prompt_for(item['question'],docs),
            'docs':[{'id':i,'title':clean(d.get('title','')),'text':clean(d.get('text',''))}
                    for i,d in enumerate(docs,1)],'edges':edges,'all_gold_answers':item['answers'],
            'gold_short':', '.join(f"{e['answer']} [{e['doc_id']}]" for e in edges)+'.',
            'retrieval_setting':'unaltered_gtr_top5','zero_lexical_support':not edges}


def main():
    from transformers import AutoTokenizer
    source=ROOT/'ALCE/data/qampari_eval_gtr_top100.json'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='42b8ae06faae183d55be6768011c456a927daddb36def447bc92a66a9cfa38ac'
    old=ROOT/'runs/qwen_clean_labels_20261010/compiled/train.jsonl'
    assert hashlib.sha256(old.read_bytes()).hexdigest()=='e2a58bd60eb1655e43f7be220da61785cf9998b897592a199f7aedbbebb66a0e'
    rows=[json.loads(x) for x in old.read_text().splitlines() if x.strip()]
    ids={r['id'] for r in rows};assert len(ids)==len(rows)==350
    raw={str(r.get('id',r.get('sample_id',i))):r for i,r in enumerate(json.loads(source.read_text()))
         if str(r.get('id',r.get('sample_id',i))) in ids}
    assert set(raw)==ids
    for path in [ROOT/'data/qampari_evidence_full/dev.jsonl',ROOT/'data/prospective_qampari_gtr5_v2_20261010/dev.jsonl']:
        dev=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        assert not ids.intersection(r['id'] for r in dev)
        assert not {norm(r['question']) for r in rows}.intersection(norm(r['question']) for r in dev)
    built=[]
    for row in rows:
        item=raw[row['id']];assert norm(row['question'])==norm(item['question'])
        built.append(convert(item,row['id']))
    tok=AutoTokenizer.from_pretrained(str(_release_model),use_fast=True)
    lengths=[len(tok(tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,
                  add_generation_prompt=True,enable_thinking=False))['input_ids']) for r in built]
    assert max(lengths)+260<=2048,'Need uniformly larger context; do not filter long rows'
    assert sum(not r['edges'] for r in built)==106 and sum(len(r['edges']) for r in built)==682
    if RUN.exists() or CACHE.exists():raise FileExistsError('Refusing existing round/cache')
    (RUN/'data').mkdir(parents=True);(RUN/'mining').mkdir();CACHE.mkdir(parents=True)
    target=RUN/'data/train_top5.jsonl'
    target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in built))
    manifest={'rows':350,'zero_lexical_support':106,'lexical_edges':682,'top_k':5,
              'rank_only_document_selection':True,'gold_used_only_for_training_annotation':True,
              'same_350_train_ids':True,'dev_id_and_normalized_question_overlap':0,
              'sealed_holdout_rows_analyzed':0,'max_prompt_tokens':max(lengths),
              'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'previous_clean_train_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),
              'data_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
              'warning':'Internal split of ALCE evaluation source, not official benchmark; lexical support is not entailment.'}
    (RUN/'data/manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
