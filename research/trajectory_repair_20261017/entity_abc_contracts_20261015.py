"""Pure entity-set contracts. Lexical support is not semantic entailment."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import hashlib
import re
import string


def normalize(text):
    text=re.sub(r'\[\d+\]','',text)
    text=''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())


def inventory(row):
    groups=[]
    for aliases in row['all_gold_answers']:
        values={normalize(a) for a in aliases if normalize(a)}
        if not values:continue
        hits=[i for i,g in enumerate(groups) if values&g['values']]
        if not hits:groups.append({'values':values,'aliases':list(aliases)})
        else:
            first=hits[0];groups[first]['values'].update(values);groups[first]['aliases']+=list(aliases)
            for i in reversed(hits[1:]):
                groups[first]['values'].update(groups[i]['values']);groups[first]['aliases']+=groups[i]['aliases'];groups.pop(i)
    result=[]
    for g in groups:
        key=hashlib.sha256('\n'.join(sorted(g['values'])).encode()).hexdigest()[:16]
        supporting=[];surfaces=[]
        for d in row['docs']:
            text=normalize(d['title']+' '+d['text'])
            matched=[a for a in g['aliases'] if re.search(r'(?:^| )'+re.escape(normalize(a))+r'(?: |$)',text)]
            if matched:
                supporting.append(d['id']);surfaces.extend((a,d['id']) for a in matched)
        result.append({'key':key,'values':sorted(g['values']),'aliases':list(dict.fromkeys(g['aliases'])),
            'supporting_doc_ids':sorted(supporting),'matched_surfaces':list(dict.fromkeys(surfaces))})
    return result


def parse_closed(text):
    """Conservative closed units plus exact character coordinates; keep tail."""
    units=[];pos=0
    for m in re.finditer(r'(.+?)(\[\d+\](?:\s*\[\d+\])*)',text,re.S):
        if m.start()!=pos:return None
        raw=m.group(1)
        if units:
            sep=re.match(r'^\s*,\s*',raw)
            if not sep:return None
            name=raw[sep.end():].strip();name_start=m.start()+sep.end()
        else:
            name=raw.strip();name_start=m.start()+len(raw)-len(raw.lstrip())
        if not name or any(x in name for x in ['[',']','\n']):return None
        refs=sorted({int(x) for x in re.findall(r'\[(\d+)\]',m.group(2))})
        units.append({'name':name,'literal_key':normalize(name),'refs':refs,
            'start':m.start(),'name_start':name_start,'end':m.end(),'surface':name+' '+m.group(2).strip()})
        pos=m.end()
    if not units:return None
    return units,text[pos:]


def identify(unit,entities):
    matches=[g['key'] for g in entities if normalize(unit['name']) in g['values']]
    return matches[0] if len(matches)==1 else None


def supported_unit(unit,entities):
    key=identify(unit,entities)
    group=next((g for g in entities if g['key']==key),None)
    return key if group and set(group['supporting_doc_ids'])&set(unit['refs']) else None


def state_at(units,index,entities):
    supported={g['key'] for g in entities if g['supporting_doc_ids']}
    past={identify(u,entities) for u in units[:index]}-{None}
    future={supported_unit(u,entities) for u in units[index:]}-{None}
    future-=past
    return {'E_supported':sorted(supported),'E_past':sorted(past),
        'E_future':sorted(future),'delta_E':sorted(supported-past)}


def classify_repeat(units,index,entities):
    state=state_at(units,index,entities)
    current=identify(units[index],entities)
    literal=units[index]['literal_key']
    repeat=current in set(state['E_past']) if current else literal in {u['literal_key'] for u in units[:index]}
    if not repeat:return None,state
    if state['E_future']:return 'B',state
    correct_past={supported_unit(u,entities) for u in units[:index]}-{None}
    no_unknown=all(supported_unit(u,entities) is not None for u in units[:index])
    if not state['delta_E'] and correct_past==set(state['E_supported']) and no_unknown:return 'C',state
    return 'UNRESOLVED_REPEAT',state


def validate_label(record):
    kind=record['type'];past=set(record['state']['E_past']);future=set(record['state']['E_future']);delta=set(record['state']['delta_E'])
    if kind=='A':
        assert delta and record['target_entity'] in delta and record['target_entity'] not in past
        assert not record['positive_has_EOS']
    elif kind=='B':
        assert future and record['target_entity'] in future and record['target_entity'] not in past
        assert not record['positive_has_EOS']
    elif kind=='C':
        assert not delta and not future and record['positive_has_EOS']
    else:raise AssertionError('Unknown label type')
    assert record['positive_ids'][:record['shared_tokens']]==record['actual_ids'][:record['shared_tokens']]
    assert len(record['positive_ids'])>record['shared_tokens']
    for site in record['pair_sites']:
        assert not set(site['allowed'])&set(site['rejected']), 'Contradictory positive/negative token sets'


def common_prefix(a,b):
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y:return i
    return min(len(a),len(b))


def selective_positions(prompt_tokens,shared,buffer=2,maximum=16):
    positions=list(range(prompt_tokens-1,max(prompt_tokens-1,shared-buffer-1)))
    if len(positions)<=maximum:return positions
    return sorted({positions[round(i*(len(positions)-1)/(maximum-1))] for i in range(maximum)})


def blocked_first_ids(tok,names):
    return sorted({tok(' '+name,add_special_tokens=False)['input_ids'][0] for name in names if tok(' '+name,add_special_tokens=False)['input_ids']})


def validate_A_native_EOS(record,eos_ids=(151643,151645)):
    validate_label(record)
    if record['type']=='A':
        n=record['shared_tokens']
        assert n==len(record['actual_ids'])-1, 'A must supervise the exact observed end decision'
        assert record['actual_ids'][n] in eos_ids, 'A negative action must be actual EOS, not citation closure'
