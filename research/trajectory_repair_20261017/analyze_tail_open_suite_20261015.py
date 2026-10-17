"""Exhaustive reporting, not a metric-based gate; shared question bootstrap."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import json
import re
import numpy as np
import qwen_tail_fidelity_20261015 as c
from score_strict_units_20261010 import score, summarize, norm

q = c.q
SOURCE = q.ROOT / 'runs/qwen_tail_fidelity_v3_20261015'
ROOT = q.ROOT / 'runs/qwen_tail_open_suite_20261015'
FIELDS = ['grounded_quality', 'answer_f1', 'mean_tokens', 'repetition_rate', 'true_cap_hit_rate']
LABEL = {'base': '原始Qwen', 'base_block': 'Qwen＋防重探针', 'anchor': '旧Anchor',
    'ce60': 'Base-CE 60步（未合并）', 'pair60': 'Base-Pair 60步（未合并）',
    'scale070': 'Pair60 静态合并 λ=0.70', 'scale085': 'Pair60 静态合并 λ=0.85',
    'scale100': 'Pair60 静态合并 λ=1.00', 'scale115': 'Pair60 静态合并 λ=1.15',
    'ce60_merged': 'CE60 静态合并 λ=1.00（数值控制）',
    'guard_ce': 'Max-Guard CE60（合并）', 'guard_pair': 'Max-Guard Pair60（合并）'}


def membership(row, pred, cited):
    units = [u.strip() for u in pred['output'].rstrip().rstrip('.').rstrip(',').split(',') if norm(u)]
    return {i for i, g in enumerate(row['edges']) if any(norm(u) in {norm(a) for a in g['aliases']} and
        (not cited or set(g.get('supporting_doc_ids', [g['doc_id']])) & {int(x) for x in re.findall(r'\[(\d+)\]', u)}) for u in units)}


def bootstrap_draws(values, samples=20000, seed=20261010):
    """Identical RNG/question resampling to paired_fixed; vectorized once/model."""
    a = np.array([[r[f] for f in ['grounded_edges', 'gold_edges', 'tokens', 'answer_f1', 'repeated_units', 'cap_hit_proxy']] for r in values], dtype=float)
    ix = np.random.default_rng(seed).integers(0, len(values), (samples, len(values)))
    rv = a[ix]
    den = rv[:, :, 1].sum(1)
    qual = np.divide(rv[:, :, 0].sum(1), den, out=np.full(samples, np.nan), where=den > 0)
    draws = np.column_stack([qual, rv[:, :, 3].mean(1), rv[:, :, 2].mean(1), (rv[:, :, 4] > 0).mean(1), rv[:, :, 5].mean(1)])
    observed = np.array([a[:, 0].sum()/a[:, 1].sum(), a[:, 3].mean(), a[:, 2].mean(), (a[:, 4] > 0).mean(), a[:, 5].mean()])
    return observed, draws


def paired_from_draws(method, baseline):
    delta, draws = method[0]-baseline[0], method[1]-baseline[1]
    result = {f: {'delta': float(delta[i]), 'ci95': np.nanquantile(draws[:, i], [.025, .975]).tolist()} for i, f in enumerate(FIELDS)}
    result.update(samples=len(draws), seed=20261010, zero_Q_denominator_draws=int(np.isnan(draws[:, 0]).sum()),
        Q_interval_uses_only_defined_denominators=True, other_intervals_use_all_draws=True,
        question_level_paired=True, no_multiple_comparison_adjustment=True)
    return result


def audit(model, predictions, baseline, oldpair, byid, split, folder):
    rows = []
    for key, row in byid.items():
        if row['eval_split'] != split:
            continue
        item = {'id': key, 'edges': row['edges']}
        for mode, cited in [('name_only', False), ('locally_cited', True)]:
            a, b, old = membership(row, predictions[key], cited), membership(row, baseline[key], cited), membership(row, oldpair[key], cited)
            item[mode] = {'base': sorted(b), 'current': sorted(a), 'added': sorted(a-b), 'lost': sorted(b-a),
                'added_names': [row['edges'][i]['aliases'] for i in sorted(a-b)],
                'lost_names': [row['edges'][i]['aliases'] for i in sorted(b-a)],
                'old_added_retained': sorted((old-b) & a), 'old_lost_recovered': sorted((b-old) & a)}
        nbase, ncur = membership(row, baseline[key], False), membership(row, predictions[key], False)
        cbase, ccur = membership(row, baseline[key], True), membership(row, predictions[key], True)
        item['Q_decomposition'] = {'new_name_with_supported_citation': sorted(ccur-nbase),
            'old_name_citation_repaired': sorted((ccur & nbase)-cbase),
            'original_supported_name_missing': sorted(cbase-ncur),
            'name_retained_citation_lost': sorted((cbase & ncur)-ccur)}
        rows.append(item)
    q.write_rows(folder / (model+'__'+split+'.jsonl'), rows)
    result = {}
    for mode in ['name_only', 'locally_cited']:
        result[mode] = {field: sum(len(r[mode][field]) for r in rows) for field in ['base', 'current', 'added', 'lost', 'old_added_retained', 'old_lost_recovered']}
        result[mode]['net'] = result[mode]['added']-result[mode]['lost']
    result['Q_decomposition'] = {field: sum(len(r['Q_decomposition'][field]) for r in rows) for field in rows[0]['Q_decomposition']}
    result['count_goal_added17_lost3_met'] = result['locally_cited']['added'] >= 17 and result['locally_cited']['lost'] <= 3
    return result


def analyze(final=False):
    folder = ROOT / ('analysis_final' if final else 'analysis_stage1')
    folder.mkdir(exist_ok=False)
    (folder / 'ledgers').mkdir()
    inputs = q.read(ROOT / 'stage1/eval69.jsonl')
    byid = {r['id']: r for r in inputs}
    train = q.read(SOURCE / 'training_source350_snapshot.jsonl')
    assert not set(byid) & {r['id'] for r in train}
    from entity_abc_contracts_20261015 import normalize
    assert not {normalize(r['question']) for r in inputs} & {normalize(r['question']) for r in train}
    models = list(LABEL) if final else list(LABEL)[:5]
    predictions, scored, tables, audits, profiles = {}, {}, {}, {}, {}
    oldpair = {p['id']: p for p in q.read(c.PREVIOUS / 'eval/base_pair/predictions.jsonl')}
    for model in models:
        if model in ['base', 'anchor']:
            path = SOURCE / 'cached_eval' / (model+'.jsonl')
        elif model == 'base_block':
            path = SOURCE / 'eval/base_block/predictions.jsonl'
        elif model in ['ce60', 'pair60']:
            path = ROOT / 'stage1/eval' / ('base_ce' if model == 'ce60' else 'base_pair') / 'predictions.jsonl'
        else:
            path = ROOT / 'profiles' / model / 'eval69/predictions.jsonl'
            status = json.loads((ROOT / 'profiles' / model / 'status.json').read_text())
            assert status['state'] == 'COMPLETE'
            profiles[model] = status
        ps = q.read(path)
        assert len(ps) == 69 and len({p['id'] for p in ps}) == 69 and {p['id'] for p in ps} == set(byid)
        predictions[model] = {p['id']: p for p in ps}
        for p in ps:
            assert p['prompt_token_ids'] == predictions['base'][p['id']]['prompt_token_ids']
            assert p['tokens'] == len(p['generated_token_ids'])
            assert p['hit_cap_without_eos'] == (len(p['generated_token_ids']) == 256 and p['generated_token_ids'][-1] not in c.STOPS)
            if model not in ['base', 'base_block', 'anchor']:
                assert not p['blocking_events']
        vals = [dict(score(byid[p['id']], dict(p, grounded_edges=0)), cap_hit_proxy=p['hit_cap_without_eos']) for p in ps]
        scored[model] = {r['id']: r for r in vals}
        q.write_rows(folder / (model+'_scored69.jsonl'), vals)
        tables[model], audits[model] = {}, {}
        for split in ['multi45', 'raw24']:
            selected = [v for v in vals if byid[v['id']]['eval_split'] == split]
            tables[model][split] = dict(summarize(selected), true_cap_hit_rate=sum(v['cap_hit_proxy'] for v in selected)/len(selected))
            audits[model][split] = audit(model, predictions[model], predictions['base'], oldpair, byid, split, folder / 'ledgers')
            cited = audits[model][split]['locally_cited']
            assert cited['current'] == round(tables[model][split]['grounded_quality'] * (176 if split == 'multi45' else 8))
    intervals = {}
    comparisons = [(m, 'base') for m in models if m != 'base'] + [('pair60', 'ce60')]
    if final:
        comparisons += [(m, 'pair60') for m in ['scale070', 'scale085', 'scale100', 'scale115', 'guard_pair']]
        comparisons += [(m, 'scale100') for m in ['scale070', 'scale085', 'scale115', 'guard_pair']]
        comparisons += [('guard_pair', 'guard_ce'), ('guard_ce', 'ce60'), ('guard_ce', 'ce60_merged')]
    for split in ['multi45', 'raw24']:
        ids = sorted(k for k, r in byid.items() if r['eval_split'] == split)
        draws = {m: bootstrap_draws([scored[m][k] for k in ids]) for m in models}
        for model, baseline in comparisons:
            intervals[model+'__minus__'+baseline+'__'+split] = paired_from_draws(draws[model], draws[baseline])
    summary = {'models': models, 'tables': tables, 'name_audits': audits, 'paired_bootstrap': intervals,
        'profiles': profiles, 'bootstrap_samples': 20000, 'seed': 20261010,
        'training_gate_used': False, 'no_inference_rules_on_methods': True, 'sealed311_reads': 0,
        'reused_development_views_not_blind_test': True, 'metrics_are_lexical_proxies_not_semantic_certification': True,
        'script_sha256': q.sha(__file__)}
    if final:
        scales = ['scale070', 'scale085', 'scale100', 'scale115']
        values = {m: {'whole_prefix': profiles[m]['fidelity']['fully_preserved_trajectories'],
            'cited_net45': audits[m]['multi45']['locally_cited']['net'],
            'Raw24_tokens': tables[m]['raw24']['mean_tokens']} for m in scales}
        pareto = [m for m in scales if not any(values[n]['whole_prefix'] >= values[m]['whole_prefix'] and values[n]['cited_net45'] >= values[m]['cited_net45'] and
            (values[n]['whole_prefix'] > values[m]['whole_prefix'] or values[n]['cited_net45'] > values[m]['cited_net45']) for n in scales if n != m)]
        feasible = [m for m in scales if values[m]['whole_prefix'] >= 110]
        best = sorted(feasible, key=lambda m: (-values[m]['cited_net45'], values[m]['Raw24_tokens'], abs(float(m[-3:])/100-1)))[0] if feasible else None
        summary['static_scan_balance'] = {'values': values, 'pareto_retention_vs_net45': pareto,
            'threshold110_feasible': feasible, 'development_selected_under110_constraint': best,
            'selection_on_reused_dev_not_confirmatory': True,
            'rule': 'among whole_prefix>=110, maximize net45, then minimize Raw24 tokens, then lambda distance to 1; else report Pareto frontier only'}
    q.dump(folder / 'summary.json', summary)
    print(json.dumps({'analysis': str(folder), 'tables': tables, 'audits': {m: audits[m]['multi45']['locally_cited'] for m in models}}, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--final', action='store_true')
    args = ap.parse_args()
    analyze(args.final)
