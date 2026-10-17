"""Reconstruct every supplementary point/ledger/CI from text-free numeric cases."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'research'))
import numpy as np
import supplementary_statistics_20261017 as frozen


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def reproduce(with_intervals=True):
    counts = {'point_values':0, 'ledger_summaries':0, 'CI_values':0}
    for package in ['package2','package3']:
        folder = ROOT/'results/supplementary_20261017'/package
        published = load(folder/'summary.json')
        public = load(folder/'question_statistics.json')
        assert len(public['cases']) == 380
        assert [c['case_id'] for c in public['cases']] == [f'case_{i:04d}' for i in range(380)]
        assert set(public['systems']) == set(published['tables'])
        for view,n,g in [('multi45',45,176),('raw24',24,8),('full311',311,484),('multi',85,428),('sparse',226,56)]:
            indices = [i for i,c in enumerate(public['cases']) if c['split']==view or
                       (view=='full311' and c['split'] in ['multi','sparse'])]
            assert len(indices)==n
            arrays = {m:np.asarray(v['values'],float)[indices] for m,v in public['systems'].items()}
            observed = {m:frozen.observed(a) for m,a in arrays.items()}
            for model,array in arrays.items():
                assert int(array[:,1].sum())==g
                for field,value in zip(frozen.FIELDS,observed[model]):
                    assert math.isclose(value,published['tables'][model][view][field],rel_tol=0,abs_tol=1e-12),(package,model,view,field)
                    counts['point_values']+=1
                records = [public['systems'][model]['attribution'][i] for i in indices]
                expected = published['ledgers'][model][view]
                for group in ['name_only','locally_cited','Q_decomposition']:
                    for key,value in expected[group].items():
                        assert sum(r[group][key] for r in records)==value,(package,model,view,group,key)
                cited = expected['locally_cited']
                decomposition = expected['Q_decomposition']
                assert cited['net']==cited['added']-cited['lost']
                assert cited['added']==decomposition['new_name_with_supported_citation']+decomposition['old_name_citation_repaired']
                assert cited['lost']==decomposition['original_supported_name_missing']+decomposition['name_retained_citation_lost']
                counts['ledger_summaries']+=1
            if not with_intervals:
                continue
            ix = np.random.default_rng(20261010).integers(0,n,(20000,n))
            draws = {m:frozen.bootstrap(a,ix) for m,a in arrays.items()}
            for comparison,expected in published['paired_bootstrap'].items():
                model,other = comparison.split('__minus__')
                baseline,subset = other.rsplit('__',1)
                if subset!=view:
                    continue
                actual = frozen.interval(observed[model]-observed[baseline],draws[model]-draws[baseline],expected['kind'])
                for field in frozen.FIELDS:
                    assert actual[field]['defined_draws']==expected[field]['defined_draws']
                    for value,wanted in zip([actual[field]['delta'],*actual[field]['ci95']],
                                            [expected[field]['delta'],*expected[field]['ci95']]):
                        assert value==wanted or (value is not None and wanted is not None and math.isclose(value,wanted,rel_tol=0,abs_tol=1e-12)),(comparison,field)
                        counts['CI_values']+=1
    assert counts['point_values']==325 and counts['ledger_summaries']==65
    assert counts['CI_values']==(1425 if with_intervals else 0)
    return {'status':'PASS','checks':counts,'bootstrap_resamples':20000 if with_intervals else 0,
            'question_rng_seed':20261010,'numeric_only_reproduction':True,
            'text_scoring_or_model_generation_revalidated':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--points-only',action='store_true',help='Skip the 20,000-draw CI reproduction')
    args=parser.parse_args()
    print(json.dumps(reproduce(not args.points_only),indent=2))
