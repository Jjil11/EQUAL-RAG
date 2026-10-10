"""View saved aggregate results with Python's standard library only."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    data=json.loads((ROOT/'results/qwen_native_consistent/summary.json').read_text(encoding='utf-8'))
    print('EQUAL-RAG | Qwen3-14B development snapshot | 2026-10-11')
    print('Not an official benchmark score or confirmed method superiority.\n')
    print('| Condition | Arm | Lexical Q (%) | Mean tokens | Full F1 (%) | True cap (%) |')
    print('|---|---|---:|---:|---:|---:|')
    for setting,s in data['settings'].items():
        for arm in ['baseline','action_pair','action_ce','action_ul']:
            t=s['table'][arm]
            print(f"| {setting} | {arm} | {100*t['grounded_quality']:.3f} | {t['mean_tokens']:.3f} | {100*t['answer_f1']:.3f} | {100*t['true_cap_hit_rate']:.2f} |")
    print('\nPair minus anchor: relative utility, question-paired 95% CI')
    for setting,s in data['settings'].items():
        t=s['comparisons']['action_pair_minus_baseline']['relative_utility']
        print(f"  {setting}: {100*t['relative_delta']:+.2f}% [{100*t['ci95'][0]:+.2f}%, {100*t['ci95'][1]:+.2f}%]")
    print('\nConfirmatory expansion ready: '+str(data['development_gates']['action_pair']['ready_for_confirmatory_expansion']))
    print('Latest-version sealed predictions: '+str(data['holdout_predictions']))
    print('Human annotations: '+str(data['human_annotations']))
    print('See docs/EXPERIMENTS.md for semantic, mechanism, latency and metric limitations.')

if __name__=='__main__':main()
