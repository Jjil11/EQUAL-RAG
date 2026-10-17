"""Current aggregate viewer: stdlib-only, no training or downloads."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    print('EQUAL-RAG | through2026-10-17 | supplementary Seed20261015')
    print('Post-unsealing supplement, not another blind test or superiority claim.\n')
    for package,name in [('package2','Qwen3-14B'),('package3','Mistral-7B')]:
        s=json.loads((ROOT/'results/supplementary_20261017'/package/'summary.json').read_text(encoding='utf-8'))
        print(name+' | Full311')
        print('| System | Q % | Full F1 % | Tokens | Repetition % | True cap % |')
        print('|---|---:|---:|---:|---:|---:|')
        for model,views in s['tables'].items():
            v=views['full311']
            print(f"| {model} | {100*v['grounded_quality']:.2f} | {100*v['answer_f1']:.2f} | {v['mean_tokens']:.2f} | {100*v['repetition_rate']:.2f} | {100*v['true_cap_hit_rate']:.2f} |")
        print()
    print('Qwen EQUAL Q is lower than strong prompting and matched SFT/DPO.')
    print('Mistral AOnly improves over Base, not established superior to SFT.')
    print('See docs/EXPERIMENTS.md for all views, CIs and limits.')


if __name__=='__main__':
    main()
