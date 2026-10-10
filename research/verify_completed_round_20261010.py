"""Validate frozen inputs, complete pairs and token-level termination; no inference."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(str(_release_root))
RUN = ROOT / 'runs/submission_20261010'
FRESH = ROOT / 'data/prospective_qampari_gtr5_v2_20261010'
ARMS = {'qwen3': ['baseline', 'semantic_eos', 'unit_ul', 'wrong_boundary', 'kl_only'],
        'mistral': ['baseline', 'semantic_eos', 'unit_ul']}
CORE_HASH = '1fa08803bfc18ce0167ca8c9a1f4f4edf6856cb6420864e8dfb59f3bef13224a'


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def mapping(rows):
    result = {row['id']: row for row in rows}
    assert len(rows) == len(result), 'Duplicate IDs'
    return result


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    state = json.loads((RUN / 'all_answers_finalization_status.json').read_text())
    assert state['status'] == 'COMPLETE'
    core = ROOT / 'code/train_verified_boundary_20261010.py'
    assert hashlib.sha256(core.read_bytes()).hexdigest() == CORE_HASH
    manifest = json.loads((FRESH / 'manifest.json').read_text())
    frozen = {}
    for split in ['dev', 'holdout']:
        path = FRESH / f'{split}.jsonl'
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == manifest['splits'][split]['sha256']
        rows = mapping(read(path))
        assert len(rows) == manifest['splits'][split]['n']
        frozen[split] = rows
    assert not (frozen['dev'].keys() & frozen['holdout'].keys())
    expected = {'regression_dev': mapping(read(ROOT / 'data/qampari_evidence_full/dev.jsonl')),
                'prospective_dev': frozen['dev']}
    report = {'status': 'PASS', 'n_predictions': 0, 'n_systems': 8,
              'training_core_sha256': CORE_HASH, 'frozen_splits_unchanged': True,
              'human_annotations': 0, 'holdout_predictions_in_round': 0,
              'training_audits': {}, 'evaluations': {}, 'termination_pairs': {},
              'input_sha256': {}, 'adapter_file_sha256': {}, 'anchor_cache_manifests': {}}
    shared = {'seed': 20260930, 'epochs': 1, 'lr': 1e-5, 'grad_accum': 8,
              'max_length': 2048, 'max_new_tokens': 256, 'trajectory_alpha': .1,
              'anchor_alpha': 10., 'anchor_kl_target': .0001, 'train_examples': 350,
              'disable_lora_dropout': True}
    for family, arms in ARMS.items():
        folder = RUN / f'{family}_frozen_comparison'
        for arm in arms:
            assert (folder / f'{arm}_status.txt').read_text().strip() == 'COMPLETE'
            if arm != 'baseline':
                train = RUN / f'{family}_{arm}' / 'train'
                audit = json.loads((train / 'boundary_training_audit.json').read_text())
                config = json.loads((train / 'train_manifest.json').read_text())
                assert audit['script_sha256'] == CORE_HASH and audit['seen_rows'] == 350
                assert all(config[key] == value for key, value in shared.items())
                assert audit['legacy_post_eot_suffix_excluded']
                assert config['disable_thinking'] == (family == 'qwen3')
                assert audit['active_boundaries'] == (0 if arm == 'kl_only' else (39 if family == 'qwen3' else 135))
                if arm in ['semantic_eos','unit_ul']:
                    assert audit['positive_boundary_losses'] > 0, 'No effective local training signal'
                assert 198 not in audit['semantic_eot_ids'], 'Post-EOT newline cannot be a stopping target'
                train_rows = mapping(read(config['train']))
                assert len(train_rows) == 350
                for subset in [*expected.values(), frozen['holdout']]:
                    assert not (train_rows.keys() & subset.keys()), 'Training/evaluation ID overlap'
                report['input_sha256'][config['train']] = digest(config['train'])
                cache = Path(config['anchor_logits_dir']) / 'cache_manifest.json'
                if cache.exists():
                    report['anchor_cache_manifests'][str(cache)] = {'sha256':digest(cache), 'manifest':json.loads(cache.read_text())}
                for adapter in [Path(config['init_adapter']), RUN / f'{family}_{arm}' / 'interp05/adapter']:
                    for file in sorted(adapter.iterdir()):
                        if file.is_file() and file.suffix in {'.json', '.safetensors', '.bin'}:
                            name = str(file.relative_to(ROOT))
                            if name not in report['adapter_file_sha256']:
                                report['adapter_file_sha256'][name] = digest(file)
                report['training_audits'][f'{family}_{arm}'] = {
                    **audit, 'elapsed_train_seconds': config['elapsed_train_seconds'],
                    'peak_memory_gib': max(x['peak_memory_gib'] for x in config['history'])}
        for setting, data in expected.items():
            keys = sorted(data)
            telemetry_by_arm = {}
            for arm in arms:
                path = folder / setting / arm
                preds = mapping(read(path / 'eval_results.jsonl'))
                telemetry = mapping(read(path / 'termination_telemetry.jsonl'))
                assert preds.keys() == data.keys() == telemetry.keys()
                assert not (preds.keys() & frozen['holdout'].keys())
                config = json.loads((path / 'termination_metrics.json').read_text())
                assert config['same_decoding_as_legacy'] and config['no_prompt_truncation']
                cap_proxy_differences = []
                for key in keys:
                    item = telemetry[key]
                    ids, stops = item['generated_token_ids'], item['eos_ids']
                    assert len(ids) == preds[key]['tokens'] and 0 < len(ids) <= 256
                    assert item['last_token_id'] == ids[-1]
                    assert item['ended_by_eos'] == (ids[-1] in stops)
                    assert item['hit_cap_without_eos'] == (len(ids) == 256 and ids[-1] not in stops)
                    if (preds[key]['tokens'] >= 256) != item['hit_cap_without_eos']:
                        cap_proxy_differences.append(key)
                report['n_predictions'] += len(preds)
                report['evaluations'][f'{family}/{setting}/{arm}'] = {
                    'n': len(preds), 'true_cap_hit_rate': config['true_cap_hit_rate'],
                    'configured_eos_ids': config['configured_eos_ids'],
                    'proxy_vs_true_cap_disagreement_ids': cap_proxy_differences}
                telemetry_by_arm[arm] = telemetry
            ix = np.random.default_rng(20261010).integers(0, len(keys), (20000, len(keys)))
            def compare(a, b):
                delta = np.array([int(telemetry_by_arm[a][k]['hit_cap_without_eos']) -
                                  int(telemetry_by_arm[b][k]['hit_cap_without_eos']) for k in keys])
                return {'delta': float(delta.mean()), 'ci95': np.quantile(delta[ix].mean(1), [.025,.975]).tolist()}
            pairs = {f'{arm}_minus_baseline': compare(arm, 'baseline') for arm in arms if arm != 'baseline'}
            pairs['semantic_eos_minus_unit_ul'] = compare('semantic_eos', 'unit_ul')
            report['termination_pairs'][f'{family}/{setting}'] = pairs
    assert report['n_predictions'] == 1280
    for path in RUN.rglob('eval_results.jsonl'):
        assert not (mapping(read(path)).keys() & frozen['holdout'].keys()), f'Holdout exposure: {path}'
    (RUN / 'integrity_and_termination_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ['status','n_predictions','n_systems','human_annotations','holdout_predictions_in_round']}, indent=2))


if __name__ == '__main__':
    main()
