import importlib.util,json,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class ReleaseTests(unittest.TestCase):
    def test_release_verifier(self):
        spec=importlib.util.spec_from_file_location('release_verifier',ROOT/'scripts/verify_release.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(module.verify()[0],[])

    def test_latest_scope_and_gate(self):
        d=json.loads((ROOT/'results/qwen_native_consistent/summary.json').read_text())
        self.assertEqual(d['new_predictions'],480)
        self.assertEqual(d['holdout_predictions'],0)
        self.assertEqual(d['human_annotations'],0)
        self.assertFalse(d['development_gates']['action_pair']['ready_for_confirmatory_expansion'])
        self.assertEqual(set(d['settings']),{'regression_dev','prospective_dev'})

    def test_nli_scope(self):
        d=json.loads((ROOT/'results/citation_nli/summary.json').read_text())
        self.assertEqual(d['predictions_screened'],1120)
        self.assertEqual(d['unique_nli_pairs'],943)
        self.assertEqual(d['truncated_pairs'],0)
        self.assertEqual(d['human_annotations'],0)

    def test_native_stop_failure_not_hidden(self):
        d=json.loads((ROOT/'results/diagnostics/training_sites.json').read_text())
        for arm in ['baseline','action_pair','action_ce','action_ul']:
            self.assertEqual(d['arms'][arm]['native_stop']['n'],72)
            self.assertEqual(d['arms'][arm]['native_stop']['eos_argmax_count'],0)

if __name__=='__main__':unittest.main()
