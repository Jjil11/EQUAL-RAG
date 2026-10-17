import ast
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results/supplementary_20261017'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


class SupplementaryPublicationTests(unittest.TestCase):
    def test_counterevidence_not_hidden(self):
        tables=load(RESULTS/'package2/summary.json')['tables']
        for baseline in ['exhaustive_prompt','matched_sft','matched_dpo']:
            self.assertGreater(tables[baseline]['full311']['grounded_quality'],tables['a025']['full311']['grounded_quality'])

    def test_exact_statistical_function_bodies(self):
        original=ast.parse((ROOT/'research/trajectory_repair_20261017/analyze_heldout311_20261015.py').read_text(encoding='utf-8'))
        pure=ast.parse((ROOT/'research/supplementary_statistics_20261017.py').read_text(encoding='utf-8'))
        functions={n.name:ast.dump(n,include_attributes=False) for n in original.body if isinstance(n,ast.FunctionDef)}
        for node in pure.body:
            if isinstance(node,ast.FunctionDef):
                self.assertEqual(functions[node.name],ast.dump(node,include_attributes=False))

    def test_native_backbone_labels_not_fabricated(self):
        c=load(RESULTS/'package3/counts.json')
        self.assertEqual((c['A'],c['C1_embedded'],c['C2'],c['total_records']),(15,15,10,25))
        self.assertEqual((c['retention_trajectories'],c['retention_positions']),(120,5317))

    def test_no_raw_answer_token_traces_in_audit(self):
        audit=load(RESULTS/'tokenizer_boundary_audit.json')
        for tokenizer in audit['results']:
            self.assertEqual(tokenizer['mismatches'],21)
            self.assertTrue(tokenizer['distinct_merged_continuation'])
            for record in tokenizer['details']:
                self.assertEqual(set(record),{'case_id','mismatch','first_different_token'})

    def test_single_seed_supplement_is_not_new_blind(self):
        for pkg in ['package2','package3']:
            result=load(RESULTS/pkg/'summary.json')
            self.assertEqual(result['single_training_seed'],20261015)
            self.assertEqual(result['bootstrap_samples'],20000)
            self.assertTrue(result['post_unsealing_supplement_not_new_blind_test'])

    def test_numeric_point_and_ledger_reproduction(self):
        spec=importlib.util.spec_from_file_location('public_reproduction',ROOT/'scripts/reproduce_supplementary.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.reproduce(False)['status'],'PASS')


if __name__=='__main__':
    unittest.main()
