from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, CACHE_ROOT as _release_cache, MODELS_ROOT as _release_models
import unittest
import json
from pathlib import Path
from completion_pairs_20261010 import repair,covered,edge_matches,true_repeat,valid_training_edges


class SafetyTests(unittest.TestCase):
    def row(self,prefix):
        return {'id':'test','trajectory_prefix':prefix,'trajectory_continuation':', Alpha [1]',
                'edges':[{'answer':'Alpha','aliases':['Alpha'], 'doc_id':1},
                         {'answer':'Beta','aliases':['Beta'], 'doc_id':2}]}

    def test_incomplete_continues(self):
        r=repair(self.row('Alpha [1]'))
        self.assertEqual(r['action'],'complete_then_stop');self.assertEqual(r['chosen'],'Alpha [1], Beta [2]')

    def test_complete_stops(self):
        self.assertEqual(repair(self.row('Alpha [1], Beta [2]'))['action'],'stop')

    def test_wrong_citation_does_not_mark_covered(self):
        r=repair(self.row('Alpha [2]'));self.assertEqual(r['missing_edges'],[0,1])

    def test_no_neighbor_credit(self):
        self.assertEqual(covered('Alpha, wrong [1]',self.row('')['edges']),set())

    def test_secondary_support(self):
        e={'answer':'Alpha','aliases':['Alpha'],'doc_id':1,'supporting_doc_ids':[1,3]}
        self.assertEqual(edge_matches('Alpha [3]',[e]),{0})

    def test_internal_comma_is_not_repetition(self):
        self.assertIsNone(true_repeat({'short':'Alpha, Jr. [1], Beta, Jr. [2]'}))

    def test_citation_group_unit_repetition(self):
        self.assertEqual(true_repeat({'short':'Alpha, Jr. [1], Beta [2], Alpha, Jr. [1]'}),
                         ('Alpha, Jr. [1], Beta [2]','Alpha, Jr. [1]'))

    def test_comma_alias_covered(self):
        self.assertEqual(covered('Alpha, Jr. [1]',[{'answer':'Alpha, Jr.','aliases':['Alpha, Jr.'],'doc_id':1}]),{0})

    def test_empty_label_removed(self):
        self.assertEqual(valid_training_edges([{'answer':'','aliases':[''],'doc_id':1}]),[])

    def test_all_existing_training_labels(self):
        local=Path(__file__).resolve().parents[1]/'artifacts/submission_20261010/final/data/qampari_evidence_full'
        root=local if local.exists() else Path(str(_release_root / 'data/qampari_evidence_full'))
        for name in ['train_trajectory_onpolicy_seed20260930.jsonl','train_trajectory_qwen3_14b_onpolicy_seed20260930.jsonl']:
            for line in (root/name).read_text(encoding='utf-8').splitlines():
                row=json.loads(line)
                row['edges']=valid_training_edges(row['edges'])
                repeated=true_repeat(row) if row.get('has_trajectory') else None
                if repeated and row['edges']:
                    prefix,unit=repeated;row['trajectory_prefix']=prefix;row['trajectory_continuation']=', '+unit
                    with self.subTest(id=row['id']):repair(row)


if __name__=='__main__':unittest.main()
