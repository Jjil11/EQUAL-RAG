import unittest
from score_strict_units_20261010 import score,paired


class StrictMetricTests(unittest.TestCase):
    def setUp(self):
        self.row={'id':'q','edges':[{'aliases':['Paris'],'doc_id':1},{'aliases':['London'],'doc_id':2}]}

    def test_neighbor_citation_not_credited(self):
        result=score(self.row,{'output':'Paris, London [1]','tokens':10,'grounded_edges':1})
        self.assertEqual(result['grounded_edges'],0)
        self.assertEqual(result['answer_precision'],1)

    def test_unit_local_correct(self):
        result=score(self.row,{'output':'Paris [1], London [2].','tokens':10,'grounded_edges':2})
        self.assertEqual(result['grounded_edges'],2)
        self.assertEqual(result['answer_f1'],1)

    def test_duplicate_not_inflate_recall(self):
        result=score(self.row,{'output':'Paris [1], Paris [1]','tokens':10,'grounded_edges':1})
        self.assertEqual(result['grounded_edges'],1)
        self.assertEqual(result['repeated_units'],1)

    def test_unsupported_unit_penalizes_precision(self):
        result=score(self.row,{'output':'Paris [1], Berlin [1]','tokens':10,'grounded_edges':1})
        self.assertEqual(result['answer_precision'],.5)

    def test_all_valid_supporting_documents(self):
        self.row['edges'][0]['supporting_doc_ids']=[1,2]
        result=score(self.row,{'output':'Paris [2]','tokens':10,'grounded_edges':0})
        self.assertEqual(result['grounded_edges'],1)

    def test_duplicate_pair_ids_rejected(self):
        result=score(self.row,{'output':'Paris [1]','tokens':10,'grounded_edges':1})
        with self.assertRaises(ValueError):paired([result,result],[result],samples=10)


if __name__=='__main__':unittest.main()
