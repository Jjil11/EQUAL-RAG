import unittest
from build_qwen_retrieval_align_20261010 import convert


class RankOnly(unittest.TestCase):
    def test_gold_does_not_choose_doc_six(self):
        item={'question':'Names?','answers':[['Hidden']],
              'docs':[{'title':str(i),'text':'Nothing'} for i in range(5)]+[{'title':'6','text':'Hidden'}]}
        r=convert(item,'fixed');self.assertEqual(len(r['docs']),5)
        self.assertEqual(r['edges'],[]);self.assertNotIn('Hidden',r['prompt'])

    def test_missing_metadata_reconstructed_and_citations_ranked(self):
        item={'question':'Names?','answers':[['Alice'],['']],
              'docs':[{'title':'x','text':'Unrelated'},{'title':'Alice','text':'Person'}]}
        r=convert(item,'fixed');self.assertEqual(len(r['edges']),1)
        self.assertEqual(r['edges'][0]['doc_id'],2);self.assertEqual(r['edges'][0]['supporting_doc_ids'],[2])
        self.assertFalse(r['zero_lexical_support'])


if __name__=='__main__':unittest.main()
