"""Check vectorized bootstrap against the existing scalar implementation."""
import unittest
import numpy as np
from score_strict_units_20261010 import paired
from finalize_qwen_priority_20261010 import paired_fast


class Bootstrap(unittest.TestCase):
    def test_scalar_vectorized_equivalence(self):
        base=[];method=[];tb={};tm={}
        for i in range(8):
            row={'id':str(i),'gold_edges':i%3,'grounded_edges':i%2 if i%3 else 0,
                 'tokens':20+i,'answer_f1':i/10,'answer_precision':i/10,'answer_recall':i/10,
                 'answer_f1_top5':i/10,'repeated_units':i%2,'cap_hit_proxy':False}
            base.append(row);m=dict(row);m['tokens']+=(-1)**i*2;m['answer_f1']+=.01
            m['grounded_edges']=min(m['gold_edges'],m['grounded_edges']+int(i%4==0));method.append(m)
            tb[str(i)]={'hit_cap_without_eos':i==3};tm[str(i)]={'hit_cap_without_eos':False}
        old=paired(method,base);new=paired_fast(method,base,tm,tb)
        for key in ['grounded_quality','mean_tokens','utility','answer_f1','repetition_rate']:
            self.assertTrue(np.allclose(old[key]['ci95'],new[key]['ci95'],atol=1e-12),key)
            self.assertAlmostEqual(old[key]['delta'],new[key]['delta'],places=12)
        self.assertAlmostEqual(old['quality_noninferiority']['one_sided_95_lower'],
                               new['quality_noninferiority']['one_sided_95_lower'],places=12)


if __name__=='__main__':unittest.main()
