"""CPU consistency tests; fixtures are never used as experiment records."""
from release_paths import ROOT as _release_root, QWEN_MODEL as _release_model, MISTRAL_MODEL as _release_mistral, CACHE_ROOT as _release_cache
import math
import unittest
import torch
import torch.nn.functional as F
from supp_runtime import *
from run_matched_baselines import dpo_loss
from run_second_backbone import guard

class Contracts(unittest.TestCase):
    def test_frozen_pool_and_order(self):
        rr=qwen_records()
        self.assertEqual(len(order(rr)),240)
        self.assertEqual(len({r['id'] for r in rr}),41)

    def test_dpo_at_reference(self):
        self.assertAlmostEqual(float(dpo_loss(torch.tensor(-4.),torch.tensor(-7.),-4.,-7.)),math.log(2),places=6)

    def test_dpo_gradient_direction(self):
        x=torch.tensor(0.,requires_grad=True);y=torch.tensor(0.,requires_grad=True)
        dpo_loss(x,y,0.,0.).backward()
        self.assertLess(float(x.grad),0);self.assertGreater(float(y.grad),0)

    def test_c2_identical_pairs_no_preference_gradient(self):
        x=torch.tensor(-2.,requires_grad=True)
        dpo_loss(x,x,-2.,-2.).backward()
        self.assertEqual(float(x.grad),0.)

    def test_sequence_logp_response_only_sum(self):
        class Fake:
            weight=torch.zeros(1)
            def get_input_embeddings(self):return self
            def __call__(self,input_ids,use_cache):
                class Output:pass
                out=Output();out.logits=torch.tensor([[[1.,2.,3.],[2.,3.,1.],[3.,1.,2.],[1.,2.,3.]]])
                return out
        value,count=sequence_logp(Fake(),[0,1,2,0],2)
        expected=F.log_softmax(torch.tensor([[2.,3.,1.],[3.,1.,2.]]),-1)[[0,1],[2,0]].sum()
        self.assertEqual(count,2);self.assertAlmostEqual(float(value),float(expected),places=6)

    def test_guard_identity_zero(self):
        ref=F.log_softmax(torch.tensor([[2.,1.,0.]]),-1)
        self.assertAlmostEqual(float(guard(ref,ref)),0.,places=6)

    def test_guard_flip_positive(self):
        ref=F.log_softmax(torch.tensor([[2.,1.,0.]]),-1)
        student=F.log_softmax(torch.tensor([[0.,1.,2.]]),-1)
        self.assertGreater(float(guard(student,ref)),1.)

    def test_unchanged_eval_sizes(self):
        rr=eval_rows()
        expected={'multi45':(45,176),'raw24':(24,8),'multi':(85,428),'sparse':(226,56)}
        for key,(n,g) in expected.items():
            group=[r for r in rr if r['eval_split']==key]
            self.assertEqual((len(group),sum(len(r['edges']) for r in group)),(n,g))

    def test_no_train_eval_overlap(self):
        train=read(SOURCE/'training_source350_snapshot.jsonl')
        self.assertFalse({r['id'] for r in train}&{r['id'] for r in eval_rows()})

    def test_mistral_compiler_fixture(self):
        from transformers import AutoTokenizer
        from entity_abc_contracts_20261015 import inventory
        import run_second_backbone_v2 as second
        tok=AutoTokenizer.from_pretrained(MISTRAL,local_files_only=True)
        source=next(r for r in read(SOURCE/'training_source350_snapshot.jsonl')
                    if sum(bool(g['supporting_doc_ids']) for g in inventory(r))>=2)
        g=next(g for g in inventory(source) if g['supporting_doc_ids'])
        name,cite=g['matched_surfaces'][0]
        records=[];pred=[]
        for typ in ['A','C2']:
            r=dict(source,id='UNIT_FIXTURE_'+typ)
            if typ=='C2':r['all_gold_answers']=[]
            pp=prompt_ids(tok,r['prompt'])
            text=tok.decode(pp,skip_special_tokens=False)+' '+name+' ['+str(cite)+']'
            full=tok.encode(text,add_special_tokens=False)
            self.assertEqual(full[:len(pp)],pp)
            generated=full[len(pp):]+[tok.eos_token_id]
            pred.append({'id':r['id'],'prompt_token_ids':pp,'generated_token_ids':generated,'ended_by_eos':True})
            records.append(r)
        import tempfile
        fixture_parent=safe(SUITE/'tests/fixtures')
        fixture_parent.mkdir(exist_ok=True)
        second.ROOT2=Path(tempfile.mkdtemp(prefix='compiler_',dir=str(fixture_parent)))
        rr,native=second.compile_records(tok,records,pred,[tok.eos_token_id])
        self.assertEqual(sorted(r['type'] for r in rr),['A','C'])
        self.assertEqual(len(native),2)

if __name__=='__main__':
    unittest.main(verbosity=2)
