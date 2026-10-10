import os
os.environ['EQUAL_QWEN_RETRIEVAL_ARM']='action_ul'
import unittest
import torch
import train_qwen_retrieval_align_20261010 as core


class RawLoss(unittest.TestCase):
    def test_wrapper_uses_ce_context_without_hinge(self):
        self.assertEqual(core.core.ARM,'action_ce')
        self.assertIs(core.core.action_loss,core.ul_loss)

    def test_same_legal_mass_and_illegal_penalty(self):
        x=torch.tensor([3.,2.,1.],requires_grad=True)
        loss,ce,ul=core.ul_loss(x,[1,2],0,False,1.)
        p=torch.softmax(x,0)
        self.assertTrue(torch.allclose(ce,-p[[1,2]].sum().log()))
        self.assertTrue(torch.allclose(ul,-torch.log1p(-p[0])))
        loss.backward();self.assertGreater(float(x.grad[0]),0.)

    def test_extreme_logits_finite(self):
        x=torch.tensor([100.,-100.,0.],requires_grad=True)
        loss,_,_=core.ul_loss(x,[1],0,False,1.)
        loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(x.grad).all())


if __name__=='__main__':unittest.main()
