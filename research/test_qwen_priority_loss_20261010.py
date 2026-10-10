"""CPU numerical tests; importing trainer does not run training."""
import os
os.environ['EQUAL_QWEN_PRIORITY_ARM']='action_pair'
import unittest
import torch
from train_qwen_priority_20261010 import action_loss


class Losses(unittest.TestCase):
    def test_single_token_ce_equivalence(self):
        x=torch.tensor([0.,1.,2.],requires_grad=True)
        loss,ce,hinge=action_loss(x,[2],0,False,1.)
        self.assertTrue(torch.allclose(ce,-torch.log_softmax(x,0)[2]))
        self.assertEqual(float(hinge),0.)
        loss.backward();self.assertLess(float(x.grad[2]),0.)

    def test_set_ce_is_probability_mass(self):
        x=torch.tensor([1.,2.,3.],requires_grad=True)
        _,ce,_=action_loss(x,[1,2],0,False,1.)
        self.assertTrue(torch.allclose(ce,-torch.softmax(x,0)[[1,2]].sum().log()))

    def test_hinge_uses_best_individual_not_sum(self):
        x=torch.tensor([3.,2.5,2.5],requires_grad=True)
        loss,ce,hinge=action_loss(x,[1,2],0,True,1.)
        self.assertAlmostEqual(float(hinge),1.5)
        loss.backward();self.assertGreater(float(x.grad[0]),0.)

    def test_satisfied_hinge_zero(self):
        x=torch.tensor([0.,2.,1.],requires_grad=True)
        _,_,hinge=action_loss(x,[1],0,True,1.)
        self.assertEqual(float(hinge),0.)


if __name__=='__main__':unittest.main()
