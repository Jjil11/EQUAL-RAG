import os
os.environ['EQUAL_QWEN_CLEAN_CONTROL']='action_ul'
import unittest
import torch
from train_qwen_clean_control_20261010 import matched_ul


class UL(unittest.TestCase):
    def test_rejected_gradient_increases_loss(self):
        x=torch.tensor([3.,1.,0.],requires_grad=True)
        loss,ce,ul=matched_ul(x,[1],0,False,1.)
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(ul),0.)
        loss.backward();self.assertGreater(float(x.grad[0]),0.);self.assertLess(float(x.grad[1]),0.)

    def test_extreme_probabilities_are_finite(self):
        x=torch.tensor([100.,-100.,0.],requires_grad=True)
        loss,_,_=matched_ul(x,[1],0,False,1.)
        loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(x.grad).all())


if __name__=='__main__':unittest.main()
