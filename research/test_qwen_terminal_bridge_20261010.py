import os,unittest
os.environ['EQUAL_QWEN_TERMINAL_ARM']='action_pair'
import torch
from train_qwen_terminal_bridge_20261010 import terminal_loss,first_loss
class Tests(unittest.TestCase):
    def test_terminal_updates_actual_end_class(self):
        logits=torch.tensor([0.,1.,2.,3.],requires_grad=True)
        loss=terminal_loss(logits,[2,3]);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertLess(float(logits.grad[2]),0.)
        self.assertLess(float(logits.grad[3]),0.);self.assertGreater(float(logits.grad[0]),0.)
    def test_eos_probability_aggregation(self):
        logits=torch.tensor([0.,1.,2.,3.]);lp=torch.log_softmax(logits,0)
        self.assertTrue(torch.allclose(terminal_loss(logits,[2,3]),-torch.logsumexp(lp[[2,3]],0)))
    def test_first_and_terminal_separate(self):
        logits=torch.tensor([0.,1.,2.,3.],requires_grad=True)
        loss,ce,contrast=first_loss(logits,[1],0)
        self.assertTrue(torch.isfinite(loss+ce+contrast))
        loss.backward();self.assertLess(float(logits.grad[1]),0.)
if __name__=='__main__':unittest.main()
