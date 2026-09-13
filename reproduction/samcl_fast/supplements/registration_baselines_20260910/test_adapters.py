import unittest
from types import SimpleNamespace
import torch
from baseline_methods import single, restore_method, method_state


class AdapterTests(unittest.TestCase):
    def test_projection_equivalence(self):
        torch.manual_seed(7)
        g=torch.randn(5,12,dtype=torch.float64)
        u=torch.linalg.qr(torch.randn(12,4,dtype=torch.float64)).Q
        self.assertTrue(torch.allclose(g-g@(u@u.T),g-(g@u)@u.T,atol=1e-12))

    def test_complete_registration_sample(self):
        b={'imgs':torch.ones(4,2,3,3,3),'masks':torch.ones(4,2,3,3,3),
           'segs':torch.zeros(4,2,3,3,3),'keypoints':[torch.ones(2,5,3)*i for i in range(4)],
           'names':['a','b','c','d']}
        s=single(b,2)
        self.assertEqual(s['names'],['c']); self.assertEqual(s['imgs'].shape[0],1)
        self.assertEqual(s['keypoints'][0].shape,(2,5,3)); self.assertEqual(s['keypoints'][0].mean(),2)
        self.assertEqual(set(s),set(b))

    def test_regularizer_state_resume(self):
        source=SimpleNamespace(checkpoint=torch.ones(3),big_omega=torch.arange(3.),small_omega=torch.ones(3)*2)
        target=SimpleNamespace()
        state=method_state(source,'si'); restore_method(target,'si',state,torch.device('cpu'))
        for k,v in state.items(): self.assertTrue(torch.equal(getattr(target,k),v))
        del state['small_omega']
        with self.assertRaises(ValueError): restore_method(target,'si',state,torch.device('cpu'))


if __name__=='__main__': unittest.main()
