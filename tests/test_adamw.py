import copy
import unittest

import torch

from bred import BReDAdamW


class AdamWTest(unittest.TestCase):
    def test_dense_group_matches_adamw(self):
        p = torch.nn.Parameter(torch.tensor([1., -2.]))
        q = torch.nn.Parameter(p.detach().clone())
        actual = BReDAdamW([{"params": [p], "quantization": False}])
        expected = torch.optim.AdamW([q])
        for grad in ([0.4, -0.7], [0., 0.2], [-0.5, 0.1]):
            p.grad = q.grad = torch.tensor(grad)
            actual.step()
            expected.step()
            torch.testing.assert_close(p, q)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and Triton")
    def test_packed_checkpoint_replays_next_update(self):
        for fmt in ("mxfp4", "mxint8"):
            with self.subTest(state_format=fmt):
                self.check_packed_resume(fmt)

    def check_packed_resume(self, fmt):
        p = torch.nn.Parameter(torch.ones(3, 35, device="cuda"))
        opt = BReDAdamW([p], global_seed=123, state_format=fmt)
        grad = torch.linspace(-1, 1, p.numel(), device="cuda").view_as(p)
        p.grad = grad
        opt.step()
        state = opt.state[p]
        self.assertEqual(state["exp_avg_packed"].shape, (3, 35 if fmt == "mxint8" else 18))
        self.assertEqual(state["exp_avg_shared_exp"].shape, (3, 2))
        q = torch.nn.Parameter(p.detach().clone())
        resumed = BReDAdamW([q])
        resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
        for param, optimizer in ((p, opt), (q, resumed)):
            param.grad = grad * 0.1
            optimizer.step()
        self.assertTrue(torch.isfinite(q).all().item())
        torch.testing.assert_close(q, p, rtol=0, atol=0)
        for name, value in opt.state[p].items():
            if torch.is_tensor(value):
                self.assertEqual(value.dtype, torch.int8)
                torch.testing.assert_close(resumed.state[q][name], value, rtol=0, atol=0)
