import copy
from itertools import product
import unittest

import torch

from bred import BReDNAdamW, BReDMARS


class MomentStorageTest(unittest.TestCase):
    def test_dense_nadamw_matches_pytorch(self):
        p = torch.nn.Parameter(torch.tensor([1., -2.]))
        q = torch.nn.Parameter(p.detach().clone())
        kwargs = dict(lr=3e-4, betas=(0.9, 0.99), weight_decay=0.1)
        actual = BReDNAdamW([p], quantization=False, **kwargs)
        expected = torch.optim.NAdam([q], decoupled_weight_decay=True, foreach=False, **kwargs)
        for grad in ([0.4, -0.7], [0., 0.2], [-0.5, 0.1]):
            p.grad = q.grad = torch.tensor(grad)
            actual.step()
            expected.step()
            torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_zero_gamma_mars_matches_adamw_below_clipping_threshold(self):
        params = [torch.nn.Parameter(torch.ones(shape)) for shape in ((2, 3), (3,))]
        refs = [torch.nn.Parameter(p.detach().clone()) for p in params]
        kwargs = dict(lr=3e-4, betas=(0.9, 0.99), weight_decay=0.1)
        actual = BReDMARS(params, quantization=False, gamma=0.,
                         betas_1d=kwargs["betas"], weight_decay_1d=0.1, **kwargs)
        expected = torch.optim.AdamW(refs, foreach=False, **kwargs)
        for value in (0.1, -0.05, 0.2):
            for p, q in zip(params, refs):
                p.grad = q.grad = torch.full_like(p, value)
            actual.step()
            expected.step()
            for p, q in zip(params, refs):
                torch.testing.assert_close(p, q)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and Triton")
    def test_mixed_groups_checkpoint_replays_updates(self):
        for cls, fmt in product((BReDNAdamW, BReDMARS), ("mxfp4", "mxint8")):
            with self.subTest(optimizer=cls.__name__, state_format=fmt):
                torch.manual_seed(7)
                params = [torch.nn.Parameter(torch.randn(shape, device="cuda"))
                          for shape in ((3, 35), (35,), (2, 35))]
                groups = lambda ps: [{"params": ps[:1]}, {"params": ps[1:2], "state_format": "mxfp4"},
                                     {"params": ps[2:], "quantization": False}]
                actual = cls(groups(params), global_seed=123, state_format=fmt)
                for step in range(3):
                    for i, p in enumerate(params):
                        p.grad = None if step == 1 and i == 0 else torch.randn_like(p) * 0.03
                    actual.step()
                self.assertNotIn("exp_avg", actual.state[params[0]])
                self.assertIn("exp_avg", actual.state[params[2]])
                self.assertEqual(actual.state[params[0]]["moment_step"], 2)
                self.assertEqual(actual.state[params[0]]["exp_avg_packed"].shape, (3, 35 if fmt == "mxint8" else 18))
                self.assertEqual(actual.state[params[0]]["exp_avg_shared_exp"].shape, (3, 2))
                refs = [torch.nn.Parameter(p.detach().clone()) for p in params]
                resumed = cls(groups(refs))
                resumed.load_state_dict(copy.deepcopy(actual.state_dict()))
                for step in range(3):
                    for i, (p, q) in enumerate(zip(params, refs)):
                        grad = None if step == 0 and i == 1 else torch.randn_like(p) * 0.03
                        p.grad = q.grad = grad
                    actual.step()
                    resumed.step()
                    for p, q in zip(params, refs):
                        torch.testing.assert_close(p, q, rtol=0, atol=0)
                        for key, value in actual.state[p].items():
                            restored = resumed.state[q][key]
                            if torch.is_tensor(value):
                                self.assertEqual(restored.dtype, value.dtype)
                                torch.testing.assert_close(restored, value, rtol=0, atol=0)
                            else:
                                self.assertEqual(restored, value)
                if cls is BReDMARS:
                    previous = actual.state[params[0]]["last_grad"].clone()
                    params[0].grad.zero_()
                    torch.testing.assert_close(actual.state[params[0]]["last_grad"], previous)
