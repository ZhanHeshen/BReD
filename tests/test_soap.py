import io
from itertools import product
import unittest

import torch

from bred import BReDSOAP


class SOAPTest(unittest.TestCase):
    def test_dense_vector_matches_adam_without_epsilon_or_decay(self):
        p = torch.nn.Parameter(torch.tensor([1., -2.]))
        q = torch.nn.Parameter(p.detach().clone())
        kwargs = dict(lr=3e-4, betas=(0.95, 0.95), eps=0., weight_decay=0.)
        actual = BReDSOAP([p], quantization=False, **kwargs)
        reference = torch.optim.AdamW([q], foreach=False, **kwargs)
        p.grad = torch.ones_like(p)
        actual.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)  # Initialization skips the update.
        for grad in ([0.4, -0.7], [0.1, 0.2], [-0.5, 0.1]):
            p.grad = q.grad = torch.tensor(grad)
            actual.step()
            reference.step()
            torch.testing.assert_close(p, q)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and Triton")
    def test_packed_checkpoint_including_initialization_and_basis_refresh(self):
        for calls_before_save, fmt in product((1, 5), ("mxfp4", "mxint8")):
            with self.subTest(calls_before_save=calls_before_save, state_format=fmt):
                self.check_resume(calls_before_save, fmt)

    def check_resume(self, calls_before_save, fmt):
        torch.manual_seed(17)
        ps = [torch.nn.Parameter(torch.randn(shape, device="cuda"))
              for shape in ((8, 35), (35,), (4, 35))]
        groups = lambda xs: [{"params": xs[:2]}, {"params": xs[2:], "quantization": False}]
        actual = BReDSOAP(groups(ps), precondition_frequency=2, max_precond_dim=16, global_seed=123, state_format=fmt)
        for step in range(calls_before_save):
            for i, p in enumerate(ps):
                p.grad = None if step == 2 and i == 0 else torch.randn_like(p) * 0.03
            actual.step()
        state = actual.state[ps[0]]
        self.assertNotIn("exp_avg", state)
        self.assertNotIn("exp_avg_sq", state)
        self.assertNotIn("GG", state)
        self.assertEqual(state["GG_packed"][0][0].dtype, torch.int8)
        self.assertEqual(state["GG_packed"][0][0].shape, (8, 8 if fmt == "mxint8" else 4))
        self.assertEqual(state["Q"][0].dtype, torch.float32)
        self.assertEqual(state["GG_packed"][1], [])
        self.assertIn("GG", actual.state[ps[2]])

        qs = [torch.nn.Parameter(p.detach().clone()) for p in ps]
        resumed = BReDSOAP(groups(qs))
        checkpoint = io.BytesIO()
        torch.save(actual.state_dict(), checkpoint)
        checkpoint.seek(0)
        resumed.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        for step in range(4):
            for i, (p, q) in enumerate(zip(ps, qs)):
                p.grad = q.grad = None if step == 1 and i == 0 else torch.randn_like(p) * 0.03
            actual.step()
            resumed.step()
            for p, q in zip(ps, qs):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
                self.check_state_equal(actual.state[p], resumed.state[q])

    def check_state_equal(self, expected, actual):
        if torch.is_tensor(expected):
            self.assertEqual(expected.dtype, actual.dtype)
            self.assertEqual(expected.device, actual.device)
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self.check_state_equal(expected[key], actual[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(type(expected), type(actual))
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                self.check_state_equal(left, right)
        else:
            self.assertEqual(expected, actual)
