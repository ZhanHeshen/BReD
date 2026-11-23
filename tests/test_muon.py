import io
import math
import unittest

import torch

from bred import BReDMuon


class MuonTest(unittest.TestCase):
    def test_dense_update_matches_diagonal_reference(self):
        # For a rectangular diagonal matrix, Newton–Schulz acts on each
        # diagonal entry separately, so we can check the full update in scalars.
        for shape in ((2, 3), (3, 2)):
            p = torch.nn.Parameter(torch.ones(shape))
            opt = BReDMuon([p], lr=0.01, momentum=0.9, weight_decay=0.1, quantization=False)
            expected = p.detach().clone()
            buf = [0., 0.]
            for grad in ((1., -2.), (-0.4, 1.5), (0., -0.2)):
                p.grad = torch.zeros_like(p)
                p.grad.diagonal().copy_(torch.tensor(grad))
                opt.step()
                buf = [0.9 * m + 0.1 * g for m, g in zip(buf, grad)]
                update = [0.1 * g + 0.9 * m for g, m in zip(grad, buf)]
                norm = math.hypot(*update) + 1e-7
                x = [v / norm for v in update]
                for _ in range(5):
                    x = [3.4445 * v - 4.775 * v**3 + 2.0315 * v**5 for v in x]
                expected.mul_(0.999)
                expected.diagonal().add_(torch.tensor(x), alpha=-0.002 * math.sqrt(max(shape)))
                torch.testing.assert_close(p, expected)
                torch.testing.assert_close(opt.state[p]["momentum_buffer"].diagonal(), torch.tensor(buf))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and Triton")
    def test_packed_momentum_and_resume(self):
        for fmt in ("mxfp4", "mxint8"):
            with self.subTest(state_format=fmt):
                self.check_packed_resume(fmt)

    def check_packed_resume(self, fmt):
        p = torch.nn.Parameter(torch.ones(35, 19, device="cuda"))
        q = torch.nn.Parameter(p.detach().clone())
        packed = BReDMuon([p], global_seed=123, state_format=fmt)
        dense = BReDMuon([q], quantization=False)
        grad = torch.linspace(-1, 1, p.numel(), device=p.device).view_as(p)
        p.grad = q.grad = grad
        packed.step()
        dense.step()
        torch.testing.assert_close(p, q, rtol=0, atol=0)
        state = packed.state[p]
        self.assertEqual(set(state), {"step", "momentum_packed", "momentum_shared_exp"})
        self.assertEqual(state["momentum_packed"].shape, (35, 19 if fmt == "mxint8" else 10))
        self.assertEqual(state["momentum_shared_exp"].shape, (35, 1))
        self.assertEqual(state["momentum_shared_exp"].dtype, torch.int8)
        self.assertEqual(state["momentum_packed"].dtype, torch.int8)
        resumed = BReDMuon([q])
        checkpoint = io.BytesIO()
        torch.save(packed.state_dict(), checkpoint)
        checkpoint.seek(0)
        resumed.load_state_dict(torch.load(checkpoint, weights_only=True))
        for param, opt in ((p, packed), (q, resumed)):
            param.grad = -grad
            opt.step()
        self.assertTrue(torch.isfinite(p).all().item())
        torch.testing.assert_close(q, p, rtol=0, atol=0)
        for key, value in state.items():
            if torch.is_tensor(value):
                self.assertEqual(resumed.state[q][key].dtype, value.dtype)
                torch.testing.assert_close(resumed.state[q][key].float(), value.float(), rtol=0, atol=0)
