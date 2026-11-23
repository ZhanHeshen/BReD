import math
import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and Triton")
class MXINT8Test(unittest.TestCase):
    def test_codes_scales_and_replay_against_reference(self):
        from bred.quantization import quantize_mx_tri, dequantize_mx_tri

        torch.manual_seed(19)
        # Odd lengths, partial blocks, noncontiguous tensors, and different axes.
        for shape, axis in [((3, 35), -1), ((35, 3), 0), ((2, 35, 3), 1)]:
            x = torch.randn((*shape, 2), device="cuda")[..., 0]
            axis = axis % x.ndim
            rows = x.movedim(axis, -1).cpu().double()
            length = rows.shape[-1]
            padded = torch.zeros((*rows.shape[:-1], math.ceil(length / 32) * 32), dtype=torch.float64)
            padded[..., :length] = rows
            blocks = padded.reshape(-1, 32)
            exponents = torch.log2(blocks.abs().amax(-1).clamp_min(2**-126)).floor().clamp_min(-127)
            scales = 2.0**exponents
            noise = torch.linspace(0.125, 0.875, blocks.shape[0], dtype=torch.float64)
            for mode in ("nearest", "dither", "subtract_dither"):
                with self.subTest(shape=tuple(x.shape), axis=axis, mode=mode):
                    normalized = blocks / scales[:, None] * 64
                    rounded = normalized.round() if mode == "nearest" else (normalized + noise[:, None]).floor()
                    expected_codes = rounded.clamp(-127, 127).to(torch.int8)
                    expected_values = expected_codes.double()
                    if mode == "subtract_dither":
                        expected_values -= noise[:, None] - 0.5
                    expected_values = expected_values / 64 * scales[:, None]
                    expected_codes = expected_codes.reshape(padded.shape)[..., :length].movedim(-1, axis)
                    expected_values = expected_values.reshape(padded.shape)[..., :length].movedim(-1, axis)
                    expected_exp = exponents.to(torch.int8).reshape(*rows.shape[:-1], -1).movedim(-1, axis)
                    dither = noise.float().cuda()
                    codes, exp = quantize_mx_tri(x, elem_format="int8", axes=axis, round=mode, dither_rng=dither)
                    self.assertEqual(codes.shape, x.shape)
                    self.assertEqual(codes.dtype, torch.int8)
                    self.assertEqual(codes.numel() + exp.numel(), x.numel() + blocks.shape[0])
                    torch.testing.assert_close(codes.cpu(), expected_codes, rtol=0, atol=0)
                    torch.testing.assert_close(exp.cpu(), expected_exp, rtol=0, atol=0)
                    decoded = dequantize_mx_tri(codes, exp, elem_format="int8", axes=axis,
                                               round=mode, dither_rng=dither)
                    torch.testing.assert_close(decoded.cpu(), expected_values.float(), rtol=0, atol=2e-7)

    def test_rounding_saturation_and_no_fp4_exception(self):
        from bred.quantization import quantize_mx_tri, dequantize_mx_tri

        x = torch.zeros(3, 35, device="cuda")
        x[0, :8] = torch.tensor([1., -1., 0.5/64, 1.5/64, 2.5/64, -0.5/64, -1.5/64, -2.5/64])
        x[0, 32:] = torch.tensor([4., 2., -4.])
        x[1, :2] = torch.tensor([1.999, -1.999])
        codes, exp = quantize_mx_tri(x, elem_format="int8")
        self.assertEqual(codes[0, :8].tolist(), [64, -64, 0, 2, 2, 0, -2, -2])
        self.assertEqual(codes[1, :2].tolist(), [127, -127])
        self.assertEqual(exp[0].tolist(), [0, 2])
        decoded = dequantize_mx_tri(codes, exp, elem_format="int8")
        torch.testing.assert_close(decoded[2], x[2], rtol=0, atol=0)

        # INT8 code 2 must replay dither too; the FP4 mantissa exception does not apply.
        x[0, 0] = 2/64
        noise = torch.full((6,), 0.25, device="cuda")
        codes, exp = quantize_mx_tri(x, elem_format="int8", round="subtract_dither", dither_rng=noise)
        decoded = dequantize_mx_tri(codes, exp, elem_format="int8", round="subtract_dither", dither_rng=noise)
        self.assertEqual(codes[0, 0].item(), 2)
        self.assertEqual(decoded[0, 0].item(), 2.25/64)
