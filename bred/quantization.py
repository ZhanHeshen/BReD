"""MXFP4 E2M1 and MXINT8 quantization with block dither replay.

Each element is encoded as one standard FP4 E2M1 nibble.  Two adjacent
elements along ``axes`` are stored in one int8 value: the even element uses
the low nibble and the odd element uses the high nibble. MXFP4 stores one
signed int8 exponent per block. MXINT8 stores one signed byte per element,
representing code / 64, with the same shared-exponent storage.
"""

import math

import torch
import triton
import triton.language as tl


FP32_MIN_NORMAL = tl.constexpr(2.0 ** -126)
_FP4_FORMATS = {"fp4", "fp4_e2m1"}


def _validate_common(tensor: torch.Tensor, elem_format: str, axes, block_size: int):
    if elem_format.lower() not in _FP4_FORMATS | {"int8"}:
        raise NotImplementedError(
            "Supported element formats are fp4_e2m1 and int8."
        )
    if not tensor.is_cuda:
        raise ValueError("The packed Triton path requires a CUDA tensor.")
    if not tensor.is_floating_point() and tensor.dtype != torch.int8:
        raise TypeError("Expected a floating-point tensor or packed int8 tensor.")
    if tensor.ndim == 0:
        raise ValueError("Scalar tensors are not supported by the MX path.")
    if block_size < 2 or block_size & (block_size - 1) or block_size % 2:
        raise ValueError("block_size must be an even power of two.")
    if not isinstance(axes, int):
        if len(axes) != 1:
            raise NotImplementedError("Only one quantization axis is supported.")
        axis = axes[0]
    else:
        axis = axes
    if not -tensor.ndim <= axis < tensor.ndim:
        raise IndexError(f"axis {axis} is out of range for a {tensor.ndim}-D tensor")
    return axis % tensor.ndim


@triton.jit
def _fp4_quantize_code(value, u, USE_DITHER: tl.constexpr):
    """Convert a value already divided by the block scale to one FP4 code."""
    abs_value = tl.abs(value)
    log2_e = tl.floor(
        tl.log2(abs_value + (abs_value == 0.0).to(tl.float32) * 1e-38)
    )
    # MXFP4 E2M1 uses exponent 0 for subnormal values and exponents 0..2.
    private_exp = tl.maximum(log2_e, 0.0)
    mantissa = value * tl.exp2(1.0 - private_exp)

    if USE_DITHER:
        mantissa = tl.floor(mantissa + u)
    else:
        sign = tl.where(mantissa >= 0.0, 1.0, -1.0)
        mantissa = sign * tl.floor(tl.abs(mantissa) + 0.5)

    sign = tl.where(mantissa >= 0.0, 1.0, -1.0)
    overflow = tl.abs(mantissa) >= 4.0
    mantissa = tl.where(overflow, mantissa * 0.5, mantissa)
    private_exp = tl.where(overflow, private_exp + 1.0, private_exp)

    exp_overflow = private_exp > 2.0
    mantissa = tl.where(exp_overflow, 3.0 * sign, mantissa)
    private_exp = tl.minimum(private_exp, 2.0)

    mantissa_i = mantissa.to(tl.int16)
    magnitude = tl.abs(mantissa_i)
    # Canonical E2M1 magnitudes:
    # code 0..7 -> 0, .5, 1, 1.5, 2, 3, 4, 6.
    magnitude_code = private_exp.to(tl.int16) * 2 + magnitude
    sign_bit = tl.where(mantissa_i < 0, 8, 0).to(tl.int16)
    return (magnitude_code | sign_bit) & 0xF


@triton.jit
def _fp4_decode_code(code, u, SUBTRACT_DITHER: tl.constexpr):
    """Decode one FP4 nibble to the block-normalized floating-point value."""
    code = code.to(tl.int16) & 0xF
    magnitude_code = code & 0x7
    negative = (code & 0x8) != 0

    private_exp = tl.where(
        magnitude_code >= 4, (magnitude_code - 2) >> 1, 0
    ).to(tl.float32)
    mantissa_magnitude = tl.where(
        magnitude_code < 4,
        magnitude_code,
        2 + (magnitude_code & 1),
    ).to(tl.float32)
    mantissa = tl.where(negative, -mantissa_magnitude, mantissa_magnitude)

    if SUBTRACT_DITHER:
        mantissa -= tl.where(mantissa_magnitude == 2, 0.0, u - 0.5)

    return mantissa * tl.exp2(private_exp - 1.0)


@triton.jit
def _int8_quantize_code(value, u, USE_DITHER: tl.constexpr):
    # MXINT8 has six fractional bits; reserve -128 for symmetric saturation.
    value = value * 64.0
    if USE_DITHER:
        rounded = tl.floor(value + u)
    else:
        lower = tl.floor(value)
        fraction = value - lower
        increment = (fraction > 0.5) | ((fraction == 0.5) & ((lower.to(tl.int32) & 1) != 0))
        rounded = lower + increment.to(tl.float32)
    return tl.minimum(tl.maximum(rounded, -127.0), 127.0).to(tl.int8)


@triton.jit
def _shared_exponent(a0, a1, use_ceil: tl.constexpr, flush_subnorms: tl.constexpr,
                     INT8: tl.constexpr = False):
    block_max = tl.max(tl.maximum(tl.abs(a0), tl.abs(a1)), axis=0)
    safe_max = block_max + (block_max == 0.0).to(tl.float32) * FP32_MIN_NORMAL
    log2_max = tl.log2(safe_max)
    if use_ceil:
        exponent = tl.ceil(log2_max)
    else:
        exponent = tl.floor(log2_max)
    exponent -= 0.0 if INT8 else 2.0  # Element maximum exponent.

    invalid = (exponent > 127.0) | (exponent != exponent)
    exponent = tl.maximum(exponent, -127.0)
    scale_exponent = tl.where(invalid, float("nan"), exponent)
    stored_exponent = tl.where(invalid, -128.0, exponent).to(tl.int8)

    if flush_subnorms:
        keep = scale_exponent > -127.0
        a0 = tl.where(keep, a0, 0.0)
        a1 = tl.where(keep, a1, 0.0)
    scale = tl.exp2(-scale_exponent)
    return a0 * scale, a1 * scale, stored_exponent


@triton.jit
def _mx_quantize_nearest_kernel(
    A_ptr,
    packed_ptr,
    shared_exp_ptr,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
    use_ceil: tl.constexpr,
    flush_subnorms: tl.constexpr,
    INT8: tl.constexpr = False,
):
    block_id = tl.program_id(0)
    pair_offsets = tl.arange(0, BLOCK_SIZE // 2)
    input_base = block_id * stride_row
    a0 = tl.load(A_ptr + input_base + pair_offsets * 2).to(tl.float32)
    a1 = tl.load(A_ptr + input_base + pair_offsets * 2 + 1).to(tl.float32)

    a0, a1, stored_exponent = _shared_exponent(
        a0, a1, use_ceil, flush_subnorms, INT8
    )
    tl.store(shared_exp_ptr + block_id, stored_exponent)

    if INT8:
        code0 = _int8_quantize_code(a0, 0.0, USE_DITHER=False)
        code1 = _int8_quantize_code(a1, 0.0, USE_DITHER=False)
        tl.store(packed_ptr + block_id * BLOCK_SIZE + pair_offsets * 2, code0)
        tl.store(packed_ptr + block_id * BLOCK_SIZE + pair_offsets * 2 + 1, code1)
    else:
        code0 = _fp4_quantize_code(a0, 0.0, USE_DITHER=False)
        code1 = _fp4_quantize_code(a1, 0.0, USE_DITHER=False)
        packed = (code0 | (code1 << 4)).to(tl.int8)
        tl.store(packed_ptr + block_id * (BLOCK_SIZE // 2) + pair_offsets, packed)


@triton.jit
def _mx_quantize_dither_kernel(
    A_ptr,
    dither_ptr,
    packed_ptr,
    shared_exp_ptr,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
    use_ceil: tl.constexpr,
    flush_subnorms: tl.constexpr,
    INT8: tl.constexpr = False,
):
    block_id = tl.program_id(0)
    pair_offsets = tl.arange(0, BLOCK_SIZE // 2)
    input_base = block_id * stride_row
    a0 = tl.load(A_ptr + input_base + pair_offsets * 2).to(tl.float32)
    a1 = tl.load(A_ptr + input_base + pair_offsets * 2 + 1).to(tl.float32)
    u = tl.load(dither_ptr + block_id).to(tl.float32)

    a0, a1, stored_exponent = _shared_exponent(
        a0, a1, use_ceil, flush_subnorms, INT8
    )
    tl.store(shared_exp_ptr + block_id, stored_exponent)

    if INT8:
        code0 = _int8_quantize_code(a0, u, USE_DITHER=True)
        code1 = _int8_quantize_code(a1, u, USE_DITHER=True)
        tl.store(packed_ptr + block_id * BLOCK_SIZE + pair_offsets * 2, code0)
        tl.store(packed_ptr + block_id * BLOCK_SIZE + pair_offsets * 2 + 1, code1)
    else:
        code0 = _fp4_quantize_code(a0, u, USE_DITHER=True)
        code1 = _fp4_quantize_code(a1, u, USE_DITHER=True)
        packed = (code0 | (code1 << 4)).to(tl.int8)
        tl.store(packed_ptr + block_id * (BLOCK_SIZE // 2) + pair_offsets, packed)


@triton.jit
def _mx_dequantize_kernel(
    packed_ptr,
    shared_exp_ptr,
    out_ptr,
    BLOCK_SIZE: tl.constexpr,
    SUBTRACT_DITHER: tl.constexpr,
    dither_ptr,
    INT8: tl.constexpr = False,
):
    block_id = tl.program_id(0)
    pair_offsets = tl.arange(0, BLOCK_SIZE // 2)
    if INT8:
        code0 = tl.load(packed_ptr + block_id * BLOCK_SIZE + pair_offsets * 2).to(tl.float32)
        code1 = tl.load(packed_ptr + block_id * BLOCK_SIZE + pair_offsets * 2 + 1).to(tl.float32)
    else:
        packed = tl.load(
            packed_ptr + block_id * (BLOCK_SIZE // 2) + pair_offsets
        ).to(tl.int16)
        code0 = packed & 0xF
        code1 = (packed >> 4) & 0xF

    stored_exponent = tl.load(shared_exp_ptr + block_id).to(tl.int16)
    shared_exponent = tl.where(
        stored_exponent == -128,
        float("nan"),
        stored_exponent.to(tl.float32),
    )
    scale = tl.exp2(shared_exponent)
    if SUBTRACT_DITHER:
        u = tl.load(dither_ptr + block_id).to(tl.float32)
    else:
        u = 0.0

    if INT8:
        offset = u - 0.5 if SUBTRACT_DITHER else 0.0
        a0 = ((code0 - offset) / 64.0) * scale
        a1 = ((code1 - offset) / 64.0) * scale
    else:
        a0 = _fp4_decode_code(code0, u, SUBTRACT_DITHER) * scale
        a1 = _fp4_decode_code(code1, u, SUBTRACT_DITHER) * scale
    output_base = block_id * BLOCK_SIZE
    tl.store(out_ptr + output_base + pair_offsets * 2, a0)
    tl.store(out_ptr + output_base + pair_offsets * 2 + 1, a1)


def _axis_permutation(ndim: int, axis: int):
    perm = list(range(ndim))
    perm.pop(axis)
    perm.append(axis)
    inv_perm = [0] * ndim
    for index, original_axis in enumerate(perm):
        inv_perm[original_axis] = index
    return perm, inv_perm


def _to_blocks(A: torch.Tensor, axis: int, block_size: int):
    perm, inv_perm = _axis_permutation(A.ndim, axis)
    A_permuted = A.permute(perm).contiguous()
    original_shape = tuple(A_permuted.shape)
    length = original_shape[-1]
    padded_length = math.ceil(length / block_size) * block_size
    padded_shape = original_shape[:-1] + (padded_length,)

    if padded_length != length:
        A_padded = A_permuted.new_zeros(padded_shape)
        A_padded[..., :length] = A_permuted
    else:
        A_padded = A_permuted

    number_of_blocks = A_padded.numel() // block_size
    return (
        A_padded.reshape(number_of_blocks, block_size).contiguous(),
        number_of_blocks,
        original_shape,
        padded_shape,
        inv_perm,
    )


def _packed_from_blocks(
    packed_blocks: torch.Tensor,
    original_shape,
    padded_shape,
    inv_perm,
    values_per_byte=2,
):
    packed_padded_shape = padded_shape[:-1] + (padded_shape[-1] // values_per_byte,)
    packed_permuted = packed_blocks.reshape(packed_padded_shape)
    packed_length = (original_shape[-1] + values_per_byte - 1) // values_per_byte
    packed_permuted = packed_permuted[..., :packed_length]
    return packed_permuted.permute(inv_perm).contiguous()


def _packed_to_blocks(
    packed: torch.Tensor,
    output_shape,
    axis: int,
    block_size: int,
    values_per_byte=2,
):
    output_shape = tuple(output_shape)
    if len(output_shape) != packed.ndim:
        raise ValueError("output_shape must have the same rank as packed.")
    perm, inv_perm = _axis_permutation(packed.ndim, axis)
    output_permuted_shape = tuple(output_shape[index] for index in perm)
    packed_permuted = packed.permute(perm).contiguous()
    expected_packed_shape = output_permuted_shape[:-1] + (
        (output_permuted_shape[-1] + values_per_byte - 1) // values_per_byte,
    )
    if tuple(packed_permuted.shape) != expected_packed_shape:
        raise ValueError(
            f"packed shape {tuple(packed.shape)} is incompatible with "
            f"output_shape {output_shape} along axis {axis}."
        )

    padded_length = (
        math.ceil(output_permuted_shape[-1] / block_size) * block_size
    )
    packed_padded_shape = output_permuted_shape[:-1] + (padded_length // values_per_byte,)
    if expected_packed_shape[-1] != packed_padded_shape[-1]:
        packed_padded = packed_permuted.new_zeros(packed_padded_shape)
        packed_padded[..., : expected_packed_shape[-1]] = packed_permuted
    else:
        packed_padded = packed_permuted

    number_of_blocks = packed_padded.numel() // (block_size // values_per_byte)
    packed_blocks = packed_padded.reshape(
        number_of_blocks, block_size // values_per_byte
    ).contiguous()
    padded_output_shape = output_permuted_shape[:-1] + (padded_length,)
    return (
        packed_blocks,
        number_of_blocks,
        output_permuted_shape,
        padded_output_shape,
        inv_perm,
    )


def _from_blocks(flat: torch.Tensor, original_shape, padded_shape, inv_perm):
    padded = flat.reshape(padded_shape)
    slices = tuple(slice(0, size) for size in original_shape)
    return padded[slices].permute(inv_perm).contiguous()


def quantize_mx_tri(
    A,
    scale_bits=8,
    elem_format="fp4_e2m1",
    shared_exp_method="max",
    axes=-1,
    block_size=32,
    round="nearest",
    flush_fp32_subnorms=False,
    use_ceil=False,
    dither_rng=None,
):
    """Quantize a CUDA tensor to MXFP4 or MXINT8 storage.

    The returned packed tensor has the same rank as ``A``; its quantized axis
    has length ``ceil(A.shape[axis] / 2)``.  ``shared_exp`` has one int8 value
    per block along that axis. For int8, the codes retain the input shape.
    """
    axis = _validate_common(A, elem_format, axes, block_size)
    int8 = elem_format.lower() == "int8"
    values_per_byte = 1 if int8 else 2
    if not A.is_floating_point():
        raise TypeError("A must be a floating-point tensor.")
    if scale_bits != 8:
        raise NotImplementedError("Packed shared exponents currently require 8 bits.")
    if shared_exp_method != "max":
        raise NotImplementedError("Only shared_exp_method='max' is supported.")
    if round not in ("nearest", "dither", "subtract_dither"):
        raise ValueError(f"Unsupported round mode: {round}")

    A_blocks, n_blocks, original_shape, padded_shape, inv_perm = _to_blocks(
        A, axis, block_size
    )
    packed_blocks = torch.empty(
        (n_blocks, block_size // values_per_byte), dtype=torch.int8, device=A.device
    )
    shared_exp_flat = torch.empty(n_blocks, dtype=torch.int8, device=A.device)

    grid = (n_blocks,)
    common = dict(
        stride_row=block_size,
        BLOCK_SIZE=block_size,
        use_ceil=use_ceil,
        flush_subnorms=flush_fp32_subnorms,
        INT8=int8,
    )
    if round == "nearest":
        _mx_quantize_nearest_kernel[grid](
            A_blocks, packed_blocks, shared_exp_flat, **common
        )
    else:
        if dither_rng is None:
            raise ValueError("dither_rng is required for dithered quantization.")
        if dither_rng.device != A.device:
            raise ValueError("dither_rng must be on the same device as A.")
        dither_flat = dither_rng.reshape(-1).contiguous()
        if dither_flat.numel() != n_blocks:
            raise ValueError(
                f"dither_rng has {dither_flat.numel()} values; expected {n_blocks}."
            )
        _mx_quantize_dither_kernel[grid](
            A_blocks, dither_flat, packed_blocks, shared_exp_flat, **common
        )

    packed_out = _packed_from_blocks(
        packed_blocks, original_shape, padded_shape, inv_perm, values_per_byte
    )
    blocks_per_row = padded_shape[-1] // block_size
    shared_exp_permuted = shared_exp_flat.reshape(
        original_shape[:-1] + (blocks_per_row,)
    )
    shared_exp_out = shared_exp_permuted.permute(inv_perm).contiguous()
    return packed_out, shared_exp_out


def dequantize_mx_tri(
    packed,
    shared_exp,
    elem_format="fp4_e2m1",
    axes=-1,
    block_size=32,
    round="nearest",
    dither_rng=None,
    output_shape=None,
):
    """Dequantize MXFP4 or MXINT8 values to float32.

    ``output_shape`` is required when the original quantized-axis length was
    odd.  Supplying it unconditionally is recommended.
    """
    axis = _validate_common(packed, elem_format, axes, block_size)
    int8 = elem_format.lower() == "int8"
    values_per_byte = 1 if int8 else 2
    if packed.dtype != torch.int8:
        raise TypeError("packed must have dtype torch.int8.")
    if shared_exp.dtype != torch.int8:
        raise TypeError("shared_exp must have dtype torch.int8.")
    if shared_exp.device != packed.device:
        raise ValueError("shared_exp and packed must be on the same device.")
    if round not in ("nearest", "dither", "subtract_dither"):
        raise ValueError(f"Unsupported round mode: {round}")

    if output_shape is None:
        output_shape = list(packed.shape)
        output_shape[axis] *= values_per_byte
    packed_blocks, n_blocks, original_shape, padded_shape, inv_perm = (
        _packed_to_blocks(packed, output_shape, axis, block_size, values_per_byte)
    )

    perm, _ = _axis_permutation(shared_exp.ndim, axis)
    shared_exp_flat = shared_exp.permute(perm).contiguous().reshape(-1)
    if shared_exp_flat.numel() != n_blocks:
        raise ValueError(
            f"shared_exp has {shared_exp_flat.numel()} values; expected {n_blocks}."
        )

    if round == "subtract_dither":
        if dither_rng is None:
            raise ValueError("dither_rng is required for subtract_dither decoding.")
        if dither_rng.device != packed.device:
            raise ValueError("dither_rng must be on the same device as packed.")
        dither_flat = dither_rng.reshape(-1).contiguous()
        if dither_flat.numel() != n_blocks:
            raise ValueError(
                f"dither_rng has {dither_flat.numel()} values; expected {n_blocks}."
            )
    else:
        # The constexpr branch does not dereference this pointer.
        dither_flat = shared_exp_flat

    out_blocks = torch.empty(
        (n_blocks, block_size), dtype=torch.float32, device=packed.device
    )
    _mx_dequantize_kernel[(n_blocks,)](
        packed_blocks,
        shared_exp_flat,
        out_blocks,
        BLOCK_SIZE=block_size,
        SUBTRACT_DITHER=round == "subtract_dither",
        dither_ptr=dither_flat,
        INT8=int8,
    )
    return _from_blocks(out_blocks, original_shape, padded_shape, inv_perm)


def block_dither(
    global_seed: int,
    n_blocks: int,
    tensor_id: int,
    state_id: int,
    step: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate block-ordered dither values for one state and step.

    The mixed seed identifies ``(global_seed, tensor_id, state_id, step)``.
    Reusing that tuple reproduces the same tensor on a fixed runtime.
    """
    mask = 0xFFFF_FFFF_FFFF_FFFF

    def mix64(value: int) -> int:
        value = (value + 0x9E37_79B9_7F4A_7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58_476D_1CE4_E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D0_49BB_1331_11EB) & mask
        return (value ^ (value >> 31)) & mask

    seed = mix64(int(global_seed) & mask)
    seed = mix64(seed ^ (int(tensor_id) & mask))
    seed = mix64(seed ^ ((int(state_id) & 0xFF) << 56))
    seed = mix64(seed ^ (int(step) & mask))
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return torch.rand(n_blocks, dtype=torch.float32, device=device, generator=gen)
