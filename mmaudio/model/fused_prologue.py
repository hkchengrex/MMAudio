"""Optional inference-only fusion for MMAudio's attention input preparation.

The kernel reads any number of packed projection streams, applies each stream's
Q/K RMSNorm and optional native RoPE matrices, and writes attention's contiguous
joint Q/K/V tensors directly. Triton is optional: unsupported calls return None
without allocating tensors, leaving the caller free to run its original path.
"""

import math
import re
from typing import Optional, Sequence

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def _has_supported_rms_norm() -> bool:
    # PyTorch 2.5 and 2.6 have different RMSNorm rounding and eps defaults.
    # Only stable 2.7+ versions use the FP32 arithmetic implemented below.
    version = re.fullmatch(r'2\.(\d+)\.\d+(?:\+[^\s]+)?', str(torch.__version__))
    return version is not None and int(version.group(1)) >= 7


_SUPPORTED_RMS_NORM = _has_supported_rms_norm()

if triton is not None:

    @triton.jit
    def _joint_qkv_norm_rope(
        projections,
        q_weights,
        k_weights,
        rotations,
        out_q,
        out_k,
        out_v,
        lengths: tl.constexpr,
        projection_strides: tl.constexpr,
        weight_strides: tl.constexpr,
        rotation_strides: tl.constexpr,
        has_rotation: tl.constexpr,
        q_eps: tl.constexpr,
        k_eps: tl.constexpr,
        NUM_STREAMS: tl.constexpr,
        HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        TOTAL_TOKENS: tl.constexpr,
        HEADS_PER_BLOCK: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        heads = tl.program_id(1).to(tl.int64) * HEADS_PER_BLOCK + tl.arange(0, HEADS_PER_BLOCK)
        batch = tl.program_id(2).to(tl.int64)
        dims = tl.arange(0, HEAD_DIM)
        pairs = tl.arange(0, HEAD_DIM // 2)
        head_mask = heads[:, None] < HEADS

        start = 0
        for stream in tl.static_range(NUM_STREAMS):
            end = start + lengths[stream]
            if (token >= start) & (token < end):
                local_token = token - start
                stride_b, stride_s, stride_c = projection_strides[stream]
                # Projection channels are (head, dimension, q/k/v), with q/k/v
                # interleaved in the innermost dimension, rather than chunked.
                offset = (batch * stride_b + local_token * stride_s +
                          (heads[:, None] * HEAD_DIM + dims[None, :]) * (3 * stride_c))
                q = tl.load(projections[stream] + offset, head_mask, other=0).to(tl.float32)
                k = tl.load(projections[stream] + offset + stride_c, head_mask,
                            other=0).to(tl.float32)
                v = tl.load(projections[stream] + offset + 2 * stride_c, head_mask, other=0)

                q_scale = tl.rsqrt(tl.sum(q * q, axis=1) / HEAD_DIM + q_eps[stream])
                k_scale = tl.rsqrt(tl.sum(k * k, axis=1) / HEAD_DIM + k_eps[stream])
                stride_wq, stride_wk = weight_strides[stream]
                wq = tl.load(q_weights[stream] + dims * stride_wq).to(tl.float32)
                wk = tl.load(k_weights[stream] + dims * stride_wk).to(tl.float32)
                # Native nn.RMSNorm in torch >= 2.7 rounds only after the weight
                # multiplication. apply_rope then explicitly lifts that result
                # back to FP32, so preserve this materialization boundary.
                q = ((q * q_scale[:, None]) * wq[None, :]).to(out_q.dtype.element_ty)
                k = ((k * k_scale[:, None]) * wk[None, :]).to(out_k.dtype.element_ty)

                if has_rotation[stream]:
                    _, stride_rs, stride_rp, stride_ri, stride_rj = rotation_strides[stream]
                    rotation_offset = local_token * stride_rs + pairs * stride_rp
                    r00 = tl.load(rotations[stream] + rotation_offset).to(tl.float32)
                    r01 = tl.load(rotations[stream] + rotation_offset + stride_rj).to(tl.float32)
                    r10 = tl.load(rotations[stream] + rotation_offset + stride_ri).to(tl.float32)
                    r11 = tl.load(rotations[stream] + rotation_offset + stride_ri + stride_rj).to(
                        tl.float32)
                    qe, qo = tl.split(q.to(tl.float32).reshape(HEADS_PER_BLOCK, HEAD_DIM // 2, 2))
                    ke, ko = tl.split(k.to(tl.float32).reshape(HEADS_PER_BLOCK, HEAD_DIM // 2, 2))
                    q_even = r00[None, :] * qe + r01[None, :] * qo
                    q_odd = r10[None, :] * qe + r11[None, :] * qo
                    k_even = r00[None, :] * ke + r01[None, :] * ko
                    k_odd = r10[None, :] * ke + r11[None, :] * ko
                    q = tl.join(q_even, q_odd).reshape(HEADS_PER_BLOCK, HEAD_DIM)
                    k = tl.join(k_even, k_odd).reshape(HEADS_PER_BLOCK, HEAD_DIM)

                output_offset = ((
                    (batch * HEADS + heads[:, None]) * TOTAL_TOKENS + token) * HEAD_DIM +
                                 dims[None, :])
                tl.store(out_q + output_offset, q, head_mask)
                tl.store(out_k + output_offset, k, head_mask)
                tl.store(out_v + output_offset, v, head_mask)
            start = end


def _resolve_eps(values: Optional[Sequence[Optional[float]]], count: int):
    if values is None:
        return (torch.finfo(torch.float32).eps, ) * count
    if len(values) != count:
        return None
    resolved = []
    for value in values:
        if value is None:
            value = torch.finfo(torch.float32).eps
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            return None
        resolved.append(float(value))
    return tuple(resolved)


def try_fused_joint_qkv_norm_rope(
    qkv_streams: Sequence[torch.Tensor],
    q_weights: Sequence[torch.Tensor],
    k_weights: Sequence[torch.Tensor],
    rotations: Sequence[Optional[torch.Tensor]],
    num_heads: int,
    *,
    q_eps: Optional[Sequence[Optional[float]]] = None,
    k_eps: Optional[Sequence[Optional[float]]] = None,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Prepare joint Q/K/V, or return None for the caller's ordinary path.

    Every projection is [B, S_i, H * D * 3], in MMAudio's interleaved
    ``(head, dimension, q/k/v)`` order. Weights are separate [D] vectors per
    stream. An optional rotation is MMAudio's [1, S_i, D / 2, 2, 2] matrix
    tensor, in FP32 or the activation dtype, used without packing or copying.
    Outputs are contiguous [B, H, sum(S_i), D], in the supplied stream order.

    Supports stable PyTorch 2.7+, CUDA BF16/FP16, and head dimensions 64/128.
    Gradients and torch.compile use the caller's ordinary path. Arithmetic
    boundaries match native RMSNorm and apply_rope; FP32 reduction order may
    still yield small differences from the native implementation.
    """
    if torch.compiler.is_compiling():
        return None
    if triton is None or not _SUPPORTED_RMS_NORM:
        return None
    count = len(qkv_streams)
    if count == 0 or len(q_weights) != count or len(k_weights) != count or len(rotations) != count:
        return None
    if not isinstance(num_heads, int) or num_heads <= 0:
        return None

    first = qkv_streams[0]
    if not isinstance(first, torch.Tensor) or first.layout != torch.strided or first.ndim != 3:
        return None
    if first.device.type != 'cuda' or first.dtype not in (torch.bfloat16, torch.float16):
        return None
    batch, _, channels = first.shape
    if batch <= 0 or batch > 65535 or num_heads > 4 * 65535:
        return None
    if channels % (num_heads * 3) != 0:
        return None
    head_dim = channels // (num_heads * 3)
    if head_dim not in (64, 128):
        return None

    resolved_q_eps = _resolve_eps(q_eps, count)
    resolved_k_eps = _resolve_eps(k_eps, count)
    if resolved_q_eps is None or resolved_k_eps is None:
        return None

    lengths = []
    projection_strides = []
    weight_strides = []
    rotation_strides = []
    has_rotation = []
    rotation_pointers = []
    for projection, wq, wk, rotation in zip(qkv_streams, q_weights, k_weights, rotations):
        for tensor in (projection, wq, wk):
            if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
                return None
            if tensor.device != first.device or tensor.dtype != first.dtype:
                return None
            if torch.is_grad_enabled() and tensor.requires_grad:
                return None
            if any(stride < 0 for stride in tensor.stride()):
                return None
        if projection.ndim != 3 or projection.shape[0] != batch or projection.shape[2] != channels:
            return None
        if wq.shape != (head_dim, ) or wk.shape != (head_dim, ):
            return None
        length = projection.shape[1]
        if rotation is not None:
            if not isinstance(rotation, torch.Tensor) or rotation.layout != torch.strided:
                return None
            if (rotation.device != first.device
                    or rotation.dtype not in (torch.float32, first.dtype)):
                return None
            if rotation.shape != (1, length, head_dim // 2, 2, 2):
                return None
            if torch.is_grad_enabled() and rotation.requires_grad:
                return None
            if any(stride < 0 for stride in rotation.stride()):
                return None
            rotation_strides.append(rotation.stride())
            rotation_pointers.append(rotation)
        else:
            # This pointer is never dereferenced in the constexpr no-RoPE arm.
            rotation_strides.append((0, 0, 0, 0, 0))
            rotation_pointers.append(projection)
        lengths.append(length)
        projection_strides.append(projection.stride())
        weight_strides.append((wq.stride(0), wk.stride(0)))
        has_rotation.append(rotation is not None)

    total_tokens = sum(lengths)
    if total_tokens == 0:
        return None
    shape = (batch, num_heads, total_tokens, head_dim)
    q = torch.empty(shape, dtype=first.dtype, device=first.device)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    heads_per_block = 4
    grid = (total_tokens, triton.cdiv(num_heads, heads_per_block), batch)
    with torch.cuda.device(first.device):
        _joint_qkv_norm_rope[grid](
            tuple(qkv_streams),
            tuple(q_weights),
            tuple(k_weights),
            tuple(rotation_pointers),
            q,
            k,
            v,
            tuple(lengths),
            tuple(projection_strides),
            tuple(weight_strides),
            tuple(rotation_strides),
            tuple(has_rotation),
            resolved_q_eps,
            resolved_k_eps,
            count,
            num_heads,
            head_dim,
            total_tokens,
            heads_per_block,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q, k, v
