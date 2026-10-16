"""Correctness and fallback coverage for MMAudio's optional inference prologue.

CUDA tests exercise the native packed [B, S, H * D * 3] projection layout;
CPU tests require neither Triton nor a CUDA installation.
"""

import copy

import pytest
import torch
from torch import nn

from mmaudio.ext.rotary_embeddings import apply_rope, compute_rope_rotations
from mmaudio.model import fused_prologue, transformer_layers
from mmaudio.model.fused_prologue import try_fused_joint_qkv_norm_rope
from mmaudio.model.transformer_layers import JointBlock, attention


def make_inputs(device='cpu',
                dtype=torch.float32,
                batch=2,
                heads=7,
                dim=64,
                lengths=(9, 5, 7),
                strided=False):
    torch.manual_seed(127)
    qkv = []
    q_weights, k_weights, rotations = [], [], []
    for i, length in enumerate(lengths):
        projection = torch.randn(batch,
                                 length,
                                 heads * dim * 3 * (2 if strided else 1),
                                 device=device,
                                 dtype=dtype)
        if strided:
            projection = projection[..., ::2]
        qkv.append(projection)
        # Unequal and nonconstant weights expose accidental sharing between streams.
        q_weights.append((torch.randn(dim, device=device) * 0.2 + 0.8 + i * 0.3).to(dtype))
        k_weights.append((torch.randn(dim, device=device) * 0.2 + 1.4 - i * 0.2).to(dtype))
        rotations.append(
            compute_rope_rotations(length, dim, 10000, freq_scaling=1.0 +
                                   i * 0.5, device=device) if i != 2 else None)
    return qkv, q_weights, k_weights, rotations


def native_reference(qkv, q_weights, k_weights, rotations, heads, q_eps=None, k_eps=None):
    """The actual eager chain: nn.RMSNorm, native matrix RoPE, then three cats."""
    result = [[], [], []]
    for i, projection in enumerate(qkv):
        batch, length, channels = projection.shape
        dim = channels // (heads * 3)
        packed = projection.reshape(batch, length, heads, dim, 3).permute(0, 2, 1, 3, 4)
        q, k, v = packed.unbind(-1)
        q_norm = nn.RMSNorm(dim,
                            eps=None if q_eps is None else q_eps[i],
                            device=projection.device,
                            dtype=projection.dtype)
        k_norm = nn.RMSNorm(dim,
                            eps=None if k_eps is None else k_eps[i],
                            device=projection.device,
                            dtype=projection.dtype)
        # Keep autograd relationships to the supplied weights in gradient tests.
        q = torch.nn.functional.rms_norm(q, (dim, ), q_weights[i], q_norm.eps)
        k = torch.nn.functional.rms_norm(k, (dim, ), k_weights[i], k_norm.eps)
        if rotations[i] is not None:
            q, k = apply_rope(q, rotations[i]), apply_rope(k, rotations[i])
        for destination, value in zip(result, (q, k, v)):
            destination.append(value)
    return tuple(torch.cat(values, dim=2) for values in result)


def original_joint_forward(block, latent, clip, text, global_c, extended_c, latent_rot, clip_rot):
    """Original JointBlock math, independent of the optimized dispatch."""
    x_qkv, x_mod = block.latent_block.pre_attention(latent, extended_c, latent_rot)
    c_qkv, c_mod = block.clip_block.pre_attention(clip, global_c, clip_rot)
    t_qkv, t_mod = block.text_block.pre_attention(text, global_c, None)
    joint = [torch.cat([x_qkv[i], c_qkv[i], t_qkv[i]], dim=2) for i in range(3)]
    out = attention(*joint)
    latent_len, clip_len = latent.shape[1], clip.shape[1]
    latent = block.latent_block.post_attention(latent, out[:, :latent_len], x_mod)
    if not block.pre_only:
        clip = block.clip_block.post_attention(clip, out[:, latent_len:latent_len + clip_len],
                                               c_mod)
        text = block.text_block.post_attention(text, out[:, latent_len + clip_len:], t_mod)
    return latent, clip, text


def make_block_inputs(device='cpu', dtype=torch.float32, batch=2, heads=1, pre_only=False):
    torch.manual_seed(37)
    dim = heads * 64
    block = JointBlock(dim, heads, mlp_ratio=1, pre_only=pre_only).to(device=device, dtype=dtype)
    block.use_fused_prologue = True
    lengths = (9, 5, 7)
    args = [torch.randn(batch, length, dim, device=device, dtype=dtype) for length in lengths]
    args.extend([
        torch.randn(batch, 1, dim, device=device, dtype=dtype),
        torch.randn(batch, lengths[0], dim, device=device, dtype=dtype)
    ])
    args.extend([
        compute_rope_rotations(lengths[0], 64, 10000, device=device),
        compute_rope_rotations(lengths[1], 64, 10000, freq_scaling=1.8, device=device)
    ])
    return block, args


def assert_tensors_equal(actual, expected):
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_cpu_declines_without_changing_inputs():
    inputs = make_inputs()
    before = [[value.clone() if value is not None else None for value in group] for group in inputs]
    with torch.no_grad():
        assert try_fused_joint_qkv_norm_rope(*inputs, 7) is None
    for group, saved in zip(inputs, before):
        for actual, expected in zip(group, saved):
            if actual is not None:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_joint_block_requires_opt_in():
    assert not JointBlock(64, 1).use_fused_prologue


def test_disabled_joint_block_preserves_original_path(monkeypatch):
    block, args = make_block_inputs(batch=1)
    block.eval()
    block.use_fused_prologue = False

    def forbidden(*args, **kwargs):
        raise AssertionError('disabled block attempted fusion')

    monkeypatch.setattr(transformer_layers, 'try_fused_joint_qkv_norm_rope', forbidden)
    with torch.no_grad():
        assert_tensors_equal(block(*args), original_joint_forward(block, *args))


@pytest.mark.parametrize('pre_only', [False, True])
def test_cpu_eval_matches_original_joint_block(pre_only):
    block, args = make_block_inputs(pre_only=pre_only)
    block.eval()
    with torch.no_grad():
        assert_tensors_equal(block(*args), original_joint_forward(block, *args))


@pytest.mark.parametrize('training', [False, True])
def test_fallback_preserves_input_and_parameter_gradients(training):
    block, args = make_block_inputs(batch=1)
    block.train(training)
    reference = copy.deepcopy(block)
    test_args = [value.detach().clone().requires_grad_(i < 5) for i, value in enumerate(args)]
    ref_args = [value.detach().clone().requires_grad_(i < 5) for i, value in enumerate(args)]
    actual, expected = block(*test_args), original_joint_forward(reference, *ref_args)
    assert_tensors_equal(actual, expected)
    sum(value.square().mean() for value in actual).backward()
    sum(value.square().mean() for value in expected).backward()
    for test_input, reference_input in zip(test_args[:5], ref_args[:5]):
        torch.testing.assert_close(test_input.grad, reference_input.grad, rtol=0, atol=0)
    for (name, parameter), (ref_name, ref_parameter) in zip(block.named_parameters(),
                                                            reference.named_parameters()):
        assert name == ref_name
        if parameter.grad is None or ref_parameter.grad is None:
            assert parameter.grad is ref_parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, ref_parameter.grad, rtol=0, atol=0)


def test_training_never_attempts_inference_fusion(monkeypatch):
    block, args = make_block_inputs(batch=1)
    block.train()

    def forbidden(*args, **kwargs):
        raise AssertionError('training attempted inference-only fusion')

    monkeypatch.setattr(transformer_layers, 'try_fused_joint_qkv_norm_rope', forbidden)
    with torch.no_grad():
        assert_tensors_equal(block(*args), original_joint_forward(block, *args))


def test_compile_falls_back_without_graph_break():
    inputs = make_inputs(batch=1, heads=1)

    def run(projection):
        fused = try_fused_joint_qkv_norm_rope([projection], [inputs[1][0]], [inputs[2][0]],
                                              [inputs[3][0]], 1)
        if fused is None:
            return projection + 1
        return fused[0]

    compiled = torch.compile(run, backend='eager', fullgraph=True)
    torch.testing.assert_close(compiled(inputs[0][0]), inputs[0][0] + 1, rtol=0, atol=0)


def test_compiled_joint_block_matches_original_fallback():
    block, args = make_block_inputs(batch=1)
    block.eval()
    compiled = torch.compile(block, backend='eager', fullgraph=True)
    with torch.no_grad():
        assert_tensors_equal(compiled(*args), original_joint_forward(block, *args))


@pytest.fixture
def cuda_device():
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required for the fused kernel')
    pytest.importorskip('triton')
    if not fused_prologue._SUPPORTED_RMS_NORM:
        pytest.skip('fusion requires stable PyTorch 2.7+ native RMSNorm semantics')
    return torch.device('cuda')


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('batch,heads,dim,lengths,strided', [
    (1, 7, 64, (250, 64, 77), False),
    (2, 14, 64, (345, 64, 77), False),
    (4, 7, 64, (9, 5, 7), True),
    (2, 3, 128, (9, 5, 7), False),
])
def test_three_stream_production_layout(cuda_device, dtype, batch, heads, dim, lengths, strided):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('device does not support BF16')
    inputs = make_inputs(cuda_device, dtype, batch, heads, dim, lengths=lengths, strided=strided)
    before = [[value.clone() if value is not None else None for value in group] for group in inputs]
    with torch.no_grad():
        actual = try_fused_joint_qkv_norm_rope(*inputs, heads)
        assert actual is not None, 'supported input silently fell back'
        expected = native_reference(*inputs, heads)
        lifted = [[value.float() if value is not None else None for value in group]
                  for group in inputs]
        fp32_reference = native_reference(*lifted, heads)
    atol, rtol = (0.035, 0.015) if dtype == torch.bfloat16 else (0.004, 0.004)
    for result, reference, accurate in zip(actual[:2], expected[:2], fp32_reference[:2]):
        assert result.shape == (batch, heads, sum(lengths), dim)
        assert result.is_contiguous()
        torch.testing.assert_close(result, reference, atol=atol, rtol=rtol)
        fused_rmse = (result.float() - accurate).square().mean().sqrt()
        native_rmse = (reference.float() - accurate).square().mean().sqrt()
        assert fused_rmse <= native_rmse * 1.1 + 1e-5
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    assert actual[2].is_contiguous()
    # The text stream is normalized but must not receive either visual RoPE table.
    for result, reference in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(result[:, :, -lengths[2]:],
                                   reference[:, :, -lengths[2]:],
                                   atol=atol,
                                   rtol=rtol)
    for group, saved in zip(inputs, before):
        for value, snapshot in zip(group, saved):
            if value is not None:
                torch.testing.assert_close(value, snapshot, rtol=0, atol=0)


@pytest.mark.parametrize('stream_count', [1, 2, 4])
def test_stream_count_is_not_hardcoded(cuda_device, stream_count):
    inputs = make_inputs(cuda_device,
                         torch.float16,
                         heads=1,
                         lengths=tuple(range(3, 3 + stream_count)))
    with torch.no_grad():
        actual = try_fused_joint_qkv_norm_rope(*inputs, 1)
        assert actual is not None
        expected = native_reference(*inputs, 1)
    for result, reference in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(result, reference, rtol=0.004, atol=0.004)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_per_stream_epsilon(cuda_device):
    inputs = list(make_inputs(cuda_device, torch.bfloat16, heads=1))
    q_eps, k_eps = [None, 1e-4, 0.125], [1e-3, None, 0.25]
    with torch.no_grad():
        actual = try_fused_joint_qkv_norm_rope(*inputs, 1, q_eps=q_eps, k_eps=k_eps)
        assert actual is not None
        expected = native_reference(*inputs, 1, q_eps=q_eps, k_eps=k_eps)
    for result, reference in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(result, reference, rtol=0.015, atol=0.035)


def test_rotation_buffers_cast_by_model_to(cuda_device):
    inputs = list(make_inputs(cuda_device, torch.bfloat16, heads=1))
    inputs[3] = [rotation.bfloat16() if rotation is not None else None for rotation in inputs[3]]
    with torch.no_grad():
        actual = try_fused_joint_qkv_norm_rope(*inputs, 1)
        assert actual is not None
        expected = native_reference(*inputs, 1)
    for result, reference in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(result, reference, rtol=0.015, atol=0.035)


@pytest.mark.parametrize('amplitude', [0.0, 1e-4, 100.0])
def test_zero_small_and_large_inputs(cuda_device, amplitude):
    inputs = list(make_inputs(cuda_device, torch.float16, heads=1))
    inputs[0] = [projection * amplitude for projection in inputs[0]]
    with torch.no_grad():
        actual = try_fused_joint_qkv_norm_rope(*inputs, 1)
        assert actual is not None
        expected = native_reference(*inputs, 1)
    for result, reference in zip(actual[:2], expected[:2]):
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, reference, rtol=0.004, atol=0.004)
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


@pytest.mark.parametrize('requires_grad', ['projection', 'weight', 'rotation'])
def test_cuda_grad_enabled_declines(cuda_device, requires_grad):
    inputs = make_inputs(cuda_device, torch.float16, heads=1)
    group = {'projection': 0, 'weight': 1, 'rotation': 3}[requires_grad]
    inputs[group][0].requires_grad_(True)
    assert try_fused_joint_qkv_norm_rope(*inputs, 1) is None


@pytest.mark.parametrize(
    'invalid', ['fp32', 'head_dimension', 'rotation_shape', 'weight_count', 'weight_dtype'])
def test_unsupported_cuda_inputs_decline(cuda_device, invalid):
    inputs = list(
        make_inputs(cuda_device,
                    torch.float16,
                    heads=1,
                    dim=48 if invalid == 'head_dimension' else 64))
    if invalid == 'fp32':
        inputs[:3] = [[value.float() for value in group] for group in inputs[:3]]
    elif invalid == 'rotation_shape':
        inputs[3][0] = inputs[3][0][..., :1]
    elif invalid == 'weight_count':
        inputs[1] = inputs[1][:-1]
    elif invalid == 'weight_dtype':
        inputs[1] = [weight.float() for weight in inputs[1]]
    with torch.no_grad():
        assert try_fused_joint_qkv_norm_rope(*inputs, 1) is None


def test_cuda_graph_replay_is_bitwise_repeatable(cuda_device):
    inputs = make_inputs(cuda_device, torch.float16, batch=2, heads=7)
    with torch.no_grad():
        expected = try_fused_joint_qkv_norm_rope(*inputs, 7)
        assert expected is not None
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = try_fused_joint_qkv_norm_rope(*inputs, 7)
        for _ in range(3):
            graph.replay()
            for result, reference in zip(actual, expected):
                assert torch.equal(result.contiguous().view(torch.uint8),
                                   reference.contiguous().view(torch.uint8))


@pytest.mark.parametrize('pre_only', [False, True])
def test_cuda_joint_block_uses_fusion(cuda_device, monkeypatch, pre_only):
    block, args = make_block_inputs(cuda_device, torch.bfloat16, heads=7, pre_only=pre_only)
    block.eval()
    called = []
    original = transformer_layers.try_fused_joint_qkv_norm_rope

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        called.append(result is not None)
        return result

    monkeypatch.setattr(transformer_layers, 'try_fused_joint_qkv_norm_rope', record)
    with torch.no_grad():
        actual = block(*args)
        expected = original_joint_forward(block, *args)
    assert called == [True]
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0.04, atol=0.04)
