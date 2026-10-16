"""Reproduce operator, JointBlock, and optional random-weight network timings.

Run from the repository root, for example:
  python benchmarks/benchmark_fused_prologue.py --output prologue.json
  python benchmarks/benchmark_fused_prologue.py --levels operator block network \
      --models small_16k large_44k_v2 --batches 1 2 --output full-forward.json

Network results time predict_flow with cached random conditioning features. They
are denoiser-forward measurements, not pretrained quality or generation latency.
Batches are literal model batches: upstream CFG invokes the two branches separately.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from unittest.mock import patch

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mmaudio.ext.rotary_embeddings import apply_rope, compute_rope_rotations
from mmaudio.model import transformer_layers
from mmaudio.model.fused_prologue import try_fused_joint_qkv_norm_rope
from mmaudio.model.networks import PreprocessedConditions, get_my_mmaudio
from mmaudio.model.sequence_config import CONFIG_16K, CONFIG_44K, SequenceConfig
from mmaudio.model.transformer_layers import JointBlock


def metadata():

    def package(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    def git(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=REPO_ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    paths = [
        'mmaudio/model/fused_prologue.py', 'mmaudio/model/transformer_layers.py',
        'benchmarks/benchmark_fused_prologue.py'
    ]
    return {
        'python': platform.python_version(),
        'torch': torch.__version__,
        'triton': package('triton'),
        'cuda_runtime': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(),
        'compute_capability': list(torch.cuda.get_device_capability()),
        'git_head': git('rev-parse', 'HEAD'),
        'git_status': git('status', '--porcelain'),
        'source_sha256': {
            path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
            for path in paths
        },
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'sdpa_flash_enabled': torch.backends.cuda.flash_sdp_enabled(),
        'sdpa_mem_efficient_enabled': torch.backends.cuda.mem_efficient_sdp_enabled(),
        'sdpa_math_enabled': torch.backends.cuda.math_sdp_enabled(),
    }


def agreement(actual, expected):
    if isinstance(actual, torch.Tensor):
        actual, expected = (actual, ), (expected, )
    result = []
    for left, right in zip(actual, expected):
        delta = left.float() - right.float()
        different = left.contiguous().view(torch.uint8) != right.contiguous().view(torch.uint8)
        result.append({
            'shape': list(left.shape),
            'dtype': str(left.dtype),
            'max_abs': delta.abs().max().item(),
            'rmse': delta.square().mean().sqrt().item(),
            'differing_elements': (left != right).count_nonzero().item(),
            'elements': left.numel(),
            'differing_bytes': different.count_nonzero().item(),
            'contiguous': left.is_contiguous(),
        })
    return result


def time_calls(function, iterations):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def measure(arms, select, args, iterations):
    """Selection happens outside timing; each ABBA round yields two samples per arm."""
    output = {}
    for mode in args.timing:
        functions, graphs = {}, {}
        for name, function in arms.items():
            select(name)
            for _ in range(args.warmup):
                function()
            torch.cuda.synchronize()
            if mode == 'graph':
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured_output = function()
                # Keep the capture output alive for the lifetime of its graph.
                graphs[name] = (graph, captured_output)
                functions[name] = graph.replay
            else:
                functions[name] = function
        samples = {name: [] for name in arms}
        order = []
        names = list(arms)
        for _ in range(args.rounds):
            for name in (names[0], names[1], names[1], names[0]):
                select(name)
                samples[name].append(time_calls(functions[name], iterations))
                order.append(name)
        medians = {name: statistics.median(values) for name, values in samples.items()}
        output[mode] = {
            'median_ms': medians,
            'raw_samples_ms': samples,
            'order': order,
            'calls_per_sample': iterations,
            'speedup': medians['native'] / medians['fused'],
            'latency_reduction_percent': 100 * (1 - medians['fused'] / medians['native'])
        }
    return output


def operator_case(batch, heads, lengths, dtype, args):
    dim = 64
    projections = [
        torch.randn(batch, length, heads * dim * 3, device='cuda', dtype=dtype)
        for length in lengths
    ]
    q_norms = nn.ModuleList([nn.RMSNorm(dim, device='cuda', dtype=dtype) for _ in lengths])
    k_norms = nn.ModuleList([nn.RMSNorm(dim, device='cuda', dtype=dtype) for _ in lengths])
    for i, (q_norm, k_norm) in enumerate(zip(q_norms, k_norms)):
        q_norm.weight.copy_(torch.randn_like(q_norm.weight) * 0.2 + 1.0 + i * 0.1)
        k_norm.weight.copy_(torch.randn_like(k_norm.weight) * 0.2 + 1.2 - i * 0.1)
    rotations = [
        compute_rope_rotations(lengths[0], dim, 10000, device='cuda'),
        compute_rope_rotations(lengths[1],
                               dim,
                               10000,
                               freq_scaling=lengths[0] / lengths[1],
                               device='cuda'), None
    ]

    def native():
        streams = []
        for projection, q_norm, k_norm, rotation in zip(projections, q_norms, k_norms, rotations):
            packed = projection.reshape(batch, projection.shape[1], heads, dim, 3)
            q, k, v = packed.permute(0, 2, 1, 3, 4).unbind(-1)
            q, k = q_norm(q), k_norm(k)
            if rotation is not None:
                q, k = apply_rope(q, rotation), apply_rope(k, rotation)
            streams.append((q, k, v))
        return tuple(torch.cat([stream[i] for stream in streams], dim=2) for i in range(3))

    def fused():
        result = try_fused_joint_qkv_norm_rope(projections, [norm.weight for norm in q_norms],
                                               [norm.weight for norm in k_norms], rotations, heads)
        if result is None:
            raise RuntimeError('fused operator declined the benchmark input')
        return result

    actual, expected = fused(), native()
    return {
        'level': 'operator',
        'batch': batch,
        'heads': heads,
        'head_dim': dim,
        'lengths_audio_video_text': lengths,
        'actual_fused_calls': 1,
        'agreement': agreement(actual, expected),
        'timings': measure({
            'native': native,
            'fused': fused
        }, lambda name: None, args, args.iterations)
    }


def count_dispatch(function):
    original = transformer_layers.try_fused_joint_qkv_norm_rope
    counts = {'attempted': 0, 'fused': 0, 'declined': 0}

    def record(*args, **kwargs):
        counts['attempted'] += 1
        result = original(*args, **kwargs)
        counts['fused' if result is not None else 'declined'] += 1
        return result

    with patch.object(transformer_layers, 'try_fused_joint_qkv_norm_rope', record):
        output = function()
    return counts, output


def model_case(level, model_name, batch, lengths, dtype, args):
    heads = 7 if model_name.startswith('small') else 14
    hidden = heads * 64
    if level == 'block':
        model = JointBlock(hidden, heads).cuda().to(dtype).eval()
        values = [
            torch.randn(batch, length, hidden, device='cuda', dtype=dtype) for length in lengths
        ]
        values.extend([
            torch.randn(batch, 1, hidden, device='cuda', dtype=dtype),
            torch.randn(batch, lengths[0], hidden, device='cuda', dtype=dtype),
            compute_rope_rotations(lengths[0], 64, 10000, device='cuda'),
            compute_rope_rotations(lengths[1],
                                   64,
                                   10000,
                                   freq_scaling=lengths[0] / lengths[1],
                                   device='cuda')
        ])

        def select(name):
            model.use_fused_prologue = name == 'fused'

        def run():
            return model(*values)

        iterations = args.block_iterations
        expected_calls = 1
    else:
        model = get_my_mmaudio(model_name).cuda().to(dtype).eval()
        # The training initializer zeros every residual gate and the output
        # convolution. Give these random nonzero values so output comparisons
        # exercise attention rather than comparing two constant-zero tensors.
        for block in model.joint_blocks:
            for stream in (block.latent_block, block.clip_block, block.text_block):
                nn.init.normal_(stream.adaLN_modulation[-1].weight, std=0.02)
                nn.init.normal_(stream.adaLN_modulation[-1].bias, std=0.02)
        for block in model.fused_blocks:
            nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)
            nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.02)
        nn.init.normal_(model.final_layer.conv.weight, std=0.02)
        nn.init.normal_(model.final_layer.conv.bias, std=0.02)
        # Native code intentionally stores rotation matrices in FP32.
        sync_len = args.sequence_config.sync_seq_len
        model.update_seq_lengths(lengths[0], lengths[1], sync_len)
        latent = torch.randn(batch, lengths[0], model.latent_dim, device='cuda', dtype=dtype)
        time = torch.full((batch, ), 0.5, device='cuda', dtype=dtype)
        conditions = PreprocessedConditions(clip_f=torch.randn(batch,
                                                               lengths[1],
                                                               hidden,
                                                               device='cuda',
                                                               dtype=dtype),
                                            sync_f=torch.randn(batch,
                                                               lengths[0],
                                                               hidden,
                                                               device='cuda',
                                                               dtype=dtype),
                                            text_f=torch.randn(batch,
                                                               lengths[2],
                                                               hidden,
                                                               device='cuda',
                                                               dtype=dtype),
                                            clip_f_c=torch.randn(batch,
                                                                 hidden,
                                                                 device='cuda',
                                                                 dtype=dtype),
                                            text_f_c=torch.randn(batch,
                                                                 hidden,
                                                                 device='cuda',
                                                                 dtype=dtype))

        def select(name):
            model.enable_fused_prologue(name == 'fused')

        def run():
            return model.predict_flow(latent, time, conditions)

        iterations = args.network_iterations
        expected_calls = len(model.joint_blocks)
    select('native')
    expected = run()
    select('fused')
    dispatch, actual = count_dispatch(run)
    if dispatch['fused'] != expected_calls or dispatch['declined']:
        raise RuntimeError(f'expected {expected_calls} fused calls, observed {dispatch}')
    return {
        'level': level,
        'model_geometry': model_name,
        'weights': 'random',
        'random_initialization': 'nonzero residual gates and output convolution',
        'batch': batch,
        'heads': heads,
        'head_dim': 64,
        'lengths_audio_video_text': lengths,
        'actual_dispatch': dispatch,
        'agreement': agreement(actual, expected),
        'timings': measure({
            'native': run,
            'fused': run
        }, select, args, iterations)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--models',
        nargs='+',
        default=['small_16k', 'medium_44k'],
        choices=['small_16k', 'small_44k', 'medium_44k', 'large_44k', 'large_44k_v2'])
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 2, 4])
    parser.add_argument('--duration', type=float, default=8.0)
    parser.add_argument('--levels',
                        nargs='+',
                        choices=['operator', 'block', 'network'],
                        default=['operator', 'block'])
    parser.add_argument('--timing',
                        nargs='+',
                        choices=['eager', 'graph'],
                        default=['eager', 'graph'])
    parser.add_argument('--dtype', choices=['bf16', 'fp16'], default='bf16')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--block-iterations', type=int, default=20)
    parser.add_argument('--network-iterations', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('a CUDA GPU and the optional Triton dependency are required')
    if min(
            args.batches +
        [args.iterations, args.block_iterations, args.network_iterations, args.rounds, args.warmup]
    ) <= 0 or args.duration <= 0:
        parser.error('batches, counts and duration must be positive')
    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float16
    torch.manual_seed(174)
    torch.backends.cudnn.benchmark = False
    result = {
        'environment': metadata(),
        'method': {
            'order': 'ABBA repeated; six samples per arm by default',
            'timing': 'CUDA events; per-call ms; optional captured graph replay',
            'network': 'random-weight predict_flow, cached synthetic conditions',
            'batch': 'literal batch; upstream CFG calls both branches separately',
            'warmup_calls_per_arm': args.warmup
        },
        'records': []
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for model_name in args.models:
            original = CONFIG_16K if model_name.endswith('16k') else CONFIG_44K
            args.sequence_config = SequenceConfig(
                duration=args.duration,
                sampling_rate=original.sampling_rate,
                spectrogram_frame_rate=original.spectrogram_frame_rate)
            lengths = [args.sequence_config.latent_seq_len, args.sequence_config.clip_seq_len, 77]
            heads = 7 if model_name.startswith('small') else 14
            for batch in args.batches:
                for level in args.levels:
                    if level == 'operator':
                        record = operator_case(batch, heads, lengths, dtype, args)
                    else:
                        record = model_case(level, model_name, batch, lengths, dtype, args)
                    result['records'].append(record)
                    args.output.write_text(json.dumps(result, indent=2) + '\n')
                    for mode, timing in record['timings'].items():
                        print(
                            f'{model_name} {level} B={batch} {mode}: '
                            f"{timing['median_ms']['native']:.4f} -> "
                            f"{timing['median_ms']['fused']:.4f} ms "
                            f"({timing['latency_reduction_percent']:+.2f}% reduction)",
                            flush=True)
                    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
