import argparse
import os
import random
import sys
import torch
import torch.distributed as dist
from typing import Tuple

import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
from deep_gemm.testing import bench_kineto


def import_baseline():
    # Load legacy implements from third-party
    deep_ep, tilelang_ops, do_bench, is_legacy_loaded = None, None, None, False
    # noinspection PyBroadException
    try:
        import deep_ep
        import importlib.util
        from tilelang.profiler.bench import do_bench
        spec = importlib.util.spec_from_file_location(
            'tilelang_ops',
            os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'third-party', 'tilelang_ops', '__init__.py'))
        tilelang_ops = importlib.util.module_from_spec(spec)
        sys.modules['tilelang_ops'] = tilelang_ops
        spec.loader.exec_module(tilelang_ops)
        is_legacy_loaded = True
    except Exception as ex:
        dist_print(f'Failed to load legacy code: {ex}, skip baseline benchmarking', once_in_node=True)
        dist_print(once_in_node=True)
    return deep_ep, tilelang_ops, do_bench, is_legacy_loaded


# TODO: skip the test for SM90
# noinspection PyUnboundLocalVariable,PyShadowingNames
def test(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(rank_idx)
    random.seed(rank_idx)

    # Settings
    num_max_tokens_per_rank = args.num_max_tokens_per_rank
    num_tokens = max(0, args.num_max_tokens_per_rank - random.randint(0, args.num_max_removed_tokens)) \
        if args.num_tokens == 0 else args.num_tokens
    hidden, intermediate_hidden = args.hidden, args.intermediate_hidden
    num_experts, num_topk = args.num_experts, args.num_topk
    num_experts_per_rank = num_experts // num_ranks
    assert num_tokens <= num_max_tokens_per_rank

    # Allocate symmetric memory
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group, num_experts,
        num_max_tokens_per_rank, num_topk,
        hidden, intermediate_hidden
    )

    # Create inputs
    # noinspection PyGlobalUndefined
    def create_inputs():
        global x, topk_idx, topk_weights, l1_weights, l2_weights, transformed_l1_weights, transformed_l2_weights
        global cumulative_local_expert_recv_stats_fused
        global cumulative_local_expert_recv_stats_baseline
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        l1_weights = torch.randn(
            (num_experts_per_rank, intermediate_hidden * 2, hidden), dtype=torch.bfloat16, device='cuda')
        l2_weights = torch.randn(
            (num_experts_per_rank, hidden, intermediate_hidden), dtype=torch.bfloat16, device='cuda')
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
        cumulative_local_expert_recv_stats_fused = torch.randint(
            0, 100, (num_experts_per_rank, ), dtype=torch.int, device='cuda')
        cumulative_local_expert_recv_stats_baseline = cumulative_local_expert_recv_stats_fused.clone()
        if args.masked_ratio > 0:
            rand_mask = torch.rand_like(topk_idx, dtype=torch.float)
            topk_idx.masked_fill_(rand_mask < args.masked_ratio, -1)
            topk_weights.masked_fill_(topk_idx < 0, 0)

        # Check SF requirements
        assert hidden % 128 == 0
        assert intermediate_hidden % 128 == 0
        assert l1_weights.shape[2] % 128 == 0 and l2_weights.shape[2] % 128 == 0

        # Cast inputs to FP8 (or FP4 under DG_USE_FP4_ACTS) with per-32 UE8M0 SF.
        # Stream A0.0b: when the flag is on, the symm buffer's `x` slot is sized
        # for packed E2M1 (`hidden/2` bytes/token), so we must quantize at the
        # source to match.
        if os.environ.get('DG_USE_FP4_ACTS', '0') != '0':
            x = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        else:
            x = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)

        # Cast grouped BF16 weights to FP4 with MN-major SF
        # TODO: merge with `cast_fp8_fp4_with_major`
        def cast_grouped_weights_to_fp4(bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            num_groups, n, k = bf16_weights.shape
            w = torch.empty((num_groups, n, k // 2), device='cuda', dtype=torch.int8)
            w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
            for i in range(num_groups):
                w[i], w_sf[i] = per_token_cast_to_fp4(bf16_weights[i], use_ue8m0=True, gran_k=32)
            w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
            return w, w_sf

        # Cast grouped BF16 weights to FP8 (e4m3) with MN-major UE8M0 SF, per-32 along K.
        # SF shape and packing are identical to the FP4 path (see support_w8a8_qa.md Q1).
        # NOTE: paddle compat 没有为 float8_e4m3fn 注册 `set_value_with_tensor`，所以
        # 不能直接 `w[i] = fp8_tensor`。先在 uint8 视图上做赋值（每个 e4m3 元素 = 1 byte），
        # 最后一次性 view 回 float8_e4m3fn。
        def cast_grouped_weights_to_fp8(bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            num_groups, n, k = bf16_weights.shape
            w_u8 = torch.empty((num_groups, n, k), device='cuda', dtype=torch.uint8)
            w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
            for i in range(num_groups):
                w_i_fp8, w_sf[i] = per_token_cast_to_fp8(bf16_weights[i], use_ue8m0=True, gran_k=32)
                w_u8[i] = w_i_fp8.view(torch.uint8)
            w = w_u8.contiguous().view(torch.float8_e4m3fn)
            w_sf = w_sf.contiguous()
            w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
            return w, w_sf

        # Cast grouped BF16 weights to FP8 with blockwise 128 quantization along K.
        # Quantizes with gran_k=128, then repeat_interleave(4) on SF so that the
        # SF shape matches per-32 layout (128/32=4 slots share the same SF value).
        def cast_grouped_weights_to_fp8_blockwise128(bf16_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            num_groups, n, k = bf16_weights.shape
            w_u8 = torch.empty((num_groups, n, k), device='cuda', dtype=torch.uint8)
            w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
            for i in range(num_groups):
                w_i_fp8, w_sf_128 = per_token_cast_to_fp8(bf16_weights[i], use_ue8m0=True, gran_k=128)
                w_u8[i] = w_i_fp8.view(torch.uint8)
                # repeat_interleave 4x: per-128 SF -> per-32 SF shape
                w_sf[i] = w_sf_128.repeat_interleave(4, dim=-1)
            w = w_u8.contiguous().view(torch.float8_e4m3fn)
            w_sf = w_sf.contiguous()
            w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
            return w, w_sf

        use_blockwise_128 = os.environ.get('DG_MEGA_MOE_USE_BLOCK_WISE_FP8', '0') != '0'
        if args.weight_dtype == 'fp8' and use_blockwise_128:
            cast_fn = cast_grouped_weights_to_fp8_blockwise128
        elif args.weight_dtype == 'fp8':
            cast_fn = cast_grouped_weights_to_fp8
        else:
            cast_fn = cast_grouped_weights_to_fp4
        l1_weights = cast_fn(l1_weights)
        l2_weights = cast_fn(l2_weights)
        transformed_l1_weights, transformed_l2_weights = deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights)

    # Run fused mega MoE
    # NOTES: copy x into buffer before each call because debug mode zeros the entire buffer
    # NOTES: copy x into buffer before each call because debug mode zeros the entire buffer
    if args.weight_dtype == 'fp8':
        fused_kernel = deep_gemm.fp8_fp4_mega_moe
    else:
        fused_kernel = deep_gemm.fp8_fp4_mega_moe
    trace_mega_moe = os.environ.get('DG_MEGA_MOE_TRACE', '0') != '0'
    def trace(message: str):
        if trace_mega_moe:
            print(f'[rank {rank_idx}] {message}', flush=True)

    def run_fused():
        trace('run_fused: copy inputs start')
        buffer.x[:num_tokens].copy_(x[0].view(buffer.x.dtype))
        buffer.x_sf[:num_tokens].copy_(x[1])
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)
        trace('run_fused: copy inputs done')

        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        # noinspection PyTypeChecker
        trace('run_fused: fused_kernel launch start')
        fused_kernel(
            y,
            transformed_l1_weights, transformed_l2_weights,
            buffer,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_fused,
            activation_clamp=args.activation_clamp,
            fast_math=bool(args.fast_math)
        )
        trace('run_fused: fused_kernel launch returned')
        if trace_mega_moe:
            torch.cuda.synchronize()
            trace('run_fused: cuda synchronize done')
        return y, cumulative_local_expert_recv_stats_fused

    dist_print('Config:', once_in_node=True)
    dist_print(f' > Tokens: {num_tokens}/{num_max_tokens_per_rank}', once_in_node=True)
    dist_print(f' > Hidden: {hidden}', once_in_node=True)
    dist_print(f' > Intermediate: {intermediate_hidden}', once_in_node=True)
    dist_print(f' > Experts: {num_topk}/{num_experts}', once_in_node=True)
    dist_print(f' > Buffer: {buffer.buffer.nbytes / 2 ** 30:.3f} GiB', once_in_node=True)
    dist_print(once_in_node=True)

    # Only do NCU profiling
    if args.ncu_profile_only:
        create_inputs()
        dist_print(f'Run fused kernel:', once_in_node=True)
        run_fused()
        dist_print(f' > Done, exiting', once_in_node=True)

        # Destroy and exit
        dist.barrier()
        buffer.destroy()
        dist.destroy_process_group()
        return

    # Non-overlapped baseline: EP dispatch + GEMM + EP combine
    deep_ep, tilelang_ops, tilelang_bench, is_legacy_loaded = import_baseline()
    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    ep_buffer = deep_ep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank, hidden=hidden,
        num_topk=num_topk, use_fp8_dispatch=True,
        explicitly_destroy=True,
        allow_multiple_reduction=False,
    ) if is_legacy_loaded else None

    use_blockwise_128 = os.environ.get('DG_MEGA_MOE_USE_BLOCK_WISE_FP8', '0') != '0'
    # Match baseline quantization granularity to the fused kernel's setting so
    # the comparison is apples-to-apples. When blockwise128 is on, the fused
    # kernel quantizes the L1 output per-128-K; the baseline must do the same
    # (per-128 tilelang SwiGLU + expand per-128 SF into 4 per-32 slots for the
    # L2 GEMM) to yield a near-bit-exact reference.
    baseline_num_per_channels = 128 if use_blockwise_128 else 32

    def run_baseline():
        recv_x, _, recv_topk_weights, handle, _ = ep_buffer.dispatch(
            x, topk_idx=topk_idx, topk_weights=topk_weights,
            cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats_baseline,
            num_experts=num_experts, expert_alignment=alignment,
            do_cpu_sync=False, do_handle_copy=False,
            do_expand=True, use_tma_aligned_col_major_sf=True,
        )
        n = recv_x[0].size(0)
        l1_y = torch.empty((n, intermediate_hidden * 2), dtype=torch.bfloat16, device='cuda')
        baseline_gemm = (deep_gemm.m_grouped_fp8_gemm_nt_contiguous
                    if args.weight_dtype == 'fp8'
                    else deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous)
        baseline_gemm(
            recv_x, l1_weights, l1_y, handle.psum_num_recv_tokens_per_expert,
            use_psum_layout=True, recipe=(1, 1, 32))
        # noinspection PyCallingNonCallable
        l1_y = tilelang_ops.swiglu_apply_weight_to_fp8(
            x=l1_y,
            topk_weights=recv_topk_weights,
            avail_tokens=handle.psum_num_recv_tokens_per_expert[-1],
            num_per_channels=baseline_num_per_channels,
            use_col_major_scales=True,
            round_scale=True,
            ue8m0_scale=True,
            output_bf16=False,
            clamp_value=args.activation_clamp,
            fast_math=bool(args.fast_math)
        )
        # When using blockwise-128 quantization, tilelang returns per-128 SF.
        # The L2 GEMM expects per-32 SF layout, so expand per-128 → per-32.
        # Each UE8M0 byte (one per-128 scale) must be replicated 4 times at
        # byte granularity to fill 4 consecutive per-32 slots.
        if baseline_num_per_channels == 128:
            l1_y_128_data, l1_y_128_sf = l1_y
            nt = l1_y_128_sf.shape[0]
            tma_aligned_nt = deep_gemm.get_tma_aligned_size(nt, 4)
            # l1_y_128_sf may be col-major strided; make contiguous first
            # Shape: (nt, packed_k) int32 where packed_k = num_128_scales / 4
            sf_contig = l1_y_128_sf.contiguous()
            # Unpack int32 → uint8: each int32 holds 4 UE8M0 bytes
            sf_uint8 = sf_contig.view(dtype=torch.uint8)  # (nt, packed_k * 4 = num_128_scales)
            # Repeat each byte 4x: per-128 → per-32
            sf_expanded = sf_uint8.repeat_interleave(4, dim=-1)  # (nt, num_32_scales)
            target_k = sf_expanded.shape[1] // 4  # num_32_scales / 4 packed int32
            sf_int32 = sf_expanded.contiguous().view(dtype=torch.int32)  # (nt, target_k)
            l1_y_sf_col = torch.empty_strided(
                (nt, target_k),
                (1, tma_aligned_nt),
                dtype=torch.int32, device='cuda')
            l1_y_sf_col.copy_(sf_int32)
            l1_y = (l1_y_128_data, l1_y_sf_col)
        l2_y = torch.empty((n, hidden), dtype=torch.bfloat16, device='cuda')
        baseline_gemm(
            l1_y, l2_weights, l2_y, handle.psum_num_recv_tokens_per_expert,
            use_psum_layout=True, recipe=(1, 1, 32))
        return ep_buffer.combine(l2_y, handle=handle)[0], cumulative_local_expert_recv_stats_baseline

    # Check correctness
    num_correctness_tests = 1 if args.num_correctness_tests is None else args.num_correctness_tests
    # noinspection PyBroadException
    if is_legacy_loaded and num_correctness_tests > 0:
        dist_print('Running correctness tests:', once_in_node=True)
        for i in range(num_correctness_tests):
            create_inputs()
            fused_results = run_fused()
            baseline_results = run_baseline()
            for idx, (fused_result, baseline_result) in enumerate(zip(fused_results, baseline_results)):
                if idx == 1:
                    # Skip cumulative stats comparison
                    continue
                if not torch.equal(fused_result, baseline_result):
                    diff_mask = (fused_result != baseline_result)
                    num_diff = diff_mask.sum().item()
                    total = fused_result.numel()
                    abs_diff = (fused_result.float() - baseline_result.float()).abs()
                    max_abs = abs_diff.max().item()
                    mean_abs = abs_diff.mean().item()
                    # Relative error: normalize by baseline magnitude
                    rel_diff = abs_diff / (baseline_result.float().abs() + 1e-6)
                    max_rel = rel_diff.max().item()
                    mean_rel = rel_diff.mean().item()
                    dist_print(f'  MISMATCH result[{idx}]: {num_diff}/{total} elements differ '
                               f'({100.0*num_diff/total:.2f}%)', once_in_node=True)
                    dist_print(f'  Max abs diff: {max_abs:.6e}, Mean abs diff: {mean_abs:.6e}', once_in_node=True)
                    dist_print(f'  Max rel diff: {max_rel:.6e}, Mean rel diff: {mean_rel:.6e}', once_in_node=True)
                    # Show a few differing positions
                    diff_positions = torch.nonzero(diff_mask, as_tuple=False)[:5]
                    for pos in diff_positions:
                        r, c = pos[0].item(), pos[1].item()
                        dist_print(f'    [{r},{c}] fused={fused_result[r,c].item():.6f} '
                                   f'baseline={baseline_result[r,c].item():.6f}', once_in_node=True)
                    assert False, f"Correctness check failed for result[{idx}]"
            if (i + 1) % 100 == 0 or i == num_correctness_tests - 1:
                dist_print(f' > Correctness test #{i + 1}/{num_correctness_tests} passed', once_in_node=True)
        dist_print(once_in_node=True)
    else:
        create_inputs()

    # Count local received tokens
    gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
    gathered_topk_idx[(gathered_topk_idx < rank_idx * num_experts_per_rank) | \
                      (gathered_topk_idx >= (rank_idx + 1) * num_experts_per_rank)] = -1
    num_recv_tokens = (gathered_topk_idx != -1).sum().item()

    # Benchmark
    t_fused = bench_kineto(
        run_fused, 'mega_moe',
        barrier=lambda: ep_buffer.barrier(use_comm_stream=False) if ep_buffer else dist.barrier(),
        trace_path=None if not args.dump_profile_traces else f'{args.dump_profile_traces}/mega_moe_rank{rank_idx}.json')
    t_baseline = tilelang_bench(run_baseline, _n_warmup=5, _n_repeat=1, backend='cudagraph', return_mode='median') / 1e3 if is_legacy_loaded else 0

    # TFLOPS: 3 matmuls (L1 left, L1 right, L2), each 2 * M * N * K
    safe_div = lambda a, b: float('nan') if b == 0 else a / b
    num_recv_tokens = int(num_recv_tokens)
    tflops = safe_div(2 * num_recv_tokens * (hidden * intermediate_hidden * 3) / 1e12, t_fused)

    # HBM bytes: weights (FP4 packed = 0.5 bytes / FP8 = 1 byte) + activations (FP8 = 1 byte) + output (BF16 = 2 bytes)
    num_touched_experts = int(torch.unique(gathered_topk_idx.flatten()).numel()) - 1 # NOTES minus 1 to exclude "-1"
    weight_bytes_per_elem_x2 = 2 if args.weight_dtype == 'fp8' else 1  # FP8: 1B, FP4: 0.5B → /2 in `// (2 if fp4 else 1)`
    weight_div = 1 if args.weight_dtype == 'fp8' else 2
    num_hbm_bytes = (
        num_touched_experts * intermediate_hidden * 2 * hidden // weight_div +   # L1 weights
        num_touched_experts * hidden * intermediate_hidden // weight_div +       # L2 weights
        num_recv_tokens * hidden +                                               # L1 acts read (FP8)
        num_recv_tokens * intermediate_hidden +                                  # L1 output write (FP8)
        num_recv_tokens * intermediate_hidden +                                  # L2 acts read (FP8)
        num_recv_tokens * hidden * 2                                             # L2 output write (BF16)
    )
    hbm_gbs = safe_div(num_hbm_bytes / 1e9, t_fused)

    # NVLink bytes: dispatch pull + combine write-back
    num_nvlink_bytes = num_recv_tokens * hidden * 3
    nvlink_gbs = safe_div(num_nvlink_bytes / 1e9, t_fused)

    # Combine reduction (serial) time approximation
    t_reduction = num_tokens * hidden * 2 * (1 + num_topk) / 6.5e12

    # Summary
    approx_factor = t_fused / (t_fused - t_reduction)
    dist_print('Performance:', once_in_node=True)
    dist_print(f' > EP: {rank_idx:2}/{num_ranks} | '
               f'{tflops:4.0f} TFLOPS | '
               f'overlap: '
               f'{tflops * approx_factor:4.0f} TFLOPS, '
               f'HBM {hbm_gbs * approx_factor:4.0f} GB/s, '
               f'NVL {nvlink_gbs * approx_factor:3.0f} GB/s | '
               f'{t_fused * 1e6:4.0f} us, '
               f'reduction: {t_reduction * 1e6:4.1f} us | '
               f'{safe_div(t_baseline, t_fused):.2f}x legacy')

    # Exit
    dist.barrier()
    buffer.destroy()
    ep_buffer.destroy() if is_legacy_loaded else None
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test PyTorch symmetric memory')

    # Resource settings
    parser.add_argument('--ncu-profile-only', action='store_true', help='Only run profiling without correctness test')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn (default: 8)')

    # Model settings
    parser.add_argument('--num-max-tokens-per-rank', type=int, default=32, help='Number of maximum tokens per rank')
    parser.add_argument('--num-tokens', type=int, default=0, help='Number of tokens per rank (follow max minus removed if 0)')
    parser.add_argument('--num-max-removed-tokens', type=int, default=0, help='Maximum number of tokens to remove')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden size')
    parser.add_argument('--intermediate-hidden', type=int, default=3072, help='Intermediate hidden size')
    parser.add_argument('--activation-clamp', type=float, default=10, help='Clamp value for activation')
    parser.add_argument('--num-experts', type=int, default=384, help='Number of experts')
    parser.add_argument('--num-topk', type=int, default=6, help='Number of expert selections')
    parser.add_argument('--masked-ratio', type=float, default=0.0, help='Mask some expert selections')
    parser.add_argument('--fast-math', type=int, default=1, help='Enable fast math (0 or 1, default: 1)')
    parser.add_argument('--weight-dtype', type=str, default='fp4', choices=['fp4', 'fp8'],
                    help='Weight dtype: fp4 (W4A8 baseline) or fp8 (W8A8)')

    # Test settings
    parser.add_argument('--num-correctness-tests', type=int, default=None, help='Pressure test')
    parser.add_argument('--dump-profile-traces', type=str, default='', help='Dump profiling trace JSONs')
    parser.add_argument('--local-rank-idx', type=int, default=None, help='Run as single process with this local rank (e.g. for NCU prof)')
    args = parser.parse_args()

    # Create dump trace directories
    if args.dump_profile_traces:
        os.makedirs(args.dump_profile_traces, exist_ok=True)

    if args.local_rank_idx is not None:
        # Single-process mode: each process is launched separately (e.g. by NCU)
        test(args.local_rank_idx, args.num_processes, args)
    else:
        # Launch tests
        num_processes = args.num_processes
        torch.multiprocessing.spawn(test, args=(num_processes, args), nprocs=num_processes)
