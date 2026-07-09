# Blockwise 128 Quantization for W8A8 MegaMoE — Implementation Record

## Overview

Implements per-128-K quantization in the L1 epilogue of MegaMoE, controlled by
`DG_MEGA_MOE_BLOCKWISE_128=1`. The "Scheduler Pair" approach dispatches 2
consecutive n_blocks to the same SM so that 4 per-32 groups (=128 columns)
share a single UE8M0 scale factor, achieving bit-exact equivalence with
standard per-token per-128-K quantization.

## Key Design Decisions

1. **Weight side**: host-only change — `cast_grouped_weights_to_fp8_blockwise128`
   quantizes at gran_k=128, then `repeat_interleave(4)` to fill per-32 SF slots.
   Zero kernel change needed for weights.

2. **Activation side**: "Scheduler Pair" in the L1 epilogue. First n_block
   stages SwiGLU float outputs to smem; second n_block combines amax from both,
   computes per-128 SF, and quantizes both n_blocks.

3. **L2 GEMM**: unchanged. It reads activations + SF written by L1 epilogue.
   The per-128 SF is replicated into 4 per-32 positions by the epilogue.

## Implementation Progress

### Phase 1: Host-side integration (DONE)
- Added `bool use_blockwise_128` to `Args` in `sm100_fp8_fp4_mega_moe.hpp`
- Added as template parameter in `generate_impl()` code generation
- Added `use_blockwise_128` to `get_mega_moe_config()` and `get_pipeline_config_for_mega_moe()` signatures in `heuristics/mega_moe.hpp`
- Added `DG_MEGA_MOE_BLOCKWISE_128` env var read in `csrc/apis/mega.hpp`
- Passed through to `sm100_fp8_fp4_mega_moe()` call

### Phase 2: Scheduler pair dispatch (DONE)
- Added `bool kUseBlockwise128 = false` template param to `MegaMoEScheduler`
- Added `kEffectiveL1BlockNs = kUseBlockwise128 ? (kNumL1BlockNs / 2) : kNumL1BlockNs`
- Modified `fetch_next_l1_block()`: uses `kEffectiveL1BlockNs` for linearization
- Modified `get_next_block()`: pair-based `n_block_idx = ... * 2` when blockwise128
- Modified `for_each_block()`: calls lambda twice per L1 pair (n_block_idx, n_block_idx+1)
- File: `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`

### Phase 3: Smem pair staging allocation (DONE)
- Added `kWGBlockM = BLOCK_M / kNumEpilogueWarpgroups` for correct sizing
- `SMEM_PAIR_STAGING_FLOAT_SIZE`: `kNumEpilogueWarpgroups * kWGBlockM * (BLOCK_N/2) * sizeof(float)`
- `SMEM_PAIR_AMAX_SIZE`: `kNumEpilogueWarpgroups * kPairAmaxPerWG * sizeof(float2)` where `kPairAmaxPerWG = kWGBlockM / 8`
- Aligned to 1024 bytes via `SMEM_PAIR_STAGING_SIZE`
- Inserted between SMEM_CD and SMEM_A/B regions
- Updated all offset calculations (smem_a, smem_b, sf_start_ptr)
- Heuristics `smem_pair_staging` uses `wg_block_m = block_m / num_epilogue_warpgroups`
- File: `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` (lines 311-335)
- File: `csrc/jit_kernels/heuristics/mega_moe.hpp` (lines 187-196)

### Phase 4: L1 Epilogue pair processing (DONE)
- Full implementation at line 1219 of `sm100_fp8_fp4_mega_moe.cuh`
- Branch: `if (block_phase == sched::BlockPhase::Linear1 and kUseBlockwise128)`
- **First n_block** (`n_block_idx % 2 == 0`):
  - Load TMEM accumulator, SwiGLU computation
  - Compute per-32 amax, cross-warp reduce
  - Stage float SwiGLU values to `smem_pair_float[WG_BLOCK_M][L1_OUT_BLOCK_N]`
  - Store reduced amax to `smem_pair_amax[s * kNumAtomsPerStore + i]`
  - Release TMEM, skip TMA store
- **Second n_block** (`n_block_idx % 2 == 1`):
  - Load TMEM accumulator, SwiGLU for second n_block
  - Per-32 amax for second n_block, cross-warp reduce
  - Combine first + second amax → per-128 amax → UE8M0 SF
  - Quantize first n_block from smem staging with shared SF → STSM → TMA store
  - Quantize second n_block from registers with shared SF → STSM → TMA store
  - Write per-128 SF to 4 consecutive k_idx positions in `l2_sf_buffer`
  - Notify L2 arrival mask with both n_block bits

### Phase 5: tvm_ffi compatibility fix (DONE)
- `_C.get_symm_buffer_size_for_mega_moe()` returns `tvm_ffi.core.Tensor` objects
  which are not subscriptable (no `__getitem__`)
- Added `_tvm_tensor_to_torch()` helper in `deep_gemm/mega/__init__.py`
- Converts views via `torch.from_dlpack(t)` in `SymmBuffer.__init__`

### Phase 6: Test weight quantization (DONE)
- Added `cast_grouped_weights_to_fp8_blockwise128()` in `test_mega_moe.py`
- Uses `per_token_cast_to_fp8(gran_k=128)` + `repeat_interleave(4, dim=-1)`
- Selected when `DG_MEGA_MOE_BLOCKWISE_128=1` and `--weight-dtype fp8`

## Verification Status

- **Build**: Passes cleanly with `build.sh`
- **Runtime**: Kernel launches and completes without CUDA errors
  - `--ncu-profile-only` mode: OK (128, 512, 8192 tokens)
  - All 4 ranks complete successfully
- **Smem config**: With blockwise128 on: `smem_size=229124` (vs 211716 without)
  - Delta = 17408 bytes = 16KB float staging + 256B amax staging + alignment
- **Correctness**: NOT YET VERIFIED (requires baseline comparison path or
  reference implementation; `tilelang_ops` baseline not available in current env)
- **Non-blockwise128 path**: Still works correctly (regression-free)

## Files Modified

| File | Changes |
|------|---------|
| `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh` | Template param, smem staging, full epilogue pair logic |
| `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh` | `kUseBlockwise128`, pair dispatch, `for_each_block` |
| `csrc/jit_kernels/heuristics/mega_moe.hpp` | Pipeline config smem accounting |
| `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp` | Pass `use_blockwise_128` to heuristics |
| `csrc/apis/mega.hpp` | Env var read, pass to kernel |
| `deep_gemm/mega/__init__.py` | tvm_ffi → torch tensor conversion |
| `tests/test_mega_moe.py` | `cast_grouped_weights_to_fp8_blockwise128` |

## Known Issues / TODO

1. **Correctness verification** blocked on lack of baseline in current env
2. The `smem_pair_float` indexing uses `s * STORE_BLOCK_M + i * ATOM_M` for row
   addressing across store-block iterations — correct for the multi-store-block
   case (`WG_BLOCK_M > STORE_BLOCK_M`), verified logically consistent
3. SF write uses `transform_sf_token_idx` for UTCCP 4×32 transpose — same as
   the non-blockwise path, applied to 4 k_idx positions per atom

## Bugfix Round 2 (amax staging lane dimension)

### Symptom
Blockwise-128 correctness test: 97.6% elements differ, mean_rel ~8.9e5, output
magnitudes ~7x too large (e.g. fused=-14272 vs baseline=-2000). Non-blockwise
W8A8 (per-32) path passes cleanly, isolating the bug to the pair epilogue.

### Root cause
In the first-n_block epilogue, the per-warp-pair reduced amax was stored to
`smem_pair_amax` with an index that OMITTED the lane (token) dimension:
```
smem_pair_amax[s*kNumAtomsPerStore*2 + (warp_idx_in_wg/2)*kNumAtomsPerStore + i]
```
Each of lanes 0..3 owns a different token pair and computes a different `my_amax`,
but they all wrote to the SAME slot → 3/4 of tokens' amax were lost. The second
n_block read this per-lane, so every lane got the single surviving (wrong) value.
Underestimated per-128 amax → SF too small → SwiGLU*sf_inv overflowed FP8 e4m3
range (saturating at 448) → dequant gave wildly wrong (large) L2 outputs.

Note: `second_amax_all` (second n_block, line ~1450) already included `lane%4`
in its index and was correct; only the first-n_block staging dropped it.

### Fix
Re-layout `smem_pair_amax` as `[atom_global][warp_pair(2)][lane(4)]` float2:
- index = `(s*kNumAtomsPerStore+i)*8 + (warp_idx_in_wg/2)*4 + (lane_idx%4)`
- `kPairAmaxPerWG = kWGBlockM` (was `(kWGBlockM/8)*2`, now 4× larger, still tiny)
- Host heuristics `smem_pair_staging` amax term updated to
  `num_epilogue_warpgroups * wg_block_m * sizeof(float2)` to match.
- Files: `impls/sm100_fp8_fp4_mega_moe.cuh` (kPairAmaxPerWG def + write + 2 reads),
  `csrc/jit_kernels/heuristics/mega_moe.hpp` (host smem sizing).
- `_C.so` rebuilt (heuristics compiled into it).

### Verification
Blocked intermittently by external GPU contention on shared node (GPUs 4,5,6
frequently occupied by other jobs at ~170GB/100% util). Test re-run pending a
free 4-GPU window.

## Bugfix Round 3 (TMA store smem reuse)

### Symptom
After the amax lane-dimension fix, blockwise-128 still needed final correctness
verification. The kernel had another precision-risk path in the second n_block
branch: the first n_block's FP8 output was TMA-stored from `smem_cd[tma_stage_idx]`,
then the same smem stage was immediately reused for the second n_block after only
`ptx::tma_store_wait<kNumTMAStoreStages - 1>()`.

With `kNumTMAStoreStages = 2`, this is `wait<1>`, which allows one TMA store to
remain pending. Since the first n_block TMA store may still be reading that smem
stage, the second n_block STSM can overwrite it and corrupt the first n_block's
L1 activation output. This presents as a precision error even when SF/amax values
are correct.

### Fix
- In `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`, changed the
  wait between first n_block TMA store and second n_block STSM to
  `ptx::tma_store_wait<0>()` so the first store is fully drained before reusing
  `smem_cd[tma_stage_idx]`.
- Also cleaned up the first-n_block amax staging write so only one warp in each
  warp-pair writes the reduced amax slot:
  `if ((warp_idx_in_wg % 2) == 0 and lane_idx < 4)`.
  The previous duplicated write was expected to be numerically benign, but it was
  still a same-address race.
- Added host-side guards in `csrc/apis/mega.hpp` and
  `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`:
  blockwise128 requires `use_fp8_acts && !use_fp4_acts`.
- Tightened `tests/test_mega_moe.py` by removing the loose blockwise128
  `mean_rel < 0.05` fallback. Blockwise128 now uses the strict correctness check
  against the per-128 baseline.

### Verification
Build/install command completed successfully:
```
source /root/paddlejob/share-storage/gpfs/system-public/wanglongzhi/miniconda3/etc/profile.d/conda.sh && conda activate dsv4_py312 && bash build.sh && pip install --force-reinstall dist/deep_gemm-*.whl
```

Blockwise-128 W8A8 test command:
```
source /root/paddlejob/share-storage/gpfs/system-public/wanglongzhi/miniconda3/etc/profile.d/conda.sh && conda activate dsv4_py312 && cd tests && bash run.sh
```

Result:
- Correctness: `Correctness test #1/1 passed`
- Performance:
  - EP0: 1795 TFLOPS, overlap 1860 TFLOPS, 3611 us, 1.25x legacy
  - EP1: 1805 TFLOPS, overlap 1870 TFLOPS, 3613 us, 1.25x legacy
  - EP2: 1798 TFLOPS, overlap 1864 TFLOPS, 3609 us, 1.25x legacy
  - EP3: 1795 TFLOPS, overlap 1861 TFLOPS, 3611 us, 1.25x legacy

Non-blockwise W8A8 regression command:
```
source /root/paddlejob/share-storage/gpfs/system-public/wanglongzhi/miniconda3/etc/profile.d/conda.sh && conda activate dsv4_py312 && DG_MEGA_MOE_USE_FP8_ACTS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 python test_mega_moe.py --weight-dtype fp8 --num-processes 4 --num-correctness-tests 1
```

Result:
- Correctness: `Correctness test #1/1 passed`
- Performance: ~2017-2029 TFLOPS, overlap ~2099-2112 TFLOPS, 1.39x legacy

The legacy A100 Triton load warning still appears, but it is benign for this fused
MegaMoE path; both blockwise and non-blockwise correctness checks passed.

## Checkpoint Tag: correctness-preserving version before TMA overlap optimization

Before starting the two-stage TMA store optimization, the correctness-passing
blockwise128 version is saved in git with tag:

```
w8a8-blockwise128-correct
```

The tag is created after committing the current blockwise128 implementation,
including the Round 3 correctness fix and this record file.

### Restore commands

To inspect this exact version without modifying the current branch:
```
git checkout w8a8-blockwise128-correct
```

To create a new working branch from the saved correctness version:
```
git switch -c restore-w8a8-blockwise128-correct w8a8-blockwise128-correct
```

To move the current branch back to the saved version and discard later local
changes, use only after confirming no work needs to be preserved:
```
git reset --hard w8a8-blockwise128-correct
```

If the tag is later pushed to a remote and restored in a fresh clone, first fetch
tags:
```
git fetch --tags
```

