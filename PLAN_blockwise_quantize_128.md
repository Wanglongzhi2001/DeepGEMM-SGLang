# PLAN: W8A8 MegaMoE 支持 Blockwise 128 量化

## 1. 需求

在 W8A8 MegaMoE 中支持 blockwise 128 量化：
- **权重**：K 维度 128 粒度量化（per-row per-128-K）
- **L1 Epilogue 激活**：K 维度（intermediate_hidden 方向）128 粒度量化（per-token per-128-K）
- **精度要求**：与标准 per-token per-128-K 量化完全等价，不接受折中

## 2. 核心难点

L1 epilogue 流式逐 n_block 处理，每个 tile 的 SwiGLU 输出只有 64 列（`L1_OUT_BLOCK_N = BLOCK_N/2 = 64`）。per-128-K 需要 128 列的 amax，但相邻的两个 n_block（各 64 列）在当前 scheduler 下可能由不同 SM 处理。

**量化 FP8 data 时 SF 必须已确定** → 必须在写 FP8 data 之前知道完整 128 列的 amax。

## 3. 方案选型

### 3.1 被否决的方案：两遍 gmem staging

将 SwiGLU BF16 结果写入 gmem 暂存 buffer，等所有 n_block 的 amax 就绪后从 gmem 读回重新量化。

**否决原因**：
- 额外显存 ~230 MB（`kNumMaxPoolTokens * intermediate_hidden * 2`）
- 额外 HBM 带宽：+3× intermediate_hidden bytes/token（BF16 write + BF16 read + FP8 write）
- 跨 SM 同步开销（amax_arrival_mask 等待）
- 预估 +10-20% kernel latency

### 3.2 被否决的方案：精度折中（per-64 量化 + per-128 SF）

数据按 per-64 粒度量化，MMA 使用 per-128 SF。**用户明确拒绝任何精度折中**。

## 4. 最终方案：Scheduler Pair 调度 + Smem 暂存延迟量化

### 4.1 核心思路

修改 scheduler，以 **n_block pair**（2 个连续 n_block）为最小 L1 调度单位。同一 SM 上**连续**处理 n_block `2k` 和 `2k+1`，使得 epilogue 可以在同一 SM 内完成 128 列的 amax 汇聚和量化，无需跨 SM 同步，无需 gmem 暂存。

### 4.2 执行流程

对于一个 n_block pair（n_block `2k` 和 `2k+1`）：

**Step 1：处理第一个 n_block（n_block_idx = 2k）**
1. 等待 MMA 完成（wait tmem_full_barriers）
2. 从 TMEM load accumulator
3. 释放 TMEM（arrive tmem_empty_barriers）
4. 执行 SwiGLU
5. 计算 per-32 amax（与现有逻辑相同）
6. **不做 FP8 量化，不写 FP8 data**
7. **将 SwiGLU float 结果暂存到 smem 暂存区**（~8 KB）
8. 将 per-32 amax 暂存到寄存器或 smem

**Step 2：处理第二个 n_block（n_block_idx = 2k+1）**
1. 等待 MMA 完成（wait tmem_full_barriers，使用下一个 stage）
2. 从 TMEM load accumulator
3. 释放 TMEM（arrive tmem_empty_barriers）
4. 执行 SwiGLU
5. 计算 per-32 amax

**Step 3：合并并量化**
1. 取 pair 内 4 个 per-32 amax（第一个 n_block 2 个 + 第二个 n_block 2 个）的 max → per-128 amax
2. 计算 UE8M0 SF = `ceil_to_ue8m0(amax_128 / 448.0)`
3. 用该 SF 量化**第一个 n_block** 的 SwiGLU 值（从 smem 暂存区读回）→ FP8，写入 smem_cd → TMA store
4. 用该 SF 量化**第二个 n_block** 的 SwiGLU 值（仍在寄存器中）→ FP8，写入 smem_cd → TMA store
5. 写 per-128 SF 到 `l2_sf_buffer`（对应 4 个 per-32 slot 写相同值）
6. 通知 `l2_arrival_mask`（两个 n_block 一起通知）

### 4.3 与 kNumEpilogueStages = 2 的兼容性

两个 n_block 的 TMEM 使用是**串行的**（先处理第一个，释放 TMEM，再处理第二个）。double buffer 的两个 stage 正好交替使用：
- 第一个 n_block 用 `accum_stage_idx = current_iter_idx % 2`
- 第二个 n_block 用 `accum_stage_idx = (current_iter_idx + 1) % 2`

MMA warp 同样串行发射两个 n_block 的 GEMM，完全兼容现有 pipeline。

### 4.4 与 2-CTA Cluster 的兼容性

当前 2-CTA cluster 中，两个 CTA 共享 `blockIdx.x`，scheduler 对两个 CTA 产生相同的 `(m_block_idx, n_block_idx)`。2-CTA 的意义是 multicast TMA load（A 的 `LOAD_BLOCK_M = BLOCK_M/2` per CTA）和 `UMMA_M = 256`。

修改为 pair 调度后：scheduler 返回的是 `n_block_idx_base`（pair 的起始 n_block），两个 CTA 仍然共享同一个 pair。因此 **2-CTA cluster 机制不受影响**。

原有 assert `kNumL1BlockNs % 2 == 0` 保证了 pair 数量为整数（`kNumL1BlockNPairs = kNumL1BlockNs / 2`）。

### 4.5 Smem 暂存区开销

第一个 n_block 的 SwiGLU 输出需要以 float 精度暂存到 step 3 才能量化。

**关键优化：逐 STORE_BLOCK_M 处理**

Epilogue 原本就是逐 `STORE_BLOCK_M` 迭代的（外层 `for s` 循环）。因此不需要暂存整个 `WG_BLOCK_M` 的数据，只需在每个 store block 迭代中暂存当前 store block 的数据即可：

- 暂存大小 per warpgroup = `STORE_BLOCK_M × L1_OUT_BLOCK_N × sizeof(float)` = 32 × 64 × 4 = **8 KB**
- 总计（2 warpgroups） = **16 KB**

SM100 有 228 KB smem，当前 kernel smem 使用远未到上限，16 KB 完全可行。

### 4.6 精度分析

- SwiGLU float 结果直接以 **float** 精度暂存在 smem
- 读回时仍是 float，然后用 per-128 SF 量化为 FP8
- 与标准 per-128-K 量化流程完全等价：SwiGLU(float) → amax(float) → SF(UE8M0) → quantize(float → FP8)
- **bit-exact，零精度损失**

## 5. Scheduler 修改

### 5.1 当前 Scheduler 逻辑

```cpp
// mega_moe.cuh line 148-159
CUTLASS_DEVICE cute::tuple<BlockPhase, uint32_t, uint32_t, uint32_t> get_next_block() {
    if (next_phase == BlockPhase::Linear1) {
        if (fetch_next_l1_block()) {
            n_block_idx = block_idx - m_block_idx * kNumL1BlockNs;
            block_idx += kNumSMs;
            return {BlockPhase::Linear1, current_local_expert_idx, m_block_idx, n_block_idx};
        }
    }
}
```

每次调用返回一个 `(phase, expert, m_block, n_block)` 四元组，block_idx 按 kNumSMs stride 递增。

### 5.2 修改后 Scheduler 逻辑

```cpp
// 新增: 以 pair 为单位的 L1 调度
// kNumL1BlockNPairs = kNumL1BlockNs / 2
// 每次 get_next_block() 仍返回单个 block，但 epilogue 内部连续处理 2 个

// 方案 A（最小改动）：scheduler 不变，epilogue 内部处理 pair
// Epilogue 在收到 n_block_idx 时：
//   - 如果 n_block_idx 是偶数：执行 step 1（SwiGLU + 暂存），不写 FP8，不通知 L2
//   - 如果 n_block_idx 是奇数：执行 step 2 + step 3（合并量化 + 写出）

// 但这要求同一 SM 连续处理偶数和奇数 n_block → 需要 scheduler 保证！

// 方案 B（推荐）：修改 scheduler，以 pair 为单位分配
// fetch_next_l1_block 按 kNumL1BlockNPairs 划分：
CUTLASS_DEVICE bool fetch_next_l1_block() {
    const auto wave_end_expert_idx = get_wave_expert_end_idx();
    while (current_local_expert_idx < wave_end_expert_idx) {
        const auto num_m_blocks = get_current_num_m_blocks();
        m_block_idx = block_idx / kNumL1BlockNPairs;  // 改用 pair 数量
        if (m_block_idx < num_m_blocks)
            return true;
        block_idx -= num_m_blocks * kNumL1BlockNPairs;
        advance_expert_idx();
    }
    return false;
}

// get_next_block() 中：
n_block_idx = (block_idx - m_block_idx * kNumL1BlockNPairs) * 2;  // 返回 pair 的起始 n_block
block_idx += kNumSMs;
// epilogue 知道要连续处理 n_block_idx 和 n_block_idx + 1
```

### 5.3 非 blockwise128 模式的兼容

当 `kUseBlockwise128 = false` 时，scheduler 使用原有逻辑（per n_block 调度）。通过 `if constexpr` 选择不同的 scheduler 路径。

```cpp
if constexpr (kUseBlockwise128) {
    // Pair-based scheduling
    constexpr uint32_t kNumL1BlockNPairs = kNumL1BlockNs / 2;
    m_block_idx = block_idx / kNumL1BlockNPairs;
    n_block_idx = (block_idx - m_block_idx * kNumL1BlockNPairs) * 2;
} else {
    // Original per-n_block scheduling
    m_block_idx = block_idx / kNumL1BlockNs;
    n_block_idx = block_idx - m_block_idx * kNumL1BlockNs;
}
```

## 6. Epilogue 修改

### 6.1 Smem 暂存区分配

在 smem layout 中新增一块用于暂存第一个 n_block SwiGLU float 结果的区域：

```cpp
// 新增 smem 暂存区（仅 kUseBlockwise128 时使用）
// 逐 STORE_BLOCK_M 处理时，只需暂存一个 store block 的数据
// Shape: [STORE_BLOCK_M, L1_OUT_BLOCK_N] float per epilogue warpgroup
constexpr uint32_t SMEM_PAIR_STAGING_SIZE = kUseBlockwise128
    ? (kNumEpilogueWarpgroups * STORE_BLOCK_M * L1_OUT_BLOCK_N * sizeof(float))
    : 0;
// 典型值: 2 * 32 * 64 * 4 = 16384 bytes = 16 KB
```

这块 smem 可以复用 smem_cd 的空间（因为第一个 n_block 不做 TMA store，smem_cd 空闲），或者单独分配。

**复用 smem_cd 的可行性**：
- smem_cd 大小 = `kNumEpilogueWarpgroups * STORE_BLOCK_M * L1_OUT_ROW_BYTES * kNumTMAStoreStages`
- FP8 模式下 = 2 × 32 × 64 × 2 = 8192 bytes（2 个 TMA stage）
- 暂存需要 = 2 × 32 × 64 × 4 = 16384 bytes（float）

暂存区比 smem_cd 大 2×。不能直接复用。

**方案**：在 smem_cd 之后追加暂存区，或者复用 smem_a/b 的某个 pipeline stage（epilogue 期间 MMA pipeline 不使用该 stage 的 smem）。

实际上，当处理第一个 n_block 时，不需要 smem_cd（因为不做 TMA store）。当处理第二个 n_block + 量化时，才需要 smem_cd。所以可以**临时复用 smem_cd 的全部空间**（两个 TMA stage 合计 8 KB）+ 额外分配 8 KB = 16 KB。

**最简方案**：直接额外分配 16 KB smem。SM100 的 228 KB 预算完全足够。

### 6.2 Epilogue 主循环修改

```cpp
// 当 kUseBlockwise128 时，epilogue 每次处理一个 n_block pair
// scheduler 返回 n_block_idx（pair 的偶数起始值）

if constexpr (kUseBlockwise128) {
    // ===== 第一个 n_block (n_block_idx) =====
    // Wait TMEM full
    const auto accum_stage_idx_0 = current_iter_idx % kNumEpilogueStages;
    const auto accum_phase_0 = (current_iter_idx++ / kNumEpilogueStages) & 1;
    tmem_full_barriers[accum_stage_idx_0]->wait(accum_phase_0);

    // Process store blocks for first n_block
    for (s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++s) {
        // Load TMEM → SwiGLU → compute per-32 amax
        // Store SwiGLU float results to smem_pair_staging
        // Store per-32 amax to registers (amax_first[])
        // Release TMEM on last atom
    }

    // ===== 第二个 n_block (n_block_idx + 1) =====
    // Wait TMEM full (next stage)
    const auto accum_stage_idx_1 = current_iter_idx % kNumEpilogueStages;
    const auto accum_phase_1 = (current_iter_idx++ / kNumEpilogueStages) & 1;
    tmem_full_barriers[accum_stage_idx_1]->wait(accum_phase_1);

    // Process store blocks for second n_block
    for (s = 0; s < WG_BLOCK_M / STORE_BLOCK_M; ++s) {
        // Load TMEM → SwiGLU → compute per-32 amax
        // Release TMEM on last atom

        // ===== 合并量化 =====
        // per-128 amax = max(amax_first[s], amax_second[s])
        // （每个 store block 对应 2 个 per-32 group：first n_block 1 个 + second n_block 1 个）
        // 实际上每个 warp pair 计算的 amax 覆盖 32 列，
        // pair 内共 4 个 per-32 amax → 取 max → per-128 amax

        // 计算 UE8M0 SF
        math::get_e4m3_sf_and_sf_inv(amax_128, sf, sf_inv);

        // 量化第一个 n_block（从 smem_pair_staging 读回 float）
        // → FP8 → store to smem_cd → TMA store (out_n = n_block_idx * L1_OUT_BLOCK_N)

        // 量化第二个 n_block（仍在寄存器中）
        // → FP8 → store to smem_cd → TMA store (out_n = (n_block_idx+1) * L1_OUT_BLOCK_N)

        // 写 SF 到 l2_sf_buffer（4 个 per-32 slot 写相同的 per-128 SF）
    }

    // 通知 L2（两个 n_block 一起）
    ptx::red_or_rel_gpu(
        workspace.get_l2_arrival_mask_ptr(pool_block_idx),
        (1ull << n_block_idx) | (1ull << (n_block_idx + 1))
    );
} else {
    // 现有 per-32 量化逻辑（不变）
}
```

### 6.3 Amax 合并细节

当前每个 n_block epilogue 中，amax 的粒度是 per-32 列：
- 每个 warp pair（2 个 warp）处理 32 列（`warp_idx_in_wg / 2` 决定 k_idx）
- 一个 n_block 有 `L1_OUT_BLOCK_N / 32 = 2` 个 per-32 group

Per-128-K 需要 4 个连续 per-32 group 的 max：
- 第一个 n_block 贡献 per-32 group `k_idx = n_block_idx * 2 + 0` 和 `n_block_idx * 2 + 1`
- 第二个 n_block 贡献 per-32 group `k_idx = (n_block_idx+1) * 2 + 0` 和 `(n_block_idx+1) * 2 + 1`

对于 pair 起始 `n_block_idx = 2k`：
- 4 个 per-32 group：k_idx = `4k, 4k+1, 4k+2, 4k+3` → 构成一个 per-128 group
- `per_128_amax = max(amax[4k], amax[4k+1], amax[4k+2], amax[4k+3])`

在 epilogue 中，第一个 n_block 的 warp pair 0 持有 amax[4k]（经过 cross-warp reduce），warp pair 1 持有 amax[4k+1]。第二个 n_block 的 warp pair 0 持有 amax[4k+2]，warp pair 1 持有 amax[4k+3]。

合并路径：
1. 第一个 n_block 处理完后，将 `amax[4k]` 和 `amax[4k+1]` 存入 smem
2. 第二个 n_block 处理后，从 smem 读回 `amax[4k]` 和 `amax[4k+1]`
3. 与当前 `amax[4k+2]` 和 `amax[4k+3]` 做 max → `per_128_amax`
4. 所有 warp 使用同一个 `per_128_amax` 计算 SF

### 6.4 SF 写入修改

当前 SF 按 per-32 粒度写入 `l2_sf_buffer`：每个 `k_idx` 位置写一个 UE8M0 byte。

Blockwise 128 下，4 个连续 `k_idx` 写入**相同的 per-128 SF**：

```cpp
if constexpr (kUseBlockwise128) {
    // 写 4 个连续 k_idx 位置，值相同
    const uint32_t k_base = (n_block_idx / 2) * 4;  // pair 对应的 per-128 group 起始
    for (uint32_t dk = 0; dk < 4; ++dk) {
        const uint32_t k_idx = k_base + dk;
        // ... 写 sf 到 l2_sf_buffer[k_idx][token]
    }
} else {
    // 现有逻辑：每个 k_idx 写独立 SF
}
```

## 7. MMA Warp 修改

### 7.1 Pair 模式下 MMA 发射

当 scheduler 返回一个 pair 时，MMA warp 需要**连续发射两个 n_block 的 GEMM**：

```cpp
if constexpr (kUseBlockwise128) {
    // 对 pair 中的两个 n_block 分别发射 GEMM
    for (uint32_t pair_offset = 0; pair_offset < 2; ++pair_offset) {
        const uint32_t actual_n_block_idx = n_block_idx + pair_offset;
        // TMA load B for actual_n_block_idx (n_idx = actual_n_block_idx * BLOCK_N)
        // Execute UMMA K-loop
        // Signal tmem_full on last K-block
    }
}
```

这与 MMA warp 现有的 persistent loop 兼容——只是每次 scheduler iteration 内部执行 2 次 GEMM 而非 1 次。

### 7.2 TMA Load 修改

对于 B 矩阵（weights），每个 n_block 加载不同的列段：
- 第一个 n_block：`n_idx = n_block_idx * BLOCK_N`
- 第二个 n_block：`n_idx = (n_block_idx + 1) * BLOCK_N`

A 矩阵（activations）对两个 n_block 相同（同一 m_block）。可以复用第一次的 A smem（如果 pipeline stage 允许）。

## 8. 代码修改清单

### 8.1 `tests/test_mega_moe.py`

```python
use_blockwise_128 = os.environ.get('DG_MEGA_MOE_BLOCKWISE_128', '0') != '0'

def cast_grouped_weights_to_fp8_blockwise128(bf16_weights):
    num_groups, n, k = bf16_weights.shape
    w_u8 = torch.empty((num_groups, n, k), device='cuda', dtype=torch.uint8)
    w_sf = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
    for i in range(num_groups):
        w_i_fp8, w_sf_128 = per_token_cast_to_fp8(bf16_weights[i], use_ue8m0=True, gran_k=128)
        w_u8[i] = w_i_fp8.view(torch.uint8)
        # repeat_interleave 4x: per-128 SF → per-32 SF shape
        w_sf[i] = w_sf_128.repeat_interleave(4, dim=-1)
    w = w_u8.contiguous().view(torch.float8_e4m3fn)
    w_sf = deep_gemm.transform_sf_into_required_layout(w_sf, n, k, (1, 32), num_groups)
    return w, w_sf

cast_fn = cast_grouped_weights_to_fp8_blockwise128 if (args.weight_dtype == 'fp8' and use_blockwise_128) \
          else (cast_grouped_weights_to_fp8 if args.weight_dtype == 'fp8' else cast_grouped_weights_to_fp4)
```

### 8.2 `scheduler/mega_moe.cuh` — Pair 调度

```cpp
// 新增模板参数
template <..., bool kUseBlockwise128 = false>
struct MegaMoEScheduler {
    // 新增 constexpr
    static constexpr uint32_t kNumL1BlockNPairs = kNumL1BlockNs / 2;
    static constexpr uint32_t kEffectiveL1BlockNs =
        kUseBlockwise128 ? kNumL1BlockNPairs : kNumL1BlockNs;

    CUTLASS_DEVICE bool fetch_next_l1_block() {
        while (current_local_expert_idx < wave_end_expert_idx) {
            const auto num_m_blocks = get_current_num_m_blocks();
            m_block_idx = block_idx / kEffectiveL1BlockNs;
            if (m_block_idx < num_m_blocks)
                return true;
            block_idx -= num_m_blocks * kEffectiveL1BlockNs;
            advance_expert_idx();
        }
        return false;
    }

    // get_next_block() 中：
    if constexpr (kUseBlockwise128) {
        n_block_idx = (block_idx - m_block_idx * kEffectiveL1BlockNs) * 2;
    } else {
        n_block_idx = block_idx - m_block_idx * kNumL1BlockNs;
    }
};
```

### 8.3 `sm100_fp8_fp4_mega_moe.cuh` — 新增模板参数

```cpp
bool kUseBlockwise128 = false,  // 新增
```

### 8.4 `sm100_fp8_fp4_mega_moe.cuh` — Smem 分配

```cpp
// Pair staging: 暂存第一个 n_block 的 SwiGLU float 结果 + amax
constexpr uint32_t SMEM_PAIR_STAGING_SIZE = kUseBlockwise128
    ? (kNumEpilogueWarpgroups * STORE_BLOCK_M * L1_OUT_BLOCK_N * sizeof(float)
       + kNumEpilogueWarpgroups * STORE_BLOCK_M / ATOM_M * sizeof(float2))  // amax staging
    : 0;

// 加入 SMEM_BEFORE_BARRIER_SIZE 计算
constexpr uint32_t SMEM_BEFORE_BARRIER_SIZE =
    SMEM_EXPERT_COUNT_SIZE + SMEM_SEND_BUFFER_SIZE + SMEM_CD_SIZE + SMEM_PAIR_STAGING_SIZE
    + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
```

### 8.5 `sm100_fp8_fp4_mega_moe.cuh` — Epilogue 重写（kUseBlockwise128 分支）

见第 6 节描述。主要改动点：
- Epilogue 内部 for_each_block 回调中，当 `kUseBlockwise128` 时执行 pair 流程
- 第一个 n_block：SwiGLU + 暂存到 smem + amax 暂存
- 第二个 n_block：SwiGLU + amax 合并 + 两次 FP8 量化 + 两次 TMA store
- L2 arrival：pair 内两个 bit 一起通知

### 8.6 `sm100_fp8_fp4_mega_moe.cuh` — MMA Warp 修改

当 `kUseBlockwise128` 时，MMA warp 在每个 scheduler iteration 内执行 2 次 K-loop（对应 pair 内两个 n_block 的 GEMM）。

### 8.7 `csrc/` Host 端

- 传入 `kUseBlockwise128` 模板参数
- 环境变量 `DG_MEGA_MOE_BLOCKWISE_128` 控制
- smem 大小计算更新（+SMEM_PAIR_STAGING_SIZE）

## 9. 性能影响

| 方面 | 影响 |
|------|------|
| 额外 HBM 带宽 | **0**（无 gmem 暂存） |
| 额外显存 | **0**（仅 smem 内暂存 ~16 KB） |
| 跨 SM 同步 | **0**（同一 SM 串行处理 pair） |
| Smem 压力 | +16 KB（STORE_BLOCK_M × 64 × 4 × kNumEpilogueWarpgroups） |
| 计算开销 | 第一个 n_block 的 SwiGLU 值需要多一次 smem write + read（~十几个 cycle） |
| 量化延迟 | 第一个 n_block 的量化延迟到第二个 n_block 完成后（增加 1 个 n_block 的 pipeline latency） |
| TMA store | 两个 n_block 的 TMA store 集中在 pair 末尾发射（可能造成短暂 TMA 拥塞） |
| Scheduler 粒度 | pair 粒度更粗 → SM 间负载不均可能略微增加 |
| 整体 kernel 性能 | 预计 **< 5% latency 增加** |

### 9.1 与两遍方案对比

| 指标 | 两遍 gmem staging | Scheduler pair（本方案） |
|------|-------------------|------------------------|
| 额外 HBM 带宽 | +3× intermediate_hidden B/token | **0** |
| 额外显存 | ~230 MB | **0** |
| 跨 SM 同步 | amax_arrival_mask 等待 | **无** |
| 预估 latency | +10-20% | **< 5%** |
| 代码复杂度 | 中等（新增 buffer + pass 2） | 中等（scheduler + epilogue pair 逻辑） |

## 10. 精度

**完全等价于标准 per-token per-128-K 量化**：
- 权重：per-row per-128-K UE8M0 SF（host 端 repeat_interleave 4x 后与 per-32 MMA 兼容）
- 激活：per-token per-128-K UE8M0 SF（epilogue 内 4 个 per-32 amax 取 max）
- SwiGLU 中间结果以 float 精度暂存在 smem，**零精度损失**
- **bit-exact** 等价于标准 per-128-K 实现

## 11. Review 记录

### Review 1
- 确认 scheduler pair 调度与 2-CTA cluster 兼容（两个 CTA 共享 blockIdx.x，pair 调度不影响 CTA 分工）
- 确认 kNumEpilogueStages=2 的 double buffer 机制与 pair 串行处理兼容（两个 n_block 交替使用两个 stage）
- 确认 smem 暂存大小（~16 KB）在 SM100 228 KB smem budget 内可行

### Review 2
- 确认 amax 合并逻辑正确：pair 内 4 个 per-32 amax → max → per-128 SF
- 确认 SF 写入 l2_sf_buffer 时 4 个连续 k_idx 写相同值的正确性
- 确认 l2_arrival_mask 通知时序：pair 内两个 n_block 全部量化完成后才通知
- 确认 MMA warp 连续发射 2 次 GEMM 的可行性（pipeline 无死锁）

### Review 3
- 确认性能预估合理（< 5% overhead 主要来自 smem staging 的额外 cycle 和 pair 粒度导致的轻微负载不均）
- 确认精度为 bit-exact（float 暂存，零精度损失）
- 确认整体方案完整性，可以开始实施
- 唯一风险点：smem 总量是否超限——需要在具体参数组合下验证 `SMEM_BEFORE_BARRIER_SIZE + SMEM_PAIR_STAGING_SIZE` 不超过 `cudaFuncAttributeMaxDynamicSharedMemorySize` 限制
