# FlashAttention-4 Blackwell SM100 `hdim=256` 2-CTA 反向传播：`M=64` TMEM 累加器布局、零缓冲 DSMEM `dS` 交换与 `AccumTileOrder` 统一后处理

> **English Version**: [fa4-sm100-hd256-bwd-m64-tmem-accum-layout.en.md](fa4-sm100-hd256-bwd-m64-tmem-accum-layout.en.md) | **返回索引 / Back to README**: [README.md](../../README.md) | [README.en.md](../../README.en.md)

## 1. 工程问题定义：为什么 `hdim=256` 在 SM100 反向内核中无法使用 `M=128` 128-Row KV Tile？

在 Blackwell SM100（每 SM 硬件上限 `512` 列 Tensor Memory / TMEM，每列 `128 lanes × 32-bit`；每 SM 最大共享内存 `227 KiB = 232,448 B`）上，FlashAttention-4 反向传播需要同时维护五类 FP32 累加器：`S`、`dP`、`dV`、`dK` 与 `dQ`。

根据 `Dao-AILab/flash-attention` 提交 `2b00d76f222c842b245fd98847223c898bc476db` 与 `965c9212e6be33c821ee9f2b461564c5c91a2503`（`flash_attn/cute/flash_bwd_sm100.py` L265–319）：
- 在传统的 **`M=128` 布局**（`tile_m = 128, tile_n = 128, acc_m64 = False, lane_rows = 1`）下，每个 TMEM Lane 对应 `1` 行，累加器占用的 TMEM 列数等于其逻辑列宽：
  $$\mathrm{cols}(dV) = d_v = 256,\quad \mathrm{cols}(dK) = d = 256,\quad \mathrm{cols}(S) = M = 128,\quad \mathrm{cols}(dP) = M = 128,\quad \mathrm{cols}(dQ) = \frac{d}{\mathrm{cta\_group\_size}}$$
  - 在 1-CTA 下，`tmem_end = 128 (S) + 256 (dV) + 128 (dP) + 256 (dK) = 768 > 512` 列，超出硬件容量 **50%**；
  - 即便切换到 2-CTA `TWO_CTA_SERIAL` 调度，若保持 `tile_n = 128`（`lane_rows = 1`），仅常驻的 `dV (256)` + `dK (256)` 就已占满全部 `512` 列（`tmem_S_offset = 512, tmem_end = 640 > 512`），根本没有空间放置 `S`、`dP` 和 `dQ`！

---

## 2. 一手源码走读：`M=64` 2-CTA TMEM 布局与零额外缓冲 DSMEM `dS` 交换

### 2.1 `M=64` 累加器布局（`tile_n = 64, acc_m64 = True, lane_rows = 2`）

`FlashAttentionBackwardSm100`（`flash_bwd_sm100.py` L103–149 `NOTE [M=64 accumulator layout]` 及 L265–290）通过将每 CTA 的 KV Tile 缩小为 **`tile_n = 64`** 并启用 **2-CTA `M=64` TMEM 布局**（`self.acc_m64 = self.use_2cta_instrs and self.tile_n == 64`），使 TMEM 的 `128` 个 Lane 折叠承载两个 `64` 行半块（`lane_rows = 2`：Lane `0..63` 存左半列 `[0, N/2)`，Lane `64..127` 存右半列 `[N/2, N)`）。此时**所有累加器的 TMEM 列宽全部减半为 `N // 2`**：

| 累加器 / 操作数 | 逻辑形状 `(rows, cols)` | `M=128` (`tile_n=128`) 列数 | `M=64` (`tile_n=64, acc_m64=True`) 列数 | `TWO_CTA_SERIAL` TMEM 偏移区间 `[start, end)` |
|---|---|---|---|---|
| **`dV`** | `(64, 256)` | `256` | **`128`** (`256 // 2`) | **`[0, 128)`** |
| **`dK`** | `(64, 256)` | `256` | **`128`** (`256 // 2`) | **`[128, 256)`** |
| **`S^T`** (`Q @ K^T`) | `(64, 128)` | `128` | **`64`** (`128 // 2`) | **`[256, 320)`** |
| **`dQ`** (`dS @ K`) | `(64, 256)` / CTA | `128` | **`128`** (`256 // 2`) | **`[384, 512)`** |
| **`dP^T`** (`dO @ V^T`) | `(64, 128)` | `128` | **`64`** (`128 // 2`) | **`[448, 512)`**（与 `dQ` 尾端复用） |
| **`P^T` / `dS^T`** | `(64, 128)` | `64` / `64` (TMEM) | **`0`**（驻留 SMEM `sP` / `sdS_dK`） | 不占用 TMEM（作为 `dV`/`dK` MMA 的 SMEM `A` 操作数） |

由此，`tmem_end = 512 <= 512`，且主循环内 `[256, 320)` (`S`) 与 `[384, 512)` (`dQ` / `dP`) 之间还留有 `64` 列安全隔离带（`[320, 384)`），彻底消除了 `hdim=256` 下的 TMEM 溢出。

### 2.2 为什么 `P / dS^T` 放入 SMEM 后可以完全省去 `sdS_xchg` 交换缓冲区？

在 2-CTA 反向模式中，计算 `S / dP / dV / dK` 时按 KV 行切分（每 CTA 持有一个 `64` 行 KV 子块和全部 `128` 列 Q），而计算 `dQ = dS @ K` 的 2-CTA `M=128` MMA 时则按 **Q 行切分**（CTA `r` 持有 `64` 行 Q 和 `64` 列 KV）。因此，两个 CTA 必须通过集群分布式共享内存（DSMEM，`mapa.shared::cluster`）交换各自计算出的 `dS` 的另一半 Q 列。

在 `M=128` 路径下，`dS^T` 位于 TMEM，必须先写出到专门的 `sdS_xchg` 缓冲区（并与 `sdQaccum` 共享地址别名屏障 `dQaccum_empty_mbar`）再发往对端 CTA。而在 `M=64`（`acc_m64 = True`）路径下（`flash_bwd_sm100.py` L483–539 `_check_dS_exchange_layouts`）：
1. `dS^T` 本身就写往 SMEM 缓冲区 `sdS_dK`（供本 CTA 的 `dK = dS^T @ Q` MMA 作为 `A` 操作数读取），其布局为 `(kv=64, q=128)` K-major，等价于 `(q=128, kv=64)` M-major；
2. 当沿 Q 维按 `cta_group_size = 2` 对半切分时，`sdS_dK[None, None, (None, peer_cta_rank)]` 恰好构成**一整块连续的 `64 × 64` 子张量**，且其 Swizzle 与元素线性偏移与对端 CTA 的 `sdS`（`dQ` MMA 的 `A` 操作数）**逐字节完全一致**！
3. 因此，`self.has_sdS_xchg = False` 且 `self.xchg_aliases_dQaccum = False`：Compute Warp 直接从本 CTA 的 `sdS_dK` 发起单次批量 `cp.async.bulk` DSMEM 传输直达对端 CTA 的 `sdS`，既省去了 `sdS_xchg` 共享内存开销，又解除了 `dS` 产生与 `sdQaccum` 归约之间的假相关屏障。
4. 同时，由于 2-CTA 跨集群 `dQ` 归约需要独立的 `sKt` Tile（`64 × 256 × 2B = 32 KiB`），为严格控制在 SM100 `227 KiB` SMEM 上限以内（仅余 `1 KiB` 裕量），`hdim=256` 配置 `dQ_reduce_ncol = 16, sdQaccum_stage = 2`（两个 `16` 列 Stage 共 `16 KiB`，实测比单个 `32` 列 Stage 更快）。

---

## 3. `AccumTileOrder` 统一抽象与无 SMEM 后处理（`bwd_accum_tile.py`）

提交 `965c9212e6be33c821ee9f2b461564c5c91a2503` 新增了 `flash_attn/cute/bwd_accum_tile.py`（L1–84），将 SM100 反向核写出 FP32 累加器 Tile `(rows, hdim)` 与 `FlashAttentionBackwardPostprocess` 读取该 Tile 的存储顺序统一为三元枚举 `AccumTileOrder`：

$$\begin{aligned}
\mathrm{idx}_{\text{ROW\_MAJOR}}(r, c) &= r \cdot d + c \\
\mathrm{idx}_{\text{TMEM\_FRAG\_1CTA}}(r, c) &= 4r + (c \bmod 4) + 4R \cdot \lfloor c / 4 \rfloor \\
\mathrm{idx}_{\text{TMEM\_FRAG\_2CTA}}(r, c) &= 4(r \bmod 64) + 64d \cdot \lfloor r / 64 \rfloor + (c' \bmod 4) + 512 \cdot \lfloor c' / 4 \rfloor + 256 \cdot \lfloor 2c / d \rfloor,\quad c' = c \bmod (d/2)
\end{aligned}$$

| 内核配置 `(use_2cta_instrs, tile_n)` | `acc_m64` | `dQaccum_order` | `dKVaccum_order` (GQA `dK/dV` 归约缓冲) | Postprocess 全局内存读取方式 |
|---|---|---|---|---|
| `(False, 128)` (1-CTA `M=128`) | `False` | `TMEM_FRAG_1CTA` | `TMEM_FRAG_1CTA` | 每线程直接从 GMEM 加载 `4` 个连续 FP32（每 Warp `512 B` 连续合并访问，**零 `cp.async` SMEM 中转**） |
| `(True, 128)` (2-CTA `M=128`) | `False` | `TMEM_FRAG_2CTA` | `TMEM_FRAG_1CTA` | `dQ` 按 2-CTA Fragment 直读寄存器；`dK/dV` 按 1-CTA Fragment 直读寄存器 |
| **`(True, 64)` (2-CTA `M=64`, `hdim=256`)** | **`True`** | **`TMEM_FRAG_2CTA`** | **`ROW_MAJOR`** | `dK/dV` 经半 Tile (`32 × 256`) 分两阶段写出为纯 Row-Major，Postprocess 按 `ROW_MAJOR` 直读寄存器 |

---

## 4. 确定性实验验证（`experiments/fa4_sm100_hd256_bwd_m64_sim/run.py`）

运行 `python experiments/fa4_sm100_hd256_bwd_m64_sim/run.py --check` 对全部 5 种 SM100 TMEM 映射配置与 3 种 `AccumTileOrder` 双射进行确定性验证：
- `hd256_1cta_m128_overflow`：`tmem_end = 768 > 512`（溢出 `256` 列）；
- `hd256_2cta_m128_serial_overflow`：`tmem_end = 640 > 512`（溢出 `128` 列）；
- `hd256_2cta_m64_serial_valid`：`acc_m64 = True, lane_rows = 2`，`dV(128) + dK(128) + S(64) + dQ(128, 尾端复用 dP=64)` 恰好满足 `tmem_end = 512`，`has_sdS_xchg = False`，`skt_separate_bytes = 32768 B`；
- `AccumTileOrder` 在 `(128, 256)` 的 `32,768` 个元素上 100% 严格双射，且与 `_check_accum_tile_orders` 的 `400` 个 2-CTA `dQ` 采样点和 `24` 个 `M=64` `dK/dV` 半块采样点零误差吻合。

---

## 5. 一手来源与证据链索引

| Source ID | 仓库与提交 / 文件路径 | 证据等级 |
|---|---|---|
| `flash-attention-sm100-hd256-m64-bwd-2026` | `Dao-AILab/flash-attention` commit `2b00d76f222c842b245fd98847223c898bc476db` (`flash_attn/cute/flash_bwd_sm100.py`, blob `9aedeeaadac1`) | `implementation` |
| `flash-attention-sm100-bwd-accum-tile-2026` | `Dao-AILab/flash-attention` commit `965c9212e6be33c821ee9f2b461564c5c91a2503` (`flash_attn/cute/bwd_accum_tile.py`, blob `9f75d17e248d`; `flash_bwd_postprocess.py`, blob `82dddba5df03`; `flash_bwd_sm100.py`, blob `67cb5b41707e`) | `implementation` |
| `flash-attention-sm100-bwd-interface-2026` | `Dao-AILab/flash-attention` commit `965c9212e6be33c821ee9f2b461564c5c91a2503` (`flash_attn/cute/interface.py`, blob `20636fa5d2e7`) | `implementation` |
