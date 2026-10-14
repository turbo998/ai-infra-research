# Innovator arch tradeoffs：Fireworks KV-Outer 字节模型、Together Sequoia 投机树 Roofline 与 Groq/Cerebras 片上 SRAM 边界

原创 Python 标准库、CPU/offline 算例；不运行 GPU、Groq LPU 或 Cerebras WSE 硬件。
关联组件正文见：
- [张量 Layout/Dtype 到编译器与 SM100 Kernel 的分派链路](../../docs/components/kernel-dispatch-layout.md)
- [AI Infra 创新架构深度研究：Fireworks、Together、Groq 与 Cerebras](../../docs/components/ai-infra-innovators.md)

## 三组可重算模型

1. **Fireworks SM100 M3 KV-Outer 稀疏注意力字节模型（复现 `fw-ai/minimax-kernels` 的 `fio1x` / `fioAct` / `cio` / `ioAlt`）**：
   - 验证 `docs/perf-results-b200.md` 与 `perf-results-b300.md` 中的分析型字节定义：
     - `fio1x = Q_once + Hkv × distinct_blocks × kv_block + partials`
     - `fioAct = Q_restream + Hkv × distinct_blocks × kv_block + partials`（其中 `Q_restream = total_pairs × (Hq / Hkv) × D × input_bytes`）
     - `ioAlt = Q_once + Hkv × total_pairs × kv_block + partials`（无 KV 复用基线）
     - `cio = partials + inv_map + final_output`
   - 对于 `Tq=4096, Hq=16, Hkv=1, topK=16, BF16`：
     - 当 `distinct_blocks = 16`（`M = 4096`）时，算例精确重算出 `fio1x = 0.294650 GB`（报告打印 `0.295`）、`fioAct = 0.546308 GB`（报告打印 `0.546`）、`cio = 0.293863 GB`（报告打印 `0.294`），而无 KV 复用的 `ioAlt` 高达 `4.588055 GB`（KV-outer 节省 `8.40×` 总前向 I/O，KV 读取本身压缩 `4096×`）。
2. **Together Sequoia 投机树验证（Tree Verification）算术强度与 Roofline 拐点**：
   - 在 `70B FP16`（`140 GB` 权重，H100 级 `3.35 TB/s` HBM、`989 TFLOP/s`）下：
     - `K = 1` 贪婪解码算术强度仅 `1.0 FLOP/B`（严重 Memory-bound，单步 `41.79 ms`）；
     - `K = 64` 投机树验证将算术强度拉升至 `64.0 FLOP/B`（仍低于硬件拐点 `295.22 FLOP/B`，故 HBM 权重读取时间不变，扣除 `3.5 ms` Draft 开销后单步 `45.29 ms`，若期望接受 `4.2` tokens，净吞吐提升 `3.88×`）；
     - 反例：当树宽盲目扩大至 `K = 512`（算术强度 `512.0 FLOP/B > 295.22 FLOP/B`）时，验证阶段跨过拐点变为 **Compute-bound**（计算耗时 `72.48 ms > 41.79 ms`），即便期望接受升至 `5.5` tokens，净加速比反而跌回 `2.94×`。
3. **Groq TSP/LPU 与 Cerebras WSE-3 的片上 SRAM 容量硬边界**：
   - 对 `70B INT8`（`70 GB` 权重，按 `75%` 权重可用率估算，预留激活/KV/暂存区）：
     - 单颗 Groq TSP/LPU（`220 MiB` SRAM，可用权重约 `173 MiB`）无法单卡放下模型，至少需要 **405 颗芯片**通过 C2C 链路组成确定性 Daisy-Chain 流水线；
     - 单片 Cerebras WSE-3（`44 GB` 片上 SRAM，可用权重约 `33 GB`）同样跨越单晶圆纯 SRAM 驻留边界（需 `>= 3` 片晶圆或依赖 MemoryX 外部权重流式加载）。

## 运行与校验

```powershell
python experiments\innovator_arch_tradeoffs\run.py
python experiments\innovator_arch_tradeoffs\run.py --check
python -m unittest tests.test_innovator_arch_tradeoffs
```
