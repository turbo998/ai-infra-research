# EAGLE-3 / MTP Speculative Tree Verification & Roofline Collapse Experiment (`own-offline`)

本实验基于 `SGLang` (`python/sglang/srt/speculative/eagle_utils.py`、`adaptive_spec_params.py`、`eagle_disaggregation.py`) 一手源码，量化链式草稿 vs 树状草稿（`N_verify = 16 / 32 / 64`）在不同接受率 $\alpha \in \{0.58, 0.74, 0.86\}$ 与并发度 $BS \in [1, 64]$ 下的期望接受长度 $\mathbb{E}[\tau]$ 与**算术强度屋脊线坍塌（Roofline Collapse）**边界：
1. **低并发带宽红利（$BS=1$）**：在代码/数学场景（$\alpha=0.86$）下，`tree_s5_k6_v32` 每步提交 `5.364` 个 Token，端到端加速达 **`2.605x`**（`tree_s6_k8_v64` 达 **`2.647x`**）；
2. **高并发屋脊线坍塌（$BS=32$）**：当 $BS=32$ 仍维持 `num_verify_tokens=32` 时，单步有效查询 Token 膨胀至 `1,024`（达到 H100 FP8 算术强度拐点 `229.6` 的 **`4.46x`**），加之 `TreeMaskMode.FULL_MASK` 掩码与非因果 KV 放大，净吞吐退化至 **`0.889x`**（在 `general_chat` $BS=16, N_{\text{verify}}=64$ 下更跌至 **`0.660x`**）；
3. **自适应降阶算法（`AdaptiveSpeculativeParams`）**：随 $BS$ 从 `1 -> 4 -> 8 -> 32` 动态将候选从 `tree_v64 -> tree_v32 -> chain_s5_k1_v6` 降阶，可在 $BS=32$ 恢复 **`2.422x`** 加速（`9,871.8 tok/s`）。

## 运行与校验

```powershell
python experiments\eagle3_speculative_tree\run.py
python experiments\eagle3_speculative_tree\run.py --check
```
