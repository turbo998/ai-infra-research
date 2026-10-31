# NCCL 2.32.3 `nccl4py` 绑定生成、NIXL Mooncake 插件治理、Mooncake TENT T-One 双栈 E2E 与 LMCache XPU 子进程探测（Phase 162）

> **证据分类**：上游源码级提交分析（`implementation`：`893470119efc`、`0d651e079119`、`f38665ac9a56`、`131900c264fb`、`2653767b80f0`、`7cf2232ee203`、`9507de50d60b`、`9a1654ad0f53`）+ 本项目确定性标准库模型验证（`own-offline`：`experiments/nccl4py_2323_rccl_nixl_mooncake_tent_tone_lmcache_xpu_phase162/run.py`）。

## 1. 核心架构问题与代码实现路径

1. **NCCL 2.32.3-1 与 `nccl4py` 原生 Python 绑定同步生成（`893470119efc`、`0d651e079119`）及 RCCL 安全依赖升级（`f38665ac9a56`、`131900c264fb`）**：
   - 直接从 NCCL `2.32.3` 头文件生成 `nccl4py` Cython/Python 绑定，确保 C ABI 符号零漂移（`0` 不匹配符号）。
2. **Mooncake TENT `T-One` 与 Classic 双栈端到端验证（`9507de50d60b`）及 NIXL Mooncake 插件治理（`2653767b80f0`）**：
   - 在每晚流水线中同时运行 TENT `T-One` 与 `classic` 两种传输模式（协议栈覆盖率从 `1` 种提升至 `2` 种，即 `2.0x`）。
3. **LMCache Intel XPU 子进程探测中的 Mooncake Wheel 导入保持（`9a1654ad0f53`）**：
   - 修复了 XPU 环境下派生子进程探测 Mooncake 时丢失 wheel `PYTHONPATH`/导入上下文的问题，将子进程探测失败数从 `4` 降至 `0`。

## 2. 确定性实验验证

```bash
python experiments/nccl4py_2323_rccl_nixl_mooncake_tent_tone_lmcache_xpu_phase162/run.py --check
```
