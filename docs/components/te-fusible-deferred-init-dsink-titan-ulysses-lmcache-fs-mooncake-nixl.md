# TransformerEngine 可融合算子延迟初始化与 THD 可学习 dSink 守卫、TorchTitan Ulysses NCCL 图注册规避与非特殊扩展词元保留、LMCache `fs_native` 字节预算与隔离 IPC / Mamba 组编辑、Mooncake EFA 双向握手无锁化与 etcd 租约及 NIXL Redis / GPUNetIO 边界加固

> **Phase 55 (`priority: 64`) · 一手源码与 Commit 级深挖**
> - `NVIDIA/TransformerEngine@53e241c5a1c978a56c784b37ee794db739c04df8` (`#3327`) & `c3081a5f2db9de31eab51689a25c70ea48068aa6` (`#3470`)
> - `pytorch/torchtitan@8899cee5bcf520cbeca62ec7b320af8d6b112428` (`#5393`) & `5e27dc87b8cc8f0f1dd0904bec60a704d9781ad8` (`#5470`)
> - `LMCache/LMCache@741b61305902e7d70fd2cf560ad025b251a19f3d` (`#5078`), `786010f4d680653ffe2bf7c88a71684c2fab9495` (`#5693`), & `32f8aa90b7dca6e15a6d7e51c4dff6e2d029b26b` (`#5702`)
> - `kvcache-ai/Mooncake@7ff6bf7f72f15aba7289bc25c87cf491e04efc6f` (`#4704`) & `292e8391b7df0a385092474374660f599958f6c4` (`#4706`)
> - `ai-dynamo/nixl@d4f72ffe6a34be610bcd6578dc0a4d83c3e176b8` (`#2419`) & `8882e2d3484676aa56502fb3d3535c6477159e14` (`#2422`)

---

## 1. TransformerEngine 可融合算子延迟初始化与 THD Learnable `dSink` 守卫 (`#3327`, `#3470`)

- **`te.ops` 显式延迟初始化契约 (`#3327`)**：为所有可融合算子（`BasicLinear`、`Bias`、`LayerNorm`、`GroupedLinear`、`Sequential`）补齐 `reset_parameters()`（其中已量化的 `BasicLinear.weight` 通过原地 `self.weight.quantize_(weight)` 保持参数引用不变），并移除 `pre_first_fuser_forward()` 中静默物化 `meta` 权重的隐式分支——若首次前向时参数仍位于 `meta` 设备则直接抛出异常。
- **cuDNN `< 9.26.0` THD Learnable Softmax 守卫收窄 (`#3470`)**：特化内核 `head_dim_v ∈ {64, 128, 256}` 已正确索引变长 `Stats`，因此仅在 `cudnn_runtime_version < 92600` 且 `head_dim_v ∉ {64, 128, 256}` 的通用内核路径上拒绝执行。

---

## 2. TorchTitan Ulysses `NCCL_GRAPH_REGISTER=0` 与非特殊扩展 Token 保留 (`#5393`, `#5470`)

- **Ulysses CP 在旧版 NCCL 下的图注册规避 (`#5393`)**：在 `NCCL < 2.32.3`（`NVIDIA/nccl#2362` 修复前）中，NCCL 可能在另一个缓冲区视图上复用未完成的 CUDA Graph IPC 映射，引发 Kimi-K3 等长序列训练出现静默算错或非法访存；`configure_nccl_for_ulysses()` 在进程组初始化前自动设置 `NCCL_GRAPH_REGISTER=0`。
- **保留 `AddedToken.special=False` 词元 (`#5470`)**：将 `_infer_special_tokens` 中的 `add_special_tokens` 替换为 `add_tokens`，使 Qwen3 的 `<think>`、`</think>`、`<tool_call>`、`</tool_call>` 在默认解码中不被误删，同时继续过滤真正的特殊词元（如 `<|im_end|>`）。

---

## 3. LMCache `fs_native` 字节预算 / 隔离 IPC / Mamba 组编辑与 Mooncake / NIXL 传输加固 (`#5078`, `#5693`, `#5702`, `#4704`, `#4706`, `#2419`, `#2422`)

- **LMCache 存储与多进程平台 (`#5078`, `#5693`, `#5702`)**：`#5078` 将 `fs_native` 读并发深度与 `num_workers` 解耦，按字节预算控制（默认 `kDefaultReadMaxBytesInFlight = 32 MiB`）；`#5693` 在隔离 IPC 下支持 GLM-5.3-Flash 等 `dim-0` 填充 KV 池视图；`#5702` 新增第 4 类 Mamba KV Cache Group Edit 以支持 Ascend 共享池切分的独立 `conv_state` / `ssm_state` 张量。
- **Mooncake EFA 双向主动握手防死锁与 etcd 租约 TTL (`#4704`, `#4706`)**：`#4704` 在 `EfaEndPoint::setupConnectionsByActive()` 发起握手 RPC 期间释放 `RWSpinlock`，改用仅主动端持有的 `active_setup_mutex_` 串行化，并引入 `generation_` 计数器拦截跨断连/切网卡的陈旧 RPC 回复，彻底消除双端同时拨号死锁；`#4706` 支持 `--ha_leader_lease_ttl_sec`（`2..60s`，默认 `5s`）。
- **NIXL Redis 事件循环保活与 GPUNetIO 边界解析 (`#2419`, `#2422`)**：`#2419` 使用 `EVLOOP_NO_EXIT_ON_EMPTY` 防止末个连接断开时 libevent 提前退出导致在途命令 Future 永久挂起；`#2422` 使用 `std::from_chars` 严格校验 GPUNetIO 元数据并按声明长度解析通知载荷，完整保留内嵌 `\x00` 的二进制载荷。

---

## 4. 确定性复现实验验证

运行 `python experiments/te-fusible-deferred-init-dsink-titan-ulysses-lmcache-fs-mooncake-nixl/run.py --check` 全部不变量通过。
