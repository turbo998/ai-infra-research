# 互连安全与指令集仿真 Phase 169：NCCL OpenSSL TLS 1.3 PSK 握手路径强制校验与 EFA-GDA 相位位直写、ROCm `rocjitsu` 8B 指令编码捕获与 `gfx1250` Rev-0 VOP3PX 陷阱 PC 修复及 NIXL/Mooncake 演进

## 1. 背景与核心问题

在加密集合通信（NCCL Crypt）、AWS EFA-GDA 头文件展开、AMD `gfx1250` 硬件异常处理以及高频 KV 传输日志系统中，存在三项突出的安全性、编译作用域与内存开销问题：
1. **NCCL OpenSSL TLS 1.3 外部 PSK 握手的绕过风险与 EFA-GDA `BIT(0)` 宏作用域泄漏**：OpenSSL 将 TLS 1.3 外部预共享密钥（External PSK）握手报告为会话复用（Session Reuse）；若客户端未同时验证对等证书并强制检查 `SSL_session_reused()`，攻击者或错误配置可能绕过预期的 PSK 认证路径。此外 EFA-GDA 中调用 `EFA_SET(&ctrl2, EFA_IO_TX_META_DESC_PHASE, wqe_phase)` 会展开依赖局部 `BIT(0)` 宏的掩码，在头文件外引发编译错误。
2. **ROCm `rocjitsu` 竞态检测器缓存反汇编字符串的内存膨胀与 `gfx1250` VOP3PX Trap PC 过度修正**：在 `rocjitsu` 竞态检测器中，为每条指令缓存完整反汇编文本（约 `64 B/指令`）会造成显著内存膨胀；同时 ROCr 二级陷阱处理程序（Trap Handler）中的 VOP3PX 异常 PC 修正逻辑仅适用于 `gfx1250 revision 0`，若无条件应用到后续步进（revision > 0）会导致错误的指令指针偏移。
3. **NIXL 每传输 `INFO` 日志刷屏与 Mooncake MoE 冷启动 JIT 超时**：NIXL 在每次 KV 传输时以 `INFO` 级别打印 `"Selected backend"` 会在每秒数千次传输下打满标准输出与日志采集管道；Mooncake 端到端测试在冷缓存下编译 MoE 算子耗时超过原默认阈值，需将启动超时提升至 `360s`。

---

## 2. 上游源码机制深挖（8 条零重叠提交）

### 2.1 NVIDIA NCCL：TLS 1.3 PSK `SSL_session_reused()` 强制校验与 EFA-GDA 位操作直写
- **`65e7e3ea64` (`2026-09-12`)**：在 `src/misc/crypt.cc` 中配置 OpenSSL 客户端上下文验证对等端证书，并拒绝任何未通过 `SSL_session_reused()` 确认使用了配置 PSK 路径的握手连接。
- **`bb4ddd1cc9` (`2026-09-11`)**：在 `src/include/nccl_device/gin/efa_gda/gin_efa_gda.h` 中用直接位操作（direct bit manipulation）替换 `EFA_SET` 相位设置调用，彻底消除对局部作用域 `BIT(0)` 宏的依赖。

### 2.2 ROCm `rocm-systems`：`rocjitsu` 指令编码捕获与 `gfx1250` Rev-0 Trap PC 限定
- **`ddbfb7d489` (`2026-10-31`, `#14223`)**：在 `emulation/rocjitsu` 竞态检测器中改为直接捕获紧凑的 `8-byte` 原始指令编码（instruction encodings）并按需反汇编，替代原先缓存 `64-byte` 反汇编字符串的做法（追踪内存占用降低 `8.0x`）。
- **`4b70275d5a` (`2026-10-31`, `#13797`)**：在 `amd_gpu_agent.cpp` 与 `trap_handler_gfx12.s` 中将 VOP3PX 陷阱期间的 PC 修正严格限定在 `gfx1250 revision 0`（`PLAT-212250`）。

### 2.3 NIXL、Mooncake & LMCache：日志降级、冷 JIT 360s 超时与 Python 3.14 支持
- **`dd2141ec1d` (`2026-10-12`, `#2321`) & `f2086f5616` (`2026-10-12`, `#2230`)**：NIXL 将每次传输的 `"Selected backend"` 日志从 `INFO` 降级为 `DEBUG`，并恢复 Device API 的用户向 Doxygen 文档注释。
- **`95346cefcb` (`2026-10-19`, `#4765`)**：Mooncake 将端到端测试中 MoE Server 启动超时提高至 `360s` 以容纳冷 JIT 构建。
- **`3981ee5145` (`2026-10-27`, `#5289`)**：LMCache 新增 Python `3.14` 运行时与构建支持。

---

## 3. 确定性离线建模与验证结果

通过 `experiments/nccl_tls_psk_efa_bit_rocjitsu_encoding_gfx1250_phase169/run.py` 验证：

| 子系统与机制 | 修复/优化前基线 | Phase 169 表现 | 核心工程影响 |
|---|---|---|---|
| NCCL Crypt TLS 1.3 PSK & EFA-GDA 相位位 | `4` 次非 PSK 握手放行；`BIT(0)` 编译错误 | **`0` 次非 PSK 放行**；**`0` 宏泄漏错误** | 强制保证 TLS 1.3 PSK 认证路径与头文件自包含 |
| ROCm `rocjitsu` (`10,000` 条指令) & `gfx1250` Trap | 缓存反汇编 `640 KB`；非 Rev-0 误修 PC | 捕获指令编码 **`80 KB`（压缩 `8.0x`）**；**`0` 误修** | 大幅降低仿真竞态检测器内存占用并确保 `gfx1250` 陷阱精确恢复 |
| NIXL 传输日志 (`5,000` tx/s) & Mooncake JIT | `5,000` 条 INFO/s；冷启动超时 | **`0` 条 INFO/s（降为 DEBUG）**；**`360s` 门限** | 消除高吞吐 KV 传输日志 I/O 抖动 |
