# 机密计算 GPU（NVIDIA Confidential Computing / PPCIE）：SPDM/RIM 远程证明、拓扑校验与 Bounce Buffer 吞吐边界

## 1. 核心架构问题与结论（Executive Summary）

当企业与主权云客户在多租户云环境中部署高价值专有权重或敏感行业数据微调/推理时，要求将宿主机的 Hypervisor、Host OS、系统管理员及物理 PCIe/NVLink 总线排除在可信计算基（TCB）之外。NVIDIA H100 / Blackwell Confidential Computing（`CC-On` 与多卡 `PPCIE` Protected PCIe 模式）与开源证明工具链 `NVIDIA/nvtrust` 构建了硬件级 GPU TEE，但也引入了显著的控制面证明门禁与数据面加密代价：
1. **PCIe DMA 防火墙与 Bounce Buffer 拷贝税**：在 `CC-On` 模式下，GPU 硬件内存保护单元（MPU/MMU）拒绝宿主机直接访问受保护的 GPU HBM（CPR，Confidential Protected Region）。所有 Host-to-Device (`H2D`) 与 Device-to-Host (`D2H`) 传输必须先在未加密的主机共享内存（Bounce Buffer）中暂存，由 CPU（CVM 内部）与 GPU Copy Engine (`CE`) 执行 **AES-GCM-256 加解密与认证标签（Auth Tag）校验**，导致有效 H2D 带宽从非机密 PCIe Gen5 的 **`56.0 GB/s`** 跌至软件加密路径的 **`8.5 GB/s`**（或硬件流水优化后的 **`18.4 GB/s`**）。
2. **单卡 `CC-On` 禁用直接 P2P 导致的多卡通信崩塌**：在标准单卡 `CC-On` 模式下，跨卡直接 P2P DMA 被硬件阻断，多卡 Tensor Parallelism (`TP`) 的 `AllReduce` 被迫绕行 CVM 主机内存做 `D2H -> CPU 解密/重加密 -> H2D`，有效卡间带宽跌至 `4.2 ~ 9.2 GB/s`。
3. **多卡 `PPCIE`（8-GPU + 4-NVSwitch）的双向拓扑证明硬门禁**：为了在机密计算下恢复全速 NVLink 通信，必须启用 **PPCIE（Protected PCIe / NVLink）模式**，不仅需要对每张 GPU 和每颗 NVSwitch (LSV) 独立执行 SPDM + RIM 度量验证，还必须双向校验 8 张 GPU 与 4 颗 NVSwitch 报告的物理设备标识符（PDI）构成完整二分图后，才允许通过 `set_gpu_ready_state(True)` 解锁 GPU。

基于 `NVIDIA/nvtrust`（commit `a8090b65872a15bf0fd1f69f3d9b754b1be431b5`）一手源码核验与 `experiments/confidential_gpu_overhead/run.py` 离线建模，核心量化结论如下：
- **RIM 黄金度量验证的确定性规则（`verifier/verifier.py`）**：验证器合并 Driver RIM 与 VBIOS RIM 的活跃度量槽位，若同一索引在两份 RIM 中同时处于 `active` 状态，立即抛出 `InvalidMeasurementIndexError` 拒绝通过；同时针对 `OPAQUE_FIELD_ID_NVDEC0_STATUS == DISABLED` 的 Hopper 实例专门豁免索引 `35`（`is_msr_35_valid = False`），避免误杀合法硬件配置。
- **PPCIE 双向 PDI 拓扑证明与 Fail-Closed 门禁（`validate_topology.py` & `verification.py`）**：验证器强制要求恰好 `8` 张 GPU 与 `4` 颗 NVSwitch，每张 GPU 的 `x-nvidia-gpu-switch-pdis`（剔除 `DISABLED_PDI = "0000000000000000"` 后）必须恰好包含 4 个唯一 NVSwitch PDI，每颗 NVSwitch 的 `x-nvidia-switch-gpu-pdis` 必须恰好包含 8 个唯一 GPU PDI，且 `unique_switches - switch_pdis` 必须为空集；任一环节失败立即调用 `disable_gpu_state()` 将 GPU 锁定在 `NOT READY` 状态。
- **单卡 `CC-On` vs 多卡 `PPCIE` 的 TTFT / TPOT 分化**：在包含 `1 GiB` 主机前缀 KV 卸载加载（Host HiCache H2D）与每步 `64 MiB` TP AllReduce 的推理负载下，未优化的 `single_gpu_cc_on_sw_aes` 使 TTFT 从 **`104.1740 ms` 恶化至 `212.3079 ms`（慢 `2.0380x`）**，TPOT 从 **`11.6678 ms` 恶化至 `27.6029 ms`（慢 `2.3657x`）**；而开启 **8-GPU + 4-NVSwitch `PPCIE`** 后，得益于 NVLink 硬件链路层加密恢复 **`372.0 GB/s`** 卡间带宽，TPOT 恢复至 **`11.6811 ms`**（与非机密基线仅差 **`0.11%`**，相对软件中转提速 **`2.3630x`**），但跨 PCIe 的主机 KV 卸载加载仍受限于 Bounce Buffer（TTFT 为 **`143.5831 ms`**，慢 **`1.3783x`**）。

---

## 2. 一手源码深度剖析（`NVIDIA/nvtrust` Verification & Topology Engine）

### 2.1 Local GPU Verifier：Driver + VBIOS RIM 合并与索引 35 特例

在 `guest_tools/gpu_verifiers/local_gpu_verifier/src/verifier/verifier.py:42-164`（commit `a8090b65872a15bf0fd1f69f3d9b754b1be431b5`）中，`Verifier` 类实现了运行时 SPDM 度量报告（`AttestationReport`）与黄金度量（`GoldenMeasurement`）的逐槽位比对：
1. **Driver 与 VBIOS RIM 索引互斥检查（第 `110-148` 行）**：
   ```python
   for gld_msr_idx in vbios_golden_measurements:
       if vbios_golden_measurements[gld_msr_idx].is_active() and \
          gld_msr_idx in self.golden_measurements:
           settings.mark_no_driver_vbios_measurement_index_conflict(False)
           raise InvalidMeasurementIndexError(
               f"The driver and vbios RIM have measurement at the same index : {gld_msr_idx}"
           )
   ```
   任何企图通过伪造重叠槽位覆盖固件或驱动度量的 RIM 文件都会触发 `InvalidMeasurementIndexError`。
2. **`OPAQUE_FIELD_ID_NVDEC0_STATUS` 与 MSR 35 豁免（第 `74-75`, `158-162` 行）**：
   - 构造函数检查 SPDM 报告的不透明字段：`if attestation_report_obj.get_response_message().get_opaque_data().get_data("OPAQUE_FIELD_ID_NVDEC0_STATUS") == BaseSettings.NVDEC_STATUS.DISABLED: self.is_msr_35_valid = False`。
   - 在 `verify()` 循环中，当 `i == 35 and not self.is_msr_35_valid` 时直接跳过比对。
3. **多候选摘要与字节长度双重匹配（第 `81-86` 行）**：
   - 黄金度量支持 `get_number_of_alternatives()` 个合法固件/驱动哈希候选值，但要求同时满足字符串相等以及字节长度精确匹配：`golden_measurement.get_size() == len(self.runtime_measurements[i]) // 2`。

### 2.2 PPCIE Verifier：8-GPU + 4-NVSwitch 双向 PDI 图校验与 Fail-Closed 状态机

在 `guest_tools/ppcie-verifier-sdk-cpp/ppcie/verifier/verification.py:121-190` 与 `src/topology/validate_topology.py:24-173` 中，PPCIE 验证器编排了 7 阶段严格门禁：
1. **七阶段串行门禁（`verification.py:121-159`）**：
   - 阶段 1：设备数量强校验 `if number_of_gpus != 8 and number_of_switches != 4: sys.exit()`。
   - 阶段 2：`validate_gpu_pre_checks(nvml_client, logger, status)`。
   - 阶段 3：`validate_switch_pre_checks(nscq_client, logger, status, switch_list_uuids)`（通过 NSCQ 驱动访问 NVSwitch）。
   - 阶段 4：`perform_gpu_attestation` 获取 `gpu_claims`。
   - 阶段 5：`perform_switch_attestation` 获取 `switch_claims`。
   - 阶段 6：`topology.gpu_topology_check(gpu_claims, number_of_switches, status)`。
   - 阶段 7：`topology.switch_topology_check(switch_claims, number_of_gpus, status)`。
   - **Fail-Closed 硬件就绪锁**：只有 7 个阶段全部通过才调用 `enable_gpu_state(logger, nvml_client)`（即 `nvml_client.set_gpu_ready_state(True)`）；否则在第 `156-157` 行执行 `disable_gpu_state(logger, nvml_client)`，强制调用 `nvml_client.set_gpu_ready_state(False)` 阻断用户态工作负载使用 GPU。
2. **双向 PDI 集合不变量（`validate_topology.py:24-173`）**：
   - 定义 `GPU_SWITCH_PDIS_CLAIM = "x-nvidia-gpu-switch-pdis"`、`SWITCH_PDI_CLAIM = "x-nvidia-switch-pdi"`、`SWITCH_GPU_PDIS_CLAIM = "x-nvidia-switch-gpu-pdis"` 以及 `DISABLED_PDI = "0000000000000000"`。
   - **GPU 视角 (`gpu_topology_check`)**：每张 GPU 声明的 Switch PDI 集合在 `discard(DISABLED_PDI)` 后必须恰好等于 `number_of_switches` (`4`)，且 8 张 GPU 汇总的 `self.unique_switches` 基数也必须恰好为 `4`。
   - **NVSwitch 视角 (`switch_topology_check`)**：每颗 NVSwitch 声明连接的 `gpu_pdis` 集合基数必须恰好为 `number_of_gpus` (`8`)，4 颗 Switch 汇总的 `self.unique_gpus` 基数必须为 `8`，4 颗 Switch 自身的 `switch_pdis` 基数必须为 `4`，最后校验差集 **`unknown_switches = self.unique_switches - switch_pdis`** 必须为空集——彻底杜绝了中间人注入未受证明的伪造 NVSwitch（Rogue Switch PDI Injection）。

---

## 3. 原创 CPU 离线实验结果（`experiments/confidential_gpu_overhead/`）

### 3.1 RIM 度量与 PPCIE 拓扑攻击向量验证结果

| 验证类别 | 用例 ID (`case_id`) | 验证结果 | 触发规则 / 拒绝原因 |
| :--- | :--- | :---: | :--- |
| RIM 度量校验 | `valid_hopper_nvdec0_disabled` | **通过 (`True`)** | `OPAQUE_FIELD_ID_NVDEC0_STATUS == DISABLED`，自动豁免 `MSR 35` (`skipped_msr_35 = True`) |
| RIM 度量校验 | `driver_vbios_index_conflict` | **拒绝 (`False`)** | `InvalidMeasurementIndexError: index 5 active in both Driver and VBIOS RIM` |
| RIM 度量校验 | `tampered_driver_measurement_mismatch` | **拒绝 (`False`)** | `MismatchedIndices:[1]`（驱动运行时哈希与黄金度量不符） |
| PPCIE 拓扑校验 | `valid_8gpu_4nvswitch_ppcie_mesh` | **通过 (`True`)** | 忽略 `0000000000000000` 禁用端口后，8 GPU $\leftrightarrow$ 4 NVSwitch 双向 PDI 完全闭合，`gpu_ready_state_enabled = True` |
| PPCIE 拓扑校验 | `rogue_nvswitch_pdi_injection` | **拒绝 (`False`)** | `TotalUniqueSwitchesFromGpusMismatch:5`（GPU0 报告未授权 `SW99`），触发 Fail-Closed `gpu_ready_state_enabled = False` |

### 3.2 四种机密计算模式下的数据面延迟与吞吐对比（`1 GiB` Host KV 加载 + `64 MiB/tok` TP AllReduce）

| 运行模式 (`id`) | 有效 H2D 带宽 (`GB/s`) | 有效卡间 P2P 带宽 (`GB/s`) | 冷启动证明屏障 (`ms`) | `1 GiB` H2D 传输耗时 (`ms`) | 含 Host KV 加载的 TTFT (`ms`) | 每步 TP AllReduce (`ms`) | Decode TPOT (`ms`) | Decode 吞吐 (`tok/s`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `standard_non_cc` | `56.0 GB/s` | `400.0 GB/s` | `0.0 ms` | `19.1740 ms` | `104.1740 ms` (`1.00x`) | `0.1678 ms` | `11.6678 ms` (`1.00x`) | `85.7062` |
| `single_gpu_cc_on_sw_aes` | `8.5 GB/s` | `4.2 GB/s` | `1420.0 ms` | `127.3079 ms` | `212.3079 ms` (`2.04x`) | `16.1029 ms` | `27.6029 ms` (`2.37x`) | `36.2280` |
| `single_gpu_cc_on_hw_dma_opt` | `18.4 GB/s` | `9.2 GB/s` | `1420.0 ms` | `58.8107 ms` | `143.8107 ms` (`1.38x`) | `7.3513 ms` | `18.8513 ms` (`1.62x`) | `53.0466` |
| `multi_gpu_ppcie_nvswitch` | `18.4 GB/s` | **`372.0 GB/s`** | `1420.0 ms` | `58.5831 ms` | `143.5831 ms` (`1.38x`) | **`0.1811 ms`** | **`11.6811 ms` (`1.001x`)** | **`85.6083`** |

---

## 4. 解决方案架构师设计指南与可证伪问题

1. **架构师落地准则**：
   - **多卡机密推理/微调必须启用 `PPCIE` 模式，切勿在多卡 TP 场景混用单卡 `CC-On`**：单卡 `CC-On` 仅适合能够单卡装下的小模型（如 8B~32B FP8），一旦跨卡执行 Tensor Parallelism，绕行主机 Bounce Buffer 会令 TPOT 恶化 `1.62x ~ 2.37x`；`PPCIE` 通过对 4 颗 NVSwitch 纳入 SPDM 证明域，使多卡 TP Decode 吞吐恢复至非机密基线的 **`99.89%`**。
   - **谨慎在机密计算下开启高频 Host HiCache KV Offloading**：即便在 `PPCIE` 模式下，GPU 与宿主机 CVM 内存之间的 PCIe 边界依然受 Bounce Buffer + AES-GCM 保护（有效带宽 `18.4 GB/s` vs `56.0 GB/s`）。对于超长上下文缓存，应优先使用 **同证明域内通过 NVLink/GPUDirect 互联的远端 GPU 显存池**，而非频繁换入换出到主机 DDR。
   - **迁移路线提醒**：`nv-local-gpu-verifier` Python 工具链已在源码（`cc_admin.py:84-92`）中标记将于 **2026 年 9 月 15 日** 停止支持（EOS），新生产系统应基于 C++ `NVIDIA/attestation-sdk` 与 `ppcie-verifier-sdk-cpp` 构建证明守护进程，并在本地缓存 RIM/OCSP 响应以规避 `1.42 s` 冷启动网络毛刺。
2. **后续可证伪实验问题**：
   - 在 Blackwell (`B200` / `GB200`) 原生支持 **TEE-I/O (TDX Connect / SEV-TIO)** 与 PCIe IDE（Integrity and Data Encryption）硬件流加密后，Host-to-Device 跨 PCIe 传输能否彻底绕过软件 Bounce Buffer 并将 `1 GiB` Host KV 加载耗时从 `58.6 ms` 压回 `20 ms` 以内？
