# FlashAttention-4 SM100 MLA KV-Head 缓存键隔离、2-CTA dQ 寄存器直载、SM103 `setmaxnreg.dec` 陷阱修复与 DeepEP/DeepGEMM/NIXL 跨层内存序及上下文安全

**English companion**: [fa4-sm100-mla-kvhead-key-2cta-dq-reg-sm103-setmaxnreg-deepep-gin-nixl-obj.en.md](fa4-sm100-mla-kvhead-key-2cta-dq-reg-sm103-setmaxnreg-deepep-gin-nixl-obj.en.md)

## 1. 背景与核心问题

在 Blackwell SM100/SM103 原生算子、GPU-Initiated Networking（GIN）混合互连以及多后端 KV/对象存储传输（NIXL）中，近期上游提交暴露并修复了五类极具代表性的底层静默数据损坏（Silent Data Corruption, SDC）与硬件非法指令陷阱：

1. **FlashAttention-4 SM100 MLA 编译缓存键碰撞导致静默漏写或越界别名**：当两个 MLA 调用具有相同的 GQA 比值 `nheads_per_kv = 64` 但不同的 KV 头数——例如 `(nheads_q=64, nheads_kv=1)` 与 `(nheads_q=128, nheads_kv=2)`——若 JIT 编译缓存键仅包含 `nheads_per_kv` 而未包含 `num_head_kv`，后一次调用会错误复用前一次编译的内核。
2. **FlashAttention-4 SM100 2-CTA Backward Postprocess 共享内存占用过高**：2-CTA MMA 的 `dQaccum`（FP32，每 tile $128 \times 128 \times 4 = 65,536\;\text{B}$）若先经过共享内存再归约，会占据双倍于 BF16 `sdQ`（$32,768\;\text{B}$）的 SMEM 配额。
3. **FlashAttention-4 SM103（GB300）`hd128`/`hd192` Decode `setmaxnreg.dec` 非法指令陷阱**：在 `q_stage=1`、CTA 512 线程（启动寄存器上限 $65536 / 512 = 128$）下，未加封顶的公式算出 `num_regs_other = 136 > 128`，导致 PTX `setmaxnreg.dec.sync.aligned.u32 136` 尝试将寄存器配额“缩减”到高于启动配额的数值，直接触发硬件非法指令异常。
4. **DeepEP GIN 门铃顺序与混合 NVLink+RDMA 系统级内存栅栏缺失，以及 DeepGEMM MegaMoE 槽位重用竞争**：`engram_fetch.cuh` 在最后一次 RDMA `get` 上若未先敲响门铃就调用 `flush_async`，会导致尾部请求挂起；`comm.cuh` 在跨 NVLink 与 RDMA 的 `ncclTeamTagWorld` 上缺少 `fence.acq_rel.sys`；DeepGEMM `mega_moe.cuh` 消费者在释放 `task_info` 空屏障前缺少 `fence.acq_rel.cta`，导致调度器提前覆写元数据。
5. **NIXL S3 加速对象引擎的无上下文工作线程注册失败与 UCX `ERR_HANDLING_MODE_NONE` 空回调段错误**：应用层工作线程（未执行过 CUDA 调用，当前 `CUcontext` 为空）调用 `S3AccelObjEngineImpl::registerMem` 时，`cuMemObjGetDescriptor` 因缺少当前上下文返回 `rc=1`（`NIXL_ERR_BACKEND`）；同时 `nixlUcxEp` 在 `UCP_ERR_HANDLING_MODE_NONE` 下仍置位 `UCP_EP_PARAM_FIELD_ERR_HANDLER`，触发远端节点退出时的空回调崩溃。

---

## 2. 第一手源码实现路径审计

### 2.1 FlashAttention-4 SM100 MLA `num_head_kv` 缓存键隔离（`#2978`, commit `94e22c906678`）

在 `flash_attn/cute/interface.py` 的 `_flash_attn_fwd` 编译缓存键元组中，新增项：
```python
num_head_kv if qv is not None else None,  # MLA kernels specialize on nheads_kv
```
**故障机理对比**（`tests/cute/test_flash_attn_mla.py`）：
- **先运行 `(64, 1)` 再运行 `(128, 2)`**：复用了单 KV 头内核，Grid 仅启动了 1 个 KV 头的 tile，导致第 2 个 KV 头（head 1）完全未被写入（残留未初始化内存或 0）。
- **先运行 `(128, 2)` 再运行 `(64, 1)`**：在单 KV 头输入上启动了 2 个 KV 头的 tile，多出的“幽灵 head 1”越界读取并覆写了下一个 token 的 Q 行！将 `num_head_kv` 纳入缓存键后彻底消除了两类碰撞。

### 2.2 FlashAttention-4 SM100 2-CTA `dQaccum` 寄存器直载与 `hd256` 确定性反向传播（`#2960`, `#2952`）

- 在 `flash_attn/cute/flash_bwd_postprocess.py`（`47e91f1f7dd8`）中，当 `self.use_2cta_instrs` 为真时，`dQaccum` 直接根据 2-SM MMA TMEM 布局（$M=128$ 跨 CTA pair 按 `m // 64 = cta_rank` 切分，CTA 内 lanes 0–63 持有列 `[0, hdim/2)`、lanes 64–127 持有列 `[hdim/2, hdim)`）从 GMEM 加载进寄存器，共享内存仅需容纳 BF16 `sdQ`（从 $65,536\;\text{B}$ 降至 $32,768\;\text{B}$，节省 **50% SMEM**）。
- 在 `3451a2a67a24`（`#2952`）中，打通了 `hd256` 2-CTA backward 的 `deterministic=True` 路径，采用无原子操作的 CTA 局部归约。

### 2.3 FlashAttention-4 SM103 `setmaxnreg.dec` 寄存器上限封顶（`#2950`, commit `037e808b7239`）

在 `flash_attn/cute/flash_fwd_sm100.py` 中，SM103 1-CTA causal `hd128`（`num_regs_softmax=176, num_regs_correction=64`）与 2-CTA `hd192` 在 `q_stage=1`、`threads_per_cta=512` 下：
$$\text{uncapped} = \left\lfloor \frac{512 - 176 \times 1 - 64}{3 - 1} \right\rfloor \div 8 \times 8 = 136 > \frac{65536}{512} = 128$$
修复后将其显式封顶于启动寄存器上限：
```python
self.num_regs_other = min(
    (512 - self.num_regs_softmax * self.q_stage - self.num_regs_correction)
    // (3 - self.q_stage) // 8 * 8,
    65536 // self.threads_per_cta,
)
```
将 `num_regs_other` 限制为 `128`，消除了 GB300 上 `hd128`/`hd192` decode 的非法指令崩溃。此外 `#2758`（`1fc0221183e7`）补齐了 SM120 的自动 `num_splits` 启发式计算。

### 2.4 DeepEP GIN 门铃/系统栅栏与 DeepGEMM MegaMoE CTA 栅栏（`#752`, `#715`, `#441`）

1. **DeepEP `engram_fetch.cuh`（`a56d6156febc`, `#752`）**：新增共享内存数组 `deferred_token_idx[kNumRDMAPeers]` 与 `deferred_src_byte_offset[kNumRDMAPeers]`，延迟每个 peer 的最后一次 RDMA `get` 并在 `flush_async` 之前显式敲响门铃，避免队列尾部请求未被硬件感知。
2. **DeepEP `comm.cuh`（`11d1cab604bb`, `#715`）**：在 `gin_barrier_wo_local_sync` 中，当 `std::is_same_v<team_t, ncclTeamTagWorld>` 时插入 `ptx::fence_acq_rel_sys()`，确保混合 NVLink + RDMA 域在网格同步前完成系统级可见性收敛。
3. **DeepGEMM `mega_moe.cuh`（`78b69000794d`, `#441`）**：在 `MegaMoEScheduler::release_task_info()` 调用 `task_info_empty_barriers[sched_stage_idx ^ 1].arrive(0u)` 之前插入 `ptx::fence_acq_rel_cta()`（`asm volatile("fence.acq_rel.cta;" ::: "memory")`），防止编译器或硬件重排导致调度器在工作 warp 读完元数据前覆写槽位。

### 2.5 NIXL S3 加速对象引擎 `scopedCudaDevice` 与 UCX 错误处理器掩码（`#2380`, `#2423`）

1. **`S3AccelObjEngineImpl::registerMem`（`d0319acbdb8e`, `#2380`）**：在 `src/plugins/obj/s3_accel/engine_impl.cpp` 中引入 RAII 守卫 `scopedCudaDevice`，先调用 `cuCtxGetCurrent(&saved_)` 保存调用线程当前上下文（即便为 `nullptr`），随后调用 `cudaSetDevice(mem.devId)` 激活目标 VRAM 所在设备的 primary context 完成 `cuMemObjGetDescriptor` 注册，最后在析构函数中通过 `cuCtxSetCurrent(saved_)` 精确还原调用线程状态。
2. **`nixlUcxEp`（`c8178f563a9a`, `#2423`）**：在 `src/plugins/ucx/ucx_utils.cpp` 中，默认 `ep_params.field_mask` 仅包含 `UCP_EP_PARAM_FIELD_REMOTE_ADDRESS | UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE`，仅当 `err_handling_mode != UCP_ERR_HANDLING_MODE_NONE` 时才追加 `UCP_EP_PARAM_FIELD_ERR_HANDLER` 并绑定 `err_cb_wrapper`。

---

## 3. 可复现实验与検証总结

配套实验 `experiments/fa4_sm100_mla_kvhead_key_2cta_dq_reg_sm103_setmaxnreg_deepep_gin_nixl_obj/run.py` 検証结果：

| 组件与提交 | 缺陷根因 | 修复前状态 | 修复后不变量 | 证据级别 |
|---|---|---|---|---|
| FA4 SM100 MLA Cache Key (`94e22c90`) | `(64,1)` 与 `(128,2)` 同为 `nheads_per_kv=64` | 碰撞：head 1 漏写或幽灵 head 越界别名 | 键入 `num_head_kv`，缓存键严格正交 | `implementation` + `own-offline` |
| FA4 2-CTA `dQaccum` (`47e91f1f`) | FP32 `dQaccum` 经由 SMEM 中转 | 占用 $65,536\;\text{B}$ SMEM | GMEM 直载寄存器，SMEM 降至 $32,768\;\text{B}$（**-50%**） | `implementation` + `own-offline` |
| FA4 SM103 `setmaxnreg` (`037e808b`) | `q_stage=1` 算出 `num_regs_other=136` | `136 > 128` 触发 GB300 非法指令陷阱 | `min(136, 65536 // 512) = 128` | `implementation` + `own-offline` |
| DeepEP / DeepGEMM (`a56d6156`, `11d1cab6`, `78b69000`) | 尾请求门铃缺失 / 缺 `sys` 与 `cta` 栅栏 | 尾包挂起、跨网弱序、槽位提前覆写 | 延迟尾 `get` 敲门铃 + `fence.acq_rel.{sys,cta}` | `implementation` + `own-offline` |
| NIXL S3 Accel & UCX (`d0319acb`, `c8178f56`) | 工作线程无 `CUcontext` / 空 `ERR_HANDLER` 位 | `rc=1 NIXL_ERR_BACKEND` / UCX 段错误 | `scopedCudaDevice` 借还上下文 + 按需置位掩码 | `implementation` + `own-offline` |
