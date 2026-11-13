# 来源、版权与公开边界

本项目的分析、图示、合成输入和程序独立编写。第三方文档、论文、标准及源码仅作为研究来源，保留原作者与其许可证/版权；不提交其全文、图片、PDF、模型、数据集或源码镜像。

NVIDIA CUDA/NCCL 文档及技术文章属于相应权利人；AMD ROCm 文档属于相应权利人；UEC 规范/发布材料属于相应权利人。源码研究链接固定到 NVIDIA/NCCL 与 linux-rdma/rdma-core 的精确提交，仅讨论实际读取范围，不重新署名其实现。NCCL 所读文件有 Apache-2.0 声明，rdma-core 所读手册有 OpenIB.org BSD 声明；这些局部声明不被扩展到其他材料。

JEDEC 标准正文没有取得，不推测其版权授权或伪称已阅读。访问受限材料只记录缺口，使用合法公开资料做有限替代。

公开推送前必须排除秘密、签名 URL、私人路径、个人/客户数据、公司内部资料及未经授权的第三方材料。程序检查只辅助审阅，不能证明“绝无泄露”。

目前不为全部内容添加统一开源授权。自有代码/文档与第三方引用的授权需分别处理；公开访问并不消除版权。

Round 92 的 DDP 梯度通信压缩研究链接 PyTorch `v2.10.0` 固定提交的 PowerSGD/default hooks 实现与版本化文档；PyTorch 保留其 BSD-3-Clause 许可和相应版权。原始合成模型、H5、报告和图示未复制第三方实现。LMSYS 2026-11-12 公告仅作为当日背景链接，不作为梯度压缩性能证据，也未镜像其文字或图片。

Round 93 的对比式训练研究链接 Sentence Transformers 固定提交的 CachedMNRL/GradCache mixin 与 GradCache 原论文 v2。Sentence Transformers 保留 Apache-2.0 许可及版权，论文保留其作者权利。独立编写的标量 encoder、InfoNCE 数值检查与内存/计算预算不是第三方实现镜像，也未复现或冒用论文的模型质量及性能结论。

Round 94 的词表并行交叉熵研究链接 NVIDIA Megatron-LM 固定提交的实际 loss 实现、旧兼容入口及上游 LICENSE。版权与文件级许可例外保留于上游；没有复制其实现或许可全文，也不从 GitHub 的 NOASSERTION 元数据臆断整个仓库许可。原创标量统计、有限差分、错误协议对照与字节/启动预算不是 GPU 基准或第三方代码镜像。

Round 95 的流式卷积状态研究链接 Kyutai Moshi 固定提交的卷积模块、reset 变更与上游 LICENSE；GitHub 仓库许可元数据为 Apache-2.0，文件中的 Kyutai/Meta 原始权利声明保留于上游。原创标量离线公式、流式 adapter、overlap-add 与缓存预算未复制实现或模型权重，也不代表 Mimi 实测、音质或服务延迟。

Round 96 的推荐稀疏 Embedding 表研究链接 Meta/PyTorch TorchRec 固定提交的 TW/RW/CW distributor、mean divisor 与 2026-11-11 subgroup root 修复；仓库许可元数据为 BSD-3-Clause，上游权利保留。原创 jagged 数值模型、表梯度有限差分、逻辑字节预算与本地 CPU 计时不复制实现，也不是 TorchRec/FBGEMM/GPU benchmark 或推荐质量评估。

Round 97 的 GNN 邻居采样研究链接 PyG 固定提交的 NeighborLoader 文档与 MIT LICENSE；seed-first、n_id、fanout/hop 与子图策略的上游权利保留。原创依赖闭包、特征梯度、确定性邻居 fixture 与独立 uniform 子集枚举未复制上游实现，不代表 PyG/GPU 执行、真实采样吞吐或训练质量。

Round 98 的 Fourier 神经算子研究链接 neuraloperator 固定 SpectralConv 与 2026-02-24 Hermitian enforcement 变更；上游 MIT 权利保留。原创可分离/直接 DFT、空间循环卷积与复数缓存预算未复制实现或模型权重；没有执行 neuraloperator/PyTorch/cuFFT，也未复现上游 GPU artifact、PDE 质量或峰值 HBM。

Round 99 的逐样本裁剪研究链接 Opacus 固定 Linear norm sampler、全局 norm、weighted replay 与 optimizer scaling；上游 Apache-2.0 权利保留。原创标量梯度、Gram 与固定噪声向量未复制实现，不是 Opacus/PyTorch 实测或私人训练；无 sampler、accountant、Gaussian/secure RNG、epsilon/delta 保证或 GPU 指标。

Round 100 的半结构化稀疏研究链接 PyTorch 固定 sparse classes、conversion 与 LICENSE；上游权利保留，GitHub 许可元数据为 NOASSERTION，不擅自确认单一 SPDX。原创 2:4 nibble、剪枝与标量乘法未复制实现；逻辑编码不冒充物理 CUTLASS/cuSPARSELt 格式，不代表 GPU 性能、FP8/ROCm 验证或模型质量。

Round 101 的 CTC 研究链接固定 pyctcdecode decoder（Apache-2.0）与 NeMo 集成；上游权利保留。原创路径枚举、prefix log-sum、独立 target forward 和 CPU 计时没有复制 decoder 实现或模型权重；合成声学概率不代表 WER/CER、校准 confidence、LM/BPE/时间戳一致性或原生 ASR/GPU 性能。

Round 102 的集合监督研究链接固定 RT-DETR matcher 与 criterion（Apache-2.0）；上游权利保留。原创穷举、subset DP、标量监督梯度和 DDP count 算术未复制检测器实现或权重，不是 Hungarian 算法实现、原生 SciPy 计时、focal/GIoU 复现或 detector AP/FPS benchmark。

Round 103 的 Neural ODE 研究链接固定 Diffrax adjoints（Apache-2.0）与 torchdiffeq adjoint（MIT）；上游权利保留。原创固定 Euler、离散 reverse、block replay 与连续增广反向时间对照未复制实现或权重；不是框架 adaptive solver 复现、online/binomial checkpointing、科学质量或 GPU/peak-memory benchmark。

Round 104 的 Gaussian rasterization 研究链接固定 gsplat tile/depth、alpha gates 与 2026-09-01 int64 offset 变更（Apache-2.0）；上游权利保留。原创 square-support CPU 分桶、混合、梯度和宽索引算术未复制实现或场景资产，不是 3D projection、ellipse culling、gsplat/CUDA、图像质量或 peak-memory benchmark。

Round 105 的 multiresolution hash-grid/LoD 研究链接固定 tiny-cuda-nn 2026-09-29 native hard/soft LoD 变更及 grid/helpers；已直接检查 LICENSE.txt 的 BSD-3-Clause 条款，GitHub license API 返回 NOASSERTION。上游权利保留。原创二维标量 gather、展开参考、碰撞 VJP、固定 LoD 权重和有界分配模型未复制 source 实现、训练权重或渲染资产；不是 native CUDA/JIT、完整二阶/LoD-input 导数、图像质量或 peak-HBM benchmark。

Round 106 的 sparse-convolution coordinate-map 研究链接固定 spconv conv/ops（Apache-2.0）；repo pin 2024-12-15，conv.py 最后变更 2023-03-23，不声称 2026 新发布。上游权利保留。原创 batched 1D scalar map、dense reference、tied adjoint、支持/行身份控制与 selected Native pair budget 未复制 source、权重或数据集；不是 native CUDA/3D kernel、模型质量或 peak-HBM 复现。
