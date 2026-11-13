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
