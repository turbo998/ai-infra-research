# UEC 1.1：规范发布怎样改变研究问题

证据状态：**announcement**。UEC 官方新闻索引标注 2026-10-12，发布正文已读；完整 PDF 规范、NIC/switch 实现、兼容列表和硬件性能尚未验证。不能将发布日和核验日的巧合泛化到其他来源。

来源：[UEC 官方发布](https://ultraethernet.org/ultra-ethernet-consortium-releases-specification-1-1-advancing-ai-and-hpc-networking/)，来源 ID `uec-1-1`，主张 ID `uec-1-1-announcement`。

## 发布正文提供的研究线索

| 机制 | 从公告取得的信息 | 架构研究问题 |
|---|---|---|
| Scale-up / UFH / Unified Local | 讨论 packet overhead、转发头及面向本地域的简化操作 | 访问、路由、ordering/atomic 和 GPU 软件语义如何对应？不能直接称“替代 NVLink” |
| UET Verbs | 讨论现有应用语义与迁移兼容、内存映射/同步组 | 哪些 verbs 语义是映射、哪些依赖 NIC/provider？应用移植需要哪些变化？ |
| In-Network Collectives | 讨论将部分归约放到网络，减少特定路径通信 | 哪些 datatype/reduction/拓扑有实现？精度、拥塞、多租户和失败回退怎样处理？ |
| CSIG | 讨论数据包携带沿路拥塞信息及端侧控制 | 遥测的可用性、开销、精度、公平性与现有运维栈如何验证？ |
| PCM | 讨论可编程拥塞管理 API | 编程接口、验证/隔离、兼容与生产运维门槛是什么？ |

上述都是公告描述的机制，不是本项目对所有条文或实现的确认。

## 为什么“通信最高 2x”不等于应用 2x

网络内归约可改变通信路径的数据移动数量，但总体训练/推理还包含计算、其他 collectives、同步、存储与恢复。就算某个通信环节改善，Amdahl 限制和 workload 分布仍影响端到端收益。

应核验基线是否同算法、拓扑、精度/归约顺序和质量；忽略 CPU/GPU、NIC/switch 的边界会得到错误分母。本项目不采用发布正文中的加速倍率作为实测或容量规划参数。

## 必须通过的采用门槛

1. 读取合法公开完整规范并固定版本/章节，明确支持 profile、操作、ordering/一致性与异常语义。
2. 核验具体 NIC/switch/provider/runtime release，而非仅厂商加入联盟或宣布兼容。
3. 在相同模型/数据/精度/拓扑和质量门槛下比较有效吞吐、尾延迟、功率、失败/恢复。
4. 保留不支持/回退路径、拥塞/多租户条件与负面结果。

规范/源码研究可继续；真实硬件实验需单独获批。当前成熟度结论仅为“官方宣布了 1.1 规范及相关机制”，不是“已经能生产替换现有 GPU fabric”。
