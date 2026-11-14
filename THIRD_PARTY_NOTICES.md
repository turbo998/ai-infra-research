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

Round 107 的 VFR timestamp/clip 研究链接固定 TorchCodec 2026-11-09 low-level APIs stable 变更、decoder PTS index 与 sampler policies（BSD-3-Clause）；不声称新发明 VFR 算法。上游权利保留。原创整数 tick interval/oracle、平均 FPS 对照、clip padding、重复 slots 和 logical tensor budget 未复制 source、编码媒体、数据集或权重；不是 TorchCodec/FFmpeg/NVDEC、实际解码帧数、视频质量或 peak-HBM benchmark。

Round 108 的 nullable List buffer 研究链接固定 Arrow 2026-11-13 required-object None guards、ArrayData offsets/validity 与 List flatten（Apache-2.0，LICENSE.txt）；合法 nullable buffers 与缺失必需对象不同。上游权利保留。原创 immutable List<int32>、显式 canonical copy、CPU 计时、buffer payload 与 SVG 未复制 source、数据集或资产；不是 native constructor crash、Arrow allocator、NumPy/LargeList execution、RSS 或 peak-HBM benchmark。

Round 109 的 symbolic shape 研究链接固定 JAX 2026-11-04 polynomial optimization、scope、univariate solving 与 export 文档（Apache-2.0）；上游权利保留。原创 affine solver、有界穷举、inequality 域检查、CPU 计时与 signature/padding policy 未复制实现、模型或数据集；不是 native parser/compiler、完整 polynomial engine、上游优化 speedup、实际 compilation events、padding-mask 等价性或 peak-HBM benchmark。

Round 110 的 count-aware SyncBatchNorm 研究链接固定 PyTorch wrapper/native moments 与 BatchNorm estimator 文档；2026-06 collective API migration、2026-07 native pin，不称十一月新发布。已检查 LICENSE primary redistribution terms，GitHub SPDX 为 NOASSERTION；上游权利保留。原创 scalar moments、exact rational/finite-difference references、CPU timing 与 selected payload 未复制实现、模型或数据集；不是 native invstd packet、CUDA Graph empty-rank、DDP、collective failure、wire latency 或 peak-HBM benchmark。

Round 111 的 distributed AUROC 研究链接固定 TorchMetrics 2026-09 compatibility cleanup、raw cat/counter SUM state、tie grouping、ROC endpoint 与 integration（Apache-2.0）；不称为新发明 ROC。上游权利保留。原创 rational ROC、pairwise oracle、阈值计数与 payload 未复制实现、权重或数据集；source-derived endpoint control 和单列 origin variant 不冒充 native failure/fix，missing-class 策略不冒充库 warning parity；不是真实模型质量、native distributed latency 或 peak-HBM benchmark。

Round 112 的 Parquet pruning 研究链接固定 DuckDB 2026-11 reader、statistics validity 与 comparison propagation（MIT）；11 月 scratch-buffer 改动仅作时间线，不把既有剪枝称为新发明。上游权利保留。原创整数 SQL 三值 interpreter、row oracle、CPU timing 与 selected IO budget 未复制实现、权重或数据集；constant-bound/nullability analogue 不冒充 native string regression，未测 Parquet 解码、object-store IO、DuckDB native latency 或 peak memory。

Round 113 的指数直方图研究链接固定 OpenTelemetry Go 2026-11 bucket-logic deduplication、2026-10 circular-storage、public interval 与 aggregation contract（Apache-2.0）；不称为指数直方图新发明。上游权利保留。原创 exact-rational mapping、immutable merge、interval oracle 与 CPU timing 未复制实现或私有观测；仅非负值、scale -10..0，counter capacity 不冒充 retained SDK total，未测 native concurrency、collector、OTLP 或生产吞吐。

Round 114 的 RoPE 研究链接固定 FlashInfer 2026-11-10 regular dimension guard 与 cached rotary/layout repair（Apache-2.0）；上游权利保留。原创 scalar planes、Q/K permutation、complex relative-angle oracle、CPU timing 与 selected payload 未复制上游实现、模型权重或私有 prompts；不冒充 native failure/fix、GPU speedup、真实 checkpoint conversion、模型质量或 peak-HBM benchmark。

Round 115 的 DLPack 研究链接固定 2026-11-04 header / Python spec（Apache-2.0）；该提交只将版本 1.3 改为 1.4，既有 ownership/stream 规则不称为新发明。上游权利保留。原创 scalar descriptor、toy capsule、partial-order enumeration、Python buffer 计时与视觉未复制上游实现；不执行 native PyCapsule/C ABI/GPU，不把枚举比例冒充竞态概率或把 buffer view 计时冒充 DLPack speedup。

Round 116 的 Mel 前端研究链接固定 Transformers 2026-11-12 audio dependency migration（Apache-2.0）；上游权利保留。原创 scalar filter、rational triangle/frame-start oracles、projection timing 与视觉未复制上游实现、模型权重或录音。Source float32-before-normalization、Slaney 与 transpose 契约已区分；不冒充 native librosa bit-exact reproduction、ASR accuracy 或 GPU benchmark。

Round 117 的采样研究链接固定 FlashInfer 2026-11-13 basic categorical batch-invariant 更新（Apache-2.0）；不自动推广到已有 filtered APIs，上游权利保留。原创 exact rational filters/CDF、supplied-uniform identity、CPU lookup 与视觉未复制上游实现或模型权重；不复现 native Philox/canonical float tree，不把 C-backed bisect/Python-loop 对照冒充 GPU sampler speedup。

Round 118 链接固定 NVIDIA/cccl DeviceRadixSort（component header SPDX BSD-3），上游权利保留；2026-11-10 修改为 ADL qualification 维护，不冒充新算法。原创 key/byte-pass、numeric/bit-string oracles、CPU timing 与视觉未复制 CUB 实现；原型显式拒绝 NaNs，不认证 native 内部 zero encoding、NaN order、workspace 或 GPU speed。

Round 119 链接固定 PyTorch 2026-11-12 implicit gradient fan-in guard（BSD-style repository license），上游权利保留。原创 scalar slots、IEEE rounding、Fraction oracle、CPU timing 与视觉未复制 autograd 实现；source native tests 不算本地复现，flag 不等于跨次 .grad accumulation ban、general reduction ban 或 float64/memory 保证。

Round 120 链接固定 PyTorch SparseAdam 与 functional API（BSD-style），既有 mask/coalesce 契约与 8 月 public-API 日期已分开，不冒充新算法。Scalar updates、Decimal/Fraction oracles、CPU grouping 与视觉原创，未复制 upstream optimizer；native precision/speed/HBM/state compaction 未复现，原型 finite-positive-denominator policy 比 native 更窄。

Round 121 链接固定 ONNX QuantizeLinear reference（Apache-2.0），上游权利保留。Exact Fraction/BigInt per-axis int8/uint8、rounding oracle、CPU timing 与视觉原创，未复制 upstream；8 月 float6 更新单列且未评测，不认证 native division/overflow、FP6/FP8/INT4、GPU speed 或 model quality。

Round 122 链接固定 MAPIE quantile helpers（BSD-3-Clause）与既有 conformal theory；11 月 10 日 educational tree 更新非新 theorem。Exact ranks、synthetic exhaustive holdouts、CPU selection 与视觉原创，未复制 upstream；coverage 只指固定 exchangeable world enumeration，不冒充 real-data/per-group/单 query confidence 或 native MAPIE 复现。

Round 123 链接固定 linear_operator Cholesky helper（MIT），3 月 modern API migration 非新算法。Scalar factor/solve、Fraction elimination、CPU cache timing 与视觉原创，未复制 upstream；selected observation-noise shift 与 native numerical rescue jitter 区分，不冒充 native rescue/GPU 或 calibrated posterior quality。

Round 124 链接 2026-11-14 PyTorch regional AOTI capture 更新及 BSD-style LICENSE。Bounded storage/view/event model、index oracles、CPU samples 与视觉原创，未复制 implementation；不复现 native arbitrary-object traversal、autograd、export/compile/GPU。Detach/live storage、clone alias guards、pending status 分开标注。

Round 125 链接固定 SciPy resample_poly 既有规则（BSD-3-Clause）。Exact rational nonzero-phase FIR、独立 dense zero-insertion oracle、CPU float paths 与视觉原创，未复制 implementation；11月 tree 更新不冒充新 resampling algorithm；custom FIR/zero boundary control 非 native Kaiser/audio quality/streaming/GPU 复现。

Round 126 链接固定 ONNX Resize reference 与9月30日 integer-output align_corners 修复（Apache-2.0）。Fraction1-Dgeometry/interpolation、独立 hat-basis oracle、CPU grid timing与视觉原创，未复制implementation；非 nativeONNX/cubic/ROI/antialias/visionquality/GPU复现。

Round 127 链接固定 Mamba classic selective-scan ordered affine operator/prefixcallback（Apache-2.0）。ScalarFractionrecurrence/tree、独立productexpansion、CPUfloatpaths与视觉原创，未复制implementation；zero-resetpacking为原创extension，非nativeAPI承诺；JulyMamba3Blackwellfix属于不同kernel、未执行，非11月新scan算法。

Round 128 链接固定 ONNX NMS reference 与10月28日 optional-scalar/omission修复（Apache-2.0）。ExactIoU/classgreedy、独立candidate-removal/cell-areaoracles、CPUgeometry与视觉原创，未复制implementation；strictscorepredicate/tiepolicy明确为pinnedreference，不认证nativebackends/GPU/detectionquality。

Round 129 链接固定 Flink boundedwatermark、statusvalve、windowoperator（Apache-2.0）。Discreteevent/windowmodel、boundary/invariantoracles、CPUaggregation与视觉原创，未复制implementation；allidlemaxflush/resumealignment按pinnedsource，变更日期查询TLS失败不虚构日期；非nativeclocks/triggers/checkpoint/sink/exactly-once复现。

Round 130 链接固定 PyTorch SDPA、CausalBias 与 MHA 文档/源码（BSD-style）。位置掩码、均匀零logits标量模型、排序位置oracle、CPU路径和视觉原创，未复制实现；Nov14regionalAOTI更新不是新掩码语义发明；空行null为原创策略，非nativeSDPA/GPU/softmax/quality复现。

Round 131 链接固定 ONNX ScatterND29 spec、reference 与11月12日mean能力变更（Apache-2.0）。整数坐标/切片模型、独立destination-first oracle、CPU sum/count与视觉原创，未复制实现；mean含原值一次、floor、negativealiases、none重复拒绝、k0whole tensor按固定合同；不认证nativeONNX/FP/atomic/GPU。

Round 132 链接固定 Dynamo LoRA HRW 与 allocator interface（所读文件Apache-2.0 SPDX）。显式score-table策略、独立subset oracle、CPUtopK/fullsort与视觉原创，未复制实现；nativeBLAKE3编码仅说明、未执行；sticky、partialunderfill、allfullbasicHRW回退与missingusageeligibility按固定策略，非nativehash/controller/load/residency/GPU认证。

Round 133 链接固定 Ceph EC/profile文档及COPYING；doc/*为CC-BY-SA-3.0。此处原创事实总结、GF256/Vandermonde矩阵、独立Lagrange oracle、CPUparity与视觉，未复制上游实现或文档文本；非Ceph/ISA-L byte parity、FastEC、耐久度或nativeperformance复现。经典条带预算与FastEC布局显式区分。

Round 134 链接固定 RedisBloom chain（RSALv2/SSPLv1/AGPLv3选择）、deps/bloom（所读BSDheader）及LICENSE。显式坐标模型、finite-universe/set oracle、层界、计数删除反例、CPUprobes与视觉原创，未复制上游实现；理想sizing/层界不等于nativeMurmur/Redis/FPR/allocator/数据库收益。

Round 135 链接2026-11-14读取的AWS S3 multipart workflow、completion API与limits。原创byte/opaque-token ledger、独立flatten oracle、HTTPbody fixture、CPUassembly与视觉，未复制上游实现或文案；无云凭据/操作，非nativeETag/checksum/SDK/network或globalcheckpointatomicity认证。

Round 136 链接固定 Redis hyperloglog.c 的 rank、register MAX 与 Ertl histogram estimator。所读文件头许可为 RSALv2 / SSPLv1 / AGPLv3 选择，部分内容另有 BSD3 来源说明；上游权利保留。原创有限 16-bit digest、classic/linear-counting teaching 模型、字符串 oracle、CPU rank paths 与视觉未复制实现；非 native Murmur/Ertl、Redis memory/accuracy/latency 或生产误差复现，渐近 RSE 非逐样本保证。

Round 137 链接固定 RoaringFormatSpec（已检查 Apache-2.0 LICENSE）与 CRoaring 11 月 9 日 Roaring64 C++ query capability patch；上游权利保留。原创 uint32 container payload codec、strict run-saving policy、interval oracle、Python CPU paths 与视觉未复制上游示例/图/实现；非完整 portable reader/writer、native runOptimize/64-bit rank/select/SIMD、memory 或 ACL freshness/latency 认证。

Round 138 链接固定 ONNX DepthToSpace/SpaceToDepth schema/reference、8 月 31 日 opset28 mode/function-body 变更（Apache-2.0）与固定 PyTorch TorchScript exporter（BSD-style）；上游权利保留。原创整数 gather/scatter、inverse/adjoint、direct-op gate、CPU planning 与视觉未复制实现；非 native converter/kernel、strided tensor、GPU/fusion、zero-copy 或 image-quality 认证。

Round 139 链接固定 PyTorch ConvTranspose1d、requested output_size 与 native slow transpose shape guard（BSD-style）；上游权利保留。原创 onechannel integer scatter、独立 finite forward-matrix oracle、adjoint、分组 payload 与 CPU timing/视觉未复制实现；output_padding<stride 为更窄原型域；非 native kernel/BLAS/cuDNN、grouped execution/workspace、streaming 或音视频质量认证。

Round 140 链接固定 liburing manpages（所读 SPDX LGPL-2.0-or-later）、LICENSE(MIT)、COPYING(LGPL-2.1)；保留文件级上游权利，不以单一许可概括仓库。原创 uint32 SQ/CQ、one-shot buffer ledger、deque oracle、CPU list paths 与视觉未复制文档或native实现；非Linux/atomics/SQPOLL/multishot/notification/cancellation/CQ-overflow或native IOPS/memory认证。

Round 141 链接固定 Linux memory policy 文档、weighted-interleave ABI与mm/mempolicy.c（source SPDX GPL-2.0-only）；保留上游权利，无代码镜像。原创 ideal first-write/page-index interval models、expanded-cycle oracle和CPU lookup/视觉非native allocator、cpuset rebind、fallback、THP、shared cache、auto weights、migration或NUMA带宽认证。

Round 142 链接 SQLite WAL/transaction/checkpoint/file-format 文档及固定public-domain wal.c；无实现镜像。原创ledger、immutable transaction oracle、CPU lookup与视觉，另有same-thread隔离SQLite合成fixture。未执行race/corruption/crash测试；本机3.49.1回补状态未认证，部署须核验官方WAL-reset修复。无原生checkpoint性能或持久性认证。

Round 143 链接11月14日SQLite public-domain dbstat fix/回归与文件格式；无实现镜像。原创field-context/leaf-header几何、opaque cells、interval oracle、CPU decoding与视觉；另读取自己生成的原生空表文件。dbstat模块缺失指标显式null，未编译执行新patched binary；非record/overflow/tree/file完整性、native dbstat性能认证。

Round 144 链接固定PyTorch all_to_all_single、NCCL测试入口及BSD-style LICENSE；权利保留，无实现镜像。原创split矩阵、单destination scalar transport、direct-owner oracle、CPU inverse lookup与视觉；无原生PyTorch/NCCL/MPI/GPU、network、PSK/TLS、零长度backend、top-k副本或async stream认证。

Round 145 链接11月14日PyTorch nccl2 TCP TLS feature、API/options/process gate及native tests；BSD-style权利保留，无实现镜像或test密钥复制。原创metadata-only automaton、declarative oracle、CPU synthetic-label preflight与60-solid视觉；不生成/读取真实key，无native crypto、communicator、GPU、并发/rotation或全面数据路径安全认证。

Round 146 链接固定PyTorch ShrinkNCCL.cpp与NVIDIA shrink/split/default/abort/sharing API事实；PyTorch BSD-style权利保留，NVIDIA事实改写，无实现镜像。原创parent-relative association、cyclic-row/checkpoint ledger、独立slice oracle、CPU planner与60-solid视觉；无native NCCL/GPU、abort/shared-resource、checkpoint I/O、optimizer/step恢复或wire/peak认证。

Round 147 链接ONNX Nov13BJT opset29Pack/Unpack schema/reference/function/tests与Apache-2.0 LICENSE；权利保留，无实现镜像。原创rank-two axis/radix reservoir、explicit-bitstream oracle、BigInt parity、CPU packing与60-solid视觉；原型strict errors不代表schema invalid-input保证，无ONNX/reference/function/ORT/GPU、数值quantization或runtime峰值认证。

Round 148 链接ONNX Nov13BJTfinite FP4/FP6 NaN文档/test澄清及Apache-2.0 LICENSE；权利保留，无实现镜像。原创exact rational RNE/zero/saturation/NaN policy、piecewise/exhaustive oracles、CPU tick lookup与60-solid视觉；policy block非native exception保证，无ONNX Cast/GPU、float32预舍入、scales/calibration或model quality认证。

Round 149 链接ONNX Nov13BJT STRING helper byte-preservation修正、TensorProto schema及Apache-2.0 LICENSE；权利保留，无实现镜像。原创typed records/有限wire subset、literal/random oracles与CPU fragment assembly；trim仅错误模型，非旧NumPy复现，无native helper/protobuf roundtrip、general parser或network性能认证。

Round 150 链接ONNX Nov13BJT ImageDecoder RGB/BGR reference修正、opset20schema/tests及Apache-2.0 LICENSE；权利保留，无实现镜像。原创post-decode六mode逻辑、alpha/palette/bilevel与channel-order oracle、CPU copied reorder；无codec/Pillow/NumPy/ONNX、Grayscale、native stride/peak或model quality认证。
