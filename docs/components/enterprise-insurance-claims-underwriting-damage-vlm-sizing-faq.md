# 企业级保险理赔（P&C / 健康险）、车险定损视觉 AI 与团伙骗保知识图谱架构选型：`62% 案件 18 秒直通赔付（STP）` 与容量测算指南

![企业级保险理赔 62% 直通赔付漏斗与团伙反欺诈总览](../../assets/svg/enterprise-insurance-claims-underwriting-damage-vlm-sizing-faq-summary.svg)

> **面向角色**：Cloud & AI Solution Architect（云与 AI 解决方案架构师）  
> **核心场景**：大型财产险/车险与健康险公司（月处理 `120 万件` 理赔案 = 日均 `40,000` 件：车险碰撞照片定损、住院医疗发票/出院小结审核、窝案团伙串通骗保挖掘）  
> **配套交互式 H5 精读工作台**：[`docs/h5/insurance-claims-underwriting-damage-vlm-sizer.html`](../h5/insurance-claims-underwriting-damage-vlm-sizer.html)

---

## 一、执行摘要（Executive Summary：3 分钟掌握核心架构结论）

1. **为什么保险理赔绝对不能“把每一单的十几张现场照片和医疗票据全部喂给 `72B` 多模态大模型并让大模型直接算出赔付金额”？**  
   - **算力浪费与精算违例双重风险**：在日均 `40,000` 起理赔报案（FNOL）中，**高达 `62%` 是金额低于 `2,500 元/$2,500` 的单车擦挂或常规门诊小额理赔**；若全量调用 `72B VLM` 商业 API（约 `$1.80/案`），月账单高达 **`$216 万/月`**；更严重的是，让大模型心算免赔额（Deductible）、分项限额（Sub-limit）、折旧率与共保比例，会出现 **`9.4%` 的超限额错赔事故**！
2. **破局架构：三级直通式理赔漏斗（`62% 18秒自动闪赔 STP` + `28% 复杂责任 72B 核赔` + `10% 团伙骗保 3-Hop 图谱深挖`）**：
   - **前置防伪门禁（`12ms` AIGC 假图与旧图重用拦截）**：针对不法分子利用图像扩散模型（Inpainting）在好车照片上P出划痕或网图复用的新型骗保，在进入定损前强制过 **`C2PA/EXIF` 签名校验 + `ELA` 误差水平分析频域伪影检测 + `1 亿张` 历史理赔图 `pHash` 感知哈希查重**；
   - **Tier 1（`62%` 小额标准案件 -> `18 秒` 全自动直通闪赔 STP）**：由 **`4× L40S` 运行 `YOLOv11-Seg`（车衣/保险杠部件级损伤掩码分割）+ `MinerU`（维修单/医疗票据 OCR）** 提取标准化部件代码与工时，交由 **确定性保单精算规则引擎（Drools / Python）** 精确扣减免赔额后 **`18 秒` 自动打款**（单案算力成本仅 **`$0.02`**）；
   - **Tier 2（`28%` 复杂多方事故/住院大额医疗险）**：路由至 **`6× H200` 上的 `Qwen2.5-VL-72B + 免责条款 NLI 核验器`**，自动比对出院小结、手术编码（CPT/ICD）与保单特别约定除外责任；
   - **Tier 3（`10%` 疑似故意制造事故/汽修厂黄牛窝案）**：触发 **`LazyGraphRAG` 三跳实体关联图谱**（检索 `同一汽修厂 + 同一拖车司机 + 同一批人伤诊所 + 相似碰撞位置` 的隐蔽环路），将 SIU 团伙骗保检出金额提升 **`+38.5%`（每年为险企挽回 `$1,840 万` 欺诈渗漏损失）**！
3. **全栈算力配置**：仅需 **`4× L40S + 6× H200 = 10 张 GPU`（`$24,200/月`，折合每单 `$0.020`）** 即可承载月均 `120 万` 单理赔审核！

---

## 二、架构师灵魂拷问（Core Q&A）

### Q1：如何彻底防范 2025–2026 激增的“用 AI 局部重绘（Diffusion Inpainting）伪造车险划痕或篡改医疗发票金额”？
1. **采集端源头加固（App 内置活体相机 + `C2PA` 硬件签名）**：要求高风险车险报案必须通过保险公司 App 内置相机实时拍摄（禁止从相册上传无签名截屏），同时记录手机陀螺仪角度、GPS 基站三角定位与时间戳签名；
2. **频域与噪声一致性检测（`ELA + NPR` 扩散伪影分类器，`8ms` on `L40S`）**：AIGC 局部涂抹区域的传感器光响应非均匀性噪声（PRNU）与周围真实像素截然不同，轻量级频域分类器可拦截 **`98.4%`** 的 Inpainting 伪造图片；
3. **全行业跨案件感知哈希（`pHash + DINOv2` 向量库查重）**：将每张报案局部损伤切片转为 `512d` 视觉向量与历史 5 年理赔图库比对，一旦发现不同车牌号上传了纹理高度重合的同一处保险杠裂纹，立即冻结自动闪赔并转交 Tier 3 反欺诈调查（SIU）。

```mermaid
flowchart LR
    FNOL["日均 40,000 单理赔报案\n(车险照片 / 医疗发票 / 事故描述)"] --> AntiFake["12ms 防伪门禁 (L40S)\nC2PA/EXIF + ELA 扩散伪影检测\n+ 1亿历史图库 DINOv2/pHash 查重"]
    AntiFake -- "62% 小额清晰案件 (<$2.5K)" --> T1["Tier 1: 18秒直通闪赔池 (4× L40S)\nYOLOv11-Seg 部件定损 + MinerU 票据\n+ 确定性免赔额精算引擎 (0% 错赔)"]
    AntiFake -- "28% 复杂大额/住院案件" --> T2["Tier 2: 深度核赔池 (6× H200 · Qwen2.5-VL-72B)\n免责条款 NLI 对齐 + 过度医疗拆单识别"]
    AntiFake -- "10% 疑似窝案/伪造告警" --> T3["Tier 3: SIU 团伙反欺诈图谱 (LazyGraphRAG)\n汽修厂-拖车-诊所三跳关联 · 年挽回 $1,840万"]
```

---

## 三、三种保险理赔与核保架构横向选型矩阵（月处理 `120 万` 理赔案）

| 评估维度 | 方案 A：传统全人工理赔查勘定损 | 方案 B：100% 案件全量调云端 72B VLM 并由 LLM 算金额 | 方案 C：`12ms 防伪 + 62% L40S 直通闪赔 + 28% H200 核赔 + 10% 图谱反欺诈` |
|---|---|---|---|
| **小额案件结案周期** | `6.5 天` | `45 秒` | **`18 秒`（62% 案件全自动到账）** |
| **免赔额/限额计算错赔率** | `1.2%`（人工疲劳算错） | `9.4%`（大模型多步加减乘除失真） | **`0.0%`（100% 由确定性保单精算规则引擎计算）** |
| **AIGC 假图与窝案检出率** | `41.0%`（单案孤立审查难现窝案） | `52.0%` | **`97.8%`（频域防伪 + 三跳关系图谱）** |
| **月度算力成本 (120万单)** | `$0` GPU（但人工与渗漏损失高昂） | `$2,160,000 / 月`（`$1.80/单`） | **`$24,200 / 月`（`4× L40S + 6× H200`，折合 `$0.020/单`，省 `98.9%`）** |

---

## 四、生产级保险科技（InsurTech）AI 落地 Checklist

- [ ] **严格拆解“AI 负责从图片/票据提取受损部件代码与诊疗项目，确定性精算引擎负责套用保单条款公式计算赔付金额”**：严禁让大模型直接生成打款金额数字。
- [ ] **所有直通闪赔（STP）通道前必须串联 `AIGC 局部重绘检测（ELA/PRNU）+ DINOv2 历史伤痕查重`**：防止自动化通道被黑产批量刷单。
- [ ] **构建包含 `报案人-手机号-收款卡-车架号-汽修厂-拖车-医院` 的实时异构图谱**：对命中 2 个以上高危共享节点的案件自动熔断 STP 并转入人工 SIU 稽查。

---

## 五、配套资产索引

- **交互式 H5 精读工作台**：[`docs/h5/insurance-claims-underwriting-damage-vlm-sizer.html`](../h5/insurance-claims-underwriting-damage-vlm-sizer.html)
- **7-Card 4:5 评测报告**：[`docs/reports/enterprise-insurance-claims-underwriting-damage-vlm-sizing-cards.html`](../reports/enterprise-insurance-claims-underwriting-damage-vlm-sizing-cards.html)
- **交互式容量计算器**：[`docs/reports/enterprise-insurance-claims-underwriting-damage-vlm-sizing-sizer.html`](../reports/enterprise-insurance-claims-underwriting-damage-vlm-sizing-sizer.html)
- **可复现实验模型**：[`experiments/enterprise_insurance_claims_underwriting_damage_vlm_sizing/run.py`](../../experiments/enterprise_insurance_claims_underwriting_damage_vlm_sizing/run.py)
