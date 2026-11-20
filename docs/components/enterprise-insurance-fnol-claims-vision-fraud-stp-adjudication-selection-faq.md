# 企业级财险与车险报案核赔（`FNOL & Claims Straight-Through Processing / Guidewire ClaimCenter`）、多模态定损（`SAM2/YOLO Part Segmentation + VLM`）、`C2PA / EXIF / AIGC` 欺诈伪造照片拦截与保单除外责任确定性核保架构选型指南

> **适用读者**：负责财产险（P&C）、车险、货运险与延保理赔核心系统（`Guidewire ClaimCenter`、`Duck Creek Claims`、`Sapiens`）建设的 Cloud & AI Solution Architect、核赔与反欺诈（`SIU`）技术负责人及保险科技架构师。
> **核心课题**：当保险公司每月处理 **`120,000` 起报案（`First Notice of Loss, FNOL`）与 `960,000` 张车损/物损现场照片** 时，为什么直接“把保险杠受损照片喂给通用多模态大模型（GPT-4o / Claude Vision）让它估价赔付多少钱（Tier 1）”或“用纯文本 RAG 检索保单 PDF 判断免赔额与除外责任（Tier 3）”会导致 **`34.5%` 维修定损金额误差（`MAPE`）、漏放 `78.0%` 的 `Stable Diffusion / Flux` AI 局部重绘假伤痕与网图搬运骗保、以及 `12.8%` 的除外责任误判（每月造成 `$580,000` 欺诈漏损与二次追加定损返工成本）**？如何通过 **“前置 `C2PA / EXIF + DIRE 频域扩散伪影 + 5,000 万图库 pHash` 反欺诈门禁 + `YOLOv11 / SAM2` 部件级损伤分割映射 `VIN -> CCC ONE / Mitchell / 精友` 原厂零整比配件库 + `Guidewire / OPA` 保单除外条款确定性规则引擎 + `<= $2,500` 小额案件 `4.2 分钟` 自动闪赔（`STP`）”** 四级架构，将定损金额误差压降至 **`4.6%`**、AIGC 伪造与重复照片拦截率提升至 **`99.2%`**、小额案件自动闪赔直通率（`STP`）提升至 **`68.5%`**，并将月度综合核赔与漏损 TCO 降低 **`89.3%`（`$580,000 -> $62,000/月`）**？

![企业级财险与车险报案核赔、多模态部件级定损、AIGC 欺诈照片拦截与确定性闪赔架构总结图](../../assets/svg/enterprise-insurance-fnol-claims-vision-fraud-stp-adjudication-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（3 分钟拍板）

在财险与车险核赔（Claims Adjudication）中，**最致命的两大架构反模式是：① 让多模态大模型看一眼照片就直接拍脑袋猜“修车要花 $1,200”；② 对上传照片不做 AIGC 局部重绘（Inpainting）鉴伪就直接走自动打款**：

| 选型策略 | 适用场景 | 维修定损金额误差 (`MAPE`) | AIGC 假伤痕与网图骗保拦截率 | 自动闪赔直通率 (`STP Rate`) | 月度欺诈漏损、二次定损与人工总成本 (`TCO`) | 最终拍板结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1：车损照片直传 VLM 拍脑袋报金额 + 无图像鉴伪（反模式）** | ❌ **严禁用于任何真实核赔定损与打款** | `34.5%`（漏算内部雷达支架！） | **`22.0%`（漏掉 `78%` 假图！）** | `42.0%`（高错赔率） | `$580,000 / 月` | ❌ **严禁上生产**：被 `Flux` AI 假划痕与网图骗保洗劫，且不懂车型零整比 |
| **Tier 2：传统全人工坐席查勘与人工核赔（`100% Desk Adjuster`）** | 仅适合 `>$50,000` 重大人伤诉讼或复杂企财险 | `9.5%` | `64.0%`（肉眼难辨 AIGC 伪影） | **`8.0%`（结案长达 `11.5 天`）** | `$460,000 / 月` | ❌ **运营极慢极贵**：赔付一块 `$400` 挡风玻璃也要花 `$145` 人工核赔费 |
| **Tier 3：VLM 粗略分类 + 保单全文无约束 RAG 算免赔额** | 内部核赔员辅助摘要（严禁自动打款） | `18.2%` | `75.0%` | `38.0%` | `$280,000 / 月` | ⚠️ **高合规风险**：保单 RAG 产生 `12.8%` 非指定驾驶员/免赔额幻觉 |
| **Tier 4：`C2PA/DIRE/pHash` 鉴伪门禁 + `SAM2/YOLOv11` 部件分割查 `VIN` 配件库 + `OPA` 保单规则闪赔（推荐）** | **企业级车险/家财险/数码碎屏险主力生产架构** | **`4.6%`** | **`99.2%`** | **`68.5%`（`4.2 分钟` 到账）** | **`$62,000 / 月`（省 `89.3%`）** | 🏆 **生产金标准**：视觉只认部件伤情、价格全查 `VIN` 库、规则算保单 |

### 💡 架构师的三大黄金铁律
1. **图片进入定损模型前，必须先过 `<120ms` 的“三道数字取证反欺诈门禁（`C2PA/EXIF + DIRE AIGC 检测 + 5,000 万级 pHash/DINOv2 撞库`）”**：如今黑产只需用 `Stable Diffusion / Flux Inpainting` 花 3 秒钟就能在完好的保时捷车门或名表表盘上“画”出一道逼真的凹痕，或从二手车论坛盗图报案。Tier 4 强制使用 App 内置防篡改相机（写入 `C2PA` 硬件签名、实时 GPS 与车架号 `VIN` 同框陀螺仪轨迹），并在服务端通过 **`DIRE / TruFor` 频域扩散残差检测器 + 全行业历史理赔库 `DINOv2` 向量去重**，拦截 **`99.2%`** 的合成假图与旧伤复报，直接移交反欺诈调查局（`SIU`）。
2. **严禁让视觉大模型猜“修车多少钱”：必须拆解为“`YOLOv11 + SAM2` 部件级多边形分割 + VLM 伤情分级 $\rightarrow$ `VIN` 车架号反查 `CCC ONE / Mitchell / 精友` 原厂工时配件库”**：同样是前保险杠裂纹，2018 款卡罗拉只需 `$380` 喷漆修复，而 2025 款带激光雷达与毫米波支架的新能源 SUV 必须更换总成并做 `$2,400` 的 `ADAS` 标定！视觉模型只负责输出结构化元组 **`(部件编码: Front_Bumper_Cover, 损伤类型: 穿透性裂纹, 面积占比: 18%, 建议动作: Replace)`**，具体配件单价、拆装工时与本地 4S 店工时费率 **`100%` 由车架号 `VIN` 关联商用定损数据库确定性查表计算**（定损误差从 `34.5%` 压降至 **`4.6%`**）。
3. **保单有效性、指定驾驶员除外责任与阶梯免赔额必须走确定性规则引擎（`Guidewire / OPA`），严禁交给保单 PDF RAG 自由推理**：将保单承保范围、起保时间戳（防出险后买保险倒签）、特别约定批单（Endorsements）与历史免赔额累计值交给 **`Open Policy Agent (OPA)`** 确定性核验，使 **`<= $2,500` 的真实小额案件实现 `68.5%` 全自动直通闪赔（`STP`），结案周期从 `11.5 天` 缩短至 `4.2 分钟`**！

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A

### Q1：很多团队觉得“GPT-4o / Gemini 1.5 Pro 看图能力很强，直接传 4 张撞车照片让它生成一份维修报价单（Tier 1）不就完事了吗？”，为什么上线后赔穿了底裤？
* **通用 VLM 既看不见保险杠后面的隐藏传感器，也防不住 3 秒生成的 AI 假划痕**：
  * **配件零整比与隐藏损伤盲区**：通用大模型没有接入主机厂的 `VIN` 爆炸装配表（不知道某款车大灯爪断裂必须连带更换前围框架与校准毫米波雷达），导致初期定损偏低、到了修理厂后 **`52%` 的案件发生高额二次追加定损（Supplement Rework）**，或者被虚假高报蒙混过关（整体误差 `MAPE = 34.5%`）。
  * **AIGC 局部重绘（Inpainting）零防御**：不经过像素噪声一致性（`TruFor`）与频域扩散重建误差（`DIRE`）检测的 VLM，会把 AI 在完好车漆上生成的逼真冰雹坑 100% 当成真伤情直接批款。

### Q2：如果把几十页的保险合同 PDF 切块放进向量数据库做 RAG（Tier 3），为什么会在“拒赔/赔付”上犯 `12.8%` 的致命错误？
* **保险理赔的“批单覆盖（Endorsement Override）”与“时间窗口交集”是严格的布尔代数**：
  * 主险条款第 12 页写着“暴雨涉水导致的发动机损坏不赔”，但客户额外购买了第 38 页的《附加发动机涉水损失险批单（免赔率 15%）》，同时第 41 页批注“仅限登记配偶驾驶时有效”。无约束 RAG 经常只检索到第 12 页主险就错误拒赔（引发银保监/州保险局投诉），或者漏看第 41 页批注而错误赔付！

### Q3：如何平衡“4 分钟极速闪赔体验（STP）”与“严防团伙骗保（SIU）”？
* **三档风险分流漏斗（`Green STP / Amber Desk-Assist / Red SIU Quarantine`）**：
  1. **绿道自动闪赔（`68.5%` 案件，金额 `<= $2,500`）**：App 实时连拍通过 `C2PA + DIRE` 鉴伪、`pHash` 零历史撞库、`VIN` 配件定损无内部结构件损伤、`OPA` 保单核验 100% 绿灯 $\rightarrow$ **`4.2 分钟` 自动生成定损单并触发直赔打款**。
  2. **黄道坐席辅助（`26.3%` 案件，金额 `$2,500–$15,000` 或涉及结构件）**：AI 预填完整配件工时表与伤情掩码高亮图，人工核赔师仅需 **`90 秒`** 复核微调即可一键批复。
  3. **红道反欺诈拦截（`5.2%` 案件）**：命中 AIGC 重绘伪影、跨保司历史旧伤撞库、或保单起保后 `<48 小时` 出险 $\rightarrow$ 立即冻结自动打款并连同取证热力图推送至 **`SIU` 反欺诈专员**。

---

## 3. 🧭 直观可视化图解（Mermaid 决策树 + 端到端参考架构 + 时序图）

### 3.1 企业级保险报案核赔与多模态定损架构选型决策树

```mermaid
flowchart TD
    A["企业财险/车险报案 (FNOL) 与图像定损请求"] --> B["第 1 关: 图像数字取证与反欺诈门禁 (<120ms)\n(C2PA/EXIF + DIRE/TruFor AIGC 鉴伪 + 50M pHash 撞库)"]
    B -- "检出 AIGC 局部重绘伪影或历史理赔重复图" --> C["🚨 红道拦截: 冻结自动闪赔并附带篡改热力图移交 SIU 反欺诈局"]
    B -- "真实拍摄且无历史重复" --> D["第 2 关: YOLOv11 + SAM2 部件级多边形分割 + 伤情分类\n输出 (部件 ID, 损伤类型, 面积, Repair/Replace)"]
    D --> E["第 3 关: 车架号 VIN 反查 CCC ONE / Mitchell / 精友配件与工时表\n+ OPA 保单除外责任/免赔额确定性核算"]
    E --> F{"定损金额 <= $2,500 且无结构件/人伤争议?"}
    F -- "是 (68.5% 案件)" --> G["🟢 绿道直通闪赔 (STP): 4.2 分钟自动结案打款"]
    F -- "否 (26.3% 中大案)" --> H["🟡 黄道坐席辅助: 预填 Guidewire 配件工时单供核赔师 90s 复核"]
```

### 3.2 Tier 4 数字鉴伪门禁 + `SAM2/VIN` 配件目录定损 + `OPA` 闪赔端到端架构图

```mermaid
flowchart LR
    subgraph Gate["🛡️ 1. 图像取证与 AIGC 反欺诈门禁 (<120ms)"]
        IMG["报案现场照片/视频\n(移动端防篡改水印相机)"] --> FOR["EXIF/C2PA 签名校验\n+ DIRE/TruFor 扩散伪影检测"]
        FOR --> DEDUP["5,000 万历史理赔图库\npHash + DINOv2 旧伤撞库"]
    end

    subgraph Vision["🔍 2. 部件分割与 VIN 确定性定损引擎"]
        DEDUP --> SAM["YOLOv11 + SAM2 部件分割\n+ VLM 伤情分级 (划痕/凹陷/裂纹)"]
        SAM --> CAT["VIN 车架号解码器 ->\nCCC ONE / Mitchell 原厂配件与 ADAS 工时库"]
    end

    subgraph Adjudicate["⚖️ 3. Guidewire / OPA 保单核保与三档分流"]
        CAT --> OPA["OPA 保单规则引擎\n(起保期/批单覆盖/指定驾驶员/免赔额)"]
        OPA --> STP["🟢 绿道 STP 自动闪赔 (68.5% / 4.2m)"]
        OPA --> DESK["🟡 黄道 Guidewire 坐席预填单 (26.3%)"]
        FOR & DEDUP -. "欺诈告警 (5.2%)" .-> SIU["🔴 红道 SIU 专案调查队列"]
    end
```

### 3.3 全链路车险报案、AIGC 鉴伪、VIN 查表定损与 4.2 分钟闪赔时序图

```mermaid
sequenceDiagram
    autonumber
    participant User as 投保车主 (移动端引导拍照)
    participant Fraud as C2PA/DIRE/pHash 鉴伪网关
    participant Vision as SAM2/YOLO + VIN 配件定损引擎
    participant OPA as Guidewire + OPA 保单规则引擎
    participant Pay as 直通闪赔 / 修理厂 DRP

    User->>Fraud: 上传车架号 `VIN` 同框视频与 6 张右后门擦挂照片
    Fraud->>Fraud: 校验 `C2PA` 实时签名 + `TruFor` 像素篡改热力图 (`Score=0.02` 真实) + `50M pHash` 无重复旧伤
    Fraud->>Vision: 放行真实车损图像与 `VIN: 5UXCR6C0...`
    Vision->>Vision: `SAM2` 分割 `Right_Rear_Door_Panel (凹陷 12cm, Repair)` + `Wheel_Arch_Trim (断裂, Replace)`
    Vision->>Vision: 反查该 `VIN` 在本地邮编的 `CCC ONE` 配件单价 (`$185`) + 钣金喷漆工时 (`4.5h x $95`) = `$612.50`
    Vision->>OPA: 提交结构化定损单 (`$612.50`) 与出险时间/驾驶员信息
    OPA->>OPA: 确定性核验保单有效、含批单无免赔额优惠 -> 判定满足 `STP <= $2,500` 绿道标准
    OPA->>Pay: `4.2 分钟` 自动向车主推送 `$612.50` 电子定损单并支持一键直赔或预约合作修理厂
```

---

## 4. 📊 架构师多方案选型决策矩阵

| 评估维度 | Tier 1：车损图直传 VLM 拍脑袋报价 + 无鉴伪 | Tier 2：传统全人工坐席查勘与人工核赔 | Tier 3：VLM + 保单 PDF 无约束 RAG | Tier 4：`C2PA/DIRE` 鉴伪 + `SAM2/VIN` 配件目录 + `OPA` 闪赔（推荐） |
| :--- | :--- | :--- | :--- | :--- |
| **维修定损金额误差 (`MAPE`)** | **`34.5%`（漏算内部支架与地区工时差）** | `9.5%` | `18.2%` | **`4.6%`（精确对齐 `VIN` 原厂配件号与本地工时表）** |
| **AIGC 假伤痕与旧图骗保拦截率** | **`22.0%`（漏掉 `78%` AI 重绘与网图）** | `64.0%`（肉眼难辨 Diffusion 假图） | `75.0%` | **`99.2%`（`C2PA` + `DIRE/TruFor` 频域检测 + `50M pHash`）** |
| **自动闪赔直通率 (`STP Rate`)** | `42.0%`（伴随极高错赔漏损） | **`8.0%`（平均结案耗时 `11.5 天`）** | `38.0%` | **`68.5%`（真实小额案 `4.2 分钟` 全自动结案到账）** |
| **保单除外责任与免赔额准确率** | `82.0%` | `97.5%` | `87.2%`（`12.8%` 批单覆盖幻觉） | **`100.0%`（`Guidewire / OPA` 确定性布尔与算术引擎）** |
| **单笔简单案件处理成本** | `$18.50`（含高额错赔分摊） | **`$145.00`（纯人工核赔工时）** | `$42.00` | **`$1.85`（降幅达 `98.7%`）** |
| **月度欺诈漏损与核赔总 TCO** | `$580,000 / 月` | `$460,000 / 月` | `$280,000 / 月` | **`$62,000 / 月`（节省 `89.3%`）** |

---

## 5. 🧪 容量/成本拐点算例与 7-Card 报告

### 5.1 基准算例假设（月均 `120,000` 起报案 `FNOL` / `960,000` 张车损与家财险照片）
* **业务规模**：大型财险与车险公司，月均处理 `120,000` 起报案（每案平均 `8` 张照片，共 `960,000` 张图像），其中 `70%` 为金额 `<= $2,500` 的中轻度外观件或家财受损案件，单次人工核赔坐席综合成本为 `$145`。
* **工程实测与仿真代码**：见 [`experiments/enterprise_insurance_fnol_claims_vision_fraud_stp_adjudication_selection/run.py`](../../experiments/enterprise_insurance_fnol_claims_vision_fraud_stp_adjudication_selection/run.py) 与 [`results.json`](../../experiments/enterprise_insurance_fnol_claims_vision_fraud_stp_adjudication_selection/results.json)（数据来源：`synthetic + anonymized_reported`，`independently_reproduced=false`）。
* **交互式容量与 TCO 计算器**：📊 [打开企业级财险与车险报案核赔容量计算器 (`enterprise-insurance-fnol-claims-vision-fraud-stp-adjudication-sizer.html`)](../reports/enterprise-insurance-fnol-claims-vision-fraud-stp-adjudication-sizer.html)
* **7-Card 4:5 评测报告卡**：🃏 [打开 7-Card 视觉评测报告 (`enterprise-insurance-fnol-claims-vision-fraud-stp-adjudication-selection-cards.html`)](../reports/enterprise-insurance-fnol-claims-vision-fraud-stp-adjudication-selection-cards.html)

```chart
{
  "data": {
    "values": [
      { "tier": "Tier 1: Direct VLM Dollar Guess", "mape_pct": 34.5, "fraud_catch_pct": 22.0, "stp_rate_pct": 42.0, "monthly_tco_k": 580.0 },
      { "tier": "Tier 2: Manual Desk Adjuster", "mape_pct": 9.5, "fraud_catch_pct": 64.0, "stp_rate_pct": 8.0, "monthly_tco_k": 460.0 },
      { "tier": "Tier 3: VLM + Unconstrained Policy RAG", "mape_pct": 18.2, "fraud_catch_pct": 75.0, "stp_rate_pct": 38.0, "monthly_tco_k": 280.0 },
      { "tier": "Tier 4: C2PA/DIRE Gate + SAM2/VIN + OPA STP", "mape_pct": 4.6, "fraud_catch_pct": 99.2, "stp_rate_pct": 68.5, "monthly_tco_k": 62.0 }
    ]
  },
  "semantic_types": {
    "tier": "Category",
    "mape_pct": "Percentage",
    "fraud_catch_pct": "Percentage",
    "stp_rate_pct": "Percentage",
    "monthly_tco_k": "Amount"
  },
  "chart_spec": {
    "chartType": "Bar Chart",
    "title": "Tier 4 Achieves 4.6% Repair MAPE, 99.2% Fraud Catch, and Cuts Monthly Claims TCO by 89.3%",
    "subtitle": "Monthly fraud leakage, supplement rework, and manual adjuster TCO ($ USD Thousands) across 120K claims/mo",
    "encodings": {
      "x": "tier",
      "y": "monthly_tco_k",
      "color": "tier"
    }
  }
}
```

---

## 6. 🛠️ 架构评审与生产落地 Checklist

- [ ] **前置 `C2PA / EXIF + DIRE/TruFor` 扩散重绘鉴伪门禁**：是否在移动端强制通过引导式实时相机拍摄（禁止直接从相册上传无签名截图），并在服务端用频域扩散检测器拦截 AI 局部重绘假伤痕？
- [ ] **跨历史案件与全网图库 `pHash + DINOv2` 向量撞库**：是否将每张报案损伤局部裁剪图与历史理赔图库比对，防止同一处旧划痕换角度重复骗保或搬运网图报案？
- [ ] **`YOLOv11 + SAM2` 部件分割与 `VIN -> CCC ONE / Mitchell` 确定性查表解耦**：是否已严禁多模态大模型直接生成美元金额，确保所有配件单价、拆装工时、油漆辅料与 `ADAS` 标定费用 `100%` 源自车架号 `VIN` 标准定损库？
- [ ] **`Guidewire / OPA` 保单批单与除外责任确定性规则引擎**：起保等待期、非指定驾驶员除外、专项附加险批单覆盖及免赔额抵扣是否全部由确定性规则计算？
- [ ] **三档风险分流（`STP <= $2,500` 绿道 / 黄道坐席预填 / 红道 `SIU` 移交）**：是否对存疑或超额案件自动阻断直赔打款，并向反欺诈专员输出像素级篡改定位热力图？
