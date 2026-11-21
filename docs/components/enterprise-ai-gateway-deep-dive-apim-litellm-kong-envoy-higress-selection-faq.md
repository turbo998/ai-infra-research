# 企业级 AI Gateway 统一接入层深度横向选型：`Azure API Management (AI Gateway)` vs. `LiteLLM Proxy` vs. `Kong AI Gateway` vs. `Envoy AI Gateway / Higress` 两阶段 Token 限流、秒级熔断与 K8s KV 感知路由指南

> **语言 / Language**: **中文 (当前)** | [English](./enterprise-ai-gateway-deep-dive-apim-litellm-kong-envoy-higress-selection-faq.en.md)  
> **定位**：面向 Cloud & AI Infrastructure Solution Architect 与企业平台工程负责人的四大主流 AI 网关（`Azure APIM`、`LiteLLM Proxy`、`Kong AI Gateway`、`Envoy AI Gateway / Higress`）横向对比、流式 SSE 两阶段加权 Token 桶算法、`PTU -> PAYGO` 秒级溢出熔断与 `K8s Gateway API Inference Extension` 选型白皮书  
> **合规与来源声明**：`synthetic + anonymized_reported`（`independently_reproduced=false`，`physical_gpu_cluster_measurement=not_measured`；完全面向通用企业 AI 平台网关架构设计与选型）

![企业级 AI Gateway 统一接入层 Azure APIM vs LiteLLM vs Kong vs Envoy AI / Higress 总结图](../../assets/svg/enterprise-ai-gateway-deep-dive-apim-litellm-kong-envoy-higress-selection-faq-summary.svg)

---

## 1. 📌 核心选型结论速览（Executive Summary）

> **一句话结论**：**传统基于请求次数（RPM）的 HTTP 网关（如原生 Nginx / ALB）完全无法治理大模型流量（1 个 `128K` Agent 请求消耗的算力是普通 `500` 字问答的 `256 倍`，且 SSE 流式响应直到最后一帧 `[DONE]` 才知道真实输出 Token 数）！在四大主流 AI Gateway 选型上：① 深度绑定 Azure OpenAI `PTU + Entra ID RBAC` 的企业首选 `Azure API Management (AI Gateway)`；② 追求最快接入 `100+` 模型厂商、虚拟 Key 发放与部门 FinOps 账单看板的中小流量中台（`<= 1,000 RPS`）首选 `LiteLLM Proxy`；③ 已有 Kong 商业底座的跨云企业首选 `Kong AI Gateway`；④ 自建 Kubernetes (`vLLM / SGLang`) 集群或超高并发（`>= 1,500 RPS`）生产底座首选 `Envoy AI Gateway / Higress`（原生实现 `K8s Gateway API Inference Extension` 的 KV Cache 水位与 LoRA 亲和性路由，C++ 数据面转发损耗仅 `0.9ms`、故障/429 切换仅 `120ms`，可降低 `64.2%` 平台总 TCO）！**

- **交互式企业 AI Gateway 选型与配额计算器**：[打开交互式计算器 (`enterprise-ai-gateway-deep-dive-apim-litellm-kong-envoy-higress-sizer.html`)](../reports/enterprise-ai-gateway-deep-dive-apim-litellm-kong-envoy-higress-sizer.html)
- **7-Card 4:5 视觉评测报告**：[打开 7-Card 报告 (`enterprise-ai-gateway-deep-dive-apim-litellm-kong-envoy-higress-selection-cards.html`)](../reports/enterprise-ai-gateway-deep-dive-apim-litellm-kong-envoy-higress-selection-cards.html)

---

## 2. 🎯 客户与架构师灵魂拷问 Q&A（Deep-Dive Architecture FAQ）

### Q1：SSE 流式输出在请求刚进来时根本不知道模型会吐多少个字，AI Gateway 是怎么做到精确按“加权 Token（TPM）”限流防爆仓的？
- **两阶段流式令牌桶算法（Two-Phase Streaming Token Bucket）**：
  1. **第一阶段：请求进入时预扣款（Pre-Flight Reservation）**：
     - 网关通过高速分词器（如 `tiktoken-rs`）或字节估算算出输入 Token 数 $N_{\text{in}}$，并加上该租户历史 P80 预估输出预算 $\hat{N}_{\text{out}}$，先从 Redis 令牌桶中原子预扣 $T_{\text{reserve}} = N_{\text{in}} + w_{\text{out}} \cdot \hat{N}_{\text{out}}$（若余额不足立即返回 `429` 或路由至降级模型）；
  2. **第二阶段：SSE 流结束时多退少补（Post-Stream Reconciliation）**：
     - 网关流式解析 SSE 响应块，当捕获到最后一帧 `usage: {prompt_tokens, completion_tokens}`（或客户端中途断连时统计已发送的 chunk 数）时，立即计算差额 $\Delta = w_{\text{out}} \cdot (N_{\text{out\_actual}} - \hat{N}_{\text{out}})$ 并异步回写修正 Redis 令牌桶！彻底杜绝恶意租户利用 `max_tokens=16384` 或 `128K` 长文打爆共享 `PTU` 底座。

### Q2：什么是 `K8s Gateway API Inference Extension`？为什么 `Envoy AI Gateway / Higress` 能让自建 `vLLM / SGLang` 集群吞吐再涨 30%？
- **传统 K8s `Service (kube-proxy / Ingress)` 的致命缺陷**：
  - 把大模型请求当成普通无状态 Web 请求做随机/轮询负载均衡——哪怕 Pod A 显存里刚刚缓存了该用户的 `32K` 代码库 KV Cache 且挂载了对应的金融 `LoRA`，请求也会被错误地分发给正在排队跑满长 Prefill 的 Pod B！
- **`Envoy AI Gateway` + `Endpoint Picker Extension (EPP)` 的解法**：
  - 实时抓取每个 `vLLM / SGLang` Pod 暴露的 Prometheus 指标（① `kv_cache_usage_perc` 显存水位、② `num_requests_waiting` 排队长度、③ 已加载的活跃 `LoRA` 列表与前缀树 Hash），将请求精准路由至**队列最短且命中 KV/LoRA 亲和性的最优 Pod**！

---

## 3. 🧭 直观可视化图解（Mermaid 四大网关选型决策树 + 两阶段 Token 限流时序）

### 3.1 企业级 AI Gateway 四选一架构选型决策树
```mermaid
flowchart TD
    A["企业级 AI Gateway 统一接入层选型"] --> B{"企业主力云环境与核心诉求？"}
    B -- "Azure 重度用户 · 主力用 Azure OpenAI PTU<br/>+ 需原厂 Entra ID / Private Link 托管" --> C["🏆 选方案 1: Azure API Management (AI Gateway)<br/>• 原生 azure-openai-token-limit & emit-metric 策略<br/>• 零自建网关运维 · 原生 PTU->PAYGO 优先池故障转移"]
    B -- "研发团队快速搭建多模型超市 (100+ 模型)<br/>+ 需现成 Virtual Keys / 部门预算账单 UI (<=1K RPS)" --> D["🏆 选方案 2: LiteLLM Proxy (Python + Postgres/Redis)<br/>• 最全 OpenAI 格式转换层 · 1小时上线分账中台"]
    B -- "存量已全面标准化 Kong Enterprise API 网关" --> E["🏆 选方案 3: Kong AI Gateway (AI Proxy/Rate Limit 插件)"]
    B -- "自建 K8s (vLLM/SGLang) 推理池 + 多云混合<br/>或超高并发 (>1,500 RPS) 严苛低延迟 (<1ms)" --> F["🏆 选方案 4: Envoy AI Gateway / Higress<br/>(+ K8s Gateway API Inference Extension)<br/>• 0.9ms 极低代理损耗 + KV/LoRA 亲和性调度"]
```

---

## 4. 📊 架构师多方案选型决策矩阵（四大 AI Gateway 横向对比）

| 对比维度 | `Azure API Management (AI Gateway)` | `LiteLLM Proxy` | `Kong AI Gateway` | `Envoy AI Gateway / Higress` |
| :--- | :--- | :--- | :--- | :--- |
| **底层技术栈与形态** | Azure 全托管 PaaS 网关 | Python (FastAPI) 开源/商业服务 | Lua / Go / Rust 插件化网关 | **C++ Envoy + Go 控制面 + K8s Gateway API** |
| **单跳网关代理延迟 (`P99`)** | `3.8 ms` | `11.4 ms`（Python 序列化开销） | `2.8 ms` | **`0.9 ms` ✓（极致高性能）** |
| **多模型协议转换覆盖度** | 聚焦 Azure OpenAI / Foundry / Bedrock | **最强（原生适配 `100+` 模型 API）** | 强（主流 `20+` 模型提供商） | 强（聚焦 OpenAI 兼容标准与云端主力） |
| **K8s `vLLM` KV/LoRA 感知调度** | 需外挂自定义路由服务 | 基础 Prefix/Latency 路由 | 基础插件支持 | **原生支持 `Gateway API Inference Extension` ✓** |
| **多租户预算账单与 Virtual Key** | 结合 Azure Monitor / APIM 订阅 | **开箱即用最完善（团队/项目/Key 级账单）** | 结合 Kong Konnect 企业版 | 结合 OpenTelemetry + Prometheus FinOps |
| **最佳搭档组合** | 独立承载 Azure 企业闭源底座 | **可作为控制面/转换层挂在 Envoy 后方** | 统一传统微服务与 AI API 网关 | **自建 K8s GPU 推理集群 + 多云混合入口首选** |

---

## 5. 🛠️ 架构评审与生产落地 Checklist

- [ ] **全面废除纯 `RPM/QPS` 限流，强制上线两阶段加权 `TPM` 令牌桶（`Pre-Flight 预扣 + SSE 结算`）**。
- [ ] **配置基于 `TTFT 超时（如 >1.5s）` 与 `HTTP 429/503` 的毫秒级分级熔断（`PTU 主区 -> PTU 备区 -> 自建 vLLM -> PAYGO 溢出`）**。
- [ ] **K8s 自建推理池接入 `Gateway API Inference Extension (Endpoint Picker)`**，严禁用无状态轮询打散 `vLLM/SGLang` 的前缀缓存。
