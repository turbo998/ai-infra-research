import math from "./academy-math.cjs";

export const esc = (text) => String(text).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
export const b = (zh, en) => {
  if (typeof zh !== "string" || typeof en !== "string" || !zh.trim() || !en.trim()) throw new Error("Both Chinese and English content fragments are required");
  return `<span data-lang="zh-CN">${zh}</span><span data-lang="en">${en}</span>`;
};
const p = (zh, en) => `<p>${b(zh, en)}</p>`;
const formula = (text) => `<div class="formula">${text}</div>`;
const note = (zh, en) => `<div class="notice">${b(zh, en)}</div>`;
const section = (id, zh, en, content) => `<section class="lesson" id="${id}"><h2>${b(zh, en)}</h2>${content}</section>`;
const faq = (zh, en, answerZh, answerEn) => `<details><summary>${b(zh, en)}</summary>${p(answerZh, answerEn)}</details>`;
const table = (headers, rows, caption) => `<div class="table-wrap"><table>${caption ? `<caption>${caption}</caption>` : ""}<thead><tr>${headers.map((h) => `<th scope="col">${h}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((v, i) => i === 0 ? `<th scope="row">${v}</th>` : `<td>${v}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
const field = (lab, key, zh, en, value, min, max, step = 1, type = "number") => `<div class="control"><label for="${lab}-${key}">${b(zh, en)}</label><input id="${lab}-${key}" data-key="${key}" name="${key}" type="${type}" value="${value}" min="${min}" max="${max}" step="${step}" required>${type === "range" ? `<output for="${lab}-${key}" data-for="${key}">${value}</output>` : ""}</div>`;
const select = (lab, key, zh, en, choices, selected) => `<div class="control"><label for="${lab}-${key}">${b(zh, en)}</label><select id="${lab}-${key}" data-key="${key}" name="${key}">${choices.map(([value, label]) => `<option value="${esc(value)}"${value === selected ? " selected" : ""}>${esc(label)}</option>`).join("")}</select></div>`;
const check = (lab, key, zh, en, checked = false) => `<div class="control"><label for="${lab}-${key}">${b(zh, en)}</label><input type="checkbox" id="${lab}-${key}" name="${key}" data-key="${key}"${checked ? " checked" : ""}></div>`;
const textarea = (lab, key, zh, en, value) => `<div class="control"><label for="${lab}-${key}">${b(zh, en)}</label><textarea id="${lab}-${key}" name="${key}" data-key="${key}" rows="18" maxlength="250000" required spellcheck="false">${esc(value)}</textarea></div>`;
const models = Object.entries(math.models).map(([id, m]) => [id, m.name]);
const gpus = Object.entries(math.gpus).map(([id, g]) => [id, g.name]);
const lab = (id, controls) => `<div class="lab" data-workbench="${id}"><form data-lab="${id}" class="controls" aria-describedby="${id}-error" novalidate>${controls}<button type="reset">${b("恢复算例", "Reset example")}</button></form><div class="results"><div id="${id}-error" class="error" role="alert" hidden></div><div class="sr-only" role="status" aria-live="polite" data-result-summary="${id}"></div><div data-results="${id}" aria-live="off"></div><noscript>${b("请启用JavaScript使用沙盘；下方公式、FAQ与算例仍可离线阅读。", "Enable JavaScript for the workbench; formulas, FAQ and worked examples remain readable offline.")}</noscript></div></div>`;
const quiz = (id, zh, en, answers, correct, explainZh, explainEn) => `<div class="quiz" data-quiz="${id}" data-correct="${correct}"><fieldset><legend>${b(zh, en)}</legend>${answers.map(([z, e], i) => `<label><input type="radio" name="quiz-${id}" value="${i}">${b(z, e)}</label>`).join("")}<button type="button" data-check-quiz>${b("核对答案", "Check answer")}</button><div class="quiz-result" role="status"></div><template data-explanation>${b(explainZh, explainEn)}</template></fieldset></div>`;
const checklist = (items) => `<div class="checklist">${items.map(([z, e]) => `<label><input type="checkbox">${b(z, e)}</label>`).join("")}</div>`;
const diagram = (id, titleZh, titleEn, content, captionZh, captionEn, height = 260) => `<figure class="chart"><svg viewBox="0 0 800 ${height}" role="img" aria-labelledby="${id}-title"><title id="${id}-title">${titleZh} / ${titleEn}</title>${content}</svg><figcaption>${b(captionZh, captionEn)}</figcaption></figure>`;
const node = (x, y, w, title, subtitle, cls = "surface") => `<rect x="${x}" y="${y}" width="${w}" height="68" rx="10" class="${cls}"/><text x="${x + w / 2}" y="${y + 27}" text-anchor="middle" class="${cls === "primary" ? "inverse" : "label"}">${title}</text><text x="${x + w / 2}" y="${y + 49}" text-anchor="middle" class="${cls === "primary" ? "inverse" : "label"}">${subtitle}</text>`;
const arrow = (x1, y1, x2, y2) => `<path d="M ${x1} ${y1} L ${x2} ${y2}" class="line secondary"/><path d="M ${x2 - 8} ${y2 - 5} L ${x2} ${y2} L ${x2 - 8} ${y2 + 5}" class="line secondary"/>`;
export const chapters = [
  { id: "metrics", file: "01-metrics.html", zh: "先看用户体验，再看机器吞吐", en: "User experience comes before peak throughput", subtitleZh: "TTFT · TPOT / ITL · E2EL · RPS · Goodput · 成本", subtitleEn: "TTFT · TPOT / ITL · E2EL · RPS · Goodput · cost", minutes: 20, sources: ["vllm", "ptu-concept", "vllm-metrics", "aiperf", "genai-perf"] },
  { id: "capacity", file: "02-tpm-tps-capacity.html", zh: "50 TPS/用户，不等于 50 TPS/集群", en: "50 TPS per user is not 50 TPS per cluster", subtitleZh: "TPM/TPS 边界 · Little's Law · 峰值规划 · PTU 权重", subtitleEn: "TPM/TPS boundaries · Little's Law · peak sizing · PTU weights", minutes: 25, sources: ["ptu", "ptu-concept", "vllm", "quota", "quota-classic"] },
  { id: "sizing", file: "03-gpu-sizing.html", zh: "放得下、吐得快、首字快，是三道不同水位线", en: "Memory fit, decode speed and first-token speed are three different bounds", subtitleZh: "Dense / MoE · GQA / MLA · 显存堆叠 · HBM / FLOPs", subtitleEn: "Dense / MoE · GQA / MLA · memory stacks · HBM / FLOPs", minutes: 30, sources: ["h100", "h200", "b200", "l40s", "deepseek-config", "deepseek", "llama", "qwen", "vllm", "gqa-tp", "mla-layout"] },
  { id: "pd", file: "04-pd-disaggregation.html", zh: "PD 分离消除一类干扰，却增加一次 KV 交接", en: "PD removes one interference source but adds a KV handoff", subtitleZh: "混部 / Chunked Prefill / PD · KV 传输 · 决策边界", subtitleEn: "Colocated / chunked prefill / PD · KV transfer · decision boundaries", minutes: 25, sources: ["vllm", "vllm-pd", "dynamo", "mooncake", "sglang", "gpudirect", "dynamo-pd-layout", "nixl-layout"] },
  { id: "parallel", file: "05-parallelism-acceleration.html", zh: "TP 分张量，PP 分层，DP 分请求，EP 分专家", en: "TP splits tensors, PP layers, DP requests, EP experts", subtitleZh: "通信原语 · NVLink / RDMA 边界 · 加速收益/代价矩阵", subtitleEn: "Collectives · NVLink / RDMA boundaries · acceleration tradeoffs", minutes: 25, sources: ["vllm", "sglang", "deepseek", "nccl", "blackwell", "tensor-parallel", "speculative", "prefix"] },
  { id: "network", file: "06-rdma-roce-nvlink.html", zh: "900 GB/s NVLink和400 Gb/s网卡，先统一口径", en: "Normalize units before comparing 900 GB/s NVLink with a 400 Gb/s NIC", subtitleZh: "GPUDirect RDMA ·RoCEv2流控 ·多轨拓扑 ·NCCL带宽", subtitleEn: "GPUDirect RDMA · RoCEv2 congestion · rail topology · NCCL bandwidth", minutes: 25, sources: ["gpudirect", "nccl", "blackwell", "h100", "ib", "ecn", "pcie", "rail-ra", "nccl-count", "pfc-headroom", "dynamo-pd-layout"] }
];
export const studyTasks = {
  metrics: ["lifecycle", "先把输出128改成1，观察TPOT的N/A；再对比trace平均/最大ITL门槛。", "Change output 128 to 1 and inspect TPOT N/A; then compare mean/max-ITL trace gates."],
  capacity: ["capacity", "先走QPS→TPS/TPM；再调高推理占比，观察并发为何由少数长请求主导。", "Start with QPS→TPS/TPM, then increase reasoning share to see long-request concurrency dominance."],
  sizing: ["sizing", "分开改S_p和驻留S；再对比TP8/TP16，别把聚合显存下界当实际配置。", "Change S_p and resident S separately; compare TP8/TP16 rather than treating an aggregate floor as a configuration."],
  pd: ["pd", "独立改变线量与D状态复制；切换P先发首字，比较TTFT和首次间隔。", "Change wire and D-state copies separately; switch to early P output and compare TTFT with the first gap."],
  parallel: ["parallel", "切四种流向图；把投机接受率从0.8降到0，检查是否反而变慢。", "Switch the four flow views; reduce speculative acceptance from 0.8 to 0 and check for a slowdown."],
  network: ["network", "核对NCCL count，再改NIC PCIe代际；注入rail故障并关闭重映射。", "Check NCCL count scope, change NIC PCIe generation, then fail a rail without remapping."]
};
export function validateContentSources(document) {
  if (!document || !Array.isArray(document.sources) || !document.sources.length || !/^\d{4}-\d{2}-\d{2}$/.test(document.reviewed_at)) throw new Error("Academy sources require a reviewed source catalog");
  const reviewed = new Date(document.reviewed_at + "T00:00:00Z");
  if (!Number.isFinite(reviewed.getTime()) || reviewed.toISOString().slice(0,10)!==document.reviewed_at) throw new Error("Source review date must be a valid calendar date");
  if (document.hardware_measured !== false || document.independently_reproduced !== false) throw new Error("Educational academy provenance must not claim independently measured hardware benchmarks");
  const ids = new Set();
  for (const source of document.sources) {
    if (!/^[a-z][a-z0-9-]*$/.test(source.id) || ids.has(source.id) || !source.title || !source.supports || !["specification", "implementation", "unverified"].includes(source.state)) throw new Error("Source catalog IDs, state and descriptions must be explicit and unique");
    const url = new URL(source.url);
    const host=url.hostname.toLowerCase();
    const octets=/^\d+\.\d+\.\d+\.\d+$/.test(host)?host.split(".").map(Number):null;
    const localIP=octets && (octets[0]===0||octets[0]===10||octets[0]===127||(octets[0]===169&&octets[1]===254)||(octets[0]===172&&octets[1]>=16&&octets[1]<=31)||(octets[0]===192&&octets[1]===168));
    if (url.protocol !== "https:" || url.username || url.password || localIP || host==="localhost" || host==="[::1]" || /^\[(fc|fd|fe80)/.test(host) || /\.(localhost|local|internal)$/.test(host)) throw new Error("Archive citations must use public HTTPS URLs without credentials");
    ids.add(source.id);
  }
  const files = new Set();
  for (const chapter of chapters) {
    if (!/^\d{2}-[a-z0-9-]+\.html$/.test(chapter.file) || files.has(chapter.file)) throw new Error("Chapter outputs must be unique local HTML basenames");
    if (chapter.sources.some((id)=>!ids.has(id)) || new Set(chapter.sources).size!==chapter.sources.length) throw new Error("Every chapter citation must resolve to one explicit source record");
    if (!studyTasks[chapter.id] || !chapter.zh || !chapter.en) throw new Error("Every chapter needs bilingual goals and a learning entry");
    files.add(chapter.file);
  }
}

export function indexContent() {
  return section("route", "今晚先走一条业务问题驱动的学习路线", "Follow a business-question learning route tonight",
    p("不从缩写背诵开始。先问「用户为什么觉得卡」→「高峰需要多大吞吐」→「买多少卡才合理」→「是否拆 P/D」→「怎么并行」→「网络瓶颈在哪里」。每册都带可调沙盘、手算复核、FAQ、自测与上线核对清单。",
      "Start with a business question, not an acronym list: why users see stalls → how much peak throughput is needed → how many GPUs are plausible → whether to split P/D → how to parallelize → where the network bottleneck sits. Each chapter includes a workbench, hand calculation, FAQ, quiz and production checklist.") +
    `<div class="grid three chapter-rail">${chapters.map((c, i) => `<article class="card module-card"><span class="chapter-number">0${i + 1}</span><span class="tag">${c.minutes} MIN · OFFLINE</span><h3>${b(c.zh, c.en)}</h3>${p(c.subtitleZh, c.subtitleEn)}<a class="button" data-carry href="${c.file}">${b("进入专题", "Open chapter")} →</a></article>`).join("")}</div>`) +
    section("quick", "三分钟总览：别让吞吐数字掩盖 SLA 失败", "Three-minute overview: throughput can conceal an SLO failure",
      table([b("客户问题", "Customer question"), b("先看哪个量", "Start with"), b("下一步动作", "Next action")], [
        [b("首字太慢", "Slow first token"), "P95/P99 TTFT + queue + prefill", b("拆开排队、输入长度、缓存命中与网络预算", "Separate queue, input length, cache hits and transport budget")],
        [b("流式断断续续", "Stuttering stream"), "P99 ITL + per-request TPOT", b("看长 Prefill 干扰、KV 读取、通信与调度", "Inspect long-prefill interference, KV reads, collectives and scheduling")],
        [b("用户多了就排队", "Queues grow under load"), "Goodput + active concurrency", b("用稳态均值做 Little's Law，再做突发压测", "Use steady-state means in Little's Law, then test bursts")],
        [b("模型到底放不放得下", "Will the model fit?"), "Total weights + KV + per-rank runtime", b("总参数定存储，激活参数定部分计算；别混淆", "Total parameters govern storage; active parameters govern part of compute")],
        [b("网络明明 400G 还很慢", "400G network still feels slow"), "Payload / effective one-way bandwidth", b("先统一 bit/byte、方向、端口数，再看拥塞", "Normalize bits/bytes, direction and port count before congestion")]
      ]) +
      note("所有 SLA 目标、负载分布、价格和曲线都是明确标记的教学假设。没有真实 GPU 运行记录，不冒充实测，也不把容量下界写成采购保证。",
        "All SLO targets, traffic distributions, prices and curves are explicitly labeled teaching assumptions. There is no hardware run record; bounds are not measurements or procurement guarantees.")) +
    section("map", "六个概念串成一张架构决策图", "Six concepts form one architecture decision map",
      diagram("academy-map", "业务到网络的六步决策", "Six decisions from business to fabric",
        node(20, 30, 230, "01 / SLO", "TTFT · ITL · Goodput", "primary") + node(285, 30, 230, "02 / Demand", "QPS × tokens · Little") + node(550, 30, 230, "03 / GPU", "Weights + KV + roofline") +
        node(20, 150, 230, "06 / Network", "RDMA · NVLink · NCCL") + node(285, 150, 230, "05 / Parallel", "TP · PP · DP · EP") + node(550, 150, 230, "04 / P-D", "Isolation vs handoff") +
        arrow(252, 64, 280, 64) + arrow(517, 64, 545, 64) + `<path d="M 665 100 L 665 140" class="line secondary"/>` +
        `<path d="M 548 185 L 520 185 M 281 185 L 253 185" class="line secondary"/>`,
        "先满足体验指标，再求达标吞吐；网络服务于并行与 KV 交接，不是孤立采购参数。",
        "Satisfy user experience before maximizing goodput; fabric supports parallelism and KV handoff, not an isolated purchasing spec.")) +
    section("study", "每次只改一个变量，才知道瓶颈转移到了哪里", "Change one variable at a time to see the bottleneck move",
      `<div class="grid"><article class="card"><h3>${b("30分钟速读", "30-minute fast route")}</h3>${p("先读01的生命周期，02的QPS→TPM，03的三道水位线。再把自己的业务QPS、输入/输出长度和目标延迟代入。", "Read the lifecycle in 01, QPS→TPM in 02 and the three bounds in 03. Substitute your traffic, token lengths and latency targets.")}</article><article class="card"><h3>${b("2–3小时精读", "2–3-hour deep route")}</h3>${p("完整走六册，每册先做顶部「今晚先试」默认算例，再按问题查深度沙盘与FAQ。把batch从1改到32、上下文从2K改到32K，观察瓶颈转移。", "Walk through six chapters, starting with each chapter's first experiment before consulting deeper workbenches/FAQ. Change batch 1→32 and context 2K→32K to see bottlenecks move.")}</article></div>` +
      p("离线使用：保留整个 h5-academy 目录，双击 index.html。每一册本身是独立 HTML，无需服务器；中英切换、明暗主题、交互、图表和自测全部内联。手机首页可左右滑动专题卡片，长图和表格可在自身区域横滑。",
        "Offline: keep the h5-academy folder and double-click index.html. Every chapter is a standalone HTML file with embedded language switching, themes, interactions, charts and quizzes. On mobile, swipe the chapter cards; wide figures and tables scroll within their own regions."));
}

function metricsContent() {
  const lifeControls = field("lifecycle", "queue", "排队 ms", "Queue ms", 120, 0, 30000) +
    field("lifecycle", "prefill", "Prefill + 首 Token 计算 ms", "Prefill + first-token compute ms", 300, 0, 60000) +
    field("lifecycle", "delivery", "网络 / 网关交付 ms", "Network / gateway delivery ms", 30, 0, 10000) +
    field("lifecycle", "output", "输出 Token 数（含首 Token）", "Output tokens (including first)", 128, 1, 32768) +
    field("lifecycle", "tpot", "后续平均 TPOT ms/token", "Mean subsequent TPOT ms/token", 20, 0.1, 1000, 0.1);
  const goodControls = field("goodput", "batch", "活跃 decode batch", "Active decode batch", 32, 1, 128, 1, "range") +
    field("goodput", "base", "基础 ITL ms（假设）", "Base ITL ms (assumption)", 12, 0.1, 200, 0.1) +
    field("goodput", "slope", "每 batch 增量 ms（假设）", "Increment per batch ms (assumption)", 0.25, 0.01, 10, 0.01) +
    field("goodput", "knee", "饱和拐点 batch（假设）", "Saturation knee batch (assumption)", 48, 1, 128) +
    field("goodput", "ttftLimit", "达标 TTFT 上限 ms", "TTFT SLO limit ms", 800, 1, 30000) +
    field("goodput", "tpotLimit", "达标平均 TPOT 上限 ms", "Mean TPOT SLO limit ms", 35, 1, 1000);
  const traces = select("trace", "gateMode", "逐请求时延协议", "Per-request latency contract", [["mean", "TTFT + mean TPOT"], ["max", "TTFT + max visible ITL"]], "max") +
    select("trace", "boundary", "在途请求的系统边界", "In-flight system boundary", [["client", "Client arrival → terminal"], ["admitted", "Server admitted → terminal"]], "client") +
    field("trace", "windowMs", "相对时间测窗 [0,W) ms", "Relative-time observation window [0,W) ms", 1000, 200, 5000, 100) +
    field("trace", "ttftLimit", "逐请求 TTFT 上限 ms", "Per-request TTFT limit ms", 300, 1, 60000) +
    field("trace", "tpotLimit", "平均 TPOT 或最大 ITL 上限 ms", "Mean TPOT or maximum ITL limit ms", 35, 1, 1000) +
    `<div class="control"><label for="trace-requireQuality">${b("同时要求 quality_pass=true（示例注释，非自动评测）", "Require quality_pass=true (example annotation, not automatic evaluation)")}</label><input type="checkbox" id="trace-requireQuality" data-key="requireQuality" checked></div>` +
    textarea("trace", "trace", "匿名数值 JSON；不放 Prompt、身份或密钥", "Anonymous numeric JSON; no prompts, identities or credentials", JSON.stringify(math.traceSamples, null, 2));
  return section("concepts", "TTFT 包含排队；平均 TPOT 不等于尾部 ITL", "TTFT includes queueing; mean TPOT is not tail ITL",
    table([b("指标", "Metric"), b("严格定义 / 单位", "Definition / unit"), b("架构含义与易错点", "Meaning and trap")], [
      ["TTFT", b("请求发出到首 Token 可见，ms 或 s", "Request sent → first visible token, ms or s"), b("包含客户端/网关/排队/Prefill/首字传输。服务端 TTFT 与端到端 TTFT 要分别命名。", "Includes client/gateway/queue/prefill/first delivery. Name server-side and end-to-end TTFT separately.")],
      ["TPOT / ITL", "(t_last − t_first) / (N_out − 1); ITL_i = t_i − t_(i−1)", b("TPOT 是一个请求的平均后续间隔；P99 ITL 捕捉短时卡顿。N_out=1 时 TPOT 未定义，不写成 0。", "TPOT is mean subsequent interval for one request; P99 ITL captures stalls. TPOT is undefined for N_out=1, not zero.")],
      ["E2EL", "t_last − t_request = TTFT + (N_out−1) × TPOT", b("适用于非投机逐 Token 可见时间。分块流式时，网络 chunk 不一定是一枚 Token。", "Based on visible-token timestamps. A streamed network chunk need not contain one token.")],
      ["RPS / QPS", b("requests/s；分别记录 offered / admitted / completed", "requests/s; distinguish offered / admitted / completed"), b("一个请求会生成多枚 Token；RPS 不是 TPS。", "One request produces many tokens; RPS is not TPS.")],
      ["Goodput", b("单位时间内同时满足约定 SLA 的请求或 Token 数", "SLO-compliant requests or tokens per unit time"), b("必须声明请求口径还是输出 Token 口径，以及双门槛。高吞吐可以伴随零 Goodput。", "Declare request vs output-token goodput and both gates. High throughput can coexist with zero goodput.")],
      ["$/1M tokens", b("成本 ÷ 同口径 Token × 10⁶", "Cost ÷ same-scope tokens × 10⁶"), b("输入/输出单价不同；集群成本应含闲置、网络、运维并用达标输出作分母。", "Input/output prices differ. Cluster cost includes idle capacity, fabric and operations; use compliant output as denominator.")]
    ])) +
    section("lifecycle", "128 个输出 Token 只有 127 个后续间隔", "128 output tokens contain 127 subsequent intervals",
      lab("lifecycle", lifeControls) +
      formula("TTFT = queue + prefill_and_first_compute + delivery<br>E2EL = TTFT + max(N_out − 1, 0) × TPOT<br>Streaming TPS ≈ 1000 / TPOT_ms, only when N_out &gt; 1") +
      p("默认手算：120 + 300 + 30 = 450 ms TTFT；E2EL = 450 + 127×20 = 2,990 ms；单用户后续流式速率约 50 tok/s。不要用 128×20 又重复计算首 Token。",
        "Default hand check: 120 + 300 + 30 = 450 ms TTFT; E2EL = 450 + 127×20 = 2,990 ms; subsequent streaming rate is about 50 tok/s. Multiplying 128×20 counts the first token twice.")) +
    section("scenarios", "语音的 300 ms 是端到端预算，不是 LLM 全部预算", "A 300 ms voice target is an end-to-end budget, not all for the LLM",
      `<div data-static-chart="scenarios"></div>` +
      table([b("五类场景", "Five scenarios"), b("教学目标（非行业 SLA）", "Teaching target (not an industry SLA)"), b("优先优化", "Prioritize")], [
        [b("实时语音", "Real-time voice"), "End-to-end turn response &lt;300 ms; LLM TTFT budget 120 ms; TPOT 15 ms", b("VAD、ASR、网络、LLM、TTS 分摊预算；首音频与首文本分开测。", "Allocate VAD, ASR, network, LLM and TTS budgets; measure first audio separately.")],
        [b("IDE 补全", "IDE completion"), "TTFT 200 ms; TPOT 15 ms; 64 output tokens", b("取消过时请求、短输出、低并发低延迟池。", "Cancel stale requests; short outputs and a low-latency pool.")],
        [b("企业对话", "Enterprise chat"), "TTFT 800 ms; TPOT 35 ms; 512 output tokens", b("RAG 输入长度、Prefix Caching、平滑流式与双门槛 Goodput。", "RAG input length, prefix caching, smooth streaming and dual-gated goodput.")],
        [b("R1 深度推理", "R1 deep reasoning"), "TTFT 2,000 ms; TPOT 50 ms; 4,096 output tokens", b("长输出预算、质量、首可见答案 vs 隐藏推理 Token；不只看首字。", "Long-output budget and quality; visible answer vs hidden reasoning tokens.")],
        [b("离线批处理", "Offline batch"), b("用小时级截止时间，不设交互 TTFT SLA", "Use an hours-scale deadline, not an interactive TTFT SLO"), b("总完成时间、成本、重试、失败率与吞吐；不要拿离线指标评价语音体验。", "Completion deadline, cost, retry/failure and throughput; not a voice-experience metric.")]
      ]) +
      note("对比图只绘制有定义的交互延迟目标。离线批处理的交互 TTFT 为 N/A，不把缺失值画成 0，也不使用不同比例的雷达图制造视觉误导。",
        "The chart only plots defined interactive latency targets. Batch TTFT is N/A, not zero. Different metric scales are not mixed into a misleading radar chart.")) +
    section("goodput", "越过 SLA 门槛后，更多 batch 反而减少达标吞吐", "Beyond the SLO gate, a larger batch can reduce goodput",
      lab("goodput", goodControls) +
      formula("Synthetic ITL = base + slope×B + 2×slope×max(0, B−knee) + deterministic_jitter<br>Synthetic TTFT_i = 100 + 4×B + 3×((13×i) mod 29)<br>Output TPS ≈ B×1000 / mean_TPOT; Goodput = TPS×(passed_requests/B)") +
      p("这是稳态 decode 近似沙盘，不是实测帕累托前沿。每个 batch 的请求数就是样本分母；同长输出 128 Token 时，达标请求比例等于达标输出比例。不同输出长度必须按 Token 加权，不能照搬比例。",
        "This is a steady-decode approximation, not a measured Pareto frontier. Batch size is the request sample denominator. With equal 128-token outputs, request and output-token pass fractions coincide; unequal lengths require token weighting.")) +
    section("trace", "平均 TPOT 20 ms 的流，仍可能每轮停顿 60 ms", "A 20 ms mean TPOT stream can still stall for 60 ms",
      lab("trace", traces) +
      p("默认是7条原创教学追踪记录，共18枚可见 Token，不是 GPU 或模型 benchmark。burst 记录的间隔为 0、0、60 ms，平均20 ms；切换「平均TPOT」与「最大可见ITL」，Goodput 会从9降到5 tok/s，虽然原始输出都是18 tok/s。失败、拒绝、质量失败和单 Token 都保留在记录中，不丢失失败分母。",
        "Default data contains seven original teaching traces and 18 visible tokens, not a GPU/model benchmark. The burst record has gaps 0,0,60 ms but mean20 ms. Switching mean-TPOT to max-visible-ITL gates changes goodput from9 to5 tok/s while raw visible throughput stays18. Failure, rejection, quality failure and single-token records remain instead of disappearing from denominators.") +
      formula("TTFT_i = first_visible_token_i − client_arrival_i<br>TPOT_i = mean(visible_token_gaps_i), only when N_out&gt;1<br>Visible E2EL_i = last_visible_token_i − arrival_i<br>Terminal latency_i = response_terminal_i − arrival_i<br>L_observed = ∫ active_requests(t)dt / window_length") +
      note("并发图按所选系统边界直接积时间面积，不把 client在途请求当GPU驻留请求。admitted边界也可能包含服务端等待，不等于GPU busy。全记录落在同一测窗时 λ×平均W 与面积L相等只是有限追踪恒等式；截断测窗会偏离，不能因此用短窗或P99做稳态容量保证。",
        "Concurrency integrates time under the chosen boundary; client in-flight requests are not GPU-resident requests. The admitted boundary may include server wait, not GPU busy. With all traces contained, λ×meanW equals areaL as a finite-trace identity. Clipping the window can break that comparison; short windows or P99 do not prove steady capacity.") +
      note("时间戳必须在同一相对时间基准；跨机时钟要先对齐。多个 Token 在一个 chunk 中可有相同可见时间，但0间隔不意味着GPU Decode为0；隐藏推理Token与供应商Usage另算。JSON仅在本页计算、不上传、不写本地存储；自填记录标记user_supplied而非已验证实测。",
        "Timestamps need one aligned relative time base. Tokens in one chunk can share a visible timestamp, but zero gaps do not mean zero GPU decode. Hidden reasoning and provider usage need separate accounting. JSON is processed only here, without upload or local storage; entered traces are user_supplied, not verified measurements.")) +
    section("instrumentation", "数学上的 N/A、Prometheus 的 0 和 benchmark 过滤不是一回事", "Mathematical N/A, Prometheus zero and benchmark filtering are different",
      table([b("采集来源", "Collection source"), b("定义差别", "Definition difference"), b("使用前动作", "Before use")], [
        ["vLLM request_time_per_output_token_seconds", b("每完成请求记录一次；N_out≤1可能以0记入histogram", "One sample per completed request; N_out≤1 may enter histograms as zero"), b("0是采集约定，不让1/0变成无限TPS；与N_out关联过滤", "Zero is a collection convention, not infinite TPS; correlate with output length")],
        ["vLLM inter_token_latency_seconds", b("每stream output event的间隔，不一定逐Token", "Gap per stream-output event, not necessarily per token"), b("说明chunk大小与权重；不能直接和逐请求TPOT均值比较", "State chunk sizes/weights; do not directly compare with per-request TPOT means")],
        ["vLLM bench serve", b("TPOT统计可排除N_out≤1，与Prometheus总体不同", "TPOT statistics can exclude N_out≤1, unlike the Prometheus population"), b("固定版本和样本总体；两个独立histogram无法精确按request join", "Fix version and population; separate histograms cannot exactly join request cohorts")],
        ["NVIDIA AIPerf", b("当前推荐的服务端点性能采集工具；不是本页自动运行", "Current recommended serving-endpoint profiling tool; not run automatically here"), b("记录模型/版本、tokenizer、load模式、测窗与失败状态", "Record model/version, tokenizer, load mode, window and failures")]
      ]) +
      p("复核到2026-11-21，GenAI-Perf 官方页面已标明停止积极开发新功能并推荐 AIPerf。这里仅提供定义与工具选型依据，不安装、不连接任何推理端点，也不引用其不同机器的示例性能来证明本沙盘。",
        "As reviewed on 2026-11-21, the official GenAI-Perf page says new-feature development is phased out and recommends AIPerf. This page only teaches definitions/tool selection; it installs nothing, connects to no endpoint and does not borrow example performance from other machines.")) +
    section("faq", "平均值、尾分位数、Token 定义必须一起交付", "Ship means, tails and token definitions together",
      faq("为什么单用户 TPS 很高，企业平台仍然很慢？", "Why can high single-user TPS coexist with a slow platform?", "测试可能没有排队或只用短输入。生产需要混合长度、持续 offered load、P95/P99 TTFT 与 ITL、拒绝率、质量门槛和 Goodput；先固定同一 tokenizer、模型版本与输出条件。", "The test may have no queue or only short inputs. Production requires mixed lengths, sustained offered load, TTFT/ITL tails, rejection rate, quality gates and goodput under the same tokenizer, model and output conditions.") +
      faq("为什么不能把每个请求的 P99 相加得到 E2EL P99？", "Why not add P99 stage latencies to get P99 E2EL?", "分位数不线性可加。要保留同一请求 trace 的相关性，先对请求求总时延，再对总时延分布取分位数。阶段预算相加只是工程上界预算。", "Quantiles are not additive. Keep correlated stage traces, compute each request's total, then take the total distribution percentile. Adding stage budgets is a planning bound.") +
      quiz("metrics", "输出只有一个 Token，TPOT 应记为什么？", "For a one-token output, what is TPOT?", [["0 ms", "0 ms"], ["未定义 / N/A", "Undefined / N/A"], ["TTFT", "TTFT"]], 1, "分母 N_out−1=0。TTFT 仍然有意义，但没有后续 Token 间隔。", "N_out−1 is zero. TTFT remains meaningful, but there is no subsequent token interval.")) +
    section("checklist", "先写清测量合同，再选引擎和 GPU", "Specify the measurement contract before choosing an engine or GPU",
      checklist([["同一模型、tokenizer、提示词分布、输出长度与质量门槛", "Same model, tokenizer, prompt distribution, output lengths and quality gate"],
        ["分别命名服务端 TTFT、端到端 TTFT、首音频、首可见答案", "Name server TTFT, end-to-end TTFT, first audio and first visible answer separately"],
        ["同时采集平均 TPOT 与 P99 ITL；记录失败、取消和拒绝", "Collect mean TPOT and P99 ITL; record failures, cancellations and rejections"],
        ["声明 Goodput 门槛、分母、时间窗口与成本范围", "Declare goodput gates, denominator, window and cost scope"]]));
}

function capacityContent() {
  const controls = select("capacity", "mode", "需求输入模式", "Demand input mode", [["qps", "Direct peak QPS"], ["business", "DAU → peak QPS"]], "qps") +
    field("capacity", "qps", "业务峰值 QPS", "Business peak QPS", 10, 0, 100000, 0.1) +
    field("capacity", "dau", "日活用户 DAU", "Daily active users", 20000, 0, 100000000) +
    field("capacity", "requests", "每用户每日请求数", "Requests per user per day", 10, 0, 10000, 0.1) +
    field("capacity", "hours", "活跃业务小时/日", "Active business hours/day", 10, 0.1, 24, 0.1) +
    field("capacity", "peak", "峰值/活跃时段平均倍率", "Peak/active-hours mean factor", 3, 1, 100, 0.1) +
    field("capacity", "headroom", "需求余量倍率", "Demand headroom multiplier", 1.3, 1, 10, 0.1) +
    field("capacity", "prompt", "平均输入 Token", "Mean input tokens", 2048, 0, 1048576) +
    field("capacity", "output", "平均输出 Token（含首字）", "Mean output tokens (including first)", 512, 1, 65536) +
    field("capacity", "ttft", "目标/假设平均 TTFT ms", "Target / assumed mean TTFT ms", 800, 0, 60000) +
    field("capacity", "tpot", "目标/假设平均 TPOT ms", "Target / assumed mean TPOT ms", 20, 0.1, 1000, 0.1) +
    field("capacity", "inputCapacity", "实例实测输入 TPS（示例值）", "Measured input TPS/instance (example)", 20000, 1, 10000000) +
    field("capacity", "outputCapacity", "实例实测输出 TPS（示例值）", "Measured output TPS/instance (example)", 1500, 1, 10000000) +
    field("capacity", "batch", "实例活跃请求上限（示例值）", "Active-request cap/instance (example)", 128, 1, 4096) +
    field("capacity", "utilization", "实测容量使用比例", "Fraction of measured capacity used", 0.7, 0.1, 1, 0.05);
  const ptus = field("ptu", "inputTPM", "输入 TPM", "Input TPM", 200000, 0, 1e12) +
    field("ptu", "outputTPM", "输出 TPM", "Output TPM", 20000, 0, 1e12) +
    field("ptu", "cache", "输入缓存命中比例", "Input cache-hit fraction", 0.5, 0, 1, 0.05) +
    field("ptu", "cachedWeight", "缓存输入权重（相对未缓存=1）", "Cached-input weight (uncached=1)", 0, 0, 100, 0.1) +
    field("ptu", "outputWeight", "输出相对输入权重", "Output/input weight", 8, 0.1, 100, 0.1) +
    field("ptu", "capacity", "每 PTU 归一化 TPM（手工录入）", "Normalized TPM/PTU (manual)", 3400, 1, 1e9) +
    field("ptu", "minimum", "部署最小 PTU（手工录入）", "Minimum deployment PTU (manual)", 15, 1, 10000) +
    field("ptu", "step", "扩容步长 PTU（手工录入）", "PTU scale increment (manual)", 5, 1, 10000);
  const costs = field("cost", "hourly", "完整集群成本 $/小时（假设）", "Whole-cluster cost $/hour (assumption)", 40, 0, 100000, 0.1) +
    field("cost", "outputTPS", "达标输出 Goodput tok/s", "Compliant output goodput tok/s", 5000, 0, 1e9) +
    field("cost", "inputTPS", "API 输入 TPS", "API input TPS", 20000, 0, 1e9) +
    field("cost", "hours", "同负载持续小时", "Hours at the same load", 8, 0, 8760, 0.1) +
    field("cost", "priceIn", "API 输入 $/1M Token（假设）", "API input $/1M tokens (assumption)", 1, 0, 1000, 0.01) +
    field("cost", "priceOut", "API 输出 $/1M Token（假设）", "API output $/1M tokens (assumption)", 5, 0, 1000, 0.01);
  const mixed = field("mixture", "qps", "业务峰值 QPS", "Business peak QPS", 10, 0, 100000, 0.1) +
    field("mixture", "headroom", "需求余量倍率", "Demand headroom", 1.3, 1, 10, 0.1) +
    field("mixture", "shortShare", "IDE 请求占比", "IDE request fraction", 0.5, 0, 1, 0.05) +
    field("mixture", "ragShare", "RAG 请求占比（剩余为推理）", "RAG fraction (remainder is reasoning)", 0.4, 0, 1, 0.05) +
    field("mixture", "inputCapacity", "同混合负载实例输入 TPS（示例）", "Instance input TPS for this mix (example)", 20000, 1, 10000000) +
    field("mixture", "outputCapacity", "同混合负载实例输出 TPS（示例）", "Instance output TPS for this mix (example)", 1500, 1, 10000000) +
    field("mixture", "utilization", "实测吞吐使用比例", "Measured-capacity fraction used", 0.7, 0.1, 1, 0.05) +
    field("mixture", "batch", "每实例同边界活动请求预算", "Same-boundary active request budget/instance", 128, 1, 4096);
  const bursts = field("burst", "baseQPS", "恢复后基线 QPS", "Post-burst base QPS", 8, 0, 100000, 0.1) +
    field("burst", "burstMultiplier", "突发倍率", "Burst multiplier", 3, 1, 100, 0.1) +
    field("burst", "burstSeconds", "突发持续秒", "Burst duration seconds", 30, 1, 600) +
    field("burst", "serviceRPS", "同负载集群可持续 RPS（示例）", "Sustainable cluster RPS for this workload (example)", 10, 0.1, 100000, 0.1) +
    field("burst", "queueBudgetMs", "仅分给排队的 TTFT 预算 ms", "TTFT budget allocated only to queueing ms", 300, 0, 60000) +
    field("burst", "horizonSeconds", "观察秒数（≥突发时长）", "Observation horizon seconds (≥burst)", 300, 1, 3600);
  const tokenVolumes = [
    ["shortInput", "短普通输入（不含缓存读/写）", "Short ordinary input (excluding cache read/write)", 100000],
    ["shortRead", "短缓存读", "Short cached input", 100000],
    ["shortWrite30", "短30分钟缓存写", "Short 30-minute cache write", 20000],
    ["shortWrite12", "短12小时缓存写（示例明确为0）", "Short 12-hour cache write (example explicitly zero)", 0],
    ["shortOutput", "短上下文输出", "Short-context output", 20000],
    ["longInput", "长普通输入（不含缓存读/写）", "Long ordinary input (excluding cache read/write)", 10000],
    ["longRead", "长缓存读", "Long cached input", 10000],
    ["longWrite30", "长30分钟缓存写", "Long 30-minute cache write", 2000],
    ["longWrite12", "长12小时缓存写（示例明确为0）", "Long 12-hour cache write (example explicitly zero)", 0],
    ["longOutput", "长上下文输出", "Long-context output", 2000]
  ];
  const normalized = select("normalized", "model", "官方归一化参数快照（非实时quota）", "Published normalization snapshot (not live quota)", Object.entries(math.ptuModels).map(([key, model]) => [key, `${model.name} / ${model.version}`]), "gpt62sol") +
    select("normalized", "deployment", "部署类别（决定最小值/步长）", "Deployment class (minimum/increment)", [["global-zone", "Global / Data Zone"], ["regional", "Regional"]], "global-zone") +
    tokenVolumes.map(([key, z, e, value]) => field("normalized", key, z + " TPM", e + " TPM", value, 0, 1e12)).join("");
  return section("concepts", "TPM=60×TPS 只在同一 Token 口径、同一窗口下成立", "TPM=60×TPS only for the same token scope and window",
    table([b("范围", "Scope"), b("公式", "Formula"), b("不要混淆", "Do not confuse")], [
      [b("单用户流式速率", "Per-user streaming"), "TPS_user ≈ 1000 / TPOT_ms", b("不含首字等待；不是系统吞吐。", "Excludes first-token wait; not system throughput.")],
      [b("系统输入吞吐", "System input throughput"), "Input TPS = completed_RPS × E[input_tokens]", b("稳定系统中 offered≈completed；过载时不成立。", "Offered≈completed only in a stable system.")],
      [b("系统输出吞吐", "System output throughput"), "Output TPS = completed_RPS × E[output_tokens]", b("计入首 Token；不要乘活跃 batch 又重复统计。", "Includes first token; do not multiply by batch again.")],
      [b("输入/输出 TPM", "Input/output TPM"), "TPM = 60 × same_scope_TPS", b("云端限流另有RPM、时间窗口和部署口径；不能照搬旧式max_tokens预留估算。", "Cloud admission also has RPM, time windows and deployment-specific accounting; do not reuse an older max_tokens reservation rule.")],
      ["Little's Law", "L = λ × E[W]", b("L 为平均在系统中请求数；W 为对应系统平均停留时间。不可代入 P99 假装严格等式。", "L is mean in-system requests and W their mean residence. Substituting P99 is not the exact law.")],
      ["PTU", b("带模型/部署权重的归一化容量", "Model/deployment-weighted normalized capacity"), b("没有通用 1 PTU = X TPM，也没有通用 PTU→GPU 换算。", "No universal PTU→TPM or PTU→GPU conversion.")]
    ])) +
    section("capacity", "峰值 QPS 同时决定输入、输出和在途并发", "Peak QPS drives input, output and in-flight concurrency",
      lab("capacity", controls) +
      formula("QPS_peak = DAU × requests_per_day × peak_factor / (active_hours × 3600)<br>λ_plan = QPS_peak × demand_headroom<br>W_plan = (mean_TTFT_ms + (N_out−1)×mean_TPOT_ms) / 1000<br>L_plan = λ_plan × W_plan<br>instances ≥ max(ceil(inputTPS/C_input), ceil(outputTPS/C_output), ceil(L/batch_cap))") +
      note("这里的目标 TTFT/TPOT 被当作规划均值假设，而非已证实均值或 P99。用实测 E[W] 替换后才是 Little's Law 的观测验证。输入和输出容量必须来自同一混合负载实测，不能拼两个独立峰值。",
        "Targets are treated as assumed planning means, not observed means or P99. Replace them with measured E[W] to verify Little's Law. Input/output capacities must come from the same mixed workload, not two independent peak tests.") +
      p("默认手算：10 QPS×1.3=13 QPS；输入 26,624 TPS→1,597,440 TPM；输出 6,656 TPS→399,360 TPM；W=0.8+511×0.02=11.02 s，平均在途 L=143.26。使用 70% 容量时，输入/输出/并发约束分别给出 2/7/2 个实例，先取 7，再压测。实例不是一张 GPU。",
        "Default hand check: 10×1.3=13 QPS; input 26,624 TPS→1,597,440 TPM; output 6,656 TPS→399,360 TPM; W=0.8+511×0.02=11.02 s and L=143.26. At 70% of example capacity, input/output/concurrency bounds require 2/7/2 instances: start from 7, then validate. An instance is not one GPU.")) +
    section("mixture", "10% 长推理请求可占约72%平均在途并发", "10% long reasoning requests can occupy about72% of mean concurrency",
      lab("mixture", mixed) +
      formula("Input TPS = λ×Σ f_i×input_tokens_i; Output TPS = λ×Σ f_i×output_tokens_i<br>E[W] = Σ f_i×[TTFT_i+(output_i−1)×TPOT_i]<br>L = Σ λ×f_i×W_i<br>Generally E[(N_out−1)×TPOT] ≠ (E[N_out]−1)×E[TPOT]") +
      p("三类固定教学画像：IDE为256输入/64输出/200ms TTFT/15ms TPOT；RAG为4096/512/800/35；推理为8192/4096/2000/50。50%/40%/10%请求混合时，E[W]=28.7215s，13计划QPS对应L=373.3795；推理类占10%请求，却贡献约71.98%在途并发。它们是各类规划均值，不是硬件 SLA。",
        "Fixed teaching profiles: IDE=256 input/64 output/200ms TTFT/15ms TPOT; RAG=4096/512/800/35; reasoning=8192/4096/2000/50. At a50%/40%/10% request mix, E[W]=28.7215s and13 planned QPS gives L=373.3795. Reasoning is10% of requests but about71.98% of mean concurrency. These are assumed class means, not hardware SLAs.") +
      note("若先平均输出长度和TPOT再相乘，默认会得到错误的17.7231s，而不是28.7215s。输出长度和每Token时间相关时，必须按同类/同请求相乘后求平均。L仍是指定系统边界的在途量，不能直接当GPU KV驻留量；实例容量需按同混合负载测量。",
        "Multiplying separately averaged output length and TPOT gives the incorrect17.7231s rather than28.7215s. Correlated lengths/token times must be multiplied within each class/request before averaging. L is in-flight population under the specified boundary, not automatically GPU KV residency; measure instance capacity on this same mix.")) +
    section("burst", "30秒突发造成的积压，可能要210秒才排空", "A30-second burst can need210 seconds to drain",
      lab("burst", bursts) +
      formula("Fluid Q_peak = max(0, λ_burst−μ)×T_burst<br>Drain time = Q_peak/(μ−λ_base), only when μ&gt;λ_base<br>Queue wait approximation = Q/μ<br>Queue-only admission budget = μ×queue_budget_seconds") +
      p("默认：服务10 RPS、基线8 QPS、突发3倍持续30秒。积压=(24−10)×30=420个请求等价量，峰值排队等待近似42秒；恢复余量只有2 RPS，需要420/2=210秒。若排队只允许300ms，流体队列预算约3个请求，多余约417个请求等价量需拒绝/转移，而不是悄悄违反 TTFT。",
        "Default: service 10 RPS, base 8 QPS, a 3× burst for 30 s. Backlog=(24−10)×30=420 request-equivalents and approximate peak wait 42 s. Recovery has only 2 RPS spare capacity, so drain takes 420/2=210 s. A 300 ms queue budget holds about 3 fluid request-equivalents; about 417 excess equivalents need rejection/routing rather than silent TTFT violations.") +
      note("这是固定平均服务速率的流体教学模型，不是离散scheduler、随机排队或P99预测。μ必须来自同负载可持续能力；R1长短混合、重试、抢占和缓存冷热会改变它。限流不创造算力，只把无限排队转成可见拒绝/降级。排队预算也不是全部 TTFT。",
        "This is a fixed-average-rate fluid teaching model, not a discrete scheduler, stochastic queue or P99 forecast. Measure μ on the same workload; reasoning mix, retries, preemption and cache state change it. Admission creates no compute: it replaces unbounded waiting with explicit rejection/degradation. Queue budget is not the whole TTFT.")) +
    section("ptu", "缓存与输出权重改变 PTU，却不改变原始 Token 计数", "Caching and output weights change PTU, not raw token counts",
      lab("ptu", ptus) +
      formula("Normalized TPM = Input TPM × [(1−cache_hit) + cache_hit×cached_weight]<br>                 + Output TPM × output_weight<br>Raw PTU = Normalized TPM / normalized_TPM_per_PTU<br>Deploy PTU = minimum + max(0, ceil((Raw PTU−minimum)/step))×step") +
      p("这个两权重沙盘的默认数值对应 Microsoft Learn 中 GPT-5.2 Data Zone 的教学算例：输入200K、输出20K、输出权重8、每PTU 3,400输入等价TPM。50%缓存且缓存权重0时得260K归一化TPM→76.47 PTU→80。复核表中此类部署最小15、步长5，但字段仍可手工改，且不是实时quota或可部署容量查询。",
        "Default weights mirror the Microsoft Learn GPT-5.2 Data Zone example: 200K input, 20K output, output weight 8 and 3,400 input-equivalent TPM/PTU. At 50% cache and zero cached weight: 260K normalized TPM→76.47 PTU→80. The reviewed table lists minimum 15/increment 5, but fields remain manual and are not live quota or deployable-capacity discovery.") +
      note("GPT-6+、多模态、长上下文档位、缓存写入等可能采用更多权重。本沙盘只适合两类输入+输出模型的讲解；额外类别必须按当前官方公式单独扩展，不能把它们默认当作免费或零。",
        "GPT-6+, multimodal inputs, context tiers and cache writes can require additional weights. This workbench only explains two input categories plus output. Extra categories must use current official formulas, not silently become free or zero.")) +
    section("normalized", "GPT-6缓存读与写都消耗PTU，不能沿用旧式免费缓存", "GPT-6 cache reads/writes consume PTU; the older free-cache rule does not apply",
      lab("normalized", normalized) +
      formula("Normalized TPM = Σ mutually_exclusive_category_TPM×category_weight<br>Raw PTU = Normalized TPM / model_version_normalized_TPM_per_PTU<br>Each token is assigned once: ordinary input OR cache read OR cache write; 30min OR12h write<br>Deploy size is rounded using the selected minimum and increment") +
      p("官方2026-11-21参数快照：GPT-6.1/6.2的短缓存读权重0.05、长缓存读0.1；短/长普通输入1/2，输出5/7.5，30分钟写1.25/2.5，12小时写2.5/5。GPT-6 Sol/Luna/Astra缓存读为0.1/0.2。短/长档位由模型版本和官方定价定义，不在此按字符数或统一128K阈值自动推断。",
        "Published parameters reviewed2026-11-21: GPT-6.1/6.2 short/long cache-read weights0.05/0.1; ordinary input1/2, output5/7.5,30-minute write1.25/2.5 and12-hour write2.5/5. GPT-6 Sol/Luna/Astra cache reads use0.1/0.2. Context tiers follow model/version/pricing definitions, not character counts or an assumed universal128K cutoff.") +
      p("默认互斥Token分类的原始总量264K TPM，正确归一化271K。gpt-6.2-sol 2026-11-17每PTU预算2,950，raw=91.86；Global/Data Zone取95 PTU，Regional按25最小/25步长取100。错误把缓存读当免费会算成90个Global PTU。这里没有查询订阅额度、地区可用性或给报价。",
        "Default mutually exclusive categories total 264K raw TPM and 271K normalized TPM. gpt-6.2-sol / 2026-11-17 has 2,950 normalized TPM/PTU: raw 91.86, rounded 95 for Global/Data Zone or 100 for Regional's minimum 25/increment 25. Incorrectly treating reads as free gives 90 Global PTU. This does not query subscription quota, regional availability or pricing.") +
      note("不能把cache write又算ordinary input，或把12小时写重复算30分钟写。旧模型输出权重也不一定等于报价比例；按具体版本与部署核对。零需求显示0仅代表不新建容量，已部署PTU的闲置费用不会自动消失。Standard RPM/TPM限流与PTU归一化预算是不同合同。",
        "Do not count cache writes again as ordinary input or 12-hour writes again as 30-minute writes. Older output weights can differ from price ratios; verify model/version/deployment. Zero demand means no new planned capacity, not free existing PTU idle time. Standard RPM/TPM limits and normalized PTU budgets are different contracts.")) +
    section("cost", "达标输出作分母，才不会奖励失败的峰值吞吐", "Use compliant output as the denominator, not failed peak throughput",
      lab("cost", costs) +
      formula("Self-hosted $/1M_output = whole_cluster_$per_hour × 10⁶ / (3600×output_goodput)<br>API period cost = hours×3600×(inputTPS×price_input + outputTPS×price_output)/10⁶") +
      p("默认自建成本：40×10⁶/(3600×5000)=$2.22/1M 达标输出。API 示例：8 小时、20K 输入 TPS×$1 + 5K 输出 TPS×$5，合计 $1,296。两种成本不是同一种商业范围：云单价、输入处理、自建闲置/冗余/人工都要对齐。输出 Goodput=0 时单位成本未定义，不显示 $0。",
        "Default self-hosted cost: 40×10⁶/(3600×5000)=$2.22/1M compliant output. API example: 8 hours of 20K input TPS at $1 and 5K output TPS at $5 totals $1,296. These are different commercial scopes; align rates, input work, idle/resilience and staffing. At zero output goodput, unit cost is undefined, not $0.")) +
    section("faq", "活跃请求数既不是在线用户数，也不是静态 batch", "Active requests are neither online users nor a fixed batch",
      faq("为什么 DAU 不能直接变成并发数？", "Why cannot DAU directly become concurrency?", "DAU 需要请求频率、活跃时段、峰值集中度与响应时间才能映射到 QPS 和在途请求。用户在线但没有生成请求，不占 decode slot。", "DAU needs request frequency, active hours, peak concentration and response time to map to QPS and in-flight requests. An online user with no generation does not occupy a decode slot.") +
      faq("一分钟没用完额度，为什么仍会429？", "Why can a request get 429 before the minute budget is exhausted?", "2026-11-21直读Microsoft Learn新/经典页面，均说明Token限流使用实际输入与生成Token，而非把配置的最大输出值预留进TPM；旧检索摘要仍见max_tokens/best_of估算，不能拿缓存摘要覆盖现行页面。RPM可能按1或10秒短窗口判断：若600 RPM按1秒窗口检查，超过10请求/秒仍可能429，哪怕整分钟少于600。按部署核对remaining/reset、retry-after-ms与实际usage；PTU是另一套利用率评估。这里只核验文档，不宣称部署实测或服务变更生效日期。",
        "Direct reads of both current/classic Microsoft Learn pages on 2026-11-21 describe actual input/generated tokens for token limits, not reserving the configured maximum output value. Older search snippets still describe max_tokens/best_of estimates; cached snippets do not override the live pages. RPM can use 1- or 10-second windows: under a 1-second check, a 600-RPM deployment can reject more than 10 requests/s despite fewer than 600 in the minute. Check remaining/reset headers, retry-after-ms and usage on the specific deployment; PTU uses different utilization-based evaluation. This verifies documentation, not runtime behavior or a service-transition date.") +
      faq("为什么输出 4K 的 R1 会比输出 512 的聊天更难规划？", "Why is 4K-token reasoning harder to size than 512-token chat?", "输出拉长 decode 停留时间，增加在途并发和驻留 KV；还可能包含不显示但消耗算力、quota 和费用的推理 Token。要区分业务可见输出、模型生成输出和供应商计费口径。", "Longer output increases decode residence, in-flight concurrency and resident KV. Hidden reasoning tokens may consume compute, quota and billing. Separate visible answer, generated output and provider accounting.") +
      quiz("capacity", "10 QPS、每请求 512 输出 Token，系统 Output TPS 是？", "At 10 QPS with 512 output tokens/request, what is system output TPS?", [["50", "50"], ["5,120", "5,120"], ["307,200", "307,200"]], 1, "10×512=5,120 tok/s；再乘 60 才是 307,200 TPM。", "10×512=5,120 tok/s; multiply by 60 for 307,200 TPM.")) +
    section("checklist", "用同一业务画像同时验证吞吐、并发和 quota", "Validate throughput, concurrency and quota against one traffic profile",
      checklist([["高峰输入、输出长度保留分布，而不只有均值", "Keep peak input/output length distributions, not only means"],
        ["压测包括排队、拒绝、超时、取消与缓存冷热两种状态", "Include queues, rejection, timeout, cancellation and cold/warm cache"],
        ["输入/输出容量为同一配置、同一混合负载测量", "Measure input/output capacities on the same configuration and mixed load"],
        ["PTU 模型版本、区域、部署类型、权重、最小值与步长独立核对", "Verify PTU model version, region, deployment, weights, minimum and increment"]]));
}

function sizingContent() {
  const controls = select("sizing", "model", "模型 / 架构预设", "Model / architecture preset", models, "llama70") +
    select("sizing", "gpu", "GPU 预设（名义 GB）", "GPU preset (nominal GB)", gpus, "h200") +
    select("sizing", "weightBytes", "权重存储精度", "Weight storage precision", [["2", "BF16 / FP16: 2 B"], ["1", "FP8: 1 B"], ["0.5", "INT4: 0.5 B"]], "1") +
    select("sizing", "kvBytes", "KV 存储精度（单独选择）", "KV precision (independent)", [["2", "BF16 / FP16: 2 B"], ["1", "FP8: 1 B (kernel-dependent)"]], "2") +
    field("sizing", "batch", "峰值 KV 驻留请求上限 B（含 Prefill/Decode）", "Peak KV-resident request cap B (prefill + decode)", 32, 1, 512, 1, "range") +
    field("sizing", "context", "每请求驻留上下文 Token S", "Resident context tokens/request S", 8192, 256, 131072, 256, "range") +
    field("sizing", "prefillTokens", "冷 Prefill 输入 Token S_p（不含未来输出）", "Cold prefill input tokens S_p (no future output)", 8192, 256, 131072, 256, "range") +
    field("sizing", "prefillBatch", "同时完整 Prefill 请求数（≤驻留 B）", "Simultaneous full-prefill requests (≤resident B)", 1, 1, 512) +
    field("sizing", "safe", "规划显存安全水位", "Planning safe memory fraction", 0.85, 0.5, 0.95, 0.01) +
    field("sizing", "runtime", "每卡运行时预留 GB（假设）", "Runtime reserve GB/GPU (assumption)", 3, 0, 1000, 0.5) +
    field("sizing", "overhead", "量化 scales / padding 开销比例", "Quantization scales / padding fraction", 0.05, 0, 1, 0.01) +
    field("sizing", "replication", "聚合 KV 复制倍率（手工）", "Aggregate KV replication (manual)", 1, 1, 64, 0.1) +
    field("sizing", "mbu", "HBM/GDDR 有效带宽比例 MBU（假设）", "HBM/GDDR bandwidth fraction MBU (assumption)", 0.6, 0.01, 1, 0.01) +
    field("sizing", "mfu", "Prefill dense BF16 MFU（假设）", "Prefill dense BF16 MFU (assumption)", 0.4, 0.01, 1, 0.01) +
    field("sizing", "tpot", "目标 TPOT ms", "Target TPOT ms", 20, 0.1, 1000, 0.1) +
    field("sizing", "ttft", "目标 TTFT ms", "Target TTFT ms", 800, 1, 60000) +
    field("sizing", "reserve", "TTFT 非算力预留 ms", "Non-compute TTFT reserve ms", 100, 0, 60000) +
    check("sizing", "mtp", "DeepSeek 额外加载 14B MTP 模块", "Load DeepSeek's extra 14B MTP module");
  const ranks = select("rank", "model", "每 rank 模型预设", "Per-rank model preset", models, "llama70") +
    select("rank", "gpu", "每 rank GPU", "GPU per rank", gpus, "h200") +
    select("rank", "tp", "Tensor Parallel degree", "Tensor parallel degree", [1, 2, 4, 8, 16, 32, 64].map((n) => [String(n), String(n)]), "16") +
    select("rank", "pp", "Pipeline stages", "Pipeline stages", [1, 2, 4, 8, 16, 32].map((n) => [String(n), String(n)]), "1") +
    select("rank", "dp", "整模型副本 DP（不是 Attention DP）", "Full-model replica DP (not attention DP)", [1, 2, 4, 8, 16].map((n) => [String(n), String(n)]), "1") +
    field("rank", "batch", "全组同时驻留请求 B", "Resident requests across all replicas B", 32, 1, 512, 1, "range") +
    field("rank", "context", "每请求峰值驻留 Token S", "Peak resident tokens/request S", 8192, 256, 131072, 256, "range") +
    select("rank", "weightBytes", "权重存储字节", "Weight storage bytes", [["2", "BF16/FP16 2 B"], ["1", "FP8 1 B"], ["0.5", "INT4 0.5 B"]], "1") +
    select("rank", "kvBytes", "KV 参考精度（MLA 实测字节优先）", "KV reference precision (measured MLA bytes override)", [["2", "BF16/FP16 2 B"], ["1", "FP8 1 B"]], "2") +
    field("rank", "overhead", "权重 metadata 预留比例", "Weight metadata allowance", 0.05, 0, 1, 0.01) +
    field("rank", "weightSkew", "stage/rank 权重不均衡预算倍率", "Stage/rank weight-skew budget multiplier", 1, 1, 3, 0.05) +
    field("rank", "runtime", "每 rank runtime/CUDA/通信预留 GB", "Runtime/CUDA/communication reserve GB/rank", 3, 0, 1000, 0.5) +
    field("rank", "safe", "安全水位", "Safe memory fraction", 0.85, 0.5, 0.95, 0.01) +
    select("rank", "mlaCache", "MLA 每 rank 缓存布局", "MLA per-rank cache layout", [["replicated", "Replicated packed latent (reference)"], ["measured", "Measured opaque per-layer row bytes"]], "replicated") +
    field("rank", "mlaLayerBytes", "实测 MLA 每 layer/Token/rank 字节", "Measured MLA bytes/layer/token/rank", 1152, 1, 1000000) +
    check("rank", "measuredMemory", "使用设备实测总可见显存（不是剩余 free）", "Use measured total visible device memory (not remaining free)") +
    field("rank", "deviceGB", "实测设备总可见 GB（十进制）", "Measured total visible device GB (decimal)", 141, 1, 2000, 0.01) +
    check("rank", "customReplicaBatch", "使用最忙副本的实测活动请求数", "Use measured busiest-replica active batch") +
    field("rank", "replicaBatch", "最忙副本活动请求（≤全组 B）", "Busiest replica batch (≤global B)", 32, 1, 512) +
    check("rank", "mtp", "checkpoint 实际加载额外 MTP 14B", "Checkpoint actually loads extra 14B MTP");
  return section("concepts", "37B 激活参数能省计算，不能把 671B 权重变没", "37B active parameters save compute, not 671B weight storage",
    `<div class="grid three"><article class="card"><h3>${b("水位线 1：放得下", "Bound 1: fit")}</h3>${p("总权重 + KV Cache + 激活、CUDA Graph、通信缓冲与碎片。先做聚合下界，再核对最忙 rank。", "Total weights + KV + activations, CUDA graphs, communication buffers and fragmentation. Start with an aggregate floor, then check the busiest rank.")}</article><article class="card"><h3>${b("水位线 2：吐得快", "Bound 2: decode")}</h3>${p("低 batch Decode 常受内存带宽限制，但 batch 放大后也会转为算力或通信瓶颈。MoE 批内专家并集会增加实际权重流量。", "Low-batch decode is often bandwidth-bound; larger batches can become compute/communication-bound. MoE's batch expert union increases weight traffic.")}</article><article class="card"><h3>${b("水位线 3：首字快", "Bound 3: prefill")}</h3>${p("Prefill 要在扣除排队、网关与交接后的 TTFT 预算内完成。长上下文注意力有近似二次项。", "Prefill must finish within TTFT after queue, gateway and handoff reserves. Long-context attention contributes an approximate quadratic term.")}</article></div>` +
    formula("M_w [GB] = (total_params_B + optional_MTP_B) × weight_bytes × (1+metadata_fraction)<br>M_KV [B] = 2×layers×KV_heads×head_dim×S×B×KV_bytes   [MHA / GQA]<br>M_MLA [B] = layers×(latent_rank+RoPE_dim)×S×B×KV_bytes [compressed cache]<br>N_mem ≥ ceil((M_w + M_KV_GB) / (GPU_GB×safe_fraction − runtime_GB_per_GPU))") +
    p("MHA 有一份每个 query head 对应的 K/V；GQA 让多个 query head 共享 KV head。MLA 存压缩 latent 与 RoPE key，不能把 128 个 attention head 直接代入标准公式。DeepSeek 默认每 Token BF16 KV = 61×(512+64)×2 = 70,272 B。",
      "MHA has K/V per query head; GQA shares fewer KV heads across query heads. MLA stores compressed latent and RoPE key, not 128 standard K/V heads. DeepSeek's modeled BF16 KV/token = 61×(512+64)×2 = 70,272 B.")) +
    section("sizing", "拖动并发和上下文，看哪道水位线先越界", "Drag batch and context to see which bound is crossed first",
      lab("sizing", controls) +
      note("沙盘是理想可切分的聚合下界，2 的幂候选只是待压测配置，不是可直接运行的 TP 配置。TP 超过 KV heads 时 KV 可能复制；MLA、DP Attention、PP、EP 的分片方式随引擎变化。逐 rank 内存峰值才是最终依据。",
        "The workbench gives an ideal aggregate-sharding floor. A power-of-two candidate is for testing, not automatically a valid TP configuration. KV may replicate when TP exceeds KV heads; MLA, attention DP, PP and EP sharding vary by engine. Per-rank peak memory is decisive.") +
      note("规格纠偏：L40S 是 GDDR6，不是 HBM，而且没有 NVLink；B200 192 GB 为设计/假设预设，DGX B200 官方交付表为 1,440 GB/8=180 GB 每卡。模型和精度预设不等于已验证 kernel 支持。",
        "Spec correction: L40S uses GDDR6, not HBM, and has no NVLink. B200 192 GB is a design/what-if preset; official DGX B200 lists 1,440 GB/8=180 GB per GPU. Model/precision presets do not prove kernel support.")) +
    section("rank", "TP=16 时，8 个 KV heads 不会变成每卡半个 head", "At TP=16, eight KV heads do not become half a head per GPU",
      lab("rank", ranks) +
      formula("GQA KV_heads_per_rank = max(1, n_KV/TP), with valid divisibility<br>KV_rank = 2×layers_on_stage×KV_heads_per_rank×d_head×S×B_replica×KV_bytes<br>MLA_rank = layers_on_stage×packed_bytes_per_layer_token_rank×S×B_replica<br>GPU_count = TP×PP×replica_DP; peak_rank = W_rank + KV_rank + runtime_rank") +
      p("vLLM LlamaAttention 在 TP 小于 KV heads 时切分 KV；TP 大于 KV heads 时复制 KV heads，并要求可整除。默认 70B、8 KV heads：TP=8 与 TP=16 都是每 rank 1 个 KV head，但 TP=16 的总 KV 预留翻倍。上方聚合下界与本沙盘输入独立，比较前先对齐模型、S、B 与精度。",
        "vLLM LlamaAttention partitions KV heads when TP is smaller and replicates them when TP exceeds KV-head count, with divisibility constraints. For default 70B/8 KV heads, TP=8 and TP=16 both have one KV head/rank, but TP=16 doubles aggregate KV reservation. This workbench and the earlier aggregate floor use independent inputs; align model, S, B and dtype before comparing.") +
      note("这是整模型 TP×PP×副本 DP 的内存近似，不替代 Attention DP+EP。权重按层数均衡分配再乘不均衡预算；embedding/head、MTP、热门专家冗余和实际 partition 仍要测量。PP 分层能省驻留，却不会把单请求的串行 Decode 自动加速。",
        "This approximates full-model TP×PP×replica-DP memory, not attention DP+EP. Weights follow balanced layer counts plus a skew budget; measure embeddings/heads, MTP, redundant experts and actual partitions. PP saves residency but does not automatically accelerate serial single-request decode.") +
      note("MLA 的压缩 latent 不等于天然按 TP 均分。参考复制布局保留完整 latent/RoPE；若 kernel 使用 opaque packed layout，请录入实测每 layer/Token/rank 字节，不能只按 head_size×dtype 猜。副本热点可覆盖均衡请求假设；所有“fits”只是当前预算下内存满足，不是 SLA 或运行成功承诺。",
        "Compressed MLA latent is not automatically evenly TP-sharded. The reference replicated layout retains full latent/RoPE; enter measured bytes/layer/token/rank for opaque layouts instead of inferring head_size×dtype. Busiest-replica measurements can override balanced routing. A 'fits' result only means this memory budget passes, not an SLO or successful-runtime guarantee.")) +
    section("roofline", "单 Token 带宽下界不能预测大 batch MoE 的速度", "A single-token bandwidth bound cannot predict large-batch MoE speed",
      formula("TPOT_weight_floor_ms ≈ active_weight_GB / (N_GPU×BW_GBps×MBU) × 1000<br>Uniform expert union = E×[1−(1−k/E)^B]<br>shared_params = (active_params−total_params×k/E)/(1−k/E)<br>batch_active_params = shared_params + routed_params×[1−(1−k/E)^B]<br>Prefill_FLOPs ≈ 2×active_params×S_p×B_prefill + 2×layers×hidden×S_p²×B_prefill<br>TTFT_compute_floor = Prefill_FLOPs / (N_GPU×dense_BF16_FLOPs×MFU)") +
      p("37B 激活只描述每 Token。32 个不同请求可能触发大量不同专家，不能仍假设整个 batch 只读 37B。沙盘给出「单 Token 乐观下界」和「均匀路由批内并集场景」两条读权重估算，并额外显示读全 KV 的时间；均匀路由只是解释模型，不是现实路由预测。",
        "37B active describes one token. A batch of 32 distinct requests can activate many experts; the batch need not read only 37B. The workbench shows a single-token optimistic bound and a uniform-routing batch-union scenario, plus a full-KV read estimate. Uniform routing explains the effect; it does not predict real routing.") +
      p("Prefill 使用 dense BF16 标称 FLOPs×假设 MFU，而不是把权重量化自动当成算力翻倍。MoE 通信、内核效率、量化转换、attention 具体实现与首 Token decode 均可能让实际 TTFT 更高。",
        "Prefill uses nominal dense BF16 FLOPs×assumed MFU, not an automatic compute doubling from quantized storage. MoE communication, kernels, dequantization, attention implementation and first-token decode can all increase real TTFT.") +
      p("冷 Prefill 输入 S_p 与峰值驻留 S 现在分别输入。例如 8192 Prompt + 最多512 输出，应为 S_p=8192、峰值 S≈8704；增加未来输出预算会增 KV，而不应凭空增加此次冷 Prefill FLOPs。B_prefill 是同时做完整 Prefill 的请求数，与驻留/decode B 分开；前缀缓存后的增量 Prefill 还需要考虑历史键长度，本公式不预测该混合情形。",
        "Cold input S_p and peak resident S are entered separately. An 8192-token prompt plus up to 512 output tokens gives S_p=8192 and peak S≈8704. Future output increases KV but must not invent extra cold-prefill FLOPs. B_prefill is separate from resident/decode B; incremental cached prefill also depends on past key length and is outside this cold-prefill formula.") +
      p("B 在本沙盘表示同一实例所有会占 KV 的峰值请求预算，而不只是已有 Decode 请求；B_prefill≤B。若 32 个 decode 外又来一个 prefill，应预算至少33个驻留请求。PD 的两个独立池应分别测算，不把不同池的 B 混到一张卡上。",
        "Here B budgets all peak KV-resident requests in one instance, not only existing decode requests, and B_prefill≤B. Thirty-two decodes plus another prefill need at least 33 resident slots. Size independent PD pools separately instead of mixing their B values on one GPU.")) +
    section("worked", "70B FP8 权重约 70.6 GB；32×8K BF16 KV 另需 85.90 GB", "70B FP8 weights are about 70.6 GB; 32×8K BF16 KV adds 85.90 GB",
      formula("Llama 70B GQA: 2×80×8×128×2 = 327,680 B/token<br>KV at S=8,192, B=32: 327,680×8,192×32 = 85,899,345,920 B = 85.90 GB = 80 GiB<br>Weights with 5% allowance: 70.6×1.05 = 74.13 GB<br>H200 safe payload/GPU: 141×0.85−3 = 116.85 GB<br>Memory floor: ceil((74.13+85.90)/116.85) = 2 GPUs") +
      p("上例只证明理想聚合内存至少 2 卡，不证明 2 卡必达 SLA。BF16 权重将权重项翻倍；FP8 KV 将 KV 项减半；长输出、KV 复制、专家冗余和额外 MTP 要独立计入。GB=10⁹ B，GiB=2³⁰ B；这里故意展示两者。",
        "This only proves an ideal aggregate floor of 2 GPUs, not that 2 meet the SLO. BF16 doubles the weight term; FP8 KV halves the KV term. Long outputs, KV replication, expert redundancy and extra MTP are separate. GB=10⁹ B and GiB=2³⁰ B; both are shown deliberately.")) +
    section("faq", "模型名、存储精度和 GPU 商品名都不是完整配置", "Model names, storage dtype and GPU branding are not full configurations",
      faq("DeepSeek 能用激活 37B 来算加载显存吗？", "Can DeepSeek loaded memory use its 37B active size?", "不能。GPU 常驻全模型的目标下，用 671B 总参数算主权重；加载额外 MTP 再加 14B。CPU/NVMe offload 是另一种部署边界，吞吐必须把搬运和 CPU 计算计入。", "No. For a GPU-resident full model, use 671B main weights and add 14B if MTP is loaded. CPU/NVMe offload is a different deployment boundary; include transfers and CPU compute.") +
      faq("MLA 是否让所有实现的 KV 都自动只占 35 KB/token？", "Does MLA guarantee 35 KB/token in every implementation?", "不保证。61×576×1=35,136 B/token 只在压缩表示且所有分量按 1 B 存储时成立。RoPE、scale、padding、TP 复制、MTP state 与 kernel 支持都会改变真实占用。", "No. 61×576×1=35,136 B/token requires compressed cache and one byte for every component. RoPE dtype, scales, padding, TP replication, MTP state and kernels can change occupancy.") +
      faq("80–85% 水位是 vLLM 当前默认值吗？", "Is an 80–85% watermark vLLM's current default?", "不是。这是可调的架构规划余量，不是引擎默认宣称。运行时测量/内存策略在版本之间变化，必须记录实际 flags、启动 profile、图捕获与稳态峰值。", "No. It is an adjustable architectural planning margin, not an engine default claim. Runtime profiling and memory policies change by version; record actual flags, startup profile, graph capture and steady-state peak.") +
      quiz("sizing", "DeepSeek 的主权重容量应使用哪个参数数？", "Which parameter count should size DeepSeek main weights?", [["37B", "37B"], ["671B（加载 MTP 时再加 14B）", "671B (add 14B when loading MTP)"], ["8 个专家", "8 experts"]], 1, "总参数决定存储，激活参数只是每 Token 计算与访问的线索。", "Total parameters govern storage; active parameters only inform per-token work and access.")) +
    section("checklist", "采购前必须从聚合估算下沉到每个 rank", "Before procurement, move from aggregate estimates to each rank",
      checklist([["核对 checkpoint 参数/geometry、上下文上限与量化 scales", "Verify checkpoint parameters/geometry, context limit and quantization scales"],
        ["检查单 rank 权重、复制 KV、CUDA Graph、通信和峰值激活", "Inspect per-rank weights, replicated KV, CUDA graphs, collectives and peak activations"],
        ["记录实际可见显存字节，不能混用名义 GB 与 GiB", "Record actual visible memory bytes; do not mix nominal GB with GiB"],
        ["用冷/热缓存和混合负载同时验证 TTFT、ITL、Goodput、质量", "Verify TTFT, ITL, goodput and quality on cold/warm caches and mixed traffic"]]));
}

function pdContent() {
  const controls = select("pd", "model", "KV 模型预设", "KV model preset", models, "llama70") +
    field("pd", "context", "待交接输入 Token 数", "Prompt tokens to hand off", 8192, 256, 131072, 256, "range") +
    select("pd", "kvBytes", "KV 存储字节", "KV storage bytes", [["2", "BF16 / FP16 2 B"], ["1", "FP8 1 B (verify kernel)"]], "1") +
    field("pd", "cache", "已在 D 池的前缀比例", "Prefix fraction already in D pool", 0, 0, 1, 0.05) +
    select("pd", "rate", "单端口单向线速 Gb/s", "One-port one-way line rate Gb/s", [["25", "25G"], ["100", "100G"], ["200", "200G"], ["400", "400G"], ["800", "800G"]], "400") +
    field("pd", "efficiency", "有效网络效率（假设）", "Effective network fraction (assumption)", 0.8, 0.05, 1, 0.05) +
    field("pd", "rails", "真正可并行的端口/轨数", "Actually parallel ports/rails", 1, 1, 16) +
    field("pd", "wireCopies", "实际跨网传输倍率（不自动等于D侧复制）", "Actual wire multiplier (not automatically D replication)", 1, 1, 16, 0.1) +
    check("pd", "customKVBytes", "使用已确认的唯一逻辑 KV 字节/Token（默认仍为示例）", "Use confirmed unique logical KV bytes/token (default remains an example)") +
    field("pd", "bytesPerToken", "唯一逻辑 KV B/Token（含真实packing，未乘复制）", "Unique logical KV B/token (packing included, before copies)", 163840, 1, 10000000) +
    field("pd", "setup", "握手/调度固定耗时 ms", "Fixed handshake/scheduling ms", 0.2, 0, 10000, 0.1) +
    field("pd", "overlap", "可证实的重叠时间 ms（默认为教学假设）", "Confirmable overlap ms (default is a teaching assumption)", 10, 0, 10000, 0.1) +
    field("pd", "qps", "KV 新请求 QPS", "New-KV request QPS", 10, 0, 100000, 0.1) +
    field("pd", "base", "基础 ITL ms（模拟）", "Base ITL ms (synthetic)", 20, 0.1, 1000, 0.1) +
    field("pd", "prefillRate", "共池 Prefill TPS（模拟）", "Colocated prefill TPS (synthetic)", 80000, 1, 10000000) +
    field("pd", "chunk", "Chunked Prefill Token 数", "Chunked prefill tokens", 1024, 1, 131072) +
    field("pd", "residual", "PD池残留尾部干扰ms（模拟）", "Residual PD-tail interference ms (synthetic)", 2, 0, 1000, 0.1) +
    check("pd", "hideColocated", "放大只比较Chunked与PD（不把省略值设0）", "Zoom to chunked versus PD (omitted values are not zero)");
  const budgets = field("pdBudget", "outputTokens", "全部生成输出Token（含首字/隐藏推理）", "All generated output tokens (first + hidden reasoning)", 512, 1, 32768) +
    field("pdBudget", "destinationCopies", "D侧引用完整KV状态倍率（手工）", "D referenced complete-state multiplier (manual)", 1, 1, 64, 0.1) +
    field("pdBudget", "prefillPoolTPS", "P池总输入TPS上界（教学值）", "Total P-pool input TPS bound (teaching value)", 200000, 1, 10000000) +
    field("pdBudget", "decodePoolTPS", "D池总输出TPS上界（教学值）", "Total D-pool output TPS bound (teaching value)", 12000, 1, 10000000) +
    field("pdBudget", "decodeSlots", "D池KV驻留请求槽（含交接等待）", "D KV-resident slots (including handoff wait)", 256, 1, 32768) +
    field("pdBudget", "meanTPOT", "D阶段平均间隔 ms（非P99）", "Mean D interval ms (not P99)", 20, 0.1, 1000, 0.1) +
    field("pdBudget", "capacityLimit", "单约束使用比例规划上限", "Planning utilization limit per constraint", 0.7, 0.1, 0.95, 0.05) +
    select("pdBudget", "firstVisible", "首个用户可见Token路径", "First user-visible token path", [["decode", "Visible after D handoff"], ["prefill", "Visible from P before handoff"]], "decode") +
    field("pdBudget", "queueP", "P排队预算 ms", "P queue budget ms", 100, 0, 10000) +
    field("pdBudget", "prefillBudget", "P计算预算/均值假设 ms", "P compute budget/mean assumption ms", 350, 0, 60000) +
    field("pdBudget", "queueD", "D排队预算 ms", "D queue budget ms", 50, 0, 10000) +
    field("pdBudget", "firstBudget", "首输出采样剩余开销 ms", "Remaining first-output sampling budget ms", 20, 0, 10000) +
    field("pdBudget", "delivery", "网关/交付预算 ms", "Gateway/delivery budget ms", 30, 0, 10000) +
    field("pdBudget", "ttftTarget", "工程 TTFT 总预算 ms", "Engineering total TTFT budget ms", 800, 1, 60000) +
    field("pdBudget", "itlTarget", "首字后首次间隔预算 ms", "First-to-second visible gap budget ms", 35, 1, 1000) +
    field("pdBudget", "prefillReplicas", "P池完整模型实例数", "Complete-model replicas in P pool", 1, 1, 256) +
    field("pdBudget", "decodeReplicas", "D池完整模型实例数", "Complete-model replicas in D pool", 1, 1, 256) +
    select("pdBudget", "weightBytes", "主权重存储字节", "Main weight storage bytes", [["2", "BF16 2 B"], ["1", "FP8 1 B"], ["0.5", "INT4 0.5 B"]], "1") +
    field("pdBudget", "weightOverhead", "权重metadata余量比例", "Weight metadata allowance", 0.05, 0, 1, 0.01) +
    check("pdBudget", "mtp", "checkpoint实际额外加载14B MTP", "Checkpoint actually adds14B MTP") +
    check("pdBudget", "compatible", "模型/Tokenizer/KV布局兼容（本例假设确认）", "Model/tokenizer/KV layout compatible (assumed in example)", true) +
    check("pdBudget", "memoryChecked", "每rank与transfer-buffer内存已检查（本例假设）", "Per-rank/transfer-buffer memory checked (example assumption)", true) +
    check("pdBudget", "colocatedPasses", "Chunked混部基线已满足业务SLO", "Chunked colocated baseline already meets business SLO");
  return section("concepts", "Prefill 和 Decode 混部会争资源，不只是争 FLOPs", "Colocated prefill and decode compete for more than FLOPs",
    p("Prefill 一次处理大量输入 Token，矩阵计算往往更密集；低 batch Decode 每步生成一枚 Token，反复读权重和 KV，通常更吃带宽。二者还竞争 kernel 调度、SM、缓存、显存容量与通信。边界并不绝对：短 Prefill 也可能受带宽/启动开销限制，大 batch Decode 也可能算力受限。",
      "Prefill processes many input tokens with denser matrix work; low-batch decode generates one token per step and repeatedly reads weights and KV, often stressing bandwidth. They also compete for kernel scheduling, SMs, caches, capacity and communication. Short prefill can be launch/bandwidth-bound; large-batch decode can be compute-bound.") +
    p("Chunked Prefill 把长输入拆小，并优先调度 Decode，降低单次阻塞，但资源仍在共池。PD 用独立资源池隔离这类干扰，代价是第二份模型驻留、KV 交接、路由与两池配比。不要把 P/D isolation 误译成「系统没有抖动」。",
      "Chunking splits long inputs and prioritizes decode, reducing individual stalls while keeping shared resources. PD isolates this interference with separate pools but adds duplicate model residency, KV handoff, routing and pool-ratio management. Isolation does not mean a jitter-free system.")) +
    section("pd", "1.34 GB KV 在 400G、80% 效率下也要约 33.55 ms", "1.34 GB of KV still needs about 33.55 ms at 400G and 80% efficiency",
      lab("pd", controls) +
      formula("Unique_KV_suffix = KV_bytes_per_token × prompt_tokens × (1−prefix_already_on_D)<br>Wire_payload = Unique_KV_suffix × confirmed_wire_multiplier<br>Effective_BW_Bps = port_Gbps×10⁹/8 × efficiency × usable_parallel_rails<br>Raw_transfer_ms = Wire_payload / Effective_BW_Bps ×1000<br>Exposed_handoff_ms = max(0, Raw_transfer_ms−verified_overlap_ms) + fixed_ms<br>Fabric_load = new_request_QPS × Wire_payload / Effective_BW_Bps") +
      p("默认 Llama 70B FP8 KV：2×80×8×128×1×8192=1,342,177,280 B。400G 单向理论 50 GB/s，80% 效率为 40 GB/s；原始传输 33.554 ms。即便重叠 10 ms，仍暴露 23.754 ms（含 0.2 ms 固定开销）。只有计算与传输确实流水重叠，才允许扣除重叠时间。",
        "Default Llama 70B FP8 KV: 2×80×8×128×8192=1,342,177,280 B. A 400G link is theoretically 50 GB/s one-way, 40 GB/s at 80% efficiency. Raw transfer is 33.554 ms. With 10 ms verified overlap and 0.2 ms fixed overhead, exposed handoff is still 23.754 ms.")) +
    section("pdBudget", "KV线量、D侧复制和首字路径必须分别预算", "Budget KV wire bytes, D replication and first-visible-token path separately",
      p("本沙盘沿用上方的模型、Prompt、KV布局、网络、线量倍率与QPS；修改上方会自动更新这里。P/D吞吐与延迟是独立的教学预算，不能由aggregate TPS直接推出单请求TTFT。",
        "This workbench inherits model, prompt, KV layout, fabric, wire multiplier and QPS from above; changes there update it. Pool throughput and latency are separate teaching budgets; aggregate TPS does not directly determine single-request TTFT.") +
      lab("pdBudget", budgets) +
      formula("μ_P = P_pool_input_TPS / input_tokens<br>μ_D = D_pool_output_TPS / D_generated_tokens<br>μ_slots = D_KV_resident_slots / mean_assumed_D_residence_seconds<br>μ_network = effective_BW / actual_wire_payload; omitted only for zero new wire bytes<br>Pipeline throughput bound = min(μ_P, μ_D, μ_slots, μ_network)") +
      p("D侧KV复制不一定等于跨网复制：例如独立网络只传一份，D侧再经NVLink广播，线量倍率仍可为1而D状态倍率为2。完整D状态还引用已缓存前缀；命中只减少新传输，不让已有KV消失。当前峰值引用状态按Prompt+输出上限预算，共享block/增量分配要另测。",
        "D KV replication need not equal wire replication: one network copy followed by a D-side NVLink broadcast can keep wire multiplier1 while referenced D state has multiplier2. Full state still references cached prefixes; a hit reduces new transfer, not existing KV. Peak referenced state budgets prompt+output, with shared blocks/incremental allocations measured separately.") +
      p("默认P上界200K输入TPS、D上界12K输出TPS、256驻留槽、10 QPS。D输出限制为12000/512=23.4375请求/s，低于P与网络约束；这是候选吞吐上界，不是实测Goodput。若首字等D路径，工程TTFT预算约573.75 ms；若P先发首字，TTFT可约500 ms，但首次后续间隔约93.75 ms，仍可能不满足35 ms流式预算。",
        "Default P bound200K input TPS, D bound12K output TPS,256 resident slots and10 QPS: D output limits12000/512=23.4375 requests/s, below P/network constraints. This is a candidate throughput bound, not measured goodput. First visible output after D gives an engineering TTFT budget about573.75 ms. Early P output can yield about500 ms TTFT yet a93.75 ms first subsequent gap, missing a35 ms streaming budget.") +
      note("这些是工程阶段预算之和，不是把各阶段P99相加。首字来自P还是D必须按真实frontend/connector确认；有的pull路径是在P完成后才传，不支持你假设的layer-wise overlap。模型、tokenizer、dtype、block size、KV layout、TP/CP/PP与内存缓冲都要验证。确认框的默认值也是教学前提，不是本机已检查。",
        "These sum engineering budgets, not stage P99s. Verify where the frontend/connector exposes the first token; some pull paths transfer only after P finishes and cannot support assumed layer-wise overlap. Validate model, tokenizer, dtype, block size, KV layout, TP/CP/PP and memory buffers. Default confirmation checkboxes are teaching premises, not local checks.")) +
    section("boundary", "没有固定的「16 卡以上必须 PD」阈值", "There is no universal 'PD is mandatory above 16 GPUs' threshold",
      table([b("判断维度", "Decision dimension"), b("优先混部+Chunked", "Prefer colocated + chunking"), b("进入 PD 验证", "Evaluate PD")], [
        [b("SLA 干扰证据", "SLO interference"), b("Chunked 已达 TTFT/ITL 双门槛", "Chunking meets both TTFT/ITL gates"), b("trace 证实 Prefill 导致 Decode 尾延迟，调 chunk 仍不达标", "Traces show prefill-driven decode tails despite chunk tuning")],
        [b("KV 交接预算", "KV handoff budget"), b("传输/握手占 TTFT 预算过大", "Transfer/handshake consumes too much TTFT"), b("暴露交接+两池排队+计算仍低于 TTFT 目标", "Exposed handoff + both queues + compute fit TTFT")],
        [b("持续网络负载", "Sustained fabric load"), b("KV 到达量≥有效带宽，或网络已被 EP 占满", "KV arrival≥effective bandwidth, or EP occupies fabric"), b("留有峰值和故障余量并通过真实 payload 压测", "Peak/failure headroom validated with real payloads")],
        [b("池化成本", "Pool cost"), b("小规模/低利用率，双池复制浪费大", "Small/low utilization with expensive duplicated pools"), b("达标 Goodput/成本提升能支付复制、运维与容灾", "Goodput/cost gain pays for duplication, operations and resilience")],
        [b("硬件与网络", "Hardware/fabric"), b("普通 TCP 也可能适合低 QPS/小 KV，先测预算", "TCP may suit low QPS/small KV; test budget"), b("RDMA/NVLink 支持且版本、注册与 topology 已验证", "RDMA/NVLink supported with verified version, registration and topology")]
      ]) +
      note("PD 不是只看 RDMA 有无或 GPU 张数的机械规则。网络负载≥100% 时，本沙盘明确提示不稳定：它不生成排队延迟假数字，也不把图上的 PD 曲线当作达标预测。",
        "PD is not a mechanical rule based on RDMA or GPU count. At fabric load≥100%, the workbench flags instability instead of inventing queue latency or claiming the PD curve predicts compliance.")) +
    section("engines", "Dynamo 是编排层，Mooncake 是 KV 中心生态，vLLM/SGLang 是引擎", "Dynamo orchestrates, Mooncake centers on KV, vLLM/SGLang execute",
      table([b("方案", "Project"), b("主要职责", "Primary role"), b("评估时真正要核对", "What to verify")], [
        ["NVIDIA Dynamo", b("跨引擎路由、PD、缓存与弹性编排", "Cross-engine routing, PD, cache and scaling"), b("后端版本、connector、KV layout、两池容量与故障恢复", "Backend version, connector, KV layout, both pools and recovery")],
        ["Mooncake", b("KV-centric 架构、Transfer Engine 与 Store 生态", "KV-centric architecture, Transfer Engine and Store ecosystem"), b("传输路径、CPU/GPU staging、注册开销、缓存复用与网络", "Transfer path, staging, registration, reuse and fabric")],
        ["vLLM", b("通用推理引擎与 KV connector 接口", "General serving engine and KV connector interfaces"), b("当前 release 的 connector、chunk 默认策略、TP/DP 支持", "Connector, chunk defaults and TP/DP support in the chosen release")],
        ["SGLang", b("推理运行时、前缀复用和 PD 生态", "Serving runtime, prefix reuse and PD ecosystem"), b("选定模型的 MLA/MoE/FP8、缓存路由与 connector 支持", "Model-specific MLA/MoE/FP8, cache routing and connector support")]
      ]) +
      p("比较的是层次与职责，不是无条件速度排行榜。固定模型、输入/输出分布、网络、引擎 commit 和 SLA 后再做 A/B；禁止拼接作者不同机器的吞吐来得出生产排序。",
        "This compares roles, not unconditional speed rankings. Run A/B with a fixed model, token distributions, network, engine commit and SLO. Do not combine author results from different hardware into a production ranking.") +
      table([b("项目", "Project"), b("2026-11-21观测的release（日期UTC）", "Release observed 2026-11-21 (dates UTC)"), b("不能由版本号直接保证", "Not guaranteed by version number")], [
        ["Dynamo", '<a href="https://github.com/ai-dynamo/dynamo/releases/tag/v1.6.2">v1.6.2</a> ·2026-11-05', b("后端、模型和KV transfer协议组合", "Backend/model/KV-transfer combination")],
        ["vLLM", '<a href="https://github.com/vllm-project/vllm/releases/tag/v0.34.0">v0.34.0</a> ·2026-11-16', b("主线文档功能全部已在此release支持", "All main-branch docs features shipped in this release")],
        ["SGLang", '<a href="https://github.com/sgl-project/sglang/releases/tag/v0.5.23">v0.5.23</a> ·2026-11-02', b("所有MLA/MoE/FP8/PD组合均兼容", "Every MLA/MoE/FP8/PD combination is compatible")],
        ["Mooncake", '<a href="https://github.com/kvcache-ai/Mooncake/releases/tag/v0.3.14">v0.3.14</a> ·2026-11-03', b("每个引擎connector均无需额外适配", "Every engine connector needs no adaptation")]
      ]) +
      note("Release表仅为GitHub元数据快照，不是已验证联合栈。索引源码表明某些混合attention recipe要求P/D同TP，某些MTP+PD组合不兼容；NIXL pull/push对PCP/DCP也有不同限制。机内NVLink可用也不等于跨Pod实际使用CUDA IPC，PID namespace等约束仍要查。",
        "Release metadata is not a validated combined stack. Indexed recipes show some hybrid-attention models need matching P/D TP and some MTP+PD combinations conflict; NIXL pull/push also differ in PCP/DCP restrictions. Available NVLink does not prove cross-Pod CUDA IPC is used; inspect PID namespaces and other constraints.")) +
    section("faq", "KV 交接影响 TTFT；隔离效果主要看 Decode 尾延迟", "KV handoff affects TTFT; isolation primarily targets decode tails",
      faq("图中的P99从哪里来？", "Where do the plotted P99 values come from?", "每点生成200个32输出Token请求，每请求31个确定性后续间隔。普通请求近似每20间隔注入一次Prefill干扰，2%请求注入更多突发；共池/Chunked/PD分别注入整段、min(S,chunk)和残留干扰。对200个每请求均值取P99 TPOT，对同批6200个间隔取P99 ITL。首字假设已经在D侧可见，不含启动交接；这是模拟，不预测网络过载。", "Each point generates 200 requests with 32 outputs and 31 deterministic subsequent gaps. Ordinary requests inject about one prefill interference per 20 gaps; 2% have additional bursts. Colocated/chunked/PD inject full, min(S,chunk) or residual interference. P99 TPOT uses 200 request means; P99 ITL uses the same 6200 gaps. First output is assumed already D-visible, excluding startup handoff; this is simulation, not overload prediction.") +
      faq("PD 交接时间能直接加到每个 decode Token 的 ITL 吗？", "Should handoff time be added to every decode token's ITL?", "通常不能。交接主要是请求开始/首字路径的一次成本；Decode 中后续抖动由 D 池排队、网络争用和运行时决定。图把交接时间与尾部模拟分开，避免重复计费。", "Usually no. Handoff is primarily a one-time request-start/first-token cost. Later decode jitter depends on D-pool scheduling, contention and runtime. Handoff and ITL simulation are shown separately.") +
      quiz("pd", "400 Gb/s 单端口的单向理论字节带宽是？", "What is the theoretical one-way byte bandwidth of a 400 Gb/s port?", [["400 GB/s", "400 GB/s"], ["50 GB/s", "50 GB/s"], ["100 GB/s（直接加双向）", "100 GB/s (sum both directions)"]], 1, "400÷8=50 GB/s。单次 KV 单向发送不能直接把双向带宽相加。", "400÷8=50 GB/s. A one-way KV transfer cannot sum both directions.")) +
    section("checklist", "先验证混部基线，再验证 PD 的净收益", "Validate the colocated baseline before PD's net benefit",
      checklist([["固定 chunked-prefill 基线并保留带请求关联的 trace", "Fix a chunked baseline and retain correlated request traces"],
        ["分别测原始传输、已重叠部分与暴露交接时间", "Measure raw transfer, overlapped portion and exposed handoff separately"],
        ["验证 KV dtype/layout、TP 重分片、prefix correctness 与失败回退", "Verify KV dtype/layout, TP reshaping, prefix correctness and failure recovery"],
        ["同时计 P 池、D 池、复制权重、网络、容灾与达标成本", "Count P/D pools, duplicate weights, fabric, resilience and compliant cost"]]));
}

function parallelContent() {
  const controls = select("parallel", "kind", "观察哪一种切分", "Which partition to inspect", [["tp", "TP / Tensor"], ["pp", "PP / Pipeline"], ["dp", "DP / Requests & Attention"], ["ep", "EP / Experts"]], "tp") +
    field("parallel", "ranks", "并行 ranks（示意）", "Parallel ranks (illustration)", 4, 2, 8, 1, "range") +
    field("parallel", "batch", "调度 microbatches", "Scheduled microbatches", 8, 1, 128, 1, "range") +
    field("parallel", "payload", "每通信步骤 payload MB（假设）", "Payload MB/communication step (assumption)", 32, 0.1, 10000, 0.1) +
    field("parallel", "bandwidth", "实测单向等效通信 GB/s（假设）", "Measured one-way equivalent GB/s (assumption)", 100, 0.1, 1000, 0.1) +
    check("parallel", "attentionDP", "DP 图切换为 Attention DP + 共享 EP", "Show attention DP + shared EP instead of replicas");
  const speculation = field("speculative", "draftTokens", "每轮候选 Token 数 γ", "Draft tokens per cycle γ", 4, 1, 16, 1, "range") +
    field("speculative", "acceptance", "每位置平均接受率 α（独立同分布假设）", "Per-position acceptance α (i.i.d. assumption)", 0.8, 0, 1, 0.01) +
    field("speculative", "targetMs", "基线 target 单步 ms", "Baseline target step ms", 20, 0.1, 1000, 0.1) +
    field("speculative", "draftMs", "每候选串行 draft ms", "Serial draft ms/token", 2, 0, 1000, 0.1) +
    field("speculative", "verifyExtraMs", "每候选额外验证 ms（线性假设）", "Extra verification ms/draft (linear assumption)", 1, 0, 1000, 0.1) +
    field("speculative", "overheadMs", "每轮固定提交/调度 ms", "Fixed commit/scheduling ms/cycle", 1, 0, 1000, 0.1);
  const caching = select("prefix", "model", "缓存 geometry 模型", "Cache-geometry model", models, "llama70") +
    select("prefix", "kvBytes", "缓存 KV 精度", "Cached KV precision", [["2", "BF16 2 B"], ["1", "FP8 1 B (verify kernel)"]], "2") +
    field("prefix", "prompt", "完整 Prompt Token", "Full prompt tokens", 8192, 256, 131072, 256, "range") +
    field("prefix", "prefix", "可共享的精确前缀 Token", "Exact shareable prefix tokens", 4096, 0, 131072, 256, "range") +
    field("prefix", "hitRate", "请求缓存命中比例（假设）", "Request cache-hit fraction (assumption)", 0.7, 0, 1, 0.05) +
    field("prefix", "prefillRate", "Prefill TPS（同形状近似）", "Prefill TPS (same-shape approximation)", 20000, 1, 10000000) +
    field("prefix", "lookupMs", "每请求查缓存开销 ms", "Cache lookup ms/request", 2, 0, 1000, 0.1) +
    field("prefix", "entries", "常驻的不同前缀条目数", "Distinct resident prefix entries", 10, 1, 1000000);
  return section("concepts", "并行度不是越大越好：切分粒度决定通信频率", "More ranks are not always better: the partition sets communication frequency",
    table([b("方式 / 切分维度", "Mode / partition"), b("通信原语", "Communication"), b("延迟/吞吐方向（条件性）", "Latency/throughput direction (conditional)"), b("放在哪里", "Placement")], [
      ["TP / weight tensors", "AllReduce / ReduceScatter + AllGather", b("单 GPU 工作/显存减少；低延迟通信可帮助 TTFT/TPOT，过度 TP 反而拖慢；单实例吞吐不一定线性。", "Less work/memory per GPU; fast collectives can help TTFT/TPOT, excessive TP can hurt. Instance throughput is not necessarily linear."), b("优先低延迟高带宽 NVLink 域，跨域必须实测", "Prefer a low-latency NVLink domain; test across domains")],
      ["PP / layers", "SendRecv activations", b("层权重分布到多卡；单请求仍依次过各 stage。microbatch 填流水线提升吞吐，但 bubble/排队可伤延迟。", "Distributes layer weights; one request still visits stages serially. Microbatches fill the pipeline but bubbles/queues can hurt latency."), b("常可跨节点；stage 平衡与链路至关重要", "Often spans nodes; stage balance and links matter")],
      ["DP / independent requests", b("副本路由；推理无训练梯度 AllReduce", "Replica routing; no training gradient AllReduce"), b("独立副本通常提升系统 TPS，不会凭空加快一个请求；也复制权重与缓存。", "Replicas usually increase system TPS, not a single request's speed; weights/cache are duplicated."), b("副本跨节点，靠路由分流与缓存亲和", "Replicas across nodes with routing and cache affinity")],
      ["Attention DP + EP", "Expert dispatch/combine AllToAll", b("Attention 在 DP 请求分片上执行，专家层进入共享 EP 域。并行组正交；不是给每张卡完整复制专家。", "Attention executes on DP request shards while expert layers enter a shared EP domain. Groups are orthogonal; not full expert replication on every GPU."), b("利用 NVLink / RDMA 高效域，计入专家热点和 all-to-all", "Use efficient NVLink/RDMA domains; account for expert skew and all-to-all")],
      ["EP / routed experts", "AllToAll dispatch → compute → combine", b("专家权重分散，降低每 rank 驻留；路由倾斜与跨机 token 交换会限制 TTFT/TPOT 和 Goodput。", "Expert weights are distributed; routing skew and cross-node token exchange constrain TTFT/TPOT and goodput."), b("MoE 才有专家维度；Dense 模型不启用 EP", "MoE has experts; dense models do not use EP")]
    ])) +
    section("parallel", "看张量流向，而不是只记 TP/PP/DP/EP 四个缩写", "Follow the tensor flow, not just four acronyms",
      lab("parallel", controls) +
      note("图解是切分语义，不是实际 NCCL 时间线。通信量、步数、融合和重叠依实现变化；单步 payload/BW 是裸传输估算，不是总 Token 时延。PP 的理想利用率 m/(m+p−1) 只适用于平衡 stage、等长 microbatch 的简化流水线。",
        "The diagram explains partitions, not an actual NCCL timeline. Payload, step counts, fusion and overlap vary. One-step payload/BW is bare transfer, not total token latency. Ideal PP utilization m/(m+p−1) assumes balanced stages and equal microbatches.")) +
    section("acceleration", "Prefix Caching 降 Prefill，投机解码赌的是验证摊薄", "Prefix caching reduces prefill; speculation amortizes verification",
      table([b("手段", "Technique"), "TTFT", "TPOT / ITL", "System TPS", b("代价 / 验证门槛", "Cost / validation gate")], [
        ["TP", b("可能改善", "Can improve"), b("可能改善或恶化", "Can improve or hurt"), b("非线性", "Nonlinear"), b("集合通信、碎片/复制与并行合法性", "Collectives, fragmentation/replication and valid partition")],
        ["PP", b("可能增大", "Can increase"), b("依 stage/流水", "Stage/pipeline-dependent"), b("充分填充时提升", "Improves when filled"), b("Bubble、慢 stage 与串行请求路径", "Bubbles, slow stages and serial request path")],
        ["DP", b("减少排队，不降单次计算", "Less queueing, not less compute"), b("通常不改善单流", "Usually not a single-stream gain"), b("副本扩容提升", "Replica scale-out gain"), b("权重复制、缓存路由与故障冗余", "Weight duplication, cache routing and failover")],
        ["EP / Attention DP", b("可能改善/恶化", "Can improve/hurt"), b("依路由和网络", "Routing/fabric-dependent"), b("MoE 高 batch 可受益", "MoE large batch can benefit"), b("AllToAll、专家热点、容量 padding", "All-to-all, expert skew and capacity padding")],
        ["Speculative decoding / MTP", b("通常不是主要目标", "Usually not the main target"), b("接受率高且验证便宜时改善", "Helps with high acceptance and cheap verification"), b("高 batch 饱和时收益递减", "Diminishing at saturated large batch"), b("Draft/MTP 额外显存与验证；质量/分布正确性", "Draft/MTP memory and verification; quality/distribution correctness")],
        ["Prefix Caching", b("缓存命中时改善", "Improves on hits"), b("通常不直接加速 Decode", "Usually not direct decode acceleration"), b("节省重复 Prefill", "Saves repeated prefill"), b("命中率、KV 内存、租户隔离、精确前缀", "Hit rate, KV memory, tenant isolation, exact prefix")],
        ["FP8 / INT4 storage", b("依 kernel / compute dtype", "Kernel/compute-dtype-dependent"), b("带宽压力可能下降", "Can reduce bandwidth pressure"), b("依质量与内核", "Quality/kernel-dependent"), b("尺度开销、反量化、硬件支持、质量回归", "Scale metadata, dequantization, hardware support, quality regression")],
        ["Chunked Prefill", b("可能增加长输入等待", "May increase long-prompt wait"), b("缓解共池尾部干扰", "Reduces colocated tail interference"), b("依 token budget", "Token-budget-dependent"), b("更小 chunk 不是无条件最优", "Smaller chunks are not always optimal")],
        ["PD", b("增加交接但隔离排队", "Adds handoff but isolates queues"), b("隔离 Prefill 干扰", "Isolates prefill interference"), b("看达标吞吐净收益", "Evaluate net goodput"), b("双池驻留、网络、路由、故障治理", "Duplicate pools, fabric, routing and recovery")]
      ], b("条件性收益/代价热力矩阵：文字是语义，颜色不是虚构百分比", "Conditional benefit/cost heatmap: text carries meaning; colors are not invented percentages")) +
      p("不要声称 FP8 必然 2× 或投机必然无损 3×。正确投机采样可保持目标分布，但 greedy、温度、约束解码、draft 接受率和实现细节都要核对。前缀缓存只命中 tokenizer 后的相同前缀，不等于语义缓存。",
        "Do not promise FP8=2× or speculation=lossless 3×. Correct speculative sampling can preserve the target distribution, but greedy mode, temperature, constrained decoding, acceptance and implementation still need validation. Prefix reuse requires the same tokenized prefix; it is not semantic caching.")) +
    section("speculative", "接受率 80% 不等于无条件 5 Token/轮", "80% acceptance does not imply five tokens every cycle",
      lab("speculative", speculation) +
      formula("E[committed_tokens] = 1 + α + α² + … + α^γ<br>Verify_ms = target_step_ms + γ×extra_verify_ms<br>Cycle_ms = γ×draft_ms + Verify_ms + fixed_overhead_ms<br>Conditional speedup = target_step_ms×E[tokens] / Cycle_ms") +
      p("ICML 2023 原始分析在接受率独立同分布的简化假设下得到几何和；最后一枚是拒绝后的修正 Token 或全部接受后的额外 Token。α=0.8、γ=4，E[N]=3.3616，而不是固定 5。默认每轮成本 4×2+(20+4×1)+1=33 ms，条件性平均服务时间 9.82 ms/Token、约 2.04×，不是硬件实测。",
        "The ICML 2023 analysis gives the geometric sum under an i.i.d. acceptance assumption. The last token is a correction after rejection or a bonus after all drafts pass. At α=0.8, γ=4, E[N]=3.3616, not a fixed five. Default cycle cost is 4×2+(20+4×1)+1=33 ms: conditional mean service 9.82 ms/token and about 2.04×, not a hardware measurement.") +
      note("多 Token 一起提交会形成突发流式：平均每 Token 服务时间不是 P99 ITL。真实验证成本随 γ、batch、上下文与 kernel 改变，MTP/EAGLE 也不一定逐 Token 串行 draft；必须换成实测 cycle breakdown 并验证采样正确性。",
        "Multi-token commits create bursty streams: mean service/token is not P99 ITL. Verification changes with γ, batch, context and kernels, and MTP/EAGLE need not draft serially. Replace costs with measured cycle breakdown and validate sampling correctness.")) +
    section("prefix", "70% 请求命中半段前缀，只避免 35% 输入计算", "70% of requests hitting half the prefix avoids only 35% of input work",
      lab("prefix", caching) +
      formula("E[avoided_input_tokens] = request_hit_rate × shared_prefix_tokens<br>E[prefill_ms] = hit_rate×hit_compute_ms + (1−hit_rate)×miss_compute_ms + lookup_ms<br>Resident_prefix_GB = distinct_entries×prefix_tokens×KV_bytes_per_token / 10⁹") +
      p("默认 8192 输入、4096 精确共享前缀、70% 请求命中，平均避免 2867.2 Token，仍处理 5324.8 Token；20K Prefill TPS 和 2 ms 查找下，409.6 ms 基线变成 268.24 ms 平均 Prefill。10 个独立 Llama 70B BF16 前缀条目需约 13.42 GB KV。命中率低或查找很贵时，净收益可为负。",
        "Default 8192 input, 4096 exact-prefix tokens and 70% request hits avoid 2867.2 tokens on average, leaving 5324.8. At 20K prefill TPS with 2 ms lookup, baseline 409.6 ms becomes mean 268.24 ms. Ten distinct Llama 70B BF16 entries use about 13.42 GB KV. Low hits or expensive lookup can produce a negative net benefit.") +
      note("此线性 Prefill 近似不计长上下文 attention 非线性。不同前缀条目按独立存储估算；共享 block、驱逐、层级 cache、跨副本复制、kernel dtype 与租户隔离要另外核对，不能把缓存当免费显存。",
        "This linear prefill approximation omits long-context attention nonlinearity. Entries are budgeted as independent storage; verify shared blocks, eviction, tiered cache, replica duplication, kernel dtype and tenant isolation. Cache is not free memory.")) +
    section("placement", "把高频同步留在快域，但别把域边界写成绝对禁令", "Keep frequent synchronization in fast domains without making absolute bans",
      diagram("domains", "并行组与快慢通信域", "Parallel groups and communication domains",
        `<rect x="20" y="20" width="330" height="180" rx="12" class="surface"/><rect x="450" y="20" width="330" height="180" rx="12" class="surface"/><text x="40" y="48">NVLink / NVSwitch domain A</text><text x="470" y="48">NVLink / NVSwitch domain B</text>` +
        node(45, 80, 130, "TP group", "frequent reduce", "primary") + node(195, 80, 130, "EP shards", "dispatch/combine") +
        node(475, 80, 130, "TP group", "local tensor") + node(625, 80, 130, "EP shards", "shared experts") +
        arrow(355, 114, 445, 114) + `<text x="400" y="84" text-anchor="middle">RDMA</text><text x="400" y="235" text-anchor="middle">PP stages / DP replicas / EP domain crossing: profile, don't assume</text>`,
        "优先在 NVLink 域内做 TP。跨域 PP/DP 常见，跨域 EP 需高质量 all-to-all；超节点/MNNVL 可扩大快域，不等于节点名就是域边界。",
        "Prefer TP within NVLink. PP/DP often span domains; EP needs efficient all-to-all. Supernodes/MNNVL expand the fast domain; a server hostname is not the domain definition.")) +
    section("faq", "DP Attention 不是训练里的梯度数据并行", "Attention DP is not training gradient data parallelism",
      faq("EP=16 就能把每用户 Decode 自动快 16 倍吗？", "Does EP=16 automatically make a user's decode 16× faster?", "不能。EP 分散专家权重，但一个 Token 仍路由少量专家；批内路由、专家负载、all-to-all、shared layers 与 attention 都影响速度。增加 EP 也可能引入更慢网络。", "No. EP distributes experts while each token uses a subset. Batch routing, expert balance, all-to-all, shared layers and attention govern speed. Larger EP can introduce slower fabric.") +
      faq("如何选 DP、TP、PP 的乘积？", "How should DP, TP and PP group sizes be chosen?", "先按合法权重/attention 切分满足每实例内存，再选 fast-domain TP 与 stage-balanced PP，最后用独立副本承接 QPS。Attention DP 和 EP 属于不同并行组，不能不看引擎语义就直接相乘。", "First satisfy per-instance memory with a valid weight/attention partition. Choose fast-domain TP and balanced PP, then add replicas for QPS. Attention DP and EP use distinct groups; do not multiply sizes without engine semantics.") +
      quiz("parallel", "Dense 模型需要 Expert Parallelism 吗？", "Does a dense model need expert parallelism?", [["需要，所有模型都有专家", "Yes, every model has experts"], ["不需要，EP 的专家维度属于 MoE", "No; EP's expert dimension belongs to MoE"]], 1, "Dense 可以 TP/PP/DP，但没有 routed expert dispatch/combine 这一维度。", "Dense models can use TP/PP/DP, but have no routed expert dispatch/combine dimension.")) +
    section("checklist", "收益必须同时通过质量、内存、通信和 SLA 四道门", "Gains must pass quality, memory, communication and SLO gates",
      checklist([["记录完整 parallel group 拓扑，而不是只写总卡数", "Record full parallel-group topology, not only GPU count"],
        ["先验证模型输出质量和采样分布，再谈量化/投机加速", "Verify output quality/sampling before quantization/speculative gains"],
        ["采集每 layer / rank 的通信、专家热点和非重叠时间", "Capture per-layer/rank communication, expert skew and non-overlapped time"],
        ["冷热 prefix 分开测，避免缓存命中掩盖真实容量", "Test cold/warm prefixes separately so cache hits do not hide capacity"]]));
}

function networkContent() {
  const controls = select("network", "operation", "NCCL 通信操作", "NCCL operation", [["allreduce", "AllReduce"], ["allgather", "AllGather"], ["reducescatter", "ReduceScatter"], ["alltoall", "AllToAll"], ["broadcast", "Broadcast"], ["reduce", "Reduce"], ["sendrecv", "Send/Recv"]], "allreduce") +
    select("network", "sizeMode", "消息尺寸口径", "Message-size scope", [["reported", "NCCL-reported total S"], ["api-count", "API count × dtype bytes (operation-specific)"]], "reported") +
    field("network", "ranks", "ranks 数 n", "Rank count n", 8, 2, 1024) +
    field("network", "sizeMB", "所选口径MB（报告S或API count字节）", "MB under selected scope (reported S or API-count bytes)", 1000, 0.001, 1e7, 0.001) +
    field("network", "timeMs", "实测平均操作时间 ms（示例）", "Measured mean operation time ms (example)", 20, 0.001, 1e6, 0.001) +
    select("network", "rate", "单端口 Gb/s", "One-port Gb/s", [["100", "100G"], ["200", "200G"], ["400", "400G"], ["800", "800G"]], "400") +
    field("network", "rails", "端口/轨数（假设同时满载）", "Ports/rails (assume parallel)", 1, 1, 16) +
    field("network", "efficiency", "payload 带宽效率", "Payload bandwidth efficiency", 0.8, 0.05, 1, 0.05);
  const congestion = field("congestion", "queue", "当前队列占缓冲百分比（示意）", "Queue occupancy % (illustration)", 55, 0, 100, 1, "range") +
    field("congestion", "ecn", "ECN 标记起点 %", "ECN marking start %", 40, 1, 95) +
    field("congestion", "pfc", "PFC Xoff 起点 %", "PFC Xoff threshold %", 80, 2, 99) +
    field("congestion", "xon", "PFC恢复Xon %", "PFC resume Xon %", 60, 0, 98) +
    field("congestion", "bufferMB", "教学队列总缓冲MB（非厂商profile）", "Teaching total queue buffer MB (not vendor profile)", 16, 0.1, 1000, 0.1) +
    field("congestion", "rate", "链路线速Gb/s", "Link line rate Gb/s", 400, 1, 3200) +
    field("congestion", "reactionUs", "暂停生效反应预算µs（示例）", "Pause reaction budget µs (example)", 10, 0, 1000, 0.1) +
    field("congestion", "frameBytes", "最大帧字节预算", "Maximum-frame byte budget", 9000, 64, 1000000) +
    field("congestion", "guardMB", "额外pipeline/cable余量MB", "Extra pipeline/cable guard MB", 0.05, 0, 100, 0.01);
  const pcieGen = [3, 4, 5].map((n) => [String(n), `PCIe Gen${n} (NRZ 128b/130b)`]);
  const lanes = [1, 2, 4, 8, 16].map((n) => [String(n), `x${n}`]);
  const bandwidths = select("networkBudget", "gpu", "仅用于NVLink规格对照的GPU", "GPU for NVLink specification comparison only", gpus, "h100") +
    field("networkBudget", "gpuEndpoints", "参与条带的GPU PCIe端点数", "GPU PCIe endpoints participating in striping", 8, 1, 64) +
    field("networkBudget", "nicPorts", "活动网络端口数", "Active network-port count", 8, 1, 64) +
    field("networkBudget", "nicCards", "活动NIC PCIe卡数（≤端口）", "Active NIC PCIe cards (≤ports)", 8, 1, 64) +
    select("networkBudget", "rate", "每端口Gb/s", "Gb/s per port", [["100", "100G"], ["200", "200G"], ["400", "400G"], ["800", "800G"]], "400") +
    select("networkBudget", "gpuGen", "实测/假设GPU端协商PCIe代际", "Observed/assumed GPU negotiated PCIe generation", pcieGen, "5") +
    select("networkBudget", "gpuLanes", "GPU端协商lanes", "GPU negotiated lanes", lanes, "16") +
    select("networkBudget", "nicGen", "NIC卡协商PCIe代际", "NIC-card negotiated PCIe generation", pcieGen, "5") +
    select("networkBudget", "nicLanes", "NIC卡协商lanes", "NIC-card negotiated lanes", lanes, "16") +
    field("networkBudget", "networkEfficiency", "网络payload效率（示例）", "Network payload efficiency (example)", 0.8, 0.05, 1, 0.05) +
    field("networkBudget", "pcieEfficiency", "编码后PCIe有效payload比例", "PCIe payload fraction after line encoding", 0.85, 0.05, 1, 0.05) +
    check("networkBudget", "crossesShared", "这条数据路径经过所填共享链路（示例设是）", "Path crosses the entered shared link (example assumes yes)", true) +
    field("networkBudget", "sharedGBps", "该共享路径单向payload GB/s", "Shared-path one-way payload GB/s", 300, 0.1, 10000, 0.1) +
    field("networkBudget", "fabricGBps", "分给此节点/任务的fabric单向GB/s", "Fabric one-way GB/s allocated to this node/task", 240, 0.1, 10000, 0.1) +
    field("networkBudget", "payloadMB", "全条带payload MB（十进制）", "Fully striped payload MB (decimal)", 1024, 0.1, 10000000, 0.1);
  const railControls = select("rail", "mapping", "教学GPU→rail映射", "Teaching GPU→rail mapping", [["balanced", "Balanced affinity / round-robin"], ["skewed", "All flows collide on first rail"]], "balanced") +
    field("rail", "gpus", "参与发送的GPU数", "Sending GPU count", 8, 1, 16) +
    field("rail", "railCount", "独立rail/端口数", "Independent rails/ports", 4, 1, 16) +
    field("rail", "payloadGB", "每GPU outgoing payload GB", "Outgoing payload GB/GPU", 1, 0.001, 100, 0.001) +
    field("rail", "rate", "每rail单向Gb/s", "One-way Gb/s per rail", 400, 1, 3200) +
    field("rail", "efficiency", "每rail payload效率", "Per-rail payload efficiency", 0.8, 0.05, 1, 0.05) +
    field("rail", "fabricGBps", "共享fabric payload预算GB/s", "Shared fabric payload budget GB/s", 300, 0.1, 10000, 0.1) +
    field("rail", "hotGpu", "热点GPU索引（<GPU数）", "Hot GPU index (<GPU count)", 0, 0, 15) +
    field("rail", "hotMultiplier", "此GPU流量倍率", "Payload multiplier for that GPU", 1, 1, 8, 0.1) +
    check("rail", "failure", "注入一条rail失效", "Inject one failed rail") +
    field("rail", "failedRail", "失效rail索引（<rail数）", "Failed rail index (<rail count)", 0, 0, 15) +
    check("rail", "reroute", "允许玩具模型重映射（非真实自动恢复）", "Allow toy remapping (not actual automatic recovery)", true);
  return section("concepts", "RDMA 是内存访问能力，RoCEv2 是承载方式，NVLink 是另一类互连", "RDMA is a memory capability, RoCEv2 a transport, NVLink another interconnect",
    table([b("层次", "Layer"), b("含义", "Meaning"), b("单位 / 比较边界", "Unit / comparison boundary")], [
      ["RDMA", b("NIC 直接读写注册内存，降低数据面 CPU 拷贝/参与", "NIC reads/writes registered memory, reducing data-plane CPU copies/work"), b("不是线缆标准，也不自动等于 GPU direct", "Not a cable standard and not automatically GPU-direct")],
      ["GPUDirect RDMA", b("NIC↔GPU 内存 DMA，绕过 CPU DRAM staging", "NIC↔GPU-memory DMA bypasses CPU-DRAM staging"), b("CPU 仍做控制、注册、调度；PCIe topology/驱动必须支持", "CPU still controls, registers and schedules; topology/drivers must support it")],
      ["RoCEv2", b("RDMA over UDP/IP/Ethernet，可路由", "RDMA over UDP/IP/Ethernet, routable"), "400/800 Gb/s per port → 50/100 GB/s one-way before overhead"],
      ["InfiniBand NDR / XDR", b("专用 RDMA fabric、原生链路流控与路由体系", "Dedicated RDMA fabric with native link flow control and routing"), "NDR 400 Gb/s; XDR 800 Gb/s (check lane/port/cable mode)"],
      ["NVLink 4 / 5", b("GPU Scale-Up 域，总聚合双向营销口径", "GPU scale-up domain; advertised aggregate bidirectional scope"), "900 / 1800 GB/s per GPU aggregate ≠ one port or one flow"],
      ["NVSwitch", b("扩展 NVLink GPU 通信域的交换网络", "Switch fabric expanding the NVLink GPU domain"), b("支持快域大于一台服务器；总带宽不能视作每请求独享", "Fast domain can exceed one server; aggregate is not dedicated per request")]
    ])) +
    section("paths", "GPUDirect 绕过主机 staging，不绕过注册和拓扑约束", "GPUDirect bypasses host staging, not registration or topology constraints",
      diagram("gpu-path", "主机 staging 与 GPU Direct 对比", "Host staging versus GPU-direct path",
        `<text x="25" y="30">Host-staged path: extra DRAM copies</text>` + node(25, 48, 155, "GPU A", "HBM") + node(210, 48, 155, "Host DRAM", "CPU control") + node(395, 48, 155, "NIC / fabric", "DMA") + node(585, 48, 190, "Host → GPU B", "staging again") +
        arrow(185, 80, 205, 80) + arrow(370, 80, 390, 80) + arrow(555, 80, 580, 80) +
        `<text x="25" y="155">GPUDirect RDMA: CPU controls, payload bypasses host DRAM</text>` +
        node(25, 178, 185, "GPU A memory", "registered region", "primary") + node(305, 178, 190, "NIC / RDMA fabric", "peer DMA") + node(590, 178, 185, "GPU B memory", "registered region", "primary") +
        arrow(215, 212, 300, 212) + arrow(500, 212, 585, 212),
        "检查 GPU/NIC PCIe 亲和、IOMMU/ACS、peer-memory 或 DMA-BUF 支持、驱动与内存注册。零 host-copy 不代表零开销。",
        "Check GPU/NIC PCIe affinity, IOMMU/ACS, peer-memory or DMA-BUF support, drivers and registration. No host copy does not mean no overhead.", 285)) +
    section("scale-domains", "Scale-Up 看 NVLink 域，Scale-Out 看网络端口，不只看主机名", "Scale-up follows NVLink domains; scale-out follows fabric, not just hostnames",
      diagram("scale-domains-figure", "NVSwitch 快域与 RDMA 网络域", "NVSwitch fast domain and RDMA network",
        `<rect x="15" y="25" width="330" height="220" rx="12" class="surface"/><rect x="455" y="25" width="330" height="220" rx="12" class="surface"/><text x="35" y="53">Scale-Up / domain A</text><text x="475" y="53">Scale-Up / domain B</text>` +
        node(35, 75, 120, "GPU 0", "NVLink", "primary") + node(190, 75, 130, "GPU 1", "NVLink", "primary") +
        node(475, 75, 120, "GPU 2", "NVLink", "primary") + node(630, 75, 130, "GPU 3", "NVLink", "primary") +
        node(90, 165, 180, "NVSwitch", "GPU fabric") + node(530, 165, 180, "NVSwitch", "GPU fabric") +
        `<path d="M 95 145 L 160 162 M 255 145 L 200 162 M 535 145 L 600 162 M 695 145 L 640 162" class="line secondary"/>` +
        arrow(350, 125, 450, 125) + `<text x="400" y="95" text-anchor="middle">NIC / RDMA</text><text x="400" y="155" text-anchor="middle">400 / 800 Gb/s</text><text x="400" y="280" text-anchor="middle">NVLink 4/5: 900/1800 GB/s per-GPU aggregate; NIC: 50/100 GB/s one-way</text>`,
        "示意中的快域可以是单服务器或跨服务器超节点；速率为规格口径，不表示每对 GPU 或每个请求独占。",
        "A fast domain may be one server or a multi-server supernode. Rates are specification scopes, not dedicated bandwidth for every GPU pair or request.", 315)) +
    section("networkBudget", "800G端口不等于100 GB/s GPU有效payload", "An800G port does not guarantee100 GB/s GPU payload",
      lab("networkBudget", bandwidths) +
      formula("PCIe Gen3/4/5 one-way encoded GB/s = GT/s_per_lane×lanes×(128/130)/8<br>Network payload = port_Gb/s/8×payload_efficiency<br>Node path bound = min(active_ports×port_payload, GPU_endpoint_PCIe, NIC_card_PCIe,<br>                      crossed_shared_links, allocated_fabric)<br>Striped transfer lower bound ms = payload_MB / path_GBps") +
      p("PCIe Gen5 x16编码后单向约63.02 GB/s；再按本例85%payload比例为53.56。即使800G端口线速100 GB/s、网络效率80%为80，单PCIe端点仍可能先卡住。双端口NIC也可能共用一条PCIe链路，不能把“端口数”当“NIC卡数”。",
        "PCIe Gen5 x16 gives about 63.02 GB/s one-way after coding, or 53.56 at the example 85% payload fraction. An 800G port is 100 GB/s line rate and 80 at 80% efficiency, yet one PCIe endpoint can bottleneck first. Dual-port NICs may share a PCIe link: port count is not NIC-card count.") +
      p("默认8个GPU端点、8张单端口400G NIC，网络payload320 GB/s；共享路径300、分配fabric240，因此上界240 GB/s。1024 MB完全条带传输下界约4.27 ms；若只有1个GPU端点参与，上界可降至53.56，不能把8张NIC带宽全给一张GPU。",
        "Default eight GPU endpoints and eight single-port 400G NICs provide 320 GB/s network payload, with shared-path 300 and allocated-fabric 240: bound 240 GB/s. A fully striped 1024 MB transfer has about 4.27 ms lower bound. With only one GPU endpoint, it can fall to 53.56; eight NICs do not bypass one GPU endpoint.") +
      note("这是单向路径预算，不是实际吞吐或P99；输入需要取两端实际链路与分配的较小值。GPU选择仅用于NVLink规格对照，PCIe代际/lanes独立填真实协商值，不按型号猜。只有真正经过共享链路才纳入约束；PCIe6/7的PAM4/FLIT不适用此128b/130b公式。各沙盘输入独立，比较前对齐。",
        "This is a one-way path budget, not measured throughput/P99. Use smaller actual endpoint links/allocations. GPU selection only supplies NVLink reference; enter actual negotiated PCIe generation/lanes independently, not from branding. Include shared links only if crossed. PCIe6/7 PAM4/FLIT does not use this128b/130b formula. Align independent workbench inputs before comparing.")) +
    section("network", "8 ranks 的 AllReduce：busbw=1.75×algbw，不是压缩收益", "At 8 ranks, AllReduce busbw=1.75×algbw, not compression",
      lab("network", controls) +
      formula("algbw [GB/s] = effective_total_S_MB / time_ms<br>API-count mode: AllGather(sendcount), ReduceScatter(recvcount), fixed AllToAll(per-peer count)<br>               need total S = count×dtype_bytes×n; AllReduce/Broadcast/Reduce/SendRecv do not multiply n<br>AllReduce busbw = algbw×2(n−1)/n<br>AllGather / ReduceScatter / fixed AllToAll busbw = algbw×(n−1)/n") +
      p("默认消息 1,000 MB、20 ms：algbw=50 GB/s；8 ranks AllReduce busbw=87.5 GB/s。busbw 是标准化通信指标，不是「网络凭空超出线速」。AllGather/ReduceScatter 的 S 要用总数组大小，不是每 rank 的单块。",
        "Default 1,000 MB in 20 ms: algbw=50 GB/s; 8-rank AllReduce busbw=87.5 GB/s. Bus bandwidth is a normalized collective metric, not magical wire speed. For AllGather/ReduceScatter, S is the total array, not one rank's shard.") +
      note("单端口网络效率预算不能直接套到所有 GPU 集合通信：操作可能走 NVLink、多个 NIC、PCIe 或混合路径。若指标超过选定预算，先核对路径、方向、总消息定义和轨数，而不是自动宣告测量错误。",
        "One-port network budgets do not apply automatically to every GPU collective; paths may use NVLink, multiple NICs, PCIe or a mix. If a metric exceeds the selected budget, check path, direction, size definition and rails before declaring an error.") +
      note("API count指本操作实际使用的count×dtype字节，不是命令行请求的名义大小；NCCL-tests可按对齐取整。AllToAllv的可变分发需看最大send/recv计数，本转换器只覆盖固定AllToAll，不能拿平均专家流量推busbw。",
        "API count means actual operation count×dtype bytes, not nominal command-line requested size; NCCL-tests can round for alignment. Variable AllToAllv uses maximum send/recv counts; this converter covers fixed AllToAll and cannot infer busbw from mean expert traffic.")) +
    section("congestion", "ECN 是提前减速，PFC 是逐跳急刹，DCQCN 是端点反馈控制", "ECN signals early, PFC brakes a hop, DCQCN controls the sender",
      lab("congestion", congestion) +
      p("传统 lossless RoCE 基线：交换机队列到 ECN 区间先标记 CE；接收端 RNIC 产生 CNP；发送端 RNIC 的 DCQCN 降速，再逐步恢复。如果队列继续逼近溢出，PFC 按 priority 对上游链路发 pause；低于 Xon 再恢复。标记/减速不是丢包；PFC 不是端到端拥塞控制。",
        "Classic lossless-RoCE baseline: the switch marks CE as queues enter the ECN range; the receiver RNIC emits CNP; sender RNIC DCQCN reduces rate then recovers. If queues keep nearing overflow, PFC pauses an upstream priority on a hop, resuming below Xon. Marking/rate reduction is not packet loss; PFC is not end-to-end congestion control.") +
      note("阈值滑块是概念图，不是厂商配置。ECN/PFC/DCQCN 不是所有现代以太网 fabric 的唯一有效组合；替代拥塞控制、receiver credits 与可承受丢包的实现要按协议和负载验证。避免 PFC 死锁、head-of-line blocking，并启用可验证的 watchdog 与遥测。",
        "Threshold sliders illustrate concepts, not vendor configuration. ECN/PFC/DCQCN is not the only valid modern fabric design; alternative congestion control, receiver credits and loss-tolerant implementations need protocol/workload validation. Address PFC deadlock/HOL blocking with verified watchdogs and telemetry.") +
      formula("Engineering headroom floor ≈ link_bytes_per_second×pause_reaction_seconds<br>                             + max_frame_bytes + pipeline/cable_guard<br>Available headroom in this toy pool = buffer_MB×(1−Xoff_percent/100)") +
      p("400G链路、10µs暂停反应预算会继续送约0.5 MB；加9KB帧与50KB余量，约需0.559 MB。16MB教学队列在80% Xoff后保留3.2MB。把反应预算增至100µs或换800G，所需headroom会急增；不能只复制“80%暂停”的百分比。厂商ASIC/line-card阈值和共享buffer规则要单独查。",
        "A400G link can send another0.5 MB over a10µs pause-response budget; with9KB frame and50KB guard, about0.559 MB is needed. A16MB teaching queue keeps3.2MB after80% Xoff. Increase response to100µs or use800G and headroom rises sharply: copying an'80% pause' threshold is insufficient. Check vendor ASIC/line-card/shared-buffer rules separately.")) +
    section("rails", "多轨拓扑要匹配 GPU↔NIC 亲和与集合通信映射", "Rail topology must match GPU↔NIC affinity and collective mapping",
      diagram("rails-topology", "两节点双轨优化示意", "Two-node, two-rail example",
        node(20, 20, 200, "Node A / GPU0 NIC0", "rail 0") + node(20, 145, 200, "Node A / GPU1 NIC1", "rail 1") +
        node(300, 20, 200, "Leaf / rail 0", "same-index affinity", "primary") + node(300, 145, 200, "Leaf / rail 1", "independent failure", "primary") +
        node(580, 20, 200, "Node B / GPU0 NIC0", "rail 0") + node(580, 145, 200, "Node B / GPU1 NIC1", "rail 1") +
        arrow(225, 54, 295, 54) + arrow(505, 54, 575, 54) + arrow(225, 179, 295, 179) + arrow(505, 179, 575, 179) +
        `<path d="M 400 90 L 400 143" class="grid-line"/><text x="420" y="122">spine / cross-rail routing</text>`,
        "同索引 GPU/NIC 可形成对应轨；跨轨通信仍需 spine/路由。可用带宽受 PCIe、亲和、oversubscription、流分配和故障轨约束，不能简单 N×线速保证。",
        "Same-index GPU/NIC pairs can form rails; cross-rail traffic still needs spine/routing. PCIe, affinity, oversubscription, flow mapping and failed rails limit usable bandwidth; N×line rate is not guaranteed.") +
      lab("rail", railControls) +
      formula("Per-rail time floor = assigned_payload_GB / effective_rail_GBps<br>Fabric time floor = total_payload_GB / allocated_fabric_GBps<br>Completion floor = max(per-rail floors, fabric floor), only if every assigned flow has a live path") +
      p("8个GPU各发1GB、4条400G rail、80%效率：均衡映射每rail2GB，下界50ms；全挤到一rail变成200ms。失去一rail后若玩具重映射到3条存活rail，负载3/3/2GB，下界75ms而不是自动“3×40GB/s”满载。若不重映射，完整集合通信无法完成，必须显示阻塞而不是成功吞吐。",
        "Eight GPUs send 1 GB each over four 400G rails at 80% efficiency: balanced 2 GB/rail gives a 50 ms floor; one-rail collision gives 200 ms. Toy remapping after one failure yields 3/3/2 GB and 75 ms, not automatic full 3×40 GB/s. Without remapping, the whole collective cannot complete: report blocked flows, not successful throughput.") +
      note("这是出站payload映射教学，不模拟NCCL图搜索、ECMP/DLB、SHARP、对端拥塞或故障恢复耗时。热点GPU、PCIe亲和与接收端也可改变结果。RA的rail-aligned拓扑是物理基础，不保证每个任务自动充分利用；真实all-to-all和KV共载需测。",
        "This outgoing-payload mapping exercise does not simulate NCCL graph search, ECMP/DLB, SHARP, receiver congestion or recovery time. Hot GPUs, PCIe affinity and receivers change outcomes. Rail-aligned RA topology is a physical foundation, not automatic full utilization; test actual all-to-all plus KV traffic.")) +
    section("transport-runtime", "同节点有NVLink，隔离Pod的KV仍可能走TCP", "A node can have NVLink while isolated-Pod KV still falls back to TCP",
      p("Dynamo索引文档说明，UCX CUDA IPC路径需进程/PID namespace与peer GPU可见性等条件；UCX1.23对无fabric handles的跨PID namespace IPC有约束，普通HGX H100也不能默认有IMEX。默认单GPU Pod可能不具备peer可见性，因而即使同节点也不走NVLink。",
        "Indexed Dynamo docs describe UCX CUDA-IPC requirements including PID namespace and peer-GPU visibility. UCX1.23 constrains cross-PID IPC without fabric handles; ordinary HGX H100 must not be assumed to have IMEX. Default single-GPU pods can lack peer visibility, so being colocated does not prove NVLink transport.") +
      p("这时请求RDMA设备可让路径转用RDMA，否则可能退到TCP；NIXL不一定因此报错，UCX_TLS也不能创造缺失的权限/可见性。要检查真实日志、device与payload测试。不要把privileged/hostPID当默认修复：多租户隔离、安全批准与合法RDMA/fabric配置必须一起权衡。",
        "Requested RDMA devices can permit an RDMA path; otherwise TCP fallback may occur without a NIXL error. UCX_TLS cannot create missing permissions/visibility. Inspect actual logs/devices/payload tests. Do not make privileged/hostPID a default fix: weigh tenant isolation, approved access and valid RDMA/fabric configuration together.")) +
    section("faq", "看 busbw 之前先看消息口径和路径", "Inspect message scope and path before busbw",
      faq("RDMA 是否完全不使用 CPU？", "Does RDMA avoid the CPU entirely?", "数据面可以绕过 host staging，但连接建立、内存注册、页固定、队列配置、控制与错误处理仍需要 CPU/驱动。GPU-initiated 通信也不取消全部控制平面。", "Payload can bypass host staging, but connection setup, registration, pinning, queues, control and errors still involve CPU/drivers. GPU-initiated communication does not remove all control-plane work.") +
      faq("NCCL busbw 可以跨不同 rank 数直接比较吗？", "Can NCCL busbw be compared across rank counts?", "可以作为标准化参考，但必须同操作、消息大小定义、路径、拓扑、dtype、warmup、迭代窗口与错误状态；小消息看 latency，大消息看带宽，并用实际训练/推理负载复验。", "It is a normalized reference, but compare the same operation, size scope, path, topology, dtype, warmup, iterations and error state. Use latency for small messages, bandwidth for large ones, then verify real workloads.") +
      quiz("network", "8 ranks AllReduce 的 busbw/algbw 系数是多少？", "What is busbw/algbw for an 8-rank AllReduce?", [["1", "1"], ["1.75", "1.75"], ["8", "8"]], 1, "2×(8−1)/8=1.75；这是通信量归一化，不是增加硬件带宽。", "2×(8−1)/8=1.75; this normalizes communication volume, not hardware bandwidth.")) +
    section("checklist", "网络验收必须覆盖微基准、拥塞和真实通信图", "Fabric acceptance needs microbenchmarks, congestion and real communication graphs",
      checklist([["ib_write_bw/NCCL payload 与 topology 对齐，记录 correctness errors", "Align RDMA/NCCL payload and topology; record correctness errors"],
        ["记录 GPU/NIC 亲和、单向/双向、端口与 oversubscription", "Record GPU/NIC affinity, direction, ports and oversubscription"],
        ["测试 ECN/CNP/PFC、watchdog、incast、拥塞与单轨故障", "Test ECN/CNP/PFC, watchdogs, incast, congestion and one-rail failure"],
        ["验证实际 EP all-to-all + KV handoff 共存的尾延迟和 Goodput", "Validate tail latency/goodput with real EP all-to-all plus KV handoff"]]));
}

export const bodies = { metrics: metricsContent, capacity: capacityContent, sizing: sizingContent, pd: pdContent, parallel: parallelContent, network: networkContent };
