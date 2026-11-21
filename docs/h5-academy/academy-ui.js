(function () {
  "use strict";
  const M = window.AcademyMath;
  const lang = () => document.documentElement.lang === "en" ? "en" : "zh";
  const t = (zh, en) => lang() === "en" ? en : zh;
  const esc = (s) => String(s).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
  const fmt = (n, digits = 2) => n === null ? "N/A" : n.toLocaleString(lang() === "en" ? "en-US" : "zh-CN", { maximumFractionDigits: digits });
  const kpi = (value, label) => `<div class="kpi"><strong>${esc(value)}</strong><span>${esc(label)}</span></div>`;
  const kpis = (items) => `<div class="kpis">${items.map(([v, l]) => kpi(v, l)).join("")}</div>`;
  const note = (text) => `<div class="notice">${esc(text)}</div>`;
  const legend = (labels) => `<div class="legend">${labels.map((l, i) => `<span><i class="${["primary", "secondary", "tertiary"][i % 3]}"></i>${esc(l)}</span>`).join("")}</div>`;
  const dataTable = (headers, rows, caption) => `<div class="table-wrap"><table><caption>${esc(caption)}</caption><thead><tr>${headers.map((v) => `<th scope="col">${esc(v)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((v, i) => i === 0 ? `<th scope="row">${esc(v)}</th>` : `<td>${esc(v)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  let chartId = 0;
  function svg(body, caption, height = 310) {
    const id = `interactive-chart-${chartId++}`;
    return `<figure class="chart"><svg viewBox="0 0 800 ${height}" role="img" aria-labelledby="${id}"><title id="${id}">${esc(caption)}</title>${body}</svg><figcaption>${esc(caption)}</figcaption></figure>`;
  }
  function plot(series, xLabel, yLabel, caption, gate = null) {
    const width = 640;
    const height = 200;
    const x0 = 100;
    const y0 = 30;
    const maxX = Math.max(...series.flatMap((s) => s.values.map((p) => p.x)), 1);
    const maxY = Math.max(...series.flatMap((s) => s.values.map((p) => p.y)), gate === null ? 0 : gate, 1) * 1.1;
    const x = (v) => x0 + width * v / maxX;
    const y = (v) => y0 + height - height * v / maxY;
    let body = "";
    for (let i = 0; i <= 4; i += 1) {
      const valueY = maxY * i / 4;
      const valueX = maxX * i / 4;
      body += `<path d="M ${x0} ${y(valueY)} H ${x0 + width}" class="grid-line"/><text x="${x0 - 12}" y="${y(valueY) + 5}" text-anchor="end">${fmt(valueY, 1)}</text>`;
      body += `<text x="${x(valueX)}" y="${y0 + height + 25}" text-anchor="middle">${fmt(valueX, 0)}</text>`;
    }
    body += `<path d="M ${x0} ${y0} V ${y0 + height} H ${x0 + width}" class="axis"/>`;
    if (gate !== null) body += `<path d="M ${x0} ${y(gate)} H ${x0 + width}" class="line tertiary" stroke-dasharray="6 5"/><text x="${x0 + width}" y="${y(gate) - 8}" text-anchor="end">SLO ${fmt(gate)}</text>`;
    series.forEach((s, index) => {
      const cls = ["primary", "secondary", "tertiary"][index % 3];
      body += `<polyline class="line ${cls}" points="${s.values.map((v) => `${x(v.x)},${y(v.y)}`).join(" ")}"/>`;
      body += s.values.map((v) => `<circle class="${cls}" cx="${x(v.x)}" cy="${y(v.y)}" r="3"><title>${esc(s.label)}: ${fmt(v.x)}, ${fmt(v.y)}</title></circle>`).join("");
    });
    body += `<text x="${x0}" y="18">${esc(yLabel)}</text><text x="${x0 + width}" y="285" text-anchor="end">${esc(xLabel)}</text>`;
    return legend(series.map((s) => s.label)) + svg(body, caption);
  }
  function barChart(rows, caption, unit = "") {
    const max = Math.max(...rows.map((r) => r.value), 1);
    const height = rows.length * 54 + 40;
    let body = "";
    rows.forEach((r, i) => {
      const y = 20 + i * 54;
      body += `<text x="10" y="${y + 23}">${esc(r.label)}</text><rect x="225" y="${y}" width="${Math.max(0, r.value / max * 435)}" height="30" rx="4" class="${["primary", "secondary", "tertiary"][i % 3]}"/><text x="${235 + r.value / max * 435}" y="${y + 22}">${fmt(r.value)} ${esc(unit)}</text>`;
    });
    return svg(body, caption, height);
  }
  const boxes = (items) => items.map((item, i) => {
    const x = 20 + i * 195;
    return `<rect x="${x}" y="50" width="175" height="90" rx="8" class="${i === 0 ? "primary" : "surface"}"/><text x="${x + 87}" y="85" text-anchor="middle" class="${i === 0 ? "inverse" : "label"}">${esc(item[0])}</text><text x="${x + 87}" y="115" text-anchor="middle" class="${i === 0 ? "inverse" : "label"}">${esc(item[1])}</text>${i < items.length - 1 ? `<path d="M ${x + 177} 95 H ${x + 192}" class="line secondary"/>` : ""}`;
  }).join("");
  function lifecycle(p) {
    const r = M.lifecycle(p);
    const segments = [{ label: t("排队", "Queue"), ms: p.queue }, { label: "Prefill", ms: p.prefill }, { label: t("交付", "Delivery"), ms: p.delivery }];
    const total = Math.max(r.ttft, 1);
    let cursor = 60;
    let body = `<text x="60" y="25">${t("TTFT 放大视图（按实际毫秒比例）", "TTFT zoom (proportional milliseconds)")}</text>`;
    segments.forEach((s, i) => {
      const w = s.ms / total * 680;
      body += `<rect x="${cursor}" y="45" width="${w}" height="38" class="${["primary", "secondary", "tertiary"][i]}"><title>${esc(s.label)} ${fmt(s.ms)} ms</title></rect>`;
      cursor += w;
    });
    body += `<text x="60" y="118">0</text><text x="740" y="118" text-anchor="end">${t("首 Token", "First token")} ${fmt(r.ttft)} ms</text>`;
    body += `<text x="60" y="155">${t("完整请求时间轴", "Full request timeline")}</text><rect x="60" y="177" width="${680 * r.ttft / Math.max(r.e2e, 1)}" height="30" class="primary"/><rect x="${60 + 680 * r.ttft / Math.max(r.e2e, 1)}" y="177" width="${680 * r.decode / Math.max(r.e2e, 1)}" height="30" class="secondary"/>`;
    body += `<text x="60" y="243">TTFT ${fmt(r.ttft)} ms</text><text x="740" y="243" text-anchor="end">E2EL ${fmt(r.e2e)} ms</text>`;
    body += `<text x="60" y="280">${t("Decode 后续间隔数", "Subsequent decode intervals")}: ${p.output - 1}; TPOT ${r.measuredTPOT === null ? "N/A" : fmt(p.tpot) + " ms"}</text>`;
    return kpis([[fmt(r.ttft) + " ms", "TTFT"], [fmt(r.e2e / 1000) + " s", "E2EL"], [fmt(r.userTPS) + (r.userTPS === null ? "" : " tok/s"), t("单用户后续流式速率", "Subsequent per-user streaming")]]) +
      legend(segments.map((s) => `${s.label} ${fmt(s.ms)} ms`)) + svg(body, t("时间轴由当前输入计算；首 Token 已计入 TTFT。", "Timeline is computed from inputs; the first token is already in TTFT.")) +
      dataTable([t("阶段", "Stage"), "ms", t("含义", "Meaning")], [[t("排队", "Queue"), fmt(p.queue), t("等待资源", "Waiting for resources")], ["Prefill", fmt(p.prefill), t("含首字计算", "Includes first-token computation")], [t("交付", "Delivery"), fmt(p.delivery), t("网关/网络", "Gateway/network")], ["Decode", fmt(r.decode), `${p.output - 1} × ${fmt(p.tpot)} ms`]], t("可手算的阶段分解", "Hand-checkable stage decomposition"));
  }
  function goodput(p) {
    const r = M.goodput(p);
    const all = Array.from({ length: 32 }, (_, i) => {
      const batch = (i + 1) * 4;
      return { batch, ...M.goodput({ ...p, batch }) };
    });
    const frontier = [{ label: t("模拟 Throughput→TPOT", "Synthetic throughput→TPOT"), values: all.map((v) => ({ x: v.throughput, y: v.meanTPOT })) }];
    const series = [{ label: "Output TPS", values: all.map((v) => ({ x: v.batch, y: v.throughput })) }, { label: "Goodput", values: all.map((v) => ({ x: v.batch, y: v.goodput })) }];
    const best = all.reduce((a, v) => v.goodput > a.goodput ? v : a, all[0]);
    return kpis([[fmt(r.throughput) + " tok/s", t("模拟输出吞吐", "Synthetic output throughput")], [fmt(r.goodput) + " tok/s", "Synthetic Goodput"], [`${r.passed}/${p.batch}`, t("双门槛达标请求", "Requests passing both gates")], [fmt(r.meanTPOT) + " ms", "Mean TPOT"], [String(best.batch), t("当前采样网格 Goodput 最优 batch", "Best goodput batch on sampled grid")], ["synthetic", t("不是实测", "Not measured")]]) +
      plot(frontier, "Output tok/s", "TPOT ms", t("教学 tradeoff 曲线；非实测帕累托前沿。", "Teaching tradeoff curve, not a measured Pareto frontier."), p.tpotLimit) +
      plot(series, "Active batch", "tok/s", t("Goodput 甜点：越过双门槛后，输出吞吐不再等于达标吞吐。", "Goodput sweet spot: beyond both gates, throughput is no longer compliant throughput.")) +
      dataTable(["Batch", "TPOT ms", "TPS", "Goodput"], all.filter((v) => [4, 16, 32, 48, 64, 96, 128].includes(v.batch)).map((v) => [v.batch, fmt(v.meanTPOT), fmt(v.throughput), fmt(v.goodput)]), t("确定性教学样本；每请求输出长度相同", "Deterministic teaching samples; equal output length/request"));
  }
  function capacity(p) {
    const qps = p.mode === "business" ? M.businessQPS(p) : p.qps;
    const r = M.capacity({ ...p, qps });
    const funnel = `<path d="M 255 20 H 545 L 510 90 H 290 Z" class="primary"/><text x="400" y="48" text-anchor="middle" class="inverse">λ_plan = ${fmt(r.qps)} QPS</text><text x="400" y="73" text-anchor="middle" class="inverse">× E[input / output tokens]</text><path d="M 400 92 V 115 H 205 V 140 M 400 115 H 595 V 140" class="line secondary"/><path d="M 60 145 H 350 L 320 215 H 90 Z" class="secondary"/><path d="M 450 145 H 740 L 710 215 H 480 Z" class="primary"/><text x="205" y="173" text-anchor="middle" class="inverse">Input ${fmt(r.inputTPS)} TPS</text><text x="205" y="199" text-anchor="middle" class="inverse">× 60</text><text x="595" y="173" text-anchor="middle" class="inverse">Output ${fmt(r.outputTPS)} TPS</text><text x="595" y="199" text-anchor="middle" class="inverse">× 60</text><path d="M 205 219 V 245 M 595 219 V 245" class="line secondary"/><rect x="60" y="250" width="290" height="55" rx="8" class="surface"/><rect x="450" y="250" width="290" height="55" rx="8" class="surface"/><text x="205" y="283" text-anchor="middle">${fmt(r.inputTPM, 0)} Input TPM</text><text x="595" y="283" text-anchor="middle">${fmt(r.outputTPM, 0)} Output TPM</text>`;
    return kpis([[fmt(r.qps), t("含需求余量的 QPS", "QPS with demand headroom")], [fmt(r.inputTPS), "Input TPS"], [fmt(r.outputTPS), "Output TPS"], [fmt(r.inputTPM, 0), "Input TPM"], [fmt(r.outputTPM, 0), "Output TPM"], [fmt(r.active), t("平均在途请求 L", "Mean in-flight requests L")]]) +
      svg(funnel, t("需求漏斗/单位转换：形状宽度不跨单位编码数值，精确需求显示在标签与表中。", "Demand funnel/unit conversion: shape widths do not encode quantities across units; labels and table carry exact demand."), 335) +
      barChart([{ label: "Input TPM", value: r.inputTPM }, { label: "Output TPM", value: r.outputTPM }], t("QPS×平均 Token 长度×60；原始 Token 流量，不是 PTU。", "QPS×mean token count×60; raw token demand, not PTU."), "tok/min") +
      dataTable([t("约束", "Constraint"), t("需求", "Demand"), t("实例下界", "Instance floor")], [
        ["Input", fmt(r.inputTPS) + " TPS", r.inputInstances],
        ["Output", fmt(r.outputTPS) + " TPS", r.outputInstances],
        [t("在途并发", "In-flight concurrency"), fmt(r.active), r.batchInstances],
        [t("取最大值", "Maximum"), `${fmt(r.residence)} s E[W]`, r.instances]
      ], t("实例数下界，非单卡数；容量示例必须替换为实测", "Instance floors, not GPU counts; replace example capacities with measurements")) +
      note(t(`假设平均停留 ${fmt(r.residence)} s；其中后续 decode 活跃 L≈${fmt(r.decodeActive)}。动态 continuous batching 的活跃数不会固定等于这个均值。`, `Assumed mean residence ${fmt(r.residence)} s; subsequent decode-active L≈${fmt(r.decodeActive)}. Continuous-batching occupancy is not fixed at this mean.`));
  }
  function ptu(p) {
    const r = M.ptu(p);
    return kpis([[fmt(r.normalized, 0), "Normalized TPM"], [fmt(r.raw), t("未取整 PTU", "Raw PTU")], [String(r.provisioned), t("演示部署 PTU", "Illustrative deployed PTU")]]) +
      barChart([{ label: "Input equivalent", value: p.inputTPM * ((1 - p.cache) + p.cache * p.cachedWeight) }, { label: "Output weighted", value: p.outputTPM * p.outputWeight }], t("按手工权重归一化，不是当前云端 quota 或 GPU 数。", "Normalized using manual weights, not current cloud quota or GPU count."), "equiv TPM") +
      note(t("若缓存写入/多模态/上下文档位还有独立权重，这个两权重演示不足以报价；请使用当前模型专用公式。", "If cache writes, multimodal or context tiers have additional weights, this two-weight demo is insufficient for a quote; use current model-specific formulas."));
  }
  function cost(p) {
    const r = M.cost(p);
    return kpis([[r.perMillionOutput === null ? "N/A" : "$" + fmt(r.perMillionOutput), t("自建 $/1M 达标输出", "Self-hosted $/1M compliant output")], ["$" + fmt(r.periodCost), t("同负载时段 API 成本示例", "API period cost at the same load")], [fmt(p.hours) + " h", t("持续时间假设", "Duration assumption")]]) +
      barChart([{ label: "API input cost", value: p.inputTPS * p.priceIn * p.hours * 3600 / 1e6 }, { label: "API output cost", value: p.outputTPS * p.priceOut * p.hours * 3600 / 1e6 }], t("手工输入单价，不是实时供应商报价；自建与 API 范围需对齐。", "Manual prices, not a live provider quote; align self-hosted and API scope."), "$");
  }
  function sizing(p) {
    const r = M.sizing(p);
    const counts = [...new Set([1, 2, 4, 8, r.candidate])].sort((a, b) => a - b);
    const max = Math.max(r.weight + r.kv + p.runtime, r.gpu.memory) * 1.08;
    const y = (v) => 240 - v / max * 190;
    let body = `<text x="20" y="24">${t("理想平均每卡显存 GB", "Ideal average per-GPU memory GB")}</text>`;
    counts.forEach((count, i) => {
      const x = 85 + i * 135;
      let used = 0;
      [[r.weight / count, "secondary"], [r.kv / count, "primary"], [p.runtime, "tertiary"]].forEach(([value, cls]) => {
        body += `<rect x="${x}" y="${y(used + value)}" width="75" height="${value / max * 190}" class="${cls}"><title>${fmt(value)} GB</title></rect>`;
        used += value;
      });
      body += `<rect x="${x - 4}" y="${y(r.gpu.memory)}" width="83" height="${r.gpu.memory / max * 190}" class="line axis"/><path d="M ${x - 8} ${y(r.gpu.memory * p.safe)} H ${x + 83}" class="grid-line"/><text x="${x + 37}" y="263" text-anchor="middle">${count} GPU</text><text x="${x + 37}" y="${Math.max(38, y(used) - 8)}" text-anchor="middle">${fmt(used, 1)}</text>`;
    });
    body += `<text x="20" y="294">${t("外框=物理容量；虚线=规划安全水位；不保证逐 rank 可实现", "Outline=physical capacity; dashed=safe margin; not guaranteed per rank")}</text>`;
    const lengths = [1024, 4096, 8192, 16384, 32768, 65536, 131072];
    const comparison = [
      { label: "MHA (counterfactual)", values: lengths.map((s) => ({ x: s, y: M.kvPerToken(r.model, p.kvBytes, "mha") * s * p.batch / 1e9 })) },
      { label: "GQA (counterfactual if MLA)", values: lengths.map((s) => ({ x: s, y: M.kvPerToken(r.model, p.kvBytes, "gqa") * s * p.batch / 1e9 })) },
      { label: "DeepSeek MLA (61 layers)", values: lengths.map((s) => ({ x: s, y: M.kvPerToken(M.models.deepseek, p.kvBytes) * s * p.batch / 1e9 })) }
    ];
    return kpis([[`${r.memoryCards}`, t("理想显存下界 GPU", "Ideal memory floor GPUs")], [`${r.bandwidthCards}`, t("权重+KV 读带宽场景下界 GPU", "Weight+KV bandwidth scenario GPUs")], [`${r.computeCards}`, t("Prefill 算力下界 GPU", "Prefill compute floor GPUs")], [`${r.candidate}`, t("待测试候选（2 的幂）", "Candidate to test (power of 2)")], [fmt(r.weight) + " GB", t("总权重含手工开销", "Total weights + manual allowance")], [fmt(r.kv) + " GB", t("驻留 KV 含复制倍率", "Resident KV + replication factor")]]) +
      legend([t("KV Cache", "KV cache"), t("总权重", "Total weights"), t("每卡运行时预留", "Runtime reserve/GPU")]) + svg(body, t("显存堆叠按公式绘制；多卡项假设理想切分，超物理容量不会被隐藏。", "Memory stacks are formula-derived; multi-GPU bars assume ideal sharding and do not hide overflow."), 325) +
      plot(comparison, "Context tokens", "Aggregate KV GB", t("KV 对比固定 B 与 dtype：MHA/GQA 用选定模型几何反事实；MLA 用 DeepSeek 的61层，不是同模型替换开关。", "Fixed B/dtype comparison: MHA/GQA use selected-model geometry counterfactuals; MLA uses DeepSeek's 61 layers, not a model replacement switch.")) +
      dataTable([t("候选卡数下的估算", "Estimates at candidate GPU count"), t("值", "Value"), t("限制", "Limit")], [
        [t("单 Token 权重乐观 TPOT", "Single-token optimistic weight TPOT"), fmt(r.optimisticTPOT) + " ms", "weights only"],
        [t("批内专家并集权重 TPOT", "Batch-union weight TPOT"), fmt(r.unionTPOT) + " ms", t("均匀路由假设", "Uniform routing assumption")],
        [t("每步读全 KV 估算", "Full-KV-read estimate/step"), fmt(r.kvReadTPOT) + " ms", t("额外流量，不含通信/算力", "Additional traffic; excludes communication/compute")],
        [t("Prefill 计算下界", "Prefill compute floor"), fmt(r.prefillMs) + " ms", t("不含预留/首字/网络", "Excludes reserve/first-token/transport")],
        [t("权重并集流量", "Batch-union weight traffic"), fmt(r.unionTraffic) + " GB/step", t("不是总加载权重", "Not all loaded weights")]
      ], t("不是 TTFT/TPOT 达标承诺", "Not a TTFT/TPOT compliance promise")) +
      (!r.contextSupported ? note(t("上下文超过预设支持范围；显存计算不会赋予模型长上下文能力。", "Context exceeds the preset range; memory calculations do not confer long-context capability.")) : "");
  }
  function pd(p) {
    const r = M.transfer(p);
    const lengths = [1024, 4096, 8192, 16384, 32768, 65536, 131072];
    const points = lengths.map((context) => ({ context, ...M.pdSamples({ ...p, context }) }));
    const labels = { colocated: t("共池", "Colocated"), chunked: "Chunked Prefill", pd: t("PD（残留干扰假设）", "PD (residual assumption)") };
    const series = Object.entries(labels).map(([key, label]) => ({ label, values: points.map((v) => ({ x: v.context, y: v.p99[key] })) }));
    const prefill = p.context / p.prefillRate * 1000;
    const max = Math.max(prefill + 5 * p.base, 1);
    const scale = (v) => v / max * 630;
    let timeline = `<text x="10" y="25">${t("共池：Prefill 对 Decode 的合成干扰", "Colocated: synthetic prefill interference")}</text><rect x="100" y="45" width="${scale(prefill)}" height="35" class="primary"/><text x="110" y="105">Prefill ${fmt(prefill)} ms</text>`;
    for (let i = 0; i < 5; i += 1) {
      timeline += `<rect x="${100 + scale(prefill + i * p.base)}" y="45" width="${scale(p.base) * 0.85}" height="35" class="secondary"/>`;
      timeline += `<rect x="${100 + i * 120}" y="173" width="90" height="32" class="secondary"><title>decode work (qualitative pipeline)</title></rect>`;
    }
    timeline += `<text x="10" y="150">${t("独立 D 池：去除 Prefill 阻塞，但仍有残留调度/通信", "Separate D pool: no prefill stall, residual scheduling/communication remains")}</text><text x="100" y="245">${t("PD 行是定性流水线示意，不与上一行共享时间比例", "PD row is a qualitative pipeline, not the same time scale")}</text>`;
    const current = M.pdSamples(p);
    return kpis([[fmt(r.bytes / 1e9) + " GB", t("新传输 KV Payload", "New KV payload")], [fmt(r.rawMs) + " ms", t("原始传输时间", "Raw transfer time")], [fmt(r.exposedMs) + " ms", t("暴露交接时间", "Exposed handoff")], [fmt(r.utilization * 100) + "%", t("单方向 fabric 负载", "One-way fabric load")], [fmt(r.requiredGbps) + " Gb/s", t("含效率的总线速需求", "Aggregate line-rate need incl. efficiency")], ["200 ITL", t("每上下文点模拟样本数", "Synthetic samples/context point")]]) +
      (r.utilization >= 1 ? note(t("警告：KV 到达量不低于有效带宽，持续队列不稳定。下面曲线不预测此时的真实 P99；增加带宽、减少交接量或限制接入。", "Warning: KV arrival meets/exceeds effective bandwidth; sustained queues are unstable. Curves do not predict real P99 under this overload. Add bandwidth, reduce handoff or control admission.")) : "") +
      svg(timeline, t("混部阻塞与分池流水线示意。", "Colocated stalls versus separate-pool pipeline illustration."), 280) +
      plot(series, "Prompt tokens", "Synthetic P99 ITL ms", t("P99 来自每点 200 个确定性模拟 ITL；nearest-rank，5% Prefill 干扰；不是硬件测量，也不保证 PD 无抖动。", "P99 from 200 deterministic synthetic ITLs/point; nearest-rank, 5% interference; no hardware measurement or zero-jitter promise.")) +
      dataTable([t("当前上下文", "Current context"), "Synthetic P99 ITL ms", t("样本分母", "Sample count")], Object.entries(labels).map(([key, label]) => [label, fmt(current.p99[key]), 200]), `${p.context} tokens · synthetic`);
  }
  function parallel(p) {
    M.integer(p.ranks, "ranks", 2, 8);
    M.integer(p.batch, "microbatches", 1, 128);
    M.number(p.payload, "payload", 0.001);
    M.number(p.bandwidth, "bandwidth", 0.001);
    const n = p.ranks;
    const utilization = p.batch / (p.batch + n - 1);
    const transferMs = p.payload / p.bandwidth;
    const mode = {
      tp: { title: t("TP：同一层的张量切片", "TP: tensor shards of one layer"), subtitle: "AllReduce / RS+AG", caption: t("一个请求进入所有 TP ranks，张量分片再归约；通信往往每层发生。", "One request uses all TP ranks; tensor shards are reduced, often each layer.") },
      pp: { title: t("PP：连续层分为 stages", "PP: layer ranges become stages"), subtitle: "SendRecv activations", caption: t("同一请求顺序经过所有 stage；更多 microbatch 填满流水线，不把单请求路径变成并行完成。", "One request visits stages serially; microbatches fill the pipeline without parallelizing its entire path.") },
      dp: { title: t("DP：请求/Attention 分片", "DP: request/attention shards"), subtitle: "Router → shards", caption: t("普通 DP 是独立副本；Attention DP + EP 时只画 attention 请求分片，expert pool 是共享并行组。", "Ordinary DP uses replicas; with attention DP + EP, these are attention request shards with a shared expert group.") },
      ep: { title: t("EP：专家分布在 ranks", "EP: experts distributed over ranks"), subtitle: "AllToAll dispatch/combine", caption: t("Token 只路由所选 experts；图展示 dispatch/combine，不表示所有请求均匀命中每张卡。", "Tokens route to selected experts; dispatch/combine does not imply uniform rank load.") }
    };
    if (!Object.hasOwn(mode, p.kind)) throw new RangeError("Unknown parallel kind");
    const m = mode[p.kind];
    let body = `<text x="20" y="28">${esc(m.title)}</text><rect x="20" y="60" width="180" height="95" rx="8" class="primary"/><text x="110" y="97" text-anchor="middle" class="inverse">${p.kind === "dp" ? "Request router" : "Input tokens"}</text><text x="110" y="127" text-anchor="middle" class="inverse">${esc(m.subtitle)}</text>`;
    const x0 = 250;
    const width = 480 / n;
    for (let i = 0; i < n; i += 1) {
      const x = x0 + i * width;
      const label = p.kind === "tp" ? `W/${n}` : p.kind === "pp" ? `stage ${i + 1}` : p.kind === "dp" ? `Req ${i + 1}` : `E${i + 1}`;
      body += `<rect x="${x}" y="65" width="${width - 10}" height="90" rx="6" class="surface"/><text x="${x + (width - 10) / 2}" y="102" text-anchor="middle" class="label">${esc(label)}</text><text x="${x + (width - 10) / 2}" y="128" text-anchor="middle" class="label">rank ${i}</text>`;
      if (p.kind !== "pp" || i === 0) body += `<path d="M 200 107 L ${x + width / 2} 50 L ${x + width / 2} 60" class="line secondary"/>`;
      if (p.kind === "pp" && i < n - 1) body += `<path d="M ${x + width - 10} 110 H ${x + width}" class="line primary"/>`;
    }
    body += `<rect x="250" y="205" width="470" height="50" rx="8" class="surface"/><text x="485" y="236" text-anchor="middle">${p.kind === "ep" ? "combine → original request shards" : p.kind === "dp" ? "independent outputs / shared expert group when MoE" : "output / communication and synchronization"}</text>`;
    return kpis([[String(n), `${p.kind.toUpperCase()} ranks`], [fmt(transferMs) + " ms", t("单步骤裸传输估算", "Bare transfer estimate/step")], [p.kind === "pp" ? fmt(utilization * 100) + "%" : "N/A", t("平衡 PP 理想流水利用率", "Ideal balanced-PP utilization")]]) +
      svg(body, m.caption) +
      note(t("图解不是实测缩放率；切分合法性、层数、heads、专家容量和通信重叠必须由实际引擎配置确认。", "The diagram is not measured scaling. Validate partition legality, layers, heads, expert capacity and overlap in the actual engine."));
  }
  function network(p) {
    const r = M.collective(p);
    M.integer(p.rails, "rails", 1, 16);
    M.number(p.rate, "rate", 0.001);
    M.number(p.efficiency, "efficiency", 0.01, 1);
    const budget = p.rate / 8 * p.rails * p.efficiency;
    return kpis([[fmt(r.algbw) + " GB/s", "algbw"], [fmt(r.busbw) + " GB/s", "busbw"], [fmt(r.factor, 4) + "×", t("标准化修正系数", "Normalization factor")], [fmt(budget) + " GB/s", t("所选单向 payload 预算", "Selected one-way payload budget")]]) +
      barChart([{ label: "algbw", value: r.algbw }, { label: "busbw", value: r.busbw }, { label: "NIC payload budget", value: budget }], t("不同层次的带宽量并列，不自动把 NIC 预算当 NVLink/混合路径上限。", "Different bandwidth scopes; NIC budget is not automatically a NVLink/mixed-path cap."), "GB/s") +
      (r.busbw > budget ? note(t("指标高于所选 NIC 预算：核对是否为 NVLink/多端口路径、size 定义与方向，不能直接认为单端口超线速。", "Metric exceeds the chosen NIC budget: verify NVLink/multiport paths, size scope and direction; do not infer one-port overspeed.")) : "") +
      dataTable(["Operation", "S definition", "Factor"], [
        ["AllReduce", t("每 rank 完整消息", "Complete per-rank message"), "2(n−1)/n"],
        ["AllGather", t("所有 ranks 聚合后的总数组", "Total gathered array"), "(n−1)/n"],
        ["ReduceScatter", t("scatter 前的总数组", "Total pre-scatter array"), "(n−1)/n"],
        ["AllToAll", t("每 rank 总发送数组", "Total send array per rank"), "(n−1)/n"],
        ["Broadcast / Reduce", t("根对应的完整消息", "Complete root-associated message"), "1"]
      ], t("NCCL-tests 消息大小口径", "NCCL-tests message-size scope"));
  }
  const pauseStates = new WeakMap();
  function congestion(p, form) {
    const r = M.congestion(p, pauseStates.get(form) || false);
    pauseStates.set(form, r.paused);
    let body = `<rect x="60" y="60" width="680" height="45" rx="6" class="surface"/><rect x="60" y="60" width="${680 * p.queue / 100}" height="45" class="${r.paused ? "tertiary" : "primary"}"/>`;
    [[p.ecn, "ECN"], [p.pfc, "Xoff"], [p.xon, "Xon"]].forEach(([value, label], i) => {
      const x = 60 + 680 * value / 100;
      const ty = 137 + i * 24;
      body += `<path d="M ${x} 47 V 110" class="line ${i === 0 ? "secondary" : "tertiary"}"/><text x="${x}" y="${ty}" text-anchor="middle">${label} ${value}%</text>`;
    });
    body += boxes([["Switch queue", r.marked ? "ECN: CE mark" : "below ECN"], ["Receiver RNIC", r.marked ? "send CNP" : "no CNP"], ["Sender RNIC", r.marked ? "DCQCN reduce" : "normal rate"], ["Upstream link", r.paused ? "PFC paused" : "PFC clear"]]).replaceAll('y="50"', 'y="220"').replaceAll('y="85"', 'y="255"').replaceAll('y="115"', 'y="285"').replaceAll(" 95", " 265");
    return kpis([[p.queue + "%", t("队列占用（示意）", "Illustrative queue occupancy")], [r.marked ? "CE / CNP" : "No mark", t("ECN 反馈状态", "ECN feedback state")], [r.paused ? "PAUSE" : "CLEAR", t("PFC 逐跳状态", "PFC hop state")]]) +
      svg(body, t("状态机保留 PFC 滞回：从 Xoff 以上降到 Xon 以下才恢复；非网络仿真器。", "PFC hysteresis is retained: after Xoff, resume only at/below Xon; not a network simulator."), 335) +
      note(t("尝试先把队列推至 Xoff 以上，再下降到 Xon 与 Xoff 之间：pause 会保持。恢复算例按钮重置状态；真实 ECN 采用概率/多阈值策略，本图只显示概念。", "Raise the queue above Xoff, then lower it between Xon and Xoff: pause remains. Reset clears state. Real ECN uses probabilistic/multithreshold policies; this chart only explains concepts."));
  }
  const renderers = { lifecycle, goodput, capacity, ptu, cost, sizing, pd, parallel, network, congestion };
  function inputValues(form) {
    const values = {};
    for (const input of form.querySelectorAll("[data-key]")) {
      if (input.disabled) continue;
      if (input.type === "checkbox") values[input.dataset.key] = input.checked;
      else if (["model", "gpu", "mode", "kind", "operation"].includes(input.dataset.key)) values[input.dataset.key] = input.value;
      else {
        if (!input.value.trim()) throw new RangeError(t("请填写所有启用的数值字段", "Fill every enabled numeric field"));
        values[input.dataset.key] = Number(input.value);
      }
    }
    return values;
  }
  function prepare(form) {
    if (form.dataset.lab === "capacity") {
      const business = form.querySelector('[data-key="mode"]').value === "business";
      for (const key of ["dau", "requests", "hours", "peak"]) form.querySelector(`[data-key="${key}"]`).disabled = !business;
      form.querySelector('[data-key="qps"]').disabled = business;
    }
    if (form.dataset.lab === "sizing") {
      const input = form.querySelector('[data-key="mtp"]');
      input.disabled = form.querySelector('[data-key="model"]').value !== "deepseek";
      if (input.disabled) input.checked = false;
    }
  }
  function render(form) {
    const workbench = form.closest("[data-workbench]");
    const errorBox = workbench.querySelector(".error");
    const results = workbench.querySelector("[data-results]");
    prepare(form);
    for (const output of form.querySelectorAll("[data-for]")) output.value = form.querySelector(`[data-key="${output.dataset.for}"]`).value;
    try {
      if (!form.checkValidity()) throw new RangeError(t("数值超出字段允许范围，或步长不正确；请检查当前输入。", "A value is outside its allowed range or step; inspect the inputs."));
      results.innerHTML = renderers[form.dataset.lab](inputValues(form), form);
      results.removeAttribute("aria-invalid");
      errorBox.hidden = true;
      errorBox.textContent = "";
    } catch (error) {
      results.replaceChildren();
      results.setAttribute("aria-invalid", "true");
      errorBox.hidden = false;
      errorBox.textContent = t("无法计算：", "Cannot calculate: ") + error.message;
      if (!(error instanceof RangeError)) {
        console.error(error);
        throw error;
      }
    }
  }
  function staticCharts() {
    const el = document.querySelector('[data-static-chart="scenarios"]');
    if (el) {
      el.innerHTML = barChart([
        { label: t("语音 LLM 子预算", "Voice LLM sub-budget"), value: 120 },
        { label: "IDE", value: 200 },
        { label: t("企业对话", "Enterprise chat"), value: 800 },
        { label: "R1", value: 2000 }
      ], t("教学 TTFT 目标，非实测和行业 SLA；离线批处理为 N/A，不绘制为零。", "Teaching TTFT targets, not measurements or industry SLAs; offline batch is N/A, not zero."), "ms");
    }
    const matrix = document.querySelector("#acceleration tbody");
    if (matrix) {
      for (const row of matrix.rows) {
        for (let i = 1; i < row.cells.length - 1; i += 1) row.cells[i].className = i === 3 ? "heat-good" : "heat-trade";
      }
    }
  }
  function carryNavigation() {
    for (const link of document.querySelectorAll('a[data-carry]')) {
      const url = new URL(link.getAttribute("href"), location.href);
      url.searchParams.set("lang", lang() === "en" ? "en" : "zh");
      url.searchParams.set("scoutTheme", document.documentElement.dataset.theme);
      link.href = url.href;
    }
  }
  function refresh() {
    document.querySelectorAll("form[data-lab]").forEach(render);
    staticCharts();
    carryNavigation();
    document.querySelector("#language").textContent = lang() === "en" ? "中文" : "English";
    document.querySelector("#theme").textContent = t(document.documentElement.dataset.theme === "dark" ? "浅色" : "深色", document.documentElement.dataset.theme === "dark" ? "Light" : "Dark");
  }
  document.querySelectorAll("form[data-lab]").forEach((form) => {
    form.addEventListener("input", () => render(form));
    form.addEventListener("change", () => render(form));
    form.addEventListener("submit", (event) => event.preventDefault());
    form.addEventListener("reset", () => {
      pauseStates.delete(form);
      setTimeout(() => render(form), 0);
    });
  });
  document.querySelector("#language").addEventListener("click", () => {
    document.documentElement.lang = lang() === "en" ? "zh-CN" : "en";
    refresh();
    document.querySelectorAll(".quiz-result").forEach((el) => { el.textContent = ""; });
  });
  document.querySelector("#theme").addEventListener("click", () => {
    document.documentElement.dataset.theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    refresh();
  });
  document.querySelector("#print").addEventListener("click", () => window.print());
  for (const quiz of document.querySelectorAll("[data-quiz]")) {
    quiz.querySelector("[data-check-quiz]").addEventListener("click", () => {
      const selected = quiz.querySelector("input:checked");
      const output = quiz.querySelector(".quiz-result");
      if (!selected) {
        output.textContent = t("先选择一个答案。", "Select an answer first.");
        return;
      }
      const correct = selected.value === quiz.dataset.correct;
      output.innerHTML = `<strong>${t(correct ? "正确。 " : "再想一想。 ", correct ? "Correct. " : "Try again. ")}</strong>` + quiz.querySelector("template").innerHTML;
    });
  }
  const progress = document.querySelector(".reading-progress");
  window.addEventListener("scroll", () => {
    const total = document.documentElement.scrollHeight - window.innerHeight;
    progress.style.width = (total <= 0 ? 100 : Math.min(100, window.scrollY / total * 100)) + "%";
  }, { passive: true });
  window.AcademyUI = Object.freeze({ refresh });
  refresh();
})();
