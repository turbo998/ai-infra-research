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
  const legend = (labels, classes = []) => `<div class="legend">${labels.map((l, i) => `<span><i class="${esc(classes[i] || ["primary", "secondary", "tertiary"][i % 3])}"></i>${esc(l)}</span>`).join("")}</div>`;
  const dataTable = (headers, rows, caption) => `<div class="table-wrap"><table><caption>${esc(caption)}</caption><thead><tr>${headers.map((v) => `<th scope="col">${esc(v)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((v, i) => i === 0 ? `<th scope="row">${esc(v)}</th>` : `<td>${esc(v)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  let chartId = 0;
  function svg(body, caption, height = 310) {
    const id = `interactive-chart-${chartId++}`;
    return `<figure class="chart"><svg viewBox="0 0 800 ${height}" role="img" aria-labelledby="${id}"><title id="${id}">${esc(caption)}</title>${body}</svg><figcaption>${esc(caption)}</figcaption></figure>`;
  }
  function plot(series, xLabel, yLabel, caption, gate = null, gateLabel = "SLO", integerTicks = false) {
    const width = 640;
    const height = 200;
    const x0 = 100;
    const y0 = 30;
    const maxX = Math.max(...series.flatMap((s) => s.values.map((p) => p.x)), 1);
    const peakY = Math.max(...series.flatMap((s) => s.values.map((p) => p.y)), gate === null ? 0 : gate, 1);
    const maxY = integerTicks ? Math.max(4, Math.ceil(peakY / 4) * 4) : peakY * 1.1;
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
    if (gate !== null) body += `<path d="M ${x0} ${y(gate)} H ${x0 + width}" class="line tertiary" stroke-dasharray="6 5"/><text x="${x0 + width}" y="${y(gate) - 8}" text-anchor="end">${esc(gateLabel)} ${fmt(gate)}</text>`;
    series.forEach((s, index) => {
      const cls = s.colorClass || ["primary", "secondary", "tertiary"][index % 3];
      body += `<polyline class="line ${cls}"${cls === "secondary" ? ' stroke-dasharray="7 4"' : cls === "tertiary" ? ' stroke-dasharray="2 4"' : ""} points="${s.values.map((v) => `${x(v.x)},${y(v.y)}`).join(" ")}"/>`;
      body += s.values.map((v) => `<circle class="${cls}" cx="${x(v.x)}" cy="${y(v.y)}" r="3"><title>${esc(s.label)}: ${fmt(v.x)}, ${fmt(v.y)}</title></circle>`).join("");
    });
    body += `<text x="${x0}" y="18">${esc(yLabel)}</text><text x="${x0 + width}" y="285" text-anchor="end">${esc(xLabel)}</text>`;
    const rows = series.flatMap((s) => s.values.map((value) => [s.label, fmt(value.x), fmt(value.y)]));
    return legend(series.map((s) => s.label), series.map((s, i) => s.colorClass || ["primary", "secondary", "tertiary"][i % 3])) + svg(body, caption) +
      `<details><summary>${t("查看图表原始坐标", "View chart coordinates")}</summary>${dataTable([t("序列", "Series"), xLabel, yLabel], rows, t("重复X表示阶梯跳变，未合并或置零", "Repeated X values preserve step transitions, without merging or zero substitution"))}</details>`;
  }
  function barChart(rows, caption, unit = "") {
    const max = Math.max(...rows.map((r) => r.value), 1);
    const height = rows.length * 54 + 40;
    let body = "";
    rows.forEach((r, i) => {
      const y = 20 + i * 54;
      body += `<text x="10" y="${y + 23}">${esc(r.label)}</text><rect x="225" y="${y}" width="${Math.max(0, r.value / max * 435)}" height="30" rx="4" class="${["primary", "secondary", "tertiary"][i % 3]}"/><text x="${235 + r.value / max * 435}" y="${y + 22}">${fmt(r.value)} ${esc(unit)}</text>`;
    });
    return svg(body, caption, height) + `<details><summary>${t("查看条形图数据", "View bar-chart data")}</summary>${dataTable([t("类别", "Category"), unit || t("值", "Value")], rows.map((row) => [row.label, fmt(row.value)]), caption)}</details>`;
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
  function trace(p) {
    if (p.trace.length > 250000) throw new RangeError(t("JSON 超过本页安全大小", "JSON exceeds this workbench's size limit"));
    let requests;
    try { requests = JSON.parse(p.trace); }
    catch (error) {
      if (!(error instanceof SyntaxError)) throw error;
      throw new RangeError(t("JSON 语法错误：需要数组、双引号与正确逗号；不回显原始内容。", "Invalid JSON: use an array, double quotes and valid commas; raw content is not echoed."));
    }
    const r = M.traceMetrics({ ...p, requests });
    const synthetic = JSON.stringify(requests) === JSON.stringify(M.traceSamples);
    const terminated = r.rows.filter((row) => row.completeInWindow).length;
    const state = { success: t("成功", "success"), failed: t("失败", "failed"), cancelled: t("取消", "cancelled"), rejected: t("拒绝", "rejected") };
    return kpis([[fmt(r.visibleOutputTPS) + " tok/s", t("测窗内原始可见输出", "Raw visible output in window")], [fmt(r.goodOutputTPS) + " tok/s", t("完成且达标的可见输出 Goodput", "Completed compliant visible-output goodput")], [`${r.goodRequests}/${terminated}`, t("计入 Goodput / 全终态请求", "Goodput requests / all terminal requests")], [fmt(r.p99ITL) + " ms", `P99 visible ITL · n=${r.itlSamples}`], [fmt(r.meanRequestTPOT) + " ms", t("有间隔终态请求的 TPOT 均值", "Mean TPOT of terminal requests with intervals")], [fmt(r.meanObservedITL) + " ms", t("全部观察间隔的均值（不同权重）", "Mean observed gap (different weighting)")]]) +
      plot([{ label: t("选定系统边界在途请求", "In-flight requests under selected boundary"), values: r.occupancy }], "Relative time ms", "Active requests", t("按时间戳直接重建阶梯并发；面积/测窗得到平均L，不是KV占用或GPU忙碌曲线。", "Step concurrency reconstructed from timestamps; area/window gives mean L, not KV occupancy or GPU-busy time."), null, "SLO", true) +
      dataTable([t("同一窗口中的量", "Quantity in the same window"), t("值", "Value"), t("口径", "Scope")], [
        ["Offered RPS", fmt(r.offeredRPS), t("arrival 事件，含拒绝", "Arrival events including rejections")],
        ["Known admitted RPS", fmt(r.admittedRPS), t("已记录 admission 事件", "Recorded admission events")],
        ["Terminal RPS", fmt(r.completedRPS), t("所有终态，含失败/取消/拒绝", "All terminal states, including failed/cancelled/rejected")],
        ["Input demand TPS", fmt(r.inputDemandTPS), t("admission 时输入需求，非Prefill完成速率", "Input demand at admission, not prefill-completion rate")],
        ["P95 TTFT", fmt(r.p95TTFT) + " ms", t("本窗观察到的首可见Token", "First visible-token events observed in window")],
        ["L observed", fmt(r.meanActive, 4), t("所选边界时间面积 / 测窗", "Boundary occupancy area / window")],
        ["λ × mean W", fmt(r.littleProduct, 4), `${fmt(r.lambda)} /s × ${fmt(r.meanResidenceMs)} ms / 1000`]
      ], synthetic ? t("synthetic：原创7请求教学记录，未实机测量", "synthetic: original seven-request teaching trace, no hardware measurement") : t("user_supplied：本地录入，未验证测量/时钟/质量", "user_supplied: local input, measurement/clocks/quality unverified")) +
      dataTable(["ID", t("终态", "Terminal state"), "Trace N_visible", "TTFT ms", "TPOT ms", "Max ITL ms", "Visible E2EL ms", "Terminal latency ms", t("品质注释", "Quality annotation"), "Goodput"], r.rows.map((row) => [
        row.id, state[row.status], row.visible_token_ms.length, fmt(row.ttft), fmt(row.tpot), fmt(row.maxITL), fmt(row.e2eVisible), fmt(row.terminalLatency),
        row.quality_pass === null ? t("未知", "unknown") : row.quality_pass ? t("标注通过", "marked pass") : t("标注未过", "marked fail"),
        row.good && row.completeInWindow ? t("计入", "counted") : !row.completeInWindow ? t("窗外终态", "terminal outside window") : row.status !== "success" ? state[row.status] : !row.latencyPass ? t("时延未达/不可评估", "latency fails/not evaluable") : t("品质未过/未知", "quality fails/unknown")
      ]), t("保留失败分母；N/A 不被写成0", "Failures remain in the denominator; N/A is not replaced with zero")) +
      note(r.closedWindow ? t("本例所有所选区间完整落在测窗内，因此面积L与λ×平均W相同。这是有限追踪恒等式，不证明生产稳态容量。", "All selected intervals are fully contained, so area L equals λ×mean W. This is a finite-trace identity, not proof of steady production capacity.") : t("输入区间不全部完整包含于测窗：边界截断/选样会使面积L和窗口终态λ×平均W不同。不要把边界偏差当成扩容结论。", "Not all intervals are fully contained: window boundaries/cohort selection can make area L differ from terminal-rate λ×mean W. Do not interpret boundary bias as a scaling conclusion.")) +
      note(t(`P99 样本只有 ${r.itlSamples} 个间隔，只用于演示定义，不可外推生产尾延迟。相同chunk时间产生0间隔，单用户速率在零均值或无间隔时为N/A。`, `P99 has only ${r.itlSamples} gap samples; it demonstrates a definition, not production tails. Equal chunk timestamps create zero gaps; per-user rate is N/A with zero mean or no intervals.`)) +
      (r.unknownAdmission ? note(t("部分 admission 时间未观测，Known admitted RPS 只是已知子集；切换 admitted 边界前必须补齐，不能填0。", "Some admission times are unobserved; Known admitted RPS is only the known subset. Fill them before choosing the admitted boundary; never insert zero.")) : "");
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
  function mixture(p) {
    const r = M.mixtureCapacity(p);
    let body = "";
    r.classes.forEach((group, i) => {
      const y = 35 + i * 85;
      const activeShare = r.active > 0 ? group.active / r.active : null;
      body += `<text x="15" y="${y + 22}">${esc(group.name)}</text><rect x="170" y="${y}" width="${480 * group.share}" height="25" rx="4" class="primary"/><text x="${180 + 480 * group.share}" y="${y + 19}">${fmt(group.share * 100)}%</text>`;
      if (activeShare !== null) body += `<rect x="170" y="${y + 33}" width="${480 * activeShare}" height="25" rx="4" class="secondary"/>`;
      body += `<text x="${180 + 480 * (activeShare || 0)}" y="${y + 52}">${activeShare === null ? "N/A" : fmt(activeShare * 100) + "%"}</text>`;
    });
    return kpis([[fmt(r.qps), t("含余量的混合 QPS", "Mixed QPS including headroom")], [fmt(r.inputTPS), "Input TPS"], [fmt(r.outputTPS), "Output TPS"], [fmt(r.meanResidence) + " s", t("按类正确求得 E[W]", "Correct class-weighted E[W]")], [fmt(r.active), t("同边界平均在途 L", "Same-boundary mean in-flight L")], [String(r.instances), t("同混合容量下实例下界", "Instance floor under same-mix capacity")]]) +
      legend([t("请求占比", "Request fraction"), t("平均在途占比", "Mean in-flight fraction")]) +
      svg(body, t("请求占比与在途占比是不同分母；长停留类主导并发。零总负载时在途份额为N/A。", "Request and in-flight fractions use different denominators; long residence dominates concurrency. In-flight fraction is N/A at zero total load."), 320) +
      dataTable([t("教学类别", "Teaching class"), "Share", "Input/Output", "W s", "Input TPS", "Output TPS", "L"], r.classes.map((group) => [group.name, fmt(group.share * 100) + "%", `${group.prompt}/${group.output}`, fmt(group.residence, 3), fmt(group.inputTPS), fmt(group.outputTPS), fmt(group.active, 3)]), t("固定教学均值；不是实测模型结果", "Fixed teaching means, not measured model results")) +
      dataTable([t("计算方式", "Calculation"), "W s", t("含义", "Meaning")], [
        ["Σ f_i W_i", fmt(r.meanResidence, 4), t("先在每类中计算停留，再平均", "Compute residence per class before averaging")],
        ["E[TTFT]+(E[N]−1)×E[TPOT]", fmt(r.incorrectShortcutW, 4), t("错误捷径：忽略长度/Token时间相关性", "Incorrect shortcut: ignores length/token-time correlation")],
        ["Input/Output/Concurrency floors", `${r.inputInstances}/${r.outputInstances}/${r.batchInstances}`, t("先取最大值，再做同负载验证", "Take the maximum, then validate the same workload")]
      ], t("容量规划必须保持同一总体", "Capacity planning must preserve the same population"));
  }
  function burst(p) {
    const r = M.burstQueue(p);
    const noDrain = t("无排空余量", "No drain headroom");
    return kpis([[fmt(r.burstRate) + " /s", t("突发到达率", "Burst arrival rate")], [fmt(r.peakQueue), t("无限队列峰值请求等价量", "Unbounded peak request-equivalents")], [fmt(r.peakWaitSeconds) + " s", t("流体峰值排队等待", "Fluid peak queue wait")], [r.drainSeconds === null ? noDrain : fmt(r.drainSeconds) + " s", t("突发后排空时间", "Post-burst drain time")], [fmt(r.cappedPeak), t("TTFT排队预算下的队列", "Queue under TTFT wait budget")], [fmt(r.rejectedDuringBurst), t("突发阶段需拒绝/转移等价量", "Burst request-equivalents to reject/route")]]) +
      plot([{ label: t("不限制接入", "Unbounded admission"), values: r.points.map((point) => ({ x: point.time, y: point.queue })) }, { label: t("按排队预算限制", "Queue-budget admission"), values: r.points.map((point) => ({ x: point.time, y: point.capped })) }], "Time seconds", "Fluid request-equivalents", t("流体队列时序；cap队列从较小峰值独立排空，不沿用无限积压的排空时间。", "Fluid queue timeline; capped queue drains independently from its smaller peak, not on the unbounded backlog's timeline."), r.queueCap, t("接入排队预算", "Admission queue budget")) +
      dataTable([t("策略", "Policy"), t("峰值等待 s", "Peak wait s"), t("恢复排空 s", "Recovery drain s")], [
        [t("无限排队", "Unbounded queue"), fmt(r.peakWaitSeconds), r.drainSeconds === null ? noDrain : fmt(r.drainSeconds)],
        [t("预算队列", "Budget-capped queue"), fmt(r.cappedWaitSeconds), r.cappedDrainSeconds === null ? noDrain : fmt(r.cappedDrainSeconds)]
      ], t("所有时间是固定μ下的教学近似，不是P99预测", "Times are teaching approximations at fixed μ, not P99 predictions")) +
      (!r.stableBase ? note(t("恢复后的基线不低于服务速率，没有稳定/恢复余量；正积压不会自行排空。流体零积压也不证明随机负载SLA。", "Post-burst base load meets/exceeds service rate: no stability/recovery headroom. Positive backlog cannot drain itself; even zero fluid backlog does not prove stochastic SLOs.")) : "");
  }
  function ptu(p) {
    const r = M.ptu(p);
    return kpis([[fmt(r.normalized, 0), "Normalized TPM"], [fmt(r.raw), t("未取整 PTU", "Raw PTU")], [String(r.provisioned), t("演示部署 PTU", "Illustrative deployed PTU")]]) +
      barChart([{ label: "Input equivalent", value: p.inputTPM * ((1 - p.cache) + p.cache * p.cachedWeight) }, { label: "Output weighted", value: p.outputTPM * p.outputWeight }], t("按手工权重归一化，不是当前云端 quota 或 GPU 数。", "Normalized using manual weights, not current cloud quota or GPU count."), "equiv TPM") +
      note(t("若缓存写入/多模态/上下文档位还有独立权重，这个两权重演示不足以报价；请使用当前模型专用公式。", "If cache writes, multimodal or context tiers have additional weights, this two-weight demo is insufficient for a quote; use current model-specific formulas."));
  }
  function normalized(p) {
    const r = M.normalizedPTU(p);
    const names = {
      shortInput: t("短普通输入", "Short ordinary input"), shortRead: t("短缓存读", "Short cached read"),
      shortWrite30: t("短30min写", "Short30min write"), shortWrite12: t("短12h写", "Short12h write"), shortOutput: t("短输出", "Short output"),
      longInput: t("长普通输入", "Long ordinary input"), longRead: t("长缓存读", "Long cached read"),
      longWrite30: t("长30min写", "Long30min write"), longWrite12: t("长12h写", "Long12h write"), longOutput: t("长输出", "Long output")
    };
    return kpis([[fmt(r.rawTokens, 0), t("互斥分类原始 TPM", "Raw TPM across exclusive categories")], [fmt(r.normalized, 0), "Normalized TPM"], [fmt(r.raw), t("未取整 PTU", "Raw PTU")], [String(r.provisioned), t("当前分类的规划 PTU", "Planned PTU for these categories")], [`${r.minimum} / ${r.step}`, t("最小值 / 扩容步长", "Minimum / scale increment")], [String(r.incorrectFreeReads), t("错误免费缓存对照，非正确方案", "Incorrect free-cache counterfactual, not a valid plan")]]) +
      barChart(r.categories.map((category) => ({ label: names[category.key], value: category.normalized })), t("每个Token只属于一个分类；条形为TPM×已复核权重，零类别是明确输入，不是未知值。", "Each token belongs to one category; bars show TPM×reviewed weight. Zero categories are explicit inputs, not unknowns."), "norm TPM") +
      dataTable([t("互斥分类", "Exclusive category"), "Raw TPM", "Weight", "Normalized TPM"], r.categories.map((category) => [names[category.key], fmt(category.tokens, 0), fmt(category.weight), fmt(category.normalized, 0)]), `${r.model.name} / ${r.model.version} · ${fmt(r.model.capacity, 0)} normalized TPM/PTU`) +
      note(t(`这是2026-11-21官方参数快照与用户输入的预算计算，没有读取Azure订阅quota或capacity。缓存读本身贡献 ${fmt(r.readContribution, 0)} normalized TPM，不能在这个模型版本中默认免费。`, `This combines a2026-11-21 published parameter snapshot with entered demand, without reading Azure quota/capacity. Cache reads contribute ${fmt(r.readContribution, 0)} normalized TPM and are not free for this model version.`));
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
  function rankMemory(p) {
    const r = M.rankMemory(p);
    const shown = r.stages.slice(0, 8);
    const max = Math.max(r.deviceGB, ...r.stages.map((stage) => stage.usedGB)) * 1.1;
    const y = (value) => 245 - value / max * 190;
    const step = 650 / shown.length;
    const width = Math.min(100, step - 20);
    let body = `<text x="20" y="25">${t("每 stage 代表 rank 的预算 GB", "Representative rank budget per stage, GB")}</text>`;
    shown.forEach((stage, i) => {
      const x = 75 + i * step + (step - width) / 2;
      let used = 0;
      [[stage.weightGB, "secondary"], [stage.kvGB, "primary"], [stage.runtimeGB, "tertiary"]].forEach(([value, cls]) => {
        body += `<rect x="${x}" y="${y(used + value)}" width="${width}" height="${value / max * 190}" class="${cls}"><title>${fmt(value)} GB</title></rect>`;
        used += value;
      });
      body += `<rect x="${x - 3}" y="${y(r.deviceGB)}" width="${width + 6}" height="${r.deviceGB / max * 190}" class="line axis"/><path d="M ${x - 6} ${y(r.deviceGB * p.safe)} H ${x + width + 6}" class="grid-line"/><text x="${x + width / 2}" y="274" text-anchor="middle">stage ${stage.stage}</text><text x="${x + width / 2}" y="${Math.max(43, y(used) - 8)}" text-anchor="middle">${fmt(used, 1)}</text>`;
    });
    body += `<text x="20" y="310">${t("外框=总可见容量；虚线=安全水位；复制副本不等于原始参数变小", "Outline=visible total capacity; dashed=safe margin; replicas do not shrink total parameters")}</text>`;
    return kpis([[`${p.tp}×${p.pp}×${p.dp}=${r.cards}`, "TP × PP × replica DP"], [fmt(r.worst.usedGB) + " GB", t("最忙 rank 预算", "Worst-rank budget")], [fmt(r.worst.marginGB) + " GB", t("距安全水位余量，可为负", "Safe-margin headroom, possibly negative")], [r.fits ? t("满足当前内存预算", "Passes this memory budget") : t("超过当前内存水位", "Exceeds this memory budget"), t("非运行/SLA 承诺", "Not a runtime/SLO promise")], [fmt(r.kvReserveGB) + " GB", t("全组 KV 预留", "Group KV reservation")], [fmt(r.effectiveKVFactor) + "×", t("复制/布局/副本取整综合倍率", "Replication/layout/replica-rounding factor")]]) +
      legend([t("每 rank KV", "KV per rank"), t("每 rank 权重预算", "Weight budget per rank"), t("每 rank runtime", "Runtime per rank")]) +
      svg(body, t(`每 stage 只画一个代表 rank，非全组总占用；显示 ${shown.length}/${r.stages.length} stages，最坏值取全部 stages。`, `One representative rank per stage, not total group usage; ${shown.length}/${r.stages.length} stages shown, worst case uses all stages.`), 345) +
      dataTable(["Stage", t("层数", "Layers"), "W GB/rank", "KV GB/rank", "Runtime GB/rank", "Total GB/rank", t("余量 GB", "Margin GB")], r.stages.map((stage) => [stage.stage, stage.layers, fmt(stage.weightGB), fmt(stage.kvGB), fmt(stage.runtimeGB), fmt(stage.usedGB), fmt(stage.marginGB)]), `${r.model.name} · ${r.gpu.name} · ${t("仅内存近似", "memory approximation only")}`) +
      note(t(`每副本活动请求按 ${r.replicaBatch} 预算；全组保留 ${r.reservedRequestSlots} 请求槽，不等于观测到的 ${p.batch} 个在途请求。每 rank 每层 KV ${fmt(r.bytesPerLayer, 0)} B/Token。${r.model.latent ? "MLA 是 packed latent/RoPE，不用普通双 K/V heads 公式。" : `普通 GQA 每 rank ${r.kvHeadsPerRank} KV head。`}`, `Each replica is budgeted for ${r.replicaBatch} active requests, reserving ${r.reservedRequestSlots} group slots, not an observation of ${p.batch} in-flight requests. KV is ${fmt(r.bytesPerLayer, 0)} B/token/layer/rank. ${r.model.latent ? "MLA uses packed latent/RoPE, not the ordinary two-K/V-head formula." : `Ordinary GQA has ${r.kvHeadsPerRank} KV heads/rank.`}`)) +
      (p.measuredMemory || (r.model.latent && p.mlaCache === "measured") ? note(t("“实测”字段是用户录入模式：默认值仍是教学示例，本页面不会检测 GPU 或自动取得真实测量。MLA 自定义 row bytes 对内存计算优先于参考 dtype。", "Measured fields are user-entry modes: defaults remain teaching examples; this page neither detects GPUs nor obtains measurements. Custom MLA row bytes override reference dtype for memory.")) : "") +
      (p.pp > 1 ? note(t("PP stage 的冷 Prefill/Decode 仍有串行路径与流水 bubble；此内存结果不能套用理想 N×HBM 带宽来承诺单请求 TPOT。", "PP stages retain serial prefill/decode paths and pipeline bubbles; this memory result cannot turn ideal N×HBM bandwidth into a single-request TPOT promise.")) : "") +
      (r.gpu.nvlink === 0 && p.tp > 1 ? note(t("选定 GPU 无 NVLink；跨卡 TP 的 PCIe/RDMA 通信成本必须独立验证。", "The selected GPU has no NVLink; independently validate PCIe/RDMA costs for multi-GPU TP.")) : "") +
      (!r.contextSupported ? note(t("上下文超过原始模型范围，内存通过也不代表模型支持。", "Context exceeds the stock model range; memory fit does not prove model support.")) : "");
  }
  function pd(p) {
    const r = M.transfer(p);
    const lengths = [1024, 4096, 8192, 16384, 32768, 65536, 131072];
    const points = lengths.map((context) => ({ context, ...M.pdSamples({ ...p, context }) }));
    const labels = { colocated: t("共池", "Colocated"), chunked: "Chunked Prefill", pd: t("PD（残留干扰假设）", "PD (residual assumption)") };
    const series = Object.entries(labels).map(([key, label], index) => ({ label, colorClass: ["primary", "secondary", "tertiary"][index], values: points.map((v) => ({ x: v.context, y: v.p99[key] })) }));
    const tpotSeries = Object.entries(labels).map(([key, label], index) => ({ label, colorClass: ["primary", "secondary", "tertiary"][index], values: points.map((v) => ({ x: v.context, y: v.p99TPOT[key] })) }));
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
    return kpis([[fmt(r.logicalBytes / 1e9) + " GB", t("新的唯一逻辑 KV", "New unique logical KV")], [fmt(r.bytes / 1e9) + " GB", t("实际线量预算，含传输倍率", "Wire budget including multiplier")], [fmt(r.rawMs) + " ms", t("原始传输时间", "Raw transfer time")], [fmt(r.exposedMs) + " ms", t("暴露交接时间", "Exposed handoff")], [fmt(r.utilization * 100) + "%", t("单方向 fabric 负载", "One-way fabric load")], [fmt(r.requiredGbps) + " Gb/s", t("含效率的总线速需求", "Aggregate line-rate need incl. efficiency")], ["200 / 6200", t("每点模拟请求 / 间隔样本", "Synthetic requests / gaps per point")]]) +
      (r.utilization >= 1 ? note(t("警告：KV 到达量不低于有效带宽，持续队列不稳定。下面曲线不预测此时的真实 P99；增加带宽、减少交接量或限制接入。", "Warning: KV arrival meets/exceeds effective bandwidth; sustained queues are unstable. Curves do not predict real P99 under this overload. Add bandwidth, reduce handoff or control admission.")) : "") +
      svg(timeline, t("混部阻塞与分池流水线示意。", "Colocated stalls versus separate-pool pipeline illustration."), 280) +
      plot(p.hideColocated ? tpotSeries.slice(1) : tpotSeries, "Prompt tokens", "Synthetic request P99 TPOT ms", t("每点200个32输出Token请求，先求每请求31间隔平均TPOT，再取请求P99；不是P99 ITL。", "Each point has200 requests with32 outputs: average31 gaps per request, then take request-P99 TPOT; not P99 ITL.") + (p.hideColocated ? t(" 此视图省略混部，数值仍在表格。", " Colocated is omitted from this view, with values retained in the table.") : "")) +
      plot(p.hideColocated ? series.slice(1) : series, "Prompt tokens", "Synthetic pooled P99 ITL ms", t("同一批模拟请求的6200个后续间隔P99；2%请求有额外突发干扰。假设首字已从D可见，不含启动交接，非实测。", "P99 over6200 gaps from the same simulated requests;2% have extra bursts. Assumes first D-visible output, excludes startup handoff and is not measured.") + (p.hideColocated ? t(" 放大两条优化路径。", " Zoomed to the two optimized paths.") : "")) +
      dataTable([t("当前上下文", "Current context"), "P99 request TPOT ms", "P99 pooled ITL ms", t("分母", "Denominators")], Object.entries(labels).map(([key, label]) => [label, fmt(current.p99TPOT[key]), fmt(current.p99[key]), "200 requests / 6200 gaps"]), `${p.context} tokens · synthetic`);
  }
  function pdBudget(p) {
    const source = document.querySelector('form[data-lab="pd"]');
    if (!source.checkValidity()) throw new RangeError(t("请先修正上方传输沙盘输入", "Fix the transfer-workbench inputs above first"));
    const combined = { ...inputValues(source), ...p };
    const r = M.pdBudget(combined);
    const readiness = {
      "keep-baseline": t("保留混部基线，非必须分离", "Keep colocated baseline; PD is not mandatory"),
      "blocked-layout": t("先修兼容/内存/上下文条件", "Resolve compatibility/memory/context first"),
      "blocked-capacity": t("容量不稳定，不能接入该负载", "Capacity unstable at this demand"),
      "blocked-latency": t("首字或首次间隔预算不满足", "First-token/first-gap budget fails"),
      "validate-headroom": t("余量不足，先验证峰值与容灾", "Insufficient headroom; validate peak/failover"),
      "trial-candidate": t("可进入真实A/B，非生产批准", "Candidate for real A/B, not production approval")
    };
    const limits = r.limits.map((limit) => [limit.key, fmt(limit.rps), fmt(combined.qps / limit.rps * 100) + "%"]);
    if (!r.tx.bytes) limits.push(["KV network", t("无新线量，不构成限制", "No new wire bytes; not limiting"), "N/A"]);
    const latency = p.firstVisible === "decode" ? [
      ["P queue", p.queueP], ["P compute", p.prefillBudget], ["KV handoff", r.tx.exposedMs], ["D queue", p.queueD], ["First sample", p.firstBudget], ["Delivery", p.delivery]
    ] : [["P queue", p.queueP], ["P compute", p.prefillBudget], ["P first sample", p.firstBudget], ["Delivery", p.delivery]];
    return kpis([[readiness[r.readiness], t("仅规划检查，不是实测验证", "Planning check, not a measurement")], [fmt(r.bottleneck.rps) + " req/s", `${t("最小容量约束", "Minimum capacity constraint")}: ${r.bottleneck.key}`], [fmt(r.candidateOutputTPS) + " tok/s", t("候选系统输出上界，非Goodput", "Candidate system-output bound, not goodput")], [fmt(r.ttftBudget) + " ms", t("首可见Token工程预算", "First-visible-token engineering budget")], [r.firstGapBudget === null ? "N/A" : fmt(r.firstGapBudget) + " ms", t("首字后首次间隔预算", "First-to-second gap budget")], [fmt(r.fullStateBytes / 1e9) + " GB", t("每请求D引用峰值KV状态，非新增分配", "D peak referenced KV/request, not new allocation")]]).replace('<div class="kpis">', `<div class="kpis" data-pd-readiness="${r.readiness}">`) +
      barChart(r.limits.map((limit) => ({ label: limit.key, value: limit.rps })), t("P输入、D输出、D驻留槽与网络上界取最小，未用aggregate TPS推首字时间。", "Take the minimum P-input/D-output/D-slot/network bound; aggregate TPS does not determine first-token latency."), "req/s") +
      dataTable([t("容量约束", "Capacity constraint"), "Upper req/s", t("目标负载占比", "Target load fraction")], limits, t("同一个传输案例与目标QPS，余量不是通用SLA", "Same transfer case/QPS; headroom is not a universal SLA")) +
      dataTable([t("首字阶段预算", "First-token stage budget"), "ms"], latency.map(([label, value]) => [label, fmt(value)]), t("工程预算之和，不是阶段P99相加", "Engineering budget sum, not summed stage P99s")) +
      dataTable([t("检查", "Check"), t("本例条件", "Case condition")], [
        [t("模型/布局/每rank内存/上下文", "Model/layout/rank-memory/context"), r.flags.compatibility ? t("按输入假设满足", "Passes entered assumptions") : t("不满足", "Fails")],
        [t("目标QPS低于每个容量上界", "Demand below every capacity bound"), r.flags.capacityStable ? t("满足模型条件", "Passes model condition") : t("不稳定", "Unstable")],
        [t("规划容量余量", "Planning capacity headroom"), r.flags.headroom ? t("满足自选余量", "Passes chosen margin") : t("需额外验证", "Needs further validation")],
        [t("TTFT与首次间隔", "TTFT and first gap"), r.flags.latency ? t("工程预算满足", "Engineering budget passes") : t("工程预算失败", "Engineering budget fails")],
        [t("P+D主权重预算", "P+D main-weight budget"), fmt(r.weightsGB) + " GB"]
      ], t("确认框也是教学假设，不会自动检查部署", "Confirmation checkboxes are teaching assumptions, not deployment checks")) +
      note(t("D缓存命中只在这里减少新线量，不自动减少P计算预算或完整引用状态。每rankKV复制可通过D侧本地广播实现，不能直接拿状态倍率乘跨网流量。完整实例数只改变权重预算；池TPS是用户提供的全池约束，不能未经测量自动线性扩容。真正Goodput/质量/成本要用同负载端到端A/B取得。", "D cache hits reduce new wire bytes here, not automatically P compute or full referenced state. Rank KV copies may use D-local broadcast; do not blindly multiply network traffic by state-copy count. Replica counts change weight budget; whole-pool TPS bounds must not be automatically scaled without measurements. Obtain actual goodput/quality/cost from same-workload end-to-end A/B."));
  }
  function parallel(p) {
    M.integer(p.ranks, "ranks", 2, 8);
    M.integer(p.batch, "microbatches", 1, 128);
    M.number(p.payload, "payload", 0.001);
    M.number(p.bandwidth, "bandwidth", 0.001);
    const n = p.ranks;
    const pipe = M.pipeline(p);
    const transferMs = p.payload / p.bandwidth;
    const node = (x, y, width, lines, cls = "surface") =>
      `<rect x="${x}" y="${y}" width="${width}" height="58" rx="8" class="${cls}"/>` +
      lines.map((line, i) => `<text x="${x + width / 2}" y="${y + 23 + i * 20}" text-anchor="middle" class="${cls === "primary" ? "inverse" : "label"}">${esc(line)}</text>`).join("");
    const connect = (x1, y1, x2, y2, cls = "secondary") =>
      `<path d="M ${x1} ${y1} C ${(x1 + x2) / 2} ${y1}, ${(x1 + x2) / 2} ${y2}, ${x2} ${y2}" class="line ${cls}"/><path d="M ${x2 - 6} ${y2 - 4} L ${x2} ${y2} L ${x2 - 6} ${y2 + 4}" class="line ${cls}"/>`;
    let body = "";
    let height;
    let caption;
    if (p.kind === "tp") {
      height = n * 70 + 145;
      const middle = 55 + n * 35;
      body = `<text x="15" y="25">TP / two-linear MLP: column-up, row-down, then sum</text>`;
      body += node(15, middle - 29, 140, ["Same input X", "[B, H]"], "primary");
      body += node(655, middle - 29, 135, ["AllReduce ΣZ_i", "Output [B,H]"]);
      for (let i = 0; i < n; i += 1) {
        const y = 55 + i * 70;
        body += node(215, y, 215, [`rank ${i}: W_up_i [H,F/${n}]`, `W_down_i [F/${n},H]`]);
        body += node(475, y, 140, [`Partial Z_${i}`, "[B,H]"]);
        body += connect(155, middle, 215, y + 29) + connect(430, y + 29, 475, y + 29) + connect(615, y + 29, 655, middle);
      }
      body += `<text x="15" y="${height - 28}">Z = Σ GELU(X W_up_i) W_down_i · toy FFN shapes, not measured traffic</text>`;
      caption = t("同一个 X 进入所有 ranks；每卡持有列切/行切权重，产生同形状部分结果再求和。忽略 gated FFN 细节，图宽不编码通信量。", "The same X reaches all ranks; column/row weight shards produce same-shaped partials that are summed. Gated-FFN details are omitted; widths do not encode traffic.");
    } else if (p.kind === "pp") {
      const shown = Math.min(p.batch, 8);
      const shownSlots = shown + n - 1;
      const cellWidth = 640 / shownSlots;
      height = 240 + n * 42;
      body = `<text x="15" y="25">PP / one request follows the complete serial layer path</text>`;
      const stageWidth = 710 / n;
      for (let i = 0; i < n; i += 1) {
        const x = 50 + i * stageWidth;
        body += node(x, 52, stageWidth - 12, [`stage ${i}`, "layer group"]);
        if (i < n - 1) body += connect(x + stageWidth - 12, 81, x + stageWidth, 81);
      }
      body += `<text x="15" y="153">Forward-only balanced pipeline · ${shown}/${p.batch} microbatches shown</text>`;
      for (let stage = 0; stage < n; stage += 1) {
        const y = 175 + stage * 42;
        body += `<text x="15" y="${y + 25}">stage ${stage}</text>`;
        for (let slot = 0; slot < shownSlots; slot += 1) {
          const microbatch = slot - stage;
          const active = microbatch >= 0 && microbatch < shown;
          const x = 130 + slot * cellWidth;
          body += `<rect x="${x}" y="${y}" width="${cellWidth - 4}" height="32" rx="4"${active ? ` data-microbatch="${microbatch}" data-stage="${stage}"` : ""} class="${active ? (microbatch === 0 ? "primary" : "secondary") : "surface"}"/>`;
          if (active) body += `<text x="${x + (cellWidth - 4) / 2}" y="${y + 22}" text-anchor="middle" class="inverse">m${microbatch + 1}</text>`;
        }
      }
      body += `<text x="15" y="${height - 28}">Full schedule: ${pipe.slots} slots; one-request path: ${pipe.latencySlots} stage slots</text>`;
      caption = t(`完整 ${p.batch} 个 microbatch 的理想利用率 ${fmt(pipe.utilization * 100)}%；只显示前 ${shown} 个的填充/排空。stage 等成本假设不预测不等长生产负载。`, `Ideal utilization for all ${p.batch} microbatches is ${fmt(pipe.utilization * 100)}%; only the first ${shown} show fill/drain. Equal stage cost does not predict mixed-length production traffic.`);
    } else if (p.kind === "dp") {
      height = n * 70 + (p.attentionDP ? 220 : 140);
      const middle = 55 + n * 35;
      body = `<text x="15" y="25">${p.attentionDP ? "Attention DP / request shards with a shared EP group" : "Inference DP / independent complete-weight replicas"}</text>`;
      body += node(15, middle - 29, 140, ["Request router", "R0...Rk"], "primary");
      for (let i = 0; i < n; i += 1) {
        const y = 55 + i * 70;
        body += node(225, y, 220, [`rank ${i}: requests shard`, p.attentionDP ? "attention weights / work" : "complete model W copy"]);
        body += node(525, y, 250, [p.attentionDP ? `Attention hidden / owner ${i}` : `Independent output / owner ${i}`, p.attentionDP ? "to shared expert group" : "no gradient AllReduce"]);
        body += connect(155, middle, 225, y + 29) + connect(445, y + 29, 525, y + 29);
      }
      if (p.attentionDP) {
        const y = 75 + n * 70;
        body += node(225, y, 550, ["Shared EP domain: dispatch → experts → combine", "Return hidden states to their request owners"]);
        for (let i = 0; i < n; i += 1) body += `<path d="M 775 ${84 + i * 70} H 790 V ${y + 29} H 775" class="line secondary"/>`;
      }
      caption = p.attentionDP ? t("Attention 请求分片送入共享专家组，并行组语义要按引擎确认；不能误画成每 rank 完整复制所有 MoE 专家。", "Attention request shards feed a shared expert group; confirm engine group semantics rather than drawing full MoE-expert copies per rank.") : t("普通 DP 把不同请求交给独立完整副本，不对推理输出做 AllReduce；系统 TPS 扩容不等于单请求变快。", "Ordinary DP sends different requests to complete replicas with no output AllReduce; system scale-out does not accelerate one request.");
    } else if (p.kind === "ep") {
      height = n * 70 + 145;
      body = `<text x="15" y="25">EP / toy 64 experts, top-2 routing · real DeepSeek uses top-8</text>`;
      const shards = Array.from({ length: n }, (_, i) => ({ start: Math.floor(i * 64 / n), end: Math.floor((i + 1) * 64 / n) - 1 }));
      const tokens = [[0, 17], [17, 33], [33, 49], [0, 49]];
      shards.forEach((shard, i) => { body += node(310, 55 + i * 70, 220, [`rank ${i}: expert shard`, `E${shard.start}...E${shard.end}`]); });
      tokens.forEach((experts, i) => {
        const y = 55 + i * Math.max(70, (n - 1) * 70 / 3);
        body += node(15, y, 170, [`Token ${String.fromCharCode(65 + i)}`, `route E${experts[0]}, E${experts[1]}`], "primary");
        body += node(615, y, 170, [`Owner ${String.fromCharCode(65 + i)}`, "weighted combine"]);
        for (const expert of experts) {
          const owner = shards.findIndex((shard) => expert >= shard.start && expert <= shard.end);
          const targetY = 84 + owner * 70;
          body += connect(185, y + 29, 310, targetY) + connect(530, targetY, 615, y + 29);
        }
      });
      height = Math.max(height, 4 * 70 + 145);
      caption = t("固定 64 专家的玩具 top-2 路由展示 dispatch 与 combine，不假装每 Token 访问全部 ranks；部分路由可留在同 rank。专家可整除/分组要求仍以真实 kernel 为准。", "A fixed 64-expert toy top-2 route shows dispatch/combine without every token visiting every rank; some routes are rank-local. Real kernels determine expert divisibility/grouping requirements.");
    } else throw new RangeError("Unknown parallel kind");
    return kpis([[String(n), `${p.kind.toUpperCase()} ranks`], [fmt(transferMs) + " ms", t("单步骤裸传输估算", "Bare transfer estimate/step")], [p.kind === "pp" ? fmt(pipe.utilization * 100) + "%" : "N/A", t("平衡 PP 理想流水利用率", "Ideal balanced-PP utilization")]]) +
      svg(body, caption, height).replace("<svg ", `<svg data-parallel-kind="${p.kind}" data-attention-dp="${Boolean(p.attentionDP)}" `) +
      note(t("图解不是实测缩放率；切分合法性、层数、heads、专家容量和通信重叠必须由实际引擎配置确认。", "The diagram is not measured scaling. Validate partition legality, layers, heads, expert capacity and overlap in the actual engine."));
  }
  function speculative(p) {
    const r = M.speculative(p);
    const acceptanceSweep = Array.from({ length: 21 }, (_, i) => ({ x: i * 5, y: M.speculative({ ...p, acceptance: i / 20 }).speedup }));
    return kpis([[fmt(r.expectedTokens, 4), t("每轮期望提交 Token", "Expected committed tokens/cycle")], [fmt(r.cycleMs) + " ms", t("完整 cycle 成本", "Whole-cycle cost")], [fmt(r.speedup) + "×", t("条件性平均服务加速", "Conditional mean-service speedup")], [fmt(r.meanServiceMs) + " ms", t("平均服务/提交 Token，非 P99 ITL", "Mean service/committed token, not P99 ITL")], [fmt(r.userTPS) + " tok/s", t("条件性单流速率", "Conditional single-stream rate")], ["i.i.d.", t("接受率假设", "Acceptance assumption")]]) +
      barChart(r.probabilities.map((value, i) => ({ label: `P[N=${i + 1}]`, value: value * 100 })), t("每轮提交长度的几何分布：最后一桶包含全部候选接受后的 bonus。坐标来自公式，非测量。", "Geometric commit-length distribution: final bucket includes the all-accepted bonus. Coordinates are formula-derived, not measured."), "%") +
      plot([{ label: t("条件性加速倍数", "Conditional speedup"), values: acceptanceSweep }], "Acceptance %", "Speedup ×", t("固定成本假设下扫接受率；1× 以下表示额外成本压过收益。", "Acceptance sweep with fixed cost assumptions; below 1× means overhead outweighs benefit."), 1, t("收支平衡", "Break-even")) +
      dataTable([t("成本组成", "Cycle component"), "ms"], [
        ["Serial draft", fmt(p.draftTokens * p.draftMs)], ["Target verification", fmt(r.verificationMs)], ["Commit/scheduler", fmt(p.overheadMs)], ["Whole cycle", fmt(r.cycleMs)]
      ], t("每轮一次完整计费，不把 draft 开销漏掉", "Count each full cycle once, including draft cost"));
  }
  function prefixCaching(p) {
    const r = M.prefixCaching(p);
    return kpis([[fmt(r.avoidedTokens), t("平均避免输入计算 Token", "Mean avoided input-work tokens")], [fmt(r.inputComputeTokens), t("仍需输入计算 Token", "Remaining input-work tokens")], [fmt(r.expectedMs) + " ms", t("平均 Prefill+lookup（假设）", "Mean prefill+lookup (assumption)")], [fmt(r.savedMs) + " ms", t("净节约，可为负", "Net saving, possibly negative")], [fmt(r.cacheGB) + " GB", t("独立前缀条目 KV 预算", "Independent prefix-entry KV budget")], [fmt(p.hitRate * p.prefix / p.prompt * 100) + "%", t("实际平均避免的输入比例", "Mean avoided-input fraction")]]) +
      barChart([{ label: "No cache", value: r.baselineMs }, { label: "Hit + lookup", value: r.hitMs }, { label: "Miss + lookup", value: r.missMs }, { label: "Weighted mean", value: r.expectedMs }], t("命中、未命中与加权平均分开；70% 请求命中不等于70%输入都免费。", "Separate hit, miss and weighted mean; 70% request hits do not make 70% of all input free."), "ms") +
      dataTable([t("条件", "Case"), t("概率", "Probability"), t("输入计算 Token", "Input-work tokens"), t("Prefill+lookup ms", "Prefill+lookup ms")], [
        ["Hit", fmt(p.hitRate * 100) + "%", p.prompt - p.prefix, fmt(r.hitMs)], ["Miss", fmt((1 - p.hitRate) * 100) + "%", p.prompt, fmt(r.missMs)], ["Mean", "100%", fmt(r.inputComputeTokens), fmt(r.expectedMs)]
      ], t("相同 Prompt 形状下的线性教学近似", "Linear teaching approximation for the same prompt shape")) +
      (!r.contextSupported ? note(t("Prompt 超过该模型原始配置上下文；缓存不会赋予长上下文能力。", "Prompt exceeds the stock model context; caching does not confer longer-context capability.")) : "");
  }
  function network(p) {
    const r = M.collective(p);
    M.integer(p.rails, "rails", 1, 16);
    M.number(p.rate, "rate", 0.001);
    M.number(p.efficiency, "efficiency", 0.01, 1);
    const budget = p.rate / 8 * p.rails * p.efficiency;
    return kpis([[fmt(r.algbw) + " GB/s", "algbw"], [fmt(r.busbw) + " GB/s", "busbw"], [fmt(r.factor, 4) + "×", t("标准化修正系数", "Normalization factor")], [fmt(budget) + " GB/s", t("所选单向payload预算", "Selected one-way payload budget")], [fmt(r.effectiveSizeMB) + " MB", t("实际计算的总S", "Effective total S")], [fmt(r.sizeMultiplier) + "×", t("API count到S倍率", "API-count to S multiplier")]]).replace('<div class="kpis">', `<div class="kpis" data-effective-size="${r.effectiveSizeMB}">`) +
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
  function networkBudget(p) {
    const r = M.networkBudget(p);
    return kpis([[fmt(r.portPayloadGBps) + " GB/s", t("单端口单向payload", "One-port one-way payload")], [fmt(r.gpuPCIe.payloadGBps) + " GB/s", t("单GPU PCIe端点payload", "One-GPU PCIe endpoint payload")], [fmt(r.nicPCIe.payloadGBps) + " GB/s", t("单NIC卡PCIe payload", "One-NIC-card PCIe payload")], [fmt(r.nodeGBps) + " GB/s", `${t("节点路径上界", "Node path bound")}: ${r.bottleneck.key}`], [fmt(r.stripedTransferMs) + " ms", t("完全条带传输下界，非实测", "Fully striped transfer floor, not measured")], [r.nvlinkOneWayGBps === null ? "N/A" : fmt(r.nvlinkOneWayGBps) + " GB/s", t("GPU NVLink单向聚合规格，非此网络路径", "GPU NVLink one-way aggregate spec, not this network path")]]) +
      barChart(r.limits.map((limit) => ({ label: limit.key, value: limit.valueGBps })), t("只对实际参与/经过的资源取最小；端口、NIC卡与GPU端点是不同数量。", "Take the minimum only across resources used/crossed; ports, NIC cards and GPU endpoints are distinct."), "GB/s") +
      dataTable([t("口径", "Scope"), "GB/s", t("限制", "Limit")], [
        ["GPU PCIe encoded one-way", fmt(r.gpuPCIe.encodedGBps), t("Gen3/4/5只扣线编码，尚未扣packet开销", "Gen3/4/5 coding only, before packet overhead")],
        ["NIC PCIe encoded one-way", fmt(r.nicPCIe.encodedGBps), t("按实际协商代际/lanes输入", "Use actual negotiated generation/lanes")],
        ["PCIe payload fraction", fmt(p.pcieEfficiency), t("额外TLP/实现效率假设", "Extra TLP/implementation efficiency assumption")],
        ["NVLink advertised bidirectional", r.gpu.nvlink === 0 ? "N/A" : fmt(r.gpu.nvlink), t("每GPU聚合规格，不是单流独享", "Per-GPU aggregate specification, not dedicated per flow")],
        ["Modeled payload/GPU", fmt(r.perGpuGBps), t("全条带均衡假设下的平均，非实际分配", "Mean under fully balanced striping, not actual allocation")]
      ], t("统一单位之后仍须统一方向与路径", "After units, align direction and path")) +
      note(t("GPU型号不决定此处PCIe协商值；所有payload效率、共享link和fabric预算是输入假设，须用两端实测较小值替换。仅单向线量用于KV传输，不把双向规格相加后给单向流使用。", "GPU branding does not determine these negotiated PCIe values. Payload efficiencies/shared-link/fabric budgets are assumptions; replace with smaller endpoint measurements. One-way KV traffic cannot use the sum of bidirectional specifications."));
  }
  function rail(p) {
    const r = M.railBudget(p);
    const height = Math.max(p.gpus, p.railCount) * 52 + 120;
    const span = height - 150;
    let body = `<text x="15" y="25">GPU outgoing payload</text><text x="345" y="25">NIC / rail demand (not measured)</text>`;
    r.rails.forEach((item, i) => {
      const y = 48 + (p.railCount === 1 ? span / 2 : i * span / (p.railCount - 1));
      body += `<rect x="345" y="${y}" width="230" height="42" rx="8" class="surface"/><text x="460" y="${y + 17}" text-anchor="middle" class="label">rail ${item.rail}${item.failed ? " · FAILED" : ""}</text><text x="460" y="${y + 34}" text-anchor="middle" class="label">${fmt(item.payloadGB)} GB planned</text>`;
    });
    r.assignments.forEach((item, i) => {
      const y = 48 + (p.gpus === 1 ? span / 2 : i * span / (p.gpus - 1));
      body += `<rect x="15" y="${y}" width="160" height="42" rx="8" class="${item.blocked ? "surface" : "primary"}"/><text x="95" y="${y + 18}" text-anchor="middle" class="${item.blocked ? "label" : "inverse"}">GPU ${item.gpu} · ${fmt(item.payloadGB)} GB</text><text x="95" y="${y + 34}" text-anchor="middle" class="${item.blocked ? "label" : "inverse"}">${item.blocked ? "BLOCKED" : "assigned"}</text>`;
      if (item.rail !== null) {
        const target = 69 + (p.railCount === 1 ? span / 2 : item.rail * span / (p.railCount - 1));
        body += `<path d="M 175 ${y + 21} C 260 ${y + 21}, 260 ${target}, 340 ${target}" class="line ${item.blocked ? "tertiary" : "secondary"}"/>`;
      }
    });
    body += `<text x="610" y="80">${r.lowerBoundMs === null ? "INCOMPLETE" : fmt(r.lowerBoundMs) + " ms floor"}</text><text x="610" y="108">max(rail, fabric)</text><text x="15" y="${height - 22}">Toy mapping only · no receiver / PCIe / route-search / recovery simulation</text>`;
    return kpis([[fmt(r.totalGB) + " GB", t("总计划payload", "Total planned payload")], [fmt(r.railGBps) + " GB/s", t("每存活rail有效带宽假设", "Payload assumption per live rail")], [r.lowerBoundMs === null ? t("完整传输无法完成", "Whole transfer cannot complete") : fmt(r.lowerBoundMs) + " ms", t("瓶颈传输下界，非P99", "Bottleneck transfer floor, not P99")], [fmt(r.effectiveGBps) + (r.effectiveGBps === null ? "" : " GB/s"), t("模型等效总带宽", "Modeled effective aggregate bandwidth")], [fmt(r.blockedGB) + " GB", t("无有效路径的计划流量", "Planned bytes without a valid path")], [String(r.rails.filter((item) => !item.failed && item.payloadGB > 0).length), t("实际有负载的存活rails", "Live rails carrying assigned load")]]).replace('<div class="kpis">', `<div class="kpis" data-rail-complete="${r.lowerBoundMs !== null}">`) +
      svg(body, t("均衡、碰撞与失效重映射改变每rail负载；失败payload保留在分母中，不伪装成功吞吐。", "Balanced/colliding/failure remapping changes rail demand; failed payload remains in the denominator rather than becoming successful throughput."), height) +
      barChart(r.rails.map((item) => ({ label: `Rail ${item.rail}${item.failed ? " FAILED" : ""}`, value: item.payloadGB })), t("这是计划需求，不是故障rail已完成的流量。", "Planned demand, not completed traffic on a failed rail."), "GB") +
      dataTable(["Rail", t("分配GPU", "Assigned GPUs"), "Planned GB", t("路径", "Path")], r.rails.map((item) => [item.rail, item.gpus.join(", ") || "—", fmt(item.payloadGB), item.failed ? t("故障，不能完成", "Failed, cannot complete") : t("存活（未实测）", "Live (not measured)")]), t("热点、上联与实际routing仍需验证", "Validate hotspots, uplinks and actual routing"));
  }
  const pauseStates = new WeakMap();
  function congestion(p, form) {
    const r = M.congestion(p, pauseStates.get(form) || false);
    const headroom = M.pfcBudget(p);
    pauseStates.set(form, r.paused);
    let body = `<rect x="60" y="60" width="680" height="45" rx="6" class="surface"/><rect x="60" y="60" width="${680 * p.queue / 100}" height="45" class="${r.paused ? "tertiary" : "primary"}"/>`;
    [[p.ecn, "ECN"], [p.pfc, "Xoff"], [p.xon, "Xon"]].forEach(([value, label], i) => {
      const x = 60 + 680 * value / 100;
      const ty = 137 + i * 24;
      body += `<path d="M ${x} 47 V 110" class="line ${i === 0 ? "secondary" : "tertiary"}"/><text x="${x}" y="${ty}" text-anchor="middle">${label} ${value}%</text>`;
    });
    body += `<text x="20" y="198">ECN / CNP / DCQCN: end-to-end feedback, not PFC</text>`;
    body += boxes([["Switch queue", r.marked ? "ECN: CE mark" : "below ECN"], ["Receiver RNIC", r.marked ? "send CNP" : "no CNP"], ["Sender RNIC", r.marked ? "DCQCN reduce" : "normal rate"]]).replaceAll('y="50"', 'y="220"').replaceAll('y="85"', 'y="255"').replaceAll('y="115"', 'y="285"').replaceAll(" 95", " 265");
    body += `<rect x="610" y="330" width="180" height="80" rx="8" class="surface"/><text x="700" y="359" text-anchor="middle">Upstream neighbor</text><text x="700" y="385" text-anchor="middle">${r.paused ? "PFC PAUSED" : "PFC CLEAR"}</text><path data-signal="pfc-hop" d="M 107 310 V 370 H 603" class="line tertiary"/><path d="M 597 365 L 603 370 L 597 375" class="line tertiary"/><text x="205" y="352">PFC: switch → upstream hop pause/resume</text>`;
    return kpis([[p.queue + "%", t("队列占用（示意）", "Illustrative queue occupancy")], [r.marked ? "CE / CNP" : "No mark", t("ECN反馈状态", "ECN feedback state")], [r.paused ? "PAUSE" : "CLEAR", t("PFC逐跳状态", "PFC hop state")], [fmt(headroom.requiredMB, 3) + " MB", t("反应期headroom工程下界", "Reaction-period headroom floor")], [fmt(headroom.availableMB, 3) + " MB", t("教学池Xoff后的余量", "Teaching-pool room beyond Xoff")], [headroom.fits ? t("只满足本例余量公式", "Passes this allowance formula only") : t("余量不足，pause后仍可能溢出", "Insufficient room; overflow possible after pause"), t("不是厂商配置验证", "Not vendor configuration validation")]]).replace('<div class="kpis">', `<div class="kpis" data-headroom-fits="${headroom.fits}">`) +
      svg(body, t("ECN/CNP是端到端反馈；PFC由交换机直接作用上游hop，保留Xoff/Xon滞回；非网络仿真器。", "ECN/CNP is end-to-end feedback; PFC acts directly from the switch on its upstream hop, retaining Xoff/Xon hysteresis; not a network simulator."), 445) +
      note(t("尝试先把队列推至 Xoff 以上，再下降到 Xon 与 Xoff 之间：pause 会保持。恢复算例按钮重置状态；真实 ECN 采用概率/多阈值策略，本图只显示概念。", "Raise the queue above Xoff, then lower it between Xon and Xoff: pause remains. Reset clears state. Real ECN uses probabilistic/multithreshold policies; this chart only explains concepts."));
  }
  const renderers = { lifecycle, trace, goodput, capacity, mixture, burst, ptu, normalized, cost, sizing, rank: rankMemory, pd, pdBudget, parallel, speculative, prefix: prefixCaching, network, networkBudget, rail, congestion };
  function inputValues(form) {
    const values = {};
    for (const input of form.querySelectorAll("[data-key]")) {
      if (input.disabled) continue;
      if (input.type === "checkbox") values[input.dataset.key] = input.checked;
      else if (input.tagName === "TEXTAREA" || ["model", "gpu", "mode", "kind", "operation", "mlaCache", "gateMode", "boundary", "deployment", "firstVisible", "sizeMode", "mapping"].includes(input.dataset.key)) values[input.dataset.key] = input.value;
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
    if (form.dataset.lab === "parallel") {
      const input = form.querySelector('[data-key="attentionDP"]');
      input.disabled = form.querySelector('[data-key="kind"]').value !== "dp";
      if (input.disabled) input.checked = false;
    }
    if (form.dataset.lab === "rank") {
      const key = form.querySelector('[data-key="model"]').value;
      const mla = Object.hasOwn(M.models, key) && M.models[key].latent !== undefined;
      const layout = form.querySelector('[data-key="mlaCache"]');
      layout.disabled = !mla;
      form.querySelector('[data-key="mlaLayerBytes"]').disabled = !mla || layout.value !== "measured";
      const mtp = form.querySelector('[data-key="mtp"]');
      mtp.disabled = !mla;
      if (mtp.disabled) mtp.checked = false;
      form.querySelector('[data-key="deviceGB"]').disabled = !form.querySelector('[data-key="measuredMemory"]').checked;
      form.querySelector('[data-key="replicaBatch"]').disabled = !form.querySelector('[data-key="customReplicaBatch"]').checked;
    }
    if (form.dataset.lab === "pd") {
      form.querySelector('[data-key="bytesPerToken"]').disabled = !form.querySelector('[data-key="customKVBytes"]').checked;
    }
    if (form.dataset.lab === "pdBudget") {
      const model = document.querySelector('form[data-lab="pd"] [data-key="model"]').value;
      const mtp = form.querySelector('[data-key="mtp"]');
      mtp.disabled = model !== "deepseek";
      if (mtp.disabled) mtp.checked = false;
    }
    if (form.dataset.lab === "networkBudget") form.querySelector('[data-key="sharedGBps"]').disabled = !form.querySelector('[data-key="crossesShared"]').checked;
    if (form.dataset.lab === "rail") form.querySelector('[data-key="failedRail"]').disabled = !form.querySelector('[data-key="failure"]').checked;
  }
  function render(form) {
    const workbench = form.closest("[data-workbench]");
    const errorBox = workbench.querySelector(".error");
    const results = workbench.querySelector("[data-results]");
    const summary = workbench.querySelector("[data-result-summary]");
    prepare(form);
    for (const output of form.querySelectorAll("[data-for]")) output.value = form.querySelector(`[data-key="${output.dataset.for}"]`).value;
    try {
      if (!form.checkValidity()) throw new RangeError(t("数值超出字段允许范围，或步长不正确；请检查当前输入。", "A value is outside its allowed range or step; inspect the inputs."));
      results.innerHTML = renderers[form.dataset.lab](inputValues(form), form);
      summary.textContent = [...results.querySelectorAll(".kpi")].slice(0, 3).map((el) => `${el.querySelector("strong").textContent} ${el.querySelector("span").textContent}`).join(". ");
      form.removeAttribute("aria-invalid");
      for (const input of form.querySelectorAll("[data-key]")) input.removeAttribute("aria-invalid");
      results.removeAttribute("aria-invalid");
      errorBox.hidden = true;
      errorBox.textContent = "";
    } catch (error) {
      results.replaceChildren();
      summary.textContent = "";
      form.setAttribute("aria-invalid", "true");
      for (const input of form.querySelectorAll("[data-key]")) {
        if (!input.disabled && !input.checkValidity()) input.setAttribute("aria-invalid", "true");
        else input.removeAttribute("aria-invalid");
      }
      results.setAttribute("aria-invalid", "true");
      errorBox.hidden = false;
      errorBox.textContent = t("无法计算：", "Cannot calculate: ") + error.message;
      if (!(error instanceof RangeError)) {
        console.error(error);
        throw error;
      }
    }
    if (form.dataset.lab === "pd") {
      const linked = document.querySelector('form[data-lab="pdBudget"]');
      if (linked) render(linked);
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
  const updateHeaderOffset = () => document.documentElement.style.setProperty("--header-offset", document.querySelector(".topbar").offsetHeight + 16 + "px");
  updateHeaderOffset();
  if (typeof ResizeObserver === "function") new ResizeObserver(updateHeaderOffset).observe(document.querySelector(".topbar"));
  else window.addEventListener("resize", updateHeaderOffset);
  window.addEventListener("scroll", () => {
    const total = document.documentElement.scrollHeight - window.innerHeight;
    progress.style.width = (total <= 0 ? 100 : Math.min(100, window.scrollY / total * 100)) + "%";
  }, { passive: true });
  window.AcademyUI = Object.freeze({ refresh });
  refresh();
})();
