(function (root) {
  "use strict";
  const models = {
    llama8: { name: "Llama 3.1 8B", total: 8.03, active: 8.03, layers: 32, heads: 32, kvHeads: 8, headDim: 128, hidden: 4096, contextLimit: 131072 },
    llama70: { name: "Llama 3.1 70B", total: 70.6, active: 70.6, layers: 80, heads: 64, kvHeads: 8, headDim: 128, hidden: 8192, contextLimit: 131072 },
    llama405: { name: "Llama 3.1 405B", total: 405, active: 405, layers: 126, heads: 128, kvHeads: 8, headDim: 128, hidden: 16384, contextLimit: 131072 },
    qwen72: { name: "Qwen2.5 72B (stock 32K)", total: 72.7, active: 72.7, layers: 80, heads: 64, kvHeads: 8, headDim: 128, hidden: 8192, contextLimit: 32768 },
    deepseek: { name: "DeepSeek V3 / original R1 671B", total: 671, active: 37, layers: 61, heads: 128, kvHeads: 8, headDim: 128, hidden: 7168, latent: 512, rope: 64, experts: 256, topK: 8, mtp: 14, contextLimit: 131072 }
  };
  const gpus = {
    h100: { name: "H100 SXM 80 GB", memory: 80, bandwidth: 3350, bf16: 989.5, nvlink: 900 },
    h200: { name: "H200 SXM 141 GB", memory: 141, bandwidth: 4800, bf16: 989.5, nvlink: 900 },
    b200: { name: "B200 192 GB (design / what-if)", memory: 192, bandwidth: 8000, bf16: 2250, nvlink: 1800 },
    b200dgx: { name: "DGX B200 180 GB / GPU", memory: 180, bandwidth: 8000, bf16: 2250, nvlink: 1800 },
    l40s: { name: "L40S 48 GB GDDR6", memory: 48, bandwidth: 864, bf16: 362.05, nvlink: 0 }
  };
  const traceSamples = [
    { id: "steady", arrival_ms: 0, admitted_ms: 20, visible_token_ms: [80, 100, 120, 140], ended_ms: 142, prompt_tokens: 512, status: "success", quality_pass: true },
    { id: "burst", arrival_ms: 10, admitted_ms: 30, visible_token_ms: [200, 200, 200, 260], ended_ms: 262, prompt_tokens: 1024, status: "success", quality_pass: true },
    { id: "cold-long", arrival_ms: 20, admitted_ms: 40, visible_token_ms: [520, 560, 600, 640], ended_ms: 642, prompt_tokens: 8192, status: "success", quality_pass: true },
    { id: "single", arrival_ms: 30, admitted_ms: 50, visible_token_ms: [150], ended_ms: 152, prompt_tokens: 64, status: "success", quality_pass: true },
    { id: "partial-failure", arrival_ms: 40, admitted_ms: 60, visible_token_ms: [180, 200], ended_ms: 220, prompt_tokens: 256, status: "failed", quality_pass: null },
    { id: "rejection", arrival_ms: 50, admitted_ms: null, visible_token_ms: [], ended_ms: 55, prompt_tokens: 128, status: "rejected", quality_pass: null },
    { id: "quality-failure", arrival_ms: 60, admitted_ms: 80, visible_token_ms: [180, 195, 210], ended_ms: 212, prompt_tokens: 512, status: "success", quality_pass: false }
  ];
  const workloadProfiles = [
    { name: "IDE", prompt: 256, output: 64, ttft: 200, tpot: 15 },
    { name: "RAG", prompt: 4096, output: 512, ttft: 800, tpot: 35 },
    { name: "Reasoning", prompt: 8192, output: 4096, ttft: 2000, tpot: 50 }
  ];
  const ptuModels = {
    gpt62sol: { name: "gpt-6.2-sol", version: "2026-11-17", capacity: 2950, shortRead: 0.05, longRead: 0.1, regionalMinimum: 25, regionalStep: 25 },
    gpt62astra: { name: "gpt-6.2-astra", version: "2026-11-17", capacity: 750, shortRead: 0.05, longRead: 0.1, regionalMinimum: 25, regionalStep: 25 },
    gpt61sol: { name: "gpt-6.1-sol", version: "2026-09-29", capacity: 3000, shortRead: 0.05, longRead: 0.1, regionalMinimum: 25, regionalStep: 25 },
    gpt6sol: { name: "gpt-6-sol", version: "2026-09-22", capacity: 3000, shortRead: 0.1, longRead: 0.2, regionalMinimum: 15, regionalStep: 5 },
    gpt6luna: { name: "gpt-6-luna", version: "2026-09-22", capacity: 59400, shortRead: 0.1, longRead: 0.2, regionalMinimum: 15, regionalStep: 5 },
    gpt6astra: { name: "gpt-6-astra", version: "2026-09-03", capacity: 600, shortRead: 0.1, longRead: 0.2, regionalMinimum: 25, regionalStep: 25 }
  };
  function number(value, name, min = 0, max = Number.MAX_SAFE_INTEGER) {
    if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) {
      throw new RangeError(`${name}: expected a finite number in [${min}, ${max}]`);
    }
    return value;
  }
  function integer(value, name, min = 0, max = Number.MAX_SAFE_INTEGER) {
    number(value, name, min, max);
    if (!Number.isInteger(value)) throw new RangeError(`${name}: expected an integer`);
    return value;
  }
  function preset(items, key) {
    if (!Object.hasOwn(items, key)) throw new RangeError("Unknown preset");
    return items[key];
  }
  function lifecycle({ queue, prefill, delivery, output, tpot }) {
    [queue, prefill, delivery].forEach((v) => number(v, "latency"));
    integer(output, "output", 1);
    number(tpot, "tpot", 0.001);
    const ttft = queue + prefill + delivery;
    const decode = (output - 1) * tpot;
    return { ttft, decode, e2e: ttft + decode, userTPS: output > 1 ? 1000 / tpot : null, measuredTPOT: output > 1 ? tpot : null };
  }
  function traceMetrics({ requests, windowMs, ttftLimit, tpotLimit, gateMode, boundary, requireQuality }) {
    number(windowMs, "measurement window", 1, 3600000);
    number(ttftLimit, "TTFT gate", 0.001);
    number(tpotLimit, "TPOT/ITL gate", 0.001);
    if (!["mean", "max"].includes(gateMode) || !["client", "admitted"].includes(boundary)) throw new RangeError("Unknown trace gate or system boundary");
    if (typeof requireQuality !== "boolean") throw new RangeError("Choose whether the quality gate is required");
    if (!Array.isArray(requests) || !requests.length || requests.length > 2000) throw new RangeError("Trace requires 1–2000 request records");
    const keys = ["id", "arrival_ms", "admitted_ms", "visible_token_ms", "ended_ms", "prompt_tokens", "status", "quality_pass"];
    const ids = new Set();
    let tokenCount = 0;
    const time = (v) => number(v, "relative timestamp ms", -1e12, 1e12);
    const inWindow = (v) => v >= 0 && v < windowMs;
    const rows = requests.map((request, index) => {
      if (!request || typeof request !== "object" || keys.some((key) => !Object.hasOwn(request, key)) || Object.keys(request).some((key) => !keys.includes(key))) {
        throw new RangeError(`Record ${index + 1}: use the exact numeric trace schema`);
      }
      if (typeof request.id !== "string" || !/^[A-Za-z0-9._-]{1,80}$/.test(request.id) || ids.has(request.id)) throw new RangeError("Trace IDs must be unique anonymous labels");
      ids.add(request.id);
      time(request.arrival_ms);
      time(request.ended_ms);
      if (request.admitted_ms !== null) time(request.admitted_ms);
      integer(request.prompt_tokens, "prompt tokens");
      if (!["success", "failed", "cancelled", "rejected"].includes(request.status)) throw new RangeError("Unknown terminal request status");
      if (request.quality_pass !== null && typeof request.quality_pass !== "boolean") throw new RangeError("Quality annotation must be true, false or null");
      if (request.ended_ms < request.arrival_ms || (request.admitted_ms !== null && (request.admitted_ms < request.arrival_ms || request.admitted_ms > request.ended_ms))) throw new RangeError("Trace times are inconsistent; align clocks and request boundaries");
      if (!Array.isArray(request.visible_token_ms) || request.visible_token_ms.length > 32768) throw new RangeError("Visible-token timestamps must be an array of at most 32768 entries");
      tokenCount += request.visible_token_ms.length;
      if (tokenCount > 100000) throw new RangeError("Trace has too many visible-token timestamps");
      const times = request.visible_token_ms;
      times.forEach((value, i) => {
        time(value);
        if (value < request.arrival_ms || value > request.ended_ms || (request.admitted_ms !== null && value < request.admitted_ms) || (i && value < times[i - 1])) throw new RangeError("Token timestamps must be nondecreasing within the request; equal chunk timestamps are allowed");
      });
      if (request.status === "rejected" && (request.admitted_ms !== null || times.length)) throw new RangeError("Rejected requests cannot have admitted work or visible output");
      const itls = times.slice(1).map((value, i) => value - times[i]);
      const ttft = times.length ? times[0] - request.arrival_ms : null;
      const tpot = itls.length ? itls.reduce((sum, value) => sum + value, 0) / itls.length : null;
      const maxITL = itls.length ? Math.max(...itls) : null;
      const e2eVisible = times.length ? times[times.length - 1] - request.arrival_ms : null;
      const intervalMetric = gateMode === "mean" ? tpot : maxITL;
      const latencyPass = ttft !== null && ttft <= ttftLimit && (intervalMetric === null || intervalMetric <= tpotLimit);
      const qualityPass = !requireQuality || request.quality_pass === true;
      const good = request.status === "success" && latencyPass && qualityPass;
      return { ...request, ttft, tpot, maxITL, e2eVisible, terminalLatency: request.ended_ms - request.arrival_ms, itls, latencyPass, qualityPass, good, completeInWindow: inWindow(request.ended_ms), outputTokensInWindow: times.filter(inWindow).length, userTPS: tpot !== null && tpot > 0 ? 1000 / tpot : null };
    });
    if (!rows.some((row) => row.arrival_ms < windowMs && row.ended_ms >= 0)) throw new RangeError("Trace does not overlap the window; use relative timestamps or change the window");
    const unknownAdmission = rows.filter((row) => row.status !== "rejected" && row.admitted_ms === null).length;
    if (boundary === "admitted" && unknownAdmission) throw new RangeError("Admitted-boundary occupancy needs timestamps for every non-rejected request; unknown admission is not zero");
    const selected = boundary === "client" ? rows : rows.filter((row) => row.admitted_ms !== null);
    const events = [];
    let residenceAreaMs = 0;
    selected.forEach((row) => {
      const start = boundary === "client" ? row.arrival_ms : row.admitted_ms;
      const clippedStart = Math.max(0, start);
      const clippedEnd = Math.min(windowMs, row.ended_ms);
      if (clippedEnd > clippedStart) {
        residenceAreaMs += clippedEnd - clippedStart;
        events.push({ time: clippedStart, delta: 1 }, { time: clippedEnd, delta: -1 });
      }
    });
    events.sort((a, b) => a.time - b.time);
    const occupancy = [{ x: 0, y: 0 }];
    let active = 0;
    for (let i = 0; i < events.length;) {
      const at = events[i].time;
      occupancy.push({ x: at, y: active });
      let delta = 0;
      while (i < events.length && events[i].time === at) delta += events[i++].delta;
      active += delta;
      occupancy.push({ x: at, y: active });
    }
    occupancy.push({ x: windowMs, y: active });
    const completed = selected.filter((row) => row.completeInWindow);
    const meanResidenceMs = completed.length ? completed.reduce((sum, row) => sum + row.ended_ms - (boundary === "client" ? row.arrival_ms : row.admitted_ms), 0) / completed.length : null;
    const seconds = windowMs / 1000;
    const lambda = completed.length / seconds;
    const allITLs = rows.flatMap((row) => row.itls.filter((_, i) => inWindow(row.visible_token_ms[i + 1])));
    const ttfts = rows.filter((row) => row.ttft !== null && inWindow(row.visible_token_ms[0])).map((row) => row.ttft);
    const requestTPOTs = rows.filter((row) => row.completeInWindow && row.tpot !== null).map((row) => row.tpot);
    const goods = rows.filter((row) => row.completeInWindow && row.good);
    const statuses = Object.fromEntries(["success", "failed", "cancelled", "rejected"].map((status) => [status, rows.filter((row) => row.completeInWindow && row.status === status).length]));
    return { rows, statuses, occupancy, meanActive: residenceAreaMs / windowMs, meanResidenceMs, lambda, littleProduct: meanResidenceMs === null ? null : lambda * meanResidenceMs / 1000, closedWindow: selected.every((row) => (boundary === "client" ? row.arrival_ms : row.admitted_ms) >= 0 && row.ended_ms < windowMs), offeredRPS: rows.filter((row) => inWindow(row.arrival_ms)).length / seconds, admittedRPS: rows.filter((row) => row.admitted_ms !== null && inWindow(row.admitted_ms)).length / seconds, completedRPS: rows.filter((row) => row.completeInWindow).length / seconds, inputDemandTPS: rows.filter((row) => row.admitted_ms !== null && inWindow(row.admitted_ms)).reduce((sum, row) => sum + row.prompt_tokens, 0) / seconds, visibleOutputTPS: rows.reduce((sum, row) => sum + row.outputTokensInWindow, 0) / seconds, goodRequestRPS: goods.length / seconds, goodOutputTPS: goods.reduce((sum, row) => sum + row.outputTokensInWindow, 0) / seconds, goodRequests: goods.length, itlSamples: allITLs.length, meanRequestTPOT: requestTPOTs.length ? requestTPOTs.reduce((sum, value) => sum + value, 0) / requestTPOTs.length : null, meanObservedITL: allITLs.length ? allITLs.reduce((sum, value) => sum + value, 0) / allITLs.length : null, p99ITL: allITLs.length ? percentile(allITLs, 0.99) : null, p95TTFT: ttfts.length ? percentile(ttfts, 0.95) : null, unknownAdmission };
  }
  function capacity(p) {
    number(p.qps, "qps");
    number(p.headroom, "headroom", 1, 10);
    integer(p.prompt, "prompt");
    integer(p.output, "output", 1);
    number(p.ttft, "ttft");
    number(p.tpot, "tpot", 0.001);
    number(p.inputCapacity, "inputCapacity", 1);
    number(p.outputCapacity, "outputCapacity", 1);
    integer(p.batch, "batch", 1);
    number(p.utilization, "utilization", 0.01, 1);
    const qps = p.qps * p.headroom;
    const latency = lifecycle({ queue: 0, prefill: p.ttft, delivery: 0, output: p.output, tpot: p.tpot });
    const inputTPS = qps * p.prompt;
    const outputTPS = qps * p.output;
    const residence = latency.e2e / 1000;
    const active = qps * residence;
    const decodeActive = qps * latency.decode / 1000;
    return { qps, inputTPS, outputTPS, inputTPM: inputTPS * 60, outputTPM: outputTPS * 60, rpm: qps * 60, residence, active, decodeActive, ...instanceBounds(inputTPS, outputTPS, active, p) };
  }
  function instanceBounds(inputTPS, outputTPS, active, p) {
    const inputInstances = Math.ceil(inputTPS / (p.inputCapacity * p.utilization));
    const outputInstances = Math.ceil(outputTPS / (p.outputCapacity * p.utilization));
    const batchInstances = Math.ceil(active / p.batch);
    return { inputInstances, outputInstances, batchInstances, instances: Math.max(inputInstances, outputInstances, batchInstances) };
  }
  function mixtureCapacity(p) {
    number(p.shortShare, "short-request share", 0, 1);
    number(p.ragShare, "RAG-request share", 0, 1);
    if (p.shortShare + p.ragShare > 1 + 1e-12) throw new RangeError("Request shares must sum to no more than one");
    const shares = [p.shortShare, p.ragShare, Math.max(0, 1 - p.shortShare - p.ragShare)];
    const classes = workloadProfiles.map((profile, i) => ({ ...profile, share: shares[i], ...capacity({ ...p, ...profile, qps: p.qps * shares[i] }) }));
    const qps = p.qps * p.headroom;
    const inputTPS = classes.reduce((sum, group) => sum + group.inputTPS, 0);
    const outputTPS = classes.reduce((sum, group) => sum + group.outputTPS, 0);
    const active = classes.reduce((sum, group) => sum + group.active, 0);
    const meanResidence = classes.reduce((sum, group) => sum + group.share * group.residence, 0);
    const meanPrompt = classes.reduce((sum, group) => sum + group.share * group.prompt, 0);
    const meanOutput = classes.reduce((sum, group) => sum + group.share * group.output, 0);
    const meanTTFT = classes.reduce((sum, group) => sum + group.share * group.ttft, 0);
    const meanTPOT = classes.reduce((sum, group) => sum + group.share * group.tpot, 0);
    const incorrectShortcutW = (meanTTFT + (meanOutput - 1) * meanTPOT) / 1000;
    return { classes, qps, inputTPS, outputTPS, inputTPM: inputTPS * 60, outputTPM: outputTPS * 60, active, meanResidence, meanPrompt, meanOutput, meanTTFT, meanTPOT, incorrectShortcutW, ...instanceBounds(inputTPS, outputTPS, active, p) };
  }
  function burstQueue({ baseQPS, burstMultiplier, burstSeconds, serviceRPS, queueBudgetMs, horizonSeconds }) {
    number(baseQPS, "base QPS");
    number(burstMultiplier, "burst multiplier", 1, 1000);
    number(burstSeconds, "burst seconds", 0.01, 3600);
    number(serviceRPS, "same-workload service RPS", 0.001);
    number(queueBudgetMs, "queue-only budget ms");
    number(horizonSeconds, "horizon seconds", burstSeconds, 86400);
    const burstRate = baseQPS * burstMultiplier;
    const excessRate = Math.max(0, burstRate - serviceRPS);
    const peakQueue = excessRate * burstSeconds;
    const queueCap = serviceRPS * queueBudgetMs / 1000;
    const cappedPeak = Math.min(peakQueue, queueCap);
    const spareRate = serviceRPS - baseQPS;
    const drainSeconds = peakQueue === 0 ? 0 : spareRate > 0 ? peakQueue / spareRate : null;
    const cappedDrainSeconds = cappedPeak === 0 ? 0 : spareRate > 0 ? cappedPeak / spareRate : null;
    const times = new Set(Array.from({ length: 61 }, (_, i) => horizonSeconds * i / 60));
    [burstSeconds, excessRate > 0 ? queueCap / excessRate : null, drainSeconds === null ? null : burstSeconds + drainSeconds, cappedDrainSeconds === null ? null : burstSeconds + cappedDrainSeconds].forEach((value) => {
      if (value !== null && value >= 0 && value <= horizonSeconds) times.add(value);
    });
    const points = [...times].sort((a, b) => a - b).map((time) => {
      const queue = time <= burstSeconds ? excessRate * time : Math.max(0, peakQueue - spareRate * (time - burstSeconds));
      const capped = time <= burstSeconds ? Math.min(queue, queueCap) : Math.min(queueCap, Math.max(0, cappedPeak - spareRate * (time - burstSeconds)));
      return { time, queue, capped };
    });
    return { burstRate, peakQueue, queueCap, cappedPeak, drainSeconds, cappedDrainSeconds, peakWaitSeconds: peakQueue / serviceRPS, cappedWaitSeconds: cappedPeak / serviceRPS, rejectedDuringBurst: Math.max(0, peakQueue - cappedPeak), baseUtilization: baseQPS / serviceRPS, stableBase: baseQPS < serviceRPS, points };
  }
  function businessQPS({ dau, requests, hours, peak }) {
    number(dau, "dau");
    number(requests, "requests");
    number(hours, "hours", 0.1, 24);
    number(peak, "peak", 1, 100);
    return dau * requests * peak / (hours * 3600);
  }
  function ptu({ inputTPM, outputTPM, cache, cachedWeight, outputWeight, capacity, minimum, step }) {
    number(inputTPM, "inputTPM");
    number(outputTPM, "outputTPM");
    number(cache, "cache", 0, 1);
    number(cachedWeight, "cachedWeight");
    number(outputWeight, "outputWeight", 0.001);
    number(capacity, "PTU capacity", 0.001);
    integer(minimum, "minimum", 1);
    integer(step, "step", 1);
    const normalized = inputTPM * ((1 - cache) + cache * cachedWeight) + outputTPM * outputWeight;
    return { normalized, ...provisioning(normalized, capacity, minimum, step) };
  }
  function provisioning(normalized, capacity, minimum, step) {
    const raw = normalized / capacity;
    const provisioned = normalized === 0 ? 0 : minimum + Math.max(0, Math.ceil((raw - minimum) / step)) * step;
    return { raw, provisioned };
  }
  function normalizedPTU(p) {
    const model = preset(ptuModels, p.model);
    if (!["global-zone", "regional"].includes(p.deployment)) throw new RangeError("Unknown PTU deployment category");
    const weights = { shortInput: 1, shortRead: model.shortRead, shortWrite30: 1.25, shortWrite12: 2.5, shortOutput: 5, longInput: 2, longRead: model.longRead, longWrite30: 2.5, longWrite12: 5, longOutput: 7.5 };
    const categories = Object.entries(weights).map(([key, weight]) => ({ key, weight, tokens: number(p[key], "mutually exclusive category TPM"), normalized: p[key] * weight }));
    const normalized = categories.reduce((sum, category) => sum + category.normalized, 0);
    const rawTokens = categories.reduce((sum, category) => sum + category.tokens, 0);
    const minimum = p.deployment === "regional" ? model.regionalMinimum : 15;
    const step = p.deployment === "regional" ? model.regionalStep : 5;
    const readContribution = categories.filter((category) => ["shortRead", "longRead"].includes(category.key)).reduce((sum, category) => sum + category.normalized, 0);
    return { model, categories, normalized, rawTokens, minimum, step, ...provisioning(normalized, model.capacity, minimum, step), incorrectFreeReads: provisioning(normalized - readContribution, model.capacity, minimum, step).provisioned, readContribution };
  }
  function kvPerToken(model, bytes, architecture = "native") {
    number(bytes, "KV bytes", 0.5, 4);
    if (!["native", "mha", "gqa", "mla"].includes(architecture)) throw new RangeError("Unknown KV architecture");
    const mode = architecture === "native" ? (model.latent ? "mla" : "gqa") : architecture;
    if (mode === "mla") {
      if (!model.latent) throw new RangeError("MLA is unavailable for this model");
      return model.layers * (model.latent + model.rope) * bytes;
    }
    const heads = mode === "mha" ? model.heads : model.kvHeads;
    return 2 * model.layers * heads * model.headDim * bytes;
  }
  function moeUnion(model, batch) {
    integer(batch, "batch", 1);
    if (!model.experts) return model.total;
    // Solve shared + routed * topK/experts = active under uniform expert sizes.
    const ratio = model.topK / model.experts;
    const shared = (model.active - model.total * ratio) / (1 - ratio);
    const union = -Math.expm1(batch * Math.log1p(-ratio));
    return shared + (model.total - shared) * union;
  }
  function sizing(p) {
    const model = preset(models, p.model);
    const gpu = preset(gpus, p.gpu);
    number(p.weightBytes, "weightBytes", 0.5, 2);
    number(p.kvBytes, "kvBytes", 1, 2);
    integer(p.context, "context", 1, 1048576);
    integer(p.prefillTokens, "cold prefill input tokens", 1, p.context);
    integer(p.batch, "batch", 1, 4096);
    integer(p.prefillBatch, "prefillBatch", 1, p.batch);
    number(p.safe, "safe", 0.5, 0.95);
    number(p.runtime, "runtime");
    number(p.overhead, "overhead", 0, 1);
    number(p.replication, "replication", 1, 64);
    number(p.mbu, "mbu", 0.01, 1);
    number(p.mfu, "mfu", 0.01, 1);
    number(p.tpot, "tpot", 0.001);
    number(p.ttft, "ttft", 0.001);
    number(p.reserve, "reserve");
    const usable = gpu.memory * p.safe - p.runtime;
    if (usable <= 0) throw new RangeError("Runtime reserve exhausts safe GPU memory");
    if (p.ttft <= p.reserve) throw new RangeError("TTFT budget must exceed non-compute reserve");
    const totalParams = model.total + (p.mtp ? (model.mtp || 0) : 0);
    const weight = totalParams * p.weightBytes * (1 + p.overhead);
    const kv = kvPerToken(model, p.kvBytes) * p.context * p.batch * p.replication / 1e9;
    const memoryCards = Math.ceil((weight + kv) / usable);
    const optimisticTraffic = model.active * p.weightBytes;
    const unionTraffic = moeUnion(model, p.batch) * p.weightBytes;
    const bandwidthCards = Math.ceil((unionTraffic + kv) / (gpu.bandwidth * p.mbu * p.tpot / 1000));
    const prefillFLOPs = 2 * model.active * 1e9 * p.prefillTokens * p.prefillBatch + 2 * model.layers * model.hidden * p.prefillTokens ** 2 * p.prefillBatch;
    const computeCards = Math.ceil(prefillFLOPs / (gpu.bf16 * 1e12 * p.mfu * (p.ttft - p.reserve) / 1000));
    const floor = Math.max(1, memoryCards, bandwidthCards, computeCards);
    const candidate = 2 ** Math.ceil(Math.log2(floor));
    return { model, gpu, weight, kv, memoryCards, bandwidthCards, computeCards, floor, candidate, runtime: p.runtime * candidate, optimisticTPOT: optimisticTraffic / (candidate * gpu.bandwidth * p.mbu) * 1000, unionTPOT: unionTraffic / (candidate * gpu.bandwidth * p.mbu) * 1000, kvReadTPOT: kv / (candidate * gpu.bandwidth * p.mbu) * 1000, prefillMs: prefillFLOPs / (candidate * gpu.bf16 * 1e12 * p.mfu) * 1000, prefillFLOPs, unionTraffic, contextSupported: p.context <= model.contextLimit };
  }
  function rankMemory(p) {
    const model = preset(models, p.model);
    const gpu = preset(gpus, p.gpu);
    integer(p.tp, "TP", 1, model.heads);
    integer(p.pp, "PP", 1, model.layers);
    integer(p.dp, "replica DP", 1, 128);
    integer(p.batch, "global resident batch", 1, 4096);
    integer(p.context, "resident context", 1, 1048576);
    number(p.weightBytes, "weightBytes", 0.5, 2);
    number(p.kvBytes, "kvBytes", 1, 2);
    number(p.safe, "safe", 0.5, 0.95);
    number(p.runtime, "runtime");
    number(p.overhead, "weight metadata", 0, 1);
    number(p.weightSkew, "weight imbalance budget", 1, 10);
    if (model.heads % p.tp !== 0) throw new RangeError("TP must divide the query-head count");
    if (!model.latent && ((model.kvHeads >= p.tp && model.kvHeads % p.tp !== 0) || (p.tp > model.kvHeads && p.tp % model.kvHeads !== 0))) {
      throw new RangeError("GQA TP and KV-head counts must divide one another");
    }
    const deviceGB = p.measuredMemory ? number(p.deviceGB, "measured total device GB", 1) : gpu.memory;
    const replicaBatch = p.customReplicaBatch ? integer(p.replicaBatch, "busiest replica batch", 1, p.batch) : Math.ceil(p.batch / p.dp);
    const weight = (model.total + (p.mtp ? (model.mtp || 0) : 0)) * p.weightBytes * (1 + p.overhead);
    const kvHeadsPerRank = model.latent ? 1 : Math.max(1, model.kvHeads / p.tp);
    let bytesPerLayer;
    if (model.latent) {
      if (!["replicated", "measured"].includes(p.mlaCache)) throw new RangeError("Choose a known MLA cache layout");
      bytesPerLayer = p.mlaCache === "measured" ? integer(p.mlaLayerBytes, "measured MLA bytes/layer/token/rank", 1) : (model.latent + model.rope) * p.kvBytes;
    } else bytesPerLayer = 2 * kvHeadsPerRank * model.headDim * p.kvBytes;
    const baseLayers = Math.floor(model.layers / p.pp);
    const extraLayers = model.layers % p.pp;
    const stages = Array.from({ length: p.pp }, (_, stage) => {
      const layers = baseLayers + (stage < extraLayers ? 1 : 0);
      const weightGB = weight * layers / model.layers / p.tp * p.weightSkew;
      const kvGB = bytesPerLayer * layers * p.context * replicaBatch / 1e9;
      const usedGB = weightGB + kvGB + p.runtime;
      return { stage, layers, weightGB, kvGB, runtimeGB: p.runtime, usedGB, marginGB: deviceGB * p.safe - usedGB };
    });
    const worst = stages.reduce((a, stage) => stage.usedGB > a.usedGB ? stage : a, stages[0]);
    const cards = p.tp * p.pp * p.dp;
    const kvReserveGB = stages.reduce((sum, stage) => sum + stage.kvGB, 0) * p.tp * p.dp;
    const idealKVGB = kvPerToken(model, p.kvBytes) * p.context * p.batch / 1e9;
    return { model, gpu, deviceGB, cards, replicaBatch, reservedRequestSlots: replicaBatch * p.dp, kvHeadsPerRank, bytesPerLayer, stages, worst, kvReserveGB, idealKVGB, effectiveKVFactor: kvReserveGB / idealKVGB, weightReserveGB: weight * p.dp * p.weightSkew, fits: worst.marginGB >= 0, contextSupported: p.context <= model.contextLimit };
  }
  function transfer(p) {
    const model = preset(models, p.model);
    integer(p.context, "context", 1);
    number(p.kvBytes, "kvBytes", 1, 2);
    number(p.cache, "cache", 0, 1);
    number(p.rate, "rate", 0.001);
    number(p.efficiency, "efficiency", 0.01, 1);
    integer(p.rails, "rails", 1, 64);
    number(p.setup, "setup");
    number(p.overlap, "overlap");
    number(p.qps, "qps");
    number(p.wireCopies, "actual wire multiplier", 1, 64);
    const bytesPerToken = p.customKVBytes ? integer(p.bytesPerToken, "unique logical KV bytes/token", 1) : kvPerToken(model, p.kvBytes);
    const fullLogicalBytes = bytesPerToken * p.context;
    const logicalBytes = fullLogicalBytes * (1 - p.cache);
    const bytes = logicalBytes * p.wireCopies;
    const bandwidth = p.rate * 1e9 / 8 * p.efficiency * p.rails;
    const rawMs = bytes / bandwidth * 1000;
    return { bytes, logicalBytes, fullLogicalBytes, bytesPerToken, bandwidth, rawMs, exposedMs: Math.max(0, rawMs - p.overlap) + p.setup, utilization: bytes * p.qps / bandwidth, requiredGbps: bytes * p.qps * 8 / 1e9 / p.efficiency };
  }
  function pdBudget(p) {
    const model = preset(models, p.model);
    const tx = transfer(p);
    integer(p.outputTokens, "all generated output tokens", 1, 32768);
    number(p.destinationCopies, "destination referenced-state multiplier", 1, 64);
    number(p.prefillPoolTPS, "prefill pool input capacity", 0.001);
    number(p.decodePoolTPS, "decode pool output capacity", 0.001);
    integer(p.decodeSlots, "decode KV-resident slots", 1, 32768);
    number(p.meanTPOT, "mean decode interval ms", 0.001);
    number(p.capacityLimit, "capacity-use planning limit", 0.01, 0.99);
    [p.queueP, p.prefillBudget, p.queueD, p.firstBudget, p.delivery].forEach((v) => number(v, "latency budget ms"));
    number(p.ttftTarget, "TTFT budget target", 0.001);
    number(p.itlTarget, "first-to-second gap target", 0.001);
    if (p.overlap > p.prefillBudget) throw new RangeError("Verified overlap cannot exceed the assumed prefill computation window");
    if (!["decode", "prefill"].includes(p.firstVisible)) throw new RangeError("Choose where the first visible token is emitted");
    if (p.firstVisible === "prefill" && p.outputTokens === 1) throw new RangeError("A one-token request ending visibly at P does not need the full PD pipeline");
    for (const flag of ["compatible", "memoryChecked", "colocatedPasses"]) if (typeof p[flag] !== "boolean") throw new RangeError("Confirm PD planning assumptions explicitly");
    integer(p.prefillReplicas, "complete-model P replicas", 1, 4096);
    integer(p.decodeReplicas, "complete-model D replicas", 1, 4096);
    number(p.weightBytes, "stored weight bytes", 0.5, 2);
    number(p.weightOverhead, "weight metadata allowance", 0, 1);
    const dOutputs = p.outputTokens - (p.firstVisible === "prefill" ? 1 : 0);
    const dComputeMs = p.firstVisible === "decode" ? p.firstBudget + (p.outputTokens - 1) * p.meanTPOT : dOutputs * p.meanTPOT;
    const dResidenceMs = tx.exposedMs + p.queueD + dComputeMs;
    number(dResidenceMs, "D resident duration budget", 0.001);
    const limits = [
      { key: "P input", rps: p.prefillPoolTPS / p.context },
      { key: "D output", rps: p.decodePoolTPS / dOutputs },
      { key: "D slots", rps: p.decodeSlots / (dResidenceMs / 1000) }
    ];
    if (tx.bytes > 0) limits.push({ key: "KV network", rps: tx.bandwidth / tx.bytes });
    const bottleneck = limits.reduce((a, value) => value.rps < a.rps ? value : a, limits[0]);
    const ttftBudget = p.queueP + p.prefillBudget + p.firstBudget + p.delivery + (p.firstVisible === "decode" ? tx.exposedMs + p.queueD : 0);
    const firstGapBudget = p.outputTokens <= 1 ? null : p.meanTPOT + (p.firstVisible === "prefill" ? tx.exposedMs + p.queueD : 0);
    const flags = {
      compatibility: p.compatible && p.memoryChecked && p.context + p.outputTokens <= model.contextLimit,
      capacityStable: limits.every((limit) => p.qps < limit.rps),
      headroom: limits.every((limit) => p.qps / limit.rps <= p.capacityLimit),
      latency: ttftBudget <= p.ttftTarget && (firstGapBudget === null || firstGapBudget <= p.itlTarget)
    };
    const readiness = p.colocatedPasses ? "keep-baseline" : !flags.compatibility ? "blocked-layout" : !flags.capacityStable ? "blocked-capacity" : !flags.latency ? "blocked-latency" : !flags.headroom ? "validate-headroom" : "trial-candidate";
    const fullStateBytes = tx.bytesPerToken * (p.context + p.outputTokens) * p.destinationCopies;
    const weightsGB = (model.total + (p.mtp ? (model.mtp || 0) : 0)) * p.weightBytes * (1 + p.weightOverhead) * (p.prefillReplicas + p.decodeReplicas);
    return { model, tx, limits, bottleneck, flags, readiness, ttftBudget, firstGapBudget, dResidenceMs, fullStateBytes, weightsGB, candidateOutputTPS: bottleneck.rps * p.outputTokens };
  }
  function percentile(values, fraction) {
    if (!Array.isArray(values) || !values.length) throw new RangeError("Percentile requires samples");
    number(fraction, "percentile", 0.001, 1);
    values.forEach((v) => number(v, "sample"));
    return [...values].sort((a, b) => a - b)[Math.ceil(fraction * values.length) - 1];
  }
  function pdSamples({ context, base, prefillRate, chunk, residual }) {
    integer(context, "context", 1);
    number(base, "base", 0.001);
    number(prefillRate, "prefillRate", 0.001);
    integer(chunk, "chunk", 1);
    number(residual, "residual");
    const block = context / prefillRate * 1000;
    const burst = Math.min(context, chunk) / prefillRate * 1000;
    const samples = { colocated: [], chunked: [], pd: [] };
    const requestTPOT = { colocated: [], chunked: [], pd: [] };
    for (let request = 0; request < 200; request += 1) {
      const sum = { colocated: 0, chunked: 0, pd: 0 };
      for (let gap = 0; gap < 31; gap += 1) {
        const interference = request % 50 === 0 ? gap % 7 === 0 : (request + gap) % 20 === 0;
        const jitter = ((request * 17 + gap * 13) % 7) * 0.15;
        const values = { colocated: base + jitter + (interference ? block : 0), chunked: base + jitter + (interference ? burst : 0), pd: base + jitter + (interference ? residual : 0) };
        for (const key of Object.keys(samples)) {
          samples[key].push(values[key]);
          sum[key] += values[key];
        }
      }
      for (const key of Object.keys(requestTPOT)) requestTPOT[key].push(sum[key] / 31);
    }
    return { samples, requestTPOT, requestCount: 200, outputTokens: 32, gapCount: 6200, p99: Object.fromEntries(Object.entries(samples).map(([key, values]) => [key, percentile(values, 0.99)])), p99TPOT: Object.fromEntries(Object.entries(requestTPOT).map(([key, values]) => [key, percentile(values, 0.99)])) };
  }
  function collective({ operation, ranks, sizeMB, timeMs, sizeMode = "reported" }) {
    integer(ranks, "ranks", 2, 65536);
    number(sizeMB, "sizeMB", 0.001);
    number(timeMs, "timeMs", 0.001);
    if (!["reported", "api-count"].includes(sizeMode)) throw new RangeError("Choose reported array size or NCCL API-count bytes");
    const factors = { allreduce: 2 * (ranks - 1) / ranks, allgather: (ranks - 1) / ranks, reducescatter: (ranks - 1) / ranks, alltoall: (ranks - 1) / ranks, broadcast: 1, reduce: 1, sendrecv: 1 };
    const factor = preset(factors, operation);
    const sizeMultiplier = sizeMode === "api-count" && ["allgather", "reducescatter", "alltoall"].includes(operation) ? ranks : 1;
    const effectiveSizeMB = sizeMB * sizeMultiplier;
    const algbw = effectiveSizeMB / timeMs;
    return { algbw, busbw: algbw * factor, factor, effectiveSizeMB, sizeMultiplier };
  }
  function pcieBandwidth(gen, lanes, payloadFraction) {
    integer(gen, "PCIe generation", 3, 5);
    if (![1, 2, 4, 8, 16].includes(lanes)) throw new RangeError("PCIe lane count must be 1,2,4,8 or16");
    number(payloadFraction, "PCIe payload efficiency", 0.01, 1);
    const gtPerLane = { 3: 8, 4: 16, 5: 32 }[gen];
    const encodedGBps = gtPerLane * lanes * (128 / 130) / 8;
    return { encodedGBps, payloadGBps: encodedGBps * payloadFraction };
  }
  function networkBudget(p) {
    const gpu = preset(gpus, p.gpu);
    integer(p.gpuEndpoints, "active GPU PCIe endpoints", 1, 256);
    integer(p.nicPorts, "active NIC ports", 1, 256);
    integer(p.nicCards, "active NIC PCIe cards", 1, p.nicPorts);
    number(p.rate, "port rate Gb/s", 0.001);
    number(p.networkEfficiency, "network payload fraction", 0.01, 1);
    number(p.fabricGBps, "fabric attributed payload budget GB/s", 0.001);
    number(p.payloadMB, "striped payload MB", 0.001);
    if (typeof p.crossesShared !== "boolean") throw new RangeError("Confirm whether this path crosses the shared link");
    if (p.crossesShared) number(p.sharedGBps, "shared path budget GB/s", 0.001);
    const gpuPCIe = pcieBandwidth(p.gpuGen, p.gpuLanes, p.pcieEfficiency);
    const nicPCIe = pcieBandwidth(p.nicGen, p.nicLanes, p.pcieEfficiency);
    const portPayloadGBps = p.rate / 8 * p.networkEfficiency;
    const limits = [
      { key: "NIC port payload", valueGBps: p.nicPorts * portPayloadGBps },
      { key: "GPU PCIe endpoints", valueGBps: p.gpuEndpoints * gpuPCIe.payloadGBps },
      { key: "NIC PCIe cards", valueGBps: p.nicCards * nicPCIe.payloadGBps },
      { key: "Fabric allocation", valueGBps: p.fabricGBps }
    ];
    if (p.crossesShared) limits.push({ key: "Shared path link", valueGBps: p.sharedGBps });
    const bottleneck = limits.reduce((a, value) => value.valueGBps < a.valueGBps ? value : a, limits[0]);
    return { gpu, gpuPCIe, nicPCIe, portPayloadGBps, limits, bottleneck, nodeGBps: bottleneck.valueGBps, perGpuGBps: bottleneck.valueGBps / p.gpuEndpoints, stripedTransferMs: p.payloadMB / bottleneck.valueGBps, nvlinkOneWayGBps: gpu.nvlink === 0 ? null : gpu.nvlink / 2 };
  }
  function railBudget(p) {
    integer(p.gpus, "GPU endpoints", 1, 64);
    integer(p.railCount, "rail count", 1, 16);
    number(p.payloadGB, "payload GB/GPU", 0.001);
    number(p.rate, "rail port Gb/s", 0.001);
    number(p.efficiency, "rail payload fraction", 0.01, 1);
    number(p.fabricGBps, "fabric payload GB/s", 0.001);
    number(p.hotMultiplier, "hot GPU payload multiplier", 1, 100);
    integer(p.hotGpu, "hot GPU index", 0, p.gpus - 1);
    if (!["balanced", "skewed"].includes(p.mapping)) throw new RangeError("Unknown rail mapping");
    if (typeof p.failure !== "boolean" || typeof p.reroute !== "boolean") throw new RangeError("Confirm rail failure and toy rerouting choices");
    if (p.failure) integer(p.failedRail, "failed rail index", 0, p.railCount - 1);
    const rails = Array.from({ length: p.railCount }, (_, rail) => ({ rail, failed: p.failure && rail === p.failedRail, gpus: [], payloadGB: 0 }));
    const live = rails.filter((rail) => !rail.failed);
    const assignments = [];
    let blockedGB = 0;
    let totalGB = 0;
    for (let gpu = 0; gpu < p.gpus; gpu += 1) {
      const payload = p.payloadGB * (gpu === p.hotGpu ? p.hotMultiplier : 1);
      totalGB += payload;
      const candidate = p.mapping === "skewed" ? 0 : gpu % p.railCount;
      const selected = p.failure && p.reroute ? (live.length ? live[p.mapping === "skewed" ? 0 : gpu % live.length] : null) : rails[candidate];
      const blocked = selected === null || selected.failed;
      if (selected !== null) {
        selected.gpus.push(gpu);
        selected.payloadGB += payload;
      }
      if (blocked) blockedGB += payload;
      assignments.push({ gpu, rail: selected === null ? null : selected.rail, payloadGB: payload, blocked });
    }
    const railGBps = p.rate / 8 * p.efficiency;
    const maxRailSeconds = Math.max(0, ...live.map((rail) => rail.payloadGB / railGBps));
    const fabricSeconds = totalGB / p.fabricGBps;
    const lowerBoundSeconds = blockedGB > 0 ? null : Math.max(maxRailSeconds, fabricSeconds);
    return { rails, assignments, totalGB, blockedGB, railGBps, maxRailSeconds, fabricSeconds, lowerBoundMs: lowerBoundSeconds === null ? null : lowerBoundSeconds * 1000, effectiveGBps: lowerBoundSeconds === null ? null : totalGB / lowerBoundSeconds };
  }
  function goodput({ batch, base, slope, knee, ttftLimit, tpotLimit }) {
    integer(batch, "batch", 1, 4096);
    [base, slope, knee, ttftLimit, tpotLimit].forEach((v) => number(v, "goodput parameter", 0.001));
    const requests = Array.from({ length: batch }, (_, i) => ({
      ttft: 100 + batch * 4 + (i * 13 % 29) * 3,
      tpot: base + slope * batch + Math.max(0, batch - knee) * slope * 2 + (i * 7 % 13) * 0.2,
      output: 128
    }));
    const meanTPOT = requests.reduce((sum, r) => sum + r.tpot, 0) / batch;
    const throughput = batch * 1000 / meanTPOT;
    const passed = requests.filter((r) => r.ttft <= ttftLimit && r.tpot <= tpotLimit).length;
    return { requests, meanTPOT, throughput, passed, goodput: throughput * passed / batch };
  }
  function cost({ hourly, outputTPS, inputTPS, hours, priceIn, priceOut }) {
    [hourly, outputTPS, inputTPS, hours, priceIn, priceOut].forEach((v) => number(v, "cost parameter"));
    return { perMillionOutput: outputTPS === 0 ? null : hourly * 1e6 / (3600 * outputTPS), periodCost: hours * 3600 * (inputTPS * priceIn + outputTPS * priceOut) / 1e6 };
  }
  function congestion({ queue, ecn, pfc, xon }, wasPaused = false) {
    [queue, ecn, pfc, xon].forEach((v) => number(v, "threshold", 0, 100));
    if (ecn >= pfc || xon >= pfc) throw new RangeError("ECN and Xon must be below PFC Xoff");
    const paused = queue >= pfc || (wasPaused && queue > xon);
    return { paused, marked: queue >= ecn, recovering: wasPaused && !paused };
  }
  function pfcBudget({ bufferMB, rate, reactionUs, frameBytes, guardMB, pfc }) {
    number(bufferMB, "teaching queue buffer MB", 0.001);
    number(rate, "link Gb/s", 0.001);
    number(reactionUs, "pause reaction microseconds");
    integer(frameBytes, "maximum frame bytes", 64);
    number(guardMB, "pipeline guard MB");
    number(pfc, "PFC Xoff percentage", 0, 100);
    const requiredMB = (rate * 1e9 / 8 * reactionUs * 1e-6 + frameBytes) / 1e6 + guardMB;
    const availableMB = bufferMB * (1 - pfc / 100);
    return { requiredMB, availableMB, marginMB: availableMB - requiredMB, fits: availableMB >= requiredMB };
  }
  function pipeline({ ranks, batch }) {
    integer(ranks, "pipeline stages", 2, 128);
    integer(batch, "microbatches", 1, 4096);
    const slots = batch + ranks - 1;
    const schedule = Array.from({ length: ranks }, (_, stage) =>
      Array.from({ length: batch }, (_, microbatch) => ({ stage, microbatch, start: stage + microbatch, end: stage + microbatch + 1 })));
    return { slots, utilization: batch / slots, latencySlots: ranks, idleSlots: ranks * (ranks - 1), schedule };
  }
  function speculative({ draftTokens, acceptance, targetMs, draftMs, verifyExtraMs, overheadMs }) {
    integer(draftTokens, "draft tokens", 1, 32);
    number(acceptance, "acceptance", 0, 1);
    number(targetMs, "target time", 0.001);
    [draftMs, verifyExtraMs, overheadMs].forEach((v) => number(v, "cycle cost"));
    const survival = Array.from({ length: draftTokens + 1 }, (_, i) => acceptance ** i);
    const expectedTokens = survival.reduce((sum, value) => sum + value, 0);
    const probabilities = survival.map((value, i) => i === draftTokens ? value : value * (1 - acceptance));
    const verificationMs = targetMs + draftTokens * verifyExtraMs;
    const cycleMs = draftTokens * draftMs + verificationMs + overheadMs;
    return { expectedTokens, survival, probabilities, verificationMs, cycleMs, meanServiceMs: cycleMs / expectedTokens, speedup: targetMs * expectedTokens / cycleMs, userTPS: 1000 * expectedTokens / cycleMs };
  }
  function prefixCaching({ model: key, kvBytes, prompt, prefix, hitRate, prefillRate, lookupMs, entries }) {
    const model = preset(models, key);
    integer(prompt, "prompt tokens", 1);
    integer(prefix, "shared prefix tokens", 0, prompt);
    number(hitRate, "prefix hit rate", 0, 1);
    number(prefillRate, "prefill rate", 0.001);
    number(lookupMs, "cache lookup");
    integer(entries, "distinct prefix cache entries", 1);
    const baselineMs = prompt / prefillRate * 1000;
    const hitMs = (prompt - prefix) / prefillRate * 1000 + lookupMs;
    const missMs = baselineMs + lookupMs;
    const expectedMs = hitRate * hitMs + (1 - hitRate) * missMs;
    const avoidedTokens = hitRate * prefix;
    const cacheGB = kvPerToken(model, kvBytes) * prefix * entries / 1e9;
    return { baselineMs, hitMs, missMs, expectedMs, avoidedTokens, inputComputeTokens: prompt - avoidedTokens, savedMs: baselineMs - expectedMs, cacheGB, contextSupported: prompt <= model.contextLimit };
  }
  const api = { models, gpus, traceSamples, workloadProfiles, ptuModels, number, integer, lifecycle, traceMetrics, capacity, mixtureCapacity, burstQueue, businessQPS, ptu, normalizedPTU, kvPerToken, moeUnion, sizing, rankMemory, transfer, pdBudget, percentile, pdSamples, collective, pcieBandwidth, networkBudget, railBudget, goodput, cost, congestion, pfcBudget, pipeline, speculative, prefixCaching };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.AcademyMath = Object.freeze(api);
})(globalThis);
