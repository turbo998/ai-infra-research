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
    const inputInstances = Math.ceil(inputTPS / (p.inputCapacity * p.utilization));
    const outputInstances = Math.ceil(outputTPS / (p.outputCapacity * p.utilization));
    const batchInstances = Math.ceil(active / p.batch);
    return { qps, inputTPS, outputTPS, inputTPM: inputTPS * 60, outputTPM: outputTPS * 60, rpm: qps * 60, residence, active, decodeActive, inputInstances, outputInstances, batchInstances, instances: Math.max(inputInstances, outputInstances, batchInstances) };
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
    const raw = normalized / capacity;
    const provisioned = normalized === 0 ? 0 : minimum + Math.max(0, Math.ceil((raw - minimum) / step)) * step;
    return { normalized, raw, provisioned };
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
    const bytes = kvPerToken(model, p.kvBytes) * p.context * (1 - p.cache);
    const bandwidth = p.rate * 1e9 / 8 * p.efficiency * p.rails;
    const rawMs = bytes / bandwidth * 1000;
    return { bytes, bandwidth, rawMs, exposedMs: Math.max(0, rawMs - p.overlap) + p.setup, utilization: bytes * p.qps / bandwidth, requiredGbps: bytes * p.qps * 8 / 1e9 / p.efficiency };
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
    for (let i = 0; i < 200; i += 1) {
      const interference = i % 20 === 0;
      const jitter = (i * 17 % 7) * 0.15;
      samples.colocated.push(base + jitter + (interference ? block : 0));
      samples.chunked.push(base + jitter + (interference ? burst : 0));
      samples.pd.push(base + jitter + (interference ? residual : 0));
    }
    return { samples, p99: Object.fromEntries(Object.entries(samples).map(([key, values]) => [key, percentile(values, 0.99)])) };
  }
  function collective({ operation, ranks, sizeMB, timeMs }) {
    integer(ranks, "ranks", 2, 65536);
    number(sizeMB, "sizeMB", 0.001);
    number(timeMs, "timeMs", 0.001);
    const factors = { allreduce: 2 * (ranks - 1) / ranks, allgather: (ranks - 1) / ranks, reducescatter: (ranks - 1) / ranks, alltoall: (ranks - 1) / ranks, broadcast: 1, reduce: 1, sendrecv: 1 };
    const factor = preset(factors, operation);
    const algbw = sizeMB / timeMs;
    return { algbw, busbw: algbw * factor, factor };
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
  const api = { models, gpus, traceSamples, number, integer, lifecycle, traceMetrics, capacity, businessQPS, ptu, kvPerToken, moeUnion, sizing, rankMemory, transfer, percentile, pdSamples, collective, goodput, cost, congestion, pipeline, speculative, prefixCaching };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.AcademyMath = Object.freeze(api);
})(globalThis);
