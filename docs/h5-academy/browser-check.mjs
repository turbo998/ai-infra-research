import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const directory = dirname(fileURLToPath(import.meta.url));
const edge = join(process.env["ProgramFiles(x86)"] || process.env.ProgramFiles || "", "Microsoft", "Edge", "Application", "msedge.exe");
if (!existsSync(edge)) throw new Error("This offline browser check requires an installed Windows Microsoft Edge.");
const captureIndex = process.argv.indexOf("--screenshots");
const screenshots = captureIndex < 0 ? null : process.argv[captureIndex + 1];
if (captureIndex >= 0 && (!screenshots || !existsSync(screenshots))) throw new Error("--screenshots requires an existing output directory.");
const onlyIndex = process.argv.indexOf("--only");
const only = onlyIndex < 0 ? null : process.argv[onlyIndex + 1]?.split(",");
if (onlyIndex >= 0 && (!only || only.some((name)=>!readdirSync(directory).includes(name) || !name.endsWith(".html")))) throw new Error("--only requires comma-separated academy HTML filenames.");
const temporary = mkdtempSync(join(tmpdir(), "ai-infra-academy-check-"));

async function probe(exercise, expectedWidth) {
  const failures = [];
  let cases = 0;
  const assert = (condition, message) => { if (!condition) failures.push(message); };
  const wait = () => new Promise((resolve) => setTimeout(resolve, 0));
  const rgba = (text) => {
    const values=text.match(/[\d.]+/g)?.map(Number);
    if (!values || values.length<3) throw new Error("Cannot parse rendered color");
    return [values[0],values[1],values[2],values[3]===undefined?1:values[3]];
  };
  const background = (el) => {
    if (!el) throw new Error("Rendered background is unresolved");
    const value=rgba(getComputedStyle(el).backgroundColor);
    if (value[3]>=1) return value.slice(0,3);
    const behind=background(el.parentElement);
    return behind.map((v,i)=>value[i]*value[3]+v*(1-value[3]));
  };
  const luminance = (color) => {
    const c=color.slice(0,3).map((v)=>v/255).map((v)=>v<=0.04045?v/12.92:((v+0.055)/1.055)**2.4);
    return 0.2126*c[0]+0.7152*c[1]+0.0722*c[2];
  };
  const contrast = (a,b) => (Math.max(luminance(a),luminance(b))+0.05)/(Math.min(luminance(a),luminance(b))+0.05);
  const validate = (form) => {
    const error = form.closest("[data-workbench]").querySelector(".error");
    const results = form.closest("[data-workbench]").querySelector("[data-results]");
    assert(error.hidden ? results.children.length > 0 : results.getAttribute("aria-invalid") === "true", "Inconsistent error/results state: " + form.dataset.lab);
    assert(results.getAttribute("aria-live")==="off", "Large result tables should not be live-announced as a whole");
    if (error.hidden) assert(form.closest("[data-workbench]").querySelector("[data-result-summary]").textContent.length>0, "Concise result status is missing");
    assert(!/NaN|Infinity/.test(results.innerHTML), "Non-finite result: " + form.dataset.lab);
    for (const axis of results.querySelectorAll("svg path.axis")) {
      assert(getComputedStyle(axis).fill === "none", "An open SVG axis was incorrectly filled as a data area");
    }
    for (const el of results.querySelectorAll("svg [x], svg [y], svg [width], svg [height], svg [cx], svg [cy]")) {
      for (const key of ["x", "y", "width", "height", "cx", "cy"]) {
        if (el.hasAttribute(key)) assert(Number.isFinite(Number(el.getAttribute(key))), "Non-finite SVG coordinate");
      }
    }
  };
  try {
    assert(window.AcademyMath && window.AcademyUI, "Math/UI initialization missing");
    assert(innerWidth === expectedWidth, "Requested viewport width is not active: " + innerWidth);
    if (innerWidth!==expectedWidth) {
      const overflow=[...document.querySelectorAll("main,.hero,.hero-aside,.lesson,.lab,.controls,.control,.results,.kpi,.sources,.toc")].map((el)=>({id:el.id,cls:el.className,right:el.getBoundingClientRect().right,width:el.getBoundingClientRect().width,scroll:el.scrollWidth})).filter((el)=>el.right>expectedWidth+1||el.scroll>el.width+1).slice(0,12);
      failures.push("Narrow-layout diagnostics: "+JSON.stringify(overflow));
    }
    assert(matchMedia("(max-width:600px)").matches === (expectedWidth <= 600), "Requested viewport media query is not active");
    assert(document.documentElement.scrollWidth <= innerWidth + 1, "Page-level horizontal overflow");
    assert(document.querySelector("#runtime-error").hidden, "Runtime error banner visible");
    for (const el of document.querySelectorAll(".lead, .kpi span, footer, a")) {
      if (!el.getClientRects().length) continue;
      const style=getComputedStyle(el);
      const minimum=parseFloat(style.fontSize)>=24 || (parseFloat(style.fontSize)>=18.66 && Number(style.fontWeight)>=700)?3:4.5;
      assert(contrast(rgba(style.color),background(el))>=minimum, "Rendered text contrast below target: "+el.tagName+" "+el.className);
    }
    for (const el of document.querySelectorAll("svg .line.primary, svg .line.secondary, svg .line.tertiary, svg path.axis")) {
      if (!el.getClientRects().length) continue;
      assert(contrast(rgba(getComputedStyle(el).stroke),background(el))>=3, "Meaningful chart stroke contrast below3:1");
    }
    const forms = [...document.querySelectorAll("form[data-lab]")];
    for (const form of forms) {
      validate(form);
      assert(form.closest("[data-workbench]").querySelector(".error").hidden, "Default example invalid: " + form.dataset.lab);
    }
    if (exercise) {
      for (const form of forms) {
        for (const input of form.querySelectorAll("[data-key]")) {
          if (input.disabled || input.tagName === "TEXTAREA") continue;
          const choices = input.tagName === "SELECT" ? [...input.options].map((o) => o.value) : input.type === "checkbox" ? [false, true] : [input.min, input.max];
          for (const value of choices) {
            form.reset();
            await wait();
            if (input.disabled) continue;
            if (input.type === "checkbox") input.checked = value;
            else input.value = String(value);
            input.dispatchEvent(new Event("input", { bubbles: true }));
            validate(form);
            cases += 1;
          }
          form.reset();
          await wait();
        }
        const numeric = [...form.querySelectorAll('input[type="number"]')].find((el) => !el.disabled);
        if (numeric) {
          numeric.value = "";
          numeric.dispatchEvent(new Event("input", { bubbles: true }));
          assert(!form.closest("[data-workbench]").querySelector(".error").hidden, "Blank input did not surface an error");
          cases += 1;
          form.reset();
          await wait();
        }
      }
      for (const quiz of document.querySelectorAll("[data-quiz]")) {
        quiz.querySelector("[data-check-quiz]").click();
        assert(quiz.querySelector(".quiz-result").textContent.length > 0, "Unanswered quiz has no prompt");
        quiz.querySelector(`input[value="${quiz.dataset.correct}"]`).checked = true;
        quiz.querySelector("[data-check-quiz]").click();
        assert(/Correct|正确/.test(quiz.querySelector(".quiz-result").textContent), "Correct quiz answer failed");
        cases += 2;
      }
      const parallel = document.querySelector('form[data-lab="parallel"]');
      if (parallel) {
        const kind = parallel.querySelector('[data-key="kind"]');
        for (const value of ["tp", "pp", "dp", "ep"]) {
          kind.value = value;
          kind.dispatchEvent(new Event("input", { bubbles: true }));
          assert(document.querySelector(`[data-parallel-kind="${value}"]`), "Parallel view did not change: " + value);
          if (value === "pp") {
            const ranks = Number(parallel.querySelector('[data-key="ranks"]').value);
            assert(document.querySelectorAll('[data-microbatch="0"]').length === ranks, "First PP microbatch does not traverse every stage");
          }
          cases += 1;
        }
        const speculative = document.querySelector('[data-results="speculative"]');
        if (speculative) {
          const expected = document.documentElement.lang === "en" ? "Break-even" : "收支平衡";
          assert(speculative.textContent.includes(expected), "Speculation plot mislabeled its economic break-even line as an SLO");
          cases += 1;
        }
        kind.value = "dp";
        kind.dispatchEvent(new Event("input", { bubbles: true }));
        const attention = parallel.querySelector('[data-key="attentionDP"]');
        assert(!attention.disabled, "Attention DP toggle should be enabled in DP mode");
        attention.checked = true;
        attention.dispatchEvent(new Event("input", { bubbles: true }));
        assert(document.querySelector('[data-attention-dp="true"]'), "Attention DP shared-expert view is missing");
        parallel.reset();
        await wait();
        cases += 1;
      }
      const rank = document.querySelector('form[data-lab="rank"]');
      if (rank) {
        const set = (key, value) => {
          const input=rank.querySelector(`[data-key="${key}"]`);
          if (input.type === "checkbox") input.checked=value;
          else input.value=String(value);
          input.dispatchEvent(new Event("input",{bubbles:true}));
        };
        set("model", "deepseek");
        set("tp", 8);
        set("mlaCache", "measured");
        assert(!rank.querySelector('[data-key="mlaLayerBytes"]').disabled, "Opaque MLA row-byte input did not enable");
        set("mlaLayerBytes", 656);
        set("kvBytes", 1);
        validate(rank);
        set("measuredMemory", true);
        set("deviceGB", 180);
        assert(!rank.querySelector('[data-key="deviceGB"]').disabled, "Measured capacity input did not enable");
        validate(rank);
        set("customReplicaBatch", true);
        set("dp", 8);
        set("replicaBatch", 32);
        validate(rank);
        rank.reset();
        await wait();
        set("model", "");
        assert(!rank.closest("[data-workbench]").querySelector(".error").hidden, "Unknown model selection did not surface an error");
        rank.reset();
        await wait();
        cases += 4;
      }
      const trace = document.querySelector('form[data-lab="trace"]');
      if (trace) {
        const gate=trace.querySelector('[data-key="gateMode"]');
        gate.value="mean";
        gate.dispatchEvent(new Event("input",{bubbles:true}));
        assert(trace.closest("[data-workbench]").querySelector("[data-results]").textContent.includes("9 tok/s"), "Mean TPOT trace gate did not retain bursty goodput");
        gate.value="max";
        gate.dispatchEvent(new Event("input",{bubbles:true}));
        assert(trace.closest("[data-workbench]").querySelector("[data-results]").textContent.includes("5 tok/s"), "Max ITL trace gate did not remove bursty goodput");
        const input=trace.querySelector('textarea[data-key="trace"]');
        input.value="invalid JSON";
        input.dispatchEvent(new Event("input",{bubbles:true}));
        assert(!trace.closest("[data-workbench]").querySelector(".error").hidden, "Malformed trace JSON did not surface an error");
        trace.reset();
        await wait();
        validate(trace);
        cases += 3;
      }
      const pd = document.querySelector('form[data-lab="pd"]');
      const pdPlan = document.querySelector('form[data-lab="pdBudget"]');
      if (pd && pdPlan) {
        const wire=pd.querySelector('[data-key="wireCopies"]');
        wire.value="4";
        wire.dispatchEvent(new Event("input",{bubbles:true}));
        assert(document.querySelector('[data-pd-readiness="blocked-capacity"]'), "PD plan did not update from the shared wire multiplier");
        pd.reset();
        await wait();
        const origin=pdPlan.querySelector('[data-key="firstVisible"]');
        origin.value="prefill";
        origin.dispatchEvent(new Event("input",{bubbles:true}));
        assert(document.querySelector('[data-pd-readiness="blocked-latency"]'), "Early P token incorrectly hid the first streaming gap");
        pdPlan.reset();
        await wait();
        const layout=pd.querySelector('[data-key="customKVBytes"]');
        layout.checked=true;
        layout.dispatchEvent(new Event("input",{bubbles:true}));
        assert(!pd.querySelector('[data-key="bytesPerToken"]').disabled, "Custom KV geometry input did not enable");
        validate(pd);
        validate(pdPlan);
        pd.reset();
        await wait();
        const zoom=pd.querySelector('[data-key="hideColocated"]');
        zoom.checked=true;
        zoom.dispatchEvent(new Event("input",{bubbles:true}));
        const lines=[...pd.closest("[data-workbench]").querySelectorAll('figure:nth-of-type(2) polyline')];
        assert(lines.length===2 && lines.some((line)=>line.classList.contains("secondary")) && lines.some((line)=>line.classList.contains("tertiary")), "Optimized-path zoom changed series colors or retained the omitted curve");
        pd.reset();
        await wait();
        cases += 4;
      }
      const network = document.querySelector('form[data-lab="network"]');
      if (network) {
        const operation=network.querySelector('[data-key="operation"]');
        operation.value="allgather";
        operation.dispatchEvent(new Event("input",{bubbles:true}));
        const mode=network.querySelector('[data-key="sizeMode"]');
        mode.value="api-count";
        mode.dispatchEvent(new Event("input",{bubbles:true}));
        assert(document.querySelector('[data-effective-size="8000"]'), "API count did not normalize to the total gathered array");
        network.reset();
        await wait();
        cases += 1;
      }
      const rail = document.querySelector('form[data-lab="rail"]');
      if (rail) {
        const failure=rail.querySelector('[data-key="failure"]');
        const reroute=rail.querySelector('[data-key="reroute"]');
        failure.checked=true;
        failure.dispatchEvent(new Event("input",{bubbles:true}));
        reroute.checked=false;
        reroute.dispatchEvent(new Event("input",{bubbles:true}));
        assert(document.querySelector('[data-rail-complete="false"]'), "Failed fixed rail mapping fabricated a complete transfer");
        reroute.checked=true;
        reroute.dispatchEvent(new Event("input",{bubbles:true}));
        assert(document.querySelector('[data-rail-complete="true"]'), "Toy failover did not restore a mapped path");
        rail.reset();
        await wait();
        cases += 2;
      }
      const congestion = document.querySelector('form[data-lab="congestion"]');
      if (congestion) {
        assert(congestion.closest("[data-workbench]").querySelector('[data-signal="pfc-hop"]'), "PFC causal path was incorrectly chained after sender DCQCN");
        const rate=congestion.querySelector('[data-key="rate"]');
        const reaction=congestion.querySelector('[data-key="reactionUs"]');
        rate.value="800";
        rate.dispatchEvent(new Event("input",{bubbles:true}));
        reaction.value="100";
        reaction.dispatchEvent(new Event("input",{bubbles:true}));
        assert(document.querySelector('[data-headroom-fits="false"]'), "Insufficient PFC headroom was accepted");
        congestion.reset();
        await wait();
        cases += 1;
      }
      const oldLanguage = document.documentElement.lang;
      document.querySelector("#language").click();
      assert(document.documentElement.lang !== oldLanguage, "Language did not change");
      document.querySelector("#language").click();
      const oldTheme = document.documentElement.dataset.theme;
      document.querySelector("#theme").click();
      assert(document.documentElement.dataset.theme !== oldTheme, "Theme did not change");
      document.querySelector("#theme").click();
      cases += 4;
      for (const a of document.querySelectorAll("a[data-carry]")) assert(new URL(a.href).searchParams.has("lang"), "Navigation lost language");
    }
    assert(window.AcademyDiagnostics.length === 0, "JS/runtime diagnostics: " + window.AcademyDiagnostics.join("; "));
    assert(window.AcademyPolicyDiagnostics.length === 0, "Unexpected CSP policy violation in true offline Edge");
    assert(performance.getEntriesByType("resource").length === 0, "External resource request in offline document");
    if (expectedWidth <= 600) {
      const rail = document.querySelector(".chapter-rail");
      if (rail) {
        assert(rail.scrollWidth > rail.clientWidth, "Mobile chapter cards have no horizontal rail");
        assert(getComputedStyle(rail).scrollSnapType.includes("mandatory"), "Mobile card rail has no snap behavior");
        rail.scrollLeft = rail.scrollWidth - rail.clientWidth;
        assert(rail.scrollLeft > 0, "Mobile chapter rail cannot scroll");
        rail.scrollLeft = 0;
        cases += 1;
      }
      const chart = [...document.querySelectorAll(".chart")].find((el) => el.scrollWidth > el.clientWidth);
      if (chart) {
        chart.scrollLeft = 50;
        assert(chart.scrollLeft > 0, "Wide mobile chart cannot scroll within its frame");
        chart.scrollLeft = 0;
        cases += 1;
      }
    }
    assert(document.documentElement.scrollWidth <= innerWidth + 1, "Page overflow after interactions");
  } catch (error) {
    failures.push(error.stack || error.message);
  }
  const result = document.createElement("pre");
  result.id = "academy-browser-check";
  result.hidden = true;
  result.textContent = JSON.stringify({ failures, cases, width: innerWidth, language: document.documentElement.lang, theme: document.documentElement.dataset.theme, errors: window.AcademyDiagnostics, policy: window.AcademyPolicyDiagnostics });
  document.body.append(result);
}

let failed = 0;
let totalCases = 0;
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const profile = join(temporary, "edge-profile");
const child = spawn(edge, ["--headless=new", "--disable-gpu", "--disable-extensions", "--disable-background-networking", "--disable-component-update", "--disable-default-apps", "--disable-sync", "--no-first-run", "--no-default-browser-check", `--user-data-dir=${profile}`, "--remote-debugging-address=127.0.0.1", "--remote-debugging-port=0", "about:blank"], { windowsHide: true, stdio: "ignore" });
let launchError = null;
child.on("error", (error) => { launchError = error; });
let client;
class DebugClient {
  constructor(socket) {
    this.socket = socket;
    this.sequence = 0;
    this.pending = new Map();
    this.exceptions = [];
    socket.addEventListener("message", (event) => {
      const message = JSON.parse(String(event.data));
      if (message.method === "Runtime.exceptionThrown") this.exceptions.push(message.params.exceptionDetails.text);
      const pending = this.pending.get(message.id);
      if (pending) {
        clearTimeout(pending.timer);
        this.pending.delete(message.id);
        if (message.error) pending.reject(new Error(message.error.message));
        else pending.resolve(message.result);
      }
    });
  }
  send(method, params = {}, sessionId) {
    const id = ++this.sequence;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error("Browser command timed out: " + method));
      }, 15000);
      this.pending.set(id, { resolve, reject, timer });
      this.socket.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
    });
  }
}
try {
  const deadline = Date.now() + 30000;
  const activePort = join(profile, "DevToolsActivePort");
  while (!existsSync(activePort)) {
    if (launchError) throw launchError;
    if (Date.now() >= deadline) throw new Error("Isolated Edge debugging endpoint did not become ready.");
    await delay(100);
  }
  const port = Number(readFileSync(activePort, "utf8").split(/\r?\n/)[0]);
  const response = await fetch(`http://127.0.0.1:${port}/json/version`, { signal: AbortSignal.timeout(10000) });
  if (!response.ok) throw new Error("Isolated Edge debugging discovery failed.");
  const { webSocketDebuggerUrl } = await response.json();
  const socket = new WebSocket(webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error("Isolated Edge connection timed out.")), 10000);
    socket.addEventListener("open", () => { clearTimeout(timer); resolve(); }, { once: true });
    socket.addEventListener("error", () => { clearTimeout(timer); reject(new Error("Isolated Edge connection failed.")); }, { once: true });
  });
  client = new DebugClient(socket);
  const files = readdirSync(directory).filter((file) => file.endsWith(".html") && (!only || only.includes(file))).sort();
  for (const file of files) {
    for (const variant of [
      { width: 1440, language: "zh", theme: "light", exercise: true },
      { width: 390, language: "en", theme: "dark", exercise: true },
      { width: 320, language: "en", theme: "dark", exercise: false },
      { width: 360, language: "zh", theme: "light", exercise: false }
    ]) {
      const script = `(${probe.toString()})(${variant.exercise},${variant.width});`;
      const hash = createHash("sha256").update(script).digest("base64");
      const original = readFileSync(join(directory, file), "utf8");
      const instrumented = original.replace("script-src ", `script-src 'sha256-${hash}' `).replace("</body>", `<script>${script}</script></body>`);
      const path = join(temporary, file);
      writeFileSync(path, instrumented, "utf8");
      const url = pathToFileURL(path);
      url.searchParams.set("lang", variant.language);
      url.searchParams.set("scoutTheme", variant.theme);
      const { targetId } = await client.send("Target.createTarget", { url: "about:blank" });
      const { sessionId } = await client.send("Target.attachToTarget", { targetId, flatten: true });
      await client.send("Emulation.setDeviceMetricsOverride", { width: variant.width, height: 1000, deviceScaleFactor: 1, mobile: variant.width <= 600 }, sessionId);
      await client.send("Runtime.enable", {}, sessionId);
      const exceptionStart = client.exceptions.length;
      await client.send("Page.navigate", { url: url.href }, sessionId);
      const probeDeadline = Date.now() + 60000;
      let resultText = null;
      while (resultText === null) {
        const { result } = await client.send("Runtime.evaluate", { expression: "document.getElementById('academy-browser-check')?.textContent ?? null", returnByValue: true }, sessionId);
        resultText = result.value;
        if (resultText === null || resultText === undefined) {
          if (Date.now() >= probeDeadline) throw new Error("Browser probe did not finish: " + file);
          resultText = null;
          await delay(50);
        }
      }
      const result = JSON.parse(resultText);
      result.failures.push(...client.exceptions.slice(exceptionStart));
      await client.send("Input.dispatchKeyEvent", { type: "keyDown", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9 }, sessionId);
      await client.send("Input.dispatchKeyEvent", { type: "keyUp", key: "Tab", code: "Tab", windowsVirtualKeyCode: 9 }, sessionId);
      const { result: focusTarget } = await client.send("Runtime.evaluate", {
        expression: "(() => { const input=document.querySelector('form[data-lab] input[type=\"range\"]:not(:disabled)'); const el=input || document.getElementById('language'); el.focus(); return {id:el.id,kind:input?'range':'language',before:input?input.value:document.documentElement.lang}; })()",
        returnByValue: true
      }, sessionId);
      const key = focusTarget.value.kind === "range" ? "ArrowRight" : "Enter";
      const keyCode = key === "ArrowRight" ? 39 : 13;
      await client.send("Input.dispatchKeyEvent", { type: "keyDown", key, code: key, windowsVirtualKeyCode: keyCode, ...(key === "Enter" ? { text: "\r", unmodifiedText: "\r" } : {}) }, sessionId);
      await client.send("Input.dispatchKeyEvent", { type: "keyUp", key, code: key, windowsVirtualKeyCode: keyCode }, sessionId);
      const { result: keyOutcome } = await client.send("Runtime.evaluate", {
        expression: `(() => { const el=document.getElementById(${JSON.stringify(focusTarget.value.id)}); return {after:${focusTarget.value.kind === "range" ? "el.value" : "document.documentElement.lang"},outline:parseFloat(getComputedStyle(el).outlineWidth),errors:window.AcademyDiagnostics,overflow:document.documentElement.scrollWidth>innerWidth}; })()`,
        returnByValue: true
      }, sessionId);
      if (keyOutcome.value.after === focusTarget.value.before || keyOutcome.value.outline < 3 || keyOutcome.value.errors.length || keyOutcome.value.overflow) {
        result.failures.push("Native keyboard/focus or narrow-page reflow failed: " + JSON.stringify({ target: focusTarget.value, outcome: keyOutcome.value }));
      }
      if (focusTarget.value.kind === "range") {
        await client.send("Runtime.evaluate", { expression: `new Promise(resolve=>{document.getElementById(${JSON.stringify(focusTarget.value.id)}).form.reset();setTimeout(resolve,10);})`, awaitPromise: true }, sessionId);
      } else {
        await client.send("Input.dispatchKeyEvent", { type: "keyDown", key: "Enter", code: "Enter", windowsVirtualKeyCode: 13, text: "\r", unmodifiedText: "\r" }, sessionId);
        await client.send("Input.dispatchKeyEvent", { type: "keyUp", key: "Enter", code: "Enter", windowsVirtualKeyCode: 13 }, sessionId);
      }
      result.cases += 1;
      if (screenshots) {
        const { data } = await client.send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false }, sessionId);
        writeFileSync(join(screenshots, `${file.replace(".html", "")}-${variant.width}.png`), Buffer.from(data, "base64"));
        const { result: bounds } = await client.send("Runtime.evaluate", { expression: "(() => { const el=document.querySelector('[data-results] figure') || document.querySelector('.chapter-rail'); if (!el) return null; const r=el.getBoundingClientRect(); return {x:r.left+scrollX,y:r.top+scrollY,width:r.width,height:r.height,scale:1}; })()", returnByValue: true }, sessionId);
        if (bounds.value) {
          const { data: figure } = await client.send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true, clip: bounds.value }, sessionId);
          writeFileSync(join(screenshots, `${file.replace(".html", "")}-${variant.width}-figure.png`), Buffer.from(figure, "base64"));
        }
        if (file === "01-metrics.html") {
          const { result: traceBounds } = await client.send("Runtime.evaluate", {
            expression: "(() => { const r=document.querySelector('[data-results=\"trace\"] figure').getBoundingClientRect(); return {x:r.left+scrollX,y:r.top+scrollY,width:r.width,height:r.height,scale:1}; })()",
            returnByValue: true
          }, sessionId);
          const { data: traceChart } = await client.send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true, clip: traceBounds.value }, sessionId);
          writeFileSync(join(screenshots, `${file.replace(".html", "")}-${variant.width}-trace.png`), Buffer.from(traceChart, "base64"));
        }
        const { result: figureBounds } = await client.send("Runtime.evaluate", {
          expression: "(() => { const items=[]; for (const result of document.querySelectorAll('[data-results]')) { let i=0; for (const el of result.querySelectorAll('figure')) { const r=el.getBoundingClientRect(); items.push({name:result.dataset.results+'-'+i++,clip:{x:r.left+scrollX,y:r.top+scrollY,width:r.width,height:r.height,scale:1}}); } } return items; })()",
          returnByValue: true
        }, sessionId);
        for (const item of figureBounds.value) {
          const { data: figure } = await client.send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true, clip: item.clip }, sessionId);
          writeFileSync(join(screenshots, `${file.replace(".html", "")}-${variant.width}-${item.name}.png`), Buffer.from(figure, "base64"));
        }
        if (file === "05-parallelism-acceleration.html") {
          for (const view of ["tp", "pp", "dp", "ep", "attention-dp"]) {
            const { result: viewBounds } = await client.send("Runtime.evaluate", {
              expression: `(() => {
                const form=document.querySelector('form[data-lab="parallel"]');
                const kind=form.querySelector('[data-key="kind"]');
                kind.value=${JSON.stringify(view === "attention-dp" ? "dp" : view)};
                kind.dispatchEvent(new Event("input",{bubbles:true}));
                const attention=form.querySelector('[data-key="attentionDP"]');
                if (!attention.disabled) { attention.checked=${view === "attention-dp"}; attention.dispatchEvent(new Event("input",{bubbles:true})); }
                const r=document.querySelector('[data-results="parallel"] figure').getBoundingClientRect();
                return {x:r.left+scrollX,y:r.top+scrollY,width:r.width,height:r.height,scale:1};
              })()`,
              returnByValue: true
            }, sessionId);
            const { data: diagram } = await client.send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true, clip: viewBounds.value }, sessionId);
            writeFileSync(join(screenshots, `${file.replace(".html", "")}-${variant.width}-${view}.png`), Buffer.from(diagram, "base64"));
          }
        }
      }
      await client.send("Target.closeTarget", { targetId });
      totalCases += result.cases;
      if (result.failures.length) {
        failed += 1;
        console.error(JSON.stringify({ file, width: variant.width, failures: result.failures }));
      } else console.log(`PASS ${file} ${result.width}px ${result.language}/${result.theme}: ${result.cases} interactions, zero JS/CSP/resource errors`);
    }
  }
} finally {
  let closeError;
  if (client) {
    try { await client.send("Browser.close"); }
    catch (error) { closeError = error; }
    finally { client.socket.close(); }
  } else if (child.exitCode === null) child.kill();
  for (let i = 0; child.exitCode === null && i < 50; i += 1) await delay(100);
  // Windows browser children can release profile handles after the launcher exits.
  const cleanupDeadline = Date.now() + 30000;
  for (;;) {
    try {
      rmSync(temporary, { recursive: true, force: true, maxRetries: 2, retryDelay: 100 });
      break;
    } catch (error) {
      if (!["EPERM", "EBUSY", "ENOTEMPTY"].includes(error.code) || Date.now() >= cleanupDeadline) {
        if (closeError) throw new AggregateError([closeError, error], "Browser shutdown and exact temporary-profile cleanup failed");
        throw error;
      }
      await delay(250);
    }
  }
  if (closeError) throw closeError;
}
console.log(`Offline browser verification: ${totalCases} interaction cases; ${failed} failed variants.`);
if (failed) process.exitCode = 1;
