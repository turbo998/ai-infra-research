import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const directory = dirname(fileURLToPath(import.meta.url));
const edge = join(process.env["ProgramFiles(x86)"] || process.env.ProgramFiles || "", "Microsoft", "Edge", "Application", "msedge.exe");
if (!existsSync(edge)) throw new Error("This offline browser check requires an installed Windows Microsoft Edge.");
const temporary = mkdtempSync(join(tmpdir(), "ai-infra-academy-check-"));
const captureIndex = process.argv.indexOf("--screenshots");
const screenshots = captureIndex < 0 ? null : process.argv[captureIndex + 1];
if (captureIndex >= 0 && (!screenshots || !existsSync(screenshots))) throw new Error("--screenshots requires an existing output directory.");

async function probe(exercise, expectedWidth) {
  const failures = [];
  let cases = 0;
  const assert = (condition, message) => { if (!condition) failures.push(message); };
  const wait = () => new Promise((resolve) => setTimeout(resolve, 0));
  const validate = (form) => {
    const error = form.closest("[data-workbench]").querySelector(".error");
    const results = form.closest("[data-workbench]").querySelector("[data-results]");
    assert(error.hidden ? results.children.length > 0 : results.getAttribute("aria-invalid") === "true", "Inconsistent error/results state: " + form.dataset.lab);
    assert(!/NaN|Infinity/.test(results.innerHTML), "Non-finite result: " + form.dataset.lab);
    for (const el of results.querySelectorAll("svg [x], svg [y], svg [width], svg [height], svg [cx], svg [cy]")) {
      for (const key of ["x", "y", "width", "height", "cx", "cy"]) {
        if (el.hasAttribute(key)) assert(Number.isFinite(Number(el.getAttribute(key))), "Non-finite SVG coordinate");
      }
    }
  };
  try {
    assert(window.AcademyMath && window.AcademyUI, "Math/UI initialization missing");
    assert(innerWidth === expectedWidth, "Requested viewport width is not active: " + innerWidth);
    assert(matchMedia("(max-width:600px)").matches === (expectedWidth <= 600), "Requested viewport media query is not active");
    assert(document.documentElement.scrollWidth <= innerWidth + 1, "Page-level horizontal overflow");
    assert(document.querySelector("#runtime-error").hidden, "Runtime error banner visible");
    const forms = [...document.querySelectorAll("form[data-lab]")];
    for (const form of forms) {
      validate(form);
      assert(form.closest("[data-workbench]").querySelector(".error").hidden, "Default example invalid: " + form.dataset.lab);
    }
    if (exercise) {
      for (const form of forms) {
        for (const input of form.querySelectorAll("[data-key]")) {
          if (input.disabled) continue;
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
    assert(document.documentElement.scrollWidth <= innerWidth + 1, "Page overflow after interactions");
  } catch (error) {
    failures.push(error.stack || error.message);
  }
  const result = document.createElement("pre");
  result.id = "academy-browser-check";
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
  const files = readdirSync(directory).filter((file) => file.endsWith(".html")).sort();
  for (const file of files) {
    for (const variant of [
      { width: 1440, language: "zh", theme: "light", exercise: true },
      { width: 390, language: "en", theme: "dark", exercise: false }
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
      if (screenshots) {
        const { data } = await client.send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false }, sessionId);
        writeFileSync(join(screenshots, `${file.replace(".html", "")}-${variant.width}.png`), Buffer.from(data, "base64"));
        const { result: bounds } = await client.send("Runtime.evaluate", { expression: "(() => { const el=document.querySelector('[data-results] figure') || document.querySelector('.chapter-rail'); if (!el) return null; const r=el.getBoundingClientRect(); return {x:r.left+scrollX,y:r.top+scrollY,width:r.width,height:r.height,scale:1}; })()", returnByValue: true }, sessionId);
        if (bounds.value) {
          const { data: figure } = await client.send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true, clip: bounds.value }, sessionId);
          writeFileSync(join(screenshots, `${file.replace(".html", "")}-${variant.width}-figure.png`), Buffer.from(figure, "base64"));
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
  if (client) {
    try { await client.send("Browser.close"); }
    finally { client.socket.close(); }
  } else if (child.exitCode === null) child.kill();
  for (let i = 0; child.exitCode === null && i < 50; i += 1) await delay(100);
  // Only remove the exact temporary directory created for this check.
  rmSync(temporary, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
}
console.log(`Offline browser verification: ${totalCases} interaction cases; ${failed} failed variants.`);
if (failed) process.exitCode = 1;
