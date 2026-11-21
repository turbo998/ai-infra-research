import { readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { chapters, bodies, studyTasks, validateContentSources, b, esc, indexContent } from "./content.mjs";

const directory = dirname(fileURLToPath(import.meta.url));
const css = readFileSync(join(directory, "academy.css"), "utf8");
const math = readFileSync(join(directory, "academy-math.cjs"), "utf8");
const ui = readFileSync(join(directory, "academy-ui.js"), "utf8");
const sources = JSON.parse(readFileSync(join(directory, "sources.json"), "utf8"));
validateContentSources(sources);
const bootstrap = `(() => {
  const param = new URLSearchParams(window.location.search).get("scoutTheme");
  const theme = ["light", "dark"].includes(param) ? param : (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", theme);
  document.documentElement.lang = new URLSearchParams(window.location.search).get("lang") === "en" ? "en" : "zh-CN";
  window.AcademyDiagnostics = [];
  window.AcademyPolicyDiagnostics = [];
  const record = (message) => {
    window.AcademyDiagnostics.push(String(message));
    const el = document.getElementById("runtime-error");
    if (el) { el.hidden = false; el.textContent = "Runtime error / 运行错误: " + message; }
  };
  window.addEventListener("error", (event) => record(event.message));
  window.addEventListener("unhandledrejection", (event) => record(event.reason && event.reason.message || event.reason));
  window.addEventListener("securitypolicyviolation", (event) => {
    window.AcademyPolicyDiagnostics.push({ directive: event.violatedDirective, blockedURI: event.blockedURI });
    if (event.violatedDirective === "connect-src") {
      const el = document.getElementById("policy-notice");
      if (el) {
        el.hidden = false;
        el.textContent = "Offline CSP blocked a connection (for example, preview-host instrumentation). Course calculations do not require it. / 离线策略已阻止连接（如预览器注入请求），课程计算不依赖该连接。";
      }
    } else record("CSP: " + event.violatedDirective);
  });
})();`;
const hash = (script) => "'sha256-" + createHash("sha256").update(script).digest("base64") + "'";
const csp = `default-src 'none'; script-src ${[bootstrap, math, ui].map(hash).join(" ")}; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; base-uri 'none'; form-action 'none'`;

function sourceSection(ids) {
  return `<section class="lesson sources" id="sources"><h2>${b("依据、适用边界与复核日期", "Sources, boundaries and review date")}</h2><p>${b(`${esc(sources.reviewed_at)}复核。公式图与延迟曲线是原创教学模型，不是硬件测量。链接仅供主动查阅；离线阅读不发网络请求。`, `Reviewed ${esc(sources.reviewed_at)}. Formula charts and latency curves are original teaching models, not hardware measurements. Links are for intentional reference visits; offline reading sends no requests.`)}</p>${sources.sources.filter((s) => ids.includes(s.id)).map((s) => `<p><a href="${esc(s.url)}" rel="noreferrer">${esc(s.title)}</a> · ${esc(s.state)}<br>${esc(s.supports)}</p>`).join("")}</section>`;
}
function build(chapter, index) {
  const isIndex = !chapter;
  const titleZh = isIndex ? "AI Infra 推理性能、智算网络与容量测算精学馆" : chapter.zh;
  const titleEn = isIndex ? "AI Infra Inference, Fabric & Capacity Academy" : chapter.en;
  const body = isIndex ? indexContent() : bodies[chapter.id]();
  const sourceIds = isIndex ? sources.sources.map((s) => s.id) : chapter.sources;
  const task = isIndex ? "" : `<div class="study-task"><h3>${b("今晚先试这一件事", "First experiment tonight")}</h3><p>${b(studyTasks[chapter.id][1], studyTasks[chapter.id][2])}</p><a class="button" href="#${studyTasks[chapter.id][0]}">${b("打开默认算例", "Open default example")} →</a></div>`;
  const sectionIds = [...body.matchAll(/<section class="lesson" id="([^"]+)"><h2>(.*?)<\/h2>/g)].map((m) => [m[1], m[2]]);
  const navLink = (c, text) => `<a class="button" data-carry href="${c.file}">${text}</a>`;
  const navigation = isIndex ? "" : `<nav class="nav-footer" aria-label="Chapter navigation">${index > 0 ? navLink(chapters[index - 1], "← " + b("上一册", "Previous")) : '<a class="button" data-carry href="index.html">← Academy</a>'}${index < chapters.length - 1 ? navLink(chapters[index + 1], b("下一册", "Next") + " →") : '<a class="button" data-carry href="index.html">Academy →</a>'}</nav>`;
  const html = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="${esc(csp)}">
<meta name="description" content="Offline bilingual Cloud &amp; AI Solution Architect academy: metrics, TPM/TPS, GPU sizing, PD, parallelism and RDMA fabric.">
<title>${esc(titleZh)} | ${esc(titleEn)}</title>
<script>${bootstrap}</script>
<style>${css}</style>
</head>
<body data-page="${isIndex ? "index" : chapter.id}">
<a href="#main" class="skip">${b("跳到正文", "Skip to content")}</a>
<header class="topbar"><div class="topbar-inner"><a class="brand" data-carry href="index.html">AI INFRA / ${b("精学馆", "ACADEMY")}</a><div class="top-actions"><button id="language" type="button">English</button><button id="theme" type="button">${b("深色", "Dark")}</button><button id="print" type="button">${b("打印 / PDF", "Print / PDF")}</button></div></div><div class="reading-progress" aria-hidden="true"></div></header>
<main id="main">
<div id="runtime-error" class="error" role="alert" hidden></div>
<div class="hero"><div><div class="eyebrow">${isIndex ? "SOLUTION ARCHITECT / LEARNING LAB" : `CHAPTER 0${index + 1} / SIX DECISIONS`}</div><h1>${b(titleZh, titleEn)}</h1><p class="lead">${isIndex ? b("六大专题，把用户体验、容量、GPU与智算网络串成可复算的架构决策。", "Six chapters connect user experience, capacity, GPUs and fabric into hand-checkable architecture decisions.") : b(chapter.subtitleZh, chapter.subtitleEn)}</p><span class="tag">ZH / EN</span><span class="tag">NO CDN</span><span class="tag">OFFLINE</span><span class="tag">SYNTHETIC ≠ MEASURED</span></div><aside class="hero-aside"><h3>${b("今天记住一句话", "One idea to remember")}</h3><p>${isIndex ? b("以用户SLA定边界，以达标Goodput定容量；显存、算力、网络都必须说清口径。", "Use user SLOs to set boundaries and compliant goodput to size capacity. Specify memory, compute and fabric scopes.") : b(chapter.zh, chapter.en)}</p><small>${b("滑块是教学实验，结果是有条件的估算，不是供应商承诺。", "Sliders are teaching experiments; outputs are conditional estimates, not vendor promises.")}</small>${task}</aside></div>
<nav class="toc" aria-label="Page contents">${sectionIds.map(([id, text]) => `<a href="#${id}">${text}</a>`).join("")}<a href="#sources">${b("依据", "Sources")}</a></nav>
${body}
${sourceSection(sourceIds)}
${navigation}
<footer><div id="policy-notice" class="notice" role="status" hidden></div><p>${b("原创双语学习材料 · 零外部依赖 · 所有可视化坐标来自页面公式或确定性教学样本。未独立复现实机性能；离线打开无外部请求，主动点击参考链接除外。", "Original bilingual learning material · No external dependencies · All visual coordinates derive from page formulas or deterministic teaching samples. No independently reproduced hardware performance. Offline opening makes no external requests; intentional reference-link visits are separate.")}</p><a href="../inference-sizing-network/README.md">${b("双语 FAQ 索引", "Bilingual FAQ index")}</a></footer>
</main>
<script>${math}</script>
<script>${ui}</script>
</body>
</html>
`;
  return { filename: isIndex ? "index.html" : chapter.file, html };
}
const outputs = [build(null, -1), ...chapters.map(build)];
if (process.argv.includes("--check")) {
  let differs = false;
  for (const { filename, html } of outputs) {
    let actual;
    try { actual = readFileSync(join(directory, filename), "utf8"); }
    catch (error) {
      if (error.code !== "ENOENT") throw error;
      console.error("Missing generated page: " + filename);
      differs = true;
      continue;
    }
    if (actual !== html) {
      console.error("Stale generated page: " + filename);
      differs = true;
    }
  }
  if (differs) process.exitCode = 1;
  else console.log("7 standalone pages are reproducible and current.");
} else {
  for (const { filename, html } of outputs) writeFileSync(join(directory, filename), html, "utf8");
  console.log("Built 7 self-contained bilingual HTML pages.");
}
