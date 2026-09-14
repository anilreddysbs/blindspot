/* BlindSpot frontend — upload -> findings -> investigate -> charts */
let REPORT = null;
let charts = {};
const $ = (id) => document.getElementById(id);

const STEPS = ["Upload dataset", "Profile data", "Anomaly detection", "Pattern discovery", "AI investigation", "Findings"];

function pipeStart() {
  $("pipe").classList.remove("hidden");
  $("pipeSteps").innerHTML = STEPS.map((s, i) => `<span class="flex items-center gap-1"><span class="step-dot" id="st${i}">${i + 1}</span><span class="hidden sm:inline">${s}</span></span>${i < STEPS.length - 1 ? '<span class="text-slate-600">→</span>' : ''}`).join("");
  pipeTick(0, "reading CSV…");
}
function pipeTick(i, msg) {
  $("pipeMsg").textContent = msg;
  $("pipeBar").style.width = (8 + (i / (STEPS.length - 1)) * 92) + "%";
  STEPS.forEach((_, j) => {
    const el = $("st" + j);
    el.className = "step-dot" + (j < i ? " step-done" : j === i ? " step-on" : "");
    if (j < i) el.textContent = "✓";
  });
}
function pipeDone() { pipeTick(STEPS.length - 1, "report ready ✓"); }

async function runAnalyze(promise) {
  pipeStart();
  const msgs = ["profiling data (mean, spread, correlations)…", "running IsolationForest anomaly detection…", "discovering contradiction groups & segments…", "clustering hidden subgroups…", "reasoning over evidence (LLM/template)…", "assembling report…"];
  let i = 0;
  const timer = setInterval(() => { if (i < msgs.length) pipeTick(Math.min(i, 5), msgs[i++]); }, 450);
  try {
    const res = await promise;
    if (!res.ok) {
      let msg = await res.text();
      try { msg = JSON.parse(msg).detail || msg; } catch (e) { /* raw text */ }
      throw new Error(msg);
    }
    REPORT = await res.json();
    clearInterval(timer); pipeDone();
    setTimeout(render, 400);
  } catch (e) {
    clearInterval(timer);
    $("pipeMsg").textContent = "⚠️ " + e.message.slice(0, 300);
  }
}

$("drop").onclick = () => $("file").click();
$("file").onchange = (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  runAnalyze(fetch("/api/analyze", { method: "POST", body: fd }));
};
["dragover", "dragenter"].forEach(ev => $("drop").addEventListener(ev, (e) => { e.preventDefault(); $("drop").classList.add("drag"); }));
["dragleave", "drop"].forEach(ev => $("drop").addEventListener(ev, (e) => { e.preventDefault(); $("drop").classList.remove("drag"); }));
$("drop").addEventListener("drop", (e) => {
  const f = e.dataTransfer.files[0];
  if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  runAnalyze(fetch("/api/analyze", { method: "POST", body: fd }));
});
document.querySelectorAll(".sampleBtn").forEach(b => b.onclick = () =>
  runAnalyze(fetch("/api/analyze-sample", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: b.dataset.sample }) })));

$("mClose").onclick = () => $("modal").classList.add("hidden");
$("modal").addEventListener("click", (e) => { if (e.target.id === "modal") $("modal").classList.add("hidden"); });

const sevIcon = { critical: "🔴", moderate: "🟡", info: "🔵" };

function render() {
  const r = REPORT;
  $("dash").classList.remove("hidden");
  $("dash").scrollIntoView({ behavior: "smooth" });
  $("dsName").textContent = r.dataset + `  ·  ${r.rows} rows × ${r.columns.length} cols`;
  $("kRows").textContent = r.rows.toLocaleString();
  $("kAnom").textContent = r.anomalies.count;
  $("kCrit").textContent = r.counts.critical;
  $("kMod").textContent = r.counts.moderate;
  $("kInfo").textContent = r.counts.info ? `+${r.counts.info} data issues` : "";
  $("vHealth").textContent = r.scores.data_health + "%";
  $("bHealth").style.width = r.scores.data_health + "%";
  $("vConf").textContent = r.scores.insight_confidence + "%";
  $("bConf").style.width = r.scores.insight_confidence + "%";
  $("vRisk").textContent = (r.scores.hidden_risk === "HIGH" ? "🔴 " : r.scores.hidden_risk === "MEDIUM" ? "🟡 " : "🟢 ") + r.scores.hidden_risk;
  $("bRisk").style.width = r.scores.hidden_risk === "HIGH" ? "88%" : r.scores.hidden_risk === "MEDIUM" ? "55%" : "22%";
  const src = (r.findings[0] && r.findings[0].narrative_source) || "";
  $("llmBadge").textContent = src.startsWith("LLM (") ? "🤖 " + src
    : src.includes("used template reasoning") ? "🧠 template reasoning (LLM unavailable)"
    : "🧠 " + src;

  if (r.spotlight) {
    $("spot").classList.remove("hidden");
    $("spotText").textContent = r.spotlight.text + " — “" + r.spotlight.title + "”";
    $("spotBtn").onclick = () => investigate(r.spotlight.finding_id);
  } else $("spot").classList.add("hidden");

  if (r.summary && r.summary.bullets) {
    $("sumCard").classList.remove("hidden");
    $("sumSrc").textContent = "· " + (r.summary.source || "");
    $("sumBullets").innerHTML = r.summary.bullets.map(b => `<li>${esc(b)}</li>`).join("");
  } else $("sumCard").classList.add("hidden");

  $("surgery").textContent = (r.surgery && r.surgery.length) ? ("🔧 " + r.surgery.join(" · ")) : "";
  $("dlBtn").onclick = async () => {
    const res = await fetch("/api/report-md", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ dataset: REPORT.dataset }) });
    const d = await res.json();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([d.markdown], { type: "text/markdown" }));
    a.download = d.filename || "blindspot-report.md";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  $("fCount").textContent = `(${r.findings.length})`;
  $("findings").innerHTML = r.findings.map((f, i) => `
    <div class="card sev-${f.severity} p-5">
      <div class="flex flex-wrap items-center gap-2 text-xs">
        <span>${sevIcon[f.severity] || "•"} ${f.severity.toUpperCase()}</span>
        <span class="chip px-2 py-0.5 text-slate-400">${f.kind}</span>
        <span class="chip px-2 py-0.5 text-slate-400">${f.count} records · ${f.pct}%</span>
        <span class="ml-auto chip px-2 py-0.5" style="color:#34d399">confidence ${f.confidence}%</span>
      </div>
      <div class="font-disp font-bold text-white mt-2">${i + 1}. ${esc(f.title)}</div>
      <p class="text-sm text-slate-300 mt-1">${esc(f.description)}</p>
      ${f.possible_factor ? `<div class="text-sm mt-2 rounded-xl px-3 py-2" style="background:rgba(251,191,36,.08);border:1px solid #5b4a1e">⚠️ Possible contributing factor: <b>${esc(f.possible_factor.variable)}</b> differs by ${f.possible_factor.diff_pct}% — investigate further.</div>` : ""}
      <div class="mt-2 text-xs text-slate-400"><b class="text-slate-300">Evidence</b><pre class="ev mt-1">${esc(f.evidence.join("\n"))}</pre></div>
      <div class="flex items-center gap-2 mt-3">
        <button class="btn-neon px-4 py-1.5 rounded-xl text-sm" onclick="investigate('${f.id}')">🔍 Investigate</button>
        <span class="text-[11px] text-slate-500">${esc(f.causation_note || "")} · <i>${esc(f.narrative_source || "")}</i></span>
      </div>
    </div>`).join("") || `<div class="card p-5 text-sm text-slate-400">No strong patterns — this dataset looks genuinely healthy. ✅</div>`;

  // preview table
  const cols = r.columns;
  $("prev").innerHTML = `<thead><tr>${cols.map(c => `<th class="text-left px-2 py-1 text-slate-400 font-medium border-b border-edge">${esc(c)}</th>`).join("")}</tr></thead><tbody>` +
    r.preview.map(row => `<tr class="border-b border-edge/50">${cols.map(c => `<td class="px-2 py-1 text-slate-300">${esc(String(row[c] ?? ""))}</td>`).join("")}</tr>`).join("") + `</tbody>`;

  drawCharts(r);
}

function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

async function investigate(fid) {
  const f = REPORT.findings.find(x => x.id === fid);
  $("modal").classList.remove("hidden");
  $("mTitle").textContent = f ? f.title : fid;
  $("mSummary").textContent = "Running drill-down across all variables…";
  $("mRows").innerHTML = "";
  try {
    const res = await fetch("/api/investigate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ dataset: REPORT.dataset, finding_id: fid }) });
    const d = await res.json();
    $("mSummary").innerHTML = `<b>Group size: ${d.group_size}.</b> ${esc(d.summary)}`;
    $("mRows").innerHTML = d.rows.map(x => `<tr class="border-b border-edge/50">
      <td class="py-1.5 font-medium text-slate-200">${esc(x.variable)}</td><td>${x.group}</td><td class="text-slate-400">${x.rest}</td>
      <td style="color:${Math.abs(x.diff_pct) >= 15 ? "#fb7185" : Math.abs(x.diff_pct) >= 7 ? "#fbbf24" : "#34d399"}">${x.diff_pct > 0 ? "+" : ""}${x.diff_pct}%</td>
      <td class="text-slate-400">${esc(x.verdict)}</td></tr>`).join("");
  } catch (e) { $("mSummary").textContent = "⚠️ " + e.message; }
}

function drawCharts(r) {
  Object.values(charts).forEach(c => c && c.destroy()); charts = {};
  Chart.defaults.color = "#8ea0c9"; Chart.defaults.borderColor = "rgba(140,160,200,.12)";
  const box = $("plots");
  box.innerHTML = "";
  const plots = (r.charts && r.charts.plots) || [];
  if (!plots.length) {
    box.innerHTML = `<div class="card p-5 text-sm text-slate-400">No chart-worthy pattern — nothing here needed a picture. The findings above say it all.</div>`;
    return;
  }
  const PALETTE = ["#7c6cff", "#38bdf8", "#34d399", "#fbbf24", "#fb7185", "#f472b6", "#a3e635"];
  plots.forEach((p, i) => {
    const card = document.createElement("div");
    card.className = "card p-5";
    card.innerHTML = `<div class="flex items-center gap-2"><div class="text-sm font-semibold text-white">${esc(p.title)}</div>` +
      (p.finding_id ? `<button class="ml-auto text-[11px] ghost rounded-lg px-2 py-0.5 shrink-0" data-f="${p.finding_id}">view ${p.finding_id} →</button>` : "") +
      `</div><div class="text-[11px] text-slate-500 mt-0.5">${esc(p.subtitle || "")}</div>` +
      `<div class="mt-2"><canvas id="plot_${p.id}"></canvas></div>`;
    box.appendChild(card);
    const btn = card.querySelector("[data-f]");
    if (btn) btn.onclick = () => investigate(btn.dataset.f);
    const ctx = card.querySelector("canvas");
    const common = { plugins: { legend: { labels: { boxWidth: 12 } } } };
    let cfg = null;
    if (p.kind === "scatter") {
      cfg = { type: "scatter", data: { datasets: p.datasets.map(d => ({ label: d.label, data: d.points.map(pt => ({ x: pt[0], y: pt[1] })), backgroundColor: d.color, pointRadius: 3.5 })) },
        options: { ...common, scales: { x: { title: { display: true, text: p.x_label } }, y: { title: { display: true, text: p.y_label } } } } };
    } else if (p.kind === "bar") {
      const showOverall = p.overall !== null && p.overall !== undefined;
      const colors = (p.labels || []).map(l => (p.highlight && String(l) === String(p.highlight)) ? "#fb7185" : "#7c6cff");
      const datasets = [{ data: p.values, backgroundColor: colors, borderRadius: 8 }];
      if (showOverall) datasets.push({ type: "line", label: "overall avg", data: p.labels.map(() => p.overall), borderColor: "#e2e8f0", borderDash: [6, 4], pointRadius: 0, borderWidth: 1.5 });
      cfg = { type: "bar", data: { labels: p.labels, datasets }, options: { ...common, plugins: { legend: { display: showOverall } } } };
    } else if (p.kind === "grouped") {
      cfg = { type: "bar", data: { labels: p.labels, datasets: p.datasets.map((d, j) => ({ label: d.label, data: d.values, backgroundColor: PALETTE[(j + i) % PALETTE.length], borderRadius: 6 })) }, options: common };
    } else if (p.kind === "hist") {
      cfg = { type: "bar", data: { labels: p.labels, datasets: [{ data: p.values, backgroundColor: "rgba(124,108,255,.6)", borderRadius: 4 }] }, options: { ...common, plugins: { legend: { display: false } } } };
    } else if (p.kind === "line") {
      cfg = { type: "line", data: { labels: p.labels, datasets: [{ data: p.values, borderColor: "#38bdf8", backgroundColor: "rgba(56,189,248,.15)", fill: true, tension: 0.3, pointRadius: 3 }] }, options: { ...common, plugins: { legend: { display: false } } } };
    }
    if (cfg) charts[p.id] = new Chart(ctx, cfg);
  });
}
