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
  $("llmBadge").textContent = src.startsWith("LLM") ? "🤖 " + src : "🧠 " + src;

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
  // clear stale labels from any previous report
  $("scatterMeta").textContent = "no scatterable numeric pair in this dataset";
  $("segMeta").textContent = "no segment split in this dataset";
  $("histMeta").textContent = "—";
  Chart.defaults.color = "#8ea0c9"; Chart.defaults.borderColor = "rgba(140,160,200,.12)";
  const ch = r.charts || {};
  if (ch.scatter) {
    $("scatterMeta").textContent = `${ch.scatter.x} vs ${ch.scatter.y} · red = ML-flagged outliers`;
    charts.s = new Chart($("chScatter"), { type: "scatter",
      data: { datasets: [
        { label: "normal", data: ch.scatter.normal.x.map((x, i) => ({ x, y: ch.scatter.normal.y[i] })), backgroundColor: "rgba(56,189,248,.45)", pointRadius: 3 },
        { label: "anomaly", data: ch.scatter.anomalous.x.map((x, i) => ({ x, y: ch.scatter.anomalous.y[i] })), backgroundColor: "#fb7185", pointRadius: 4 }]},
      options: { plugins: { legend: { labels: { boxWidth: 12 } } }, scales: { x: { title: { display: true, text: ch.scatter.x } }, y: { title: { display: true, text: ch.scatter.y } } } } });
  }
  if (ch.segment_bars) {
    $("segMeta").textContent = `${ch.segment_bars.num} by ${ch.segment_bars.cat} · dashed = overall avg`;
    charts.b = new Chart($("chSeg"), { type: "bar",
      data: { labels: ch.segment_bars.labels, datasets: [{ data: ch.segment_bars.values, backgroundColor: ["#7c6cff", "#38bdf8", "#34d399", "#fbbf24", "#fb7185"], borderRadius: 8 }]},
      options: { plugins: { legend: { display: false } } } });
  }
  if (ch.histogram) {
    $("histMeta").textContent = `distribution of ${ch.histogram.column}`;
    const mid = ch.histogram.bins.slice(1).map((b, i) => ((b + ch.histogram.bins[i]) / 2).toFixed(0));
    charts.h = new Chart($("chHist"), { type: "bar",
      data: { labels: mid, datasets: [{ data: ch.histogram.counts, backgroundColor: "rgba(124,108,255,.6)", borderRadius: 4 }]},
      options: { plugins: { legend: { display: false } } } });
  }
}
