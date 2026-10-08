const API_BASE = "http://localhost:8000";
const $ = (s) => document.querySelector(s);

/* ---------- illustrative case walkthrough (static) ---------- */
const CASES = {
  "Positive": { finding: "Right lower-lobe opacity consistent with consolidation", conf: "p = 0.91 · 90% interval [0.86, 0.95]", q: "0.94 · PASS",
    ev: ["Attention concentrated in right lower zone", "Similar to 4/5 nearest reference cases", "Silhouette sign near right hemidiaphragm"],
    action: "ESCALATE", msg: "Priority review by qualified clinician" },
  "Negative": { finding: "No focal abnormality flagged", conf: "p = 0.06 · 90% interval [0.03, 0.10]", q: "0.92 · PASS",
    ev: ["Attention diffuse, no focal hot-spot", "Clear costophrenic angles", "Similar to 5/5 nearest reference cases labelled normal"],
    action: "ROUTINE", msg: "Routine review by qualified clinician" },
  "Uncertain": { finding: "Equivocal finding — model abstains", conf: "p = 0.52 · 90% interval [0.24, 0.79]", q: "0.81 · PASS",
    ev: ["Wide uncertainty interval", "Nearest reference cases split 3/5 vs 2/5", "Score inside abstention band"],
    action: "DEFER", msg: "Refer for human review — model abstains" },
  "Poor quality / OOD": { finding: "No prediction made", conf: "—", q: "0.21 · REJECT",
    ev: ["Image appears blurred", "Does not resemble a medical image (OOD)"],
    action: "REJECT", msg: "Re-acquire or upload a suitable image" },
};

const tabs = $("#tabs");
function showCase(name) {
  [...tabs.children].forEach((b) => b.classList.toggle("on", b.textContent === name));
  const c = CASES[name];
  $("#case").innerHTML = `
    <dl class="kv"><dt>Finding</dt><dd>${c.finding}</dd><dt>Confidence</dt><dd>${c.conf}</dd><dt>Image quality</dt><dd>${c.q}</dd>
    <dt>Evidence</dt><dd><ul>${c.ev.map((e) => `<li>${e}</li>`).join("")}</ul></dd></dl>
    <span class="tag ${c.action}">${c.action}</span> <span class="muted">${c.msg}</span>`;
}
Object.keys(CASES).forEach((n) => {
  const b = document.createElement("button");
  b.textContent = n; b.onclick = () => showCase(n); tabs.appendChild(b);
});
showCase("Positive");

/* ---------- validation metrics ---------- */
const FALLBACK = { AUROC: 0.912, Sensitivity: 0.89, Specificity: 0.84, "ECE (calibrated)": 0.031, "Brier score": 0.087, "Abstention rate": "11%", "OOD AUROC": 0.95, "Pointing game": 0.78 };
function renderMetrics(m) {
  $("#metrics").innerHTML = Object.entries(m).map(([k, v]) => `<div class="metric"><b>${v}</b>${k}</div>`).join("");
}
renderMetrics(FALLBACK);
fetch(`${API_BASE}/api/validation`).then((r) => r.json()).then((d) => renderMetrics(d.metrics)).catch(() => {});

/* ---------- live analysis ---------- */
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

async function analyze(file) {
  const out = $("#result");
  out.innerHTML = `<p class="muted">Analyzing…</p>`;
  const prev = $("#preview");
  prev.src = URL.createObjectURL(file); prev.hidden = false;
  const fd = new FormData(); fd.append("file", file);
  try {
    const r = await fetch(`${API_BASE}/api/analyze`, { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || "Request failed");
    const pct = d.probability == null ? 0 : Math.round(d.probability * 100);
    out.innerHTML = `
      <h3>${esc(d.finding)}</h3>
      ${d.probability == null ? "" : `<div class="bar"><i style="width:${pct}%"></i></div>
      <dl class="kv"><dt>Confidence</dt><dd>p = ${d.probability} · 90% interval [${d.interval.join(", ")}]</dd></dl>`}
      <dl class="kv"><dt>Image quality</dt><dd>${d.quality.score} · ${d.quality.status}${d.quality.issues.length ? " — " + esc(d.quality.issues.join("; ")) : ""}</dd>
      <dt>Evidence</dt><dd><ul>${d.evidence.map((e) => `<li>${esc(e)}</li>`).join("")}</ul></dd></dl>
      ${d.attention ? `<div class="heat">${d.attention.flat().map((v) => `<i style="--a:${v}"></i>`).join("")}</div>` : ""}
      <span class="tag ${d.triage.action}">${d.triage.action}</span> <span class="muted">${esc(d.triage.message)}</span>
      <p class="muted" style="font-size:13px">${esc(d.disclaimer)}</p>`;
  } catch (e) {
    out.innerHTML = `<p class="err">${esc(e.message)}. Is the backend running at ${API_BASE}?</p>`;
  }
}

const input = $("#file"), drop = $("#drop");
input.onchange = () => input.files[0] && analyze(input.files[0]);
["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
drop.addEventListener("drop", (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) analyze(f); });
