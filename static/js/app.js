function $(id) { return document.getElementById(id); }

function setVal(id, val) {
  const el = $(id);
  if (!el) return;
  el.value = val;
}

function setText(id, txt) {
  const el = $(id);
  if (!el) return;
  el.textContent = txt;
}

function clearForm() {
  document.querySelectorAll("#churnForm input, #churnForm select, #churnForm textarea").forEach(el => {
    if (el.type === "checkbox") el.checked = false;
    else el.value = "";
  });

  // keep SHAP enabled by default (optional)
  const shap = $("includeShap");
  if (shap) shap.checked = true;
}

function formToJSON() {
  const data = {};

  // include shap switches
  const includeShap = $("includeShap");
  data.include_shap = includeShap ? includeShap.checked : true;
  data.shap_top_n = 10;

  document.querySelectorAll("#churnForm input, #churnForm select").forEach(el => {
    const name = el.name;
    if (!name) return;

    // skip includeShap checkbox (handled above)
    if (el.id === "includeShap") return;

    if (el.type === "checkbox") {
      // if you add other checkboxes later
      data[name] = el.checked ? 1 : 0;
      return;
    }

    const raw = (el.value ?? "").toString().trim();
    if (raw === "") return; // omit blanks => backend uses baseline mean

    data[name] = Number(raw);
  });

  return data;
}

function showError(msg) {
  const box = $("errBox");
  if (!box) return;
  box.textContent = msg;
  box.classList.remove("d-none");
}

function clearError() {
  const box = $("errBox");
  if (!box) return;
  box.classList.add("d-none");
  box.textContent = "";
}

function badgeStyle(risk) {
  if (risk === "RED") return { badge: "text-bg-danger", bar: "bg-danger" };
  if (risk === "YELLOW") return { badge: "text-bg-warning", bar: "bg-warning" };
  return { badge: "text-bg-success", bar: "bg-success" };
}

/* ---------- Gauge ---------- */
let _gaugeAngle = -90;

function setGauge(probPct) {
  const needle = $("gNeedle");
  const valEl = $("gValue");
  if (valEl) valEl.textContent = `${Number(probPct).toFixed(2)}%`;
  if (!needle) return;

  const target = -90 + (Math.max(0, Math.min(100, probPct)) / 100) * 180;

  // animate a bit
  const start = _gaugeAngle;
  const end = target;
  const dur = 350;
  const t0 = performance.now();

  function step(now) {
    const t = Math.min(1, (now - t0) / dur);
    const eased = 1 - Math.pow(1 - t, 3); // easeOutCubic
    const ang = start + (end - start) * eased;
    needle.style.transform = `translateX(-50%) rotate(${ang}deg)`;
    if (t < 1) requestAnimationFrame(step);
    else _gaugeAngle = end;
  }
  requestAnimationFrame(step);
}

/* ---------- Engineered preview ---------- */
function formatMaybeNumber(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number" && Number.isFinite(v)) {
    // show fewer decimals for readability
    const abs = Math.abs(v);
    if (abs >= 1000) return v.toFixed(0);
    if (abs >= 100) return v.toFixed(1);
    return v.toFixed(4);
  }
  return String(v);
}

function renderEngineered(engineered) {
  const body = $("engTableBody");
  if (!body) return;

  body.innerHTML = "";
  if (!engineered || typeof engineered !== "object") {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="2" class="text-soft">No engineered preview returned.</td>`;
    body.appendChild(tr);
    return;
  }

  Object.entries(engineered).forEach(([k, v]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><code class="text-light">${k}</code></td>
      <td class="text-end">${formatMaybeNumber(v)}</td>
    `;
    body.appendChild(tr);
  });
}

/* ---------- Recommendation ---------- */
function renderRecommendation(rec) {
  if (!rec) return;

  setText("recUrgency", `Urgency: ${rec.urgency} • Risk: ${rec.risk_level}`);
  setText("recMessage", rec.message || "");

  const reasons = $("recReasons");
  if (reasons) {
    reasons.innerHTML = "";
    (rec.reasons || []).forEach(r => {
      const li = document.createElement("li");
      li.textContent = r;
      reasons.appendChild(li);
    });
    if ((rec.reasons || []).length === 0) {
      const li = document.createElement("li");
      li.className = "text-soft";
      li.textContent = "No major risk signals detected from the provided inputs.";
      reasons.appendChild(li);
    }
  }

  const offers = $("recOffers");
  if (offers) {
    offers.innerHTML = "";
    (rec.recommended_offers || []).forEach(o => {
      const li = document.createElement("li");
      li.textContent = o;
      offers.appendChild(li);
    });
  }
}

/* ---------- SHAP ---------- */
function renderSHAP(shap) {
  const tableBody = $("shapTableBody");
  const barsWrap = $("shapBars");
  const baseEl = $("shapBase");
  const statusEl = $("shapStatus");

  if (tableBody) tableBody.innerHTML = "";
  if (barsWrap) barsWrap.innerHTML = "";
  if (baseEl) baseEl.textContent = "";

  if (!shap || shap.available !== true) {
    if (statusEl) statusEl.textContent = shap?.error ? `Unavailable (${shap.error})` : "Unavailable";
    return;
  }

  if (statusEl) statusEl.textContent = "Available";

  const feats = shap.top_features || [];
  if (baseEl && typeof shap.base_value === "number") {
    baseEl.textContent = `Base value: ${shap.base_value.toFixed(6)} (model log-odds baseline)`;
  }

  const maxAbs = Math.max(...feats.map(f => Math.abs(f.shap_value)), 1e-9);

  // Bars
  if (barsWrap) {
    feats.forEach(f => {
      const row = document.createElement("div");
      row.className = "shap-bar-row";

      const label = document.createElement("div");
      label.className = "shap-bar-label";
      label.textContent = f.feature;

      const track = document.createElement("div");
      track.className = "shap-bar-track";

      const fill = document.createElement("div");
      const isUp = f.direction === "increases_churn";
      fill.className = "shap-bar-fill " + (isUp ? "up" : "down");
      fill.style.width = `${(Math.abs(f.shap_value) / maxAbs) * 100}%`;

      const val = document.createElement("div");
      val.className = "shap-bar-val";
      val.textContent = (f.shap_value >= 0 ? "+" : "") + Number(f.shap_value).toFixed(4);

      track.appendChild(fill);
      row.appendChild(label);
      row.appendChild(track);
      row.appendChild(val);
      barsWrap.appendChild(row);
    });
  }

  // Table
  if (tableBody) {
    feats.forEach(f => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${f.feature}</td>
        <td class="text-end">${Number(f.value).toFixed(2)}</td>
        <td class="text-end">${(f.shap_value >= 0 ? "+" : "") + Number(f.shap_value).toFixed(4)}</td>
        <td>${f.direction === "increases_churn" ? "↑ increases churn" : "↓ decreases churn"}</td>
      `;
      tableBody.appendChild(tr);
    });
  }
}

/* ---------- Main render ---------- */
function renderResult(data) {
  const empty = $("resultEmpty");
  const box = $("resultBox");
  if (empty) empty.classList.add("d-none");
  if (box) box.classList.remove("d-none");

  const prob = Number(data.churn_probability_pct);
  const risk = data.risk_level;

  // big text (if exists)
  setText("probText", `${prob.toFixed(2)}%`);

  // gauge
  setGauge(prob);

  // risk badge
  const styles = badgeStyle(risk);
  const badge = $("riskBadge");
  if (badge) {
    badge.textContent = risk;
    badge.className = `badge fs-6 ${styles.badge}`;
  }

  // optional progress bar (if your HTML still has it)
  const bar = $("probBar");
  if (bar) {
    bar.style.width = `${Math.max(0, Math.min(100, prob))}%`;
    bar.className = `progress-bar ${styles.bar}`;
  }

  renderRecommendation(data.recommendation);
  renderSHAP(data.shap);
  renderEngineered(data.engineered);
}

/* ---------- Presets ---------- */
const PRESET_GREEN = {
  aon: 3000,
  arpu_6: 450, arpu_7: 460, arpu_8: 480,
  total_og_mou_6: 1200, total_og_mou_7: 1300, total_og_mou_8: 1600,
  total_ic_mou_6: 1400, total_ic_mou_7: 1500, total_ic_mou_8: 2200,
  total_rech_amt_6: 600, total_rech_amt_7: 650, total_rech_amt_8: 700,
  total_rech_num_6: 8, total_rech_num_7: 8, total_rech_num_8: 9,
  last_day_rch_amt_6: 199, last_day_rch_amt_7: 199, last_day_rch_amt_8: 249,
  roam_og_mou_6: 0, roam_og_mou_7: 0, roam_og_mou_8: 0,
  roam_ic_mou_6: 0, roam_ic_mou_7: 0, roam_ic_mou_8: 0,
  fb_user_8: 1
};

// Updated stable YELLOW (~43-44% under current backend)
const PRESET_YELLOW = { ...PRESET_GREEN, total_og_mou_8: 1000, total_ic_mou_8: 1500 };

const PRESET_RED = {
  aon: 30,
  arpu_6: 800, arpu_7: 600, arpu_8: 20,
  total_og_mou_6: 1200, total_og_mou_7: 700, total_og_mou_8: 5,
  total_ic_mou_6: 1500, total_ic_mou_7: 900, total_ic_mou_8: 0,
  total_rech_amt_6: 900, total_rech_amt_7: 600, total_rech_amt_8: 10,
  total_rech_num_6: 10, total_rech_num_7: 6, total_rech_num_8: 1,
  last_day_rch_amt_6: 200, last_day_rch_amt_7: 100, last_day_rch_amt_8: 10,
  roam_og_mou_6: 0, roam_og_mou_7: 0, roam_og_mou_8: 300,
  roam_ic_mou_6: 0, roam_ic_mou_7: 0, roam_ic_mou_8: 100,
  fb_user_8: 0
};

function loadPreset(preset) {
  clearForm();
  Object.entries(preset).forEach(([k, v]) => setVal(k, v));
}

/* ---------- Bootstrp tooltips init ---------- */
function initTooltips() {
  const Tooltip = window.bootstrap?.Tooltip;
  if (!Tooltip) return;
  document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
    try { new Tooltip(el); } catch (_) {}
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initTooltips();

  $("btnLoadGreen")?.addEventListener("click", () => loadPreset(PRESET_GREEN));
  $("btnLoadYellow")?.addEventListener("click", () => loadPreset(PRESET_YELLOW));
  $("btnLoadRed")?.addEventListener("click", () => loadPreset(PRESET_RED));
  $("btnClear")?.addEventListener("click", () => clearForm());

  $("churnForm")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    clearError();

    try {
      const payload = formToJSON();

      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed");

      renderResult(data);
    } catch (err) {
      showError(err.message || String(err));
    }
  });
});