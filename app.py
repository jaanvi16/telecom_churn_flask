from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template


# -------------------------
# Paths + Load Artifacts
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "final_xgboost_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURES_PATH = MODELS_DIR / "feature_names.json"
META_PATH = MODELS_DIR / "meta.json"
SHAP_PATH = MODELS_DIR / "shap_explainer.pkl"

MODEL = joblib.load(MODEL_PATH)
SCALER = joblib.load(SCALER_PATH)

MODEL_FEATURES = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
MODEL_FEATURES = list(MODEL_FEATURES)
FEATURE_SET = set(MODEL_FEATURES)

META = json.loads(META_PATH.read_text(encoding="utf-8"))

if len(MODEL_FEATURES) != len(SCALER.mean_):
    raise ValueError(
        f"Mismatch: feature_names.json={len(MODEL_FEATURES)} "
        f"scaler.mean_={len(SCALER.mean_)}"
    )

# CRUCIAL FIX: baseline defaults = training means for each feature
BASELINE_DEFAULTS: Dict[str, float] = dict(zip(MODEL_FEATURES, SCALER.mean_))


# -------------------------
# Try to load SHAP Explainer
# -------------------------
SHAP_EXPLAINER = None
SHAP_LOAD_ERROR = None
try:
    import shap  # noqa: F401

    if SHAP_PATH.exists():
        SHAP_EXPLAINER = joblib.load(SHAP_PATH)
    else:
        SHAP_LOAD_ERROR = "shap_explainer.pkl not found"
except Exception as e:
    SHAP_LOAD_ERROR = str(e)
    SHAP_EXPLAINER = None


# -------------------------
# Core 19 Inputs (Required)
# -------------------------
CORE_19 = [
    "aon",
    "arpu_6", "arpu_7", "arpu_8",
    "total_og_mou_6", "total_og_mou_7", "total_og_mou_8",
    "total_ic_mou_6", "total_ic_mou_7", "total_ic_mou_8",
    "total_rech_amt_6", "total_rech_amt_7", "total_rech_amt_8",
    "total_rech_num_6", "total_rech_num_7", "total_rech_num_8",
    "last_day_rch_amt_6", "last_day_rch_amt_7", "last_day_rch_amt_8",
]


# -------------------------
# Engineered preview keys (for UI)
# -------------------------
ENGINEERED_PREVIEW_KEYS = [
    "total_mou_6", "total_mou_7", "total_mou_8",
    "arpu_trend", "arpu_trend_7_8", "arpu_decline_rate", "avg_arpu", "arpu_std",
    "og_mou_trend", "ic_mou_trend", "avg_og_mou", "avg_ic_mou", "og_mou_std", "aug_jun_og_ratio",
    "rech_amt_trend", "avg_rech_amt", "rech_num_trend",
    "tenure_months", "tenure_segment",
    "is_high_value", "is_roaming_user",
    "avg_data_rech",
]


# -------------------------
# Helpers
# -------------------------
def to_float(x: Any) -> float:
    if x is None:
        raise ValueError("Missing value")
    if isinstance(x, str):
        x = x.strip()
        if x == "":
            raise ValueError("Empty value")
    return float(x)


def to_bool(x: Any, default: bool = True) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        v = x.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        if v in {"0", "false", "no", "off"}:
            return False
    return default


def set_if_exists(row: Dict[str, float], name: str, value: float) -> None:
    if name in FEATURE_SET and np.isfinite(value):
        row[name] = float(value)


def compute_tenure_months(aon: float) -> float:
    # replicate: (aon/30).round(0)
    return float(np.round(aon / 30.0, 0))


def compute_tenure_segment(tenure_months: float) -> float:
    """
    replicate:
    pd.cut(tenure_months, bins=[0,6,12,24,36,10000], labels=[5,4,3,2,1]).astype(float).fillna(3)
    with pandas default right=True => (0,6], (6,12], ...
    """
    fillna = float(META.get("tenure_fillna", 3))
    bins = META.get("tenure_bins", [0, 6, 12, 24, 36, 10000])

    if (not np.isfinite(tenure_months)) or (tenure_months <= bins[0]) or (tenure_months > bins[-1]):
        return fillna

    if tenure_months <= 6:
        return 5.0
    if tenure_months <= 12:
        return 4.0
    if tenure_months <= 24:
        return 3.0
    if tenure_months <= 36:
        return 2.0
    return 1.0


def risk_bucket(p: float) -> str:
    if p >= 0.70:
        return "RED"
    if p >= 0.40:
        return "YELLOW"
    return "GREEN"


def build_engineered_preview(row: Dict[str, float]) -> Dict[str, float]:
    preview: Dict[str, float] = {}
    for k in ENGINEERED_PREVIEW_KEYS:
        if k in row and row[k] is not None:
            try:
                preview[k] = float(row[k])
            except Exception:
                pass
    return preview


# -------------------------
# Feature Engineering (match Colab)
# -------------------------
def build_row(payload: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Returns:
      X_raw_df (1 x 153) in correct feature order
      row_dict (raw values after feature engineering) for recommendation logic + preview
    """
    row: Dict[str, float] = dict(BASELINE_DEFAULTS)

    # core 19 required
    for f in CORE_19:
        row[f] = to_float(payload.get(f))

    # advanced optional: overwrite any provided features that exist in model features
    for k, v in payload.items():
        if k in FEATURE_SET and k not in CORE_19:
            if isinstance(v, bool):
                row[k] = 1.0 if v else 0.0
            else:
                try:
                    row[k] = float(v)
                except Exception:
                    pass

    # engineered features - replicate your Colab logic
    arpu6, arpu7, arpu8 = row["arpu_6"], row["arpu_7"], row["arpu_8"]
    og6, og7, og8 = row["total_og_mou_6"], row["total_og_mou_7"], row["total_og_mou_8"]
    ic6, ic7, ic8 = row["total_ic_mou_6"], row["total_ic_mou_7"], row["total_ic_mou_8"]
    ramt6, ramt7, ramt8 = row["total_rech_amt_6"], row["total_rech_amt_7"], row["total_rech_amt_8"]
    rnum6, rnum7, rnum8 = row["total_rech_num_6"], row["total_rech_num_7"], row["total_rech_num_8"]
    aon = row["aon"]

    # Revenue features
    set_if_exists(row, "arpu_trend", arpu8 - arpu6)
    set_if_exists(row, "arpu_trend_7_8", arpu8 - arpu7)
    set_if_exists(row, "arpu_decline_rate", (arpu6 - arpu8) / (arpu6 + 1.0))

    # avg_arpu from Colab: mean of arpu_6/7/8 + engineered arpu_* features created above
    avg_arpu_components = [
        row.get("arpu_6", 0.0),
        row.get("arpu_7", 0.0),
        row.get("arpu_8", 0.0),
        row.get("arpu_trend", 0.0),
        row.get("arpu_trend_7_8", 0.0),
        row.get("arpu_decline_rate", 0.0),
    ]
    set_if_exists(row, "avg_arpu", float(np.mean(avg_arpu_components)))

    # Call usage features
    set_if_exists(row, "og_mou_trend", og8 - og6)
    set_if_exists(row, "ic_mou_trend", ic8 - ic6)
    set_if_exists(row, "avg_og_mou", float(np.mean([og6, og7, og8])))
    set_if_exists(row, "avg_ic_mou", float(np.mean([ic6, ic7, ic8])))

    set_if_exists(row, "total_mou_6", og6 + ic6)
    set_if_exists(row, "total_mou_7", og7 + ic7)
    set_if_exists(row, "total_mou_8", og8 + ic8)

    set_if_exists(row, "aug_jun_og_ratio", (og8 + 1.0) / (og6 + 1.0))

    # Recharge features
    set_if_exists(row, "rech_amt_trend", ramt8 - ramt6)
    set_if_exists(row, "avg_rech_amt", float(np.mean([ramt6, ramt7, ramt8])))
    set_if_exists(row, "rech_num_trend", rnum8 - rnum6)

    # Tenure
    tenure_months = compute_tenure_months(aon)
    set_if_exists(row, "tenure_months", tenure_months)
    set_if_exists(row, "tenure_segment", compute_tenure_segment(tenure_months))

    # is_high_value (use fixed threshold saved from training run)
    if "is_high_value" in FEATURE_SET:
        thr = float(META.get("high_value_q75_threshold", 182))
        row["is_high_value"] = 1.0 if row.get("avg_arpu", 0.0) >= thr else 0.0

    # is_roaming_user:
    # Only recompute if payload provides roaming fields; else keep baseline mean.
    if "is_roaming_user" in FEATURE_SET:
        provided_roam_keys = [k for k in payload.keys() if "roam_" in k.lower()]
        if provided_roam_keys:
            roam_sum = 0.0
            for k in provided_roam_keys:
                try:
                    roam_sum += float(payload[k])
                except Exception:
                    pass
            row["is_roaming_user"] = 1.0 if roam_sum > 0 else 0.0

    # avg_data_rech: only compute if user provided at least 2 av_rech_amt_data_* fields
    if "avg_data_rech" in FEATURE_SET:
        provided_data_keys = [k for k in payload.keys() if "av_rech_amt_data" in k]
        vals = []
        for k in provided_data_keys:
            try:
                vals.append(float(payload[k]))
            except Exception:
                pass
        if len(vals) >= 2:
            row["avg_data_rech"] = float(np.mean(vals))

    # std features (match pandas .std default ddof=1)
    if "arpu_std" in FEATURE_SET:
        row["arpu_std"] = float(np.std([arpu6, arpu7, arpu8], ddof=1))
    if "og_mou_std" in FEATURE_SET:
        row["og_mou_std"] = float(np.std([og6, og7, og8], ddof=1))

    # Build DF in correct order + sanitize
    X_raw = pd.DataFrame([row], columns=MODEL_FEATURES)
    X_raw = X_raw.replace([np.inf, -np.inf], np.nan)

    if X_raw.isna().any(axis=None):
        for col in X_raw.columns[X_raw.isna().any()].tolist():
            X_raw.loc[0, col] = BASELINE_DEFAULTS[col]

    row_sanitized = X_raw.iloc[0].to_dict()
    return X_raw, row_sanitized


# -------------------------
# Recommendation Engine
# -------------------------
def build_recommendation(p: float, row: Dict[str, float]) -> Dict[str, Any]:
    risk = risk_bucket(p)

    arpu_trend = float(row.get("arpu_trend", 0.0))
    arpu_decline_rate = float(row.get("arpu_decline_rate", 0.0))
    og_trend = float(row.get("og_mou_trend", 0.0))
    ic_trend = float(row.get("ic_mou_trend", 0.0))
    rech_trend = float(row.get("rech_amt_trend", 0.0))
    tenure_seg = float(row.get("tenure_segment", 3.0))
    is_roaming = float(row.get("is_roaming_user", 0.0))

    reasons = []
    offers = []

    if (arpu_trend < 0) or (arpu_decline_rate > 0.20):
        reasons.append("Revenue is declining recently (ARPU drop / decline rate high).")
        offers.append("Offer 10–30% discount or budget plan upgrade to reduce price sensitivity.")

    if (og_trend < 0) or (ic_trend < 0):
        reasons.append("Customer engagement is dropping (call usage trending down).")
        offers.append("Offer unlimited calling bundle / bonus talktime to increase engagement.")

    if rech_trend < 0:
        reasons.append("Recharge behavior is weakening (recharge amount trending down).")
        offers.append("Offer recharge cashback / extra validity / bonus data on next recharge.")

    if is_roaming >= 0.5:
        reasons.append("Roaming behavior observed (customer may value roaming packs).")
        offers.append("Offer discounted roaming add-on / international pack.")

    if tenure_seg >= 4:
        reasons.append("Customer is relatively new (higher churn risk segment).")
        offers.append("Onboarding + proactive support call, plus welcome-back offer.")

    offers = list(dict.fromkeys(offers))
    if not offers:
        offers = ["Maintain engagement with periodic value offers and monitor usage trends."]

    if risk == "RED":
        urgency = "Call within 24 hours"
        message = "High churn risk. Prioritize immediate retention outreach."
    elif risk == "YELLOW":
        urgency = "Engage within 7 days"
        message = "Medium churn risk. Send targeted offer and monitor closely."
    else:
        urgency = "Monitor monthly"
        message = "Low churn risk. Maintain engagement and upsell where appropriate."

    return {
        "risk_level": risk,
        "urgency": urgency,
        "message": message,
        "reasons": reasons[:5],
        "recommended_offers": offers[:5],
    }


# -------------------------
# SHAP Explanation
# -------------------------
def shap_explain_top(
    X_scaled: np.ndarray,
    X_raw_df: pd.DataFrame,
    top_n: int = 10
) -> Dict[str, Any]:
    if SHAP_EXPLAINER is None:
        return {"available": False, "error": SHAP_LOAD_ERROR or "SHAP explainer not loaded"}

    try:
        exp = SHAP_EXPLAINER(X_scaled)
        values = exp.values
        base_values = exp.base_values

        # If values are (1, n_features, 2) pick positive class
        if values.ndim == 3:
            values = values[:, :, 1]

        shap_row = values[0]
        base = float(base_values[0]) if np.ndim(base_values) else float(base_values)

    except Exception:
        values = SHAP_EXPLAINER.shap_values(X_scaled)
        if isinstance(values, list):
            values = values[1]  # positive class
        shap_row = values[0]
        base = None

    feature_names = list(X_raw_df.columns)
    raw_vals = X_raw_df.iloc[0].to_dict()

    idx = np.argsort(np.abs(shap_row))[::-1][:top_n]
    top = []
    for i in idx:
        f = feature_names[i]
        sv = float(shap_row[i])
        top.append({
            "feature": f,
            "value": float(raw_vals.get(f, 0.0)),
            "shap_value": sv,
            "direction": "increases_churn" if sv > 0 else "decreases_churn"
        })

    return {
        "available": True,
        "base_value": base,
        "top_features": top
    }


# -------------------------
# Predict wrapper
# -------------------------
def predict_all(payload: Dict[str, Any]) -> Dict[str, Any]:
    include_shap = to_bool(payload.get("include_shap", True), default=True)
    top_n = int(payload.get("shap_top_n", 10) or 10)

    X_raw, row_dict = build_row(payload)
    X_scaled = SCALER.transform(X_raw)

    p = float(MODEL.predict_proba(X_scaled)[0, 1])
    risk = risk_bucket(p)

    rec = build_recommendation(p, row_dict)
    engineered = build_engineered_preview(row_dict)

    result = {
        "churn_probability_pct": round(p * 100, 2),
        "risk_level": risk,
        "recommendation": rec,
        "engineered": engineered,
    }

    if include_shap:
        result["shap"] = shap_explain_top(X_scaled, X_raw, top_n=top_n)
    else:
        result["shap"] = {"available": False, "error": "disabled_by_request"}

    return result


# -------------------------
# Flask App
# -------------------------
app = Flask(__name__)


@app.get("/")
def home():
    try:
        return render_template("index.html")
    except Exception:
        return "Telecom Churn Flask backend is running. POST JSON to /predict"


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        if request.form:
            data = request.form.to_dict()
        else:
            return jsonify({"error": "Send JSON body (Content-Type: application/json)"}), 400

    try:
        missing = [f for f in CORE_19 if f not in data or str(data[f]).strip() == ""]
        if missing:
            return jsonify({"error": f"Missing required fields: {missing}"}), 400

        result = predict_all(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)