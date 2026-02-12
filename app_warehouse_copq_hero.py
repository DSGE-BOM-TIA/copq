# Open in Notepad++ (UTF-8). Save As: app_warehouse_copq_hero.py
# Run: python -m streamlit run app_warehouse_copq_hero.py


from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =========================
# Helpers
# =========================
def money(x: float) -> str:
    try:
        return "${:,.0f}".format(float(x))
    except Exception:
        return str(x)

def pct(x: float) -> str:
    try:
        return "{:.2%}".format(float(x))
    except Exception:
        return str(x)

def ppm_to_rate(ppm: float) -> float:
    return float(ppm) / 1_000_000.0

def load_image_as_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# =========================
# Page + Hero theme
# =========================
st.set_page_config(page_title="Fulfillment COPQ Hero Calculator", page_icon="📦", layout="wide")

hero_b64 = ""
try:
    hero_b64 = load_image_as_base64("hero_infographic.png")
except Exception:
    pass

st.markdown(f"""
<style>
  .stApp {{
    background:
      linear-gradient(rgba(9,14,26,0.92), rgba(9,14,26,0.94)),
      url('data:image/png;base64,{hero_b64}') no-repeat top center fixed;
    background-size: cover;
  }}
  .hero-card {{
    padding: 18px 18px 14px 18px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
  }}
  .pill {{
    display:inline-block; padding: 6px 10px; border-radius: 999px;
    background: rgba(66,153,225,0.18);
    border: 1px solid rgba(66,153,225,0.35);
    font-weight: 800; font-size: 12px;
  }}
  .muted {{ opacity: 0.82; }}
  .tiny {{ font-size: 12px; opacity: 0.78; }}
</style>
""", unsafe_allow_html=True)

# =========================
# Model
# =========================
@dataclass
class Tri:
    low: float
    mode: float
    high: float

def tri_vals(tri: Tri) -> Tuple[float, float, float]:
    return tri.low, tri.mode, tri.high

@dataclass
class DefectCost:
    name: str
    ppm: Tri
    cost: Tri

def compute_copq(orders: float, opps_per_order: float, defects: Dict[str, DefectCost]) -> pd.DataFrame:
    """
    Scenario table for BEST / LIKELY / WORST.
    Defect count = orders * opportunities/order * (PPM / 1,000,000)
    COPQ = defects * cost_per_defect
    """
    rows = []
    for scenario, pick in [("BEST", 0), ("LIKELY", 1), ("WORST", 2)]:
        total = 0.0
        for d in defects.values():
            ppm = tri_vals(d.ppm)[pick]
            c = tri_vals(d.cost)[pick]
            defect_count = orders * opps_per_order * ppm_to_rate(ppm)
            cost_total = defect_count * c
            total += cost_total
            rows.append({
                "Scenario": scenario,
                "Defect Type": d.name,
                "PPM": ppm,
                "Defects (count)": defect_count,
                "Cost/Defect": c,
                "COPQ Cost": cost_total
            })
        rows.append({
            "Scenario": scenario,
            "Defect Type": "TOTAL",
            "PPM": np.nan,
            "Defects (count)": np.nan,
            "Cost/Defect": np.nan,
            "COPQ Cost": total
        })
    return pd.DataFrame(rows)

def kpi_rates(defects: Dict[str, DefectCost]) -> Dict[str, float]:
    """Converts LIKELY PPM into accuracy-like KPIs + perfect order proxy."""
    mispick_ppm = tri_vals(defects["mispick"].ppm)[1]
    damage_ppm = tri_vals(defects["damage"].ppm)[1]
    late_ppm = tri_vals(defects["late"].ppm)[1]
    label_ppm = tri_vals(defects["label"].ppm)[1]

    pick_accuracy = 1.0 - ppm_to_rate(mispick_ppm)
    damage_free = 1.0 - ppm_to_rate(damage_ppm)
    on_time = 1.0 - ppm_to_rate(late_ppm)
    label_accuracy = 1.0 - ppm_to_rate(label_ppm)

    perfect_order = pick_accuracy * damage_free * on_time * label_accuracy
    return {
        "Pick Accuracy": pick_accuracy,
        "Damage-Free Rate": damage_free,
        "On-Time Rate": on_time,
        "Label Accuracy": label_accuracy,
        "Perfect Order (proxy)": perfect_order,
    }

# =========================
# Sidebar: Deterministic inputs
# =========================
st.sidebar.markdown("### 📦 Fulfillment Inputs (Deterministic)")
orders = st.sidebar.number_input("Orders shipped per month", min_value=0.0, value=250000.0, step=1000.0)
opps = st.sidebar.number_input("Opportunities per order", min_value=1.0, value=4.0, step=1.0,
                              help="Example: pick, pack, label, load = 4 opportunities per order.")

st.sidebar.markdown("### 🧩 Defect Rates (PPM) — Best / Likely / Worst")
st.sidebar.caption("PPM = defects per million opportunities. If you only know %, use: PPM = % × 10,000")

def ppm_slider(name, default):
    return st.sidebar.slider(f"{name} PPM (best, likely, worst)", 0.0, 20000.0, default)

def cost_slider(name, default):
    return st.sidebar.slider(f"{name} cost/defect $ (best, likely, worst)", 0.0, 5000.0, default)

damage_ppm = ppm_slider("Damage", (300.0, 900.0, 2500.0))
damage_cost = cost_slider("Damage", (20.0, 80.0, 250.0))

mispick_ppm = ppm_slider("Mispick / Wrong item", (200.0, 1200.0, 4000.0))
mispick_cost = cost_slider("Mispick / Wrong item", (25.0, 120.0, 400.0))

late_ppm = ppm_slider("Late / SLA miss", (80.0, 400.0, 1200.0))
late_cost = cost_slider("Late / SLA miss", (10.0, 60.0, 200.0))

label_ppm = ppm_slider("Label / carton ID error", (50.0, 250.0, 900.0))
label_cost = cost_slider("Label / carton ID error", (8.0, 45.0, 150.0))

st.sidebar.markdown("### ⚖️ Optional scope")
include_customer_churn = st.sidebar.checkbox("Include customer churn proxy", value=False)
if include_customer_churn:
    churn_ppm = ppm_slider("Customer loss event (proxy)", (1.0, 5.0, 20.0))
    churn_cost = cost_slider("Customer loss $ impact (proxy)", (200.0, 900.0, 3000.0))
else:
    churn_ppm = (0.0, 0.0, 0.0)
    churn_cost = (0.0, 0.0, 0.0)

defects = {
    "damage": DefectCost("Damage", Tri(*map(float, damage_ppm)), Tri(*map(float, damage_cost))),
    "mispick": DefectCost("Mispick / Wrong item", Tri(*map(float, mispick_ppm)), Tri(*map(float, mispick_cost))),
    "late": DefectCost("Late / SLA miss", Tri(*map(float, late_ppm)), Tri(*map(float, late_cost))),
    "label": DefectCost("Label / carton ID error", Tri(*map(float, label_ppm)), Tri(*map(float, label_cost))),
    "churn": DefectCost("Customer loss (proxy)", Tri(*map(float, churn_ppm)), Tri(*map(float, churn_cost))),
}

# =========================
# Hero header (dynamic)
# =========================
left, right = st.columns([1.55, 1])
with left:
    st.markdown(f"""
    <div class="hero-card">
      <div class="pill">Dynamic COPQ Calculator • Shipping/Fulfillment • Best ↔ Worst</div>
      <h1 style="margin:10px 0 6px 0;">Fulfillment COPQ Hero</h1>
      <div class="muted">Input volume + PPM defect rates + handling costs. Get best/likely/worst COPQ, plus KPIs for management, CFO, process engineering, and customers.</div>
      <div class="tiny" style="margin-top:10px;">Conversion reminder: <b>PPM = % × 10,000</b> (example: 0.12% = 1,200 PPM)</div>
    </div>
    """, unsafe_allow_html=True)
with right:
    if hero_b64:
        st.image("hero_infographic.png", caption="Hero theme", use_container_width=True)
    else:
        st.info("Place hero_infographic.png in the same folder as this app to show the hero image.")

st.divider()

# =========================
# Compute + KPIs
# =========================
df = compute_copq(orders, opps, defects)
totals = df[df["Defect Type"] == "TOTAL"].set_index("Scenario")["COPQ Cost"].to_dict()
best = float(totals.get("BEST", 0.0))
likely = float(totals.get("LIKELY", 0.0))
worst = float(totals.get("WORST", 0.0))

kpis = kpi_rates(defects)

k1, k2, k3, k4 = st.columns(4)
k1.metric("COPQ (Best / month)", money(best))
k2.metric("COPQ (Likely / month)", money(likely))
k3.metric("COPQ (Worst / month)", money(worst))
k4.metric("COPQ Range", f"{money(best)} → {money(worst)}")

k5, k6, k7, k8, k9 = st.columns(5)
k5.metric("Pick Accuracy (proxy)", pct(kpis["Pick Accuracy"]))
k6.metric("Damage-Free Rate (proxy)", pct(kpis["Damage-Free Rate"]))
k7.metric("On-Time Rate (proxy)", pct(kpis["On-Time Rate"]))
k8.metric("Label Accuracy (proxy)", pct(kpis["Label Accuracy"]))
k9.metric("Perfect Order (proxy)", pct(kpis["Perfect Order (proxy)"]))

st.divider()

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["1) COPQ Breakdown", "2) Best↔Worst Visual", "3) Compare to Actuals", "4) Industry Targets (Reference)"])

with tab1:
    st.subheader("COPQ Breakdown (Best / Likely / Worst)")
    st.dataframe(df, use_container_width=True)

    st.markdown("### Executive summary (Likely)")
    likely_rows = df[(df["Scenario"] == "LIKELY") & (df["Defect Type"] != "TOTAL")].copy()
    likely_rows = likely_rows.sort_values("COPQ Cost", ascending=False)
    st.dataframe(likely_rows[["Defect Type","PPM","Defects (count)","Cost/Defect","COPQ Cost"]], use_container_width=True)

with tab2:
    st.subheader("Scenario visual (monthly COPQ)")
    fig = plt.figure()
    plt.bar(["Best","Likely","Worst"], [best, likely, worst])
    plt.ylabel("COPQ $ / month")
    plt.title("Best vs Likely vs Worst COPQ")
    st.pyplot(fig)

    st.subheader("Where COPQ comes from (Likely)")
    likely_rows = df[(df["Scenario"] == "LIKELY") & (df["Defect Type"] != "TOTAL")].copy()
    likely_rows = likely_rows.sort_values("COPQ Cost", ascending=False)

    fig2 = plt.figure()
    plt.bar(likely_rows["Defect Type"], likely_rows["COPQ Cost"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("$ / month")
    plt.title("COPQ contributors (Likely)")
    st.pyplot(fig2)

with tab3:
    st.subheader("Compare to Actuals")
    st.caption("Upload CSV columns: month,copq_actual. Optional: pick_accuracy,on_time,damage_free,label_accuracy")
    up = st.file_uploader("Upload actuals CSV", type=["csv"])
    if up is None:
        st.info("Upload actual monthly COPQ to compare to best/likely/worst bands.")
    else:
        act = pd.read_csv(up)
        if "copq_actual" not in act.columns:
            st.error("CSV must contain: copq_actual (and optional month).")
        else:
            if "month" not in act.columns:
                act["month"] = range(1, len(act) + 1)
            act = act.sort_values("month")
            act["copq_best"] = best
            act["copq_likely"] = likely
            act["copq_worst"] = worst

            st.dataframe(act, use_container_width=True)

            fig = plt.figure()
            plt.plot(act["month"], act["copq_actual"], label="Actual COPQ")
            plt.plot(act["month"], act["copq_likely"], label="Model (Likely)")
            plt.plot(act["month"], act["copq_best"], label="Best")
            plt.plot(act["month"], act["copq_worst"], label="Worst")
            plt.xlabel("Month")
            plt.ylabel("COPQ ($)")
            plt.title("Actual COPQ vs Modeled Best/Likely/Worst")
            plt.legend()
            st.pyplot(fig)

with tab4:
    st.subheader("Industry targets (reference)")
    st.markdown("""
Use these as **conversation anchors** (targets vary by operation, SKU mix, and customer requirements):

- **Order picking accuracy**: best-in-class often cited around **≥99.9%**; typical ranges are lower.
- **Best-in-class DC picking accuracy** is also cited around **≥99.68%** in some benchmarking materials.
- **Warehouse order accuracy targets** are often referenced around **99.5%–99.9%** as “best-in-class” targets.

Tip: convert your PPM inputs into an accuracy proxy:
- **Accuracy ≈ 1 − (PPM/1,000,000)** (for that defect type)
""")

# Quick export
st.divider()
st.download_button("Download COPQ breakdown (CSV)", data=df.to_csv(index=False), file_name="copq_breakdown_best_likely_worst.csv", mime="text/csv")
