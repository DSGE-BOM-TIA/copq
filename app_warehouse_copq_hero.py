# Save As: app_warehouse_copq_hero.py
# Repo must include: app_warehouse_copq_hero.py + requirements.txt + hero_infographic.png (optional)

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------
# Helpers
# -------------------------
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

def inv_norm_cdf(p: float) -> float:
    """Inverse CDF for standard normal distribution (Acklam approximation)."""
    p = min(max(p, 1e-12), 1 - 1e-12)
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2*math.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

def sigma_from_ppm(ppm: float) -> float:
    # yield ~ 1 - ppm/1e6
    y = 1.0 - min(max(ppm_to_rate(ppm), 1e-12), 1 - 1e-12)
    return inv_norm_cdf(y)

def load_image_as_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def traffic_light(value: float, target: float, warn_band: float = 0.002) -> str:
    if value >= target:
        return "🟢"
    if value >= target - warn_band:
        return "🟡"
    return "🔴"

# -------------------------
# Page + Theme
# -------------------------
st.set_page_config(page_title="Fulfillment COPQ Hero", page_icon="📦", layout="wide")

hero_b64 = ""
try:
    hero_b64 = load_image_as_base64("hero_infographic.png")
except Exception:
    hero_b64 = ""

st.markdown(f"""
<style>
.stApp {{
  background:
    linear-gradient(rgba(6,10,18,0.97), rgba(6,10,18,0.975)),
    url('data:image/png;base64,{hero_b64}') no-repeat top center fixed;
  background-size: cover;
}}
html, body, [class*="css"] {{
  color: #FFFFFF !important;
  font-size: 18px !important;
}}
h1 {{ font-size: 46px !important; font-weight: 900 !important; }}
h2 {{ font-size: 32px !important; font-weight: 850 !important; }}
h3 {{ font-size: 24px !important; font-weight: 800 !important; }}
.hero-card {{
  padding: 22px;
  border-radius: 18px;
  border: 1px solid rgba(255,255,255,0.22);
  background: rgba(0,0,0,0.55);
  backdrop-filter: blur(12px);
  box-shadow: 0 10px 30px rgba(0,0,0,0.40);
}}
.pill {{
  display:inline-block; padding: 7px 12px; border-radius: 999px;
  background: rgba(0,229,255,0.14);
  border: 1px solid rgba(0,229,255,0.35);
  font-weight: 900; font-size: 13px;
}}
[data-testid="stMetricValue"] {{
  font-size: 36px !important;
  font-weight: 900 !important;
  color: #00E5FF !important;
}}
[data-testid="stMetricLabel"] {{
  font-size: 18px !important;
  font-weight: 850 !important;
}}
section[data-testid="stSidebar"] {{
  background-color: rgba(10,16,30,0.98) !important;
  border-right: 1px solid rgba(255,255,255,0.10);
}}
section[data-testid="stSidebar"] * {{
  font-size: 17px !important;
  color: #FFFFFF !important;
}}
button[data-baseweb="tab"] {{
  font-size: 18px !important;
  font-weight: 850 !important;
  color: #FFFFFF !important;
}}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Inputs
# -------------------------
st.sidebar.markdown("## Inputs")
st.sidebar.markdown("### 📦 Volume")
N = st.sidebar.number_input("Orders shipped per month (N)", min_value=0.0, max_value=10_000_000.0, value=250_000.0, step=1000.0)
O = st.sidebar.number_input("Opportunities per order (O)", min_value=1.0, max_value=20.0, value=4.0, step=1.0)
st.sidebar.caption("PPM = defects per million opportunities. If you only know %, use **PPM = % × 10,000**.")

def defect_block(name, ppm_default, cost_default, ppm_max=20000.0, cost_max=10000.0):
    st.sidebar.markdown(f"### {name}")
    ppm_best = st.sidebar.number_input(f"{name} PPM (Best)", min_value=0.0, max_value=float(ppm_max), value=float(ppm_default[0]), step=1.0, key=f"{name}_ppm_b")
    ppm_lik  = st.sidebar.number_input(f"{name} PPM (Likely)", min_value=0.0, max_value=float(ppm_max), value=float(ppm_default[1]), step=1.0, key=f"{name}_ppm_l")
    ppm_wst  = st.sidebar.number_input(f"{name} PPM (Worst)", min_value=0.0, max_value=float(ppm_max), value=float(ppm_default[2]), step=1.0, key=f"{name}_ppm_w")
    c_best   = st.sidebar.number_input(f"{name} Cost/defect $ (Best)", min_value=0.0, max_value=float(cost_max), value=float(cost_default[0]), step=1.0, key=f"{name}_c_b")
    c_lik    = st.sidebar.number_input(f"{name} Cost/defect $ (Likely)", min_value=0.0, max_value=float(cost_max), value=float(cost_default[1]), step=1.0, key=f"{name}_c_l")
    c_wst    = st.sidebar.number_input(f"{name} Cost/defect $ (Worst)", min_value=0.0, max_value=float(cost_max), value=float(cost_default[2]), step=1.0, key=f"{name}_c_w")
    ppm_best, ppm_lik, ppm_wst = sorted([ppm_best, ppm_lik, ppm_wst])
    c_best, c_lik, c_wst = sorted([c_best, c_lik, c_wst])
    return (ppm_best, ppm_lik, ppm_wst), (c_best, c_lik, c_wst)

damage_ppm, damage_cost = defect_block("Damage", (300.0, 900.0, 2500.0), (20.0, 80.0, 250.0))
mispick_ppm, mispick_cost = defect_block("Mispick", (200.0, 1200.0, 4000.0), (25.0, 120.0, 400.0))
late_ppm, late_cost = defect_block("Late/SLA", (80.0, 400.0, 1200.0), (10.0, 60.0, 200.0))
label_ppm, label_cost = defect_block("Label Error", (50.0, 250.0, 900.0), (8.0, 45.0, 150.0))

include_churn = st.sidebar.checkbox("Include customer churn proxy", value=False)
if include_churn:
    churn_ppm, churn_cost = defect_block("Churn Proxy", (1.0, 5.0, 20.0), (200.0, 900.0, 3000.0), ppm_max=2000.0, cost_max=20000.0)
else:
    churn_ppm, churn_cost = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

st.sidebar.markdown("## Investment + ROI")
investment = st.sidebar.number_input("One-time improvement cost ($)", min_value=0.0, max_value=50_000_000.0, value=50_000.0, step=1000.0)
monthly_run_cost = st.sidebar.number_input("Ongoing monthly cost ($)", min_value=0.0, max_value=5_000_000.0, value=5_000.0, step=500.0)
expected_reduction = st.sidebar.slider("Expected COPQ reduction (%)", min_value=0.0, max_value=90.0, value=20.0, step=1.0) / 100.0

# -------------------------
# Compute (best/likely/worst)
# -------------------------
def defect_costs(ppm_trip, cost_trip):
    def one(ppm, c):
        dcount = N * O * ppm_to_rate(ppm)
        return dcount, dcount * c
    b = one(ppm_trip[0], cost_trip[0])
    l = one(ppm_trip[1], cost_trip[1])
    w = one(ppm_trip[2], cost_trip[2])
    return b, l, w

streams = {
    "Damage": defect_costs(damage_ppm, damage_cost),
    "Mispick": defect_costs(mispick_ppm, mispick_cost),
    "Late/SLA": defect_costs(late_ppm, late_cost),
    "Label Error": defect_costs(label_ppm, label_cost),
}
if include_churn:
    streams["Churn Proxy"] = defect_costs(churn_ppm, churn_cost)

best = sum(v[0][1] for v in streams.values())
likely = sum(v[1][1] for v in streams.values())
worst = sum(v[2][1] for v in streams.values())

# Top driver in likely
top_driver = max(streams.items(), key=lambda kv: kv[1][1][1])[0] if streams else "—"
top_driver_cost = streams[top_driver][1][1] if streams else 0.0

# KPI proxies
pick_acc = 1.0 - ppm_to_rate(mispick_ppm[1])
dmg_free = 1.0 - ppm_to_rate(damage_ppm[1])
on_time = 1.0 - ppm_to_rate(late_ppm[1])
lab_acc = 1.0 - ppm_to_rate(label_ppm[1])
perfect = pick_acc * dmg_free * on_time * lab_acc

combined_ppm_likely = damage_ppm[1] + mispick_ppm[1] + late_ppm[1] + label_ppm[1]
sigma_approx = sigma_from_ppm(combined_ppm_likely)

# ROI
monthly_savings_net = (likely * expected_reduction) - monthly_run_cost
annual_savings = monthly_savings_net * 12.0
payback_months = (investment / monthly_savings_net) if monthly_savings_net > 0 else float("inf")

# -------------------------
# Header + Math
# -------------------------
st.markdown("""
<div class="hero-card">
  <div class="pill">Bold • Polished • Decision-ready</div>
  <h1 style="margin:10px 0 6px 0;">Fulfillment COPQ Hero Calculator</h1>
  <div style="opacity:0.9;">
    Input <b>volume + PPM + cost per incident</b> → see <b>best/likely/worst COPQ</b>, <b>top loss driver</b>, and <b>ROI/payback</b>.
  </div>
</div>
""", unsafe_allow_html=True)

with st.expander("📘 Math model (purpose • formulas • symbols)", expanded=True):
    st.write("Purpose: quantify Cost of Poor Quality (COPQ) for shipping/fulfillment defects and support ROI decisions.")
    st.latex(r"r = \frac{PPM}{1{,}000{,}000}")
    st.latex(r"D_i = N \cdot O \cdot r_i")
    st.latex(r"COPQ_i = D_i \cdot C_i")
    st.latex(r"COPQ_{TOTAL} = \sum_i COPQ_i")
    st.markdown("""
**Symbols**
- **N**: orders per month  
- **O**: opportunities per order  
- **PPM**: defects per million opportunities  
- **Dᵢ**: defect count (expected)  
- **Cᵢ**: cost per defect  
""")

# -------------------------
# KPIs + lights
# -------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("COPQ (Best / mo)", money(best))
c2.metric("COPQ (Likely / mo)", money(likely))
c3.metric("COPQ (Worst / mo)", money(worst))
c4.metric("Top Loss Driver (Likely)", f"{top_driver} • {money(top_driver_cost)}")

st.markdown("## Quality KPI Targets (proxy)")
target_pick = 0.9990
target_damagefree = 0.9980
target_ontime = 0.9900
target_label = 0.9990
target_perfect = 0.9860

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric(f"{traffic_light(pick_acc, target_pick)} Pick Accuracy", pct(pick_acc))
k2.metric(f"{traffic_light(dmg_free, target_damagefree)} Damage-Free", pct(dmg_free))
k3.metric(f"{traffic_light(on_time, target_ontime)} On-Time", pct(on_time))
k4.metric(f"{traffic_light(lab_acc, target_label)} Label Accuracy", pct(lab_acc))
k5.metric(f"{traffic_light(perfect, target_perfect)} Perfect Order", pct(perfect))

st.markdown(f"**Combined defect PPM (Likely)** ≈ **{combined_ppm_likely:,.0f}** → **Sigma (approx)** ≈ **{sigma_approx:.2f}**")
st.divider()

# -------------------------
# ROI panel
# -------------------------
st.markdown("## Investment Case (ROI + Payback)")
r1,r2,r3 = st.columns(3)
r1.metric("Expected reduction", pct(expected_reduction))
r2.metric("Net monthly savings", money(monthly_savings_net))
r3.metric("Net annual savings", money(annual_savings))

if math.isfinite(payback_months):
    st.success(f"Estimated payback: **{payback_months:.1f} months**")
else:
    st.warning("Payback not achieved with current assumptions (net monthly savings ≤ 0).")

# -------------------------
# Narrated Mode
# -------------------------
st.divider()
st.markdown("## Narrated Mode")
aud = st.radio("Audience lens:", ["CFO", "Operations", "Process Engineer"], horizontal=True)

if aud == "CFO":
    st.markdown(f"""
- Likely COPQ: **{money(likely)}/month** (**{money(likely*12)}/year**)  
- Biggest driver: **{top_driver}** (**{money(top_driver_cost)}/month**)  
- With a **{pct(expected_reduction)}** reduction, net savings ≈ **{money(monthly_savings_net)}/month**.
""")
elif aud == "Operations":
    st.markdown(f"""
- Fix first: **{top_driver}**  
- That stream is the highest-leverage lever on COPQ today.
""")
else:
    st.latex(r"D = N \times O \times (PPM / 1{,}000{,}000)")
    st.latex(r"COPQ = D \times Cost")
    st.write("Reduce PPM via root-cause elimination and process controls.")

# -------------------------
# Tabs
# -------------------------
tab1, tab2 = st.tabs(["Breakdown", "Visuals"])

with tab1:
    rows=[]
    for name, trio in streams.items():
        rows.append({"Defect": name, "Best COPQ": trio[0][1], "Likely COPQ": trio[1][1], "Worst COPQ": trio[2][1]})
    out = pd.DataFrame(rows).sort_values("Likely COPQ", ascending=False)
    st.dataframe(out, use_container_width=True)

with tab2:
    fig = plt.figure()
    plt.bar(["Best","Likely","Worst"], [best, likely, worst])
    plt.ylabel("COPQ $ / month")
    st.pyplot(fig)

    fig2 = plt.figure()
    plt.bar(out["Defect"], out["Likely COPQ"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("$ / month")
    st.pyplot(fig2)
