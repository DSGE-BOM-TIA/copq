# Save As: app_warehouse_copq_hero.py
# Deploy with requirements.txt and hero_infographic.png

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fulfillment COPQ Hero", layout="wide")

def money(x):
    return "${:,.0f}".format(float(x))

def ppm_to_rate(ppm):
    return ppm / 1_000_000

st.title("Fulfillment COPQ Hero Calculator")

st.sidebar.header("Volume Inputs")
N = st.sidebar.number_input("Orders per Month (N)", 0.0, 10000000.0, 250000.0)
O = st.sidebar.number_input("Opportunities per Order (O)", 1.0, 20.0, 4.0)

st.sidebar.header("Defect Inputs (PPM & Cost)")

def defect_block(name, ppm_default, cost_default):
    st.sidebar.subheader(name)
    ppm_best = st.sidebar.number_input(f"{name} PPM Best", 0.0, 20000.0, ppm_default[0])
    ppm_likely = st.sidebar.number_input(f"{name} PPM Likely", 0.0, 20000.0, ppm_default[1])
    ppm_worst = st.sidebar.number_input(f"{name} PPM Worst", 0.0, 20000.0, ppm_default[2])
    cost_best = st.sidebar.number_input(f"{name} Cost Best", 0.0, 10000.0, cost_default[0])
    cost_likely = st.sidebar.number_input(f"{name} Cost Likely", 0.0, 10000.0, cost_default[1])
    cost_worst = st.sidebar.number_input(f"{name} Cost Worst", 0.0, 10000.0, cost_default[2])
    return (ppm_best, ppm_likely, ppm_worst), (cost_best, cost_likely, cost_worst)

damage_ppm, damage_cost = defect_block("Damage",(300,900,2500),(20,80,250))
mispick_ppm, mispick_cost = defect_block("Mispick",(200,1200,4000),(25,120,400))
late_ppm, late_cost = defect_block("Late",(80,400,1200),(10,60,200))
label_ppm, label_cost = defect_block("Label Error",(50,250,900),(8,45,150))

def compute(ppm, cost):
    best = N*O*ppm_to_rate(ppm[0])*cost[0]
    likely = N*O*ppm_to_rate(ppm[1])*cost[1]
    worst = N*O*ppm_to_rate(ppm[2])*cost[2]
    return best, likely, worst

defects = {
    "Damage": compute(damage_ppm, damage_cost),
    "Mispick": compute(mispick_ppm, mispick_cost),
    "Late": compute(late_ppm, late_cost),
    "Label Error": compute(label_ppm, label_cost)
}

best = sum(v[0] for v in defects.values())
likely = sum(v[1] for v in defects.values())
worst = sum(v[2] for v in defects.values())

col1,col2,col3,col4 = st.columns(4)
col1.metric("Best COPQ", money(best))
col2.metric("Likely COPQ", money(likely))
col3.metric("Worst COPQ", money(worst))
col4.metric("Range", f"{money(best)} → {money(worst)}")

st.divider()

st.header("Narrated Mode")

audience = st.radio("Audience View:",["CFO","Operations","Process Engineer"],horizontal=True)

if audience=="CFO":
    st.write(f"Likely Monthly COPQ: {money(likely)}")
    st.write(f"Annual Impact: {money(likely*12)}")
    st.write("Reducing COPQ directly improves EBITDA.")

elif audience=="Operations":
    top = max(defects.items(), key=lambda x: x[1][1])[0]
    st.write(f"Top Defect Driver (Likely): {top}")
    st.write("Target this defect stream first for maximum impact.")

elif audience=="Process Engineer":
    st.latex(r"D = N * O * (PPM / 1,000,000)")
    st.latex(r"COPQ = D * Cost")
    st.write("Reduce PPM through root cause elimination.")

st.divider()

st.subheader("Scenario Comparison")
fig = plt.figure()
plt.bar(["Best","Likely","Worst"],[best,likely,worst])
plt.ylabel("COPQ per Month")
st.pyplot(fig)

st.subheader("Defect Breakdown")
df = pd.DataFrame({
    "Defect": defects.keys(),
    "Best": [v[0] for v in defects.values()],
    "Likely": [v[1] for v in defects.values()],
    "Worst": [v[2] for v in defects.values()]
})
st.dataframe(df)
