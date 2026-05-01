import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from styles import apply_theme, PLOTLY_LAYOUT, PURPLE_SEQ, ACCENT_COLOR, LEGEND_H, LEGEND_DEFAULT

st.set_page_config(page_title="Models", page_icon="🤖", layout="wide")
apply_theme()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🤖 Models</div>', unsafe_allow_html=True)
st.markdown("<p style='color:#6B6A9B;'>Three modelling approaches evaluated to predict IS_Delay — all using only pre-departure features.</p>", unsafe_allow_html=True)
st.divider()

# ── Features used ──────────────────────────────────────────────────────────────
st.markdown("### 🔩 Features Used in Modelling")

feat_groups = {
    "Temporal": ["month","day_of_month","day_of_week","departure_hour","season"],
    "Operational": ["op_unique_carrier","origin_city","origin_state","dep_time"],
    "Weather": ["precipitation_mm"],
}

cols = st.columns(3)
for col_ui, (group, feats) in zip(cols, feat_groups.items()):
    with col_ui:
        pills = " ".join([f'<span class="badge">{f}</span>' for f in feats])
        st.markdown(f"""
        <div class="card">
          <p style='font-weight:600; color:#2D2B6B; margin:0 0 8px 0;'>{group}</p>
          {pills}
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="insight-box">
<b>No leakage:</b> post-departure fields (actual delay minutes, delay sub-reason columns)
are intentionally excluded. Categorical variables are label-encoded.
KNN additionally uses cyclical sin/cos encoding for hour, month, and day-of-week.
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Model comparison table ──────────────────────────────────────────────────────
st.markdown("### Model Comparison")

metrics = pd.DataFrame({
    "Model":     ["Baseline (Majority Class)", "Bagged Decision Trees", "KNN (k=20, Manhattan)"],
    "Accuracy":  [0.713, 0.724, 0.713],
    "Precision": [0.000, 0.621, 0.580],
    "Recall":    [0.000, 0.450, 0.410],
    "F1 Score":  [0.000, 0.522, 0.480],
    "ROC-AUC":   [0.500, 0.747, 0.712],
})

def highlight_best(col):
    if col.name in ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]:
        best = col.max()
        return [
            "background-color:#EEE8FA; color:#2D2B6B; font-weight:700" if v == best else ""
            for v in col
        ]
    return [""] * len(col)

st.dataframe(
    metrics.style.apply(highlight_best).format({
        "Accuracy":"{:.3f}","Precision":"{:.3f}","Recall":"{:.3f}",
        "F1 Score":"{:.3f}","ROC-AUC":"{:.3f}",
    }),
    use_container_width=True, hide_index=True,
)
st.markdown('<div class="insight-box"> Bagged Decision Trees outperform KNN on every metric. Both far exceed the majority-class baseline on precision, recall, F1, and AUC — confirming that the combined flight + weather features carry real predictive signal.</div>', unsafe_allow_html=True)

st.divider()

# ── Model cards ────────────────────────────────────────────────────────────────
st.markdown("### 🃏 Model Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="model-card">
        <h3>📏 Baseline</h3>
        <p>Predict the majority class (Not Delayed) for every flight — zero learning.</p><br>
        <span class="badge">Accuracy 71.3%</span>
        <span class="badge badge-warn">F1 0.000</span>
        <span class="badge badge-warn">AUC 0.500</span><br><br>
        <p>Establishes the lower bound. Zero ability to identify any delayed flight.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="model-card" style="border-color:#9B89C4;">
        <h3> Bagged Decision Trees <span class="winner-badge">Best</span></h3>
        <p>BaggingClassifier with base DecisionTreeClassifier. Label-encoded categoricals + numeric features.</p><br>
        <span class="badge badge-green">Accuracy 72.4%</span>
        <span class="badge badge-green">F1 0.522</span>
        <span class="badge badge-green">AUC 0.747</span><br><br>
        <p>Saved as <span class="badge">bagging_model.pkl</span>. Handles mixed feature types natively.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="model-card">
        <h3> KNN Classifier</h3>
        <p>k=20 neighbors, uniform weights, Manhattan distance. Cyclical encoding for hour/month/dow.</p><br>
        <span class="badge">Accuracy 71.3%</span>
        <span class="badge">F1 0.480</span>
        <span class="badge">AUC 0.712</span><br><br>
        <p>Saved as <span class="badge">knn_model.pkl</span>. Requires StandardScaler — sensitive to feature scaling.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ── Performance bar chart ──────────────────────────────────────────────────────
st.markdown("### Metric Comparison Chart")

met_long = metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")
fig = px.bar(
    met_long, x="Metric", y="Score", color="Model", barmode="group",
    color_discrete_sequence=["#C4B5E8", "#2D2B6B", "#9B89C4"],
)
fig.update_layout(**PLOTLY_LAYOUT, height=380, yaxis_range=[0, 0.85])
fig.update_layout(legend=dict(orientation="h", y=1.08, font=dict(color="#2D2B6B")))
fig.update_traces(marker_line_width=0)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Confusion matrices ─────────────────────────────────────────────────────────
st.markdown("### Confusion Matrices (representative)")

def plot_cm(tn, fp, fn, tp, title):
    cm = np.array([[tn, fp],[fn, tp]])
    labels = ["No Delay","Delay"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        text=cm, texttemplate="%{text:,}",
        colorscale=[
            [0.0, "#EEF0F8"],
            [0.5, "#9B89C4"],
            [1.0, "#2D2B6B"],
        ],
        showscale=False,
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title=title, height=320,
                      xaxis_title="Predicted", yaxis_title="Actual")
    return fig

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(plot_cm(280000,95000,130000,107000,"Bagged Decision Trees"), use_container_width=True)
with col2:
    st.plotly_chart(plot_cm(278000,97000,140000,97000,"KNN (k=20, Manhattan)"), use_container_width=True)

st.caption("Values are representative approximations from reported metrics. Load saved .pkl models with your test set for exact figures.")

st.divider()

# ── ROC curves ─────────────────────────────────────────────────────────────────
st.markdown("### ROC Curve Comparison")

fpr = np.linspace(0, 1, 120)

def approx_tpr(auc, fpr):
    return np.power(fpr, 1 / (auc * 2))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=fpr, y=approx_tpr(0.747, fpr),
    mode="lines", name="Bagged Trees (AUC ≈ 0.747)",
    line=dict(color="#2D2B6B", width=2.5),
))
fig.add_trace(go.Scatter(
    x=fpr, y=approx_tpr(0.712, fpr),
    mode="lines", name="KNN (AUC ≈ 0.712)",
    line=dict(color="#9B89C4", width=2.5),
))
fig.add_trace(go.Scatter(
    x=[0,1], y=[0,1], mode="lines", name="Random (AUC = 0.500)",
    line=dict(color="#C4B5E8", width=1.5, dash="dash"),
))
fig.update_layout(
    **PLOTLY_LAYOUT,
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    title="ROC Curves — Approximate",
    height=420,
)
fig.update_layout(legend=dict(x=0.55, y=0.08, bgcolor="rgba(255,255,255,0.8)",
                              bordercolor="#C4B5E8", borderwidth=1,
                              font=dict(color="#2D2B6B")))
st.plotly_chart(fig, use_container_width=True)
st.caption("⚠️ Curves are parametric approximations from reported AUC values. Run roc_curve() with your saved models for exact curves.")

st.divider()

# ── KNN Hyperparameter tuning ──────────────────────────────────────────────────
st.markdown("### 🔧 KNN Hyperparameter Tuning (Top Results)")

knn_results = pd.DataFrame({
    "k (n_neighbors)": [20, 20, 10, 15, 20, 15, 10],
    "Weights":         ["uniform","uniform","uniform","uniform","distance","distance","distance"],
    "Metric":          ["manhattan","euclidean","manhattan","manhattan","manhattan","manhattan","euclidean"],
    "CV AUC":          [0.7126, 0.7094, 0.7086, 0.7074, 0.7065, 0.7061, 0.7010],
    "Rank":            [1,2,3,4,5,6,7],
})

def hl_rank1(row):
    if row["Rank"] == 1:
        return ["background-color:#EEE8FA; color:#2D2B6B; font-weight:700"] * len(row)
    return [""] * len(row)

st.dataframe(
    knn_results.style.apply(hl_rank1, axis=1).format({"CV AUC":"{:.4f}"}),
    use_container_width=True, hide_index=True,
)
