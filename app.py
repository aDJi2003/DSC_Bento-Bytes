"""
app.py — Dashboard interaktif prediksi keberhasilan UMKM (Tim Bento Bytes).
Versi presentasi: kontrol ambang global di sidebar (efek langsung terlihat di
panel mana pun), tombol skenario siap-pakai, dan penjelasan lokal yang hidup.

Jalankan:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from utils import engineer_features, FEATURE_META, RAW_FEATURES, TARGET

st.set_page_config(page_title="Dashboard UMKM — Bento Bytes",
                   page_icon="📊", layout="wide")

PRIMARY, GREEN, RED, GREY = "#3f72af", "#2e8b57", "#d1495b", "#9aa5b1"

# ---- gaya: perbesar font agar nyaman dilihat saat diproyeksikan ----
st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
h1 {font-size: 2.1rem !important;}
.stTabs [data-baseweb="tab"] {font-size: 1.05rem; padding: 8px 18px;}
[data-testid="stMetricValue"] {font-size: 1.9rem;}
</style>
""", unsafe_allow_html=True)


# ---------------- artefak (cached) ----------------
@st.cache_resource
def load_model(): return joblib.load("artifacts/model.joblib")
@st.cache_data
def load_metrics(): return joblib.load("artifacts/metrics.joblib")
@st.cache_data
def load_data(): return joblib.load("artifacts/data_fe.joblib")
@st.cache_data
def load_coefs(): return joblib.load("artifacts/coefs.joblib")

model = load_model()
metrics = load_metrics()
df = load_data()
coefs = load_coefs()
scaler = model.named_steps["scaler"]
clf = model.named_steps["model"]

# ---------------- preset skenario ----------------
PRESETS = {
    "🟢 UMKM Berpotensi": dict(Age=32, Education=4, Initial_Capital=1, Financial_Record_Keeping=1,
        Internet_Usage=1, Business_Plan=1, Marketing_Effort=6, Partnership=1,
        Parent_Business_Experience=1, Industry_Experience=12, Owner_Gender=1, Professional_Advice=6),
    "🔴 UMKM Berisiko": dict(Age=55, Education=2, Initial_Capital=0, Financial_Record_Keeping=0,
        Internet_Usage=0, Business_Plan=0, Marketing_Effort=3, Partnership=0,
        Parent_Business_Experience=0, Industry_Experience=2, Owner_Gender=1, Professional_Advice=2),
    "🟡 Kasus Ambigu": dict(Age=38, Education=3, Initial_Capital=1, Financial_Record_Keeping=1,
        Internet_Usage=1, Business_Plan=0, Marketing_Effort=4, Partnership=0,
        Parent_Business_Experience=0, Industry_Experience=6, Owner_Gender=1, Professional_Advice=4),
}

# init session_state untuk input & threshold
for feat, meta in FEATURE_META.items():
    st.session_state.setdefault(f"in_{feat}", meta["default"])
st.session_state.setdefault("threshold", 0.50)

def apply_preset(profile):
    for k, v in profile.items():
        st.session_state[f"in_{k}"] = v


# ================= SIDEBAR (kontrol global, selalu terlihat) =================
with st.sidebar:
    st.markdown("## ⚙️ Kontrol")
    st.markdown("**Ambang keputusan (threshold)**")
    st.slider("Ambang", 0.05, 0.95, key="threshold", step=0.01,
              label_visibility="collapsed")
    thr = st.session_state["threshold"]
    st.caption(f"UMKM diprediksi **berhasil** bila probabilitas ≥ **{thr:.2f}**. "
               "Geser untuk melihat efeknya langsung di panel kanan.")
    st.divider()
    st.markdown("**Skenario contoh** (klik untuk memuat profil):")
    for name, prof in PRESETS.items():
        st.button(name, use_container_width=True,
                  on_click=apply_preset, args=(prof,))
    st.divider()
    st.caption("Tim **Bento Bytes** · Model final: Logistic Regression (L1)")

thr = st.session_state["threshold"]

# ================= HEADER + KPI STRIP =================
st.title("📊 Prediksi Keberhasilan UMKM")
k1, k2, k3, k4 = st.columns(4)
k1.metric("F1-score (CV)", f"{metrics['cv_f1']:.3f}")
k2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
cm0 = metrics["cm"]
k3.metric("Recall (Berhasil)", f"{cm0[1,1]/(cm0[1,0]+cm0[1,1]):.3f}")
k4.metric("Total UMKM", len(df))

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Prediktor", "🧠 Mengapa?", "🔎 Data", "📈 Performa"])

# ======================================================================
# TAB 1 — PREDIKTOR
# ======================================================================
with tab1:
    st.markdown("#### Susun profil UMKM → lihat peluang keberhasilannya seketika")
    st.caption("Gunakan tombol **Skenario contoh** di sidebar untuk memuat profil "
               "dalam satu klik, atau atur manual di bawah.")

    colIn, colOut = st.columns([1, 1.25])

    with colIn:
        st.markdown("**Profil UMKM**")
        for feat in RAW_FEATURES:
            meta = FEATURE_META[feat]
            if meta["type"] == "bool":
                st.toggle(meta["label"], key=f"in_{feat}")
            else:
                st.slider(meta["label"], meta["min"], meta["max"], key=f"in_{feat}")

    with colOut:
        inputs = {f: st.session_state[f"in_{f}"] for f in RAW_FEATURES}
        row = engineer_features(pd.DataFrame([inputs]))
        proba = float(model.predict_proba(row)[0, 1])
        verdict = "BERHASIL" if proba >= thr else "TIDAK BERHASIL"
        color = GREEN if proba >= thr else RED

        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=proba * 100,
            number={"suffix": "%", "font": {"size": 52, "color": color}},
            gauge={"axis": {"range": [0, 100], "tickwidth": 1},
                   "bar": {"color": color, "thickness": 0.7},
                   "steps": [{"range": [0, thr*100], "color": "#f7e0e0"},
                             {"range": [thr*100, 100], "color": "#dcefe0"}],
                   "threshold": {"line": {"color": "black", "width": 4},
                                 "value": thr*100}}))
        gauge.update_layout(height=300, margin=dict(t=30, b=10, l=30, r=30))
        st.plotly_chart(gauge, use_container_width=True)
        st.markdown(
            f"<div style='text-align:center;font-size:1.6rem;font-weight:700;"
            f"color:{color}'>{verdict}</div>"
            f"<div style='text-align:center;color:gray'>pada ambang {thr:.2f} "
            f"(garis hitam) · skor kesiapan manajerial "
            f"{int(row['Management_Readiness'].iloc[0])}/4</div>",
            unsafe_allow_html=True)

# ======================================================================
# TAB 2 — MENGAPA (interpretasi global)
# ======================================================================
with tab2:
    st.markdown("#### Faktor apa yang paling menentukan keberhasilan UMKM?")
    cdf = coefs.reset_index(); cdf.columns = ["Fitur", "Koefisien"]
    fig = px.bar(cdf, x="Koefisien", y="Fitur", orientation="h",
                 color="Koefisien", color_continuous_scale=[RED, "#eeeeee", GREEN],
                 color_continuous_midpoint=0)
    fig.update_layout(height=480, coloraxis_showscale=False)
    fig.add_vline(x=0, line_color="black", line_width=1)
    st.plotly_chart(fig, use_container_width=True)
    st.success("**Temuan utama:** *Management_Readiness* (kesiapan manajerial "
               "kumulatif: modal + pencatatan keuangan + internet + rencana bisnis) "
               "mendominasi seluruh fitur. Faktor demografis (usia, gender, "
               "pendidikan) nyaris tak berpengaruh, keberhasilan ditentukan oleh "
               "**apa yang dilakukan** pemilik, bukan **siapa** pemiliknya. "
               "Karena faktor penentunya dapat dilatih, ia menjadi sasaran "
               "intervensi pembinaan yang tepat.")

# ======================================================================
# TAB 3 — DATA
# ======================================================================
with tab3:
    st.markdown("#### Pola keberhasilan dalam data")
    c1, c2 = st.columns(2)
    with c1:
        rate = (df.groupby("Management_Readiness")[TARGET].mean()*100).reset_index()
        rate.columns = ["Skor Kesiapan Manajerial", "Tingkat Keberhasilan (%)"]
        fig = px.bar(rate, x="Skor Kesiapan Manajerial", y="Tingkat Keberhasilan (%)",
                     text_auto=".0f", color="Tingkat Keberhasilan (%)",
                     color_continuous_scale="Blues")
        fig.update_layout(height=400, coloraxis_showscale=False,
                          title="Keberhasilan naik tajam seiring kesiapan manajerial")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        feat = st.selectbox("Jelajahi fitur lain:",
                            ["Professional_Advice", "Industry_Experience",
                             "Marketing_Effort", "Education", "Age"])
        g = (df.groupby(feat)[TARGET].mean()*100).reset_index()
        g.columns = [feat, "Tingkat Keberhasilan (%)"]
        fig = px.line(g, x=feat, y="Tingkat Keberhasilan (%)", markers=True)
        fig.update_traces(line_color=PRIMARY, line_width=3)
        fig.update_layout(height=400, title=f"Keberhasilan vs {feat}")
        st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# TAB 4 — PERFORMA (bereaksi langsung ke ambang di sidebar)
# ======================================================================
with tab4:
    st.markdown(f"#### Performa model pada ambang **{thr:.2f}** "
                "*(ubah di sidebar — panel ini ikut berubah seketika)*")
    proba_oof = metrics["oof_proba"]; y = metrics["y_true"]
    pred = (proba_oof >= thr).astype(int)
    tp = int(((pred==1)&(y==1)).sum()); fp = int(((pred==1)&(y==0)).sum())
    fn = int(((pred==0)&(y==1)).sum()); tn = int(((pred==0)&(y==0)).sum())
    prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9); f1 = 2*prec*rec/(prec+rec+1e-9)

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Precision", f"{prec:.3f}")
    mc2.metric("Recall", f"{rec:.3f}")
    mc3.metric("F1-score", f"{f1:.3f}")

    c1, c2 = st.columns(2)
    with c1:
        cm = np.array([[tn, fp], [fn, tp]])
        labels = ["Tidak Berhasil", "Berhasil"]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=labels, y=labels, labels=dict(x="Prediksi", y="Aktual"))
        fig.update_layout(height=380, coloraxis_showscale=False,
                          title=f"Confusion Matrix (ambang {thr:.2f})")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics["fpr"], y=metrics["tpr"], mode="lines",
                                 line=dict(color=PRIMARY, width=3),
                                 name=f"AUC={metrics['roc_auc']:.3f}"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                 line=dict(color="gray", dash="dash"), name="Acak"))
        fig.update_layout(height=380, title="Kurva ROC",
                          xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Perbandingan model**")
    comp = metrics["comparison"].copy()
    comp[["F1","ROC_AUC","Bal_Acc"]] = comp[["F1","ROC_AUC","Bal_Acc"]].round(3)
    st.dataframe(comp, use_container_width=True, hide_index=True)
