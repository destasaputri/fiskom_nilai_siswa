import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(page_title="Dashboard Analisis Hasil Siswa", layout="wide")

# ==========================================================
# 🌸 SOFT PINK THEME (TIDAK TOO MUCH)
# ==========================================================
st.markdown("""
<style>
body {
    background-color: #fff5fa;
}
.main {
    background: linear-gradient(to right, #fff0f6, #ffe6f2);
}
h1, h2, h3 {
    color: #c2185b;
}
div[data-testid="metric-container"] {
    background-color: #ffe6f2;
    border: 1px solid #f8bbd0;
    padding: 12px;
    border-radius: 12px;
}
.stButton>button {
    background-color: #ec407a;
    color: white;
    border-radius: 8px;
}
.stButton>button:hover {
    background-color: #d81b60;
    color: white;
}
section[data-testid="stSidebar"] {
    background-color: #fff0f6;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# IDENTITAS DI KIRI ATAS (ATAS JUDUL)
# ==========================================================
st.markdown(
    "<p style='text-align:left; font-weight:500; color:#ad1457;'>"
    "Desta Saputri<br>NIM: 06111282429040"
    "</p>",
    unsafe_allow_html=True
)

st.title("🌸📊 Dashboard Analisis Hasil Siswa")
st.markdown("### Analisis Performa Akademik Berbasis Data")

st.divider()

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_excel("data_simulasi_50_siswa_20_soal.xlsx")
indikator = df.apply(pd.to_numeric, errors="coerce")

# ==========================================================
# NAVIGASI HALAMAN (NEXT BUTTON)
# ==========================================================
if "page" not in st.session_state:
    st.session_state.page = 0

def next_page():
    st.session_state.page += 1

def prev_page():
    st.session_state.page -= 1

# ==========================================================
# PAGE 0 – KPI
# ==========================================================
if st.session_state.page == 0:

    mean_scores = indikator.mean()
    nilai_rata2_kelas = indikator.mean(axis=1).mean()

    def kategori_nilai(x):
        if x >= 85: return "Sangat Baik"
        elif x >= 75: return "Baik"
        elif x >= 65: return "Cukup"
        else: return "Perlu Bimbingan"

    col1, col2, col3 = st.columns(3)
    col1.metric("Rata-rata Kelas", f"{nilai_rata2_kelas:.2f}")
    col2.metric("Kategori", kategori_nilai(nilai_rata2_kelas))
    col3.metric("Jumlah Siswa", len(df))

    st.button("Next ➝", on_click=next_page)

# ==========================================================
# PAGE 1 – RATA-RATA PER SOAL
# ==========================================================
elif st.session_state.page == 1:

    st.header("Rata-rata Nilai per Soal")

    mean_scores = indikator.mean()

    fig_soal, ax_soal = plt.subplots(figsize=(10,4))
    ax_soal.bar(mean_scores.index, mean_scores.values, color="#ec407a")
    ax_soal.set_ylabel("Rata-rata Nilai")
    ax_soal.tick_params(axis='x', rotation=90)

    st.pyplot(fig_soal)

    col1, col2 = st.columns(2)
    col1.button("⬅ Previous", on_click=prev_page)
    col2.button("Next ➝", on_click=next_page)

# ==========================================================
# PAGE 2 – KORELASI
# ==========================================================
elif st.session_state.page == 2:

    st.header("Korelasi Antar Soal")

    corr = indikator.corr()

    fig_corr, ax_corr = plt.subplots(figsize=(7,6))
    im = ax_corr.imshow(corr, cmap="RdPu", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax_corr)

    ax_corr.set_xticks(range(len(corr.columns)))
    ax_corr.set_yticks(range(len(corr.columns)))
    ax_corr.set_xticklabels(corr.columns, rotation=90)
    ax_corr.set_yticklabels(corr.columns)

    st.pyplot(fig_corr)

    col1, col2 = st.columns(2)
    col1.button("⬅ Previous", on_click=prev_page)
    col2.button("Next ➝", on_click=next_page)

# ==========================================================
# PAGE 3 – REGRESI
# ==========================================================
elif st.session_state.page == 3:

    st.header("Analisis Regresi")

    target_index = 19

    X = sm.add_constant(indikator.drop(indikator.columns[target_index], axis=1))
    y = indikator.iloc[:, target_index]

    model = sm.OLS(y, X, missing="drop").fit()
    coef = model.params[1:]
    r2 = model.rsquared

    fig_reg, ax_reg = plt.subplots(figsize=(10,5))
    ax_reg.bar(coef.index, coef.values, color="#ec407a")
    ax_reg.axhline(0, linestyle="--")
    ax_reg.tick_params(axis='x', rotation=90)

    st.pyplot(fig_reg)

    st.success(f"Soal paling berpengaruh: {coef.abs().idxmax()}")
    st.info(f"Model menjelaskan {r2*100:.1f}% variasi nilai")

    col1, col2 = st.columns(2)
    col1.button("⬅ Previous", on_click=prev_page)
    col2.button("Next ➝", on_click=next_page)

# ==========================================================
# PAGE 4 – SEGMENTASI
# ==========================================================
elif st.session_state.page == 4:

    st.header("Segmentasi Performa Siswa")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(indikator.fillna(indikator.mean()))

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_label = kmeans.fit_predict(X_scaled)

    indikator_cluster = indikator.copy()
    indikator_cluster["Cluster"] = cluster_label

    cluster_mean = indikator_cluster.groupby("Cluster").mean()

    labels = indikator.columns.tolist()
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig_rad = plt.figure(figsize=(6,6))
    ax_rad = plt.subplot(polar=True)

    for i, row in cluster_mean.iterrows():
        values = row[labels].tolist()
        values += values[:1]
        ax_rad.plot(angles, values, label=f"Cluster {i}")
        ax_rad.fill(angles, values, alpha=0.2)

    ax_rad.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax_rad.set_title("Radar Segmentasi")
    ax_rad.legend(loc="upper right")

    st.pyplot(fig_rad)

    col1, col2 = st.columns(2)
    col1.button("⬅ Previous", on_click=prev_page)
    col2.button("Kembali ke Awal", on_click=lambda: st.session_state.update(page=0))
