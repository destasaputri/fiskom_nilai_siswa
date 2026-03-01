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
st.set_page_config(page_title="Pink Dashboard", layout="wide")

# ==========================================================
# 🎀 FULL PINK THEME
# ==========================================================
st.markdown("""
<style>
html, body, [class*="css"]  {
    background-color: #ffe6f2;
}

.main {
    background: linear-gradient(to bottom right, #ffd6ec, #fff0f8);
}

h1, h2, h3 {
    color: #cc0066;
    text-align: center;
}

div[data-testid="metric-container"] {
    background-color: #fff0f6;
    border: 2px solid #ff99cc;
    padding: 15px;
    border-radius: 20px;
}

.stButton>button {
    background-color: #ff66b2;
    color: white;
    border-radius: 20px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

.stButton>button:hover {
    background-color: #ff3385;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================
st.title("🌸 Dashboard Analisis Hasil Siswa 🌸")
st.markdown("### 💖 Desta Saputri 💖")
st.markdown("NIM: 06111282429040")

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_excel("data_simulasi_50_siswa_20_soal.xlsx")
indikator = df.apply(pd.to_numeric, errors="coerce")

# ==========================================================
# NAVIGASI SLIDE
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
    st.header("💗 Ringkasan Performa Kelas")

    mean_scores = indikator.mean()
    nilai_rata2_kelas = indikator.mean(axis=1).mean()

    col1, col2 = st.columns(2)
    col1.metric("📈 Rata-rata Kelas", f"{nilai_rata2_kelas:.2f}")
    col2.metric("👩‍🎓 Jumlah Siswa", len(df))

    st.button("Next ➝", on_click=next_page)

# ==========================================================
# PAGE 1 – RATA-RATA PER SOAL
# ==========================================================
elif st.session_state.page == 1:
    st.header("🌺 Rata-rata Nilai per Soal")

    mean_scores = indikator.mean()

    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(mean_scores.index, mean_scores.values, color="#ff66b2")
    ax.set_ylabel("Rata-rata")
    ax.tick_params(axis='x', rotation=90)

    st.pyplot(fig)

    col1, col2 = st.columns(2)
    col1.button("⬅ Previous", on_click=prev_page)
    col2.button("Next ➝", on_click=next_page)

# ==========================================================
# PAGE 2 – KORELASI
# ==========================================================
elif st.session_state.page == 2:
    st.header("🌸 Korelasi Antar Soal")

    corr = indikator.corr()

    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(corr, cmap="RdPu", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

    st.pyplot(fig)

    col1, col2 = st.columns(2)
    col1.button("⬅ Previous", on_click=prev_page)
    col2.button("Next ➝", on_click=next_page)

# ==========================================================
# PAGE 3 – REGRESI
# ==========================================================
elif st.session_state.page == 3:
    st.header("🌷 Analisis Regresi")

    target_index = 19  # Soal 20

    X = sm.add_constant(indikator.drop(indikator.columns[target_index], axis=1))
    y = indikator.iloc[:, target_index]

    model = sm.OLS(y, X, missing="drop").fit()
    coef = model.params[1:]

    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(coef.index, coef.values, color="#ff66b2")
    ax.axhline(0, linestyle="--")
    ax.tick_params(axis='x', rotation=90)

    st.pyplot(fig)

    st.success(f"Soal paling berpengaruh: {coef.abs().idxmax()} 💖")

    col1, col2 = st.columns(2)
    col1.button("⬅ Previous", on_click=prev_page)
    col2.button("Next ➝", on_click=next_page)

# ==========================================================
# PAGE 4 – SEGMENTASI
# ==========================================================
elif st.session_state.page == 4:
    st.header("🎀 Segmentasi Siswa")

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

    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(polar=True)

    for i, row in cluster_mean.iterrows():
        values = row[labels].tolist()
        values += values[:1]
        ax.plot(angles, values, label=f"Cluster {i}")
        ax.fill(angles, values, alpha=0.2)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Radar Segmentasi 🌸")
    ax.legend()

    st.pyplot(fig)

    col1, col2 = st.columns(2)
    col1.button("⬅ Previous", on_click=prev_page)
    col2.button("🔄 Kembali ke Awal", on_click=lambda: st.session_state.update(page=0))
