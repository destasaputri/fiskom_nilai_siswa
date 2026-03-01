import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================================
# KONFIGURASI
# ==========================================================
st.set_page_config(page_title="Dashboard Analisis", layout="wide")

# ==========================================================
# 🌸 FULL PINK BACKGROUND
# ==========================================================
st.markdown("""
<style>
html, body, .stApp {
    background-color: #ffd9ec;
}

.main {
    background-color: #ffc2e0;
}

h1, h2, h3 {
    color: #ad1457;
}

div[data-testid="metric-container"] {
    background-color: #ffe6f2;
    border-radius: 12px;
    padding: 15px;
}

.stButton>button {
    background-color: #d81b60;
    color: white;
    border-radius: 8px;
}

.stButton>button:hover {
    background-color: #ad1457;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# IDENTITAS (KIRI ATAS)
# ==========================================================
st.markdown(
    "<p style='text-align:left; font-weight:500; color:#880e4f;'>"
    "Desta Saputri<br>NIM: 06111282429040"
    "</p>",
    unsafe_allow_html=True
)

st.title("🌸 Dashboard Analisis Hasil Siswa")

st.divider()

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_excel("data_simulasi_50_siswa_20_soal.xlsx")
indikator = df.apply(pd.to_numeric, errors="coerce")

# ==========================================================
# INTERAKTIF NAVIGASI
# ==========================================================
if "page" not in st.session_state:
    st.session_state.page = 0

def next_page():
    if st.session_state.page < 4:
        st.session_state.page += 1

def prev_page():
    if st.session_state.page > 0:
        st.session_state.page -= 1

# ==========================================================
# PAGE 0 – KPI
# ==========================================================
if st.session_state.page == 0:

    st.header("Ringkasan Performa")

    nilai_rata2_kelas = indikator.mean(axis=1).mean()

    col1, col2 = st.columns(2)
    col1.metric("Rata-rata Kelas", f"{nilai_rata2_kelas:.2f}")
    col2.metric("Jumlah Siswa", len(df))

    st.button("Next ➝", on_click=next_page)

# ==========================================================
# PAGE 1 – RATA-RATA PER SOAL
# ==========================================================
elif st.session_state.page == 1:

    st.header("Rata-rata Nilai per Soal")

    mean_scores = indikator.mean()

    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(mean_scores.index, mean_scores.values, color="#c2185b")
    ax.set_ylabel("Rata-rata")
    ax.tick_params(axis='x', rotation=90)

    st.pyplot(fig)

    soal = st.selectbox("Lihat detail soal:", mean_scores.index)
    st.info(f"Rata-rata nilai {soal}: {mean_scores[soal]:.2f}")

    col1, col2 = st.columns(2)
    col1.button("⬅ Previous", on_click=prev_page)
    col2.button("Next ➝", on_click=next_page)

# ==========================================================
# PAGE 2 – KORELASI
# ==========================================================
elif st.session_state.page == 2:

    st.header("Korelasi Antar Soal")

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
# PAGE 3 – REGRESI INTERAKTIF
# ==========================================================
elif st.session_state.page == 3:

    st.header("Analisis Regresi Interaktif")

    target_soal = st.slider("Pilih Soal Target", 1, 20, 20)
    target_index = target_soal - 1

    X = sm.add_constant(indikator.drop(indikator.columns[target_index], axis=1))
    y = indikator.iloc[:, target_index]

    model = sm.OLS(y, X, missing="drop").fit()
    coef = model.params[1:]

    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(coef.index, coef.values, color="#ad1457")
    ax.axhline(0, linestyle="--")
    ax.tick_params(axis='x', rotation=90)

    st.pyplot(fig)

    st.success(f"Soal paling berpengaruh: {coef.abs().idxmax()}")

    col1, col2 = st.columns(2)
    col1.button("⬅ Previous", on_click=prev_page)
    col2.button("Next ➝", on_click=next_page)

# ==========================================================
# PAGE 4 – SEGMENTASI
# ==========================================================
elif st.session_state.page == 4:

    st.header("Segmentasi Performa")

    jumlah_cluster = st.slider("Jumlah Cluster", 2, 5, 3)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(indikator.fillna(indikator.mean()))

    kmeans = KMeans(n_clusters=jumlah_cluster, random_state=42, n_init=10)
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
    ax.legend(loc="upper right")

    st.pyplot(fig)

    col1, col2 = st.columns(2)
    col1.button("⬅ Previous", on_click=prev_page)
    col2.button("Kembali ke Awal", on_click=lambda: st.session_state.update(page=0))
