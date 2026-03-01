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
# SOFT PINK THEME
# ==========================================================
st.markdown("""
<style>
html, body, .stApp { background-color: #ffd9ec; }
.main { background-color: #ffc2e0; }
h1, h2, h3 { color: #880e4f; }
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
# IDENTITAS
# ==========================================================
st.markdown(
    "<p style='text-align:left; font-weight:500; color:#6a1b9a;'>"
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
# NAVIGASI
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
# PAGE 0 – KPI + GRAFIK KECIL
# ==========================================================
if st.session_state.page == 0:

    st.header("Ringkasan Performa Siswa")

    total_nilai = indikator.sum(axis=1)

    jumlah_siswa = len(df)
    rata_kelas = total_nilai.mean()
    skor_tertinggi = total_nilai.max()
    skor_terendah = total_nilai.min()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah Siswa", jumlah_siswa)
    col2.metric("Rata-rata Kelas", f"{rata_kelas:.2f}")
    col3.metric("Skor Tertinggi", f"{skor_tertinggi:.0f}")
    col4.metric("Skor Terendah", f"{skor_terendah:.0f}")

    st.divider()

    st.subheader("Distribusi Total Nilai")

    fig, ax = plt.subplots(figsize=(5,3))
    ax.hist(total_nilai, bins=8)
    ax.set_xlabel("Total Nilai")
    ax.set_ylabel("Jumlah Siswa")
    st.pyplot(fig)

    st.button("Next ➝", on_click=next_page)

# ==========================================================
# PAGE 1 – RATA-RATA SOAL (2 GRAFIK)
# ==========================================================
elif st.session_state.page == 1:

    st.header("Rata-rata Nilai per Soal")

    mean_scores = indikator.mean()

    st.subheader("Grafik 1 – Semua Soal")
    fig1, ax1 = plt.subplots(figsize=(8,3))
    ax1.bar(mean_scores.index, mean_scores.values)
    ax1.tick_params(axis='x', rotation=90)
    st.pyplot(fig1)

    st.subheader("Grafik 2 – Detail Soal")
    soal_detail = st.selectbox("Pilih Soal:", mean_scores.index)

    fig2, ax2 = plt.subplots(figsize=(5,3))
    ax2.bar(soal_detail, mean_scores[soal_detail])
    ax2.set_ylim(0, indikator.max().max())
    st.pyplot(fig2)

    col1, col2 = st.columns(2)
    col1.button("⬅ Previous", on_click=prev_page)
    col2.button("Next ➝", on_click=next_page)

# ==========================================================
# PAGE 2 – KORELASI (2 GRAFIK)
# ==========================================================
elif st.session_state.page == 2:

    st.header("Korelasi Antar Soal")

    corr = indikator.corr()

    st.subheader("Grafik 1 – Heatmap Keseluruhan")
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(corr, cmap="RdPu", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    st.pyplot(fig)

    st.subheader("Grafik 2 – Detail Korelasi Soal")
    soal_korelasi = st.selectbox("Pilih Soal:", corr.columns)

    fig2, ax2 = plt.subplots(figsize=(8,3))
    ax2.bar(corr.columns, corr[soal_korelasi])
    ax2.axhline(0, linestyle="--")
    ax2.tick_params(axis='x', rotation=90)
    st.pyplot(fig2)

    col1, col2 = st.columns(2)
    col1.button("⬅ Previous", on_click=prev_page)
    col2.button("Next ➝", on_click=next_page)

# ==========================================================
# PAGE 3 – REGRESI (2 GRAFIK)
# ==========================================================
elif st.session_state.page == 3:

    st.header("Analisis Regresi")

    target_soal = st.slider("Pilih Soal Target", 1, 20, 20)
    target_index = target_soal - 1

    X = sm.add_constant(indikator.drop(indikator.columns[target_index], axis=1))
    y = indikator.iloc[:, target_index]

    model = sm.OLS(y, X, missing="drop").fit()
    coef = model.params[1:]

    st.subheader("Grafik 1 – Semua Koefisien")
    fig1, ax1 = plt.subplots(figsize=(8,3))
    ax1.bar(coef.index, coef.values)
    ax1.axhline(0, linestyle="--")
    ax1.tick_params(axis='x', rotation=90)
    st.pyplot(fig1)

    st.subheader("Grafik 2 – Detail Koefisien")
    soal_detail = st.selectbox("Pilih Variabel:", coef.index)

    fig2, ax2 = plt.subplots(figsize=(5,3))
    ax2.bar(soal_detail, coef[soal_detail])
    ax2.axhline(0, linestyle="--")
    st.pyplot(fig2)

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

    fig = plt.figure(figsize=(5,5))
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
