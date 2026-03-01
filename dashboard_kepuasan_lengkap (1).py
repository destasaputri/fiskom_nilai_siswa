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
st.set_page_config(page_title="Dashboard Analisis Hasil Siswa", layout="wide")

st.markdown("""
<style>
html, body, .stApp { background-color: #ffd9ec; }
h1, h2, h3 { color: #880e4f; }
.stButton>button {
    background-color: #d81b60;
    color: white;
    border-radius: 8px;
}
.stButton>button:hover {
    background-color: #ad1457;
}
</style>
""", unsafe_allow_html=True)

# IDENTITAS
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

# PALETTE PINK (KONSISTEN UNTUK SEMUA DIAGRAM SOAL)
pink_colors = [
    "#f8bbd0", "#f48fb1", "#f06292", "#ec407a",
    "#e91e63", "#d81b60", "#c2185b", "#ad1457",
    "#880e4f", "#ff80ab", "#ff4081", "#ff1a75",
    "#ff66a3", "#ff3385", "#ff99cc", "#ff4da6",
    "#ffb3d9", "#ff6699", "#ff0066", "#ff5c8a"
]

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
# PAGE 0 – KPI + DISTRIBUSI
# ==========================================================
if st.session_state.page == 0:

    total_nilai = indikator.sum(axis=1)

    jumlah_siswa = len(df)
    rata_kelas = total_nilai.mean()
    skor_tertinggi = total_nilai.max()
    skor_terendah = total_nilai.min()

    st.markdown("""
    <style>
    .kpi-container {display:flex; gap:20px; margin-bottom:20px;}
    .kpi-box {
        background-color:#ffffff;
        padding:25px;
        border-radius:12px;
        text-align:center;
        width:100%;
        box-shadow:0px 2px 8px rgba(0,0,0,0.05);
    }
    .kpi-title {font-size:16px; color:#555;}
    .kpi-value {font-size:42px; font-weight:bold;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-box">
            <div class="kpi-title">👥 Total Partisipan</div>
            <div class="kpi-value">{jumlah_siswa}</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-title">📊 Rata-rata Kelas</div>
            <div class="kpi-value">{rata_kelas:.1f}</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-title">🏆 Skor Tertinggi</div>
            <div class="kpi-value">{skor_tertinggi:.0f}</div>
        </div>
        <div class="kpi-box">
            <div class="kpi-title">⚠ Skor Terendah</div>
            <div class="kpi-value">{skor_terendah:.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Distribusi Total Nilai (50 Siswa)")

    fig, ax = plt.subplots(figsize=(5,3))
    ax.hist(total_nilai, bins=len(total_nilai), edgecolor='black')
    st.pyplot(fig)

    st.button("Next ➝", on_click=next_page)

# ==========================================================
# PAGE 1 – RATA-RATA PER SOAL
# ==========================================================
elif st.session_state.page == 1:

    st.header("Rata-rata Nilai per Soal")

    mean_scores = indikator.mean()

    fig1, ax1 = plt.subplots(figsize=(8,3))
    ax1.bar(mean_scores.index, mean_scores.values,
            color=pink_colors[:len(mean_scores)])
    ax1.tick_params(axis='x', rotation=90)
    st.pyplot(fig1)

    soal_detail = st.selectbox("Pilih Soal:", indikator.columns)
    warna = pink_colors[list(indikator.columns).index(soal_detail)]

    fig2, ax2 = plt.subplots(figsize=(5,3))
    ax2.bar(soal_detail, mean_scores[soal_detail], color=warna)
    st.pyplot(fig2)

    col1, col2 = st.columns(2)
    col1.button("⬅ Previous", on_click=prev_page)
    col2.button("Next ➝", on_click=next_page)

# ==========================================================
# PAGE 2 – KORELASI
# ==========================================================
elif st.session_state.page == 2:

    st.header("Korelasi Antar Soal")

    corr = indikator.corr()

    fig, ax = plt.subplots(figsize=(6,5))
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
# PAGE 3 – REGRESI LENGKAP
# ==========================================================
elif st.session_state.page == 3:

    st.header("Analisis Regresi Interaktif")

    target_soal = st.selectbox(
        "Pilih Soal Target (Variabel Dependen):",
        indikator.columns
    )

    X = sm.add_constant(indikator.drop(columns=[target_soal]))
    y = indikator[target_soal]

    model = sm.OLS(y, X, missing="drop").fit()
    coef = model.params.drop("const")

    warna_bar = [
        pink_colors[list(indikator.columns).index(col)]
        for col in coef.index
    ]

    fig1, ax1 = plt.subplots(figsize=(9,3))
    ax1.bar(coef.index, coef.values, color=warna_bar)
    ax1.axhline(0, linestyle="--")
    ax1.tick_params(axis='x', rotation=90)
    st.pyplot(fig1)

    soal_detail = st.selectbox("Pilih Soal Prediktor:", coef.index)
    warna_detail = pink_colors[
        list(indikator.columns).index(soal_detail)
    ]

    fig2, ax2 = plt.subplots(figsize=(5,3))
    ax2.bar(soal_detail, coef[soal_detail], color=warna_detail)
    ax2.axhline(0, linestyle="--")
    st.pyplot(fig2)

    st.success(
        f"Soal paling berpengaruh terhadap {target_soal}: "
        f"{coef.abs().idxmax()}"
    )

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
