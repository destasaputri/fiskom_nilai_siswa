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

/* Background */
html, body, .stApp { 
    background-color: #ffd9ec; 
}

/* Judul */
h1, h2, h3 { 
    color: #880e4f; 
}

/* Tombol */
.stButton>button {
    background-color: #d81b60;
    color: white;
    border-radius: 8px;
}
.stButton>button:hover {
    background-color: #ad1457;
}

/* KPI jadi pink */
div[data-testid="metric-container"] {
    background-color: #ffe6f2;
    border: 2px solid #f06292;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.05);
}

div[data-testid="metric-container"] label {
    color: #ad1457 !important;
    font-weight: 600;
}

div[data-testid="metric-container"] div {
    color: #880e4f !important;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

st.title("🌸 Dashboard Analisis Hasil Siswa")
st.divider()

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_excel("data_simulasi_50_siswa_20_soal.xlsx")
indikator = df.apply(pd.to_numeric, errors="coerce")

# PALETTE PINK
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

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total Siswa", len(df))
    col2.metric("📊 Rata-rata", f"{total_nilai.mean():.1f}")
    col3.metric("🏆 Tertinggi", f"{total_nilai.max():.0f}")
    col4.metric("⚠ Terendah", f"{total_nilai.min():.0f}")

    st.subheader("Distribusi Total Nilai (50 Bin)")

    fig, ax = plt.subplots(figsize=(6,3))
    ax.hist(total_nilai, bins=50, edgecolor='black')
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
# PAGE 3 – REGRESI TOTAL NILAI
# ==========================================================
elif st.session_state.page == 3:

    st.header("Analisis Regresi")
    st.subheader("Pengaruh Setiap Soal terhadap Total Nilai")

    total_nilai = indikator.sum(axis=1)

    X = sm.add_constant(indikator)
    y = total_nilai

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

    soal_detail = st.selectbox("Detail Soal:", coef.index)
    warna_detail = pink_colors[
        list(indikator.columns).index(soal_detail)
    ]

    fig2, ax2 = plt.subplots(figsize=(5,3))
    ax2.bar(soal_detail, coef[soal_detail], color=warna_detail)
    ax2.axhline(0, linestyle="--")
    st.pyplot(fig2)

    arah = "positif" if coef[soal_detail] > 0 else "negatif"

    st.info(
        f"Koefisien {soal_detail} = {coef[soal_detail]:.3f} ({arah})"
    )

    st.success(
        f"Soal paling berkontribusi: {coef.abs().idxmax()}"
    )

    st.write(f"R² Model: {model.rsquared:.3f}")

    col1, col2 = st.columns(2)
    col1.button("⬅ Previous", on_click=prev_page)
    col2.button("Next ➝", on_click=next_page)

# ==========================================================
# PAGE 4 – SEGMENTASI
# ==========================================================
elif st.session_state.page == 4:

    st.header("Segmentasi Performa Siswa")

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
