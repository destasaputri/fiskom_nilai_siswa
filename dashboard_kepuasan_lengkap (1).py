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
# CUSTOM CSS GIRLY THEME
# ==========================================================
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #ffe6f2, #fff0f5);
}
h1, h2, h3 {
    color: #d63384;
}
div[data-testid="metric-container"] {
    background-color: #fff0f6;
    border: 2px solid #ff99cc;
    padding: 15px;
    border-radius: 15px;
}
.sidebar .sidebar-content {
    background-color: #ffe6f2;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================
st.title("🌸📊 Dashboard Analisis Hasil Siswa 🌸")
st.markdown("### 💖 Analisis Performa Akademik Berbasis Data 💖")
st.markdown("**Desta Saputri**  \nNIM: 06111282429040")

st.divider()

# ==========================================================
# SIDEBAR INTERAKTIF
# ==========================================================
st.sidebar.title("🎀 Pengaturan Analisis")

jumlah_cluster = st.sidebar.slider("Jumlah Cluster", 2, 5, 3)

target_soal = st.sidebar.selectbox(
    "Pilih Soal untuk Analisis Regresi",
    options=list(range(1, 21)),
    index=19
)

st.sidebar.markdown("🌷 Dashboard dibuat dengan Streamlit")
st.sidebar.markdown("💗 Tema Girly Pink Edition")

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_excel("data_simulasi_50_siswa_20_soal.xlsx")
indikator = df.apply(pd.to_numeric, errors="coerce")

# ==========================================================
# KPI PERFORMA
# ==========================================================
mean_scores = indikator.mean()
nilai_rata2_kelas = indikator.mean(axis=1).mean()

def kategori_nilai(x):
    if x >= 85: return "Sangat Baik 💎"
    elif x >= 75: return "Baik 🌷"
    elif x >= 65: return "Cukup 🌸"
    else: return "Perlu Bimbingan 💔"

col1, col2, col3 = st.columns(3)
col1.metric("📈 Rata-rata Kelas", f"{nilai_rata2_kelas:.2f}")
col2.metric("🏷️ Kategori", kategori_nilai(nilai_rata2_kelas))
col3.metric("👩‍🎓 Jumlah Siswa", len(df))

st.divider()

# ==========================================================
# ANALISIS PER SOAL
# ==========================================================
st.header("🌺 Rata-rata Nilai per Soal")

fig_soal, ax_soal = plt.subplots(figsize=(9,4))
ax_soal.bar(mean_scores.index, mean_scores.values)
ax_soal.set_ylabel("Rata-rata Nilai")
ax_soal.set_title("Rata-rata Nilai Tiap Soal")
ax_soal.tick_params(axis='x', rotation=90)

st.pyplot(fig_soal)

soal_terendah = mean_scores.idxmin()
st.info(f"💡 Soal paling menantang: **{soal_terendah}**")

st.divider()

# ==========================================================
# ANALISIS KORELASI
# ==========================================================
st.header("🌸 Korelasi Antar Soal")

corr = indikator.corr()

fig_corr, ax_corr = plt.subplots(figsize=(7,6))
im = ax_corr.imshow(corr, cmap="RdPu", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax_corr)

ax_corr.set_xticks(range(len(corr.columns)))
ax_corr.set_yticks(range(len(corr.columns)))
ax_corr.set_xticklabels(corr.columns, rotation=90)
ax_corr.set_yticklabels(corr.columns)

st.pyplot(fig_corr)

st.divider()

# ==========================================================
# ANALISIS REGRESI (INTERAKTIF)
# ==========================================================
st.header("🌷 Analisis Regresi Interaktif")

target_index = target_soal - 1
X = sm.add_constant(indikator.drop(indikator.columns[target_index], axis=1))
y = indikator.iloc[:, target_index]

model = sm.OLS(y, X, missing="drop").fit()
coef = model.params[1:]
r2 = model.rsquared

fig_reg, ax_reg = plt.subplots(figsize=(9,4))
ax_reg.bar(coef.index, coef.values)
ax_reg.axhline(0, linestyle="--")
ax_reg.set_title(f"Koefisien Regresi untuk Soal {target_soal}")

st.pyplot(fig_reg)

st.info(f"📊 Nilai R²: **{r2:.2f}**")
st.success(f"🌟 Soal paling berpengaruh: **{coef.abs().idxmax()}**")

st.divider()

# ==========================================================
# SEGMENTASI INTERAKTIF
# ==========================================================
st.header("🎀 Segmentasi Performa Siswa")

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

fig_rad = plt.figure(figsize=(6,6))
ax_rad = plt.subplot(polar=True)

for i, row in cluster_mean.iterrows():
    values = row[labels].tolist()
    values += values[:1]
    ax_rad.plot(angles, values, label=f"Cluster {i}")
    ax_rad.fill(angles, values, alpha=0.2)

ax_rad.set_thetagrids(np.degrees(angles[:-1]), labels)
ax_rad.set_ylim(0, indikator.max().max())
ax_rad.set_title("Radar Segmentasi Siswa 🌸")
ax_rad.legend(loc="upper right")

st.pyplot(fig_rad)

st.success("🌷 Dashboard Interaktif Girly Siap Digunakan 💗")
