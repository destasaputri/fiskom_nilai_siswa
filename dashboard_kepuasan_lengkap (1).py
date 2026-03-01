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
# 🌸 CUSTOM CSS GIRLY THEME
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
section[data-testid="stSidebar"] {
    background-color: #ffe6f2;
}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================
st.title("🌸📊 Dashboard Analisis Hasil Siswa 🌸")
st.markdown("### 💖 Analisis Performa Akademik Berbasis Data 💖")
st.markdown("**Nama:** Desta Saputri  \n**NIM:** 06111282429040")

st.divider()

# ==========================================================
# SIDEBAR INTERAKTIF
# ==========================================================
st.sidebar.title("🎀 Pengaturan Dashboard")

jumlah_cluster = st.sidebar.slider("Jumlah Cluster", 2, 5, 3)

target_soal = st.sidebar.selectbox(
    "Pilih Soal untuk Analisis Regresi",
    options=list(range(1, 21)),
    index=19
)

st.sidebar.markdown("🌷 Girly Pink Theme Activated")
st.sidebar.markdown("💗 Powered by Streamlit")

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
# 🌺 RATA-RATA PER SOAL
# ==========================================================
st.header("🌺 Rata-rata Nilai per Soal")

fig_soal, ax_soal = plt.subplots(figsize=(10,4))
bars = ax_soal.bar(mean_scores.index, mean_scores.values, color="#ff66b2")
ax_soal.set_ylabel("Rata-rata Nilai")
ax_soal.set_title("Distribusi Rata-rata Tiap Soal")
ax_soal.tick_params(axis='x', rotation=90)

st.pyplot(fig_soal)

soal_terendah = mean_scores.idxmin()
st.info(f"💡 Soal paling menantang adalah **{soal_terendah}**")

st.divider()

# ==========================================================
# 🌸 KORELASI
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
# 🌷 ANALISIS REGRESI INTERAKTIF ESTETIK
# ==========================================================
st.header("🌷 Analisis Regresi Interaktif")

st.markdown("""
Analisis ini menunjukkan **soal mana yang paling memengaruhi**
soal yang kamu pilih.  
Semakin tinggi batang grafik 💗 → semakin besar pengaruhnya.
""")

target_index = target_soal - 1

X = sm.add_constant(indikator.drop(indikator.columns[target_index], axis=1))
y = indikator.iloc[:, target_index]

model = sm.OLS(y, X, missing="drop").fit()
coef = model.params[1:]
r2 = model.rsquared

# Slider filter pengaruh
threshold = st.slider(
    "🌸 Tampilkan soal dengan pengaruh minimum:",
    0.0,
    float(abs(coef).max()),
    0.5
)

coef_filtered = coef[abs(coef) >= threshold]

fig_reg, ax_reg = plt.subplots(figsize=(10,5))

bars = ax_reg.bar(coef_filtered.index, coef_filtered.values)

for bar in bars:
    if bar.get_height() > 0:
        bar.set_color("#ff66b2")
    else:
        bar.set_color("#ffb3d9")

ax_reg.axhline(0, linestyle="--")
ax_reg.set_title(f"Pengaruh Soal terhadap Soal {target_soal} 💖")
ax_reg.set_ylabel("Besarnya Pengaruh")
ax_reg.tick_params(axis='x', rotation=90)

st.pyplot(fig_reg)

soal_dominan = coef.abs().idxmax()
nilai_dominan = coef.abs().max()

st.success(
    f"💎 Soal paling berpengaruh terhadap Soal {target_soal} adalah "
    f"**{soal_dominan}** dengan kekuatan {nilai_dominan:.2f}"
)

st.info(f"📊 Model menjelaskan sekitar **{r2*100:.1f}%** variasi nilai")

st.markdown("### 🌸 Ranking Pengaruh Soal")

ranking = coef.abs().sort_values(ascending=False)

for i, (nama, nilai) in enumerate(ranking.items(), start=1):
    st.progress(float(nilai / ranking.max()))
    st.write(f"{i}. {nama} → {nilai:.2f}")

st.divider()

# ==========================================================
# 🎀 SEGMENTASI SISWA
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

st.success("🌷 Dashboard Girly Interaktif Siap Digunakan 💗✨")
