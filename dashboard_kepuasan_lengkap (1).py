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
st.title("📊 Dashboard Analisis Hasil 50 Siswa (20 Soal)")
st.markdown("Analisis performa siswa berbasis data")

# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_excel("data_simulasi_50_siswa_20_soal.xlsx")

# Ambil semua kolom numerik (20 soal)
indikator = df.apply(pd.to_numeric, errors="coerce")

# ==========================================================
# KPI PERFORMA
# ==========================================================
mean_scores = indikator.mean()
nilai_rata2_kelas = indikator.mean(axis=1).mean()

def kategori_nilai(x):
    if x >= 85: return "Sangat Baik"
    elif x >= 75: return "Baik"
    elif x >= 65: return "Cukup"
    else: return "Kurang"

col1, col2, col3 = st.columns(3)
col1.metric("📈 Rata-rata Kelas", f"{nilai_rata2_kelas:.2f}")
col2.metric("🏷️ Kategori Kelas", kategori_nilai(nilai_rata2_kelas))
col3.metric("👥 Jumlah Siswa", len(df))

st.divider()

# ==========================================================
# 3️⃣ ANALISIS PER SOAL
# ==========================================================
st.header("📊 Rata-rata Nilai per Soal")

fig_soal, ax_soal = plt.subplots(figsize=(8,4))
ax_soal.bar(mean_scores.index, mean_scores.values)
ax_soal.set_ylabel("Rata-rata Nilai")
ax_soal.set_title("Rata-rata Nilai Tiap Soal")
ax_soal.tick_params(axis='x', rotation=90)

st.pyplot(fig_soal)

soal_terendah = mean_scores.idxmin()
st.info(f"📌 Soal paling sulit: **{soal_terendah}**")

st.divider()

# ==========================================================
# 4️⃣ ANALISIS KORELASI
# ==========================================================
st.header("🔎 Korelasi Antar Soal")

corr = indikator.corr()

fig_corr, ax_corr = plt.subplots(figsize=(7,6))
im = ax_corr.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax_corr)

ax_corr.set_xticks(range(len(corr.columns)))
ax_corr.set_yticks(range(len(corr.columns)))
ax_corr.set_xticklabels(corr.columns, rotation=90)
ax_corr.set_yticklabels(corr.columns)

st.pyplot(fig_corr)

st.divider()

# ==========================================================
# 5️⃣ ANALISIS REGRESI
# ==========================================================
st.header("📈 Analisis Regresi (Prediksi Soal 20)")

X = sm.add_constant(indikator.iloc[:, 0:19])
y = indikator.iloc[:, 19]

model = sm.OLS(y, X, missing="drop").fit()

coef = model.params[1:]
r2 = model.rsquared

fig_reg, ax_reg = plt.subplots(figsize=(8,4))
ax_reg.bar(coef.index, coef.values)
ax_reg.axhline(0, linestyle="--")
ax_reg.set_title("Koefisien Regresi")

st.pyplot(fig_reg)
st.info(f"📊 Nilai R²: **{r2:.2f}**")
st.success(f"🔑 Soal paling berpengaruh terhadap Soal 20: **{coef.abs().idxmax()}**")

st.divider()

# ==========================================================
# 6️⃣ SEGMENTASI SISWA
# ==========================================================
st.header("🎯 Segmentasi Performa Siswa")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(indikator.fillna(indikator.mean()))

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_label = kmeans.fit_predict(X_scaled)

indikator_cluster = indikator.copy()
indikator_cluster["Cluster"] = cluster_label

cluster_mean = indikator_cluster.groupby("Cluster").mean()

cluster_mean = cluster_mean.sort_values(by=indikator.columns[-1], ascending=False)
cluster_mean["Kategori"] = ["Tinggi", "Sedang", "Rendah"]

labels = indikator.columns.tolist()
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig_rad = plt.figure(figsize=(6,6))
ax_rad = plt.subplot(polar=True)

for i, row in cluster_mean.iterrows():
    values = row[labels].tolist()
    values += values[:1]
    ax_rad.plot(angles, values, label=row["Kategori"])
    ax_rad.fill(angles, values, alpha=0.2)

ax_rad.set_thetagrids(np.degrees(angles[:-1]), labels)
ax_rad.set_ylim(0, indikator.max().max())
ax_rad.set_title("Radar Chart Segmentasi Siswa")
ax_rad.legend(loc="upper right")

st.pyplot(fig_rad)

st.success("✅ Dashboard siap digunakan untuk analisis akademik")
