# 🏥 Klasifikasi Tingkat Obesitas — UAS Pembelajaran Mesin

**Ujian Akhir Semester Genap 2025/2026 | Universitas Dian Nuswantoro**  
Kelompok: A11.4407 | Mata Kuliah: Pembelajaran Mesin

---

## 📌 Problem Statement

Obesitas merupakan masalah kesehatan global yang meningkat pesat. Proyek ini membangun sistem **klasifikasi tingkat obesitas** menggunakan machine learning berdasarkan kebiasaan makan, aktivitas fisik, dan data fisik individu.

**Target:** Mengklasifikasikan seseorang ke dalam 7 kategori obesitas:
- Insufficient_Weight, Normal_Weight
- Overweight_Level_I, Overweight_Level_II
- Obesity_Type_I, Obesity_Type_II, Obesity_Type_III

---

## 📂 Struktur Proyek

```
UAS MESIN/
├── data/
│   └── ObesityDataSet.csv          # Dataset (504 baris, 17 kolom)
├── notebook/
│   ├── 01_EDA_Preprocessing.ipynb  # Soal 2: EDA & Preprocessing
│   ├── 02_Modeling_Evaluation.ipynb # Soal 3: Modeling & Evaluasi
│   └── run_pipeline.py             # Script untuk menjalankan pipeline
├── models/                         # File model tersimpan (.pkl)
├── outputs/                        # Visualisasi & hasil evaluasi
├── app/
│   └── app.py                      # Soal 4: Aplikasi Streamlit
└── README.md
```

---

## 🚀 Cara Menjalankan

> **Catatan:** Semua perintah di bawah menggunakan path **relatif** sehingga bisa langsung dijalankan di laptop/komputer siapapun tanpa perlu mengubah path. Cukup clone/copy folder ini ke mana saja, lalu ikuti langkah berikut.

### 1. Clone / Copy Proyek

Pastikan struktur folder proyek sudah seperti ini di perangkat kamu:
```
UAS MESIN/
├── app/
├── data/
├── models/
├── notebook/
├── outputs/
└── README.md
```

### 2. Buka Terminal di Root Folder Proyek

- **Windows:** Klik kanan di dalam folder `UAS MESIN` → *Open in Terminal* (atau *Open PowerShell window here*)
- **VS Code:** Buka folder `UAS MESIN`, lalu buka terminal terintegrasi (`Ctrl + `` `)

### 3. Install Dependensi

```bash
python -m pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib
```

### 4. Jalankan Pipeline (Training Model)

> Langkah ini hanya perlu dilakukan **sekali**. Hasilnya tersimpan di folder `models/` dan `outputs/`.

```bash
python notebook/run_pipeline.py
```

### 5. Jalankan Aplikasi Streamlit

```bash
python -m streamlit run app/app.py
```

Browser akan otomatis terbuka di `http://localhost:8501`. Jika tidak terbuka otomatis, copy URL tersebut ke browser secara manual.

---

## 🤖 Model yang Digunakan

| Model | Best Param | Test Accuracy | F1-Score |
|-------|-----------|---------------|----------|
| KNN | k=18 | ~35% | ~34% |
| **Decision Tree** | **depth=5** | **~77%** | **~77%** |
| Random Forest | n=200 | ~76% | ~76% |

**Model Terbaik: Decision Tree (depth=5)** — dipilih berdasarkan F1-Score tertinggi pada test set.

---

## 📊 Dataset

- **Sumber:** [Kaggle — Obesity Levels Dataset](https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels)
- **Ukuran:** 504 sampel × 17 kolom
- **Kelas:** 7 kategori (seimbang, 72 sampel/kelas)
- **Fitur:** data fisik, kebiasaan makan, aktivitas fisik, gaya hidup

---

## 📱 Fitur Aplikasi Streamlit

1. **Dashboard EDA** — visualisasi interaktif distribusi data
2. **Demo Prediksi** — input data pasien → prediksi kategori obesitas
3. **Evaluasi Model** — confusion matrix, metrik perbandingan
4. **Interpretasi Hasil** — insights bisnis & rekomendasi kesehatan
5. **Dokumentasi** — penjelasan dataset & metodologi

---

## 🛠️ Teknologi

- Python 3.x
- scikit-learn (ML models)
- Pandas, NumPy (data processing)
- Matplotlib, Seaborn (visualisasi)
- Streamlit (web application)
- Joblib (model serialization)
