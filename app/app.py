"""
app.py — Streamlit Application: Klasifikasi Tingkat Obesitas
UAS Pembelajaran Mesin — Genap 2025/2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, warnings
warnings.filterwarnings('ignore')

# ─── PATH CONFIG ────────────────────────────────────────────
BASE        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE, 'data', 'ObesityDataSet.csv')
MODEL_PATH  = os.path.join(BASE, 'models')
OUTPUT_PATH = os.path.join(BASE, 'outputs')

# ─── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Obesitas",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 12px; text-align: center;
        color: white; margin-bottom: 24px;
    }
    .metric-card {
        background: #f0f2f6; border-radius: 10px;
        padding: 16px; text-align: center; border-left: 4px solid #667eea;
    }
    .best-model-badge {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white; padding: 8px 16px; border-radius: 20px;
        font-weight: bold; display: inline-block;
    }
    .insight-box {
        background: #e8f4fd; border-left: 4px solid #2196F3;
        padding: 12px; border-radius: 6px; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── LOAD RESOURCES ─────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    resources = {}
    try:
        resources['scaler']       = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
        resources['le_target']    = joblib.load(os.path.join(MODEL_PATH, 'label_encoder.pkl'))
        resources['le_dict']      = joblib.load(os.path.join(MODEL_PATH, 'feature_encoders.pkl'))
        resources['feature_cols'] = joblib.load(os.path.join(MODEL_PATH, 'feature_cols.pkl'))
        resources['knn']          = joblib.load(os.path.join(MODEL_PATH, 'knn_model.pkl'))
        resources['dt']           = joblib.load(os.path.join(MODEL_PATH, 'dt_model.pkl'))
        resources['rf']           = joblib.load(os.path.join(MODEL_PATH, 'rf_model.pkl'))
        resources['best_model']   = joblib.load(os.path.join(MODEL_PATH, 'best_model.pkl'))
        resources['best_name']    = joblib.load(os.path.join(MODEL_PATH, 'best_model_name.pkl'))
        resources['best_scaled']  = joblib.load(os.path.join(MODEL_PATH, 'best_model_uses_scaled.pkl'))
        resources['results_df']   = joblib.load(os.path.join(MODEL_PATH, 'results_df.pkl'))
    except FileNotFoundError as e:
        st.error(f"❌ File model tidak ditemukan: {e}\n\nJalankan `notebook/run_pipeline.py` terlebih dahulu.")
        st.stop()
    return resources

df = load_data()
res = load_models()
df['BMI'] = df['Weight'] / (df['Height'] ** 2)

ORDER_CAT = ['Insufficient_Weight','Normal_Weight','Overweight_Level_I',
             'Overweight_Level_II','Obesity_Type_I','Obesity_Type_II','Obesity_Type_III']
PALETTE   = ['#2196F3','#4CAF50','#FFC107','#FF9800','#FF5722','#E91E63','#9C27B0']


# ─── SIDEBAR ────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/health-data.png", width=80)
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pilih Halaman:", [
    "Beranda",
    "Dashboard EDA",
    "Demo Prediksi",
    "Evaluasi Model",
    "Interpretasi Hasil",
    "Dokumentasi"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** {len(df)} sampel · {len(df.columns)} fitur")
st.sidebar.markdown(f"**Model Terbaik:** `{res['best_name']}`")

# ════════════════════════════════════════════════════════════
# HALAMAN 1: BERANDA
# ════════════════════════════════════════════════════════════
if page == "Beranda":
    st.markdown("""
    <div class='main-header'>
        <h1>Sistem Klasifikasi Tingkat Obesitas</h1>
        <p>Menggunakan Machine Learning untuk Prediksi Risiko Obesitas</p>
        <p><em>UAS Pembelajaran Mesin — Genap 2025/2026</em></p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sampel", f"{len(df):,}")
    with col2:
        st.metric("Jumlah Fitur", "16 fitur")
    with col3:
        st.metric("Kelas Target", "7 kategori")
    with col4:
        best_acc = res['results_df']['Accuracy'].max()
        st.metric("Best Accuracy", f"{best_acc:.2%}")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Tentang Proyek")
        st.markdown("""
        Proyek ini mengklasifikasikan **tingkat obesitas** seseorang berdasarkan:
        - Kebiasaan makan (FAVC, FCVC, CAEC, CALC)
        - Aktivitas fisik (FAF, TUE)
        - Konsumsi air (CH2O)
        - Data fisik (Tinggi, Berat, BMI)
        - Riwayat keluarga
        - Mode transportasi

        **Tujuan:** Membantu deteksi dini risiko obesitas sebagai langkah preventif kesehatan.
        """)
    with col_b:
        st.subheader("7 Kategori Obesitas")
        for i, cat in enumerate(ORDER_CAT):
            color = PALETTE[i]
            st.markdown(f"<span style='color:{color}; font-size:16px'>●</span> **{cat.replace('_', ' ')}**", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Alur Kerja Proyek")
    steps = ["1. Data Acquisition", "2. EDA & Preprocessing", "3. Model Training", "4. Evaluasi & Tuning", "5. Deployment"]
    cols = st.columns(len(steps))
    for col, step in zip(cols, steps):
        col.info(step)


# ════════════════════════════════════════════════════════════
# HALAMAN 2: DASHBOARD EDA
# ════════════════════════════════════════════════════════════
elif page == "Dashboard EDA":
    st.title("Dashboard Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Overview Data", "Distribusi", "Hubungan Antar Fitur"])

    with tab1:
        st.subheader("Preview Dataset")
        st.dataframe(df.head(10), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Statistik Deskriptif")
            st.dataframe(df.describe().round(3), use_container_width=True)
        with col2:
            st.subheader("Info Kualitas Data")
            quality = pd.DataFrame({
                'Kolom': df.columns,
                'Tipe Data': df.dtypes.values.astype(str),
                'Missing (%)': (df.isnull().sum().values / len(df) * 100).round(2),
                'Unik': df.nunique().values
            })
            st.dataframe(quality, use_container_width=True)
            st.success("Tidak ada missing values | Tidak ada duplikat")

    with tab2:
        st.subheader("Distribusi Kelas Target")
        fig, ax = plt.subplots(figsize=(10, 4))
        vc = df['NObeyesdad'].value_counts().reindex(ORDER_CAT)
        bars = ax.bar(vc.index, vc.values, color=PALETTE)
        ax.set_title('Distribusi Kelas NObeyesdad', fontweight='bold')
        ax.set_xlabel('Kategori Obesitas')
        ax.set_ylabel('Jumlah Sampel')
        plt.xticks(rotation=30, ha='right')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{int(bar.get_height())}', ha='center', va='bottom')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.info("Dataset seimbang sempurna — setiap kelas memiliki tepat 72 sampel.")

        st.subheader("Distribusi Fitur Numerik per Kategori")
        sel_feat = st.selectbox("Pilih Fitur:", ['Age','Height','Weight','BMI','FAF','CH2O','FCVC','TUE'])
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=df, x='NObeyesdad', y=sel_feat, order=ORDER_CAT, palette=PALETTE, ax=ax)
        ax.set_title(f'Distribusi {sel_feat} per Kategori Obesitas', fontweight='bold')
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.subheader("Heatmap Korelasi Fitur Numerik")
        num_cols = ['Age','Height','Weight','FCVC','NCP','CH2O','FAF','TUE','BMI']
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(9, 7))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', mask=mask, ax=ax, square=True)
        ax.set_title('Heatmap Korelasi', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Scatter Plot Interaktif")
        col1, col2 = st.columns(2)
        x_feat = col1.selectbox("Sumbu X:", num_cols, index=0)
        y_feat = col2.selectbox("Sumbu Y:", num_cols, index=8)
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, cat in enumerate(ORDER_CAT):
            sub = df[df['NObeyesdad'] == cat]
            ax.scatter(sub[x_feat], sub[y_feat], c=PALETTE[i], label=cat, alpha=0.7, s=40)
        ax.set_xlabel(x_feat)
        ax.set_ylabel(y_feat)
        ax.set_title(f'{x_feat} vs {y_feat}', fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ════════════════════════════════════════════════════════════
# HALAMAN 3: DEMO PREDIKSI
# ════════════════════════════════════════════════════════════
elif page == "Demo Prediksi":
    st.title("Demo Prediksi Tingkat Obesitas")
    st.info(f"**Model aktif:** {res['best_name']}")

    st.subheader("Masukkan Data Pasien")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Data Fisik**")
        gender    = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        age       = st.slider("Usia (tahun)", 14, 80, 25)
        height    = st.slider("Tinggi Badan (m)", 1.45, 2.00, 1.70, step=0.01)
        weight    = st.slider("Berat Badan (kg)", 30.0, 170.0, 70.0, step=0.5)

    with col2:
        st.markdown("**Kebiasaan Makan**")
        family_hist = st.selectbox("Riwayat Keluarga Overweight", ["yes", "no"])
        favc  = st.selectbox("Sering Makan Tinggi Kalori (FAVC)", ["yes", "no"])
        fcvc  = st.slider("Frekuensi Sayur (FCVC, 1-3)", 1.0, 3.0, 2.0, step=0.1)
        ncp   = st.slider("Jumlah Makan Utama (NCP)", 1.0, 4.0, 3.0, step=0.5)
        caec  = st.selectbox("Makan di Antara Waktu Makan (CAEC)",
                              ["no", "Sometimes", "Frequently", "Always"])
        calc  = st.selectbox("Konsumsi Alkohol (CALC)",
                              ["no", "Sometimes", "Frequently", "Always"])

    with col3:
        st.markdown("**Gaya Hidup**")
        smoke  = st.selectbox("Merokok", ["no", "yes"])
        ch2o   = st.slider("Konsumsi Air Harian (liter)", 1.0, 3.0, 2.0, step=0.1)
        scc    = st.selectbox("Monitor Kalori (SCC)", ["no", "yes"])
        faf    = st.slider("Frekuensi Aktivitas Fisik (FAF, 0-3)", 0.0, 3.0, 1.0, step=0.1)
        tue    = st.slider("Waktu Layar/Teknologi (TUE, jam)", 0.0, 2.0, 1.0, step=0.1)
        mtrans = st.selectbox("Transportasi Utama (MTRANS)",
                               ["Public_Transportation", "Automobile", "Motorbike",
                                "Bike", "Walking"])

    bmi = weight / (height ** 2)
    st.markdown(f"**BMI Terhitung:** `{bmi:.2f}` kg/m²")

    if st.button("Prediksi Sekarang", type="primary", use_container_width=True):
        # Build input row
        le_dict     = res['le_dict']
        feature_cols = res['feature_cols']
        scaler      = res['scaler']
        le_target   = res['le_target']

        input_raw = {
            'Gender': le_dict['Gender'].transform([gender])[0],
            'Age': age, 'Height': height, 'Weight': weight,
            'family_history_with_overweight': le_dict['family_history_with_overweight'].transform([family_hist])[0],
            'FAVC': le_dict['FAVC'].transform([favc])[0],
            'FCVC': fcvc, 'NCP': ncp,
            'CAEC': le_dict['CAEC'].transform([caec])[0],
            'SMOKE': le_dict['SMOKE'].transform([smoke])[0],
            'CH2O': ch2o,
            'SCC': le_dict['SCC'].transform([scc])[0],
            'FAF': faf, 'TUE': tue,
            'CALC': le_dict['CALC'].transform([calc])[0],
            'MTRANS': le_dict['MTRANS'].transform([mtrans])[0],
            'BMI': bmi
        }
        input_df = pd.DataFrame([input_raw])[feature_cols]

        best_model = res['best_model']
        best_scaled = res['best_scaled']

        if best_scaled:
            input_arr = scaler.transform(input_df)
        else:
            input_arr = input_df.values

        pred_encoded = best_model.predict(input_arr)[0]
        pred_label   = le_target.inverse_transform([pred_encoded])[0]

        # Coba dapatkan probabilitas
        try:
            proba = best_model.predict_proba(input_arr)[0]
            has_proba = True
        except Exception:
            has_proba = False

        # Warna sesuai kategori
        color_map = dict(zip(ORDER_CAT, PALETTE))
        pred_color = color_map.get(pred_label, '#333')

        st.markdown("---")
        st.markdown(f"""
        <div style='background:{pred_color}22; border:2px solid {pred_color};
                    border-radius:12px; padding:20px; text-align:center; margin:16px 0;'>
            <h2 style='color:{pred_color}; margin:0;'>Hasil Prediksi</h2>
            <h1 style='color:{pred_color};'>{pred_label.replace('_', ' ')}</h1>
            <p style='color:#555;'>BMI: <b>{bmi:.2f}</b> kg/m² | Model: <b>{res['best_name']}</b></p>
        </div>
        """, unsafe_allow_html=True)

        if has_proba:
            st.subheader("Probabilitas Tiap Kelas")
            proba_df = pd.DataFrame({'Kategori': le_target.classes_, 'Probabilitas': proba})
            proba_df = proba_df.sort_values('Probabilitas', ascending=True)
            fig, ax = plt.subplots(figsize=(8, 4))
            colors_p = [color_map.get(c, '#ccc') for c in proba_df['Kategori']]
            ax.barh(proba_df['Kategori'], proba_df['Probabilitas'], color=colors_p)
            ax.set_xlabel('Probabilitas')
            ax.set_title('Distribusi Probabilitas Prediksi', fontweight='bold')
            ax.axvline(0.5, color='red', linestyle='--', alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Rekomendasi
        st.subheader("Rekomendasi Kesehatan")
        recs = {
            'Insufficient_Weight': "Berat badan kurang. Tingkatkan asupan kalori bergizi, konsultasikan dengan ahli gizi.",
            'Normal_Weight': "Berat badan ideal. Pertahankan pola makan sehat dan olahraga rutin.",
            'Overweight_Level_I': "Kelebihan berat badan ringan. Kurangi kalori, perbanyak aktivitas fisik.",
            'Overweight_Level_II': "Kelebihan berat badan sedang. Konsultasi dokter dan program diet terstruktur.",
            'Obesity_Type_I': "Obesitas tingkat I. Segera konsultasi dokter. Olahraga rutin dan diet ketat.",
            'Obesity_Type_II': "Obesitas tingkat II. Penanganan medis diperlukan. Risiko penyakit kardiovaskular tinggi.",
            'Obesity_Type_III': "Obesitas tingkat III (morbid). Segera hubungi dokter spesialis untuk penanganan intensif."
        }
        st.info(recs.get(pred_label, "Konsultasikan dengan dokter."))


# ════════════════════════════════════════════════════════════
# HALAMAN 4: EVALUASI MODEL
# ════════════════════════════════════════════════════════════
elif page == "Evaluasi Model":
    st.title("Evaluasi & Perbandingan Model")

    results_df = res['results_df']

    st.subheader("Tabel Perbandingan Performa (Test Set)")
    display_df = results_df.set_index('Model').round(4)
    st.dataframe(display_df.style.highlight_max(axis=0, color='#c6efce'), use_container_width=True)

    best_row = results_df.loc[results_df['F1-Score'].idxmax()]
    st.markdown(f"""
    <div style='text-align:center; padding:10px;'>
        <span class='best-model-badge'>Model Terbaik: {best_row['Model']}
        — F1-Score: {best_row['F1-Score']:.4f}
        — Accuracy: {best_row['Accuracy']:.4f}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Visualisasi Perbandingan Metrik")
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.25
    colors = ['royalblue', 'forestgreen', 'darkorange']
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (_, row) in enumerate(results_df.iterrows()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, [row[m] for m in metrics], width,
                      label=row['Model'], color=colors[i], alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_title('Perbandingan Performa Model — Test Set', fontsize=13, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Confusion Matrix")
    model_choice = st.selectbox("Pilih Model:", results_df['Model'].tolist())

    model_map = {}
    for _, row in results_df.iterrows():
        name = row['Model']
        if 'KNN' in name:
            model_map[name] = ('knn', True)
        elif 'Decision Tree' in name:
            model_map[name] = ('dt', False)
        else:
            model_map[name] = ('rf', False)

    key, use_scaled = model_map[model_choice]
    mdl = res[key]
    scaler = res['scaler']

    X_test_arr = np.load(os.path.join(MODEL_PATH, 'X_test.npy'))
    X_test_raw = np.load(os.path.join(MODEL_PATH, 'X_test_raw.npy'))
    y_test     = np.load(os.path.join(MODEL_PATH, 'y_test.npy'))

    preds = mdl.predict(X_test_arr if use_scaled else X_test_raw)
    cm = __import__('sklearn.metrics', fromlist=['confusion_matrix']).confusion_matrix(y_test, preds)

    short_names = ['Insuff.', 'Normal', 'OW_I', 'OW_II', 'Ob_I', 'Ob_II', 'Ob_III']
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=short_names, yticklabels=short_names)
    ax.set_title(f'Confusion Matrix — {model_choice}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Feature Importance (Random Forest)")
    feature_cols = res['feature_cols']
    rf_model = res['rf']
    importances = rf_model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors_fi = ['#d73027' if v > feat_imp.median() else '#4575b4' for v in feat_imp.values]
    ax.barh(feat_imp.index, feat_imp.values, color=colors_fi)
    ax.set_title('Feature Importance — Random Forest', fontsize=12, fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════
# HALAMAN 5: INTERPRETASI HASIL
# ════════════════════════════════════════════════════════════
elif page == "Interpretasi Hasil":
    st.title("Interpretasi Hasil & Insights")

    st.subheader("5 Insights Utama dari Data")
    insights = [
        ("Insight 1: BMI sebagai Indikator Utama",
         "BMI meningkat secara monoton dari Insufficient Weight (~16) hingga Obesity Type III (~45+). "
         "BMI adalah fitur paling informatif untuk klasifikasi obesitas.", "04_insight1_bmi_per_kategori.png"),
        ("Insight 2: Korelasi Weight-BMI Sangat Tinggi",
         "Weight dan BMI berkorelasi hampir sempurna (r ≈ 0.97). "
         "Berat badan adalah prediktor langsung indeks massa tubuh.", "05_insight2_heatmap_korelasi.png"),
        ("Insight 3: Riwayat Keluarga Berpengaruh Signifikan",
         "Lebih dari 80% penderita Obesity Type I-III memiliki riwayat keluarga overweight, "
         "mengindikasikan faktor genetik memainkan peran penting.", "06_insight3_riwayat_keluarga.png"),
        ("Insight 4: Aktivitas Fisik Rendah, Risiko Obesitas Tinggi",
         "Penderita Obesity Type III memiliki frekuensi aktivitas fisik (FAF) yang jauh lebih rendah. "
         "Sedentary lifestyle merupakan faktor risiko utama.", "07_insight4_aktivitas_fisik.png"),
        ("Insight 5: Obesitas Berat Dominan di Usia Produktif",
         "Obesity Type III lebih banyak ditemukan pada rentang usia 20-40 tahun, "
         "usia produktif yang justru berisiko tinggi karena gaya hidup modern.", "08_insight5_age_vs_bmi.png"),
    ]

    for i, (title, desc, fname) in enumerate(insights):
        with st.expander(f"Insight {i+1}: {title.split(': ', 1)[-1]}", expanded=(i == 0)):
            fpath = os.path.join(OUTPUT_PATH, fname)
            col1, col2 = st.columns([1.5, 1])
            with col1:
                if os.path.exists(fpath):
                    st.image(fpath, use_container_width=True)
                else:
                    st.warning("Gambar belum tersedia. Jalankan pipeline terlebih dahulu.")
            with col2:
                st.markdown(f"<div class='insight-box'>{desc}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Mengapa Decision Tree / Random Forest Terpilih?")
    st.markdown("""
    | Aspek | KNN | Decision Tree | Random Forest |
    |-------|-----|---------------|---------------|
    | Accuracy | Rendah (~35%) | Tinggi (~77%) | Tinggi (~76%) |
    | Interpretabilitas | Rendah | **Tinggi** | Sedang |
    | Kecepatan Prediksi | Lambat | **Cepat** | Sedang |
    | Overfitting | Risiko tinggi | Risiko sedang | **Rendah** |
    | Feature Importance | Tidak tersedia | Tersedia | **Tersedia** |

    **Rekomendasi:**
    - Gunakan model ini sebagai skrining awal di fasilitas kesehatan
    - Integrasikan dengan aplikasi mobile untuk self-assessment pasien
    - Fitur riwayat keluarga dan BMI harus selalu disertakan dalam pengumpulan data kesehatan
    """)

# ════════════════════════════════════════════════════════════
# HALAMAN 6: DOKUMENTASI
# ════════════════════════════════════════════════════════════
elif page == "Dokumentasi":
    st.title("Dokumentasi Proyek")

    tab1, tab2, tab3 = st.tabs(["Dataset", "Metodologi", "Cara Penggunaan"])

    with tab1:
        st.subheader("Dataset: ObesityDataSet.csv")
        st.markdown("""
        **Sumber:** Kaggle / UCI Machine Learning Repository
        ([Estimation of Obesity Levels Based On Eating Habits and Physical Condition](https://www.kaggle.com/datasets/fatemehmehrparvar/obesity-levels))

        **Ukuran:** 504 baris × 17 kolom (16 fitur + 1 target)

        | Fitur | Tipe | Deskripsi |
        |-------|------|-----------|
        | Gender | Kategorikal | Jenis kelamin |
        | Age | Numerik | Usia dalam tahun |
        | Height | Numerik | Tinggi badan (meter) |
        | Weight | Numerik | Berat badan (kg) |
        | family_history_with_overweight | Biner | Riwayat keluarga overweight |
        | FAVC | Biner | Kebiasaan makan tinggi kalori |
        | FCVC | Numerik | Frekuensi konsumsi sayur (1-3) |
        | NCP | Numerik | Jumlah makan utama per hari |
        | CAEC | Ordinal | Makan di antara waktu makan |
        | SMOKE | Biner | Status merokok |
        | CH2O | Numerik | Konsumsi air harian (liter) |
        | SCC | Biner | Monitor asupan kalori |
        | FAF | Numerik | Frekuensi aktivitas fisik (0-3) |
        | TUE | Numerik | Waktu penggunaan teknologi (jam) |
        | CALC | Ordinal | Frekuensi konsumsi alkohol |
        | MTRANS | Kategorikal | Mode transportasi |
        | **NObeyesdad** | **Target** | **Kategori tingkat obesitas** |
        """)

    with tab2:
        st.subheader("Alur Metodologi")
        st.markdown("""
        ```
        Data Acquisition
            └── ObesityDataSet.csv (Kaggle)

        Exploratory Data Analysis
            ├── Analisis kualitas data (missing values, outlier, duplikat)
            ├── Analisis univariat & multivariat
            └── 5 visualisasi insight utama

        Preprocessing
            ├── Feature Engineering: BMI = Weight / Height²
            ├── Label Encoding (fitur kategorikal)
            ├── StandardScaler (untuk KNN)
            └── Stratified Train-Val-Test Split (60-20-20)

        Modeling & Tuning
            ├── KNN (k=1..20)
            ├── Decision Tree (max_depth=2..15)
            └── Random Forest (n_estimators=[10,30,50,100,150,200])

        Evaluasi
            ├── Accuracy, Precision, Recall, F1-Score
            ├── Confusion Matrix
            └── Feature Importance

        Deployment
            └── Streamlit Web Application
        ```
        """)

    with tab3:
        st.subheader("Cara Menjalankan Aplikasi")
        st.code("""
# 1. Install dependensi
python -m pip install streamlit scikit-learn pandas numpy matplotlib seaborn joblib

# 2. Jalankan pipeline (training model), cukup sekali
python notebook/run_pipeline.py

# 3. Jalankan aplikasi Streamlit
python -m streamlit run app/app.py
        """, language="bash")

        st.subheader("Struktur Folder Proyek")
        st.code("""
UAS MESIN/
├── data/
│   └── ObesityDataSet.csv
├── notebook/
│   ├── 01_EDA_Preprocessing.ipynb
│   ├── 02_Modeling_Evaluation.ipynb
│   └── run_pipeline.py
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── ... (model files)
├── outputs/
│   └── ... (visualisasi PNG)
└── app/
    └── app.py
        """, language="")

# ─── FOOTER ─────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:13px;'>"
    "Sistem Klasifikasi Obesitas | UAS Pembelajaran Mesin Genap 2025/2026 | "
    "Universitas Dian Nuswantoro</div>",
    unsafe_allow_html=True
)
