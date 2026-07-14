"""
run_pipeline.py
Menjalankan seluruh pipeline: EDA, Preprocessing, Training, dan Evaluasi model.
Menghasilkan semua file yang dibutuhkan di folder outputs/ dan models/.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(BASE, 'data', 'ObesityDataSet.csv')
OUTPUT_PATH = os.path.join(BASE, 'outputs')
MODEL_PATH  = os.path.join(BASE, 'models')
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = (10, 6)

print("=" * 55)
print("  PIPELINE PEMBELAJARAN MESIN — KLASIFIKASI OBESITAS")
print("=" * 55)

# ============================================================
# BAGIAN 1: LOAD & EDA
# ============================================================
print("\n[1/5] Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")
print(f"  Kelas target: {df['NObeyesdad'].unique()}")

# -------- Visualisasi 1: Distribusi target --------
fig, ax = plt.subplots(figsize=(12, 5))
order = df['NObeyesdad'].value_counts().index
sns.countplot(data=df, x='NObeyesdad', order=order, ax=ax, palette='Set2')
ax.set_title('Distribusi Kelas Target (NObeyesdad)', fontsize=14, fontweight='bold')
ax.set_xlabel('Tingkat Obesitas')
ax.set_ylabel('Jumlah')
plt.xticks(rotation=30, ha='right')
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '01_distribusi_target.png'), dpi=150)
plt.close()

# -------- Visualisasi 2: Distribusi numerik --------
num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    axes[i].hist(df[col], bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    axes[i].set_title(col)
plt.suptitle('Distribusi Fitur Numerik', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '02_distribusi_numerik.png'), dpi=150)
plt.close()

# -------- BMI --------
df['BMI'] = df['Weight'] / (df['Height'] ** 2)
order_cat = ['Insufficient_Weight','Normal_Weight','Overweight_Level_I',
             'Overweight_Level_II','Obesity_Type_I','Obesity_Type_II','Obesity_Type_III']

# -------- Visualisasi 3 (Insight 1): BMI per kategori --------
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df, x='NObeyesdad', y='BMI', order=order_cat, ax=ax, palette='RdYlGn_r')
ax.set_title('Insight 1: Distribusi BMI per Kategori Obesitas', fontsize=13, fontweight='bold')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '04_insight1_bmi_per_kategori.png'), dpi=150)
plt.close()

# -------- Visualisasi 4 (Insight 2): Heatmap korelasi --------
corr = df[num_cols + ['BMI']].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', mask=mask, ax=ax,
            square=True, linewidths=0.5)
ax.set_title('Insight 2: Heatmap Korelasi Fitur Numerik', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '05_insight2_heatmap_korelasi.png'), dpi=150)
plt.close()

# -------- Visualisasi 5 (Insight 3): Riwayat keluarga --------
ct = pd.crosstab(df['NObeyesdad'], df['family_history_with_overweight'])
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(12, 6))
ct_pct.loc[order_cat].plot(kind='bar', ax=ax, color=['#FF6B6B','#4ECDC4'], width=0.7)
ax.set_title('Insight 3: Riwayat Keluarga Overweight per Kategori (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Kategori Obesitas')
ax.set_ylabel('Persentase (%)')
ax.legend(['Tidak', 'Ya'], title='Riwayat Keluarga')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '06_insight3_riwayat_keluarga.png'), dpi=150)
plt.close()

# -------- Visualisasi: Aktivitas fisik --------
fig, ax = plt.subplots(figsize=(12, 6))
sns.violinplot(data=df, x='NObeyesdad', y='FAF', order=order_cat, ax=ax,
               palette='muted', inner='box')
ax.set_title('Insight 4: Aktivitas Fisik per Kategori Obesitas', fontsize=13, fontweight='bold')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '07_insight4_aktivitas_fisik.png'), dpi=150)
plt.close()

# -------- Visualisasi: Age vs BMI scatter --------
palette_map = {
    'Insufficient_Weight': '#2196F3', 'Normal_Weight': '#4CAF50',
    'Overweight_Level_I': '#FFC107',  'Overweight_Level_II': '#FF9800',
    'Obesity_Type_I': '#FF5722',       'Obesity_Type_II': '#E91E63',
    'Obesity_Type_III': '#9C27B0'
}
fig, ax = plt.subplots(figsize=(12, 7))
for cat, color in palette_map.items():
    subset = df[df['NObeyesdad'] == cat]
    ax.scatter(subset['Age'], subset['BMI'], c=color, label=cat, alpha=0.7, s=50)
ax.set_title('Insight 5: Hubungan Usia dan BMI', fontsize=13, fontweight='bold')
ax.set_xlabel('Usia (tahun)')
ax.set_ylabel('BMI (kg/m²)')
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '08_insight5_age_vs_bmi.png'), dpi=150)
plt.close()

print("  ✅ Visualisasi EDA selesai")

# ============================================================
# BAGIAN 2: PREPROCESSING
# ============================================================
print("\n[2/5] Preprocessing...")
df_clean = df.copy()
df_clean['BMI'] = df_clean['Weight'] / (df_clean['Height'] ** 2)

cat_cols_to_encode = ['Gender', 'family_history_with_overweight', 'FAVC',
                       'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
le_dict = {}
for col in cat_cols_to_encode:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    le_dict[col] = le

le_target = LabelEncoder()
df_clean['NObeyesdad_encoded'] = le_target.fit_transform(df_clean['NObeyesdad'])

feature_cols = ['Gender','Age','Height','Weight','family_history_with_overweight',
                'FAVC','FCVC','NCP','CAEC','SMOKE','CH2O','SCC','FAF','TUE','CALC','MTRANS','BMI']

X = df_clean[feature_cols]
y = df_clean['NObeyesdad_encoded']

# 60-20-20 stratified split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test     = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# Simpan
np.save(os.path.join(MODEL_PATH, 'X_train.npy'), X_train_scaled)
np.save(os.path.join(MODEL_PATH, 'X_val.npy'),   X_val_scaled)
np.save(os.path.join(MODEL_PATH, 'X_test.npy'),  X_test_scaled)
np.save(os.path.join(MODEL_PATH, 'y_train.npy'), y_train.values)
np.save(os.path.join(MODEL_PATH, 'y_val.npy'),   y_val.values)
np.save(os.path.join(MODEL_PATH, 'y_test.npy'),  y_test.values)
np.save(os.path.join(MODEL_PATH, 'X_train_raw.npy'), X_train.values)
np.save(os.path.join(MODEL_PATH, 'X_val_raw.npy'),   X_val.values)
np.save(os.path.join(MODEL_PATH, 'X_test_raw.npy'),  X_test.values)

joblib.dump(scaler,      os.path.join(MODEL_PATH, 'scaler.pkl'))
joblib.dump(le_target,   os.path.join(MODEL_PATH, 'label_encoder.pkl'))
joblib.dump(le_dict,     os.path.join(MODEL_PATH, 'feature_encoders.pkl'))
joblib.dump(feature_cols,os.path.join(MODEL_PATH, 'feature_cols.pkl'))
print("  ✅ Preprocessing & data split selesai")

# ============================================================
# BAGIAN 3: TRAINING
# ============================================================
print("\n[3/5] Training models...")
class_names = le_target.classes_

# --- KNN ---
k_range = range(1, 21)
val_scores_knn = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    val_scores_knn.append(accuracy_score(y_val.values, knn.predict(X_val_scaled)))
best_k = list(k_range)[np.argmax(val_scores_knn)]
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train_scaled, y_train)
print(f"  KNN: best_k={best_k}, val_acc={max(val_scores_knn):.4f}")

# --- Decision Tree ---
depth_range = range(2, 16)
val_scores_dt = []
for d in depth_range:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train.values, y_train)
    val_scores_dt.append(accuracy_score(y_val.values, dt.predict(X_val.values)))
best_depth = list(depth_range)[np.argmax(val_scores_dt)]
dt_model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_model.fit(X_train.values, y_train)
print(f"  DT: best_depth={best_depth}, val_acc={max(val_scores_dt):.4f}")

# --- Random Forest ---
n_trees = [10, 30, 50, 100, 150, 200]
val_scores_rf = []
for n in n_trees:
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(X_train.values, y_train)
    val_scores_rf.append(accuracy_score(y_val.values, rf.predict(X_val.values)))
best_n = n_trees[np.argmax(val_scores_rf)]
rf_model = RandomForestClassifier(n_estimators=best_n, random_state=42, n_jobs=-1)
rf_model.fit(X_train.values, y_train)
print(f"  RF: best_n={best_n}, val_acc={max(val_scores_rf):.4f}")

# ============================================================
# BAGIAN 4: EVALUASI
# ============================================================
print("\n[4/5] Evaluating models...")

y_pred_knn = knn_model.predict(X_test_scaled)
y_pred_dt  = dt_model.predict(X_test.values)
y_pred_rf  = rf_model.predict(X_test.values)

def get_metrics(y_true, y_pred, name):
    return {
        'Model': name,
        'Accuracy':  accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall':    recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1-Score':  f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

results = pd.DataFrame([
    get_metrics(y_test.values, y_pred_knn, f'KNN (k={best_k})'),
    get_metrics(y_test.values, y_pred_dt,  f'Decision Tree (depth={best_depth})'),
    get_metrics(y_test.values, y_pred_rf,  f'Random Forest (n={best_n})')
])

print("\n  ╔══ TABEL PERBANDINGAN PERFORMA MODEL ══╗")
print(results.set_index('Model').round(4).to_string())
results.to_csv(os.path.join(OUTPUT_PATH, 'model_comparison.csv'), index=False)

# -------- Confusion matrices --------
short_names = ['Insuff.', 'Normal', 'OW_I', 'OW_II', 'Ob_I', 'Ob_II', 'Ob_III']
for preds, name, fname in [
    (y_pred_knn, f'KNN (k={best_k})',           '14_cm_knn.png'),
    (y_pred_dt,  f'Decision Tree (d={best_depth})', '15_cm_dt.png'),
    (y_pred_rf,  f'Random Forest (n={best_n})',  '16_cm_rf.png')
]:
    cm = confusion_matrix(y_test.values, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=short_names, yticklabels=short_names)
    ax.set_title(f'Confusion Matrix — {name}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, fname), dpi=150)
    plt.close()

# -------- Bar chart perbandingan --------
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.25
fig, ax = plt.subplots(figsize=(12, 6))
colors = ['royalblue', 'forestgreen', 'darkorange']
for i, row in results.iterrows():
    offset = (i - 1) * width
    bars = ax.bar(x + offset, [row[m] for m in metrics], width,
                  label=row['Model'], color=colors[i], alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.12)
ax.set_title('Perbandingan Performa Model — Test Set', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '13_model_comparison_bar.png'), dpi=150)
plt.close()

# -------- Feature importance --------
importances = rf_model.feature_importances_
feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 7))
colors_fi = ['#d73027' if v > feat_imp.median() else '#4575b4' for v in feat_imp.values]
ax.barh(feat_imp.index, feat_imp.values, color=colors_fi)
ax.set_title('Feature Importance — Random Forest', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, '17_feature_importance.png'), dpi=150)
plt.close()

# ============================================================
# BAGIAN 5: SIMPAN MODEL
# ============================================================
print("\n[5/5] Saving models...")
joblib.dump(knn_model, os.path.join(MODEL_PATH, 'knn_model.pkl'))
joblib.dump(dt_model,  os.path.join(MODEL_PATH, 'dt_model.pkl'))
joblib.dump(rf_model,  os.path.join(MODEL_PATH, 'rf_model.pkl'))

best_idx = results['F1-Score'].idxmax()
best_model_name = results.loc[best_idx, 'Model']
if 'Random Forest' in best_model_name:
    best_model = rf_model
    best_uses_scaled = False
elif 'KNN' in best_model_name:
    best_model = knn_model
    best_uses_scaled = True
else:
    best_model = dt_model
    best_uses_scaled = False

joblib.dump(best_model,        os.path.join(MODEL_PATH, 'best_model.pkl'))
joblib.dump(best_uses_scaled,  os.path.join(MODEL_PATH, 'best_model_uses_scaled.pkl'))
joblib.dump(best_model_name,   os.path.join(MODEL_PATH, 'best_model_name.pkl'))
joblib.dump(results,           os.path.join(MODEL_PATH, 'results_df.pkl'))

print(f"\n🏆 Model Terbaik: {best_model_name}")
print(f"   Accuracy : {results.loc[best_idx, 'Accuracy']:.4f}")
print(f"   F1-Score : {results.loc[best_idx, 'F1-Score']:.4f}")
print("\n✅ Pipeline selesai! Semua file tersimpan.")
print(f"   outputs/ → {len(os.listdir(OUTPUT_PATH))} file")
print(f"   models/  → {len(os.listdir(MODEL_PATH))} file")
