# ============================================================================
# TRAIN_MODEL.PY  (v2 — Binary Classification)
# ============================================================================
# PERUBAHAN dari v1:
#
#   v1 — Klasifikasi 99 kelas (next_city = 1..99)
#        Confidence rata-rata 7.6%, fallback 49.2%
#
#   v2 — Binary classification (label = 0 atau 1)
#        Model belajar: "apakah kandidat ini yang harus dipilih?"
#        Tidak ada kelas out-of-range → tidak ada fallback
#        Confidence jauh lebih tinggi
#
# Cara kerja saat inferensi (app.py):
#   1. Ambil semua kota yang belum dikunjungi
#   2. Bangun fitur untuk SETIAP kandidat
#   3. Model memberi skor (proba kelas 1) untuk masing-masing
#   4. Pilih kandidat dengan skor tertinggi
#   → Tidak pernah out-of-range karena kita yang filter unvisited
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import (confusion_matrix, classification_report,
                                      accuracy_score, roc_curve, auc,
                                      ConfusionMatrixDisplay)

print("=" * 70)
print("  TRAINING MODEL v2 — BINARY CLASSIFICATION")
print("  Label: 1 = kota ini dipilih greedy | 0 = tidak dipilih")
print("=" * 70)

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ============================================================================
# 1. LOAD DATASET
# ============================================================================

print("\n[1] LOAD DATASET")
print("-" * 50)

DATASET_PATH = "data/tsp_step_dataset_v2.csv"
if not os.path.exists(DATASET_PATH):
    print(f"[ERROR] File tidak ditemukan: {DATASET_PATH}")
    print("        Jalankan dulu: python generate_training_data.py")
    exit(1)

df = pd.read_csv(DATASET_PATH)
print(f"    Shape          : {df.shape}")
print(f"    Label = 1      : {(df['label']==1).sum():,}")
print(f"    Label = 0      : {(df['label']==0).sum():,}")
print(f"    Rasio 1:0      : 1 : {(df['label']==0).sum() // (df['label']==1).sum()}")


# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================

print("\n[2] FEATURE ENGINEERING")
print("-" * 50)

drop_cols    = ['instance_id', 'step', 'candidate_city', 'label']
feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols].values
y = df['label'].values

print(f"    Fitur          : {len(feature_cols)}")
print(f"    Fitur list     : {feature_cols}")
print(f"    Total sampel   : {len(X):,}")

joblib.dump(feature_cols, "models/feature_cols.pkl")


# ============================================================================
# 3. TRAIN-TEST SPLIT (berdasarkan instance_id)
# ============================================================================

print("\n[3] TRAIN-TEST SPLIT")
print("-" * 50)

instance_ids  = df['instance_id'].unique()
n_test_inst   = max(1, int(len(instance_ids) * 0.2))
test_inst_ids = set(np.random.RandomState(42).choice(
    instance_ids, size=n_test_inst, replace=False
))

train_mask = ~df['instance_id'].isin(test_inst_ids)
test_mask  =  df['instance_id'].isin(test_inst_ids)

X_train = X[train_mask]
X_test  = X[test_mask]
y_train = y[train_mask]
y_test  = y[test_mask]

print(f"    Train baris    : {len(X_train):,}")
print(f"    Test baris     : {len(X_test):,}")


# ============================================================================
# 4. SCALING
# ============================================================================

print("\n[4] FEATURE SCALING")
print("-" * 50)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")
print("    StandardScaler disimpan: models/scaler.pkl")


# ============================================================================
# 5. TRAINING
# ============================================================================

print("\n[5] TRAINING RANDOM FOREST (Binary)")
print("-" * 50)
print("    Sedang training...")

start = time.time()

model = RandomForestClassifier(
    n_estimators      = 200,
    max_depth         = 20,
    min_samples_split = 5,
    min_samples_leaf  = 2,
    max_features      = 'sqrt',
    class_weight      = 'balanced',  # handle imbalance label 1 vs 0
    n_jobs            = -1,
    random_state      = 42,
)

model.fit(X_train, y_train)
elapsed = time.time() - start

print(f"    Selesai dalam {elapsed:.1f} detik")
joblib.dump(model, "models/tsp_model.pkl")
print("    Model disimpan: models/tsp_model.pkl")


# ============================================================================
# 6. EVALUASI BINARY
# ============================================================================

print("\n[6] EVALUASI MODEL (Binary)")
print("-" * 50)

y_pred      = model.predict(X_test)
y_proba     = model.predict_proba(X_test)[:, 1]   # skor kelas 1
train_acc   = model.score(X_train, y_train)
test_acc    = accuracy_score(y_test, y_pred)

print(f"    Train Accuracy : {train_acc:.4f} ({train_acc*100:.1f}%)")
print(f"    Test Accuracy  : {test_acc:.4f}  ({test_acc*100:.1f}%)")

# ── Evaluasi cara kerja saat inferensi ──
# Simulasi: untuk setiap langkah (step), apakah model memilih
# kandidat dengan skor tertinggi = kota yang greedy pilih?
print("\n    Simulasi inferensi (top-1 dari kandidat per langkah):")
df_test = df[test_mask].copy()
df_test['score'] = y_proba
df_test['pred']  = y_pred

correct_steps = 0
total_steps   = 0

for (inst_id, step_id), group in df_test.groupby(['instance_id', 'step']):
    if len(group) == 0:
        continue
    # Kota dengan skor tertinggi = pilihan model
    best_idx   = group['score'].idxmax()
    is_correct = group.loc[best_idx, 'label'] == 1
    if is_correct:
        correct_steps += 1
    total_steps += 1

step_accuracy = correct_steps / total_steps if total_steps > 0 else 0
print(f"    Step accuracy  : {step_accuracy:.4f} ({step_accuracy*100:.1f}%)")
print(f"    (% langkah di mana model pilih kota yang sama dengan greedy)")

# Classification report
report = classification_report(y_test, y_pred,
                                target_names=['Tidak Dipilih','Dipilih Greedy'],
                                zero_division=0)
print("\n    Classification Report:")
print(report)

with open("outputs/classification_report.txt", "w") as f:
    f.write("Classification Report — TSP Imitation Learning v2 (Binary)\n")
    f.write("=" * 60 + "\n")
    f.write(f"Train Accuracy  : {train_acc:.4f}\n")
    f.write(f"Test Accuracy   : {test_acc:.4f}\n")
    f.write(f"Step Accuracy   : {step_accuracy:.4f}\n\n")
    f.write(report)
print("    Disimpan: outputs/classification_report.txt")


# ============================================================================
# 7. CONFUSION MATRIX
# ============================================================================

print("\n[7] CONFUSION MATRIX")
print("-" * 50)

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Tidak Dipilih (0)', 'Dipilih Greedy (1)'],
            yticklabels=['Tidak Dipilih (0)', 'Dipilih Greedy (1)'],
            ax=ax)
ax.set_title("Confusion Matrix — Binary Classification\nTSP Imitation Learning v2",
             fontsize=13, fontweight='bold')
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual",    fontsize=11)
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Disimpan: outputs/confusion_matrix.png")


# ============================================================================
# 8. ROC CURVE & AUC SCORE  (fitur opsional)
# ============================================================================

print("\n[8] ROC CURVE & AUC SCORE")
print("-" * 50)

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc     = auc(fpr, tpr)

print(f"    AUC Score : {roc_auc:.4f}")

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color='#3266ad', lw=2,
        label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
        label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.15, color='#3266ad')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate',  fontsize=11)
ax.set_title('ROC Curve — TSP Imitation Learning v2',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/roc_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Disimpan: outputs/roc_curve.png")


# ============================================================================
# 9. K-FOLD CROSS VALIDATION  (fitur opsional)
# ============================================================================

print("\n[9] K-FOLD CROSS VALIDATION (5-Fold)")
print("-" * 50)
print("    Menjalankan 5-fold CV... (mungkin butuh beberapa menit)")

# Pakai subset untuk CV agar lebih cepat
cv_size   = min(10000, len(X_train))
idx_cv    = np.random.RandomState(42).choice(len(X_train), cv_size, replace=False)
X_cv      = X_train[idx_cv]
y_cv      = y_train[idx_cv]

skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_cv, y_cv, cv=skf, scoring='accuracy', n_jobs=-1)

print(f"    CV Scores  : {cv_scores}")
print(f"    CV Mean    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

with open("outputs/classification_report.txt", "a") as f:
    f.write("\n\nK-Fold Cross Validation (5-Fold)\n")
    f.write("=" * 40 + "\n")
    f.write(f"CV Scores : {cv_scores}\n")
    f.write(f"CV Mean   : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}\n")


# ============================================================================
# 10. FEATURE IMPORTANCE
# ============================================================================

print("\n[10] FEATURE IMPORTANCE")
print("-" * 50)

importances = model.feature_importances_
feat_imp_df = pd.DataFrame({
    'feature'   : feature_cols,
    'importance': importances
}).sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(9, 6))
colors  = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feat_imp_df)))
ax.barh(feat_imp_df['feature'][::-1],
        feat_imp_df['importance'][::-1],
        color=colors[::-1], edgecolor='white')
ax.set_title("Feature Importance\nTSP Imitation Learning v2 (Binary)",
             fontsize=13, fontweight='bold')
ax.set_xlabel("Importance Score", fontsize=11)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()

print("    Top fitur terpenting:")
for _, r in feat_imp_df.head(5).iterrows():
    print(f"      {r['feature']:30s} : {r['importance']:.4f}")
print("    Disimpan: outputs/feature_importance.png")


# ============================================================================
# 11. TRAINING SUMMARY CHART
# ============================================================================

print("\n[11] TRAINING SUMMARY CHART")
print("-" * 50)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Bar: train vs test accuracy
axes[0].bar(['Train', 'Test', 'Step (Inf.)'],
            [train_acc, test_acc, step_accuracy],
            color=['#3266ad', '#e87c3e', '#27ae60'],
            edgecolor='white', width=0.4)
axes[0].set_ylim(0, 1.1)
axes[0].set_title("Akurasi Model", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Accuracy")
for i, v in enumerate([train_acc, test_acc, step_accuracy]):
    axes[0].text(i, v + 0.02, f"{v*100:.1f}%",
                 ha='center', fontweight='bold', fontsize=11)
axes[0].grid(axis='y', alpha=0.3)

# ROC Curve mini
axes[1].plot(fpr, tpr, color='#3266ad', lw=2)
axes[1].plot([0,1],[0,1],'--',color='gray',lw=1)
axes[1].fill_between(fpr, tpr, alpha=0.15, color='#3266ad')
axes[1].set_title(f"ROC Curve (AUC={roc_auc:.3f})",
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
axes[1].grid(True, alpha=0.3)

# CV scores
axes[2].bar(range(1, 6), cv_scores,
            color='#3266ad', edgecolor='white', alpha=0.85)
axes[2].axhline(cv_scores.mean(), color='red', linestyle='--',
                label=f'Mean={cv_scores.mean():.3f}')
axes[2].set_title("5-Fold Cross Validation", fontsize=12, fontweight='bold')
axes[2].set_xlabel("Fold"); axes[2].set_ylabel("Accuracy")
axes[2].set_ylim(0, 1.1); axes[2].legend(); axes[2].grid(axis='y', alpha=0.3)

plt.suptitle("Training Summary — TSP Imitation Learning v2 (Binary)",
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("outputs/training_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("    Disimpan: outputs/training_summary.png")


# ============================================================================
# 12. SIMPAN METADATA
# ============================================================================

model_meta = {
    'formulation'      : 'binary',
    'train_accuracy'   : float(train_acc),
    'test_accuracy'    : float(test_acc),
    'step_accuracy'    : float(step_accuracy),
    'roc_auc'          : float(roc_auc),
    'cv_mean'          : float(cv_scores.mean()),
    'cv_std'           : float(cv_scores.std()),
    'n_features'       : len(feature_cols),
    'n_classes'        : 2,
    'n_estimators'     : model.n_estimators,
    'feature_cols'     : feature_cols,
    'n_train_samples'  : len(X_train),
    'n_test_samples'   : len(X_test),
}
joblib.dump(model_meta, "models/model_meta.pkl")


# ============================================================================
# RINGKASAN AKHIR
# ============================================================================

print("\n" + "=" * 70)
print("  TRAINING SELESAI (Binary Classification)")
print("=" * 70)
print(f"  Train Accuracy  : {train_acc*100:.1f}%")
print(f"  Test Accuracy   : {test_acc*100:.1f}%")
print(f"  Step Accuracy   : {step_accuracy*100:.1f}%  ← ini yang paling relevan")
print(f"  AUC Score       : {roc_auc:.4f}")
print(f"  CV Mean         : {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
print(f"\n  Model disimpan  : models/tsp_model.pkl")
print(f"\n  Interpretasi Step Accuracy:")
print(f"  - {step_accuracy*100:.1f}% langkah diprediksi SAMA PERSIS dengan greedy")
print(f"  - Tidak ada lagi fallback saat inferensi!")
print(f"  - Confidence jauh lebih tinggi (2 kelas vs 99 kelas)")
print(f"\n  Lanjutkan dengan: python app.py")
print("=" * 70)
