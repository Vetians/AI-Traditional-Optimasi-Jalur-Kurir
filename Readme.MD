# AI Traveling Salesman Problem — Imitation Learning

## Konsep Proyek

Model ML mempelajari pola keputusan algoritma Greedy TSP, lalu berdiri
sendiri sebagai AI pencari rute — tanpa perlu menjalankan algoritma lagi.

```
FASE TRAINING (sekali saja):
  Greedy TSP → rekam setiap keputusan → training model ML

FASE PRODUCTION:
  User input koordinat → Model ML prediksi rute langsung
```

---

## Struktur Folder

```
tsp_project/
│
├── data/
│   ├── tsp_dataset.csv          ← taruh dataset Kaggle di sini
│   └── tsp_step_dataset.csv     ← otomatis dibuat
│
├── models/                      ← otomatis dibuat
│   ├── tsp_model.pkl
│   ├── scaler.pkl
│   ├── feature_cols.pkl
│   └── model_meta.pkl
│
├── outputs/                     ← otomatis dibuat
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   ├── feature_importance.png
│   ├── training_summary.png
│   └── hasil_rute.png
│
├── generate_training_data.py    ← STEP 1
├── train_model.py               ← STEP 2
├── app.py                       ← STEP 3 (production)
└── README.md
```

---

## Cara Menjalankan

### Step 1 — Siapkan Dataset
Letakkan `tsp_dataset.csv` dari Kaggle ke folder `data/`

### Step 2 — Generate Training Data
```bash
python generate_training_data.py
```
Output: `data/tsp_step_dataset.csv`

### Step 3 — Training Model
```bash
python train_model.py
```
Output: `models/tsp_model.pkl` + grafik evaluasi di `outputs/`

### Step 4 — Jalankan AI
```bash
python app.py
```

---

## Penjelasan Teknis

### Kenapa Random Forest?
- Tidak perlu scaling yang sempurna
- Robust terhadap outlier koordinat
- Menghasilkan feature importance
- Akurasi tinggi untuk dataset tabular

### Apa itu "Fallback"?
Saat model memprediksi kota yang sudah dikunjungi atau out of range,
sistem otomatis fallback ke kota terdekat yang belum dikunjungi.
Idealnya fallback = 0 (model sepenuhnya yakin di semua langkah).

### Interpretasi Akurasi
- Akurasi = % langkah yang identik dengan keputusan greedy
- 70% = 7 dari 10 langkah sama persis dengan greedy
- Rute tetap valid meski tidak 100% identik

---

## Library yang Dibutuhkan
```
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```