# ============================================================================
# APP.PY v2 — AI TSP (PRODUCTION, Binary Classification)
# ============================================================================
# PERUBAHAN dari v1:
#
#   v1 — Model prediksi 1 kelas dari 99 (indeks kota absolut)
#        Fallback 49.2% karena sering prediksi kota out-of-range
#
#   v2 — Model memberi SKOR untuk setiap kandidat kota unvisited
#        Pilih kandidat dengan skor tertinggi
#        Tidak pernah fallback karena kita yang suplai kandidat valid
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import joblib
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("  AI TRAVELING SALESMAN PROBLEM  v2")
print("  Binary Classification — Tanpa Fallback")
print("=" * 70)


# ============================================================================
# 1. LOAD MODEL
# ============================================================================

def load_model():
    required = ["models/tsp_model.pkl", "models/scaler.pkl",
                "models/feature_cols.pkl", "models/model_meta.pkl"]
    for path in required:
        if not os.path.exists(path):
            print(f"\n[ERROR] File tidak ditemukan: {path}")
            print("        Jalankan dulu:")
            print("          1. python generate_training_data.py")
            print("          2. python train_model.py")
            sys.exit(1)

    model     = joblib.load("models/tsp_model.pkl")
    scaler    = joblib.load("models/scaler.pkl")
    feat_cols = joblib.load("models/feature_cols.pkl")
    meta      = joblib.load("models/model_meta.pkl")

    print(f"\n[OK] Model dimuat (Binary Classification)")
    print(f"     Train accuracy : {meta['train_accuracy']*100:.1f}%")
    print(f"     Test accuracy  : {meta['test_accuracy']*100:.1f}%")
    print(f"     Step accuracy  : {meta.get('step_accuracy', 0)*100:.1f}%")
    print(f"     AUC Score      : {meta.get('roc_auc', 0):.4f}")
    return model, scaler, feat_cols, meta


# ============================================================================
# 2. INPUT KOTA
# ============================================================================

def input_kota_manual():
    print("\n" + "=" * 70)
    print("  INPUT KOTA")
    print("=" * 70)

    while True:
        try:
            n = int(input("Jumlah kota (3-50): "))
            if 3 <= n <= 50:
                break
            print("  Masukkan antara 3 sampai 50.")
        except ValueError:
            print("  Masukkan angka bulat.")

    city_names = []
    coords     = []

    print(f"\nMasukkan nama dan koordinat {n} kota.")
    print("Format: x y  (contoh: 10.5 23.7)\n")

    for i in range(n):
        nama = input(f"Nama kota {i+1}: ").strip() or f"Kota {i+1}"
        while True:
            try:
                raw  = input(f"Koordinat {nama} (x y): ").strip()
                x, y = map(float, raw.split())
                break
            except ValueError:
                print("  Format salah. Contoh: 10.5 23.7")
        city_names.append(nama)
        coords.append([x, y])

    return city_names, np.array(coords)


def input_kota_contoh():
    city_names = [
        "Jakarta", "Surabaya", "Bandung", "Semarang", "Medan",
        "Makassar", "Palembang", "Yogyakarta", "Malang", "Denpasar"
    ]
    coords = np.array([
        [106.85, -6.21], [112.75, -7.25], [107.61, -6.91],
        [110.42, -6.99], [ 98.67,  3.58], [119.43, -5.13],
        [104.76, -2.99], [110.36, -7.80], [112.63, -7.98],
        [115.22, -8.65],
    ])
    return city_names, coords


# ============================================================================
# 3. DISTANCE MATRIX
# ============================================================================

def build_distance_matrix(coords):
    n    = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                dist[i][j] = np.sqrt(dx**2 + dy**2)
    return dist


# ============================================================================
# 4. BANGUN FITUR KANDIDAT (format v2)
# ============================================================================

def build_candidate_features(current, candidate, coords, dist_matrix,
                               visited, step_idx, feat_cols):
    """
    Bangun fitur untuk 1 kandidat kota di 1 langkah.
    Format HARUS sama dengan yang dipakai saat training.
    """
    n              = len(coords)
    unvisited_ids  = [j for j in range(n) if not visited[j]]
    dists_unvis    = [dist_matrix[current][j] for j in unvisited_ids]
    n_unvis        = len(unvisited_ids)

    min_dist = min(dists_unvis) if dists_unvis else 0
    max_dist = max(dists_unvis) if dists_unvis else 0
    avg_dist = np.mean(dists_unvis) if dists_unvis else 0
    std_dist = np.std(dists_unvis)  if dists_unvis else 0

    # Rank kandidat ini di antara unvisited
    dist_val = dist_matrix[current][candidate]
    sorted_dists = sorted(dists_unvis)
    rank = sorted_dists.index(dist_val) if dist_val in sorted_dists else 0

    feat_dict = {
        'dist_to_candidate'   : dist_val,
        'dist_ratio_avg'      : dist_val / (avg_dist + 1e-9),
        'dist_sq'             : dist_val ** 2,
        'delta_x'             : coords[candidate][0] - coords[current][0],
        'delta_y'             : coords[candidate][1] - coords[current][1],
        'candidate_x'         : coords[candidate][0],
        'candidate_y'         : coords[candidate][1],
        'current_x'           : coords[current][0],
        'current_y'           : coords[current][1],
        'n_unvisited'         : n_unvis,
        'step_progress'       : step_idx / max(n - 1, 1),
        'avg_dist_unvisited'  : avg_dist,
        'min_dist_unvisited'  : min_dist,
        'max_dist_unvisited'  : max_dist,
        'dist_std_unvisited'  : std_dist,
    }

    feat_vector = np.array([feat_dict.get(col, 0.0) for col in feat_cols])
    return feat_vector


# ============================================================================
# 5. PREDIKSI RUTE (tanpa fallback)
# ============================================================================

def predict_route(model, scaler, feat_cols, coords, dist_matrix):
    """
    Prediksi rute step by step menggunakan binary model.

    Untuk setiap langkah:
    1. Ambil semua kota yang belum dikunjungi (kandidat valid)
    2. Bangun fitur untuk SETIAP kandidat
    3. Skor tiap kandidat dengan model (proba kelas 1)
    4. Pilih kandidat dengan skor tertinggi

    TIDAK ADA FALLBACK karena kandidat selalu valid.
    """
    n            = len(coords)
    visited      = [False] * n
    visited[0]   = True
    route        = [0]
    total_dist   = 0
    step_details = []
    current      = 0

    for step_idx in range(n - 1):
        # Semua kandidat valid (belum dikunjungi)
        candidates = [j for j in range(n) if not visited[j]]

        # Bangun fitur semua kandidat sekaligus
        feat_matrix = np.array([
            build_candidate_features(
                current, cand, coords, dist_matrix,
                visited, step_idx, feat_cols
            )
            for cand in candidates
        ])

        # Skor tiap kandidat
        feat_scaled = scaler.transform(feat_matrix)
        scores      = model.predict_proba(feat_scaled)[:, 1]  # proba kelas 1

        # Pilih yang skornya tertinggi
        best_idx    = int(np.argmax(scores))
        next_city   = candidates[best_idx]
        confidence  = float(scores[best_idx])

        step_details.append({
            'step'         : step_idx,
            'from_city'    : current,
            'to_city'      : next_city,
            'distance'     : dist_matrix[current][next_city],
            'confidence'   : confidence,
            'n_candidates' : len(candidates),
            'fallback_used': False,   # tidak pernah fallback di v2
        })

        total_dist       += dist_matrix[current][next_city]
        visited[next_city] = True
        route.append(next_city)
        current = next_city

    total_dist += dist_matrix[current][0]
    route.append(0)

    return route, total_dist, step_details


# ============================================================================
# 6. VISUALISASI
# ============================================================================

def visualize_route(city_names, coords, route, total_dist,
                    step_details, save_path="outputs/hasil_rute.png"):
    os.makedirs("outputs", exist_ok=True)

    avg_conf = np.mean([s['confidence'] for s in step_details]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#f8f9fa')

    # Panel kiri: Peta rute
    ax = axes[0]
    ax.set_facecolor('#f8f9fa')

    for i in range(len(route) - 1):
        src, dst = route[i], route[i + 1]
        ax.annotate("", xy=(coords[dst][0], coords[dst][1]),
                    xytext=(coords[src][0], coords[src][1]),
                    arrowprops=dict(arrowstyle='->', color='#3498db',
                                    lw=1.8, mutation_scale=12))

    for i, (nama, (x, y)) in enumerate(zip(city_names, coords)):
        is_start = (i == 0)
        color    = '#e74c3c' if is_start else '#3498db'
        marker   = '*' if is_start else 'o'
        size     = 300 if is_start else 120
        order    = route.index(i) + 1 if i in route[:-1] else '–'

        ax.scatter(x, y, c=color, s=size, zorder=5,
                   edgecolors='white', linewidths=1.5, marker=marker)
        ax.annotate(
            f"{'★' if is_start else str(order)}. {nama}",
            (x, y), textcoords="offset points", xytext=(8, 6),
            fontsize=8.5, color='#2c3e50',
            fontweight='bold' if is_start else 'normal',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      alpha=0.85, edgecolor='#bdc3c7', linewidth=0.8)
        )

    ax.set_title(
        f"Rute Hasil AI v2 (Binary Classification)\n"
        f"Total Jarak: {total_dist:.4f}  |  Avg Confidence: {avg_conf:.1f}%",
        fontsize=11, fontweight='bold', color='#2c3e50'
    )
    ax.set_xlabel("Koordinat X", fontsize=9)
    ax.set_ylabel("Koordinat Y", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.text(0.02, 0.02,
            f"Model: Random Forest (Binary)\nKota: {len(city_names)}\nFallback: 0 langkah ✓",
            transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.85, edgecolor='#bdc3c7'))

    # Panel kanan: Confidence per langkah
    ax2 = axes[1]
    ax2.set_facecolor('#f8f9fa')

    steps_list = [s['step'] + 1      for s in step_details]
    confs      = [s['confidence']*100 for s in step_details]

    ax2.bar(steps_list, confs, color='#27ae60', edgecolor='white', alpha=0.85)
    ax2.axhline(y=avg_conf, color='orange', linestyle='--',
                linewidth=1.5, label=f'Rata-rata: {avg_conf:.1f}%')
    ax2.set_title("Confidence Model per Langkah\n(Semua langkah = prediksi model, tanpa fallback ✓)",
                  fontsize=11, fontweight='bold', color='#2c3e50')
    ax2.set_xlabel("Langkah ke-", fontsize=9)
    ax2.set_ylabel("Confidence (%)", fontsize=9)
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Grafik disimpan: {save_path}")


# ============================================================================
# 7. TAMPILKAN HASIL
# ============================================================================

def tampilkan_hasil(city_names, route, total_dist, elapsed, step_details):
    avg_conf = np.mean([s['confidence'] for s in step_details]) * 100

    print("\n" + "=" * 70)
    print("  HASIL AI TSP v2 — BINARY CLASSIFICATION")
    print("=" * 70)

    print("\n  RUTE OPTIMAL (diprediksi model ML):")
    print("  " + " → ".join([city_names[r] for r in route]))

    print(f"\n  Total jarak    : {total_dist:.4f}")
    print(f"  Jumlah kota    : {len(city_names)}")
    print(f"  Waktu prediksi : {elapsed*1000:.2f} ms")
    print(f"  Avg confidence : {avg_conf:.1f}%")
    print(f"  Fallback       : 0 langkah ✓  (tidak ada fallback di v2)")

    print("\n  DETAIL LANGKAH:")
    print(f"  {'Step':>4}  {'Dari':15} {'→  Ke':15} {'Jarak':>8}  {'Conf':>7}  {'Kandidat':>9}")
    print("  " + "-" * 68)
    for s in step_details:
        print(f"  {s['step']+1:>4}  "
              f"{city_names[s['from_city']]:15} "
              f"→  {city_names[s['to_city']]:15} "
              f"{s['distance']:>8.3f}  "
              f"{s['confidence']*100:>6.1f}%  "
              f"{s['n_candidates']:>8} kota")
    print(f"\n  Total: {total_dist:.4f}")


# ============================================================================
# 8. MAIN
# ============================================================================

def main():
    os.makedirs("outputs", exist_ok=True)

    model, scaler, feat_cols, meta = load_model()

    print("\nPilih mode input:")
    print("  1. Input manual (koordinat sendiri)")
    print("  2. Contoh 10 kota Indonesia")
    mode = input("\nPilih (1/2, default=2): ").strip() or "2"

    if mode == "1":
        city_names, coords = input_kota_manual()
    else:
        city_names, coords = input_kota_contoh()
        print(f"\n[INFO] Menggunakan {len(city_names)} kota contoh:")
        for i, (nm, (x, y)) in enumerate(zip(city_names, coords)):
            print(f"  {i:2}. {nm:15} ({x:.2f}, {y:.2f})")

    dist_matrix = build_distance_matrix(coords)

    print(f"\n[INFO] Model ML memprediksi rute untuk {len(city_names)} kota...")
    start_time = time.time()
    route, total_dist, step_details = predict_route(
        model, scaler, feat_cols, coords, dist_matrix
    )
    elapsed = time.time() - start_time
    print(f"[OK] Prediksi selesai dalam {elapsed*1000:.2f} ms")

    tampilkan_hasil(city_names, route, total_dist, elapsed, step_details)

    print("\n[INFO] Membuat visualisasi...")
    visualize_route(city_names, coords, route, total_dist, step_details)

    print("\n" + "=" * 70)
    lagi = input("Coba dengan kota lain? (y/n, default=n): ").strip().lower()
    if lagi == 'y':
        main()


if __name__ == "__main__":
    main()
