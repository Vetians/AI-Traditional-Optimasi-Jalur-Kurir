# ============================================================================
# GENERATE_TRAINING_DATA.PY  (v2 — Ranking Formulation)
# ============================================================================
# PERUBAHAN UTAMA dari v1:
#
#   v1 — Label = indeks kota ABSOLUT (misal: next_city = 47)
#        Problem: kelas 1–99 global, beda instance beda makna
#
#   v2 — Label = RANK kota di antara unvisited cities (selalu = 0)
#        Karena greedy SELALU pilih kota terdekat → rank 0
#        Tapi kita simpan fitur RELATIF dari unvisited cities
#        sehingga model belajar "kota mana yang paling layak dipilih"
#
# Pendekatan baru:
#   - 1 baris = 1 KANDIDAT kota (bukan 1 langkah)
#   - Setiap langkah menghasilkan (n_unvisited) baris kandidat
#   - Fitur: properti kandidat relatif terhadap posisi saat ini
#   - Label: 1 = kota ini yang greedy pilih, 0 = tidak dipilih
#   - Ini mengubah problem dari klasifikasi 99-kelas → BINARY
#
# Dengan binary classification:
#   - Model belajar: "apakah kota ini layak dipilih berikutnya?"
#   - Tidak ada masalah kelas out-of-range
#   - Tidak ada fallback karena model hanya pilih dari unvisited
#   - Confidence jauh lebih tinggi (2 kelas vs 99 kelas)
# ============================================================================

import numpy as np
import pandas as pd
import ast
import os
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("  GENERATE TRAINING DATA v2 — RANKING / BINARY FORMULATION")
print("=" * 70)
print()
print("  Pendekatan baru: Binary Classification")
print("  Label: 1 = kota ini dipilih greedy, 0 = tidak dipilih")
print("  Tidak ada lagi 99 kelas global!")
print()


# ============================================================================
# 1. FUNGSI GREEDY TSP — rekam dengan format baru
# ============================================================================

def greedy_tsp_steps_v2(coords, dist_matrix):
    """
    Jalankan Greedy TSP dan rekam setiap langkah dalam format BARU.

    Format lama (v1):
        1 langkah = 1 baris, label = indeks kota absolut

    Format baru (v2):
        1 langkah = N baris (N = jumlah kota yang belum dikunjungi)
        Setiap baris = 1 kandidat kota
        Label: 1 jika ini kota yang dipilih greedy, 0 jika tidak

    Fitur per kandidat (relatif, bukan absolut):
        - dist_to_candidate   : jarak dari posisi saat ini ke kandidat
        - dist_rank           : peringkat jarak (0 = terdekat)
        - dist_ratio_min      : rasio jarak vs kota terdekat
        - delta_x, delta_y    : selisih koordinat
        - candidate_x/y       : koordinat kandidat
        - current_x/y         : koordinat posisi saat ini
        - n_unvisited         : jumlah kota yang belum dikunjungi
        - step_progress       : progress perjalanan (0.0 = awal, 1.0 = akhir)
        - avg_dist_unvisited  : rata-rata jarak ke semua unvisited
        - min_dist_unvisited  : jarak minimum ke unvisited
        - max_dist_unvisited  : jarak maksimum ke unvisited
    """
    n          = len(coords)
    visited    = [False] * n
    visited[0] = True
    route      = [0]
    steps      = []
    total_dist = 0
    current    = 0

    for step_idx in range(n - 1):
        # Daftar kota yang belum dikunjungi
        unvisited_ids = [j for j in range(n) if not visited[j]]

        # Jarak ke semua unvisited
        dists_to_unvisited = [dist_matrix[current][j] for j in unvisited_ids]

        # Kota terdekat = pilihan greedy
        nearest_idx  = int(np.argmin(dists_to_unvisited))
        nearest_city = unvisited_ids[nearest_idx]
        nearest_dist = dists_to_unvisited[nearest_idx]

        # Statistik jarak untuk normalisasi
        min_dist = min(dists_to_unvisited)
        max_dist = max(dists_to_unvisited)
        avg_dist = np.mean(dists_to_unvisited)
        n_unvis  = len(unvisited_ids)

        # Buat 1 baris per kandidat kota (urutan asli, bukan sorted)
        for cand_city, dist_val in zip(unvisited_ids, dists_to_unvisited):
            is_chosen = 1 if cand_city == nearest_city else 0

            row = {
                # ── Meta (tidak dipakai sebagai fitur) ──
                'instance_id'          : None,
                'step'                 : step_idx,
                'candidate_city'       : cand_city,

                # ── Fitur (TANPA dist_rank dan dist_ratio_min — itu leakage!) ──
                # dist_rank selalu 0 untuk label=1 → model langsung "curang"
                # dist_ratio_min selalu 1.0 untuk label=1 → sama saja bocor
                'dist_to_candidate'    : dist_val,
                'dist_ratio_avg'       : dist_val / (avg_dist + 1e-9),
                'dist_sq'              : dist_val ** 2,
                'delta_x'              : coords[cand_city][0] - coords[current][0],
                'delta_y'              : coords[cand_city][1] - coords[current][1],

                # ── Fitur konteks ──
                'candidate_x'          : coords[cand_city][0],
                'candidate_y'          : coords[cand_city][1],
                'current_x'            : coords[current][0],
                'current_y'            : coords[current][1],
                'n_unvisited'          : n_unvis,
                'step_progress'        : step_idx / (n - 1),
                'avg_dist_unvisited'   : avg_dist,
                'min_dist_unvisited'   : min_dist,
                'max_dist_unvisited'   : max_dist,
                'dist_std_unvisited'   : np.std(dists_to_unvisited),

                # ── Label ──
                'label'                : is_chosen,   # 1 = dipilih greedy
            }
            steps.append(row)

        # Lanjut ke kota berikutnya
        total_dist     += nearest_dist
        visited[nearest_city] = True
        route.append(nearest_city)
        current = nearest_city

    total_dist += dist_matrix[current][0]
    route.append(0)

    return steps, route, total_dist


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
# 2. LOAD DATASET
# ============================================================================

def load_dataset(file_path):
    print(f"[1] Load dataset: {file_path}")
    df = pd.read_csv(file_path)
    print(f"    Total instance : {len(df)}")
    return df


# ============================================================================
# 3. GENERATE DATASET BARU
# ============================================================================


# ============================================================================
# 3. GENERATE + SIMPAN LANGSUNG KE DISK (batch, anti OOM)
# ============================================================================

def generate_step_dataset(df, max_instances=None, output_path=None, batch_size=200):
    """
    Generate dataset dengan batch writing ke CSV agar tidak OOM.
    Setiap batch_size instance langsung ditulis ke disk, RAM dibersihkan.
    """
    if max_instances:
        df = df.head(max_instances)

    n_total = len(df)
    print(f"\n[2] Generate kandidat dari {n_total} instance...")
    print(f"    Batch size     : {batch_size} instance per flush ke disk")

    total_rows     = 0
    total_chosen   = 0
    total_routes   = []
    n_cities_list  = []
    start_time     = time.time()
    header_written = False
    batch_rows     = []

    for enum_idx, (idx, row) in enumerate(df.iterrows()):
        if isinstance(row['city_coordinates'], str):
            coords = np.array(ast.literal_eval(row['city_coordinates']))
        else:
            coords = np.array(row['city_coordinates'])

        n = len(coords)
        n_cities_list.append(n)

        if 'distance_matrix' in df.columns and pd.notna(row.get('distance_matrix')):
            if isinstance(row['distance_matrix'], str):
                dist_matrix = np.array(ast.literal_eval(row['distance_matrix']))
            else:
                dist_matrix = np.array(row['distance_matrix'])
        else:
            dist_matrix = build_distance_matrix(coords)

        steps, route, total_dist = greedy_tsp_steps_v2(coords, dist_matrix)
        total_routes.append(total_dist)

        for step in steps:
            step['instance_id'] = idx
            batch_rows.append(step)

        # Flush ke disk setiap batch_size instance
        if (enum_idx + 1) % batch_size == 0 or (enum_idx + 1) == n_total:
            batch_df = pd.DataFrame(batch_rows)

            meta_cols  = ['instance_id', 'step', 'candidate_city']
            label_col  = ['label']
            feat_cols  = [c for c in batch_df.columns if c not in meta_cols + label_col]
            final_cols = meta_cols + feat_cols + label_col
            batch_df   = batch_df[[c for c in final_cols if c in batch_df.columns]]

            batch_df.to_csv(output_path,
                            mode='w' if not header_written else 'a',
                            header=not header_written,
                            index=False)
            header_written = True

            chosen        = int((batch_df['label'] == 1).sum())
            total_rows   += len(batch_df)
            total_chosen += chosen
            elapsed       = time.time() - start_time
            print(f"    Progress: {enum_idx+1}/{n_total} instance | "
                  f"{total_rows:,} baris | {total_chosen:,} chosen | {elapsed:.1f}s")

            batch_rows = []   # Bersihkan RAM

    elapsed = time.time() - start_time
    print(f"\n    Selesai dalam  : {elapsed:.1f} detik")
    print(f"    Total kandidat : {total_rows:,}")
    print(f"    Label = 1      : {total_chosen:,}  (dipilih greedy)")
    print(f"    Label = 0      : {total_rows - total_chosen:,}  (tidak dipilih)")
    print(f"    Rasio pos/neg  : 1 : {(total_rows - total_chosen) // max(total_chosen, 1)}")
    print(f"    Jumlah kota    : min={min(n_cities_list)}, max={max(n_cities_list)}")
    print(f"    Dataset        : {output_path}  ({os.path.getsize(output_path)/1024/1024:.1f} MB)")

    return {
        'n_instances'   : n_total,
        'total_rows'    : total_rows,
        'n_chosen'      : total_chosen,
        'mean_distance' : float(np.mean(total_routes)),
    }


# ============================================================================
# 4. MAIN
# ============================================================================

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    DATASET_PATH  = "data/tsp_dataset.csv"
    OUTPUT_PATH   = "data/tsp_step_dataset_v2.csv"
    MAX_INSTANCES = None   # None = pakai semua 2783 instance

    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] File tidak ditemukan: {DATASET_PATH}")
        exit(1)

    df   = load_dataset(DATASET_PATH)
    meta = generate_step_dataset(df, max_instances=MAX_INSTANCES,
                                  output_path=OUTPUT_PATH, batch_size=200)

    print("\n" + "=" * 70)
    print("  SELESAI — Format Baru (Binary Classification)")
    print("=" * 70)
    print(f"  Instance   : {meta['n_instances']}")
    print(f"  Kandidat   : {meta['total_rows']:,}")
    print(f"  Label 1    : {meta['n_chosen']:,} (greedy pilih)")
    print(f"  Dataset    : {OUTPUT_PATH}")
    print(f"  Lanjutkan  : python train_model.py")
    print("=" * 70)
