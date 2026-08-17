# ============================================================================
# TEST_COMPARE.PY — Uji Model AI vs Greedy Murni
# ============================================================================
# Menjalankan beberapa kasus uji dan membandingkan:
#   - Rute hasil model AI (v2)
#   - Rute hasil greedy murni (nearest neighbor)
#   - Selisih total jarak
#   - Avg confidence per kasus
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 0. LOAD MODEL
# ============================================================================

def load_model():
    required = ["models/tsp_model.pkl", "models/scaler.pkl",
                "models/feature_cols.pkl", "models/model_meta.pkl"]
    for path in required:
        if not os.path.exists(path):
            print(f"[ERROR] File tidak ditemukan: {path}")
            print("        Jalankan dulu generate_training_data.py + train_model.py")
            sys.exit(1)
    model     = joblib.load("models/tsp_model.pkl")
    scaler    = joblib.load("models/scaler.pkl")
    feat_cols = joblib.load("models/feature_cols.pkl")
    return model, scaler, feat_cols


# ============================================================================
# 1. KASUS UJI
# ============================================================================

TEST_CASES = [
    {
        "name": "10 Kota Indonesia (default app.py)",
        "cities": ["Jakarta","Surabaya","Bandung","Semarang","Medan",
                   "Makassar","Palembang","Yogyakarta","Malang","Denpasar"],
        "coords": np.array([
            [106.85,-6.21],[112.75,-7.25],[107.61,-6.91],[110.42,-6.99],
            [98.67,3.58],[119.43,-5.13],[104.76,-2.99],[110.36,-7.80],
            [112.63,-7.98],[115.22,-8.65],
        ]),
    },
    {
        "name": "8 Kota Grid Teratur (mudah)",
        "cities": ["A","B","C","D","E","F","G","H"],
        "coords": np.array([
            [0,0],[10,0],[20,0],[20,10],
            [20,20],[10,20],[0,20],[0,10],
        ]),
    },
    {
        "name": "8 Kota Cluster (2 cluster berjauhan)",
        "cities": ["C1a","C1b","C1c","C1d","C2a","C2b","C2c","C2d"],
        "coords": np.array([
            [0,0],[2,1],[1,3],[3,2],       # cluster kiri
            [20,0],[22,1],[21,3],[23,2],   # cluster kanan
        ]),
    },
    {
        "name": "12 Kota Acak Sedang",
        "cities": [f"K{i+1}" for i in range(12)],
        "coords": np.array([
            [15,72],[33,41],[58,85],[76,23],[42,67],[91,54],
            [27,18],[63,39],[84,71],[11,55],[49,12],[70,88],
        ]),
    },
    {
        "name": "15 Kota Acak Besar",
        "cities": [f"K{i+1}" for i in range(15)],
        "coords": np.array([
            [10,20],[30,80],[50,10],[70,60],[90,30],
            [20,50],[40,40],[60,90],[80,20],[15,70],
            [55,55],[75,75],[35,25],[65,45],[45,85],
        ]),
    },
    {
        "name": "10 Kota Zigzag (greedy sering salah)",
        "cities": [f"Z{i+1}" for i in range(10)],
        "coords": np.array([
            [0,0],[100,1],[1,2],[99,3],[2,4],
            [98,5],[3,6],[97,7],[4,8],[96,9],
        ]),
    },
]


# ============================================================================
# 2. GREEDY MURNI
# ============================================================================

def greedy_pure(coords, dist_matrix):
    n        = len(coords)
    visited  = [False] * n
    visited[0] = True
    route    = [0]
    total    = 0
    current  = 0
    for _ in range(n - 1):
        unvisited = [j for j in range(n) if not visited[j]]
        dists     = [dist_matrix[current][j] for j in unvisited]
        best      = unvisited[int(np.argmin(dists))]
        total    += dist_matrix[current][best]
        visited[best] = True
        route.append(best)
        current = best
    total += dist_matrix[current][0]
    route.append(0)
    return route, total


# ============================================================================
# 3. AI ROUTE (salin dari app.py)
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


def build_candidate_features(current, candidate, coords, dist_matrix,
                               visited, step_idx, feat_cols):
    n             = len(coords)
    unvisited_ids = [j for j in range(n) if not visited[j]]
    dists_unvis   = [dist_matrix[current][j] for j in unvisited_ids]
    n_unvis       = len(unvisited_ids)
    min_dist = min(dists_unvis) if dists_unvis else 0
    max_dist = max(dists_unvis) if dists_unvis else 0
    avg_dist = np.mean(dists_unvis) if dists_unvis else 0
    dist_val = dist_matrix[current][candidate]
    feat_dict = {
        'dist_to_candidate'  : dist_val,
        'dist_ratio_avg'     : dist_val / (avg_dist + 1e-9),
        'dist_sq'            : dist_val ** 2,
        'delta_x'            : coords[candidate][0] - coords[current][0],
        'delta_y'            : coords[candidate][1] - coords[current][1],
        'candidate_x'        : coords[candidate][0],
        'candidate_y'        : coords[candidate][1],
        'current_x'          : coords[current][0],
        'current_y'          : coords[current][1],
        'n_unvisited'        : n_unvis,
        'step_progress'      : step_idx / max(n - 1, 1),
        'avg_dist_unvisited' : avg_dist,
        'min_dist_unvisited' : min_dist,
        'max_dist_unvisited' : max_dist,
        'dist_std_unvisited' : np.std(dists_unvis) if dists_unvis else 0,
    }
    return np.array([feat_dict.get(col, 0.0) for col in feat_cols])


def ai_route(model, scaler, feat_cols, coords, dist_matrix):
    n          = len(coords)
    visited    = [False] * n
    visited[0] = True
    route      = [0]
    total      = 0
    confs      = []
    current    = 0
    for step_idx in range(n - 1):
        candidates  = [j for j in range(n) if not visited[j]]
        feat_matrix = np.array([
            build_candidate_features(current, c, coords, dist_matrix,
                                     visited, step_idx, feat_cols)
            for c in candidates
        ])
        feat_scaled = scaler.transform(feat_matrix)
        scores      = model.predict_proba(feat_scaled)[:, 1]
        best_idx    = int(np.argmax(scores))
        next_city   = candidates[best_idx]
        confs.append(float(scores[best_idx]))
        total      += dist_matrix[current][next_city]
        visited[next_city] = True
        route.append(next_city)
        current = next_city
    total += dist_matrix[current][0]
    route.append(0)
    return route, total, confs


# ============================================================================
# 4. JALANKAN SEMUA KASUS UJI + VISUALISASI
# ============================================================================

def run_all(model, scaler, feat_cols):
    os.makedirs("outputs", exist_ok=True)
    results = []

    print("\n" + "=" * 72)
    print(f"  {'Kasus Uji':<38} {'AI':>10} {'Greedy':>10} {'Selisih':>8}  {'Conf':>6}")
    print("=" * 72)

    for tc in TEST_CASES:
        name   = tc["name"]
        cities = tc["cities"]
        coords = tc["coords"]
        dist   = build_distance_matrix(coords)

        r_greedy, d_greedy = greedy_pure(coords, dist)
        r_ai, d_ai, confs  = ai_route(model, scaler, feat_cols, coords, dist)

        diff     = d_ai - d_greedy
        diff_pct = diff / d_greedy * 100
        avg_conf = np.mean(confs) * 100

        # tanda: AI lebih baik, sama, atau lebih buruk
        if abs(diff) < 0.001:
            sign = "="
        elif diff < 0:
            sign = "✓ AI lebih baik"
        else:
            sign = "✗ AI lebih buruk"

        print(f"  {name:<38} {d_ai:>10.2f} {d_greedy:>10.2f} "
              f"{diff_pct:>+7.1f}%  {avg_conf:>5.1f}%  {sign}")

        results.append({
            "name": name, "cities": cities, "coords": coords,
            "r_ai": r_ai, "d_ai": d_ai, "confs": confs,
            "r_greedy": r_greedy, "d_greedy": d_greedy,
            "diff_pct": diff_pct,
        })

    print("=" * 72)

    # Ringkasan
    ai_wins    = sum(1 for r in results if r["diff_pct"] < -0.001)
    ties       = sum(1 for r in results if abs(r["diff_pct"]) <= 0.001)
    ai_loses   = sum(1 for r in results if r["diff_pct"] > 0.001)
    avg_diff   = np.mean([r["diff_pct"] for r in results])
    print(f"\n  AI menang : {ai_wins}/{len(results)} kasus")
    print(f"  Seri      : {ties}/{len(results)} kasus")
    print(f"  AI kalah  : {ai_loses}/{len(results)} kasus")
    print(f"  Rata-rata selisih jarak: {avg_diff:+.2f}%")

    return results


# ============================================================================
# 5. PLOT PERBANDINGAN
# ============================================================================

def plot_comparison(results, save_path="outputs/test_compare.png"):
    n_cases = len(results)
    fig, axes = plt.subplots(n_cases, 2, figsize=(16, 4.5 * n_cases))
    fig.patch.set_facecolor('#f8f9fa')

    for row, r in enumerate(results):
        coords  = r["coords"]
        cities  = r["cities"]
        r_ai    = r["r_ai"]
        r_gr    = r["r_greedy"]
        d_ai    = r["d_ai"]
        d_gr    = r["d_greedy"]
        confs   = r["confs"]
        diff    = r["diff_pct"]

        for col, (route, label, dist_val) in enumerate([
            (r_ai,  f"AI  ({np.mean(confs)*100:.0f}% conf)", d_ai),
            (r_gr, "Greedy murni", d_gr),
        ]):
            ax = axes[row][col]
            ax.set_facecolor('#f8f9fa')

            # Gambar rute
            color = '#3498db' if col == 0 else '#e67e22'
            for i in range(len(route) - 1):
                s, t = route[i], route[i+1]
                ax.annotate("", xy=(coords[t][0], coords[t][1]),
                            xytext=(coords[s][0], coords[s][1]),
                            arrowprops=dict(arrowstyle='->', color=color,
                                           lw=1.6, mutation_scale=10))

            for i, (nm, (x, y)) in enumerate(zip(cities, coords)):
                is_start = (i == 0)
                ax.scatter(x, y,
                           c='#e74c3c' if is_start else color,
                           s=250 if is_start else 80,
                           zorder=5, edgecolors='white', linewidths=1.2,
                           marker='*' if is_start else 'o')
                order = route.index(i) + 1 if i in route[:-1] else '-'
                ax.annotate(
                    f"{'★' if is_start else str(order)}.{nm}",
                    (x, y), textcoords="offset points", xytext=(6, 5),
                    fontsize=7.5, color='#2c3e50',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.8, edgecolor='#bdc3c7', linewidth=0.6)
                )

            diff_str = f"  ({diff:+.1f}% vs greedy)" if col == 0 else ""
            ax.set_title(
                f"{r['name']}\n{label} | Jarak: {dist_val:.2f}{diff_str}",
                fontsize=9, fontweight='bold', color='#2c3e50'
            )
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.set_xlabel("X", fontsize=8)
            ax.set_ylabel("Y", fontsize=8)

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\n[OK] Visualisasi disimpan: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("  TSP: AI v2 vs Greedy Murni — Kasus Uji Lengkap")
    print("=" * 72)

    model, scaler, feat_cols = load_model()
    results = run_all(model, scaler, feat_cols)
    plot_comparison(results)

    print("\n[SELESAI] Hasil tersimpan di outputs/test_compare.png")