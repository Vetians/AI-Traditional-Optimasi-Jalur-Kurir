import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import sys
import os

# Set page config for wide layout and title
st.set_page_config(page_title="AI TSP Solver", page_icon="AI", layout="wide")

# Custom CSS for Premium UI/UX
st.markdown("""
<style>
    /* Global background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Title styling */
    h1, h2, h3 {
        color: #00d2ff !important;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    h3 {
        color: #f8fafc !important;
        margin-top: 1.5rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5), 0 4px 6px -2px rgba(0, 0, 0, 0.3);
        border: 1px solid #334155;
        color: white;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #38bdf8;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #38bdf8;
        margin-top: 8px;
    }
    .metric-label {
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
        font-weight: 600;
    }
    
    /* Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        width: 100%;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 210, 255, 0.4);
    }
    div.stDownloadButton > button:first-child {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    
    /* Info/Warning Boxes */
    div[data-testid="stExpander"] {
        border-radius: 8px;
        border-color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# Ensure app.py can be imported correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import load_model, predict_route, build_distance_matrix
except ImportError:
    st.error("Gagal mengimpor modul app.py. Pastikan file berada di folder yang sama.")
    st.stop()

# --- MODEL LOADING ---
@st.cache_resource(show_spinner=False)
def get_model():
    """Load model with caching so it's not reloaded on every interaction"""
    try:
        return load_model()
    except SystemExit:
        return None, None, None, None
    except Exception as e:
        return None, None, None, None

model, scaler, feat_cols, meta = get_model()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/854/854929.png", width=80)
    st.title("Navigasi Menu")
    menu = st.radio(
        "",
        ("Prediksi TSP", "Tentang Aplikasi", "Profil Pengembang"),
        label_visibility="collapsed"
    )
    st.markdown("---")

# ============================================================================
# HALAMAN 1: PREDIKSI TSP (MAIN APP)
# ============================================================================
if menu == "Prediksi TSP":
    st.title("AI Traveling Salesman Problem")
    st.markdown("**Sistem Prediksi Rute Optimal Berbasis Machine Learning (Binary Classification)**")
    
    if model is None:
        st.error("Model belum dilatih atau file model tidak ditemukan!")
        st.info("Jalankan python generate_training_data.py lalu python train_model.py terlebih dahulu.")
        st.stop()

    with st.sidebar:
        st.header("Pengaturan Data")
        st.markdown("Pilih metode input untuk memasukkan data kota:")
        
        input_method = st.radio(
            "Metode Input:",
            ("Data Contoh (10 Kota)", "Unggah File CSV", "Input Manual"),
            label_visibility="collapsed"
        )
        
        df = None
        
        if input_method == "Data Contoh (10 Kota)":
            st.info("Menggunakan dataset bawaan 10 kota di Indonesia.")
            city_names = [
                "Jakarta", "Surabaya", "Bandung", "Semarang", "Medan",
                "Makassar", "Palembang", "Yogyakarta", "Malang", "Denpasar"
            ]
            coords = [
                [106.85, -6.21], [112.75, -7.25], [107.61, -6.91],
                [110.42, -6.99], [ 98.67,  3.58], [119.43, -5.13],
                [104.76, -2.99], [110.36, -7.80], [112.63, -7.98],
                [115.22, -8.65],
            ]
            df = pd.DataFrame({"Nama Kota": city_names, "X": [c[0] for c in coords], "Y": [c[1] for c in coords]})
            
        elif input_method == "Unggah File CSV":
            st.markdown("**Format CSV:** Harus punya kolom `Nama Kota`, `X`, dan `Y`.")
            uploaded_file = st.file_uploader("", type=["csv"])
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    # Validasi kolom dulu sebelum dropna agar tidak error
                    if not all(col in df.columns for col in ["Nama Kota", "X", "Y"]):
                        st.error("CSV tidak valid. Pastikan kolom 'Nama Kota', 'X', 'Y' tersedia.")
                        df = None
                    else:
                        # Baru bersihkan baris kosong setelah kolom dipastikan valid
                        df = df.dropna(subset=["Nama Kota", "X", "Y"])
                        st.success(f"{len(df)} kota berhasil dimuat dari file!")
                except Exception as e:
                    st.error(f"Gagal membaca file: {e}")
                    
        elif input_method == "Input Manual":
            st.markdown("Edit koordinat kota langsung di bawah ini:")
            default_df = pd.DataFrame({
                "Nama Kota": ["Kota A", "Kota B", "Kota C", "Kota D", "Kota E"],
                "X": [10.0, 25.0, 40.0, 50.0, 30.0],
                "Y": [20.0, 50.0, 10.0, 40.0, 60.0]
            })
            df = st.data_editor(default_df, num_rows="dynamic", hide_index=True)

        st.markdown("---")
        predict_btn = st.button("Prediksi Rute Optimal")

    # --- MAIN CONTENT AREA ---
    if df is not None:
        # Memastikan tidak ada row kosong dari editable table
        df = df.dropna(subset=["Nama Kota", "X", "Y"])
        
        with st.expander("Lihat Data Koordinat yang Digunakan", expanded=False):
            st.dataframe(df, use_container_width=True, hide_index=True)

        if predict_btn:
            city_names = df["Nama Kota"].astype(str).tolist()
            coords = df[["X", "Y"]].values
            
            # Handle nama ganda: hanya kota duplikat ke-2 dst yang diberi indeks angka
            seen = {}
            unique_city_names = []
            for name in city_names:
                if city_names.count(name) > 1:
                    seen[name] = seen.get(name, 0) + 1
                    unique_city_names.append(f"{name} ({seen[name]})")
                else:
                    unique_city_names.append(name)
            city_names = unique_city_names
            
            if len(city_names) < 3:
                st.warning("Minimal membutuhkan 3 kota untuk mencari rute TSP.")
            elif len(city_names) > 1000:
                st.warning("Aplikasi dibatasi maksimal 1000 kota agar performa visualisasi browser tetap stabil.")
            else:
                with st.spinner("Model AI sedang menganalisis dan merumuskan rute terbaik..."):
                    dist_matrix = build_distance_matrix(coords)
                    
                    start_time = time.time()
                    route, total_dist, step_details = predict_route(
                        model, scaler, feat_cols, coords, dist_matrix
                    )
                    elapsed_time = time.time() - start_time
                    avg_conf = np.mean([s['confidence'] for s in step_details]) * 100
                    
                    # Animasi jika confidence tinggi
                    if avg_conf > 85.0:
                        st.balloons()
                        
                    # --- METRICS ---
                    st.markdown("### Ringkasan Hasil AI")
                    c1, c2, c3, c4 = st.columns(4)
                    
                    with c1:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Jarak</div><div class="metric-value">{total_dist:.2f}</div></div>', unsafe_allow_html=True)
                    with c2:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Rata-rata Confidence</div><div class="metric-value">{avg_conf:.1f}%</div></div>', unsafe_allow_html=True)
                    with c3:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Waktu Eksekusi</div><div class="metric-value">{elapsed_time*1000:.0f} ms</div></div>', unsafe_allow_html=True)
                    with c4:
                        accuracy = meta.get("test_accuracy", 0) * 100
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Akurasi Model ML</div><div class="metric-value">{accuracy:.1f}%</div></div>', unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Penjelasan edukatif hasil metrik
                    st.info(f"Analisis: Model Machine Learning menyelesaikan pencarian rute dalam waktu **{elapsed_time*1000:.0f} ms**, jauh lebih cepat daripada menghitung manual. Dengan tingkat rata-rata keyakinan sebesar **{avg_conf:.1f}%**, model sangat yakin bahwa setiap keputusan langkah dari satu kota ke kota lainnya adalah rute yang paling efisien berdasarkan fitur jarak yang dipelajarinya.")
                    
                    # --- MAP VISUALIZATION ---
                    st.markdown("### Visualisasi Peta (Interaktif)")
                    
                    fig = go.Figure()
                    
                    route_x = [coords[i][0] for i in route]
                    route_y = [coords[i][1] for i in route]

                    # Add connections (Lines)
                    fig.add_trace(go.Scatter(
                        x=route_x, 
                        y=route_y,
                        mode='lines',
                        line=dict(color='rgba(0, 210, 255, 0.6)', width=2),
                        hoverinfo='skip',
                        name='Jalur'
                    ))
                    
                    # Add nodes with sequence numbers
                    for seq, node_idx in enumerate(route[:-1]):
                        x, y = coords[node_idx]
                        name = city_names[node_idx]
                        is_start = (seq == 0)
                        
                        marker_color = '#e74c3c' if is_start else '#3498db'
                        marker_size = 22 if is_start else 14
                        
                        fig.add_trace(go.Scatter(
                            x=[x], y=[y],
                            mode='markers+text',
                            marker=dict(
                                size=marker_size,
                                color=marker_color,
                                line=dict(width=2, color='white'),
                                symbol='star' if is_start else 'circle'
                            ),
                            text=[f"<b>{'Start' if is_start else str(seq+1)}</b>"],
                            textposition="middle center" if not is_start else "top center",
                            textfont=dict(color='white' if not is_start else '#e74c3c', size=10 if not is_start else 14),
                            hoverinfo='text',
                            hovertext=f"<b>{name}</b><br>Langkah ke-{seq+1}<br>X: {x}<br>Y: {y}",
                            name=name
                        ))
                    
                    # Draw arrows to indicate direction of route
                    for i in range(len(route) - 1):
                        src_idx = route[i]
                        dst_idx = route[i+1]
                        
                        dx = coords[dst_idx][0] - coords[src_idx][0]
                        dy = coords[dst_idx][1] - coords[src_idx][1]
                        
                        # Hindari menggambar panah jika dua kota berada di koordinat yang persis sama
                        if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                            fig.add_annotation(
                                x=coords[dst_idx][0] - dx*0.1, 
                                y=coords[dst_idx][1] - dy*0.1,
                                ax=coords[src_idx][0] + dx*0.1, 
                                ay=coords[src_idx][1] + dy*0.1,
                                xref='x', yref='y', axref='x', ayref='y',
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1.5,
                                arrowwidth=1.5,
                                arrowcolor='rgba(0, 210, 255, 0.8)'
                            )

                    fig.update_layout(
                        plot_bgcolor='#1e293b',
                        paper_bgcolor='#0e1117',
                        margin=dict(l=10, r=10, t=10, b=10),
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=True, gridcolor='#334155', zeroline=False),
                        yaxis=dict(showgrid=True, gridcolor='#334155', zeroline=False),
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # --- STEP DETAILS & EXPORT ---
                    st.markdown("### Log Keputusan AI (Detail Langkah)")
                    
                    table_data = []
                    for s in step_details:
                        table_data.append({
                            "Langkah Ke-": s["step"] + 1,
                            "Kota Asal": city_names[s["from_city"]],
                            "Kota Tujuan": city_names[s["to_city"]],
                            "Jarak": round(s["distance"], 3),
                            "AI Confidence (%)": round(s['confidence']*100, 2),
                            "Kandidat Tersisa": s["n_candidates"]
                        })
                    
                    result_df = pd.DataFrame(table_data)
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    
                    # Tombol Download hasil rute ke CSV
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Data Tabel (CSV)",
                        data=csv,
                        file_name='hasil_rute_tsp.csv',
                        mime='text/csv',
                    )

# ============================================================================
# HALAMAN 2: TENTANG APLIKASI
# ============================================================================
elif menu == "Tentang Aplikasi":
    st.title("Tentang AI TSP Solver")
    
    st.markdown("""
    ### Apa itu Traveling Salesman Problem (TSP)?
    **Traveling Salesman Problem** atau TSP adalah salah satu permasalahan optimasi klasik di dunia ilmu komputer. 
    Bayangkan seorang *salesman* (agen penjualan) yang harus mengunjungi sebuah daftar kota. Aturannya:
    1. Dia harus mengunjungi **setiap kota tepat satu kali**.
    2. Dia harus kembali ke **kota tempat dia memulai**.
    Tantangannya adalah menemukan rute/jalur manakah yang menghasilkan **jarak total paling pendek**.

    Meskipun terdengar sederhana, secara matematis, mencari jalur paling optimal sangatlah rumit (berstatus *NP-Hard*). Jika Anda memiliki 10 kota, ada ratusan ribu kemungkinan rute. Namun jika Anda memiliki 50 kota, jumlah kemungkinannya jauh melampaui jumlah atom di alam semesta!
    
    ---
    
    ### Bagaimana Website Ini Bekerja Menggunakan AI?
    Alih-alih mencoba semua kemungkinan (*Brute Force*), website ini menggunakan pendekatan cerdas melalui **Machine Learning**.
    
    Kita telah melatih sebuah model **Random Forest Classifier (Binary Classification)**. Algoritma ini meniru insting pengambilan keputusan (Imitation Learning). Di setiap titik pemberhentian, AI akan menganalisis sisa kota yang belum dikunjungi. Ia akan memberikan *score* (nilai probabilitas / *confidence*) ke semua sisa kota tersebut berdasarkan fitur-fiturnya (jarak, posisi, dsb), dan memutuskan: *"Kota A adalah kandidat langkah terbaik selanjutnya"*.
    
    Hasilnya? Model ini mampu menyusun rute untuk puluhan kota secara instan hanya dalam hitungan milidetik, tanpa kehabisan tenaga komputasi!
    """)
    
    st.info("Catatan Teknis: Model yang digunakan di aplikasi web ini adalah model Random Forest statis yang disesuaikan untuk berjalan efisien dalam browser.")

# ============================================================================
# HALAMAN 3: PROFIL PENGEMBANG
# ============================================================================
elif menu == "Profil Pengembang":
    st.title("Tim Pengembang")
    st.markdown("Aplikasi AI TSP Solver ini dikembangkan oleh tim yang berdedikasi di bidang *Artificial Intelligence* dan Rekayasa Perangkat Lunak.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image("outputs/tian.jpeg", width=120)
        st.markdown("### Christian Melhan 412024003")
        st.markdown("**AI & ML Engineer**")
        st.write("Fokus pada pemodelan *Machine Learning*, pembuatan fitur (Feature Engineering), dan *tuning* algoritma Random Forest.")
        
    with col2:
        st.image("outputs/felix.jpeg", width=120)
        st.markdown("### Felix Wong 412024004")
        st.markdown("**Data Scientist**")
        st.write("Bertanggung jawab atas pengumpulan dataset, analisis jarak spasial, dan validasi *Distance Matrix* TSP.")
        
    with col3:
        st.image("outputs/amka.jpeg", width=120)
        st.markdown("### Krisantus Amka Ginting 412024025")
        st.markdown("**Frontend / UI Developer**")
        st.write("Merancang arsitektur aplikasi *Streamlit*, memoles antarmuka UI/UX interaktif, dan integrasi Plotly.")
