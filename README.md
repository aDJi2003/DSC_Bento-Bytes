# Dashboard Prediksi Keberhasilan UMKM — Tim Bento Bytes

Dashboard interaktif untuk babak final, dibangun dengan **Streamlit**.
Berjalan sepenuhnya secara lokal saat presentasi.

## Isi folder
```
dashboard/
├── app.py            # Aplikasi dashboard (yang dijalankan saat presentasi)
├── train.py          # Skrip pelatihan — dijalankan SEKALI untuk membuat artefak
├── utils.py          # Fungsi bersama (feature engineering, metadata fitur)
├── requirements.txt  # Daftar pustaka beserta versi terkunci
├── data/
│   └── umkm_success.csv
└── artifacts/        # Hasil train.py (model & metrik precomputed)
```

## Cara menjalankan (lokal)

### 1. Siapkan environment (sekali saja)
Disarankan memakai virtual environment agar bersih dan tidak bentrok dengan
paket lain di komputer Anda.

```bash
# masuk ke folder dashboard
cd dashboard

# buat & aktifkan virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# pasang semua pustaka
pip install -r requirements.txt
```

### 2. Latih model & buat artefak (sekali saja)
Langkah ini memindahkan semua komputasi berat (pelatihan, cross-validation,
SHAP) ke awal, sehingga dashboard berjalan instan.

```bash
python train.py
```
Akan terbentuk folder `artifacts/` berisi model dan metrik. Jika berhasil,
terminal menampilkan tabel perbandingan model di akhir.

### 3. Jalankan dashboard
```bash
streamlit run app.py
```
Browser akan otomatis terbuka di `http://localhost:8501`. Jika tidak,
buka alamat tersebut secara manual.

## Empat panel dashboard
1. **Prediktor Interaktif** — atur profil UMKM (slider & toggle), lihat
   probabilitas keberhasilan berubah secara langsung lewat gauge.
2. **Interpretasi** — koefisien model dan pengaruh SHAP; menjawab "mengapa
   model memprediksi demikian".
3. **Eksplorasi Data** — pola keberhasilan terhadap kesiapan manajerial dan
   fitur lain yang dapat dipilih.
4. **Performa Model** — confusion matrix, kurva ROC, slider ambang keputusan
   interaktif, dan tabel perbandingan model.

> Catatan: slider ambang di panel Performa juga memengaruhi keputusan di panel
> Prediktor Interaktif, sehingga juri dapat melihat dampak ambang secara langsung.

## Jika ingin memperbarui model
Cukup jalankan ulang `python train.py`, lalu `streamlit run app.py`.
Tidak perlu mengubah `app.py`.

## Troubleshooting
- **Konflik versi numpy/numba**: gunakan virtual environment baru dan pasang
  hanya dari `requirements.txt`. Hindari mencampur dengan environment Kaggle.
- **XGBoost/LightGBM tidak terpasang**: `train.py` otomatis melewatinya;
  perbandingan model tetap berjalan dengan model lain.
- **Port 8501 sudah dipakai**: jalankan `streamlit run app.py --server.port 8502`.
