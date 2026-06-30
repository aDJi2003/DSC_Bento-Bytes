# Dashboard Prediksi Keberhasilan UMKM

Aplikasi web interaktif untuk memprediksi keberhasilan Usaha Mikro, Kecil, dan Menengah (UMKM) sekaligus memvisualisasikan cara kerja model prediksi bagi pengambil keputusan. Dibangun oleh Tim Bento Bytes untuk Lomba Data Science.

Dashboard ini bukan sekadar antarmuka prediksi. Ia dirancang sebagai alat eksplorasi yang transparan: pengguna dapat menyusun profil UMKM, melihat probabilitas keberhasilannya secara langsung, memahami alasan di balik setiap prediksi, menelaah pola dalam data, dan mengevaluasi performa model pada berbagai ambang keputusan.

## Daftar Isi

1. [Akses Cepat](#akses-cepat)
2. [Gambaran Umum](#gambaran-umum)
3. [Fitur](#fitur)
4. [Arsitektur dan Desain](#arsitektur-dan-desain)
5. [Struktur Proyek](#struktur-proyek)
6. [Tentang Model](#tentang-model)
7. [Menjalankan Secara Lokal](#menjalankan-secara-lokal)
8. [Deployment](#deployment)
9. [Alur Pembaruan Model](#alur-pembaruan-model)
10. [Pemecahan Masalah](#pemecahan-masalah)
11. [Catatan Teknis](#catatan-teknis)
12. [Lisensi dan Atribusi](#lisensi-dan-atribusi)

## Akses Cepat

Dashboard sudah ter-deploy dan dapat diakses langsung tanpa instalasi apa pun:

**https://bento-bytes-dashboard.streamlit.app/**

Tautan tersebut dapat dibuka pada peramban mana pun, baik di komputer maupun ponsel. Tidak diperlukan akun atau konfigurasi tambahan.

Jika Anda ingin menjalankan, memodifikasi, atau mempelajari kode secara lokal, lihat bagian [Menjalankan Secara Lokal](#menjalankan-secara-lokal).

## Gambaran Umum

Proyek ini menjawab satu pertanyaan praktis: faktor apa yang menentukan keberhasilan sebuah UMKM, dan dapatkah keberhasilan tersebut diprediksi untuk mengarahkan program pembinaan yang tepat sasaran.

Model prediksi dibangun dari dataset berisi 250 UMKM dengan 12 fitur prediktor. Tantangan utamanya ada dua: kelas yang timpang (hanya sekitar 24,8 persen UMKM tergolong berhasil) dan ukuran sampel yang kecil. Kedua tantangan ini memandu seluruh keputusan teknis, mulai dari pemilihan metrik evaluasi, strategi validasi, hingga pemilihan model akhir.

Dashboard ini adalah lapisan presentasi dari hasil tersebut. Tujuannya membuat model dapat dipahami dan dipercaya, bukan sekadar menghasilkan angka. Setiap prediksi dapat dijelaskan, setiap pola dalam data dapat ditelusuri, dan setiap konsekuensi dari pengaturan ambang keputusan dapat dilihat secara langsung.

## Fitur

Dashboard terdiri dari empat panel utama yang dapat diakses melalui tab, dengan satu kontrol ambang keputusan global yang ditempatkan di sidebar.

### Panel Prediktor Interaktif

Pengguna menyusun profil UMKM melalui slider dan sakelar, atau memuat salah satu skenario contoh yang sudah disiapkan (UMKM berpotensi, UMKM berisiko, dan kasus ambigu). Probabilitas keberhasilan ditampilkan secara langsung melalui gauge, lengkap dengan verdict berhasil atau tidak berhasil berdasarkan ambang yang sedang aktif. Di bawahnya, sebuah grafik kontribusi fitur menjelaskan faktor mana yang mendorong prediksi ke arah berhasil dan mana yang menariknya ke arah sebaliknya, khusus untuk profil yang sedang ditampilkan.

### Panel Interpretasi

Panel ini menjawab pertanyaan mengapa model memprediksi seperti itu, pada tingkat global. Ditampilkan koefisien model final beserta pengaruh setiap fitur, serta ringkasan nilai SHAP yang mengukur kontribusi tiap fitur terhadap keseluruhan prediksi. Panel ini menegaskan bahwa model bukan kotak hitam.

### Panel Eksplorasi Data

Panel ini menyajikan pola dalam data secara interaktif. Pengguna dapat melihat bagaimana tingkat keberhasilan meningkat seiring naiknya skor kesiapan manajerial, serta menjelajahi hubungan antara keberhasilan dan fitur lain yang dapat dipilih sendiri.

### Panel Performa Model

Panel ini menampilkan confusion matrix, kurva ROC, metrik precision, recall, dan F1, serta tabel perbandingan beberapa model. Seluruh metrik dihitung dari prediksi out-of-fold agar mencerminkan performa yang jujur, bukan performa pada data latih.

### Kontrol Ambang Keputusan Global

Slider ambang keputusan ditempatkan di sidebar sehingga selalu terlihat di tab mana pun. Ketika digeser, efeknya langsung tampak: di panel Prediktor, garis ambang pada gauge bergeser dan verdict dapat berbalik; di panel Performa, confusion matrix beserta metrik precision dan recall ikut berubah seketika. Desain ini memungkinkan pengguna memahami trade-off antara menjangkau lebih banyak UMKM berpotensi (recall tinggi) dan intervensi yang lebih tepat sasaran (precision tinggi) tanpa perlu melatih ulang model.

## Arsitektur dan Desain

Keputusan arsitektur terpenting pada proyek ini adalah pemisahan tegas antara tahap pelatihan dan tahap penyajian.

Notebook analisis melatih dan mengevaluasi model setiap kali dijalankan. Pendekatan tersebut tidak cocok untuk aplikasi web yang harus responsif, karena pengguna tidak boleh menunggu proses cross-validation atau perhitungan SHAP setiap kali menggeser slider. Oleh sebab itu, seluruh komputasi berat dipindahkan ke sebuah skrip pelatihan terpisah yang dijalankan satu kali. Hasilnya, berupa model terlatih beserta seluruh artefak pendukung, disimpan ke disk. Aplikasi dashboard kemudian hanya memuat artefak jadi tersebut dan menggunakannya untuk prediksi instan.

Pemisahan ini diwujudkan melalui tiga berkas Python:

- `train.py` berisi seluruh logika pelatihan. Ia memuat data, membangun fitur, menyetel hyperparameter melalui cross-validation, melatih model final pada seluruh data, menghitung metrik evaluasi dan nilai SHAP, lalu menyimpan semuanya ke folder `artifacts`.
- `app.py` berisi aplikasi dashboard. Ia memuat artefak hasil `train.py` dan menyusun antarmuka interaktif. Berkas ini tidak melatih apa pun.
- `utils.py` berisi fungsi yang dipakai bersama oleh keduanya, terutama logika rekayasa fitur. Menempatkan logika ini di satu tempat memastikan input yang diterima dashboard diperlakukan persis sama seperti saat pelatihan, sehingga mencegah inkonsistensi antara pelatihan dan penyajian (training-serving skew).

Aplikasi memanfaatkan mekanisme caching bawaan Streamlit. Pemuatan model di-cache sebagai resource, sedangkan pemuatan data di-cache sebagai data, sehingga keduanya tidak dimuat ulang setiap kali pengguna berinteraksi.

## Struktur Proyek

```
dashboard/
├── app.py              Aplikasi dashboard Streamlit (titik masuk utama)
├── train.py            Skrip pelatihan, menghasilkan seluruh artefak
├── utils.py            Fungsi bersama: rekayasa fitur dan metadata fitur
├── requirements.txt    Daftar dependensi dengan versi terkunci
├── data/
│   └── umkm_success.csv    Dataset mentah
└── artifacts/          Keluaran train.py (tidak diedit manual)
    ├── model.joblib        Pipeline model final (scaler dan estimator)
    ├── metrics.joblib      Metrik evaluasi dan hasil cross-validation
    ├── data_fe.joblib      Dataset dengan fitur hasil rekayasa
    ├── shap.joblib         Nilai SHAP yang sudah dihitung
    └── coefs.joblib        Koefisien model untuk interpretasi
```

## Tentang Model

Model final adalah Logistic Regression dengan regularisasi L1. Model ini dipilih bukan karena kesederhanaannya semata, melainkan karena terbukti unggul melalui perbandingan sistematis terhadap enam algoritma menggunakan stratified 5-fold cross-validation.

Ringkasan keputusan dan hasil:

- Algoritma yang dibandingkan meliputi Logistic Regression, LightGBM, Deep Learning (MLP), XGBoost, Random Forest, dan Decision Tree. Logistic Regression menempati posisi teratas berdasarkan F1-score.
- Hyperparameter yang disetel melalui RandomizedSearchCV adalah kekuatan regularisasi (C), jenis penalti (L1 atau L2), dan solver. Konfigurasi terbaik adalah penalti L1 dengan solver liblinear dan C sekitar 1,27.
- Ketidakseimbangan kelas ditangani dengan pembobotan kelas (class weighting), bukan oversampling sintetis, dengan pertimbangan bahwa pada data kecil, sampel sintetis berisiko menghasilkan pola palsu.
- Performa akhir: F1-score 0,919 pada cross-validation, ROC-AUC 0,993, dan recall kelas berhasil 0,984.

Prediktor terkuat adalah indeks Management_Readiness, yaitu fitur hasil rekayasa yang menggabungkan empat praktik manajerial: kecukupan modal awal, pencatatan keuangan, penggunaan internet, dan kepemilikan rencana bisnis. Indeks ini memiliki korelasi 0,67 dengan keberhasilan, jauh lebih kuat dibanding fitur mana pun secara individual. Studi ablasi mengonfirmasi bahwa menghapus fitur manajerial menurunkan F1 secara drastis, sedangkan menghapus fitur demografis nyaris tidak berpengaruh.

## Menjalankan Secara Lokal

Bagian ini ditujukan bagi siapa pun yang ingin menjalankan, memodifikasi, atau mempelajari kode di komputer sendiri. Untuk sekadar menggunakan dashboard, cukup akses tautan pada bagian [Akses Cepat](#akses-cepat).

### Prasyarat

- Python versi 3.9 atau lebih baru.
- Git, jika Anda mengkloning dari repository.

### Langkah 1: Dapatkan kode

Kloning repository, lalu masuk ke folder proyek:

```bash
git clone https://github.com/aDJi2003/DSC_Bento-Bytes.git
cd DSC_Bento-Bytes
```

### Langkah 2: Siapkan virtual environment

Penggunaan virtual environment sangat disarankan agar dependensi proyek tidak bercampur dengan instalasi Python global di komputer Anda.

```bash
python -m venv venv
```

Aktifkan environment. Pada Windows:

```bash
venv\Scripts\activate
```

Pada macOS atau Linux:

```bash
source venv/bin/activate
```

### Langkah 3: Pasang dependensi

```bash
pip install -r requirements.txt
```

Berkas `requirements.txt` mengunci rentang versi setiap pustaka untuk menjaga kompatibilitas. Pustaka XGBoost dan LightGBM bersifat opsional; jika tidak terpasang, skrip pelatihan akan melewatinya secara otomatis tanpa menimbulkan error.

### Langkah 4: Latih model dan hasilkan artefak

```bash
python train.py
```

Langkah ini menjalankan seluruh komputasi berat satu kali dan menyimpan hasilnya ke folder `artifacts`. Jika berhasil, terminal akan menampilkan tabel perbandingan model di akhir proses. Langkah ini wajib dijalankan sebelum dashboard dapat berfungsi, karena dashboard memuat artefak yang dihasilkannya.

Penting: artefak harus dibuat oleh versi scikit-learn yang sama dengan yang akan menjalankan dashboard. Jika Anda memuat artefak yang dilatih pada versi scikit-learn berbeda, dapat muncul error kompatibilitas. Melatih ulang melalui langkah ini menyelesaikan masalah tersebut.

### Langkah 5: Jalankan dashboard

```bash
streamlit run app.py
```

Peramban akan terbuka otomatis pada `http://localhost:8501`. Jika tidak, buka alamat tersebut secara manual. Untuk menghentikan aplikasi, tekan Ctrl+C pada terminal.

## Deployment

Dashboard di-deploy menggunakan Streamlit Community Cloud yang terhubung langsung ke repository GitHub. Alur kerjanya bersifat continuous deployment: setiap perubahan yang di-push ke branch yang dipantau akan memicu Streamlit Cloud membangun dan menerbitkan ulang aplikasi secara otomatis.

### Cara kerja deployment

1. Kode proyek, termasuk folder `artifacts` yang berisi model terlatih, disimpan dalam repository GitHub.
2. Aplikasi di Streamlit Community Cloud dikonfigurasi untuk menunjuk ke repository tersebut, dengan `app.py` sebagai berkas utama.
3. Streamlit Cloud membaca `requirements.txt` untuk memasang seluruh dependensi pada lingkungan server.
4. Setiap commit baru ke branch yang dipantau memicu pembangunan ulang otomatis. Tidak ada langkah manual yang diperlukan untuk memperbarui versi yang sudah tayang.

### Hal yang perlu diperhatikan saat deployment

- Folder `artifacts` harus ikut di-commit ke repository. Streamlit Cloud tidak menjalankan `train.py`, sehingga aplikasi bergantung sepenuhnya pada artefak yang sudah ada di repository. Apabila artefak tidak disertakan, aplikasi akan gagal memuat model.
- Versi pustaka pada `requirements.txt` harus konsisten dengan versi yang dipakai saat menghasilkan artefak, terutama scikit-learn, untuk menghindari error kompatibilitas saat memuat model di server.
- Ukuran berkas artefak pada proyek ini kecil, sehingga aman disimpan langsung di repository tanpa memerlukan Git LFS.

## Alur Pembaruan Model

Jika model perlu diperbarui, misalnya setelah menambah data atau mengubah fitur, ikuti alur berikut.

1. Lakukan perubahan pada `train.py` atau `utils.py` sesuai kebutuhan.
2. Jalankan kembali `python train.py` di lingkungan lokal untuk menghasilkan artefak yang baru.
3. Verifikasi dashboard secara lokal dengan `streamlit run app.py` untuk memastikan semuanya berfungsi.
4. Commit perubahan kode beserta folder `artifacts` yang telah diperbarui.
5. Push ke branch yang dipantau Streamlit Cloud. Aplikasi yang tayang akan diperbarui secara otomatis.

Berkas `app.py` umumnya tidak perlu diubah saat memperbarui model, karena ia hanya membaca artefak.

## Pemecahan Masalah

Berikut beberapa kendala yang umum dijumpai beserta penyelesaiannya.

### Error saat memuat model terkait atribut yang hilang

Gejala ini biasanya muncul dalam bentuk pesan mengenai atribut yang tidak ditemukan pada objek model. Penyebabnya adalah perbedaan versi scikit-learn antara saat artefak dibuat dan saat artefak dimuat. Penyelesaiannya adalah melatih ulang model dengan menjalankan `python train.py` pada lingkungan yang sama dengan yang menjalankan dashboard, sehingga artefak dan pemuat berada pada versi yang konsisten.

### Konflik versi numpy atau numba

Gejala ini berupa peringatan atau error terkait versi numpy yang tidak kompatibel dengan pustaka lain. Penyelesaiannya adalah menggunakan virtual environment yang bersih dan memasang dependensi hanya melalui `requirements.txt`, yang sudah mengunci numpy pada versi di bawah 2.4. Hindari mencampur lingkungan proyek ini dengan lingkungan lain yang memasang versi numpy berbeda.

### XGBoost atau LightGBM tidak terpasang

Jika kedua pustaka opsional ini tidak tersedia, `train.py` akan melewatinya secara otomatis dan perbandingan model tetap berjalan dengan algoritma lain. Tidak diperlukan tindakan khusus, kecuali Anda memang ingin menyertakan keduanya dalam perbandingan.

### Port 8501 sudah digunakan

Jika port default Streamlit sedang dipakai proses lain, jalankan dashboard pada port berbeda:

```bash
streamlit run app.py --server.port 8502
```

### Aplikasi yang tayang tidak ikut berubah setelah push

Pastikan perubahan di-push ke branch yang benar, yaitu branch yang dipantau oleh konfigurasi aplikasi di Streamlit Community Cloud. Periksa juga log pembangunan pada dasbor pengelolaan aplikasi di Streamlit Cloud untuk melihat apakah terjadi error saat proses build.

## Catatan Teknis

- Bahasa dan kerangka kerja: Python dengan Streamlit sebagai kerangka aplikasi web.
- Pustaka utama: scikit-learn untuk pemodelan, pandas dan numpy untuk pengolahan data, Plotly untuk visualisasi interaktif, SHAP untuk interpretasi model, serta joblib untuk serialisasi artefak.
- Strategi versi: seluruh dependensi dikunci pada rentang versi tertentu di `requirements.txt` untuk memastikan reprodusibilitas dan menghindari konflik dependensi.
- Reprodusibilitas: random seed ditetapkan pada proses pelatihan agar hasil dapat direproduksi.
- Visualisasi: seluruh grafik bersifat interaktif dan dihasilkan secara dinamis dari artefak, bukan gambar statis.

## Lisensi dan Atribusi

Proyek ini dikembangkan oleh Tim Bento Bytes sebagai bagian dari partisipasi pada Lomba Data Science. Dataset dan ketentuan penggunaannya mengikuti aturan penyelenggara lomba.
