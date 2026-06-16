"""
utils.py — Fungsi bersama untuk pelatihan (train.py) dan dashboard (app.py).

Menaruh feature engineering di satu tempat memastikan input dashboard
diperlakukan PERSIS sama seperti saat pelatihan (mencegah training-serving skew).
"""
import pandas as pd

# Kolom penyusun indeks kesiapan manajerial
MGMT_COLS = ["Initial_Capital", "Financial_Record_Keeping",
             "Internet_Usage", "Business_Plan"]

# Confounder untuk analisis kausal (dipakai opsional di dashboard)
CONFOUNDERS = ["Education", "Industry_Experience", "Professional_Advice",
               "Parent_Business_Experience", "Age"]

TARGET = "Success"

# Metadata fitur untuk membangun kontrol input di dashboard
# (label, tipe, rentang/min, max, nilai default)
FEATURE_META = {
    "Age":                        {"label": "Usia pemilik (tahun)",        "type": "int",  "min": 18, "max": 65, "default": 35},
    "Education":                  {"label": "Pendidikan (1=SD ... 5=S2+)", "type": "int",  "min": 1,  "max": 5,  "default": 3},
    "Initial_Capital":            {"label": "Modal awal memadai",          "type": "bool", "default": 1},
    "Financial_Record_Keeping":   {"label": "Pencatatan keuangan baik",    "type": "bool", "default": 1},
    "Internet_Usage":             {"label": "Menggunakan internet",        "type": "bool", "default": 1},
    "Business_Plan":              {"label": "Memiliki rencana bisnis",     "type": "bool", "default": 1},
    "Marketing_Effort":           {"label": "Upaya pemasaran (1-7)",       "type": "int",  "min": 1,  "max": 7,  "default": 4},
    "Partnership":                {"label": "Memiliki kemitraan",          "type": "bool", "default": 0},
    "Parent_Business_Experience": {"label": "Pengalaman bisnis orang tua", "type": "bool", "default": 0},
    "Industry_Experience":        {"label": "Pengalaman industri (tahun)", "type": "int",  "min": 0,  "max": 30, "default": 5},
    "Owner_Gender":               {"label": "Jenis kelamin (1=L, 0=P)",    "type": "bool", "default": 1},
    "Professional_Advice":        {"label": "Konsultasi profesional (1-7)","type": "int",  "min": 1,  "max": 7,  "default": 3},
}

RAW_FEATURES = list(FEATURE_META.keys())


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Membuat 4 fitur turunan. Identik dengan notebook kompetisi."""
    d = data.copy()
    d["Management_Readiness"] = d[MGMT_COLS].sum(axis=1)
    d["Experience_Ratio"]     = d["Industry_Experience"] / (d["Age"] - 17)
    d["Advice_x_Plan"]        = d["Professional_Advice"] * d["Business_Plan"]
    d["Digital_Marketing"]    = d["Internet_Usage"] * d["Marketing_Effort"]
    return d


def load_raw(path: str = "data/umkm_success.csv") -> pd.DataFrame:
    """Memuat dataset mentah."""
    return pd.read_csv(path)
