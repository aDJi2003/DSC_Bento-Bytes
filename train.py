"""
train.py — Dijalankan SEKALI untuk menghasilkan artefak yang dipakai dashboard.

Memindahkan semua komputasi berat (pelatihan, CV, SHAP) ke sini, sehingga
app.py cukup memuat file hasil dan berjalan instan.

Jalankan:  python train.py
Output:    artifacts/  (model.joblib, metrics.joblib, dll.)
"""
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     cross_val_predict, RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                             precision_recall_curve, average_precision_score)

from utils import (load_raw, engineer_features, RAW_FEATURES, TARGET,
                   MGMT_COLS)

SEED = 42
np.random.seed(SEED)
ART = "artifacts"

import os
os.makedirs(ART, exist_ok=True)


def main():
    print("1/6  Memuat & menyiapkan data ...")
    raw = load_raw()
    df = engineer_features(raw)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    print("2/6  Menyetel model final (RandomizedSearchCV) ...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(class_weight="balanced",
                                     max_iter=2000, random_state=SEED)),
    ])
    search = RandomizedSearchCV(
        pipe,
        {"model__C": np.logspace(-3, 2, 30),
         "model__penalty": ["l1", "l2"],
         "model__solver": ["liblinear"]},
        n_iter=30, scoring="f1", cv=cv, random_state=SEED, n_jobs=-1)
    search.fit(X, y)
    best = search.best_estimator_
    print(f"     Best params: {search.best_params_} | F1(CV)={search.best_score_:.3f}")

    print("3/6  Menghitung prediksi out-of-fold & metrik ...")
    oof_proba = cross_val_predict(best, X, y, cv=cv, method="predict_proba")[:, 1]
    oof_pred = (oof_proba >= 0.5).astype(int)
    cm = confusion_matrix(y, oof_pred)
    fpr, tpr, _ = roc_curve(y, oof_proba)
    prec, rec, pr_thr = precision_recall_curve(y, oof_proba)

    # perbandingan beberapa model untuk panel performa
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    models = {
        "Logistic Regression": best,
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier
        spw = (y == 0).sum() / (y == 1).sum()
        models["XGBoost"] = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                          scale_pos_weight=spw, eval_metric="logloss", random_state=SEED)
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                                            class_weight="balanced", random_state=SEED, verbose=-1)
    except Exception:
        pass

    comparison = []
    for name, m in models.items():
        r = cross_validate(m, X, y, cv=cv,
                           scoring={"f1": "f1", "roc_auc": "roc_auc", "bal_acc": "balanced_accuracy"})
        comparison.append({"Model": name,
                           "F1": r["test_f1"].mean(),
                           "ROC_AUC": r["test_roc_auc"].mean(),
                           "Bal_Acc": r["test_bal_acc"].mean()})
    comparison = pd.DataFrame(comparison).sort_values("F1", ascending=False)

    print("4/6  Melatih model final pada SELURUH data ...")
    best.fit(X, y)

    print("5/6  Menghitung SHAP (linear explainer) ...")
    try:
        import shap
        Xt = best.named_steps["scaler"].transform(X)
        explainer = shap.LinearExplainer(best.named_steps["model"], Xt,
                                         feature_names=list(X.columns))
        shap_values = explainer(Xt)
        shap_payload = {"values": shap_values.values,
                        "base": np.array(shap_values.base_values),
                        "feature_names": list(X.columns),
                        "X_std": Xt}
    except Exception as e:
        print(f"     SHAP dilewati: {e}")
        shap_payload = None

    print("6/6  Menyimpan artefak ...")
    joblib.dump(best, f"{ART}/model.joblib")
    joblib.dump({
        "oof_proba": oof_proba, "oof_pred": oof_pred,
        "y_true": y.values, "cm": cm,
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc_score(y, oof_proba),
        "prec": prec, "rec": rec, "pr_thr": pr_thr,
        "avg_precision": average_precision_score(y, oof_proba),
        "comparison": comparison,
        "best_params": search.best_params_,
        "cv_f1": search.best_score_,
    }, f"{ART}/metrics.joblib")
    joblib.dump(df, f"{ART}/data_fe.joblib")
    if shap_payload is not None:
        joblib.dump(shap_payload, f"{ART}/shap.joblib")

    # importance koefisien (untuk interpretasi global sederhana)
    coefs = pd.Series(best.named_steps["model"].coef_[0],
                      index=X.columns).sort_values()
    joblib.dump(coefs, f"{ART}/coefs.joblib")

    print("\nSelesai. Artefak tersimpan di folder 'artifacts/'.")
    print(comparison.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
