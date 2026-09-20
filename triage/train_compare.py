"""Week 3: compare anomaly detectors on UNSW-NB15 (trained on normal traffic only).

Usage:
    python triage/train_compare.py
    python triage/train_compare.py --attack-frac 0.05 --slow-n 10000
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

DATA = Path("data/unsw")
# Fields Suricata flow events can provide or derive:
#   dur~age, spkts/dpkts~pkts_toserver/toclient, sbytes/dbytes~bytes_toserver/toclient,
#   rate, sload, dload, smean, dmean are derived from those.
PORTABLE_COLS = ["dur", "spkts", "dpkts", "sbytes", "dbytes",
                 "rate", "sload", "dload", "smean", "dmean"]
NON_FEATURE = {"id", "label", "attack_cat", "proto", "service", "state"}


def add_proto(out: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    proto = df["proto"].astype(str).str.lower()
    out["proto_tcp"] = (proto == "tcp").astype(float)
    out["proto_udp"] = (proto == "udp").astype(float)
    out["proto_other"] = (~proto.isin(["tcp", "udp"])).astype(float)
    return out


def make_features(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = np.log1p(df[cols].astype(float).clip(lower=0))
    return add_proto(out, df)


def run_model(name, make_model, X_fit, X_val, X_te, y_te, max_fit, rng):
    if max_fit and len(X_fit) > max_fit:
        X_fit = X_fit[rng.choice(len(X_fit), max_fit, replace=False)]
    model = make_model()

    t0 = time.perf_counter()
    model.fit(X_fit)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    s_te = -model.score_samples(X_te)      # higher = more anomalous
    pred_s = time.perf_counter() - t0
    s_val = -model.score_samples(X_val)    # held-out NORMAL data -> threshold

    thr = np.percentile(s_val, 95)         # target ~5% false positive rate
    y_hat = (s_te > thr).astype(int)
    fpr = ((y_hat == 1) & (y_te == 0)).sum() / max((y_te == 0).sum(), 1)
    row = {
        "model": name, "fit_s": fit_s, "predict_s": pred_s,
        "roc_auc": roc_auc_score(y_te, s_te),
        "pr_auc": average_precision_score(y_te, s_te),
        "precision": precision_score(y_te, y_hat, zero_division=0),
        "recall": recall_score(y_te, y_hat, zero_division=0),
        "f1": f1_score(y_te, y_hat, zero_division=0),
        "fpr": fpr,
    }
    return row, model, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slow-n", type=int, default=10000,
                    help="max training rows for LOF / One-Class SVM (they scale badly)")
    ap.add_argument("--attack-frac", type=float, default=None,
                    help="subsample attacks in the test set to this share (e.g. 0.05)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    train = pd.read_csv(DATA / "UNSW_NB15_training-set.csv")
    test = pd.read_csv(DATA / "UNSW_NB15_testing-set.csv")
    if a.attack_frac:
        norm = test[test["label"] == 0]
        att_all = test[test["label"] == 1]
        n_att = int(a.attack_frac / (1 - a.attack_frac) * len(norm))
        att = att_all.sample(n=min(n_att, len(att_all)), random_state=a.seed)
        test = pd.concat([norm, att]).sample(frac=1, random_state=a.seed)

    print(f"train: {train.shape}, test: {test.shape}, "
          f"test attack share: {test['label'].mean():.2%}")

    normal = train[train["label"] == 0].sample(frac=1, random_state=a.seed)
    n_val = int(0.2 * len(normal))
    val_df, fit_df = normal.iloc[:n_val], normal.iloc[n_val:]
    print(f"normal train rows: fit={len(fit_df)}, val={len(val_df)}")

    full_cols = [c for c in train.columns if c not in NON_FEATURE]
    feature_sets = {"portable": PORTABLE_COLS, "full": full_cols}
    models = {
        "IsolationForest": (lambda: IsolationForest(
            n_estimators=200, random_state=a.seed, n_jobs=-1), None),
        "LOF": (lambda: LocalOutlierFactor(
            n_neighbors=20, novelty=True, n_jobs=-1), a.slow_n),
        "OneClassSVM": (lambda: OneClassSVM(nu=0.05, gamma="scale"), a.slow_n),
    }

    rows, saved = [], {}
    for fname, cols in feature_sets.items():
        F_fit = make_features(fit_df, cols)
        scaler = StandardScaler().fit(F_fit)
        X_fit = scaler.transform(F_fit)
        X_val = scaler.transform(make_features(val_df, cols))
        X_te = scaler.transform(make_features(test, cols))
        y_te = test["label"].to_numpy()
        for mname, (maker, max_fit) in models.items():
            print(f"running {fname} / {mname} ...", flush=True)
            row, model, thr = run_model(mname, maker, X_fit, X_val, X_te,
                                        y_te, max_fit, rng)
            row["features"] = fname
            rows.append(row)
            saved[(fname, mname)] = (scaler, model, thr, list(F_fit.columns))

    res = pd.DataFrame(rows)[["features", "model", "roc_auc", "pr_auc", "precision",
                              "recall", "f1", "fpr", "fit_s", "predict_s"]]
    print()
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    Path("eval").mkdir(exist_ok=True)
    res.to_csv("eval/week3_results.csv", index=False)

    scaler, model, thr, feat_names = saved[("portable", "IsolationForest")]
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"scaler": scaler, "model": model, "threshold": thr,
                 "features": feat_names, "preprocessing": "log1p then StandardScaler"},
                "models/iforest_portable.joblib")
    print("\nsaved -> eval/week3_results.csv, models/iforest_portable.joblib")


if __name__ == "__main__":
    main()
