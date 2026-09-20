"""Week 3 (part 2): per-category recall, precision@k, supervised baseline, scaling.

Usage:
    python triage/analyze.py
    python triage/analyze.py --svm-max 20000 --slow-n 10000
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from train_compare import DATA, NON_FEATURE, PORTABLE_COLS, make_features

KS = [100, 500, 1000, 2000]


def make_models(seed, slow_n):
    return {
        "IsolationForest": (lambda: IsolationForest(
            n_estimators=200, random_state=seed, n_jobs=-1), None),
        "LOF": (lambda: LocalOutlierFactor(
            n_neighbors=20, novelty=True, n_jobs=-1), slow_n),
        "OneClassSVM": (lambda: OneClassSVM(nu=0.05, gamma="scale"), slow_n),
    }


def metrics(y, score, flagged):
    fpr = ((flagged == 1) & (y == 0)).sum() / max((y == 0).sum(), 1)
    return {
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "precision": precision_score(y, flagged, zero_division=0),
        "recall": recall_score(y, flagged, zero_division=0),
        "f1": f1_score(y, flagged, zero_division=0),
        "fpr": fpr,
    }


def scaling_experiment(X_fit, X_te, models, sizes, svm_max, seed):
    rng = np.random.default_rng(seed)
    probe = X_te[rng.choice(len(X_te), min(20000, len(X_te)), replace=False)]
    rows = []
    for n in sizes:
        idx = rng.choice(len(X_fit), n, replace=False)
        for mname, (maker, _) in models.items():
            if mname == "OneClassSVM" and n > svm_max:
                continue
            print(f"  scaling: n={n} {mname}", flush=True)
            m = maker()
            t0 = time.perf_counter()
            m.fit(X_fit[idx])
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            m.score_samples(probe)
            pred_s = time.perf_counter() - t0
            rows.append({"n_train": n, "model": mname,
                         "fit_s": fit_s, "predict_s": pred_s})
    return pd.DataFrame(rows)


def plot_scaling(df):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -> skipping plot (pip install matplotlib)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in ((axes[0], "fit_s", "Fit time"),
                           (axes[1], "predict_s", "Predict time (20k rows)")):
        for mname, g in df.groupby("model"):
            ax.plot(g["n_train"], g[col], marker="o", label=mname)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("training rows")
        ax.set_ylabel("seconds")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    Path("docs").mkdir(exist_ok=True)
    fig.savefig("docs/scaling.png", dpi=150)
    print("saved -> docs/scaling.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slow-n", type=int, default=10000)
    ap.add_argument("--svm-max", type=int, default=20000,
                    help="largest training size for One-Class SVM in the scaling test")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    Path("eval").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    train = pd.read_csv(DATA / "UNSW_NB15_training-set.csv")
    test = pd.read_csv(DATA / "UNSW_NB15_testing-set.csv")
    y = test["label"].to_numpy()
    cats = test["attack_cat"].astype(str).to_numpy()

    normal = train[train["label"] == 0].sample(frac=1, random_state=a.seed)
    n_val = int(0.2 * len(normal))
    val_df, fit_df = normal.iloc[:n_val], normal.iloc[n_val:]

    # realistic subset: all normal test rows + attacks down to 5% share
    norm_idx = np.where(y == 0)[0]
    att_idx = np.where(y == 1)[0]
    n_att = int(0.05 / 0.95 * len(norm_idx))
    real_idx = np.concatenate(
        [norm_idx, rng.choice(att_idx, min(n_att, len(att_idx)), replace=False)])
    y_real = y[real_idx]

    full_cols = [c for c in train.columns if c not in NON_FEATURE]
    feature_sets = {"portable": PORTABLE_COLS, "full": full_cols}
    models = make_models(a.seed, a.slow_n)
    attack_cats = sorted(set(cats[y == 1]))

    recall_tbl = pd.DataFrame(index=attack_cats + ["NORMAL (false positive rate)"])
    counts = [int((cats == c).sum()) for c in attack_cats] + [int((y == 0).sum())]
    recall_tbl.insert(0, "n_test", counts)
    pk_rows, saved_lof = [], None
    Xs = {}

    # ---- 1) unsupervised models: per-category recall + precision@k -----------
    for fname, cols in feature_sets.items():
        F_fit = make_features(fit_df, cols)
        scaler = StandardScaler().fit(F_fit)
        X_fit = scaler.transform(F_fit)
        X_val = scaler.transform(make_features(val_df, cols))
        X_te = scaler.transform(make_features(test, cols))
        Xs[fname] = (X_fit, X_te)
        for mname, (maker, max_fit) in models.items():
            print(f"fitting {fname} / {mname} ...", flush=True)
            Xf = X_fit
            if max_fit and len(Xf) > max_fit:
                Xf = Xf[rng.choice(len(Xf), max_fit, replace=False)]
            model = maker().fit(Xf)
            s_te = -model.score_samples(X_te)
            thr = np.percentile(-model.score_samples(X_val), 95)
            flagged = s_te > thr
            col = f"{fname}/{mname}"
            recall_tbl[col] = [flagged[cats == c].mean() for c in attack_cats] + \
                              [flagged[y == 0].mean()]
            order = np.argsort(-s_te[real_idx])
            ranked = y_real[order]
            for k in KS:
                if k <= len(ranked):
                    pk_rows.append({"features": fname, "model": mname, "k": k,
                                    "precision_at_k": ranked[:k].mean()})
            if (fname, mname) == ("portable", "LOF"):
                saved_lof = {"scaler": scaler, "model": model, "threshold": thr,
                             "features": list(F_fit.columns),
                             "preprocessing": "log1p then StandardScaler"}

    print("\n=== Recall per attack category (threshold = 95th pct of held-out normal) ===")
    print(recall_tbl.to_string(float_format=lambda v: f"{v:.3f}"))
    recall_tbl.to_csv("eval/week3_per_category.csv")

    pk = pd.DataFrame(pk_rows).pivot_table(
        index=["features", "model"], columns="k", values="precision_at_k")
    print(f"\n=== Precision@k on realistic split "
          f"({y_real.mean():.1%} attacks, {len(y_real)} rows) ===")
    print(pk.to_string(float_format=lambda v: f"{v:.3f}"))
    pk.to_csv("eval/week3_precision_at_k.csv")

    if saved_lof:
        joblib.dump(saved_lof, "models/lof_portable.joblib")
        print("saved -> models/lof_portable.joblib")

    # ---- 2) supervised baseline ----------------------------------------------
    sup_rows = []
    for fname, cols in feature_sets.items():
        print(f"fitting RandomForest ({fname}) ...", flush=True)
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=a.seed)
        rf.fit(make_features(train, cols), train["label"])
        proba = rf.predict_proba(make_features(test, cols))[:, 1]
        flagged = (proba >= 0.5).astype(int)
        for split, idx in (("balanced", np.arange(len(y))), ("realistic 5%", real_idx)):
            row = {"features": fname, "split": split}
            row.update(metrics(y[idx], proba[idx], flagged[idx]))
            sup_rows.append(row)
    sup = pd.DataFrame(sup_rows)
    print("\n=== Supervised baseline: RandomForest (threshold 0.5) ===")
    print(sup.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    sup.to_csv("eval/week3_supervised.csv", index=False)

    # ---- 3) scaling experiment (portable features) ----------------------------
    print("\n=== Scaling experiment (portable features) ===")
    X_fit, X_te = Xs["portable"]
    sizes = sorted({n for n in [1000, 2000, 5000, 10000, 20000, len(X_fit)]
                    if n <= len(X_fit)})
    sc = scaling_experiment(X_fit, X_te, models, sizes, a.svm_max, a.seed)
    print(sc.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    sc.to_csv("eval/week3_scaling.csv", index=False)
    plot_scaling(sc)


if __name__ == "__main__":
    main()
