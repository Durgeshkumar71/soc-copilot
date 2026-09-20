"""Train an Isolation Forest on Suricata-style window features."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "n_flows",
    "n_dest_ips",
    "n_dest_ports",
    "bytes_out",
    "bytes_in",
    "pkts_out",
    "mean_age",
    "unanswered_ratio",
    "bytes_ratio",
    "src_is_private",
]


def main():
    path = Path("data/features.csv")

    df = pd.read_csv(path)

    print("Loaded:", df.shape)

    X = df[FEATURES].copy()

    # Convert boolean values to numbers.
    X["src_is_private"] = X["src_is_private"].astype(int)

    # Replace invalid values.
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Log-transform highly skewed traffic-count/byte features.
    LOG_FEATURES = [
        "n_flows",
        "n_dest_ips",
        "n_dest_ports",
        "bytes_out",
        "bytes_in",
        "pkts_out",
        "mean_age",
        "bytes_ratio",
    ]

    X[LOG_FEATURES] = np.log1p(X[LOG_FEATURES].clip(lower=0))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )

    print("Training Isolation Forest...")
    model.fit(X_scaled)

    scores = -model.score_samples(X_scaled)

    threshold = np.percentile(scores, 95)

    print("Score range:", scores.min(), "to", scores.max())
    print("Threshold:", threshold)

    Path("models").mkdir(exist_ok=True)

    joblib.dump(
        {
            "scaler": scaler,
            "model": model,
            "threshold": threshold,
            "features": FEATURES,
            "log_features": LOG_FEATURES,
        },
        "models/iforest_suricata_window.joblib",
    )

    print("Saved -> models/iforest_suricata_window.joblib")


if __name__ == "__main__":
    main()
