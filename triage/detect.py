"""SOC-Copilot: detect anomalies from Suricata window features."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from severity import calculate_severity


MODEL_PATH = "models/iforest_suricata_window.joblib"
FEATURE_PATH = "data/features.csv"
OUTPUT_PATH = "eval/detection_results.csv"


def preprocess(df, features, log_features):
    """Prepare features exactly like training."""
    X = df[features].copy()

    # Convert boolean to integer
    if "src_is_private" in X.columns:
        X["src_is_private"] = X["src_is_private"].astype(int)

    # Replace invalid values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Apply same log transformation used during training
    X[log_features] = np.log1p(
        X[log_features].clip(lower=0)
    )

    return X


def explain(row):
    """Generate human-readable explanation for an anomaly."""

    reasons = []

    # Destination port diversity
    if row["n_dest_ports"] >= 50:
        reasons.append(
            f"high destination-port diversity "
            f"({int(row['n_dest_ports'])} ports)"
        )

    # Unanswered connections
    if row["unanswered_ratio"] >= 0.80:
        reasons.append(
            f"very high unanswered ratio "
            f"({row['unanswered_ratio']:.0%})"
        )

    # High flow volume
    if row["n_flows"] >= 100:
        reasons.append(
            f"high flow volume ({int(row['n_flows'])} flows)"
        )

    # Many destination IPs
    if row["n_dest_ips"] >= 10:
        reasons.append(
            f"many destination IPs "
            f"({int(row['n_dest_ips'])})"
        )

    # Outbound traffic much higher than inbound
    if row["bytes_out"] > (row["bytes_in"] * 2):
        reasons.append(
            "outbound traffic significantly exceeds inbound traffic"
        )

    # Fallback explanation
    if not reasons:
        reasons.append(
            "feature combination differs from learned normal traffic"
        )

    reason_text = "; ".join(reasons)

    # Pattern classification
    if (
        row["n_dest_ports"] >= 50
        and row["unanswered_ratio"] >= 0.80
    ):
        pattern = "Possible port-scan / failed-connection pattern"

    elif row["n_flows"] >= 100:
        pattern = "High connection activity"

    elif row["n_dest_ips"] >= 10:
        pattern = "Multiple-destination network activity"

    else:
        pattern = "Unusual network behavior"

    return reason_text, pattern


def main():

    print("=== SOC-COPILOT ANOMALY DETECTION ===")

    # --------------------------------------------------
    # Load trained model
    # --------------------------------------------------

    bundle = joblib.load(MODEL_PATH)

    model = bundle["model"]
    scaler = bundle["scaler"]
    threshold = bundle["threshold"]
    features = bundle["features"]
    log_features = bundle["log_features"]

    print(f"Threshold: {threshold:.4f}")

    # --------------------------------------------------
    # Load feature data
    # --------------------------------------------------

    df = pd.read_csv(FEATURE_PATH)

    if df.empty:
        print("\nNo feature rows found.")
        return

    # --------------------------------------------------
    # Prepare ML features
    # --------------------------------------------------

    X = preprocess(
        df,
        features,
        log_features
    )

    # Apply scaler
    X_scaled = scaler.transform(X)

    # --------------------------------------------------
    # Calculate anomaly scores
    # --------------------------------------------------

    scores = -model.score_samples(X_scaled)

    df["anomaly_score"] = scores

    # --------------------------------------------------
    # Detect anomalies
    # --------------------------------------------------

    df["status"] = np.where(
        df["anomaly_score"] >= threshold,
        "ANOMALY",
        "NORMAL"
    )

    # Create empty output columns
    df["reason"] = ""
    df["pattern"] = ""
    df["severity"] = "NONE"

    # --------------------------------------------------
    # Explain anomalies + calculate severity
    # --------------------------------------------------

    for idx, row in df.iterrows():

        if row["status"] == "ANOMALY":

            reason, pattern = explain(row)

            severity = calculate_severity(row)

            df.at[idx, "reason"] = reason
            df.at[idx, "pattern"] = pattern
            df.at[idx, "severity"] = severity

    # --------------------------------------------------
    # Display alerts
    # --------------------------------------------------

    alerts = df[df["status"] == "ANOMALY"]

    print("\n=== ALERTS ===")

    if alerts.empty:

        print("\nNo anomalies detected.")

    else:

        for _, row in alerts.iterrows():

            print("\n----------------------------------------")

            print(f"Source IP : {row['src_ip']}")
            print(f"Window    : {row['window']}")
            print(f"Score     : {row['anomaly_score']:.4f}")
            print(f"Severity  : {row['severity']}")
            print(f"Pattern   : {row['pattern']}")
            print(f"Reasons   : {row['reason']}")

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    total = len(df)
    anomalies = len(alerts)
    normal = total - anomalies

    print("\n=== SUMMARY ===")

    print(f"Total windows     : {total}")
    print(f"Anomalies         : {anomalies}")
    print(f"Normal            : {normal}")

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    Path("eval").mkdir(exist_ok=True)

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"\nSaved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
