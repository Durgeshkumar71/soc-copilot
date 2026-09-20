"""Suricata flow events -> per-source-IP windowed features for anomaly detection."""
import ipaddress
import json
import sys
from pathlib import Path

import pandas as pd


def load_flows(path) -> pd.DataFrame:
    rows = []
    with open(Path(path), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event_type") != "flow":
                continue
            fl = ev.get("flow", {})
            rows.append({
                "start": fl.get("start"),
                "src_ip": ev.get("src_ip"),
                "dest_ip": ev.get("dest_ip"),
                "dest_port": ev.get("dest_port"),
                "proto": ev.get("proto"),
                "pkts_toserver": fl.get("pkts_toserver", 0),
                "pkts_toclient": fl.get("pkts_toclient", 0),
                "bytes_toserver": fl.get("bytes_toserver", 0),
                "bytes_toclient": fl.get("bytes_toclient", 0),
                "age": fl.get("age", 0),
                "alerted": bool(fl.get("alerted", False)),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["start"] = pd.to_datetime(df["start"], errors="coerce")
    return df


def build_features(flows: pd.DataFrame, window: str = "1min") -> pd.DataFrame:
    df = flows.dropna(subset=["start", "src_ip"]).copy()
    df["window"] = df["start"].dt.floor(window)
    df["unanswered"] = (df["pkts_toclient"] == 0).astype(int)
    feats = (
        df.groupby(["src_ip", "window"])
        .agg(
            n_flows=("dest_ip", "size"),
            n_dest_ips=("dest_ip", "nunique"),
            n_dest_ports=("dest_port", "nunique"),
            bytes_out=("bytes_toserver", "sum"),
            bytes_in=("bytes_toclient", "sum"),
            pkts_out=("pkts_toserver", "sum"),
            mean_age=("age", "mean"),
            unanswered_ratio=("unanswered", "mean"),
            any_alert=("alerted", "max"),  # weak label for evaluation ONLY
        )
        .reset_index()
    )
    feats["bytes_ratio"] = feats["bytes_out"] / (feats["bytes_in"] + 1)
    feats["src_is_private"] = feats["src_ip"].map(lambda ip: ipaddress.ip_address(ip).is_private)
    return feats


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python triage/features.py <path-to-eve.json>")
    flows = load_flows(sys.argv[1])
    print("flows:", flows.shape)
    feats = build_features(flows)
    print("feature rows:", feats.shape)
    pd.set_option("display.width", 200)
    print(feats.describe().T)
    out = Path("data/features.csv")
    feats.to_csv(out, index=False)
    print("saved ->", out)
