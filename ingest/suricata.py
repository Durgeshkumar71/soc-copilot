"""Suricata eve.json -> pandas DataFrame (alert events only)."""
import json
import sys
from pathlib import Path

import pandas as pd


def load_alerts(path) -> pd.DataFrame:
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
            if ev.get("event_type") != "alert":
                continue
            alert = ev.get("alert", {})
            http = ev.get("http", {})
            rows.append({
                "timestamp": ev.get("timestamp"),
                "src_ip": ev.get("src_ip"),
                "src_port": ev.get("src_port"),
                "dest_ip": ev.get("dest_ip"),
                "dest_port": ev.get("dest_port"),
                "proto": ev.get("proto"),
                "signature": alert.get("signature"),
                "signature_id": alert.get("signature_id"),
                "category": alert.get("category"),
                "severity": alert.get("severity"),
                "action": alert.get("action"),
                # attacker-controlled fields: never pass raw into an LLM prompt
                "http_user_agent": http.get("http_user_agent"),
                "http_hostname": http.get("hostname"),
                "http_url": http.get("url"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python ingest/suricata.py <path-to-eve.json>")
    alerts = load_alerts(sys.argv[1])
    print("shape:", alerts.shape)
    print(alerts.head())
    if not alerts.empty:
        print(alerts["signature"].value_counts().head(10))
