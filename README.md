# SOC Copilot

A multi-stage SOC alert triage system that reduces alert fatigue with unsupervised ML,
enriches suspicious activity with threat intelligence, and drafts analyst-ready summaries.

## Pipeline

1. **Ingest** - Suricata `eve.json` (alerts and flows)
2. **Triage** - per-source-IP windowed features + Isolation Forest anomaly scoring
3. **Enrichment** - AlienVault OTX / VirusTotal reputation lookups, MITRE ATT&CK mapping
4. **Response** - LLM-generated summary and suggested mitigations (human approval required)

## Status

Work in progress. Currently: log ingestion and flow feature engineering.

## Security note

Fields such as User-Agent, hostname and URL are attacker-controlled. They are kept in
separate columns and are sanitized before reaching any LLM prompt.
