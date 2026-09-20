def calculate_severity(row):
    score = 0

    # Port diversity
    if row["n_dest_ports"] >= 100:
        score += 3
    elif row["n_dest_ports"] >= 50:
        score += 2
    elif row["n_dest_ports"] >= 20:
        score += 1

    # Unanswered connections
    if row["unanswered_ratio"] >= 0.90:
        score += 3
    elif row["unanswered_ratio"] >= 0.70:
        score += 2
    elif row["unanswered_ratio"] >= 0.50:
        score += 1

    # Flow volume
    if row["n_flows"] >= 500:
        score += 2
    elif row["n_flows"] >= 100:
        score += 1

    if score >= 5:
        return "HIGH"
    elif score >= 3:
        return "MEDIUM"
    else:
        return "LOW"
