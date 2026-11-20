
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

RAW = ROOT / "datasets" / "raw"
INCOMING = ROOT / "datasets" / "incoming"

RAW.mkdir(parents=True, exist_ok=True)
INCOMING.mkdir(parents=True, exist_ok=True)


def generate_credit(n, seed):
    rng = np.random.default_rng(seed)

    age = rng.integers(20, 70, n)
    job = rng.integers(0, 4, n)
    housing = rng.choice(["own", "rent", "free"], n)
    saving = rng.choice(["little", "moderate", "rich"], n)
    checking = rng.choice(["little", "moderate", "rich"], n)

    amount = rng.integers(250, 15000, n)
    duration = rng.integers(4, 60, n)

    purpose = rng.choice(
        ["car", "furniture", "radio/TV",
         "education", "business", "repairs"],
        n
    )

    score = (
        0.03 * (age - 40)
        + 0.00012 * (8000 - amount)
        - 0.025 * (duration - 24)
        + 0.4 * (housing == "own")
        + 0.5 * (saving == "rich")
        + rng.normal(0, 1, n)
    )

    # 1 = good credit, 0 = bad credit
    credit_risk = (score >= np.median(score)).astype(int)

    return pd.DataFrame({
        "Age": age,
        "Sex": rng.choice(["male", "female"], n),
        "Job": job,
        "Housing": housing,
        "Saving accounts": saving,
        "Checking account": checking,
        "Credit amount": amount,
        "Duration": duration,
        "Purpose": purpose,
        "credit_risk": credit_risk
    })


def generate_fraud(n, seed, start_id):
    rng = np.random.default_rng(seed)

    features = rng.normal(0, 1, (n, 28))
    amount = rng.exponential(100, n)

    # Synthetic fraud signal
    fraud_score = (
        0.5 * features[:, 0]
        - 0.4 * features[:, 3]
        + 0.002 * amount
        + rng.normal(0, 1, n)
    )

    # Keep approximately 15% fraud records.
    threshold = np.quantile(fraud_score, 0.85)
    fraud_class = (fraud_score >= threshold).astype(int)

    data = {
        "id": np.arange(start_id, start_id + n),
    }

    for i in range(28):
        data[f"V{i+1}"] = features[:, i]

    data["Amount"] = amount
    data["Class"] = fraud_class

    return pd.DataFrame(data)


# Old training datasets
credit_old = generate_credit(200, 42)
fraud_old = generate_fraud(200, 43, 1)

# New incoming datasets
credit_new = generate_credit(80, 44)
fraud_new = generate_fraud(80, 45, 201)


# Save CSV files
credit_old.to_csv(RAW / "GermanCredit.csv", index=False)
fraud_old.to_csv(RAW / "creditcard_2023.csv", index=False)

credit_new.to_csv(INCOMING / "credit_new.csv", index=False)
fraud_new.to_csv(INCOMING / "fraud_new.csv", index=False)


print("\nSYNTHETIC DATASETS CREATED SUCCESSFULLY!")

for name, df in [
    ("Credit Old", credit_old),
    ("Credit New", credit_new),
    ("Fraud Old", fraud_old),
    ("Fraud New", fraud_new)
]:
    print(f"\n{name}: {df.shape}")
    print(df.iloc[:, -1].value_counts())
