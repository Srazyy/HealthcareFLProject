"""
Dataset Generation Script for Healthcare FL Benchmark.

Generates a realistic 3,000-sample medical claim verification dataset
comprising verified (label=0) and unverified/misinformation (label=1)
clinical claims. Saves the dataset to:
  - data/medical_claims_3k.parquet
  - data/medical_claims_3k.csv
"""

import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from collections import Counter

# Import generator from partition module
from src.data.partition import _generate_medical_claims


def main(n: int = 3000, seed: int = 42):
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = data_dir / "medical_claims_3k.parquet"
    csv_path = data_dir / "medical_claims_3k.csv"

    print(f"Generating {n} synthetic medical claims (seed={seed})...")
    claims, labels = _generate_medical_claims(n=n, seed=seed)

    df = pd.DataFrame({
        "claim_id": [f"MED-{i+1:05d}" for i in range(n)],
        "claim": claims,
        "label": labels,
        "label_name": ["verified" if y == 0 else "unverified" for y in labels],
    })

    # Save to Parquet
    df.to_parquet(parquet_path, index=False)
    print(f"Saved Parquet: {parquet_path} ({os.path.getsize(parquet_path) / 1024:.1f} KB)")

    # Save to CSV for human inspection
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV:     {csv_path} ({os.path.getsize(csv_path) / 1024:.1f} KB)")

    # Print summary & balance
    counts = Counter(labels)
    print("-" * 50)
    print("Dataset Summary:")
    print(f"  Total records: {len(df)}")
    print(f"  Verified (label=0):   {counts[0]} ({100.0 * counts[0] / len(df):.1f}%)")
    print(f"  Unverified (label=1): {counts[1]} ({100.0 * counts[1] / len(df):.1f}%)")
    print("-" * 50)
    print("Sample Records:")
    for idx in range(min(3, len(df))):
        row = df.iloc[idx]
        print(f"[{row['claim_id']}] ({row['label_name']}): {row['claim']}")


if __name__ == "__main__":
    main()
