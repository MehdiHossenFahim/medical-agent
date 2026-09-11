"""
Generates small SYNTHETIC CSV files that mimic the column names/types of the three
real Kaggle datasets used in this project. This lets you run the whole pipeline
(csv_to_sqlite -> tools -> agent) without needing a Kaggle account first.

Replace the generated files in data/ with the real Kaggle CSVs before drawing any
real conclusions -- see data/README.md for download instructions.

Usage:
    python scripts/generate_sample_data.py [--rows 500]
"""
import argparse
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def make_heart_disease(n: int, rng: np.random.Generator) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "age": rng.integers(29, 78, n),
            "sex": rng.integers(0, 2, n),
            "cp": rng.integers(0, 4, n),
            "trestbps": rng.integers(94, 201, n),
            "chol": rng.integers(126, 565, n),
            "fbs": rng.integers(0, 2, n),
            "restecg": rng.integers(0, 3, n),
            "thalach": rng.integers(71, 203, n),
            "exang": rng.integers(0, 2, n),
            "oldpeak": np.round(rng.uniform(0, 6.2, n), 1),
            "slope": rng.integers(0, 3, n),
            "ca": rng.integers(0, 5, n),
            "thal": rng.integers(0, 4, n),
            "target": rng.integers(0, 2, n),
        }
    )
    return df


def make_cancer(n: int, rng: np.random.Generator) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Age": rng.integers(20, 90, n),
            "Gender": rng.integers(0, 2, n),  # 0=Male, 1=Female
            "BMI": np.round(rng.uniform(15, 40, n), 2),
            "Smoking": rng.integers(0, 2, n),
            "GeneticRisk": rng.integers(0, 3, n),
            "PhysicalActivity": np.round(rng.uniform(0, 10, n), 2),
            "AlcoholIntake": np.round(rng.uniform(0, 5, n), 2),
            "CancerHistory": rng.integers(0, 2, n),
            "Diagnosis": rng.integers(0, 2, n),  # 0=No cancer, 1=Cancer
        }
    )
    return df


def make_diabetes(n: int, rng: np.random.Generator) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Pregnancies": rng.integers(0, 17, n),
            "Glucose": rng.integers(44, 199, n),
            "BloodPressure": rng.integers(24, 122, n),
            "SkinThickness": rng.integers(0, 99, n),
            "Insulin": rng.integers(0, 846, n),
            "BMI": np.round(rng.uniform(18, 67, n), 1),
            "DiabetesPedigreeFunction": np.round(rng.uniform(0.08, 2.42, n), 3),
            "Age": rng.integers(21, 81, n),
            "Outcome": rng.integers(0, 2, n),
        }
    )
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    make_heart_disease(args.rows, rng).to_csv(os.path.join(DATA_DIR, "heart.csv"), index=False)
    make_cancer(args.rows, rng).to_csv(os.path.join(DATA_DIR, "cancer.csv"), index=False)
    make_diabetes(args.rows, rng).to_csv(os.path.join(DATA_DIR, "diabetes.csv"), index=False)

    print(f"Wrote {args.rows} synthetic rows each to:")
    print(f"  {os.path.join(DATA_DIR, 'heart.csv')}")
    print(f"  {os.path.join(DATA_DIR, 'cancer.csv')}")
    print(f"  {os.path.join(DATA_DIR, 'diabetes.csv')}")


if __name__ == "__main__":
    main()
