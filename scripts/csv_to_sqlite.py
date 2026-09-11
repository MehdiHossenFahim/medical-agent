"""
Converts the three raw CSV files in data/ into three SQLite databases in db/,
with meaningful table names and SQL column types inferred from the pandas dtypes.

    data/heart.csv    -> db/heart_disease.db  (table: heart_disease_records)
    data/cancer.csv   -> db/cancer.db         (table: cancer_records)
    data/diabetes.csv -> db/diabetes.db       (table: diabetes_records)

Usage:
    python scripts/csv_to_sqlite.py
"""
import os
import sqlite3

import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "db")

DATASETS = [
    {
        "csv": os.path.join(DATA_DIR, "heart.csv"),
        "db": os.path.join(DB_DIR, "heart_disease.db"),
        "table": "heart_disease_records",
    },
    {
        "csv": os.path.join(DATA_DIR, "cancer.csv"),
        "db": os.path.join(DB_DIR, "cancer.db"),
        "table": "cancer_records",
    },
    {
        "csv": os.path.join(DATA_DIR, "diabetes.csv"),
        "db": os.path.join(DB_DIR, "diabetes.db"),
        "table": "diabetes_records",
    },
]


def pandas_dtype_to_sql(dtype) -> str:
    """Map a pandas dtype to a reasonably specific SQLite column type."""
    kind = dtype.kind
    if kind in ("i", "u"):
        return "INTEGER"
    if kind == "f":
        return "REAL"
    if kind == "b":
        return "BOOLEAN"
    return "TEXT"


def convert(csv_path: str, db_path: str, table_name: str) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Missing {csv_path}. Either download the real Kaggle CSV (see data/README.md) "
            f"or run `python scripts/generate_sample_data.py` for a synthetic demo dataset."
        )

    df = pd.read_csv(csv_path)
    # Normalize column names: strip whitespace, keep original casing (SQLite is case-insensitive
    # for identifiers but we quote them to be safe with mixed-case Kaggle column names).
    df.columns = [c.strip() for c in df.columns]

    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')

    col_defs = ", ".join(f'"{col}" {pandas_dtype_to_sql(dtype)}' for col, dtype in df.dtypes.items())
    cur.execute(f'CREATE TABLE "{table_name}" ({col_defs})')
    conn.commit()

    df.to_sql(table_name, conn, if_exists="append", index=False)

    # Quick sanity check + helpful index on any obvious target/outcome column.
    row_count = cur.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
    conn.close()

    print(f"[OK] {csv_path} -> {db_path} :: table '{table_name}' ({row_count} rows, "
          f"{len(df.columns)} columns)")


def main():
    for spec in DATASETS:
        convert(spec["csv"], spec["db"], spec["table"])


if __name__ == "__main__":
    main()
