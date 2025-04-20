import pandas as pd
import sqlite3
from pathlib import Path


def load_from_database(db_path: Path, query: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def save_to_csv(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    db_path = data_dir / "Database.db"
    output_csv = data_dir / "Insurance_Prediction.csv"

    query = "SELECT * FROM Insurance_Prediction"
    df = load_from_database(db_path, query)
    save_to_csv(df, output_csv)
    print(f"✅ Insurance_Prediction.csv saved at {output_csv}")
