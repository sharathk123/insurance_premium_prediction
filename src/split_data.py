import pandas as pd
from pathlib import Path


def split_dataset(csv_path: Path, output_dir: Path) -> None:
    df = pd.read_csv(csv_path)

    train_df = df.iloc[:700_000]
    eval_df = df.iloc[700_000:900_000]
    live_df = df.iloc[900_000:]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    eval_df.to_csv(output_dir / "eval.csv", index=False)
    live_df.to_csv(output_dir / "live.csv", index=False)

    print("✅ Data split into train.csv, eval.csv, and live.csv")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    csv_path = data_dir / "Insurance_Prediction.csv"
    split_dataset(csv_path, data_dir)
