import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import numpy as np

# Paths
base_dir = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_FILE = base_dir / "model" / "insurance_model.pkl"
EVAL_PATH = DATA_DIR / "eval.csv"

# Load the model
model = joblib.load(MODEL_FILE)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the insurance dataset.
    """
    # Replace empty strings with NaN globally
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    # Define columns
    numeric_cols = ['age', 'bmi', 'children']
    categorical_cols = ['gender', 'smoker', 'region', 'medical_history',
                        'family_medical_history', 'exercise_frequency',
                        'occupation', 'coverage_level']

    # Convert numeric columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill missing 'medical_history' with "none"
    df['medical_history'] = df['medical_history'].fillna('none')

    # Strip whitespace and lowercase for categorical columns
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Drop rows with remaining NaNs in essential columns (except medical_history)
    essential_cols = numeric_cols + [col for col in categorical_cols if col != 'medical_history']
    df.dropna(subset=essential_cols, inplace=True)

    return df


def evaluate_model():
    # Load the evaluation dataset
    df_eval = pd.read_csv(EVAL_PATH)

    # Preprocess the evaluation data (same as training data)
    df_eval_cleaned = preprocess_data(df_eval)

    # Separate features and target variable
    X_eval = df_eval_cleaned.drop(columns=["charges"])
    y_eval = df_eval_cleaned["charges"]

    # Make predictions on the evaluation set
    preds = model.predict(X_eval)

    # Calculate performance metrics
    mae = mean_absolute_error(y_eval, preds)
    rmse = np.sqrt(mean_squared_error(y_eval, preds))
    r2 = r2_score(y_eval, preds)

    # Print the evaluation results
    print(f"📊 MAE: {mae:.2f}")
    print(f"📊 RMSE: {rmse:.2f}")
    print(f"📊 R^2 Score: {r2:.4f}")


if __name__ == "__main__":
    evaluate_model()
