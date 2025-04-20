import pandas as pd
import numpy as np


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    numeric_cols = ['age', 'bmi', 'children']
    categorical_cols = ['gender', 'smoker', 'region', 'medical_history',
                        'family_medical_history', 'exercise_frequency',
                        'occupation', 'coverage_level']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['medical_history'] = df['medical_history'].fillna('none')

    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()

    essential_cols = numeric_cols + [col for col in categorical_cols if col != 'medical_history']
    df.dropna(subset=essential_cols, inplace=True)

    return df
