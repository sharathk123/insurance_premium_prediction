import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocess import preprocess_data

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    train_path = base_dir / "data" / "train.csv"
    df = pd.read_csv(train_path)
    df = preprocess_data(df)

    X = df.drop("charges", axis=1)
    y = df["charges"]

    numeric_features = ['age', 'bmi', 'children']
    categorical_features = ['gender', 'smoker', 'region', 'medical_history',
                            'family_medical_history', 'exercise_frequency',
                            'occupation', 'coverage_level']

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, base_dir / "model"/ "insurance_model.pkl")
    joblib.dump(X.columns.tolist(),  base_dir / "model"/ "feature_columns.pkl")
    print(f"✅ Model trained and saved to {base_dir / 'model/insurance_model.pkl'}")

    #preds = pipeline.predict(X)
    #print(f"📊 MAE: {mean_absolute_error(y, preds):.2f}")
    #print(f"📊 RMSE: {mean_squared_error(y, preds):.2f}")
    #print(f"📊 R^2 Score: {r2_score(y, preds):.4f}")
