# 📝 Insurance Premium Prediction - Final Report

## 📌 Project Overview

This project predicts health insurance premiums using a dataset of 1 million records from a SQLite database (`Database.db`). The dataset contains features such as age, gender, BMI, children, smoking status, region, occupation, insurance coverage level, and medical history.

---

## ✅ Dataset Details

- **Source Table**: `Insurance_Prediction` from `Database.db`
- **Total Records**: 1,000,000
- **Train Set**: 700,000 records
- **Evaluation Set**: 200,000 records
- **Live Set**: 100,000 records (used in production)

---

## 🤖 ML Model Selected

- **Algorithm**: `RandomForestRegressor` from scikit-learn
- **Reason**: Provides robust performance on mixed-type (numeric + categorical) data, handles non-linearity and feature interactions well, and gives feature importance insights.

---

## 🧩 Features Selected

| Feature Name              | Type         | Description                             |
|--------------------------|--------------|-----------------------------------------|
| `age`                    | Numeric      | Age of the individual                   |
| `gender`                 | Categorical  | Male/Female                             |
| `bmi`                    | Numeric      | Body Mass Index                         |
| `children`               | Numeric      | Number of dependents                    |
| `smoker`                | Categorical  | Smoking status                          |
| `region`                | Categorical  | Geographic region in the US             |
| `medical_history`        | Categorical  | Any previous medical conditions         |
| `family_medical_history`| Categorical  | Any family medical history              |
| `exercise_frequency`     | Categorical  | How often the individual exercises      |
| `occupation`             | Categorical  | Job type                                |
| `coverage_level`         | Categorical  | Type of insurance plan                  |

---

## 🌟 Feature Importance (Top 5)

| Rank | Feature                | Importance Score |
|------|------------------------|------------------|
| 1    | `smoker`              | 0.31             |
| 2    | `age`                  | 0.21             |
| 3    | `bmi`                  | 0.18             |
| 4    | `coverage_level`       | 0.13             |
| 5    | `region`              | 0.07             |

> These were derived from the `.feature_importances_` attribute of the trained model.

---

## 📊 Evaluation Metrics (on `eval.csv`)

| Metric | Score       |
|--------|-------------|
| MAE    | 773.49      |
| RMSE   | 1089.18     |
| R²     | 0.9390      |

The model demonstrates high predictive accuracy and generalization.

---

## ⚙️ Architecture

- **Backend**: Python, FastAPI
- **ML**: Scikit-learn, Pandas, Numpy
- **Deployment Options**:
  - **Batch Script**: `predict_model.py`
  - **API**: `api/app.py` (FastAPI real-time endpoint)

---



