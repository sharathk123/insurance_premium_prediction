# 🏥 Insurance Premium Prediction API

This project uses machine learning to predict health insurance premiums in the US based on demographic and lifestyle features. It supports both batch processing and real-time inference using a FastAPI-powered API.

1. Set up Virtual Environment
   
       python -m venv env
       source env/bin/activate  # On Windows: env\Scripts\activate
       pip install -r requirements.txt
2. Extract Data

        python src/read_data.py
3. Split Dataset
        
        python src/split_data.py
4. Preprocess & Train Model

        python src/model_train.py
4. Predict on Evaluation or Live Set

        python src/model_predict.py
# Run the FastAPI Server
        uvicorn api.app:app --reload
# Access Swagger UI at:
    http://127.0.0.1:8000/docs