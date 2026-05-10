import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load models and scalers
try:
    linear_reg = joblib.load('models/linear_regression_model.pkl')
    decision_tree = joblib.load('models/decision_tree_model.pkl')
    logistic_reg = joblib.load('models/logistic_regression_model.pkl')
    
    scaler_lr = joblib.load('models/scaler_linear_regression.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

def prepare_features(data_dict):
    """Convert input dictionary to model-ready features"""
    try:
        # Create DataFrame from input
        df = pd.DataFrame([data_dict])
        
        low_card = ['fuel_type', 'is_accident', 'clean_title']
        for col in low_card:
            if col in df.columns:
                df = pd.get_dummies(df, columns=[col], drop_first=True)
        
        high_card = ['brand', 'transmission', 'ext_col', 'int_col']
        for col in high_card:
            if col in df.columns:
                if col in label_encoders:
                    df[col + '_encoded'] = label_encoders[col].transform(df[col])
                df = df.drop(col, axis=1)
        
        return df
    except Exception as e:
        return None

def predict_linear_regression(features):
    """Predict price using Linear Regression"""
    try:
        features_scaled = scaler_lr.transform(features)
        prediction = linear_reg.predict(features_scaled)[0]
        return round(prediction, 2)
    except Exception as e:
        return {"error": str(e)}

def predict_decision_tree(features):
    """Predict price using Decision Tree"""
    try:
        prediction = decision_tree.predict(features)[0]
        return round(prediction, 2)
    except Exception as e:
        return {"error": str(e)}

def predict_logistic_regression(features):
    """Predict price category using Logistic Regression"""
    try:
        features_scaled = scaler_lr.transform(features)
        prediction = logistic_reg.predict(features_scaled)[0]
        probabilities = logistic_reg.predict_proba(features_scaled)[0]
        
        classes = ['LOW', 'MEDIUM', 'HIGH']
        return {
            "category": prediction,
            "probabilities": {classes[i]: round(prob, 4) for i, prob in enumerate(probabilities)}
        }
    except Exception as e:
        return {"error": str(e)}