import joblib
import pandas as pd
import numpy as np

# Load model once (cached)
try:
    model = joblib.load('models/demand_model.pkl')
    print("XGBoost model loaded successfully")
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None

# The exact feature names your model was trained with
TRAINED_FEATURES = [
    'active_customers', 'new_connections', 'festival_indicator', 'price',
    'dayofweek', 'month', 'year', 'is_weekend',
    'demand_lag_1', 'demand_lag_7', 'demand_lag_30',
    'demand_roll_mean_30'
]

def predict_demand(new_data_df):
    """
    Safe prediction function:
    - Selects only trained columns
    - Converts to float
    - Returns array of predictions or fallback value on error
    """
    if model is None:
        return np.array([0.0] * len(new_data_df))  # safe fallback

    try:
        # Keep only columns the model knows
        X = new_data_df[TRAINED_FEATURES].copy()
        
        # Force numeric types (very important!)
        X = X.astype(float)
        
        # Predict
        preds = model.predict(X)
        
        # Optional: clip unrealistic values
        preds = np.clip(preds, 0, 100000)
        
        return preds
    
    except KeyError as ke:
        missing_cols = list(set(TRAINED_FEATURES) - set(new_data_df.columns))
        print(f"KeyError - Missing columns: {missing_cols}")
        return np.array([0.0] * len(new_data_df))  # fallback
    
    except ValueError as ve:
        print(f"ValueError during prediction: {ve}")
        return np.array([0.0] * len(new_data_df))
    
    except Exception as e:
        print(f"Unexpected prediction error: {e}")
        return np.array([0.0] * len(new_data_df))