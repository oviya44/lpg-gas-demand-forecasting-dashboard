import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error   # ← updated import
import joblib
from preprocess import load_and_prepare_data

df = load_and_prepare_data()

features = [
    'active_customers', 'new_connections', 'festival_indicator', 'price',
    'dayofweek', 'month', 'year', 'is_weekend',
    'demand_lag_1', 'demand_lag_7', 'demand_lag_30',
    'demand_roll_mean_30'
]

X = df[features]
y = df['demand']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = xgb.XGBRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# Quick evaluation
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = root_mean_squared_error(y_test, preds)          # ← this is the correct modern way

print(f"MAE: {mae:.2f}   RMSE: {rmse:.2f}")

joblib.dump(model, 'models/demand_model.pkl')
print("Model saved.")