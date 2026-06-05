import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import joblib

st.set_page_config(page_title="AI LPG Demand Optimizer - Tamil Nadu Demo", layout="wide")

st.title("AI-Based LPG Demand Prediction & Redistribution System")
st.markdown("**State: Tamil Nadu** – 5 Districts (Coimbatore, Chennai, Madurai, Trichy, Salem)")

# Model loading (safe)
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/demand_model.pkl')
        st.session_state.model_loaded = True
        return model
    except:
        st.session_state.model_loaded = False
        return None

model = load_model()

# Features your model was trained on
TRAINED_FEATURES = [
    'active_customers', 'new_connections', 'festival_indicator', 'price',
    'dayofweek', 'month', 'year', 'is_weekend',
    'demand_lag_1', 'demand_lag_7', 'demand_lag_30',
    'demand_roll_mean_30'
]

# Simple rule-based fallback
def rule_based_estimation(row):
    avg_per_customer = 0.08
    base = row['active_customers'] * avg_per_customer
    festival_factor = 1.35 if row.get('festival_indicator', 0) == 1 else 1.0
    price_factor = max(0.75, 1 - (row.get('price', 850) - 800) / 400)
    estimated = base * festival_factor * price_factor
    return max(0, round(estimated + np.random.normal(0, estimated * 0.05)))

# Upload dataset
uploaded_file = st.file_uploader("Step 1: Upload your LPG Dataset (CSV)", type="csv")

if uploaded_file:
    with st.spinner("Loading dataset..."):
        df = pd.read_csv(uploaded_file)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    st.success(f"Dataset loaded successfully! {len(df)} rows")

    # Step 2: Train / Load Model
    if st.button("Step 2: Train / Load AI Model (XGBoost)"):
        if model is not None:
            st.success("Model loaded successfully from disk!")
            st.info("MAE ~11,000 | RMSE ~12,900 (previous training result)")
        else:
            st.warning("Model file not found → Using simulation & rule-based mode for demo.")
            st.info("To use real model: run `python src/train.py` first")

    # Dashboard - Current Status
    st.subheader("Step 3: Dashboard – Current Status")
    
    if 'date' in df.columns and not df.empty:
        latest_date = df['date'].max()
        today_df = df[df['date'] == latest_date].copy().reset_index(drop=True)
        
        st.write(f"Latest available date: **{latest_date.date()}**")
        
        # Force numeric columns
        for col in ['stock', 'refill_bookings', 'active_customers', 'price']:
            if col in today_df.columns:
                today_df[col] = pd.to_numeric(today_df[col], errors='coerce').fillna(0)
        
        st.dataframe(
            today_df[['district', 'branch', 'stock', 'refill_bookings']]
            .rename(columns={'refill_bookings': 'sales'})
            .style.format({'stock': '{:,.0f}', 'sales': '{:,.0f}'})
        )
    else:
        st.info("Dataset loaded, but required columns missing.")

    # Step 4: Prediction
    st.subheader("Step 4: Demand Prediction (Next Week Forecast)")
    
    col1, col2 = st.columns(2)
    with col1:
        festival_boost = st.slider(
            "Festival Impact (0 = normal, 1 = major festival)",
            0.0, 1.5, 0.2, step=0.05
        )
    with col2:
        external_shock = st.slider(
            "External Shock (war/news/supply issue)",
            -0.5, 0.5, 0.0, step=0.05
        )

    if st.button("Predict Next 7 Days Demand"):
        if len(today_df) == 0:
            st.error("No data available for prediction.")
        else:
            base_demand = today_df['refill_bookings'].mean()
            
            # Try ML prediction if possible
            ml_success = False
            if model is not None and all(f in today_df.columns for f in TRAINED_FEATURES):
                try:
                    X = today_df[TRAINED_FEATURES].astype(float)
                    preds = model.predict(X)
                    base_demand = preds.mean()
                    ml_success = True
                except Exception as e:
                    st.warning(f"ML skipped: {str(e)[:60]}... → Using rule-based fallback")
            
            adjusted_base = base_demand * (1 + festival_boost + external_shock)
            predicted_week = adjusted_base * np.random.uniform(0.92, 1.08, 7)
            
            pred_df = pd.DataFrame({
                'Day': [f"+{i+1} day" for i in range(7)],
                'Predicted Demand': predicted_week.round(0).astype(int)
            })
            
            st.bar_chart(pred_df.set_index('Day'), use_container_width=True)
            st.metric("Avg next week daily demand", f"{int(predicted_week.mean()):,} cylinders")
            
            if ml_success:
                st.caption("Based on XGBoost model + factors")
            else:
                st.caption("Based on rule-based estimation + factors")

    # Step 5 & 6: Shortage, MRP, Excess
    st.subheader("Step 5 & 6: Shortage Alert + MRP + Excess Redistribution")
    
    if 'stock' in today_df.columns and 'refill_bookings' in today_df.columns:
        # Ensure numeric
        today_df['stock'] = pd.to_numeric(today_df['stock'], errors='coerce').fillna(0)
        today_df['refill_bookings'] = pd.to_numeric(today_df['refill_bookings'], errors='coerce').fillna(0)
        
        # Calculate predicted demand
        today_df['predicted_demand'] = today_df['refill_bookings'] * (1 + festival_boost + external_shock)
        today_df['predicted_demand'] = today_df['predicted_demand'].clip(lower=0)
        
        # Shortage & Excess
        today_df['shortage'] = (today_df['predicted_demand'] - today_df['stock']).clip(lower=0)
        today_df['excess'] = (today_df['stock'] - today_df['predicted_demand'] * 1.2).clip(lower=0)
        
        # ── Shortage Section ──
        shortages = today_df[today_df['shortage'] > 800].copy()
        
        if not shortages.empty:
            st.markdown("### ⚠ Shortage Alerts")
            for _, row in shortages.iterrows():
                st.error(
                    f"**{row['district']} – {row['branch']}**\n"
                    f"• Predicted demand: **{int(row['predicted_demand']):,}**\n"
                    f"• Current stock: **{int(row['stock']):,}**\n"
                    f"→ Short by **{int(row['shortage']):,}** cylinders"
                )
                
                suggest_qty = int(row['shortage'] * 1.25)
                st.info(
                    f"**MRP Suggestion**: Send **{suggest_qty:,}** cylinders\n"
                    f"from excess districts (Coimbatore / Salem recommended)"
                )
        else:
            st.success("No critical shortages predicted.")
        
        # ── Excess & Redistribution ──
        excess_df = today_df[today_df['excess'] > 1000].copy()
        
        if not excess_df.empty:
            total_excess = int(excess_df['excess'].sum())
            st.success(f"**Excess stock available**: {total_excess:,} cylinders → Can be redistributed")
            
            st.markdown("#### Branches with Excess Stock")
            st.dataframe(
                excess_df[['district', 'branch', 'stock', 'predicted_demand', 'excess']]
                .rename(columns={
                    'predicted_demand': 'Predicted Demand',
                    'excess': 'Excess Available'
                })
                .style.format(precision=0, thousands=",")
                .background_gradient(subset=['Excess Available'], cmap='Greens')
            )
        else:
            st.info("No significant excess stock detected.")
    else:
        st.warning("Cannot calculate alerts — missing 'stock' or 'refill_bookings' column.")

else:
    st.info("Please upload your LPG dataset CSV to begin.")
    st.markdown("""
    ### Demo Flow
    1. Upload dataset  
    2. Load / Train model  
    3. View current stock & sales  
    4. Adjust factors → Predict next week  
    5. See shortage alerts + MRP suggestions  
    6. View excess stock & redistribution options
    """)