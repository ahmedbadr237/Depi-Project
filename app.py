# ──────────────────────────────────────────────────────────────
# 1. IMPORTS & SETUP
# ──────────────────────────────────────────────────────────────
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator

# ──────────────────────────────────────────────────────────────
# 2. MODEL WRAPPER CLASS
# ──────────────────────────────────────────────
class TreeBasedModels(BaseEstimator):
    def __init__(self, model=None):
        self.model = model
    def predict(self, X):
        return self.model.predict(X)

# ──────────────────────────────────────────────────────────────
# 3. PATHS & CONSTANTS
# ──────────────────────────────────────────────────────────────
MODEL_DIR = 'models'
DATA_PATH = 'notebooks/Data/history_features.csv'

DEMAND_MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model_demand.pkl')
SALES_MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model_sales.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'sales_channel_encoder.pkl')

# ──────────────────────────────────────────────────────────────
# 4. LOAD MODELS AND ENCODER
# ──────────────────────────────────────────────
demand_model = pickle.load(open(DEMAND_MODEL_PATH, 'rb'))
sales_model = pickle.load(open(SALES_MODEL_PATH, 'rb'))
encoder = pickle.load(open(ENCODER_PATH, 'rb'))

# ──────────────────────────────────────────────────────────────
# 5. FEATURE LISTS
# ──────────────────────────────────────────────────────────────
demand_features = ['_StoreID', '_ProductID', 'Discount Applied', 'Unit Cost', 'Unit Price',
       'year', 'day_of_month', 'week_of_year', 'month_sin', 'month_cos',
       'day_of_week_sin', 'day_of_week_cos', 'Sales Channel_In-Store',
       'Sales Channel_Online', 'Sales Channel_Wholesale', 'lag_1', 'lag_2',
       'lag_3', 'lag_4', 'lag_5', 'rolling_mean_7', 'rolling_std_7',
       'rolling_min_7', 'rolling_max_7', 'ewm_mean_7', 'ewm_std_7',
       'rolling_mean_14', 'rolling_std_14', 'rolling_min_14', 'rolling_max_14',
       'ewm_mean_14', 'ewm_std_14', 'rolling_mean_30', 'rolling_std_30',
       'rolling_min_30', 'rolling_max_30', 'ewm_mean_30', 'ewm_std_30',
       'diff_1']
  
sales_features = ['_StoreID', '_ProductID', 'Order Quantity', 'Discount Applied',
       'Unit Price', 'year', 'day_of_month', 'week_of_year', 'month_sin',
       'month_cos', 'day_of_week_sin', 'day_of_week_cos',
       'Sales Channel_In-Store', 'Sales Channel_Online',
       'Sales Channel_Wholesale']

# ──────────────────────────────────────────────────────────────
# 6. LOAD HISTORICAL DATA
# ──────────────────────────────────────────────────────────────
if os.path.exists(DATA_PATH):
    history_data = pd.read_csv(DATA_PATH)
else:
    st.error(f"File not found: {DATA_PATH}")

# ──────────────────────────────────────────────────────────────
# 7. FUNCTION TO EXTRACT HISTORICAL STATS
# ──────────────────────────────────────────────────────────────
def get_historical_values(store_id: int, product_id: int, order_date: str) -> dict:
    order_date = pd.to_datetime(order_date)
    mask = (
        (history_data['_StoreID'] == store_id) &
        (history_data['_ProductID'] == product_id) &
        (pd.to_datetime(history_data['OrderDate']) < order_date)
    )

    if mask.any():
        recent = history_data[mask].sort_values('OrderDate', ascending=False)
        values = recent['Order Quantity']
        
        # Calculate proper lags
        lags = {f'lag_{i}': values.iloc[i-1] if i-1 < len(values) else values.median() 
                for i in range(1, 6)}
        
        # Calculate proper rolling statistics for each window
        rolling_stats = {}
        for window in [7, 14, 30]:
            window_data = values.head(window)
            rolling_stats.update({
                f'rolling_mean_{window}': window_data.mean(),
                f'rolling_std_{window}': window_data.std(),
                f'rolling_min_{window}': window_data.min(),
                f'rolling_max_{window}': window_data.max()
            })
        
        # Calculate proper EWM statistics
        ewm_stats = {}
        for window in [7, 14, 30]:
            span = window
            window_data = values.head(window)
            ewm_obj = window_data.ewm(span=span)
            ewm_stats.update({
                f'ewm_mean_{window}': ewm_obj.mean().iloc[-1],
                f'ewm_std_{window}': ewm_obj.std().iloc[-1]
            })
        
        # Calculate diff
        diff_1 = values.iloc[0] - values.iloc[1] if len(values) > 1 else 0
        
        return {**lags, **rolling_stats, **ewm_stats, 'diff_1': diff_1}
    else:
        # Fallback values using global statistics
        default_stats = {}
        values = history_data['Order Quantity']
        
        # Default lags
        for i in range(1, 6):
            default_stats[f'lag_{i}'] = values.median()
        
        # Default rolling and ewm stats
        for window in [7, 14, 30]:
            default_stats.update({
                f'rolling_mean_{window}': values.mean(),
                f'rolling_std_{window}': values.std(),
                f'rolling_min_{window}': values.min(),
                f'rolling_max_{window}': values.max(),
                f'ewm_mean_{window}': values.ewm(span=window).mean().iloc[-1],
                f'ewm_std_{window}': values.ewm(span=window).std().iloc[-1]
            })
        
        default_stats['diff_1'] = 0
        return default_stats

# ──────────────────────────────────────────────────────────────
# 8. STREAMLIT STYLING
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .header-card { background-color: rgba(0,0,0,0.6); padding: 1.5rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem; }
    .prediction-card { background-color: #141414; padding: 1rem; border-radius: 10px; margin: 0.5rem; text-align: center; font-weight: bold; }
    .profit-value { color: #00ff00; font-size: 1.2em; }
    .stApp { background-image: url("https://img.freepik.com/free-photo/dynamic-data-visualization-3d_23-2151904311.jpg"); background-size: cover; background-position: center; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-card">
    <h2>Sales and Demand Forecasting</h2>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# 9. INPUT FIELDS
# ──────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    sales_channel = st.selectbox("Sales Channel", ['In-Store', 'Online', 'Wholesale'])
    store_id = st.selectbox("Store ID", [i for i in range(1,368)])
    order_date = st.date_input("Order Date", value=pd.to_datetime("2023-01-01"))

with col2:
    unit_cost = st.number_input("Unit Cost", value=1000)
    unit_price = st.number_input("Unit Price", value=1500)
    discount_applied = st.number_input("Discount Applied", value=0.0, step=0.01, format="%.2f")

product_id = st.selectbox("Product ID",[i for i in range(1,48)])
# ──────────────────────────────────────────────────────────────
# 10. PREDICTION PIPELINE
# ──────────────────────────────────────────────────────────────
if st.button("Predict Demand and Sales", type="primary"):
    try:
        # Build base input
        base_data = pd.DataFrame({
            '_StoreID': [store_id], '_ProductID': [product_id], 'Discount Applied': [discount_applied],
            'Unit Price': [unit_price], 'year': [order_date.year], 'day_of_month': [order_date.day],
            'week_of_year': [order_date.isocalendar()[1]],
            'month_sin': [np.sin(2 * np.pi * order_date.month / 12)],
            'month_cos': [np.cos(2 * np.pi * order_date.month / 12)],
            'day_of_week_sin': [np.sin(2 * np.pi * order_date.weekday() / 7)],
            'day_of_week_cos': [np.cos(2 * np.pi * order_date.weekday() / 7)]
        })

        # Encode sales channel
        channel_arr = np.array([sales_channel]).reshape(-1, 1)
        encoded_channel = encoder.transform(channel_arr).toarray()[0]
        for name, val in zip(encoder.get_feature_names_out(['Sales Channel']), encoded_channel):
            base_data[name] = val

        # Demand input with historical features
        demand_input = base_data.copy()
        demand_input['Unit Cost'] = unit_cost
        hist_features = get_historical_values(store_id, product_id, order_date)

        # Add all historical features at once
        for feature_name, value in hist_features.items():
            demand_input[feature_name] = value

        demand_input = demand_input[demand_features]

        # Predict demand
        demand_pred = demand_model.predict(demand_input)[0]

        # Build sales input and predict
        sales_input = base_data.copy()
        sales_input['Order Quantity'] = demand_pred
        sales_input = sales_input[sales_features]
        sales_pred = sales_model.predict(sales_input)[0]

        # Output
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="prediction-card">Predicted Demand: <span class="profit-value">{demand_pred:0.0f} units</span></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="prediction-card">Predicted Sales: <span class="profit-value">${sales_pred:,.2f}</span></div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error in prediction pipeline: {str(e)}")
