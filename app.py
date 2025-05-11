import pickle
import pandas as pd
import streamlit as st
import os
import numpy as np
from sklearn.base import BaseEstimator
# Load the model and scaler
# Add this after imports and before loading models
class TreeBasedModels(BaseEstimator):
    def __init__(self, model=None):
        self.model = model
        
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        return self.model.predict(X)
demand_model_path = os.path.join('models', 'xgboost_model_demand.pkl')
sales_model_path = os.path.join('models', 'xgboost_model_sales.pkl')
encoder_path = os.path.join('models', 'sales_channel_encoder.pkl')
demand_model = pickle.load(open(demand_model_path, 'rb'))
sales_model = pickle.load(open(sales_model_path, 'rb'))
encoder = pickle.load(open(encoder_path, 'rb'))

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

history_data = pd.read_csv(r"notebooks\Data\history_features.csv")

# Check sales range in history data
def get_historical_values(store_id, product_id, order_date):
    """
    Get the last known value for a specific store and product combination
    """
    order_date = pd.to_datetime(order_date)
    
    # More precise date filtering
    mask = (history_data['_StoreID'] == store_id) & \
           (history_data['_ProductID'] == product_id) & \
           (pd.to_datetime(history_data['OrderDate']) < order_date)
    
    if mask.any():
        # Get actual historical statistics instead of using same value
        recent_data = history_data[mask].sort_values('OrderDate', ascending=False)
        return {
            'last_value': recent_data['Order Quantity'].iloc[0],
            'rolling_mean': recent_data['Order Quantity'].head(30).mean(),
            'rolling_std': recent_data['Order Quantity'].head(30).std(),
            'rolling_min': recent_data['Order Quantity'].head(30).min(),
            'rolling_max': recent_data['Order Quantity'].head(30).max()
        }
    else:
        # Return dictionary with summary statistics
        return {
            'last_value': history_data['Order Quantity'].median(),
            'rolling_mean': history_data['Order Quantity'].mean(),
            'rolling_std': history_data['Order Quantity'].std(),
            'rolling_min': history_data['Order Quantity'].min(),
            'rolling_max': history_data['Order Quantity'].max()
        }

# In your DataFrame creation

# Custom CSS
st.markdown(
    """
    <style>
    .header-card {
        background-color: rgba(0,0,0,0.6);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .prediction-card {
        background-color: #141414;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        text-align: center;
        font-weight: bold;
    }
    .profit-value {
        color: #00ff00;  /* Bright green color */
        font-size: 1.2em;
    }
    .stApp {
        background-image: url("https://img.freepik.com/free-photo/dynamic-data-visualization-3d_23-2151904311.jpg");
        background-size: cover;
        background-position: center;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

# Header Card
st.markdown(
    """
    <div class="header-card">
        <h2>Sales and Demand Forecasting</h2>
    </div>
    """,
    unsafe_allow_html=True
)



# Input fields
col1, col2 = st.columns(2)
with col1:
    sales_channel = st.selectbox("Sales Channel", ['In-Store', 'Online', 'Distributor', 'Wholesale'])
    store_id = st.number_input("Store ID", min_value=1, value=1, step=1,max_value=367)
    order_date = st.date_input("Order Date", value=pd.to_datetime("2023-01-01"))
    product_id = st.number_input("Product ID", min_value=1, value=1, step=1,max_value=47)
    

with col2:
    unit_cost = st.number_input("Unit Cost", value=0)
    unit_price = st.number_input("Unit Price", value=0)
    discount_percentage = st.number_input("Discount Percentage", value=0.0, step=0.01, format="%.2f")
    discount_applied = st.number_input("Discount Applied", value=0.0, step=0.01, format="%.2f")


submit = st.button("Predict Demand and Sales", type="primary")
# Check sales range in history data
if submit:
    try:
        # Create base input data with all common features
        base_data = pd.DataFrame({
            '_StoreID': [store_id],
            '_ProductID': [product_id],
            'Discount Applied': [discount_applied],
            'Unit Price': [unit_price],
            'year': [order_date.year],
            'day_of_month': [order_date.day],
            'week_of_year': [order_date.isocalendar()[1]],
            'month_sin': [np.sin(2 * np.pi * order_date.month / 12)],
            'month_cos': [np.cos(2 * np.pi * order_date.month / 12)],
            'day_of_week_sin': [np.sin(2 * np.pi * order_date.weekday() / 7)],
            'day_of_week_cos': [np.cos(2 * np.pi * order_date.weekday() / 7)]
        })

        # Encode sales channel
        sales_channel_array = np.array([sales_channel]).reshape(-1, 1)
        encoded_channel = encoder.transform(sales_channel_array).toarray()[0]
        channel_names = encoder.get_feature_names_out(['Sales Channel'])
        for name, value in zip(channel_names, encoded_channel):
            base_data[name] = value

        # Create demand input data
        demand_input = base_data.copy()
        demand_input['Unit Cost'] = unit_cost

        # Add historical features using get_historical_values function
        hist_values = get_historical_values(store_id, product_id, order_date)
        
        # Get historical statistics
        hist_stats = get_historical_values(store_id, product_id, order_date)

        # Add lag features using last known value
        for i in range(1, 6):
            demand_input[f'lag_{i}'] = hist_stats['last_value']

        # Add rolling and ewm features using appropriate statistics
        for window in [7, 14, 30]:
            demand_input[f'rolling_mean_{window}'] = hist_stats['rolling_mean']
            demand_input[f'rolling_std_{window}'] = hist_stats['rolling_std']
            demand_input[f'rolling_min_{window}'] = hist_stats['rolling_min']
            demand_input[f'rolling_max_{window}'] = hist_stats['rolling_max']
            demand_input[f'ewm_mean_{window}'] = hist_stats['rolling_mean']
            demand_input[f'ewm_std_{window}'] = hist_stats['rolling_std']

        # Add diff feature
        demand_input['diff_1'] = hist_stats['last_value'] - hist_stats['rolling_mean']
        
        # Ensure correct column order for demand model
        demand_input = demand_input[demand_features]
        
        # Create sales input data and make predictions
        sales_input = base_data.copy()
        demand_pred = demand_model.predict(demand_input)[0]
        sales_input['Order Quantity'] = demand_pred
        sales_input = sales_input[sales_features]
        sales_pred = sales_model.predict(sales_input)[0]
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="prediction-card">Predicted Demand: '
                f'<span class="profit-value">{demand_pred:0.0f} units</span></div>',
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f'<div class="prediction-card">Predicted Sales: '
                f'<span class="profit-value">${sales_pred:,.2f}</span></div>',
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"Error in prediction pipeline: {str(e)}")
        st.error(f"Input shape: {demand_input.shape if 'demand_input' in locals() else 'not created'}")

st.markdown('</div>', unsafe_allow_html=True)
