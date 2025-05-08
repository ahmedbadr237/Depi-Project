import pickle
import pandas as pd
import streamlit as st

# Load the model and scaler
model_path = 'models\\xgboost_model.pkl'
scaler_path = 'models\\scaler.pkl'
encoder_path = 'models\\label_encoder.pkl'
model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))
encoder = pickle.load(open(encoder_path, 'rb'))
feature_names = model.feature_names_in_.tolist()

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
        margin-top: 1rem;
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
        <h2>Sales Forecasting</h2>
    </div>
    """,
    unsafe_allow_html=True
)



# Input fields
col1, col2 = st.columns(2)
with col1:
    sales_channel = st.selectbox("Sales Channel", ['In-Store', 'Online', 'Distributor', 'Wholesale'])
    order_quantity = st.number_input("Order Quantity", min_value=0, value=0, step=1)
    discount_applied = st.number_input("Discount Applied", value=0.0, step=0.01, format="%.2f")

with col2:
    unit_cost = st.number_input("Unit Cost", value=0)
    unit_price = st.number_input("Unit Price", value=0)
    discount_percentage = st.number_input("Discount Percentage", value=0.0, step=0.01, format="%.2f")

total_revenue = st.number_input("Total Revenue", value=0)

submit = st.button("Predict Profit", type="primary")

if submit:
    try:
        input_data = pd.DataFrame({
            feature_names[0]: [sales_channel],
            feature_names[1]: [order_quantity],
            feature_names[2]: [float(discount_applied)],
            feature_names[3]: [float(unit_cost)],
            feature_names[4]: [float(unit_price)],
            feature_names[5]: [float(discount_percentage)],  
            feature_names[6]: [float(total_revenue)]
        })
        
        input_data['Sales Channel'] = encoder.transform(input_data['Sales Channel'])
        scaling = ["Sales Channel", "Order Quantity", "Discount Applied", "Unit Cost", "Unit Price", "Total Revenue"]
        input_data[scaling] = scaler.transform(input_data[scaling])
        
        profit_pred = model.predict(input_data)[0]
        st.markdown(f'<div class="prediction-card">Predicted Profit: <span class="profit-value">${profit_pred:.2f}</span></div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error in prediction pipeline: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)