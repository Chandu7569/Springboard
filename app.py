import streamlit as st
import pandas as pd
import joblib

# --- Load Model & Preprocessor ---
try:
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('model.joblib')
except FileNotFoundError:
    st.error("Files missing! Put 'preprocessor.joblib' and 'model.joblib' in same folder.")
    st.stop()

# --- Predict Function ---
def predict_shipment(data):
    try:
        transformed = preprocessor.transform(data)
        prob = model.predict_proba(transformed)[0][1]
        threshold = 0.75
        prediction = 1 if prob >= threshold else 0
        return prediction, prob
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return None, None

# --- UI ---
st.title("📦 Delivery Time Prediction App")
st.write("Fill details to predict shipment delivery time:")

col1, col2 = st.columns(2)

with col1:
    warehouse_block = st.selectbox("Warehouse Block", ['A', 'B', 'C', 'D', 'E', 'F'])
    shipment_mode = st.selectbox("Mode of Shipment", ['Ship', 'Flight', 'Road'])
    customer_care_calls = st.number_input("Customer Care Calls", min_value=0, max_value=20, value=3)
    customer_rating = st.slider("Customer Rating", 1, 5, 3)  # ✅ added
    cost_of_product = st.number_input("Cost of Product (₹)", min_value=1, value=100)

with col2:
    prior_purchases = st.number_input("Prior Purchases", min_value=0, max_value=20, value=5)
    product_importance = st.selectbox("Product Importance", ['Low', 'Medium', 'High'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    discount_offered = st.number_input("Discount Offered (%)", min_value=0, max_value=100, value=10)
    weight = st.number_input("Weight (Kg)", min_value=0.1, value=5.0)

# ✅ Derived Feature: Cost to Weight Ratio
cost_to_weight_ratio = cost_of_product / (weight * 1000)

# --- Predict Button ---
if st.button("Predict Delivery Status"):
    input_df = pd.DataFrame([{
        'Warehouse_block': warehouse_block,
        'Mode_of_Shipment': shipment_mode,
        'Customer_care_calls': customer_care_calls,
        'Customer_rating': customer_rating,            # ✅ added
        'Cost_of_the_Product': cost_of_product,
        'Prior_purchases': prior_purchases,
        'Product_importance': product_importance,
        'Gender': gender,
        'Discount_offered': discount_offered,
        'Weight_in_gms': weight * 1000,
        'Cost_to_Weight_Ratio': cost_to_weight_ratio  # ✅ added
    }])

    pred, prob = predict_shipment(input_df)

    if pred is not None:
        st.subheader("📊 Prediction Result")
        st.write(f"Probability of On-Time Delivery: **{prob:.2f}**")

        if pred == 1:
            st.success("✅ Shipment will reach **ON TIME**")
        else:
            st.error("⚠️ Shipment may be **DELAYED**")
