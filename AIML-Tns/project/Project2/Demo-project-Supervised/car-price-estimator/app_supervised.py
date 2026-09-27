import streamlit as st
import requests

st.set_page_config(
    page_title="Car Price Estimator", 
    page_icon="🚗", 
    layout="centered"
)

st.title("🚗 Used Car Price Estimator")
st.markdown("Predict estimated resale market value using Supervised Machine Learning (Decision Tree Regression).")
st.divider()

with st.form("car_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Vehicle Age (Years)", min_value=1, max_value=20, value=4)
        
        fuel = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "Electric"])
        
    with col2:
        mileage = st.number_input(
            "Total Mileage (KM)", 
            min_value=1000, 
            max_value=250000, 
            value=45000, 
            step=5000
        )
        
    submit_btn = st.form_submit_button("Calculate Estimated Value", use_container_width=True)

if submit_btn:
    payload = {
        "age_years": int(age),
        "mileage_km": float(mileage),
        "fuel_type": fuel
    }
    
    API_URL = "http://127.0.0.1:8000/predict"
    
    with st.spinner("Calculating market valuation..."):
        try:
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                price = result["estimated_price_usd"]
                
                st.subheader("Valuation Summary")
                
                st.metric("Estimated Market Price", f"${price:,.2f}")
                
                st.info(f"Vehicle Profile: {age} year old {fuel} car with {mileage:,} KM recorded.")
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to FastAPI server. Ensure main_supervised.py is running on port 8000.")