import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load model and column names
@st.cache_resource
def load_model():
    model_path = 'models/price_model.pkl'
    columns_path = 'models/model_columns.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(columns_path):
        return None, None
        
    model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)
    return model, model_columns

model, model_columns = load_model()

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏠 House Price Prediction System")
st.write("Enter the details of the house below to estimate its price.")

if model is None:
    st.error("Model not found! Please run the training script first.")
else:
    col1, col2 = st.columns(2)
    
    with col1:
        size = st.number_input("Size (sq ft)", min_value=500, max_value=10000, value=1500, step=50)
        bedrooms = st.slider("Bedrooms", 1, 10, 3)
        bathrooms = st.slider("Bathrooms", 1, 5, 2)
        
    with col2:
        age = st.slider("Age of House (years)", 0, 100, 10)
        location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])

    if st.button("Predict Price", type="primary"):
        # Create a dataframe for the input
        input_data = pd.DataFrame({
            'Size': [size],
            'Bedrooms': [bedrooms],
            'Bathrooms': [bathrooms],
            'Age': [age],
            'Location': [location]
        })
        
        # Preprocess input (One-hot encoding)
        input_data = pd.get_dummies(input_data, columns=['Location'])
        
        # Ensure all columns from training exist
        for col in model_columns:
            if col not in input_data.columns:
                input_data[col] = 0
                
        # Reorder columns to match training data
        input_data = input_data[model_columns]
        
        # Predict
        prediction = model.predict(input_data)[0]
        
        st.success(f"### Estimated Price: ${prediction:,.2f}")
        
        # Verification / Explanation text
        st.info(f"""
        **Prediction Details:**
        - Size: {size} sq ft
        - Bedrooms: {bedrooms}
        - Location: {location}
        """)
