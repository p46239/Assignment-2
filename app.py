
import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.title('My Streamlit App')
st.write('Hello, Streamlit!')

st.subheader('Rossmann Sales Prediction')
st.write('This is a placeholder for your Streamlit application content related to Rossmann Sales Prediction.')
st.write('You can add more interactive elements, visualizations, and predictions here.')

# Load the saved models and data
@st.cache_resource # Cache the model loading for efficiency
def load_model():
    model_data = joblib.load('sales_model.sav')
    return model_data

model_data = load_model()
lin_reg = model_data['lin_reg']
log_reg = model_data['log_reg']
scaler = model_data['scaler']
FEATURE_COLUMNS = model_data['feature_columns']
store_info = model_data['store_info']

st.write("Models and associated data loaded successfully!")
st.write(f"Loaded {len(FEATURE_COLUMNS)} features.")
st.write("First 5 feature columns:", FEATURE_COLUMNS[:5])

# Example of displaying store info
st.subheader("Store Sales Statistics")
st.dataframe(store_info.head())
