import streamlit as st
import pandas as pd
import numpy as np
import pickle

#load the model
with open('linear_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Sidebar inputs for user data
st.title("House Price Prediction App")

st.sidebar.header("Input Features")
bedrooms = st.sidebar.slider("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.slider("Bathrooms", 1.0, 5.0, 2.0)
sqft_living = st.sidebar.number_input("Square Feet Living", 500, 10000, 1500)
sqft_lot = st.sidebar.number_input("Square Feet Lot", 500, 20000, 5000)
floors = st.sidebar.slider("Floors", 1, 3, 1)
grade = st.sidebar.slider("Grade", 1, 13, 7)
sqft_above = st.sidebar.number_input("Square Feet Above", 500, 10000, 1200)
sqft_basement = st.sidebar.number_input("Square Feet Basement", 0, 5000, 0)

# Collect inputs into a DataFrame
input_data = pd.DataFrame({
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'sqft_living': [sqft_living],
    'sqft_lot': [sqft_lot],
    'floors': [floors],
    'grade': [grade],
    'sqft_above': [sqft_above],
    'sqft_basement': [sqft_basement]
})

# Predict using the loaded model
prediction = model.predict(input_data)

# Display prediction result
st.subheader("Prediction Result")
st.write(f"Estimated House Price: ${prediction[0]:,.2f}")