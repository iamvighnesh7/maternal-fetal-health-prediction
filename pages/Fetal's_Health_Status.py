#Gautami

import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('fetal_health_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit interface
st.title('Fetal Health Prediction')

st.write("""
    Enter the values for the following features to predict the fetal health:
""")

# User input for the top 5 features
abnormal_short_term_variability = st.number_input('Abnormal Short Term Variability (Short-term fluctuation level)')
mean_value_of_short_term_variability = st.number_input('Mean Value of Short Term Variability (Average fluctuation)')
percentage_of_time_with_abnormal_long_term_variability = st.number_input('Percentage of Time with Abnormal Long Term Variability (Percentage of time with irregular fluctuations)')
histogram_mean = st.number_input('Histogram Mean (Average health measurement)')
histogram_mode = st.number_input('Histogram Mode (Most common value)')

# Create a prediction button
if st.button('Predict'):
    # Prepare the data for prediction
    input_data = np.array([[abnormal_short_term_variability, mean_value_of_short_term_variability, 
                            percentage_of_time_with_abnormal_long_term_variability, histogram_mean, 
                            histogram_mode]])
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    st.write('1: Normal, 2: Suspect, 3: Pathological')
    # Display the prediction result
    st.write(f'Predicted fetal health class: {prediction[0]}')




