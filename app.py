import streamlit as st
import pickle   
import numpy as np


with open("model.pkl", "rb") as file:
    model = pickle.load(file)
with open("scaler.pkl", "rb") as file:  
    scaler = pickle.load(file)
st.title(" Male OR Female 🚀 ")
st.markdown("Amir Rizk🚀🚀🚀")

long_hair = st.selectbox("Select the long hair:", [0, 1])
nose_wide = st.selectbox("Select the nose wide:", [0, 1])
nose_long = st.selectbox("Select the nose long:", [0, 1])
lips_thin = st.selectbox("Select the lips thin:", [0, 1])
distance_nose_to_lip_long = st.selectbox("Select the distance:", [0, 1])

forehead_width_cm = st.number_input("Enter the Forehead width:", min_value=11.4, max_value=15.5, value=13.0)
forehead_height_cm = st.number_input("Enter the Forehead height:", min_value=5.1, max_value=7.1, value=6.0)

if st.button("Predict 💡"):
    # ✅ تحويل البيانات إلى NumPy array
    input_data = np.array([[long_hair, forehead_width_cm, forehead_height_cm, nose_wide, nose_long, lips_thin, distance_nose_to_lip_long]], dtype=np.float64)

    # ✅ تأكد إنك بتستخدم الـ Scaler
    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)

    st.success(f" (Male = 1 , Female =0) Prediction= :  {prediction[0]}")


