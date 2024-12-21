import streamlit as st
import requests
import pandas as pd
import os
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
import joblib 

st.title("🚀 Audience_Rating prediction for Tomatoes_movies_dataset_by_zoho 😃")
options = ['G', 'PG', 'PG-13', 'R', 'NC17', 'NR']
value1 = st.selectbox("Enter value for Rating", options)
value2 = st.number_input("Enter value for Runtime in Minutes ", format="%.2f")
options1 = ['Rotten', 'Fresh' ,'Certified Fresh']
value3 = st.selectbox("Enter value for Tomatometer Status", options1)
value4 = st.number_input("Enter value Tomatometer Rating", format="%.2f")
value5 = st.number_input("Enter value Tomatometer Count", format="%.2f")

value1=[i for i,j in enumerate(options) if value1==j][0]
value3=[i for i,j in enumerate(options1) if value3==j][0]


scaler=joblib.load('scaler.h5')
scaler_y=joblib.load('scaler_y.h5')
loaded_model = load_model('ann_model.h5')
example=pd.DataFrame([value1,value2,value3,value4,value5]).T
example=scaler.transform(example)
# Use the loaded model for predictions or further training
y_pred = loaded_model.predict(example)
y_pred = scaler_y.inverse_transform(y_pred)

st.write(f"Audiance Rating is {round(float(y_pred[0]),2)}")



