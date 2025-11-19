import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.keras')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open('onehot_encoder_geography.pkl', 'rb') as file:
    onehot_encoder_geopraphy = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title('Customer churn pridiction')

# input user
geography = st.selectbox('Geography', onehot_encoder_geopraphy.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credict_score = st.number_input('Credict score')
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider('Tenure',0, 10)
num_of_product = st.slider("Number of product", 1, 4)
has_cr_card = st.selectbox("Has Cr Card", [0, 1])
is_activate_number = st.selectbox("Is Activate Number", [0,1])

input_data = {
    "CreditScore": [credict_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_product],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_activate_number],
    "EstimatedSalary": [estimated_salary]
}

geo_encoded = onehot_encoder_geopraphy.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geopraphy.get_feature_names_out(["Geography"]))

input_df = pd.DataFrame(input_data)
input_df = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_df)
prediction = model.predict(input_data_scaled)
predict_propba = prediction[0][0]

st.write("Churn propbability: ",predict_propba)
if predict_propba > 0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")