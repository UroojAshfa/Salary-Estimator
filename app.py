import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle 

# Load the trained model
model = tf.keras.models.load_model('model.keras')

# Load encoders and scaler
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)



# Streamlit UI
st.title("Estimated Salary Prediction")

# User inputs
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
credit_score = st.number_input('Credit Score')
balance = st.number_input('Balance')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.select_slider('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
has_exited = st.selectbox('Has the person exited?', [0, 1])

# Encode categorical variables
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

gender_encoded = label_encoder_gender.transform([gender])[0]

# Create input DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [has_exited]
})

# Merge with encoded geography
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# Scale input data
input_data_scaled = scaler.transform(input_data)

# Predict Salary
predicted_salary = model.predict(input_data_scaled)[0][0]

# Display Results
st.subheader("Predicted Estimated Salary")
st.write(f"Estimated Salary: **${predicted_salary:,.2f}**")
