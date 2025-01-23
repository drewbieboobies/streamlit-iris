import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained Random Forest model and scaler
rf_model = joblib.load('random_forest_model.pkl')  # Replace with the correct path
scaler = joblib.load('scaler.pkl')  # Assuming the scaler is saved separately

# Define the feature order based on model training
feature_columns = ['closest_mrt_dist', 'cbd_dist', 'floor_area_sqm', 'years_remaining', 'town_BEDOK', 'town_BISHAN', 'town_BUKIT BATOK', 'town_BUKIT MERAH', 'town_BUKIT PANJANG', 'town_BUKIT TIMAH', 'town_CENTRAL AREA', 'town_CHOA CHU KANG', 'town_CLEMENTI', 'town_GEYLANG', 'town_HOUGANG', 'town_JURONG EAST', 'town_JURONG WEST', 'town_KALLANG/WHAMPOA', 'town_MARINE PARADE', 'town_PASIR RIS', 'town_PUNGGOL', 'town_QUEENSTOWN', 'town_SEMBAWANG', 'town_SENGKANG', 'town_SERANGOON', 'town_TAMPINES', 'town_TOA PAYOH', 'town_WOODLANDS', 'town_YISHUN', 'flat_type_2 ROOM', 'flat_type_3 ROOM', 'flat_type_4 ROOM', 'flat_type_5 ROOM', 'flat_type_EXECUTIVE', 'flat_type_MULTI-GENERATION', 'storey_range_01 TO 05', 'storey_range_04 TO 06', 'storey_range_06 TO 10', 'storey_range_07 TO 09', 'storey_range_10 TO 12', 'storey_range_11 TO 15', 'storey_range_13 TO 15', 'storey_range_16 TO 18', 'storey_range_16 TO 20', 'storey_range_19 TO 21', 'storey_range_21 TO 25', 'storey_range_22 TO 24', 'storey_range_25 TO 27', 'storey_range_26 TO 30', 'storey_range_28 TO 30', 'storey_range_31 TO 33', 'storey_range_31 TO 35', 'storey_range_34 TO 36', 'storey_range_36 TO 40', 'storey_range_37 TO 39', 'storey_range_40 TO 42', 'flat_model_Apartment', 'flat_model_DBSS', 'flat_model_Improved', 'flat_model_Improved-Maisonette', 'flat_model_Maisonette', 'flat_model_Model A', 'flat_model_Model A-Maisonette', 'flat_model_Model A2', 'flat_model_Multi Generation', 'flat_model_New Generation', 'flat_model_Premium Apartment', 'flat_model_Premium Maisonette', 'flat_model_Simplified', 'flat_model_Standard', 'flat_model_Terrace', 'flat_model_Type S1']
# Main Page Title
st.title("HDB Resale Price Prediction App")
st.write("This app predicts the **HDB Resale Price** based on various features.")

def user_input_features():
    st.subheader("Enter Details:")
    # Numeric Inputs
    closest_mrt_dist = st.number_input('Distance to Closest MRT (meters)', min_value=0.0, max_value=10000.0, value=441.785021, step=0.1)
    cbd_dist = st.number_input('Distance to CBD (meters)', min_value=0.0, max_value=10000.0, value=2715.822202, step=0.1)
    floor_area_sqm = st.number_input('Floor Area (m²)', min_value=0.0, max_value=200.0, value=68.0, step=0.1)
    years_remaining = st.number_input('Years Remaining on Lease', min_value=0, max_value=99, value=66, step=1)

    # Categorical Inputs
    town = st.selectbox('Select Town', [
        'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH',
        'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST',
        'JURONG WEST', 'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN',
        'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN'
    ])
    flat_type = st.selectbox('Select Flat Type', ['2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION'])
    storey_range = st.selectbox('Select Storey Range', [
        '01 TO 05', '04 TO 06', '06 TO 10', '07 TO 09', '10 TO 12', '11 TO 15',
        '13 TO 15', '16 TO 18', '16 TO 20', '19 TO 21', '21 TO 25', '22 TO 24',
        '25 TO 27', '26 TO 30', '28 TO 30', '31 TO 33', '31 TO 35', '34 TO 36',
        '36 TO 40', '37 TO 39', '40 TO 42'
    ])
    flat_model = st.selectbox('Select Flat Model', [
        'Apartment', 'DBSS', 'Improved', 'Improved-Maisonette', 'Maisonette', 'Model A',
        'Model A-Maisonette', 'Model A2', 'Multi Generation', 'New Generation',
        'Premium Apartment', 'Premium Maisonette', 'Simplified', 'Standard', 'Terrace', 'Type S1'
    ])

    # Prepare data in the correct order
    data = {
        'closest_mrt_dist': closest_mrt_dist,
        'cbd_dist': cbd_dist,
        'floor_area_sqm': floor_area_sqm,
        'years_remaining': years_remaining
    }

    # One-hot encode categorical variables
    categorical_data = [
        1 if town == col.split('_')[1] else 0 for col in feature_columns if col.startswith('town_')
    ] + [
        1 if flat_type == col.split('_')[1] else 0 for col in feature_columns if col.startswith('flat_type_')
    ] + [
        1 if storey_range == col.split('_')[1] else 0 for col in feature_columns if col.startswith('storey_range_')
    ] + [
        1 if flat_model == col.split('_')[1] else 0 for col in feature_columns if col.startswith('flat_model_')
    ]

    # Combine numeric and categorical data
    numeric_df = pd.DataFrame([data])
    categorical_df = pd.DataFrame([categorical_data])
    combined_df = pd.concat([numeric_df, categorical_df], axis=1)
    combined_df.columns = feature_columns  # Ensure correct column order

    return combined_df

# Collect user inputs
df = user_input_features()

if st.button("Predict"):
    st.subheader('User Input Parameters')
    st.write(df)

    # Normalize the numeric features
    #numeric_columns = ['closest_mrt_dist', 'cbd_dist', 'floor_area_sqm', 'years_remaining']
    #df[numeric_columns] = scaler.transform(df[numeric_columns])

    # Predict the resale price using the trained Random Forest model
    prediction = rf_model.predict(df)

    st.subheader('Prediction')
    st.write(f'The predicted resale price is: ${prediction[0]:,.2f}')
