import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the pre-trained RandomForest model
model = pickle.load(open('app/model/rf_model.pkl', 'rb'))

# Load the dataset for default values
data = pd.read_csv('datasets/UNSW_NB15_cleaned.csv')

# Extract feature names from the dataset
all_features = data.drop(columns=["attack_cat_encoded", "is_anomaly"]).columns.tolist()

# Define feature importances (replace with actual values or load from CSV)
feature_importances = pd.read_csv('datasets/feature_importances.csv')

# Set the threshold for top features (e.g., top 10 features)
top_features = feature_importances.head(10)['Feature'].tolist()
low_importance_features = [f for f in all_features if f not in top_features]

# Calculate default values for less important features
default_values = data[low_importance_features].mean().to_dict()

st.title("Network Traffic Anomaly Detection")
st.write("Adjust the features to simulate network traffic and determine the likelihood of an attack and its type.")
st.write("Happy Hacking!")

# Collect user input for the top features
input_features = {}
for feature in top_features:
    if data[feature].dtype == 'float64' or data[feature].dtype == 'int64':
        input_features[feature] = st.number_input(
            feature,
            min_value=float(data[feature].min()),
            max_value=float(data[feature].max()),
            value=float(data[feature].mean()),
        )
    else:
        unique_values = data[feature].unique().tolist()
        input_features[feature] = st.selectbox(feature, unique_values)

# Add default values for less important features
for feature, default_value in default_values.items():
    input_features[feature] = default_value

# Convert user inputs and defaults to a DataFrame
input_features_df = pd.DataFrame([input_features])

# Encode categorical features
le_service = LabelEncoder()
le_service.classes_ = ["http", "ftp", "smtp", "dns", "other"]
input_features_df["service_encoded"] = le_service.fit_transform(input_features_df["service"])

le_state = LabelEncoder()
le_state.classes_ = ["INT", "FIN", "CON", "REQ", "RST"]
input_features_df["state_encoded"] = le_state.fit_transform(input_features_df["state"])

# Drop the original categorical columns
input_features_df = input_features_df.drop(columns=["service", "state"])

# Scale the features using StandardScaler (ensure scaler is fitted)
scaler = StandardScaler()
scaler.fit(data.drop(columns=["attack_cat_encoded", "is_anomaly"]))
scaled_features = scaler.transform(input_features_df)

# Predict using the loaded model
attack_proba = model.predict_proba(scaled_features)
predicted_class = model.predict(scaled_features)

# Map the predicted class to attack types
attack_mapping = pd.read_csv('datasets/attack_cat_mapping.csv', index_col=0).to_dict()["Encoded Value"]
predicted_attack = list(attack_mapping.keys())[list(attack_mapping.values()).index(predicted_class[0])]

# Display the results
st.write(f"### Predicted Attack Type: {predicted_attack}")
st.write(f"### Likelihood of Attack: {attack_proba[0][predicted_class[0]]:.2%}")
