import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


@st.cache_resource
def load_model():
    return pickle.load(open('app/model/rf_model.pkl', 'rb'))


@st.cache_data
def load_dataset():
    return pd.read_csv('datasets/UNSW_NB15_cleaned.csv')


@st.cache_data
def load_feature_importances():
    return pd.read_csv('datasets/feature_importances.csv')


def collect_user_inputs(data, top_features):
    input_features = {}
    for feature in top_features:
        if data[feature].dtype == 'float64' or data[feature].dtype == 'int64':
            input_features[feature] = st.number_input(
                feature,
                min_value=float(data[feature].min()),
                max_value=float(data[feature].max()),
                value=float(data[feature].mean()),
                help=f"Range: {data[feature].min()} - {data[feature].max()}, Mean: {data[feature].mean():.2f}"
            )
        else:
            unique_values = data[feature].unique().tolist()
            input_features[feature] = st.selectbox(feature, unique_values)
    return input_features


def get_default_values(data, low_importance_features):
    default_values = {}
    for feature in low_importance_features:
        if data[feature].dtype == 'float64' or data[feature].dtype == 'int64':
            default_values[feature] = data[feature].mean()
        else:
            default_values[feature] = data[feature].mode()[0]
    return default_values


def preprocess_inputs(input_features, data, default_values):
    input_features_df = pd.DataFrame([input_features])

    for feature, default_value in default_values.items():
        if feature not in input_features_df:
            input_features_df[feature] = default_value

    input_features_df = input_features_df[data.drop(columns=["attack_cat_encoded", "is_anomaly"]).columns]

    scaler = StandardScaler()
    scaler.fit(data.drop(columns=["attack_cat_encoded", "is_anomaly"]))
    scaled_features = scaler.transform(input_features_df)
    return scaled_features


def make_prediction(model, scaled_features):
    attack_proba = model.predict_proba(scaled_features)
    predicted_class = model.predict(scaled_features)

    attack_mapping = pd.read_csv('datasets/attack_cat_mapping.csv', index_col=0).to_dict()["Encoded Value"]
    predicted_attack = list(attack_mapping.keys())[list(attack_mapping.values()).index(predicted_class[0])]

    return predicted_attack, attack_proba[0][predicted_class[0]]


def main():
    model = load_model()
    data = load_dataset()
    feature_importances = load_feature_importances()

    top_features = feature_importances.head(5)['Feature'].tolist()
    low_importance_features = [f for f in data.columns if f not in top_features and f not in ["attack_cat_encoded", "is_anomaly"]]
    default_values = get_default_values(data, low_importance_features)

    st.title("Network Traffic Anomaly Detection")
    st.write("Adjust the features to simulate network traffic and determine the likelihood of an attack and its type.")
    st.write("Happy Hacking!")

    user_inputs = collect_user_inputs(data, top_features)

    for feature, default_value in default_values.items():
        user_inputs[feature] = default_value

    scaled_features = preprocess_inputs(user_inputs, data, default_values)

    predicted_attack, attack_likelihood = make_prediction(model, scaled_features)

    st.write(f"### Predicted Attack Type: {predicted_attack}")
    st.write(f"### Likelihood of Attack: {attack_likelihood:.2%}")


main()