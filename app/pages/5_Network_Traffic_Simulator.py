import streamlit as st
import pandas as pd 
import pickle
import plotly.express as px


model = pickle.load(open('app/model/rf_model.pkl', 'rb'))

st.title("Network Traffic Anomaly Detection")

st.write("Adjust the features to simluate network traffic and determine the likelihood of an attack and its type.")
st.write("Happy Hacking!")

