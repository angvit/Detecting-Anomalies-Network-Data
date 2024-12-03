import pandas as pd
import streamlit as st
import pickle

st.write('hello')

@st.cache_data
def load_network_data(fp):
    print("Running load_network_data...")
    df = pd.read_csv(fp)
    st.write(df.head())

fp = './datasets/UNSW_NB15_merged.csv'
df = load_network_data(fp)