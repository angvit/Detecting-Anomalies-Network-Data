import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json


@st.cache_data
def load_network_data(fp):
    print('Loading data...')

    # read in the csv via the link
    df = pd.read_csv(fp)

    return(df)

st.title("Datasets")

st.subheader("UNSW_NB15 Merged Dataset")
st.write("This dataset is a combination of the datasets: \n- UNSW-NB15_1.csv \n- UNSW-NB15_2.csv \n- UNSW-NB15_3.csv \n- UNSW-NB15_4.csv")
fp = 'datasets/UNSW_NB15_merged.csv'
df = load_network_data(fp) 
st.dataframe(df.head(10))

st.divider()

st.subheader("UNSW_NB15 Cleaned Dataset")
st.write("Here is the dataset after cleaning!")
fp = 'datasets/UNSW_NB15_cleaned.csv'
df = load_network_data(fp) 
st.dataframe(df.head(10))