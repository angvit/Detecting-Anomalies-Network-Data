import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.header("Data Visualizations")

st.warning("Upon closer inspection of the attack_cat column of the dataset, we noticed the number of 'normal' values overpowered the attacks.")

@st.cache_data
def load_network_data(fp):
    print('Loading data...')
    return pd.read_csv(fp)

fp = 'datasets/UNSW_NB15_cleaned.csv'
df = load_network_data(fp)

df = df.sample(10000, random_state=42)

fig = px.bar(
    df,
    x='attack_cat',
    title='Count of each Attack Category',
    labels={'attack_cat': 'Attack Category', 'count': 'Count'},
    color_discrete_sequence=['#FFD700']
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("### Corelation Matrix")
st.image('app/images/content/correlation_matrix.png', use_container_width=True)
