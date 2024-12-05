import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="Hello, we're the Network Guardians. It's nice to meet you",
    page_icon="👋",
)

st.write("# Meet the team 👋")

col1, col2, col3 = st.columns(3, vertical_alignment='top')

with col1:
    st.markdown("## Brianna Persaud")
    st.image("https://static.streamlit.io/examples/cat.jpg")

with col2:
    st.markdown("## Isabel Loci")
    st.image("https://static.streamlit.io/examples/dog.jpg")

with col3:
    st.markdown("## Angelo Vitalino")
    st.image("https://static.streamlit.io/examples/owl.jpg")


