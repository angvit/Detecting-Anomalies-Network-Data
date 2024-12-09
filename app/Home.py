import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide"
)


st.title("🌐 Detecting Anomalies in Network Traffic Data 🌐")

st.markdown("Link to github repo: **https://github.com/angvit/Detecting-Anomalies-Network-Data**")
st.markdown("### Background")
st.write("The most common risks to a network's security are attacks such as brute force, denial of service (DOS), and backdoor attacks from within a network. " 
        "The undpredictable nature of network behavior makes it necessary to have a proactive approach to detect and prevent such attacks.")

st.markdown("### Project Purpose")
st.write("The project aims to detect anomalies in network traffic data using machine learning algorithms.")

st.markdown("### Dataset")
st.markdown("The dataset used for this project is the **[UNSW-NB15 dataset](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15/discussion?sort=undefined)**, "
            "which contains real network traffic data and synthetic attacks.")

st.markdown("### Steps in the Project")
st.write("Here are the key steps we followed in this project: ")
st.markdown("""
1. **Understanding the dataset**: Analyzed the UNSW-NB15 dataset to identify key features and classes.
2. **Data Preprocessing**:
    - Concatenated files into one merged dataset 
    - Filled in missing values
    - Dropped unnecessary columns
    - Label Encoded the categorical features
    - Standardized Features
3. **Exploratory Data Analysis (EDA)**: 
    - Visualized distributions of features and their unique values
    - Plotted features grouped by categories, normal and attack
4. **Random Forest Model**:
    - Found optimal hyperparameters using GridSearchCV
    - Detected anomalies and multi-classification of attack types
5. **Model Evaluation**:
    - Used metrics such as accuracy, precision, recall, f1, classification report, and confusion matrices
""")