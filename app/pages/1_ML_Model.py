import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


st.set_page_config(
    page_title="ML Model",
    page_icon="🌲",
    layout="centered"
)


@st.cache_data
def load_data():
    data = pd.read_csv('datasets/UNSW_NB15_cleaned.csv')
    return data


@st.cache_data
def load_attack_mapping():
    mapping = pd.read_csv('datasets/attack_cat_mapping.csv')
    return mapping


st.markdown("# 🌲 Random Forest for Anomaly Detection 🌲")
st.markdown("")

st.markdown("### Why Random Forest Algorithm?")
st.markdown(""" 
We chose to implement Random Forest due to its advantages:
- **Feature Importance**: It provides the importances of the individual features contributing to the predictions.
- **Handling of Class Imbalances**: The `class_weight='balanced'` hyperparameter adjusts for imbalances in the dataset.
- **Flexibility**: It performs well on datasets with mixed types (categorical and numerical).
""")
st.markdown("---")  # Horizontal line to separate sections
st.markdown("")

# Feature Importance Section
st.markdown("### Which Features Contributed Most to the Model?")
st.markdown("")

feature_importances = pd.read_csv("./datasets/feature_importances.csv")
feature_importances = feature_importances.sort_values(by="Importance", ascending=False)
top_feature_importances = feature_importances.head(10)

# Plot the horizontal bar chart
fig = px.bar(
    top_feature_importances,
    x='Importance',
    y='Feature',
    orientation='h',  
    title='Top 10 Feature Importances',
    labels={'Importance': 'Feature Importance', 'Feature': 'Feature Name'},
    color='Importance',  
    color_continuous_scale='Viridis'
)

fig.update_layout(
    xaxis_title="Importance",
    yaxis_title="Features",
    template="plotly_white",
    title_font=dict(size=20),
    title_x=0.25,  # Center the title
    yaxis=dict(autorange="reversed") 
)

# Render the chart in Streamlit
st.plotly_chart(fig, use_container_width=True)
st.markdown("")

st.markdown("#### What Do These Features Represent?")
st.markdown("""
- **`sbytes`**: Total bytes sent from the source IP. High values might indicate data exfiltration.
- **`smeansz`**: Average size of packets sent by the source. Unusual sizes may indicate attacks like DoS.
- **`sttl`**: Time-to-live value of packets from the source. Deviations may suggest reconnaissance activity.
- **`ct_dst_src_ltm`**: Number of recent connections between source and destination IPs. High counts may indicate scanning attacks.
- **`ct_state_ttl`**: Count of distinct state-TTL pairs. Irregularities often occur in packet-based attacks.
""")
st.markdown("---")
st.markdown("")

# Class Distribution Section
st.markdown("### Class Distribution in the Dataset")
st.markdown("")

data = load_data()
attack_mapping = load_attack_mapping()

class_distribution = data['attack_cat_encoded'].value_counts().reset_index()
class_distribution.columns = ['Encoded Class', 'Count']

# Map encoded classes to attack types
class_distribution['Class'] = class_distribution['Encoded Class'].map(
    dict(zip(attack_mapping['Encoded Value'], attack_mapping['Unnamed: 0']))
)

# Sort for better visualization
class_distribution = class_distribution.sort_values(by='Count', ascending=False)

# Create a bar chart
fig = px.bar(
    class_distribution,
    x='Class',
    y='Count',
    title='Class Distribution in the Dataset',
    labels={'Class': 'Attack Type', 'Count': 'Number of Instances'},
    color='Count',
    color_continuous_scale='Blues',
    text='Count',
)
fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')

st.plotly_chart(fig)

st.markdown("")
st.markdown(
    """
    In this dataset, there are total nine categories of attack and 'normal' is non-attack.

    As shown above, the distribution is highly imbalanced. The 'normal' traffic class 
    constitutes the majority of instances, while certain attack types, such as `worm` 
    and `shellcode`, have very few samples. These imbalances pose significant challenges for the model.

    To address these imbalances, we applied the following techniques:
    
    - **Class Weights**: The Random Forest model was trained using the `class_weight='balanced'` parameter, ensuring penalties for misclassifying minority classes.
    - **Downsampling**: We reduced the size of the majority class ('normal') to balance it with the attack classes.
    
    Despite the imbalance, our model achieved strong performance metrics, particularly for detecting rare attack types like `shellcode`.
    """
)
st.markdown("---")
st.markdown("")

# Confusion Matrix Section
st.markdown("### Confusion Matrix")
st.markdown("")

fig = go.Figure(data=go.Heatmap(
        z=[[57560, 665],[4, 58221]],
        x=["Normal", "Anomaly"],
        y=["Normal", "Anomaly"],
        colorscale="haline",
        showscale=True,
        text=[[57560, 665],[4, 58221]],  
        texttemplate="%{text}",  
        textfont={"size": 18},  
        hoverinfo="z"  
    ))

fig.update_layout(
        title="Confusion Matrix for Random Forest Model",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis=dict(autorange="reversed"),  
        template="plotly_white"
    )

st.plotly_chart(fig, use_container_width=True)
st.markdown("---")
st.markdown("")

# Evaluation Metrics Section
st.markdown("### Evaluation Metrics")
st.markdown("")

st.markdown("We evaluated the model using Stratified K-Fold Cross Validation, which produced the following **on average** metrics:")

evaluation_metrics = {
    "Accuracy": "92.94%",
    "Precision": "95.24%",
    "Recall": "92.94%",
    "F1-Score": "93.85%",
}

st.dataframe(pd.DataFrame(evaluation_metrics.items(), columns=["Metric", "Value"]))
