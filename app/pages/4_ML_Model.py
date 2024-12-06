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

st.markdown("# 🌲 Random Forest for Anomaly Detection 🌲")

st.markdown("### Why Random Forest algorithm?")
st.markdown("""
We chose to implement Random Forest due to its advantages:
- **Feature Importance**: It provides the importances of the individual features contributing to the predictions.
- **Handling of Class Imbalances**: The `class_weight='balanced'` hyperparameter adjusts for imbalances in the dataset.
- **Flexibility**: It performs well on datasets with mixed types (categorical and numerical).
""")

st.markdown("### Model Hyperparameters")
st.markdown("""
To determine the optimal hyperparameters for the model we tried RandomizedSearchCV, GridSearchCV:
- **Max Depth**: 22
- **Min Samples Split**: 6
- **Number of Estimators**: 300
- **Class Weight**: 'balanced'
""")

st.markdown("### Evaluation Metrics")
st.markdown("We evaluated the model using Stratified K-Fold Cross Validation, which produced the resulting **on average** metrics:")

evaluation_metrics = {
    "Accuracy": "92.94%",
    "Precision": "95.24%",
    "Recall": "92.94%",
    "F1-Score": "93.85%",
}

st.dataframe(pd.DataFrame(evaluation_metrics.items(), columns=["Metric", "Value"]))


st.markdown("### Feature Importances")
st.markdown("Random forest ranks features by how much they contribute to model predictions. Below is plot of the most important features: ")

feature_importances = pd.read_csv("././datasets/feature_importances.csv")

fig = px.bar(
    feature_importances,
    x='Importance',
    y='Feature',
    orientation='h',  
    title='Top 10 Feature Importances',
    labels={'Importance': 'Feature Importance', 'Feature': 'Feature Name'},
    color_discrete_sequence=['#FFD700']
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("### Confusion Matrix")

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
