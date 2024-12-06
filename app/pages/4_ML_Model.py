import streamlit as st

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


