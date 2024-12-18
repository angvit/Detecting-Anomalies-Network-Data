import streamlit as st

st.set_page_config(
    page_title="Meet the Team",
    page_icon="👥",
)

st.write("# Meet the Team 👋")
st.markdown("\n")

def display_teammate(image_url, name, description, github_url, linkedin_url):  # Added linkedin_url parameter
    col1, col2 = st.columns([1, 3])  
    with col1:
        st.image(image_url, width=200) 
    with col2:
        st.markdown(f"### {name}")
        st.markdown(description)
        st.markdown(f"[GitHub]({github_url}) | [LinkedIn]({linkedin_url})", unsafe_allow_html=True)  

teammates = [
    {
        "name": "Brianna Persaud",
        "description": "Data scientist specializing in network security and anomaly detection.",
        "image_url": "app/images/teammates/brianna.jpg",
        "github_url": "https://github.com/brina0781",
        "linkedin_url": "https://www.linkedin.com/in/brianna-persaud-014872285/",
    },
    {
        "name": "Isabel Loci",
        "description": "DevOps enthusiast with expertise in cloud-native technologies.",
        "image_url": "app/images/teammates/isabel.jpg",
        "github_url": "https://github.com/belbop",
        "linkedin_url": "https://www.linkedin.com/in/isabel-loci/",
    },
    {
        "name": "Angelo Vitalino",
        "description": "Software engineer passionate about machine learning and visualization.",
        "image_url": "app/images/teammates/angelo.jpg",
        "github_url": "https://github.com/angvit",
        "linkedin_url": "https://www.linkedin.com/in/angelo-vitalino/",
    },
]

for teammate in teammates:
    display_teammate(
        image_url=teammate["image_url"],
        name=teammate["name"],
        description=teammate["description"],
        github_url=teammate["github_url"],
        linkedin_url=teammate["linkedin_url"] 
    )
    st.markdown("---")
