import streamlit as st

# Page config
st.set_page_config(
    page_title="AquaVision",
    page_icon="💧", 
    layout="wide",
    initial_sidebar_state="expanded",
)

dashboard = st.Page("views/Dashboard.py", title="💧 Dashboard")
smart_assistant = st.Page("views/SmartAssistant.py", title="🤖 Smart Assistant")
forecasting = st.Page("views/Forecast.py", title="📊 Forecast")
available_data = st.Page("views/AvailableData.py", title="📈 Available Data")


pg = st.navigation(pages=[dashboard, smart_assistant, forecasting, available_data])

st.logo("assets/aquavision-logo.png")

st.sidebar.selectbox("Language", ["English", "French (coming soon)", "Arabic (coming soon)"], disabled=True)

st.sidebar.write("###### Made by the AquaVision team: [riyad](https://www.linkedin.com/in/riyad-rachidi/) & [wahib](https://www.linkedin.com/in/abkarimohamedwahib/)")

pg.run()
