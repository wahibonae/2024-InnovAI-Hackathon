import streamlit as st
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import pandas as pd
from utils.genai_utils import generate_response
from utils.lstm_utils import AIWaterForcasting, draw

load_dotenv()

# Initialize session state to persist input
if 'input_text' not in st.session_state:
    st.session_state.input_text = ''

st.title("🤖 Smart Assistant")
st.write("##### Get insights on your region's water situation with our analysis reports and AI predictions for informed water management.")

st.write("")

# Quick access buttons for common questions
predefined_questions = [
    "What is the water stress level next year?",
    "Describe the water situation",
    "How can we limit next year's water usage?",
    "Create a water management plan"
]

st.write("#### Question Suggestions")
cols = st.columns(4)
for i, question in enumerate(predefined_questions):
    with cols[i]:
        if st.button(question, key=f"q_{i}", use_container_width=True):
            st.session_state.input_text = question
            st.session_state.form_submitted = True

# Main input form
with st.form("smart_assistant_form", clear_on_submit=False):
    col1, col2 = st.columns(2)

    # Question input
    with col1:
        input_text = st.text_input(
            "Enter your question here", 
            value=st.session_state.input_text
        )
    
    # Country selector
    with col2:
        df = pd.read_csv("lstm/african_countries_data.csv")
        countries = df["Country"].unique()
        country = st.selectbox(
            "Country",
            ("", *countries),
            format_func=lambda x: "Select a country" if x == "" else x,
        )

    submitted = st.form_submit_button("Submit")

    # Handle form submission
    if submitted or st.session_state.get('form_submitted', False):
        st.session_state.form_submitted = False  # Reset submission flag
        
        with st.spinner("Generating response..."):
            if country != "":
                output, show_predictions = generate_response(input_text, country)   
                
                # Show response with predictions if needed
                if show_predictions:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(output)
                    
                    # Display prediction charts
                    with col2:
                        st.subheader("Predictions")
                        Predictor = AIWaterForcasting(
                            country,
                            'lstm/african_countries_data.csv',
                            'lstm/af.keras',
                            'lstm/standard_scaler.pkl'
                        )
                        
                        # Create tabs for different metrics
                        tab_labels = [
                            "AFW agriculture", 
                            "AFW domestic", 
                            "AFW industry", 
                            "Water stress", 
                            "Water Use Efficiency"
                        ]
                        tabs = st.tabs(tab_labels)
                        
                        # Generate charts for each metric
                        for i, tab in enumerate(tabs):
                            with tab:
                                chart = draw(Predictor, i)
                                st.pyplot(chart)
                                st.write("Note: Dotted lines represent predictions")
                else:
                    st.write(output)


