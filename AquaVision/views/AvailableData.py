import streamlit as st
import os
import base64
import pandas as pd

st.title("📈 Available Data")
st.write("##### Explore the available data that is used by AquaVision in AI predictions and generations.")
st.write("##### Coming soon, you'll be able to upload files and update the database!")

st.write("")
st.write("")

# CSV section
st.write("### :blue-background[CSV Files]")
st.write("View the available data used by AquaVision in the CSV files below.")
tab = st.tabs(["african_countries_data.csv"])
tabs = [tab]

# Display CSV data with country filter
for i, tab in enumerate(tabs):
    # Load the dataset
    df = pd.read_csv('lstm/african_countries_data.csv')
    countries = df["Country"].unique()
    
    # Country dropdown with placeholder
    country = st.selectbox(
        "Country",
        ("", *countries),
        format_func=lambda x: "Select a country" if x == "" else x,
    )
    
    # Filter data if country is selected
    if country:
        df = df[df["Country"] == country]
    edited_df = st.data_editor(df, num_rows="dynamic")

st.write("")
st.write("")

# PDF section
st.write("### :blue-background[PDF Files]")
st.write("View and download the available PDF files below used as knowledge base for AquaVision.")
st.write("")

# Create a grid of PDF download buttons
pdf_files = os.listdir("knowledge_base")
columns = st.columns(len(pdf_files))

for i, pdf_file in enumerate(pdf_files):
    # Load and display each PDF
    with open(f"knowledge_base/{pdf_file}", "rb") as f:
        pdf = f.read()
    columns[i].markdown(f"###### {pdf_file}")
    columns[i].download_button(
        label="View",
        data=pdf,
        file_name=pdf_file,
        mime="application/pdf",
    )