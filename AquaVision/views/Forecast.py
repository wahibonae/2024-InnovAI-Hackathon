import streamlit as st
import pandas as pd
import numpy as np
from utils.lstm_utils import AIWaterForcasting
from utils.genai_utils import analyze_forecast, generate_overall_summary
import plotly.graph_objects as go
import altair as alt
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

st.title("📊 Forecast")
st.write("##### Explore future predictions for water usage and efficiency metrics across African countries.")

st.write("")
st.write("")

# Load data and prep unique countries
df = pd.read_csv("lstm/african_countries_data.csv")
countries = sorted(df["Country"].unique())

# Country selection
selected_country = st.selectbox(
    "Select a country to forecast water parameters",
    [""] + list(countries),
    format_func=lambda x: "Select a country" if x == "" else x,
)

if selected_country:
    # Set up the LSTM predictor and get historical + future data
    Predictor = AIWaterForcasting(selected_country,'lstm/african_countries_data.csv','lstm/af.keras','lstm/standard_scaler.pkl')
    historical_data = Predictor.getPastData()
    predictions = np.array(Predictor.Predict(5))
    
    # Create year ranges for plotting
    historical_years = list(range(2000, 2000 + len(historical_data)))
    future_years = list(range(historical_years[-1] + 1, historical_years[-1] + 6))
    
    # Water metrics we're tracking
    parameters = [
        "Agriculture Fresh Water",
        "Domestic Fresh Water",
        "Industrial Fresh Water",
        "Water Stress",
        "Water Use Efficiency"
    ]

    # Cache analysis results to avoid recomputing
    @st.cache_data(show_spinner=f"Analyzing parameter values...")
    def get_parameter_analysis(country, hist_data, pred_data, param_idx, param_name):
        return analyze_forecast(country, hist_data, pred_data, param_idx, param_name)

    # Cache the overall summary
    @st.cache_data(show_spinner=f"Analyzing {selected_country}'s water situation...")
    def get_overall_summary(country, hist_data, pred_data, params):
        return generate_overall_summary(country, hist_data, pred_data, params)

    # Display overall country summary
    st.write("")
    st.write("")
    st.write("### :blue-background[Overall Summary]")
    st.success(get_overall_summary(selected_country, historical_data, predictions, parameters))
    
    # Dialog popup for detailed parameter analysis
    @st.dialog("Parameter Analysis")
    def show_parameter_analysis(param_name, param_index):
        analysis = get_parameter_analysis(selected_country, historical_data, predictions, param_index, param_name)
        
        st.subheader(f"{param_name} Analysis")
        st.write("##### Summary")
        st.write(analysis.summary)
        
        st.write("##### Explanation")
        st.write(analysis.explanation)
        
        st.write("##### Recommended Solutions")
        for solution in analysis.solutions:
            st.write(f"• {solution}")

    # Create a 2-column layout for parameter visualizations
    for i in range(0, len(parameters), 2):
        col1, col2 = st.columns(2)
        
        for j, column in enumerate([col1, col2]):
            if i + j < len(parameters):
                param = parameters[i + j]
                with column:
                    st.write("")
                    st.write("")
                    
                    # Parameter header with analysis button
                    header_col1, header_col2 = st.columns([4, 1])
                    with header_col1:
                        st.write(f"### :blue-background[{param}]")
                    with header_col2:
                        if st.button("View Analysis", key=f"analysis_{i+j}"):
                            show_parameter_analysis(param, i+j)
                    
                    # Create trend visualization
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot both historical and predicted trends
                    ax.plot(historical_years + future_years, 
                            list(historical_data[:, i + j].astype(float)) + list(predictions[:, i + j].astype(float)), 
                            marker='o', 
                            linewidth=2, 
                            color='#FF0000',
                            linestyle='--',
                            label='Predicted')
                    
                    ax.plot(historical_years, 
                            historical_data[:, i + j].astype(float), 
                            marker='o', 
                            linewidth=2, 
                            color='#0000FF',
                            label='Historical')

                    # Add visual aids for better readability
                    ax.axvline(x=historical_years[-1], color='red', linestyle=':', alpha=0.7)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.set_xlabel('Year')
                    ax.set_ylabel(param)
                    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Show comparison metrics between historical and predicted values
                    if len(predictions) > 0:
                        last_historical = historical_data[-1, i + j].astype(float)
                        last_predicted = predictions[-1, i + j].astype(float)
                        percent_change = ((last_predicted - last_historical) / last_historical) * 100
                        
                        # Center the metrics using columns
                        _, metric_col1, metric_col2, _ = st.columns([1, 2, 2, 1])
                        with metric_col1:
                            st.metric(
                                f"Last Historical Value ({historical_years[-1]})",
                                f"{last_historical:.2f}"
                            )
                        with metric_col2:
                            st.metric(
                                f"Predicted Value ({future_years[-1]})",
                                f"{last_predicted:.2f}",
                                f"{percent_change:+.2f}%"
                            )
                    st.write("")
