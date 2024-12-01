import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import plotly.graph_objects as go
from utils.news_utils import news

st.title("💧 Dashboard")
st.write("##### Explore water usage trends and efficiency metrics across Morocco.")

st.write("")
st.write("")

# News feed section with expandable content
st.write(f"### :blue-background[📰 Water News Feed]")
with st.expander("🔎 Click to stay updated with the latest water news in Morocco ->"):
    # Manual refresh option for news data
    if st.button("Refresh News"):
        st.cache_data.clear()

    news_items = news()
    if news_items:
        for news in news_items:
            st.success(f"Title: {news['title']}\n\nLink: {news['link']}\n\nSummary: {news['summary']}")
    else:
        st.error(f"No search results for: situation de l'eau au Maroc.")


st.write("")
st.write("")

# Load Morocco's historical data
df = pd.read_csv("lstm/african_countries_data.csv")
df = df[df["Country"] == "Morocco"]

# Split dashboard into two columns for water usage visualizations
col1, col2 = st.columns(2)

# Left side - Bar chart showing water use by sector over time
with col1:
    st.subheader(":blue-background[Annual Fresh Water Use]")
    
    # Sample every 5 years to avoid overcrowding
    df_5years = df[df['Year'] % 5 == 0]
    df_melted = pd.melt(
        df_5years,
        id_vars=['Year'],
        value_vars=['AFW agriculture', 'AFW domestic', 'AFW industry'],
        var_name='Category',
        value_name='Value'
    )
    
    with st.expander("What is this?"):
        st.write(f'''
            Annual Fresh Water (AFW) represents the volume of water used by each sector, shown every 5 years from 
            {df_5years['Year'].min()} to {df_5years['Year'].max()}.
        ''')
    
    # Create grouped bar chart using Altair
    chart = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X('Year:O'),
        y='Value:Q',
        color='Category:N',
        xOffset='Category:N'
    ).properties(
        height=400
    ).configure_legend(
        orient='bottom',
        titleOrient='left',
        padding=10
    )
    
    st.altair_chart(chart, use_container_width=True)

# Right side - Donut chart showing current distribution
with col2:
    st.subheader(":blue-background[Total Annual Fresh Water Use]")
    latest_year = df['Year'].max()
    latest_data = df[df['Year'] == latest_year].iloc[0]
    
    with st.expander("What is this?"):
        st.write(f'''
            Distribution of Annual Fresh Water usage across sectors for {latest_year}.
        ''')
    
    # Prep data for the donut chart
    values = [
        latest_data['AFW agriculture'],
        latest_data['AFW domestic'],
        latest_data['AFW industry']
    ]
    labels = ['Agriculture', 'Domestic', 'Industry']
    
    # Create donut chart with centered legend
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.7)])
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=400,
        margin=dict(t=10)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Bottom section for water metrics
col1, col2 = st.columns(2)

# Left side - Water stress
with col1:
    st.subheader(":blue-background[Water Stress Level]")
    
    with st.expander("What is this?"):
        st.write(f'''
            Water stress shows the ratio between water withdrawal and available resources from {df['Year'].min()} to {df['Year'].max()}.
        ''')
    
    # Create line plot for stress levels
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Year'], df['Water stress'], marker='o', linewidth=2, color='#1f77b4')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Year')
    ax.set_ylabel('Stress Level')
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)

# Right side - water Use metrics
with col2:
    st.subheader(":blue-background[Water Use Efficiency]")
    
    with st.expander("What is this?"):
        st.write(f'''
            Water Use Efficiency measures how effectively water is used across all sectors from {df['Year'].min()} to {df['Year'].max()}.
        ''')
    
    # Create line plot for water use efficiency metric
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Year'], df['Water Use Efficiency'], marker='o', linewidth=2, color='#2ca02c')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Year')
    ax.set_ylabel('Efficiency')
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)