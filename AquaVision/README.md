# AquaVision 💧
## AI-Powered Water Management Companion

[![Built by](https://img.shields.io/badge/Built%20by-Wahib-blue)](https://github.com/wahibonae)
[![Built by](https://img.shields.io/badge/Built%20by-Riyad-blue)](https://github.com/riyad03)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-red)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange)](https://tensorflow.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-4.0%2B-purple)](https://openai.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.1%2B-yellow)](https://langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1.0%2B-green)](https://www.langchain.com/langgraph/)

## Abstract

### Background & Problem Statement
Morocco is currently experiencing a severe water crisis, marked by a dramatic 70% reduction in rainfall. The challenge extends beyond the scarcity itself - government agencies and decision-makers lack efficient tools and clear information for water resource management, leading to suboptimal water allocation and increased risk of shortages.

### Impact & Proposed Solution
AquaVision serves as an AI-powered companion for government agencies and decision-makers across Morocco and Africa. By combining advanced AI technologies, such as GenAI and LSTM models, with comprehensive data analysis, it transforms complex water management decisions into actionable insights, enabling more informed and efficient water resource allocation.

### Project Outcomes & Deliverables
1. **Interactive Dashboard**: Visualization of Morocco's water situation and latest water-related news
2. **AI Smart Assistant**: Water situation aware assistant for more usable knowledge
3. **Predictive Analytics**: Forecasting of critical water parameters (AFW, Water stress level, Water use efficiency)
4. **Data Transparency**: Access to data and knowledge base used by AquaVision

## Features

### 1. Insightful Dashboard
- Latest water-related news in Morocco
- Intuitive graphs displaying various water metrics
- Water situation monitoring

### 2. Smart Water Assistant
- AI-powered assistant helping leaders understand water challenges/situation quickly
- Country-specific detailed reports
- Complex water challenges simplified into actionable insights

### 3. Advanced Forecasting
- 5-year predictions for critical water parameters (Annual Freshwater Withdrawals, Water stress level, Water use efficiency)
- Summarized overview of predictions
- Explaination & Analysis for each water metric's predictions

### 4. Available Data
- Transparent access to data used by AquaVision
- CSV files with water metrics
- PDF knowledge base from international organizations (UNCCD, EDA, etc.)

## Project Structure
```
aquavision/
├── knowledge_base/     # PDF documentation files
├── knowledge_db/       # Local vector database for embeddings
├── lstm/              # ML models and data
│   ├── af.keras       # Trained LSTM model
│   ├── standard_scaler.pkl
│   ├── african_countries_data.csv # Water metrics data
│   ├── aquavisionforecast.ipynb # LSTM model training notebook
│   └── lstmodel.h5
├── utils/             # Utility functions
├── views/             # Streamlit page views
├── requirements.txt   # Dependencies
└── streamlit_app.py   # Main application file
```

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/MoroccoAI/2024-InnovAI-Hackathon/
cd aquavision
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with:
```
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key
```

5. Run the application:
```bash
streamlit run streamlit_app.py
```

## Technologies Used
- Python
- Streamlit
- TensorFlow/Keras
- LangChain
- LangGraph
- OpenAI
- ChromaDB
- Pandas
- Matplotlib
- Serper.dev

## Future Development
- Multi-language support (French and Arabic)
- Enhanced citizen engagement
- Expanded regional coverage
- Advanced water management tools

## Built by
- [Mohamed Wahib ABKARI](https://www.linkedin.com/in/abkarimohamedwahib/)
- [Riyad RACHIDI](https://www.linkedin.com/in/riyad-rachidi/)

## Our Goal
Make water management smarter, more efficient, and more sustainable through AI-powered insights and predictions.

Thank you for MoroccoAI! 💧
