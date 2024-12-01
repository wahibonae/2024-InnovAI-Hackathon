from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import UnstructuredURLLoader
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# Simple prompt template for news summarization
news_prompt = """
    You are a news summarizer. Write a summary of the following in 50-100 words:
"""

@st.cache_data
def news():
    """Fetches and summarizes recent news about water situation in Morocco.
    Uses Google News API and GPT for summaries."""
    
    # Get last week's news using Serper API
    search = GoogleSerperAPIWrapper(
        type="news", 
        tbs="qdr:w1", 
        serper_api_key=os.getenv("SERPER_API_KEY")
    )
    result_dict = search.results("situation de l'eau au Maroc")
    news_items = []
    
    if result_dict['news']:
        # Process top 5 news articles
        for i, item in zip(range(5), result_dict['news']):
            # Load article content with browser headers to avoid blocks
            loader = UnstructuredURLLoader(
                urls=[item['link']], 
                ssl_verify=False, 
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                }
            )
            data = loader.load()
            
            # Setup GPT for summarization
            llm = ChatOpenAI(
                temperature=0.7, 
                model="gpt-3.5-turbo-0125", 
                streaming=True
            )
            
            # Create and run the summarization chain
            prompt_template = news_prompt + """
                {text}
            """
            prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
            chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
            summary = chain.run(data)
            
            # Store results
            news_items.append({
                'title': item['title'],
                'link': item['link'],
                'summary': summary
            })
            
    return news_items
