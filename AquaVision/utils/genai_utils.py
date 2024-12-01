import langgraph
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from typing_extensions import List, TypedDict
from langchain_core.prompts.chat import ChatPromptTemplate
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from utils.lstm_utils import get_water_data, get_predictions
from openai import OpenAI

# Initialize LLM and embeddings models
llm = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# LangGraph Graph
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    country: str
    generation: str
    documents: List[str]
    water_data: str
    predictions: str
    prediction_made: bool


def retriever_setup():
    """Sets up or loads the vector store for document retrieval.
    Creates a new one from PDFs if it doesn't exist yet."""
    if not os.path.exists("./knowledge_db"):
        directory_path = "knowledge_base/"
        
        # Load all PDFs from the knowledge base directory
        loader = DirectoryLoader(directory_path, glob="*.pdf", loader_cls=PyPDFLoader)
        data = loader.load()

        # Split docs into smaller chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(data)

        # Create and save the vector store
        vector_store = Chroma.from_documents(splits, embeddings, persist_directory="./knowledge_db")
        vector_store.persist()
        print("Vector store created and persisted")
    else:
        # Load existing vector store
        vector_store = Chroma(embedding_function=embeddings, persist_directory="./knowledge_db")

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    return retriever




def retrieve(state: GraphState):
    print("---RETRIEVE---")
    question = state["question"]
    country = state["country"]

    retriever = retriever_setup()
    documents = retriever.invoke("In the context of {country}, {question}")
    print("Retrieved documents:", documents)

    return {"documents": documents}



def get_past_data(state: GraphState):
    print("---GET PAST DATA---")
    country = state["country"]

    water_data = get_water_data(country)
    print("Water data:", water_data)

    return {"water_data": water_data}


def predict(state: GraphState):
    print("---PREDICT---")
    country = state["country"]
    
    predictions = get_predictions(country)
    print("Predictions:", predictions)

    return {"predictions": predictions}



def generate(state: GraphState):
    """Main response generation function that combines context, historical data, and predictions."""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    water_data = state["water_data"]
    predictions = state.get("predictions", "")

    prediction_made = predictions is not None and len(str(predictions)) > 0

    # Prompt engineering for the water management expert persona
    system = """You are AquaVision, an advanced water management expert. Your task is to generate a detailed, comprehensive answer/report in response to the user's query using markdown format. You should consider the following elements:
    
    1. Historical Data: Includes analysis of the past decade's water parameter data of the specified country.
    2. Current Context: Factors in the given "context" which contains country-specific water information, or/and management solutions, plans, and policies recommended by international water organizations.
    3. Future Projections: Integrates predictions of water parameters for the coming years, if provided.
    
    Instructions:
    * Always base your answers on your own knowledge and the provided data, ensuring coherence between historical trends, current conditions, and future projections.
    * If you lack sufficient information to provide a comprehensive answer, clearly state that you don't know.
    * The users are decision makers, give them tailored, specific, and actionable solutions, emphasizing innovative and strategic approaches to addressing the water situation in the country that will help them get a better understanding to the question and the water situation in their country.
    * The markdown format's titles should not have H1 or H2 titles."""

    # Build and execute the generation chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "****** Context: {context} \n\n {water_data} \n\n" + 
         ("{predictions}" if predictions else "") + "****** Question: {question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({
        "context": documents, 
        "water_data": water_data, 
        "predictions": predictions, 
        "question": question
    })
        
    return {"generation": generation, "prediction_made": prediction_made}



# Data model
class PredictionDecision(BaseModel):
    """Binary score for predicting the water situation in the country."""

    binary_score: str = Field(
        description="Question needs to consider the prediction of water situation in the country in the next years, 'yes' or 'no'"
    )

def decide_to_predict(state):
    """Determines if the question requires future predictions based on its content."""
    question = state["question"]
    country = state["country"]

    client = OpenAI()

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a grader assessing whether a question requires water situation prediction analysis. \n If the question asks about or implies needing information about future water conditions, availability, scarcity, or related metrics in the country, then it should answer with prediction context. \n Give a binary score 'yes' or 'no' score to indicate whether the question needs water situation prediction context."},
            {"role": "user", "content": f"User question: In the context of {country}, {question}"},
        ],
        response_format=PredictionDecision
    )
    
    if completion.choices[0].message.parsed.binary_score.lower() == "yes":
        return "predict"
    else:
        return "generate"



workflow = StateGraph(GraphState)

# Define then nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("get_past_data", get_past_data)
workflow.add_node("predict", predict)
workflow.add_node("generate", generate)

# Build the graph
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "get_past_data")
workflow.add_conditional_edges(
    "get_past_data",
    decide_to_predict,
    {
        "predict": "predict",
        "generate": "generate"
    },
)
workflow.add_edge("predict", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()


def generate_response(question, country):
    output = app.invoke({"question": question, "country": country})
    return output["generation"], output["prediction_made"]

# Functions used by the Forecast page
class WaterParameterAnalysis(BaseModel):
    """Analysis for a single water parameter."""
    summary: str = Field(description="A concise summary of the parameter's situation")
    explanation: str = Field(description="Brief explanation of the trends and implications")
    solutions: list[str] = Field(description="List of potential solutions or recommendations")

def analyze_forecast(country, historical_data, predictions, parameter_index, parameter_name):
    structured_llm = llm.with_structured_output(WaterParameterAnalysis)
    
    # Format the data for the prompt
    historical_values = historical_data[:, parameter_index].astype(float).tolist()
    predicted_values = predictions[:, parameter_index].astype(float).tolist()
    
    system = """You are a water management expert AI. Analyze the historical data and predictions for a specific water parameter.
    Provide a very concise analysis that includes:
    1. A brief summary of the situation (1-2 sentences)
    2. A clear explanation of the trends and their implications (2-3 sentences)
    3. 2-3 specific, actionable solutions or recommendations
    
    Keep all responses concise and focused."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", """Analyze the water situation for {country}'s {parameter}:
        Historical values (2000 onwards): {historical}
        Predicted values (next 5 years): {predictions}""")
    ])

    chain = prompt | structured_llm

    analysis = chain.invoke({
        "country": country,
        "parameter": parameter_name,
        "historical": historical_values,
        "predictions": predicted_values
    })
    
    return analysis
    
def generate_overall_summary(country, historical_data, predictions, parameters):
    """
    Generate a comprehensive summary of the water situation and next steps.
    Returns a markdown formatted string with the overall analysis.
    """
    # Format the data for all parameters
    data_summary = []
    for i, param in enumerate(parameters):
        historical_values = historical_data[:, i].astype(float).tolist()
        predicted_values = predictions[:, i].astype(float).tolist()
        data_summary.append(f"{param}:\n- Historical: {historical_values}\n- Predicted: {predicted_values}")
    
    system = """You are a water management expert AI. Create a very comprehensive summary of (in 100-150 words) of the country's overall water situation.
    Focus on the interconnections between different water parameters and their combined impact.
    
    Your response should include:
    1. Overall Situation (the general water outlook)
    2. Key Findings (highlighting the most critical trends and relationships between parameters)
    3. Strategic Recommendations (high-impact, actionable steps)
    
    Keep the response very concise, focused, and actionable for decision-makers."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", """Analyze the overall water situation for {country}:
        
        Parameters and their values:
        {data_summary}
        
        Provide a strategic overview that considers all parameters together.""")
    ])

    chain = prompt | llm | StrOutputParser()

    summary = chain.invoke({
        "country": country,
        "data_summary": "\n".join(data_summary)
    })
    
    return summary
    