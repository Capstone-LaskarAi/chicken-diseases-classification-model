import os
import asyncio
import streamlit as st
from dotenv import load_dotenv

# import langchain
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.agents import create_tool_calling_agent
from langchain import hub
from langchain_community.vectorstores import Pinecone
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.tools import tool
from pinecone import Pinecone as PineconeClient

# Load environment variables
load_dotenv()

# Set Azure OpenAI environment variables
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_KEY")
os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"

# Caches to reduce latency
_EMBEDDINGS = None
_VECTORSTORE = None
_RETRIEVER = None
_LLM_AZURE = {}
_PINECONE_CLIENT = None
_PINECONE_INDEX = None

def _get_pinecone_client():
    global _PINECONE_CLIENT
    if _PINECONE_CLIENT is None:
        _PINECONE_CLIENT = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
    return _PINECONE_CLIENT

def _get_pinecone_index():
    global _PINECONE_INDEX
    if _PINECONE_INDEX is None:
        pc = _get_pinecone_client()
        _PINECONE_INDEX = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    return _PINECONE_INDEX

def _get_embeddings():
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = AzureOpenAIEmbeddings(
            model="text-embedding-3-large",
            azure_deployment="text-embedding-3-large",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version="2024-02-01"
        )
    return _EMBEDDINGS

def _get_vectorstore():
    global _VECTORSTORE
    if _VECTORSTORE is None:
        _VECTORSTORE = Pinecone.from_existing_index(
            index_name=os.getenv("PINECONE_INDEX_NAME"),
            embedding=_get_embeddings(),
            namespace="chickbot_docs"
        )
    return _VECTORSTORE

# Initialize cached instances
pc = _get_pinecone_client()
index = _get_pinecone_index()
embeddings = _get_embeddings()
vector_store = _get_vectorstore()

# initiating llm - support both Azure and Ollama
def get_llm(llm_choice="Azure OpenAI", azure_deployment_name="gpt-4o"):
    """Get LLM based on choice with caching"""
    global _LLM_AZURE
    
    if llm_choice == "Azure OpenAI":
        # Cache LLM instances per deployment
        if azure_deployment_name not in _LLM_AZURE:
            _LLM_AZURE[azure_deployment_name] = AzureChatOpenAI(
                model="gpt-4o",
                azure_deployment=azure_deployment_name,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version="2024-02-01",
                temperature=0.2,
                max_tokens=1500,
                streaming=True,
                timeout=20
            )
        return _LLM_AZURE[azure_deployment_name]
    else:
        # For Ollama, we'll use a simple fallback since langchain-community doesn't have direct Ollama support
        # We'll handle this in the pipeline
        return None

# pulling prompt from hub
prompt = hub.pull("hwchase17/openai-functions-agent")

# function to check if question is related to chicken/poultry disease management
def is_poultry_related_question(question: str) -> bool:
    """Check if the question is related to chicken/poultry disease management."""
    # Keywords related to poultry and diseases
    poultry_keywords = [
        'ayam', 'unggas', 'chicken', 'poultry', 'bebek', 'itik', 'burung', 
        'penyakit', 'disease', 'sakit', 'gejala', 'symptoms', 'pengobatan', 
        'treatment', 'obat', 'medicine', 'vaksin', 'vaccine', 'pencegahan', 
        'prevention', 'ternak', 'peternakan', 'farm', 'kandang', 'coop',
        'virus', 'bakteri', 'parasit', 'infeksi', 'infection', 'flu burung',
        'newcastle', 'tetelo', 'berak kapur', 'snot', 'crd', 'coryza',
        'kolera', 'cholera', 'cacingan', 'kutu', 'tungau', 'stress panas', 
        'tabel', 'healthy', 'detected', 'image', 'analysis', 'coccidiosis',
        'salmonella', 'feces', 'fecal', 'kotoran', 'tinja', 'diagnos',
        'rekomendasi', 'recommendation', 'saran', 'advice', 'tindakan', 'action',
        'apa', 'bagaimana', 'kenapa', 'mengapa', 'how', 'what', 'why', 'when'
    ]
    
    question_lower = question.lower()
    return any(keyword.lower() in question_lower for keyword in poultry_keywords)

# creating the retriever tool
@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# combining all tools
tools = [retrieve]

# RAG Pipeline using Agent for Azure OpenAI
def rag_pipeline_agent_azure(query, chat_history=None, predicted_disease=None, azure_deployment_name="gpt-4o"):
    """Complete RAG pipeline using LangChain agent for Azure OpenAI"""
    try:
        # Get Azure LLM
        llm = get_llm("Azure OpenAI", azure_deployment_name)
        
        # Create agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
        # Create a comprehensive query if we have disease context
        if predicted_disease:
            enhanced_query = f"Penyakit {predicted_disease}: {query}"
        else:
            enhanced_query = query
            
        # If we have predicted_disease, always process the query (don't check keywords)
        # Only check keywords if there's no predicted_disease context
        if predicted_disease or is_poultry_related_question(enhanced_query):
            # invoking the agent
            result = agent_executor.invoke({
                "input": enhanced_query, 
                "chat_history": chat_history or []
            })
            return result["output"]
        else:
            # provide default response for out-of-context questions
            return "Maaf ya, aku belum nemu info yang cocok sama pertanyaan kamu dari data yang ada."
            
    except Exception as e:
        st.error(f"Error in RAG pipeline: {str(e)}")
        return f"Terjadi kesalahan dalam memproses pertanyaan: {str(e)}"

# RAG Pipeline using Ollama (fallback to simple implementation)
async def rag_pipeline_ollama(query, chat_history=None, predicted_disease=None):
    """RAG pipeline for Ollama - simplified implementation"""
    import aiohttp
    
    try:
        # Search for relevant documents
        retrieved_docs = vector_store.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Create prompt for Ollama
        if predicted_disease:
            enhanced_query = f"Penyakit {predicted_disease}: {query}"
        else:
            enhanced_query = query
            
        prompt_text = f"""Kamu adalah dokter hewan ahli penyakit unggas yang bertugas memberikan penjelasan komprehensif tentang penyakit pada ayam.

        Panduan:
        1. Kalau datanya ketemu, jawab dengan jelas, singkat, dan langsung ke intinya, pakai bahasa yang santai dan mudah dipahami.
        2. Kalau nggak nemu informasi yang relevan, jawab pakai kalimat ini: "Maaf ya, aku belum nemu info yang cocok sama pertanyaan kamu dari data yang ada."
        3. Jangan jawab pakai pengetahuan umum atau ngarang ya — fokus cuma ke data yang ada di vector store.
        4. Tulis jawabannya dalam bentuk paragraf pendek, langsung ke poin pentingnya, tetap ramah, dan gunakan emote yang relevan.

        Context from veterinary knowledge base:
        {context}

        Pertanyaan: {enhanced_query}
        """
        
        # If we have predicted_disease, always process the query (don't check keywords)
        # Only check keywords if there's no predicted_disease context
        if not predicted_disease and not is_poultry_related_question(enhanced_query):
            return "Maaf ya, aku belum nemu info yang cocok sama pertanyaan kamu dari data yang ada."
        
        # Call Ollama API
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": prompt_text,
                    "stream": False
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("response", "No response generated")
                else:
                    error_text = await response.text()
                    return f"Failed to generate recommendation. Error: {error_text}"
                    
    except Exception as e:
        st.error(f"Error in Ollama RAG pipeline: {str(e)}")
        return f"Terjadi kesalahan dalam memproses pertanyaan: {str(e)}"

# Main RAG Pipeline function
async def rag_pipeline(query, llm_choice, azure_deployment_name=None, user_question=None, predicted_disease=None, chat_history=None):
    """Complete RAG pipeline for generating recommendations"""
    try:
        # Use the question or query as the main input
        main_query = user_question if user_question else query
        
        # Convert chat_history to LangChain message format if needed
        langchain_history = []
        if chat_history:
            for msg in chat_history:
                if msg.get("role") == "user":
                    langchain_history.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    langchain_history.append(AIMessage(content=msg.get("content", "")))
        
        if llm_choice == "Azure OpenAI":
            # Use agent-based approach for Azure OpenAI
            result = rag_pipeline_agent_azure(
                main_query,
                chat_history=langchain_history,
                predicted_disease=predicted_disease,
                azure_deployment_name=azure_deployment_name or "gpt-4o"
            )
            return result
        else:
            # Use Ollama approach
            result = await rag_pipeline_ollama(
                main_query,
                chat_history=langchain_history,
                predicted_disease=predicted_disease
            )
            return result
            
    except Exception as e:
        st.error(f"Error in RAG pipeline: {str(e)}")
        return f"Terjadi kesalahan dalam memproses pertanyaan: {str(e)}"

# Legacy compatibility functions for backward compatibility
async def generate_recommendation_azure_async(disease, context="", deployment_name=None, user_question=None):
    """Legacy compatibility function for Azure OpenAI"""
    return await rag_pipeline(
        query=disease,
        llm_choice="Azure OpenAI",
        azure_deployment_name=deployment_name,
        user_question=user_question,
        predicted_disease=disease
    )

async def generate_recommendation_ollama_async(disease, context="", user_question=None):
    """Legacy compatibility function for Ollama"""
    return await rag_pipeline(
        query=disease,
        llm_choice="Ollama",
        user_question=user_question,
        predicted_disease=disease
    )