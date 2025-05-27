import os
import asyncio
import aiohttp
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.embeddings import OllamaEmbeddings
from supabase.client import Client, create_client

# Load environment variables
load_dotenv()

# Supabase & Embeddings setup
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)
embeddings = OllamaEmbeddings(model="bge-m3:latest")
vector_store = SupabaseVectorStore(
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
    chunk_size=1000,
)

# Get embeddings from Ollama
async def get_embeddings_async(text, model="bge-m3"):
    """Get embeddings from Ollama API asynchronously"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/embeddings",
                json={"model": model, "prompt": text}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("embedding")
                else:
                    st.error(f"Error from Ollama embeddings API: {await response.text()}")
                    return None
    except Exception as e:
        st.error(f"Error getting embeddings: {str(e)}")
        return None

# Synchronous version for simpler usage
def get_embeddings(text, model="bge-m3"):
    """Synchronous wrapper for get_embeddings_async"""
    return asyncio.run(get_embeddings_async(text, model))

# Query Supabase for similar documents
async def query_supabase_async(vector_store, query, top_k=3):
    """Query vector store for similar documents"""
    try:
        # Use similarity_search from vector_store
        docs = vector_store.similarity_search(query, k=top_k)
        # Format results to be similar to Pinecone
        matches = []
        for doc in docs:
            matches.append({
                "metadata": {"text": doc.page_content}
            })
        return {"matches": matches}
    except Exception as e:
        st.error(f"Error querying Supabase: {str(e)}")
        return None

# Generate recommendation using Ollama API
async def generate_recommendation_ollama_async(disease, context=""):
    """Generate veterinary recommendations using Ollama API"""
    prompt = f"""You are an expert veterinarian specializing in poultry diseases. 
    Based on an image analysis of chicken feces, the system has detected: {disease}.
    
    Context from veterinary knowledge base:
    {context}
    
    Please provide:
    1. A brief explanation of this condition
    2. Common symptoms to look for in the chickens
    3. Recommended immediate actions for the farmer
    4. Prevention measures
    
    Keep your response concise and informative.
    """
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": prompt,
                    "stream": False
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("response", "No response generated")
                else:
                    error_text = await response.text()
                    st.error(f"Error from Ollama API: {error_text}")
                    return f"Failed to generate recommendation. Error: {error_text}"
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"

# Generate recommendation using Azure OpenAI API
async def generate_recommendation_azure_async(disease, context=""):
    """Generate veterinary recommendations using Azure OpenAI API"""
    prompt = f"""You are an expert veterinarian specializing in poultry diseases. 
    Based on an image analysis of chicken feces, the system has detected: {disease}.
    
    Context from veterinary knowledge base:
    {context}
    
    Please provide:
    1. A brief explanation of this condition
    2. Common symptoms to look for in the chickens
    3. Recommended immediate actions for the farmer
    4. Prevention measures
    
    Keep your response concise and informative.
    """
    
    try:
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key = os.environ.get("AZURE_OPENAI_KEY")
        deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
        }
        
        payload = {
            "messages": [{"role": "system", "content": "You are a veterinary expert assistant."}, 
                         {"role": "user", "content": prompt}],
            "max_tokens": 800
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{azure_endpoint}/openai/deployments/{deployment_name}/chat/completions?api-version=2025-01-01-preview",
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    st.error(f"Error from Azure OpenAI API: {error_text}")
                    return f"Failed to generate recommendation. Error: {error_text}"
    except Exception as e:
        return f"Error connecting to Azure OpenAI: {str(e)}"

# RAG Pipeline
async def rag_pipeline(disease, llm_choice):
    """Complete RAG pipeline for generating recommendations"""
    try:
        # 1. Create query from disease
        query = f"chicken disease {disease} symptoms treatment prevention"
        
        # 2. Search Supabase for relevant documents
        results = await query_supabase_async(vector_store, query)
        if not results or not results.get("matches"):
            context = "No relevant information found in the knowledge base."
        else:
            context = "\n\n".join([match.get("metadata", {}).get("text", "") 
                                for match in results.get("matches", [])])
        
        # 3. Generate response based on LLM choice
        if llm_choice == "Azure OpenAI":
            recommendation = await generate_recommendation_azure_async(disease, context)
        else:  # Ollama
            recommendation = await generate_recommendation_ollama_async(disease, context)
            
        return recommendation
    
    except Exception as e:
        st.error(f"Error in RAG pipeline: {str(e)}")
        # Fallback to generate a response without RAG
        if llm_choice == "Azure OpenAI":
            return await generate_recommendation_azure_async(disease, "")
        else:
            return await generate_recommendation_ollama_async(disease, "")