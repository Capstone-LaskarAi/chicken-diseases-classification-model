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
    prompt = f"""Sebagai dokter hewan yang berpengalaman, berikan rekomendasi yang komprehensif dan mudah dipahami dalam bahasa Indonesia dengan format berikut:

    **🔍 PENJELASAN KONDISI**
    
    Jelaskan dengan bahasa yang sederhana apa itu {disease} dan mengapa kondisi ini terjadi pada ayam. Berikan informasi yang menenangkan namun informatif untuk peternak yang mungkin khawatir dengan kondisi ternaknya.

    Context from veterinary knowledge base:
    {context}
    
    **🚨 GEJALA-GEJALA YANG PERLU DIPERHATIKAN**
    
    • **Pada kotoran:** [Deskripsikan perubahan warna, tekstur, dan konsistensi]
    • **Perilaku ayam:** [Gejala behavioral yang mudah diamati]
    • **Kondisi fisik:** [Tanda-tanda fisik pada ayam]
    • **Nafsu makan & minum:** [Perubahan pola makan dan minum]

    **⚡ TINDAKAN SEGERA UNTUK PETERNAK**
    
    • **Langkah darurat (24 jam pertama):** [Tindakan prioritas tinggi]
    • **Isolasi dan pengamatan:** [Cara mengisolasi ayam yang sakit]
    • **Manajemen pakan dan air:** [Penyesuaian pemberian makan]
    • **Kapan harus memanggil dokter hewan:** [Indikator untuk konsultasi profesional]

    **🛡️ LANGKAH PENCEGAHAN JANGKA PANJANG**
    
    • **Sanitasi kandang:** [Tips kebersihan kandang yang praktis]
    • **Manajemen pakan:** [Kualitas dan cara pemberian pakan]
    • **Program vaksinasi:** [Jadwal vaksinasi yang direkomendasikan]
    • **Monitoring kesehatan rutin:** [Checklist harian untuk peternak]

    **💡 TIPS PRAKTIS DARI DOKTER HEWAN**
    
    Berikan 2-3 tips khusus yang bisa langsung diterapkan peternak, termasuk bahan-bahan alami atau metode sederhana yang bisa membantu pemulihan atau pencegahan.

    **⚠️ PERHATIAN KHUSUS**
    
    Sampaikan hal-hal penting yang perlu diwaspadai dan kapan kondisi ini bisa menjadi serius. Berikan motivasi dan dukungan kepada peternak.

    Gunakan bahasa yang ramah, empati, mudah dipahami, dan praktis untuk peternak Indonesia. Berikan penjelasan dengan gaya konsultasi dokter hewan yang profesional namun hangat.
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
async def generate_recommendation_azure_async(disease, context="", deployment_name=None):
    """Generate veterinary recommendations using Azure OpenAI API"""
    prompt = f"""Sebagai dokter hewan yang berpengalaman, berikan rekomendasi yang komprehensif dan mudah dipahami dalam bahasa Indonesia dengan format berikut:

    🔍 PENJELASAN KONDISI
    
    Jelaskan dengan bahasa yang sederhana apa itu {disease} dan mengapa kondisi ini terjadi pada ayam. Berikan informasi yang menenangkan namun informatif untuk peternak yang mungkin khawatir dengan kondisi ternaknya.

    Context from veterinary knowledge base:
    {context}

    🚨 GEJALA-GEJALA YANG PERLU DIPERHATIKAN
    
    • Pada kotoran: [Deskripsikan perubahan warna, tekstur, dan konsistensi]
    • Perilaku ayam: [Gejala behavioral yang mudah diamati]
    • Kondisi fisik: [Tanda-tanda fisik pada ayam]
    • Nafsu makan & minum: [Perubahan pola makan dan minum]

    ⚡ TINDAKAN SEGERA UNTUK PETERNAK
    
    • Langkah darurat (24 jam pertama): [Tindakan prioritas tinggi]
    • Isolasi dan pengamatan: [Cara mengisolasi ayam yang sakit]
    • Manajemen pakan dan air: [Penyesuaian pemberian makan]
    • Kapan harus memanggil dokter hewan: [Indikator untuk konsultasi profesional]

    🛡️ LANGKAH PENCEGAHAN JANGKA PANJANG
    
    • Sanitasi kandang: [Tips kebersihan kandang yang praktis]
    • Manajemen pakan: [Kualitas dan cara pemberian pakan]
    • Program vaksinasi: [Jadwal vaksinasi yang direkomendasikan]
    • Monitoring kesehatan rutin: [Checklist harian untuk peternak]

    💡 TIPS PRAKTIS DARI DOKTER HEWAN
    
    Berikan 2-3 tips khusus yang bisa langsung diterapkan peternak, termasuk bahan-bahan alami atau metode sederhana yang bisa membantu pemulihan atau pencegahan.

    ⚠️ PERHATIAN KHUSUS

    - Kalau nggak nemu informasi yang relevan, jawab pakai kalimat ini:
      "Maaf ya, aku belum nemu info yang cocok sama pertanyaan kamu dari data yang ada."

    - Kamu hanya boleh jawab pertanyaan seputar project yang di kerjain sama Tim Eztrip, diluar itu untuk pertanyaan yang kurang relevan jawab pakai kalimat ini:
       "Maaf ya, pertanyaannya diluar konteks nih! coba tanyakan hal lain yang relevan terkait penanganan dan pencegahan penyakit ayam"
    
    - Sampaikan hal-hal penting yang perlu diwaspadai dan kapan kondisi ini bisa menjadi serius. Berikan motivasi dan dukungan kepada peternak.

    - Gunakan bahasa yang ramah, empati, mudah dipahami, dan praktis untuk peternak Indonesia. Berikan penjelasan dengan gaya friendly gen z.
    - Buat menggunakan format paragraf yang mudah dibaca, dengan poin-poin penting yang jelas dan terstruktur.
    """
    
    try:
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key = os.environ.get("AZURE_OPENAI_KEY")
        # Menggunakan deployment_name dari argumen fungsi, bukan dari env var secara langsung di sini
        # deployment_name_to_use = deployment_name if deployment_name else os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        if not deployment_name:
            # Fallback atau error jika tidak ada deployment_name yang disediakan, sesuai kebutuhan
            # Untuk saat ini, kita asumsikan deployment_name akan selalu ada jika Azure dipilih
            # Atau bisa menggunakan default dari environment variable jika tidak ada yang dipilih
            default_deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
            if not default_deployment_name:
                raise ValueError("Azure OpenAI deployment name not provided and no default in environment.")
            deployment_name_to_use = default_deployment_name
            st.warning(f"Azure deployment name not explicitly selected, using default: {deployment_name_to_use}")
        else:
            deployment_name_to_use = deployment_name

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
                f"{azure_endpoint}/openai/deployments/{deployment_name_to_use}/chat/completions?api-version=2025-01-01-preview",
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
async def rag_pipeline(disease, llm_choice, azure_deployment_name=None):
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
            recommendation = await generate_recommendation_azure_async(disease, context, deployment_name=azure_deployment_name)
        else:  # Ollama
            recommendation = await generate_recommendation_ollama_async(disease, context)
            
        return recommendation
    
    except Exception as e:
        st.error(f"Error in RAG pipeline: {str(e)}")
        # Fallback to generate a response without RAG
        if llm_choice == "Azure OpenAI":
            return await generate_recommendation_azure_async(disease, "", deployment_name=azure_deployment_name)
        else:
            return await generate_recommendation_ollama_async(disease, "")