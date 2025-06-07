import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_openai import AzureOpenAIEmbeddings
from PyPDF2 import PdfReader
from pathlib import Path
from pinecone import Pinecone as PineconeClient

# load environment variables
load_dotenv()

# Initialize Pinecone with new SDK format
pc = PineconeClient(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# initiate embeddings model
embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-3-large",
    openai_api_version="2023-05-15",
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_KEY"),
    chunk_size=1000,
)

# Function to check if PDF is encrypted
def is_encrypted_pdf(path):
    try:
        reader = PdfReader(path)
        return reader.is_encrypted
    except:
        return True  # treat error as encrypted

# Get all PDF file paths
doc_dir = Path("documents")
pdf_paths = list(doc_dir.glob("*.pdf"))

# Filter out encrypted PDFs
valid_pdf_paths = [p for p in pdf_paths if not is_encrypted_pdf(p)]

print(f"📄 Total PDF ditemukan: {len(pdf_paths)}")
print(f"✅ PDF yang akan diproses: {len(valid_pdf_paths)}")
print(f"🚫 PDF terenkripsi dilewati: {len(pdf_paths) - len(valid_pdf_paths)}")

# Load all valid documents individually
documents = []
for path in valid_pdf_paths:
    loader = PyPDFLoader(str(path))
    try:
        docs = loader.load()
        documents.extend(docs)
    except Exception as e:
        print(f"⚠️ Gagal memuat {path.name}: {e}")

# split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# upload to Pinecone vector store with updated SDK syntax
vector_store = Pinecone.from_documents(
    docs,
    embeddings,
    index_name=index_name,
    namespace="chickbot_docs"
)

print(f"✅ Berhasil mengunggah {len(docs)} chunks ke Pinecone index '{index_name}'")
