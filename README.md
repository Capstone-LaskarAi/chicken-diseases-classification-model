# 🐔 Chickbot: Chicken Disease Classification Based on Fecal Images

## Brief Description

This project aims to detect and classify chicken diseases based on fecal images using a deep learning model (MobileNetV2). The system helps farmers with early, automated, and efficient disease detection in poultry.

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone -b dev-1 https://github.com/Capstone-LaskarAi/chicken-diseases-classification-model.git Chickbot-Project
cd Chickbot-Project
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```bash
  source venv/bin/activate
  ```

### 4. Install Required Libraries

```bash
pip install -r requirements.txt
```

---

## 🔑 API & Environment Configuration

### 5. Create a Pinecone Account & Configure Index

1. **Register at [Pinecone.io](https://www.pinecone.io/).**
2. After logging in, go to the **API Keys** section in the Pinecone dashboard.
3. Click **Create API Key**, give it a name, and copy the generated API Key.
4. Go to the **Indexes** menu, click **Create Index**.
   - Enter an index name (e.g., `chickbot-index`).
   - Set the dimension (e.g., 1536 for OpenAI embeddings).
   - Choose metric: cosine.
   - Select environment (e.g., `gcp-starter`).
5. Note down the **Index Name** and **Environment** for your `.env` configuration.

### 6. Create an API Key for OpenAI or Azure OpenAI

- **OpenAI:**
  1. Visit [OpenAI API Keys](https://platform.openai.com/api-keys).
  2. Click **Create new secret key** and copy the API Key.

- **Azure OpenAI:**
  1. Go to the Azure portal and create an Azure OpenAI resource.
  2. Note the **Endpoint** and create an **API Key** from the resource menu.

---

### 7. Add API Keys to `.env`

1. Rename `.env.example` to `.env`.
2. Edit the `.env` file and add:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   PINECONE_INDEX=your_pinecone_index
   OPENAI_API_KEY=your_openai_api_key
   # or if using Azure:
   AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   ```
3. Save the `.env` file.

---

## ⚙️ Running the Project Scripts

1. **Upload knowledge base to Pinecone:**
   ```bash
   python RAG-configuration.py
   ```
2. **(Optional) Recreate the model if needed:**
   ```bash
   python convert_model.py
   ```
   *(This script is specifically for handling the model's batch_shape if required.)*

3. **Run the main application:**
   ```bash
   streamlit run app.py
   ```

---

## 📂 File Explanations

### 1. `app.py`
- **Function:** Main Streamlit app. Provides the user interface for image upload, disease prediction, and LLM (RAG) interaction.

### 2. `model_utils.py`
- **Function:** Utilities for loading the model, image preprocessing, and disease prediction.

### 3. `LLM_service.py`
- **Function:** LLM and RAG pipeline integration for document-based Q&A.

### 4. `RAG-configuration.py`
- **Function:** RAG pipeline configuration, document upload to Pinecone.

### 5. `model/updated_model.h5`
- **Function:** Trained deep learning model file.

### 6. `model/LICENSE` & `LICENSE`
- **Function:** MIT License.

### 7. `documents/`
- **Function:** Folder for PDF documents used as the RAG knowledge base.

### 8. `README.md`
- **Function:** Complete project documentation.

---

## 📝 Additional Notes

- **Dataset:** Chicken fecal images from Kaggle, resized to 224x224 pixels.
- **Model:** MobileNetV2 with ImageNet transfer learning, 4 disease classes.
- **Augmentation:** Rotation, zoom, flipping, and balancing with SMOTE.
- **Evaluation:** Validation accuracy 97%, test accuracy 95.9%.
- **RAG:** Document-based Q&A feature using Pinecone and LLM.
