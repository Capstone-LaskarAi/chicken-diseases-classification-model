import streamlit as st
import asyncio
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import logging
import os

# Import dari modul terpisah
from model_utils import load_model, preprocess_image, predict_disease, CLASS_LABELS
from LLM_service import rag_pipeline
from langchain_core.messages import HumanMessage, AIMessage

# Fungsi untuk menyimpan gambar yang diupload
def save_uploaded_image(uploaded_file):
    if uploaded_file is not None:
        # Create uploads directory if it doesn't exist
        os.makedirs("uploads", exist_ok=True)
        
        # Generate a unique filename with timestamp
        import uuid
        file_extension = os.path.splitext(uploaded_file.name)[1]
        unique_filename = f"uploads/image_{uuid.uuid4()}{file_extension}"
        
        # Save the file
        with open(unique_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return unique_filename
    return None

# UI Setup
def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    st.set_page_config(
        page_title="Chicken Disease Classification",
        page_icon="🐔",
        layout="wide"
    )
      # Custom CSS
    st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 4px 4px 0px 0px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .healthy {
        background-color: rgba(0, 128, 0, 0.2);
        border: 1px solid green;
    }
    .disease {
        background-color: rgba(255, 165, 0, 0.2);
        border: 1px solid orange;
    }
    .severe {
        background-color: rgba(255, 0, 0, 0.2);
        border: 1px solid red;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🐔 Chickbot: Chicken Disease Classification & AI Vet")
    
    # Initialize session state variables if they don't exist
    if 'predicted_disease' not in st.session_state:
        st.session_state.predicted_disease = None
    if 'confidence' not in st.session_state:
        st.session_state.confidence = None
    if 'recommendation' not in st.session_state:
        st.session_state.recommendation = None
    if 'processed_image' not in st.session_state:
        st.session_state.processed_image = None
    if 'llm_choice' not in st.session_state:
        st.session_state.llm_choice = "Ollama (Llama 3.2)"
    if 'azure_deployment_name' not in st.session_state:
        st.session_state.azure_deployment_name = "gpt-4.1" # Default Azure model    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []  # Store as simple dict format for compatibility
    
    # Create tabs
    tab1, tab2 = st.tabs(["Upload & Diagnose", "AI Vet Chatbot"])
    
    # LLM selection in sidebar
    with st.sidebar:
        st.title("Settings")
        llm_choice_options = ["Ollama (Llama 3.2)", "Azure OpenAI"]
        llm_choice = st.radio(
            "Select LLM for recommendations:",
            llm_choice_options,
            index=llm_choice_options.index(st.session_state.llm_choice) # Set initial value from session state
        )
        st.session_state.llm_choice = llm_choice

        # Azure model selection - only show if Azure OpenAI is selected
        if st.session_state.llm_choice == "Azure OpenAI":
            azure_model_options = ["gpt-4.1", "gpt-4o", "gpt-4o-mini", "grok-3-mini"]
            azure_deployment_name = st.selectbox(
                "Select Azure OpenAI Model:",
                azure_model_options,
                index=azure_model_options.index(st.session_state.azure_deployment_name) # Set initial value
            )
            st.session_state.azure_deployment_name = azure_deployment_name
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app uses AI to identify chicken diseases from feces images and provide veterinary recommendations.
          **Models used:**
        - Image classification: MobileNetV2
        - Text embedding: text-embedding-3-large (Azure OpenAI)
        - LLM: Llama 3.2 (Ollama) or Gpt-4.1 (Azure)
        """)
    
    # Tab 1: Upload & Diagnose
    with tab1:
        st.header("Upload Chicken Feces Image")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        col1, col2 = st.columns([1, 1])
        
        if uploaded_file is not None:
            # Display uploaded image
            with col1:
                st.subheader("Uploaded Image")
                img = Image.open(uploaded_file)
                st.image(img, width=350, caption="Uploaded Image")
                
                # Save image for record-keeping
                saved_path = save_uploaded_image(uploaded_file)
                if saved_path:
                    st.session_state.saved_image_path = saved_path
            
            # Process image and make prediction when button is clicked
            if st.button("Analyze Image"):
                with st.spinner("Analyzing image..."):
                    # Load model
                    model = load_model()
                    
                    if model is not None:
                        # Preprocess image
                        img_array = preprocess_image(img)
                        st.session_state.processed_image = img_array
                        
                        # Make prediction
                        predicted_disease, confidence = predict_disease(model, img_array)
                        
                        if predicted_disease:
                            st.session_state.predicted_disease = predicted_disease
                            st.session_state.confidence = confidence
                        else:
                            st.error("Failed to make a prediction. Please try again.")
            
            # Display prediction results
            with col2:
                if st.session_state.predicted_disease:
                    st.subheader("Diagnosis Results")
                    
                    # Determine CSS class based on disease
                    css_class = "healthy" if st.session_state.predicted_disease == "Healthy" else "disease"
                    if st.session_state.predicted_disease in ["Newcastle Disease", "Coccidiosis"]:
                        css_class = "severe"
                    
                    # Display prediction with proper styling
                    st.markdown(f"""
                    <div class="prediction-box {css_class}">
                        <h3>Predicted Condition: {st.session_state.predicted_disease}</h3>
                        <p>Confidence: {st.session_state.confidence:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Plot confidence bars for all classes
                    if st.session_state.processed_image is not None:
                        model = load_model()
                        if model is not None:
                            try:
                                # Use batch_size=1 and disable verbose output
                                prediction = model.predict(st.session_state.processed_image, batch_size=1, verbose=0)[0]
                                
                                fig, ax = plt.subplots(figsize=(10, 4))
                                bars = ax.barh(CLASS_LABELS, prediction, color=['skyblue' if i != np.argmax(prediction) else 'orange' for i in range(len(prediction))])
                                ax.set_xlabel('Confidence')
                                ax.set_title('Prediction Confidence by Class')
                                
                                # Add percentage labels to the bars
                                for bar in bars:
                                    width = bar.get_width()
                                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2%}', 
                                           va='center', fontsize=8)
                                
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error generating confidence chart: {str(e)}")
                        else:
                            st.error("Could not load model to generate confidence chart.")
                    
                    # Message to direct user to the next tab
                    st.success("✅ Analysis complete! Go to the 'AI Vet Chatbot' tab for recommendations.")
    
    # Tab 2: AI Vet Chatbot
    with tab2:
        st.header("AI Veterinary Recommendations")
        
        if st.session_state.predicted_disease:
            st.info(f"Based on the image analysis, the system has detected: **{st.session_state.predicted_disease}**")
            
            if st.button("Generate Recommendations"):
                with st.spinner("Generating expert recommendations..."):
                    start_time = time.time()
                      # Generate recommendation using RAG pipeline with selected LLM
                    llm_selection = "Azure OpenAI" if st.session_state.llm_choice == "Azure OpenAI" else "Ollama"
                    azure_deployment_to_use = st.session_state.azure_deployment_name if llm_selection == "Azure OpenAI" else None
                    
                    recommendation = asyncio.run(rag_pipeline(
                        st.session_state.predicted_disease, 
                        llm_selection,
                        azure_deployment_name=azure_deployment_to_use,
                        chat_history=st.session_state.chat_history
                    ))
                    
                    st.session_state.recommendation = recommendation
                    
                    end_time = time.time()
                    generation_time = end_time - start_time
            
            # Display recommendation if available
            if st.session_state.recommendation:
                st.subheader("Veterinary Recommendations")
                st.markdown(st.session_state.recommendation)
                
                # Disclaimer
                st.markdown("---")
                st.markdown("""
                **Disclaimer**: This is an AI-generated recommendation based on image analysis. 
                Always consult with a qualified veterinarian for a proper diagnosis and treatment plan.
                """)
        else:
            st.warning("Please upload and analyze an image in the 'Upload & Diagnose' tab first.")
        
        # Chat history section
        st.subheader("Ask Follow-up Questions")
        
        # Create a container for chat messages
        chat_container = st.container()
          # Display chat history in the container
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input - this will naturally appear at the bottom
        if st.session_state.predicted_disease:  # Only show if we have a prediction
            user_question = st.chat_input("Ask a question about this condition...")
            
            if user_question:
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                  # Generate response
                with st.spinner("Thinking..."):
                    # Use the new rag_pipeline with chat history
                    llm_selection = "Azure OpenAI" if st.session_state.llm_choice == "Azure OpenAI" else "Ollama"
                    azure_deployment_to_use = st.session_state.azure_deployment_name if llm_selection == "Azure OpenAI" else None
                    
                    response = asyncio.run(rag_pipeline(
                        st.session_state.predicted_disease,
                        llm_selection,
                        azure_deployment_name=azure_deployment_to_use,
                        user_question=user_question,
                        predicted_disease=st.session_state.predicted_disease,
                        chat_history=st.session_state.chat_history
                    ))
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    
                    # Rerun to update the display
                    st.rerun()
        else:
            st.info("Upload and analyze an image first to enable the chat feature.")

if __name__ == "__main__":
    main()