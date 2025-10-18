import streamlit as st
import asyncio
import time
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import logging
import os
import re
from pathlib import Path

# Import dari modul terpisah
from model_utils import load_model, preprocess_image, predict_disease, CLASS_LABELS
from LLM_service import rag_pipeline
from langchain_core.messages import HumanMessage, AIMessage
from dummy_test_data import list_samples

# Helper: per-word typing animation for assistant responses
def render_typing_text(text: str, placeholder, delay: float = 0.03):
    """Display text with per-word typing animation while preserving whitespace"""
    tokens = re.split(r"(\s+)", text)
    buffer = ""
    for token in tokens:
        buffer += token
        placeholder.markdown(buffer + " ▌")
        time.sleep(delay)
    placeholder.markdown(buffer)

# Helper: Show loading animation
def show_loading_animation():
    """Display animated loading indicator"""
    spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    placeholder = st.empty()
    for frame in spinner_frames:
        placeholder.markdown(f"**{frame} Processing your request...**")
        time.sleep(0.1)
    placeholder.empty()
    return placeholder

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
    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .loading-dots {
        display: inline-block;
        animation: pulse 1.5s infinite;
    }
    /* Fixed chat input at bottom */
    .fixed-chat-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #0e1117;
        padding: 1rem;
        border-top: 1px solid #30363d;
        z-index: 100;
    }
    .chat-history-container {
        padding-bottom: 120px;
        overflow-y: auto;
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
        st.session_state.llm_choice = "Azure OpenAI"
    if 'azure_deployment_name' not in st.session_state:
        st.session_state.azure_deployment_name = "gpt-4o" # Default Azure model
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []  # Store as simple dict format for compatibility
    if 'demo_img_path' not in st.session_state:
        st.session_state.demo_img_path = None
    if 'demo_img_label' not in st.session_state:
        st.session_state.demo_img_label = None
    if 'saved_image_path' not in st.session_state:
        st.session_state.saved_image_path = None
    if 'show_typing_animation' not in st.session_state:
        st.session_state.show_typing_animation = False
    if 'chat_pending_llm' not in st.session_state:
        st.session_state.chat_pending_llm = False
    if 'pending_question' not in st.session_state:
        st.session_state.pending_question = None
    if 'switch_to_tab2' not in st.session_state:
        st.session_state.switch_to_tab2 = False

    # Create tabs
    tab1, tab2 = st.tabs(["Upload & Diagnose", "AI Vet Chatbot"])

    if st.session_state.switch_to_tab2:
        st.session_state.switch_to_tab2 = False
        st.markdown(
            """
            <script>
            const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
            if (tabs.length > 1) {
                tabs[1].click();
            }
            </script>
            """,
            unsafe_allow_html=True,
        )
    
    # LLM selection in sidebar
    with st.sidebar:
        st.title("Settings")
        llm_choice_options = ["Azure OpenAI", "Ollama (Llama 3.2)"]
        llm_choice = st.radio(
            "Select LLM for recommendations:",
            llm_choice_options,
            index=llm_choice_options.index(st.session_state.llm_choice) # Set initial value from session state
        )
        st.session_state.llm_choice = llm_choice

        # Azure model selection - only show if Azure OpenAI is selected
        if st.session_state.llm_choice == "Azure OpenAI":
            azure_model_options = ["gpt-4.1", "gpt-4o", "DeepSeek-R1-0528"]
            default_idx = azure_model_options.index(st.session_state.azure_deployment_name) if st.session_state.azure_deployment_name in azure_model_options else 1
            azure_deployment_name = st.selectbox(
                "Select Azure OpenAI Model:",
                azure_model_options,
                index=default_idx
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
        
        st.markdown("---")
        st.markdown("### Sample Images (QA/Testing)")
        samples = list_samples()
        if samples:
            classes = sorted(samples.keys())
            sel_cls = st.selectbox("Class", classes, index=0)
            files = samples.get(sel_cls, [])
            if files:
                sel_file = st.selectbox("File", files, index=0, format_func=lambda p: Path(p).name)
                if st.button("Load sample image"):
                    st.session_state.demo_img_path = sel_file
                    st.session_state.demo_img_label = sel_cls
                    # reset previous outputs and chat history
                    st.session_state.predicted_disease = None
                    st.session_state.confidence = None
                    st.session_state.recommendation = None
                    st.session_state.chat_history = []
                    st.session_state.processed_image = None
                    st.session_state.show_typing_animation = False
                    st.rerun()
        else:
            st.info("Sample folder not found or empty.")
    with tab1:
        st.header("Upload Chicken Feces Image")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        col1, col2 = st.columns([1, 1])
        
        # Source image: uploaded file takes priority, otherwise demo
        img = None
        src_caption = None
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            src_caption = "Uploaded Image"
            saved_path = save_uploaded_image(uploaded_file)
            if saved_path:
                st.session_state.saved_image_path = saved_path
        elif st.session_state.demo_img_path:
            try:
                img = Image.open(st.session_state.demo_img_path)
                src_caption = f"Sample Image — {st.session_state.demo_img_label}"
            except Exception as e:
                st.warning(f"Failed to open sample image: {e}")
                st.session_state.demo_img_path = None
        
        if img is not None:
            # Display uploaded image
            with col1:
                st.subheader(src_caption)
                st.image(img, width=350, caption=src_caption)
            
            # Action buttons
            action_col1, action_col2 = st.columns(2)

            with action_col1:
                analyze_clicked = st.button("Analyze Image", use_container_width=True)

            with action_col2:
                chat_clicked = st.button(
                    "💬 Go to AI Vet Chatbot",
                    use_container_width=True,
                    disabled=st.session_state.predicted_disease is None,
                    key="go_to_chatbot_tab1"
                )

            if analyze_clicked:
                with st.spinner("Analyzing image..."):
                    model = load_model()

                    if model is not None:
                        img_array = preprocess_image(img)
                        st.session_state.processed_image = img_array

                        predicted_disease, confidence = predict_disease(model, img_array)

                        if predicted_disease:
                            st.session_state.predicted_disease = predicted_disease
                            st.session_state.confidence = confidence
                        else:
                            st.error("Failed to make a prediction. Please try again.")

            if st.session_state.predicted_disease and chat_clicked:
                st.session_state.switch_to_tab2 = True
                st.rerun()
            
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

            # Button untuk generate recommendation
            if st.button("Generate Veterinary Recommendations", use_container_width=True, key="generate_rec_tab2"):
                with st.spinner("🔄 Generating expert recommendations..."):
                    llm_selection = "Azure OpenAI" if st.session_state.llm_choice == "Azure OpenAI" else "Ollama"
                    azure_deployment_to_use = st.session_state.azure_deployment_name if llm_selection == "Azure OpenAI" else None

                    recommendation = asyncio.run(rag_pipeline(
                        st.session_state.predicted_disease,
                        llm_selection,
                        azure_deployment_name=azure_deployment_to_use,
                        chat_history=[]
                    ))

                st.session_state.recommendation = recommendation
                st.session_state.show_typing_animation = True
                st.session_state.chat_history = [{"role": "assistant", "content": recommendation}]
                st.rerun()

            if st.session_state.recommendation:
                st.subheader("Veterinary Recommendations")

                if st.session_state.show_typing_animation:
                    placeholder = st.empty()
                    render_typing_text(st.session_state.recommendation, placeholder)
                    st.session_state.show_typing_animation = False
                else:
                    st.markdown(st.session_state.recommendation)

                st.markdown("---")
                st.markdown(
                    """
                    **Disclaimer**: This is an AI-generated recommendation based on image analysis. 
                    Always consult with a qualified veterinarian for a proper diagnosis and treatment plan.
                    """
                )
        else:
            st.warning("Please upload and analyze an image in the 'Upload & Diagnose' tab first.")

        st.markdown("---")
        st.subheader("Ask Follow-up Questions")

        # Display chat history
        st.markdown('<div class="chat-history-container">', unsafe_allow_html=True)
        chat_messages_to_display = st.session_state.chat_history[1:] if len(st.session_state.chat_history) > 1 else []

        for message in chat_messages_to_display:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        st.markdown('</div>', unsafe_allow_html=True)

        # Trigger LLM response if a question is pending
        if st.session_state.chat_pending_llm and st.session_state.pending_question:
            with st.chat_message("assistant"):
                with st.spinner("Preparing expert response..."):
                    llm_selection = "Azure OpenAI" if st.session_state.llm_choice == "Azure OpenAI" else "Ollama"
                    azure_deployment_to_use = (
                        st.session_state.azure_deployment_name if llm_selection == "Azure OpenAI" else None
                    )

                    response = asyncio.run(
                        rag_pipeline(
                            st.session_state.predicted_disease,
                            llm_selection,
                            azure_deployment_name=azure_deployment_to_use,
                            user_question=st.session_state.pending_question,
                            chat_history=st.session_state.chat_history[:-1],
                        )
                    )

            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.session_state.chat_pending_llm = False
            st.session_state.pending_question = None
            st.rerun()

        if st.session_state.predicted_disease:
            st.markdown('<div class="fixed-chat-input">', unsafe_allow_html=True)
            user_question = st.chat_input("Ask a question about this condition...")
            st.markdown('</div>', unsafe_allow_html=True)

            if user_question:
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                st.session_state.pending_question = user_question
                st.session_state.chat_pending_llm = True
                st.rerun()
        else:
            st.info("Upload and analyze an image first to enable the chat feature.")

if __name__ == "__main__":
    main()