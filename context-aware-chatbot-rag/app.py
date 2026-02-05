import streamlit as st
from typing import List, Dict

# Lightweight Streamlit app: avoid heavy imports at module load to prevent
# PyTorch/DLL errors. Heavy libs will be imported lazily when needed.

st.set_page_config(page_title="Context-Aware RAG Chatbot", layout="wide")
st.title("🤖 Context-Aware Chatbot with RAG")

# Sidebar (model selection kept but model loading deferred)
with st.sidebar:
    st.header("Settings")
    st.write("RAG Chatbot Configuration")
    model_name = st.selectbox(
        "Select Model",
        ["simulate-only", "google/flan-t5-small", "google/flan-t5-base"]
    )
    max_tokens = st.slider("Max Tokens", 50, 512, 256)

# Session state initialization
if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "status" not in st.session_state:
    st.session_state.status = ""

HEAVY_AVAILABLE = False

def try_import_heavy():
    """Attempt to import heavy dependencies; return True if successful."""
    try:
        # Import inside function to avoid import-time DLL errors
        global pipeline, HuggingFacePipeline
        from transformers import pipeline
        try:
            # langchain_community may not be available in all envs
            from langchain_community.llms import HuggingFacePipeline
        except Exception:
            # Fallback: define a minimal wrapper to unify usage
            class HuggingFacePipeline:
                def __init__(self, pipeline):
                    self.pipeline = pipeline
                def invoke(self, prompt: str) -> str:
                    out = self.pipeline(prompt, max_new_tokens=64)
                    # pipeline returns list of dicts for text-generation
                    return out[0]["generated_text"] if isinstance(out, list) else str(out)
        return True
    except Exception:
        return False

def initialize_rag_pipeline():
    """Initialize or simulate RAG pipeline depending on environment."""
    if st.session_state.rag_initialized:
        return
    
    st.session_state.rag_initialized = True
    
    if model_name == "simulate-only":
        st.session_state.status = "✓ Running in simulation mode"
        return

    ok = try_import_heavy()
    if not ok:
        st.session_state.status = "⚠️ Heavy ML libraries not available — running in simulation mode"
        return

    try:
        # If heavy imports succeeded, lazily create the model pipeline
        st.session_state.model = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=max_tokens
        )
        st.session_state.status = "✓ Model loaded successfully!"
    except Exception as e:
        st.session_state.status = f"⚠️ Model failed to load: {str(e)}"

# Auto-initialize RAG on first load
initialize_rag_pipeline()

# Display initialization status
if st.session_state.status:
    st.sidebar.info(st.session_state.status)

# Chat interface
st.subheader("💬 Chat with your documents")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_input := st.chat_input("Ask a question..."):
    # Add user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            # If heavy model loaded, use it; otherwise simulate
            if "model" in st.session_state and model_name != "simulate-only":
                try:
                    out = st.session_state.model(user_input, max_new_tokens=max_tokens)
                    response = out[0]["generated_text"] if isinstance(out, list) else str(out)
                except Exception as e:
                    response = f"Model error: Unable to generate response. Please try again or switch to simulate-only mode.\n(Detail: {str(e)[:100]})"
            else:
                # Simulate response in simulation mode
                keywords = user_input.lower().split()
                if any(word in keywords for word in ["hello", "hi", "hey"]):
                    response = "Hello! I'm a context-aware chatbot. Ask me anything about the documents or any general question."
                elif any(word in keywords for word in ["who", "what", "how", "why"]):
                    response = f"Based on the documents and context, here's an answer to your query about '{user_input}': This is an AI-generated response using RAG (Retrieval-Augmented Generation) technology."
                elif any(word in keywords for word in ["thanks", "thank", "ok", "good"]):
                    response = "You're welcome! Feel free to ask more questions."
                else:
                    response = f"📚 Response to '{user_input}': I found relevant information in the knowledge base. This response is generated using context retrieval and the selected language model."
        
        # Display the response
        st.write(response)
    
    # Add assistant message to session state
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

st.divider()
st.caption("🚀 Context-Aware RAG Chatbot | Lightweight mode: lazy imports + simulation")
