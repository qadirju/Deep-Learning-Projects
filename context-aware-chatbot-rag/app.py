import streamlit as st
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Context-Aware RAG Chatbot", layout="wide")
st.title("🤖 Context-Aware Chatbot with RAG")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    st.write("RAG Chatbot Configuration")
    mode = st.selectbox(
        "Select Mode",
        ["Full RAG (with LLM)", "Simulate Only"]
    )
    max_tokens = st.slider("Max Tokens", 50, 512, 256)

# Session state initialization
if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "status" not in st.session_state:
    st.session_state.status = ""
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

def initialize_rag_pipeline():
    """Initialize the full RAG pipeline with vector store, retriever, and LLM"""
    if st.session_state.rag_initialized:
        return
    
    try:
        with st.spinner("🔄 Initializing RAG pipeline..."):
            # Import required libraries
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_community.vectorstores import FAISS
            from langchain_community.llms import HuggingFacePipeline
            from transformers import pipeline as hf_pipeline
            
            # Sample documents (same as notebook)
            sample_documents = [
                """LangChain is a framework for developing applications powered by language models. 
                It enables applications that are data-aware and agentic, allowing them to interact with 
                their environment and use external tools for computation and information retrieval.""",
                
                """Retrieval-Augmented Generation (RAG) combines retrieval and generation capabilities. 
                It retrieves relevant documents from a knowledge base and uses them to augment the prompt 
                for better, more contextual responses from language models.""",
                
                """Vector databases like FAISS store embeddings of documents, enabling semantic search. 
                When a user query is converted to embeddings, the database finds similar documents 
                based on vector similarity, which is faster than traditional keyword matching.""",
                
                """Sentence Transformers are pre-trained models that encode text into dense vector representations. 
                These embeddings capture semantic meaning, allowing documents with similar meaning to have 
                similar vectors regardless of exact wording."""
            ]
            
            # 1. Create text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=50,
                separators=["\n\n", "\n", " ", ""]
            )
            
            # 2. Split documents
            document_chunks = text_splitter.create_documents(sample_documents)
            st.session_state.status = f"✓ Split {len(document_chunks)} chunks"
            
            # 3. Initialize embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # 4. Create FAISS vector store
            vector_store = FAISS.from_documents(document_chunks, embeddings)
            
            # 5. Create retriever
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            # 6. Initialize conversation memory
            class ConversationMemory:
                def __init__(self, memory_key: str = "chat_history"):
                    self.memory_key = memory_key
                    self.messages = []
                
                def add_message(self, role: str, content: str):
                    self.messages.append({"role": role, "content": content})
                
                def get_memory(self) -> List[Dict]:
                    return self.messages
                
                def clear(self):
                    self.messages = []
            
            memory = ConversationMemory(memory_key="chat_history")
            
            # 7. Initialize LLM
            hf_pipe = hf_pipeline(
                "text-generation",
                model="google/flan-t5-small",
                max_new_tokens=max_tokens
            )
            llm = HuggingFacePipeline(pipeline=hf_pipe)
            
            # 8. Create RAG Chain
            class RAGChain:
                def __init__(self, llm, retriever, memory):
                    self.llm = llm
                    self.retriever = retriever
                    self.memory = memory
                
                def retrieve_documents(self, query: str) -> List[str]:
                    docs = self.retriever.invoke(query)
                    return [doc.page_content for doc in docs]
                
                def format_context(self, retrieved_docs: List[str]) -> str:
                    return "\n\n".join([f"📄 {doc[:150]}..." for doc in retrieved_docs])
                
                def generate_response(self, query: str) -> Dict:
                    retrieved_docs = self.retrieve_documents(query)
                    context = self.format_context(retrieved_docs)
                    
                    chat_history = self.memory.get_memory()
                    history_text = "\n".join([f"{msg['role']}: {msg['content'][:80]}" for msg in chat_history[-4:]])  # Last 2 turns
                    
                    rag_prompt = f"""Context from knowledge base:
{context}

Chat History:
{history_text}

User Query: {query}

Provide a helpful response:"""
                    
                    response = self.llm.invoke(rag_prompt)
                    
                    self.memory.add_message("user", query)
                    self.memory.add_message("assistant", response)
                    
                    return {
                        "response": response,
                        "source_documents": retrieved_docs
                    }
            
            rag_chain = RAGChain(llm=llm, retriever=retriever, memory=memory)
            st.session_state.rag_chain = rag_chain
            st.session_state.rag_initialized = True
            st.session_state.status = "✓ RAG pipeline initialized successfully!"
            st.success("✅ RAG pipeline ready!")
            
    except Exception as e:
        st.session_state.status = f"⚠️ Failed to initialize: {str(e)[:100]}"
        st.error(f"Error: {str(e)}")

# Initialize based on mode
if mode == "Full RAG (with LLM)" and not st.session_state.rag_initialized:
    initialize_rag_pipeline()

# Display status
if st.session_state.status:
    st.sidebar.info(st.session_state.status)

# Chat interface
st.subheader("💬 Chat with your documents")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if user_input := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            if mode == "Full RAG (with LLM)" and st.session_state.rag_chain:
                try:
                    result = st.session_state.rag_chain.generate_response(user_input)
                    response = result['response']
                    sources = result['source_documents']
                    
                    # Display response
                    st.write(response)
                    
                    # Display sources
                    with st.expander("📚 Retrieved Sources"):
                        for i, source in enumerate(sources, 1):
                            st.write(f"**Source {i}:** {source[:200]}...")
                    
                except Exception as e:
                    response = f"Error generating response: {str(e)[:100]}"
                    st.error(response)
            else:
                # Simulation mode
                keywords = user_input.lower().split()
                if any(word in keywords for word in ["hello", "hi", "hey"]):
                    response = "Hello! I'm a context-aware RAG chatbot. Ask me anything about the documents or any topic."
                elif any(word in keywords for word in ["langchain", "rag", "retrieval", "faiss", "embedding"]):
                    response = "Great question! RAG combines retrieval and generation to provide context-aware responses. The system searches a knowledge base for relevant documents and uses them to augment the LLM's response."
                else:
                    response = f"📚 I found relevant information about '{user_input}'. This is an AI-generated response using Retrieval-Augmented Generation (RAG)."
                st.write(response)
        
        # Store assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

st.divider()
st.caption("🚀 Context-Aware RAG Chatbot | Powered by LangChain + FAISS + HuggingFace")
