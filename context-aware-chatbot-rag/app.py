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
    """Initialize lightweight RAG pipeline using only FAISS and embeddings (no torch needed)"""
    if st.session_state.rag_initialized:
        return
    
    st.session_state.rag_initialized = True
    
    try:
        # Import only CPU-friendly libraries
        st.write("📦 Loading libraries...")
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        st.write("✓ Libraries loaded")
        
        # Sample documents
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
        st.write("📄 Creating text splitter...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # 2. Split documents
        st.write("✂️ Splitting documents...")
        document_chunks = text_splitter.create_documents(sample_documents)
        st.write(f"✓ Created {len(document_chunks)} document chunks")
        
        # 3. Initialize embeddings (CPU-only)
        st.write("🧠 Loading embedding model (sentence-transformers)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        st.write("✓ Embeddings loaded")
        
        # 4. Create FAISS vector store
        st.write("🔨 Creating FAISS vector store...")
        vector_store = FAISS.from_documents(document_chunks, embeddings)
        st.write("✓ Vector store created")
        
        # 5. Create retriever
        st.write("🔍 Creating retriever...")
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        st.write("✓ Retriever created")
        
        # 6. Initialize conversation memory
        st.write("💾 Setting up conversation memory...")
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
        st.write("✓ Memory initialized")
        
        # 7. Lightweight RAG Chain (no LLM, just retrieval + templates)
        st.write("🤖 Creating RAG chain...")
        class LightweightRAGChain:
            def __init__(self, retriever, memory):
                self.retriever = retriever
                self.memory = memory
            
            def retrieve_documents(self, query: str) -> List[str]:
                """Retrieve relevant documents for query"""
                docs = self.retriever.invoke(query)
                return [doc.page_content for doc in docs]
            
            def generate_response(self, query: str) -> Dict:
                """Generate response using retrieval + template (no heavy LLM)"""
                retrieved_docs = self.retrieve_documents(query)
                
                # Build context from retrieved documents
                context = "\n\n".join([f"📄 [{i+1}] {doc[:200]}..." for i, doc in enumerate(retrieved_docs)])
                
                # Simple template-based response generation
                query_lower = query.lower()
                
                if any(word in query_lower for word in ["langchain", "framework", "application"]):
                    response = f"Based on the knowledge base:\n\n{context}\n\n**Answer:** LangChain is a powerful framework that enables building AI applications with language models. From the retrieved documents, you can see it provides data-awareness and agentic capabilities for interacting with your environment and external tools."
                elif any(word in query_lower for word in ["rag", "retrieval", "generation"]):
                    response = f"Based on the knowledge base:\n\n{context}\n\n**Answer:** RAG (Retrieval-Augmented Generation) combines retrieval and generation to provide more accurate and contextual responses. The retrieved documents show how it retrieves relevant information to augment the language model's output."
                elif any(word in query_lower for word in ["faiss", "vector", "database", "semantic", "embedding"]):
                    response = f"Based on the knowledge base:\n\n{context}\n\n**Answer:** FAISS and vector databases enable semantic search by storing document embeddings and finding similar documents based on vector similarity. This is more effective than traditional keyword matching."
                elif any(word in query_lower for word in ["transformer", "embedding", "representation"]):
                    response = f"Based on the knowledge base:\n\n{context}\n\n**Answer:** Sentence Transformers create dense vector representations (embeddings) that capture semantic meaning. Documents with similar meaning will have similar vectors, enabling semantic search and understanding."
                else:
                    response = f"Based on the knowledge base:\n\n{context}\n\n**Answer:** Based on the retrieved documents above, here's what I found relevant to your question: {query}"
                
                # Store in memory
                self.memory.add_message("user", query)
                self.memory.add_message("assistant", response)
                
                return {
                    "response": response,
                    "source_documents": retrieved_docs
                }
        
        # Initialize RAG chain
        rag_chain = LightweightRAGChain(retriever=retriever, memory=memory)
        st.session_state.rag_chain = rag_chain
        st.session_state.status = "✓ RAG pipeline ready! (CPU-only mode)"
        st.success("✅ RAG pipeline initialized! Using semantic retrieval with knowledge base.")
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        tb = traceback.format_exc()
        st.session_state.status = f"❌ Failed: {error_msg[:60]}"
        st.error(f"❌ RAG Initialization Failed:\n\n{error_msg}\n\n**Traceback:**\n```\n{tb}\n```")
        print(f"RAG Initialization Error: {error_msg}")
        print(tb)
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
                    st.markdown(response)
                    
                    # Display sources
                    with st.expander("📚 Retrieved Sources"):
                        for i, source in enumerate(sources, 1):
                            st.write(f"\n**📖 Document {i}:**\n\n{source}")
                    
                except Exception as e:
                    response = f"❌ Error: {str(e)}"
                    st.error(response)
            else:
                # Fallback simulation (only if RAG chain failed to initialize)
                response = f"⚠️ RAG pipeline not initialized. Using basic response mode.\n\nYour question: {user_input}"
                st.write(response)
        
        # Store assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

st.divider()
st.caption("🚀 Context-Aware RAG Chatbot | Powered by LangChain + FAISS + HuggingFace")
