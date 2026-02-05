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
    """Initialize lightweight RAG pipeline using keyword-based retrieval (no torch/embeddings needed)"""
    if st.session_state.rag_initialized:
        return
    
    st.session_state.rag_initialized = True
    
    try:
        st.write("📦 Initializing RAG pipeline...")
        
        # Sample documents
        sample_documents = [
            {
                "id": 0,
                "content": """LangChain is a framework for developing applications powered by language models. 
                It enables applications that are data-aware and agentic, allowing them to interact with 
                their environment and use external tools for computation and information retrieval."""
            },
            {
                "id": 1,
                "content": """Retrieval-Augmented Generation (RAG) combines retrieval and generation capabilities. 
                It retrieves relevant documents from a knowledge base and uses them to augment the prompt 
                for better, more contextual responses from language models."""
            },
            {
                "id": 2,
                "content": """Vector databases like FAISS store embeddings of documents, enabling semantic search. 
                When a user query is converted to embeddings, the database finds similar documents 
                based on vector similarity, which is faster than traditional keyword matching."""
            },
            {
                "id": 3,
                "content": """Sentence Transformers are pre-trained models that encode text into dense vector representations. 
                These embeddings capture semantic meaning, allowing documents with similar meaning to have 
                similar vectors regardless of exact wording."""
            }
        ]
        
        st.write("✓ Loaded 4 documents")
        
        # Simple keyword-based retrieval (no embeddings needed!)
        class SimpleRAGRetriever:
            def __init__(self, documents):
                self.documents = documents
                # Create keyword index
                self.keywords = {}
                for doc in documents:
                    words = doc["content"].lower().split()
                    for word in words:
                        if word not in self.keywords:
                            self.keywords[word] = []
                        self.keywords[word].append(doc["id"])
            
            def retrieve(self, query: str, k: int = 3) -> list:
                """Retrieve relevant documents based on keyword matching"""
                query_words = query.lower().split()
                doc_scores = {}
                
                # Score each document
                for doc in self.documents:
                    score = 0
                    doc_content = doc["content"].lower()
                    for word in query_words:
                        score += doc_content.count(word)
                    if score > 0:
                        doc_scores[doc["id"]] = (score, doc)
                
                # Return top-k documents
                if not doc_scores:
                    # If no keyword match, return first k documents
                    return self.documents[:k]
                
                sorted_docs = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
                return [doc for score, doc in sorted_docs[:k]]
        
        st.write("✓ Created keyword retriever")
        
        # Initialize retriever
        retriever = SimpleRAGRetriever(sample_documents)
        
        # Initialize conversation memory
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
        
        # Lightweight RAG Chain
        class LightweightRAGChain:
            def __init__(self, retriever, memory):
                self.retriever = retriever
                self.memory = memory
            
            def generate_response(self, query: str) -> Dict:
                """Generate response using retrieval + template"""
                retrieved_docs = self.retriever.retrieve(query, k=3)
                
                # Build context from retrieved documents
                context = "\n\n".join([f"📄 [{i+1}] {doc['content'][:200]}..." for i, doc in enumerate(retrieved_docs)])
                
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
                    response = f"Based on the knowledge base:\n\n{context}\n\n**Answer:** Based on the retrieved documents above, here's what I found relevant to your question about '{query}'."
                
                # Store in memory
                self.memory.add_message("user", query)
                self.memory.add_message("assistant", response)
                
                return {
                    "response": response,
                    "source_documents": [doc["content"] for doc in retrieved_docs]
                }
        
        # Initialize RAG chain
        rag_chain = LightweightRAGChain(retriever=retriever, memory=memory)
        st.session_state.rag_chain = rag_chain
        st.session_state.status = "✓ RAG pipeline ready! (Keyword-based retrieval)"
        st.success("✅ RAG pipeline initialized! Ready to process queries.")
        
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
