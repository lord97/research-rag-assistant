"""
Streamlit UI for the Research RAG Assistant.
This provides a simple web interface for users to interact with the system.
"""

import streamlit as st
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from src.config import Config
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.query_engine import RAGQueryEngine

# Configure Streamlit page
st.set_page_config(
    page_title="Research Paper Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """
    Initialize Streamlit session state variables.
    Session state persists data across reruns within a session.
    """
    if "topic_id" not in st.session_state:
        st.session_state.topic_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "papers_uploaded" not in st.session_state:
        st.session_state.papers_uploaded = False

def validate_config():
    """
    Validates configuration and shows error if API key is missing.
    """
    try:
        Config.validate()
        return True
    except ValueError as e:
        st.error(f"Configuration Error: {str(e)}")
        st.info("Please add your GOOGLE_API_KEY to the .env file")
        st.stop()
        return False

def sidebar():
    """
    Creates the sidebar for topic management and file upload.
    """
    st.sidebar.title("📚 Research RAG Assistant")
    st.sidebar.markdown("---")
    
    # Section 1: Topic Management
    st.sidebar.subheader("1️⃣ Research Topic")
    
    topic_input = st.sidebar.text_input(
        "Enter your research topic:",
        placeholder="e.g., Machine Learning in Healthcare",
        help="This will be used to organize your papers"
    )
    
    if st.sidebar.button("Create/Load Topic", type="primary"):
        if topic_input:
            st.session_state.topic_id = topic_input
            st.session_state.messages = []  # Clear chat when changing topics
            st.sidebar.success(f"Topic set: {topic_input}")
        else:
            st.sidebar.error("Please enter a topic name")
    
    # Show current topic
    if st.session_state.topic_id:
        st.sidebar.info(f"Current Topic: **{st.session_state.topic_id}**")
    
    st.sidebar.markdown("---")
    
    # Section 2: File Upload
    st.sidebar.subheader("2️Upload Research Papers")
    
    if not st.session_state.topic_id:
        st.sidebar.warning("Please create a topic first")
        return
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF files:",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more research papers in PDF format"
    )
    
    if uploaded_files and st.sidebar.button("Process Papers"):
        process_papers(uploaded_files)
    
    st.sidebar.markdown("---")
    
    # Section 3: Info
    st.sidebar.subheader("ℹAbout")
    st.sidebar.markdown("""
    This tool helps researchers interact with their papers using AI.
    
    **How to use:**
    1. Enter your research topic
    2. Upload related PDF papers
    3. Ask questions about the papers
    
    **Features:**
    - Semantic search across papers
    - Source citations
    - Natural language queries
    """)

def process_papers(uploaded_files):
    """
    Processes uploaded PDF files and creates/updates vector store.
    
    Args:
        uploaded_files: List of uploaded file objects from Streamlit
    """
    with st.spinner("Processing papers... This may take a minute."):
        try:
            # Initialize processors
            doc_processor = DocumentProcessor()
            vector_manager = VectorStoreManager()
            
            # Save uploaded files to disk
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = doc_processor.save_uploaded_file(
                    uploaded_file, 
                    st.session_state.topic_id
                )
                file_paths.append(file_path)
            
            # Process PDFs: extract text and create chunks
            chunks = doc_processor.process_pdfs(file_paths)
            
            # Create/update vector store with chunks
            vector_manager.add_documents(chunks, st.session_state.topic_id)
            
            st.session_state.papers_uploaded = True
            st.sidebar.success(f"✓ Processed {len(uploaded_files)} papers successfully!")
            
        except Exception as e:
            st.sidebar.error(f"Error processing papers: {str(e)}")

def chat_interface():
    """
    Main chat interface for asking questions about the papers.
    """
    st.title("Chat with Your Research Papers")
    
    # Check if topic is set and papers are uploaded
    if not st.session_state.topic_id:
        st.info("Please create a research topic in the sidebar to get started")
        return
    
    # Check if papers have been uploaded
    vector_manager = VectorStoreManager()
    if not vector_manager.topic_exists(st.session_state.topic_id):
        st.info("Please upload research papers in the sidebar to start asking questions")
        return
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("View Sources"):
                        for source in message["sources"]:
                            st.markdown(f"""
                            **Source {source['number']}:** {source['filename']} (Page {source['page']})
                            
                            *Excerpt:* {source['content']}
                            
                            ---
                            """)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your papers..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Query the RAG system
                query_engine = RAGQueryEngine()
                result = query_engine.query(prompt, st.session_state.topic_id)
                
                if result["error"]:
                    st.error(result["error"])
                    response = result["error"]
                    sources = []
                else:
                    response = result["answer"]
                    sources = result["sources"]
                    
                    # Display answer
                    st.markdown(response)
                    
                    # Display sources
                    if sources:
                        with st.expander("View Sources"):
                            for source in sources:
                                st.markdown(f"""
                                **Source {source['number']}:** {source['filename']} (Page {source['page']})
                                
                                *Excerpt:* {source['content']}
                                
                                ---
                                """)
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "sources": sources
        })

def main():
    """
    Main function that runs the Streamlit app.
    This is the entry point of the application.
    """
    # Initialize session state
    initialize_session_state()
    
    # Validate configuration (API keys, etc.)
    if not validate_config():
        return
    
    # Create sidebar for topic and file management
    sidebar()
    
    # Create main chat interface
    chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
        @ Mohamed Bachir SANOU | 
        <a href='https://github.com/lord97/research-rag-assistant.git' target='_blank'>GitHub</a>
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()