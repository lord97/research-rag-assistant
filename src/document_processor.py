"""
Document Processor module.
Handles loading PDFs, extracting text, and splitting into chunks.
"""

from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import shutil
from src.config import Config

class DocumentProcessor:
    """
    Processes PDF documents for the RAG system.
    Handles file upload, text extraction, and chunking.
    """
    
    def __init__(self):
        """
        Initialize the document processor with a text splitter.
        The text splitter breaks long documents into manageable chunks.
        """
        # Create a text splitter that breaks documents intelligently
        # It tries to split at natural boundaries (paragraphs, sentences)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,  # Maximum size of each chunk
            chunk_overlap=Config.CHUNK_OVERLAP,  # Overlap between chunks
            length_function=len,  # How to measure chunk size
            separators=["\n\n", "\n", " ", ""]  # Split priorities
        )
    
    def save_uploaded_file(self, uploaded_file, topic_id: str) -> str:
        """
        Saves an uploaded file to the uploads directory.
        
        Args:
            uploaded_file: File object from Streamlit uploader
            topic_id: Unique identifier for the research topic
            
        Returns:
            str: Path to the saved file
        """
        # Create a folder for this specific topic
        topic_folder = os.path.join(Config.UPLOADS_PATH, topic_id)
        os.makedirs(topic_folder, exist_ok=True)
        
        # Save the file
        file_path = os.path.join(topic_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Loads and extracts text from a single PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List[Document]: List of document objects with text content
        """
        try:
            # Use LangChain's PDF loader to extract text
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Add filename to metadata for citation purposes
            for doc in documents:
                doc.metadata["source"] = os.path.basename(file_path)
            
            return documents
        
        except Exception as e:
            raise Exception(f"Error loading PDF {file_path}: {str(e)}")
    
    def load_multiple_pdfs(self, file_paths: List[str]) -> List[Document]:
        """
        Loads multiple PDF files and combines them.
        
        Args:
            file_paths: List of paths to PDF files
            
        Returns:
            List[Document]: Combined list of all documents
        """
        all_documents = []
        
        for file_path in file_paths:
            try:
                docs = self.load_pdf(file_path)
                all_documents.extend(docs)
                print(f"✓ Loaded: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"✗ Failed to load {file_path}: {str(e)}")
        
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits documents into smaller chunks for better retrieval.
        
        Why chunking?
        - LLMs have token limits
        - Smaller chunks = more precise retrieval
        - Better context matching
        
        Args:
            documents: List of documents to split
            
        Returns:
            List[Document]: List of document chunks
        """
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk index to metadata for tracking
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
        
        return chunks
    
    def process_pdfs(self, file_paths: List[str]) -> List[Document]:
        """
        Complete processing pipeline: load PDFs and split into chunks.
        This is the main method you'll call from outside.
        
        Args:
            file_paths: List of PDF file paths
            
        Returns:
            List[Document]: Processed and chunked documents ready for embedding
        """
        print("Loading PDFs...")
        documents = self.load_multiple_pdfs(file_paths)
        
        print(f"Loaded {len(documents)} pages from {len(file_paths)} PDFs")
        
        print("Splitting into chunks...")
        chunks = self.split_documents(documents)
        
        print(f"Created {len(chunks)} chunks")
        
        return chunks
    
    def delete_topic_files(self, topic_id: str):
        """
        Deletes all files associated with a topic.
        Useful for cleanup or when user wants to start over.
        
        Args:
            topic_id: The topic identifier
        """
        topic_folder = os.path.join(Config.UPLOADS_PATH, topic_id)
        if os.path.exists(topic_folder):
            shutil.rmtree(topic_folder)
            print(f"Deleted files for topic: {topic_id}")