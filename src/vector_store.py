"""
Vector Store Manager module.
Manages the Chroma vector database for storing and retrieving document embeddings.
"""

from typing import List, Optional
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
import os
import shutil
from src.config import Config

class VectorStoreManager:
    """
    Manages vector database operations for the RAG system.
    Handles creating, loading, and querying vector stores.
    """
    
    def __init__(self):
        """
        Initialize the vector store manager with embeddings model.
        Embeddings convert text into numerical vectors that capture meaning.
        """
        # Initialize Google's embedding model
        # This converts text to vectors (lists of numbers)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            google_api_key=Config.GOOGLE_API_KEY
        )
    
    def _get_collection_name(self, topic_id: str) -> str:
        """
        Generates a valid collection name from topic ID.
        Chroma has naming restrictions, so we sanitize the topic ID.
        
        Args:
            topic_id: Raw topic identifier
            
        Returns:
            str: Sanitized collection name
        """
        # Replace spaces and special characters with underscores
        # Chroma collection names must be alphanumeric + underscores
        return topic_id.replace(" ", "_").replace("-", "_").lower()
    
    def _get_persist_directory(self, topic_id: str) -> str:
        """
        Gets the directory path where this topic's vector DB is stored.
        
        Args:
            topic_id: Topic identifier
            
        Returns:
            str: Full path to the persistence directory
        """
        return os.path.join(Config.VECTOR_DB_PATH, self._get_collection_name(topic_id))
    
    def create_vector_store(
        self, 
        documents: List[Document], 
        topic_id: str
    ) -> Chroma:
        """
        Creates a new vector store from documents.
        This is called when user first uploads papers for a topic.
        
        Process:
        1. Takes document chunks
        2. Generates embeddings for each chunk
        3. Stores embeddings + text in Chroma
        4. Saves to disk for persistence
        
        Args:
            documents: List of document chunks to store
            topic_id: Unique identifier for this research topic
            
        Returns:
            Chroma: The created vector store object
        """
        print(f"Creating embeddings for {len(documents)} chunks...")
        
        # Get the directory where we'll save this vector DB
        persist_directory = self._get_persist_directory(topic_id)
        
        # Create the vector store
        # This automatically:
        # - Generates embeddings for all documents
        # - Stores them in Chroma
        # - Saves to disk
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self._get_collection_name(topic_id),
            persist_directory=persist_directory
        )
        
        print(f"Vector store created and saved to {persist_directory}")
        
        return vectorstore
    
    def load_vector_store(self, topic_id: str) -> Optional[Chroma]:
        """
        Loads an existing vector store from disk.
        Used when user returns to a topic they've already created.
        
        Args:
            topic_id: Topic identifier
            
        Returns:
            Chroma: Loaded vector store, or None if doesn't exist
        """
        persist_directory = self._get_persist_directory(topic_id)
        
        # Check if this topic's vector store exists
        if not os.path.exists(persist_directory):
            return None
        
        try:
            # Load the existing vector store from disk
            vectorstore = Chroma(
                collection_name=self._get_collection_name(topic_id),
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
            
            print(f"Loaded existing vector store for topic: {topic_id}")
            return vectorstore
        
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return None
    
    def add_documents(
        self, 
        documents: List[Document], 
        topic_id: str
    ) -> Chroma:
        """
        Adds new documents to an existing vector store.
        Used when user uploads additional papers to a topic.
        
        Args:
            documents: New document chunks to add
            topic_id: Topic identifier
            
        Returns:
            Chroma: Updated vector store
        """
        # Try to load existing vector store
        vectorstore = self.load_vector_store(topic_id)
        
        if vectorstore is None:
            # No existing store, create new one
            return self.create_vector_store(documents, topic_id)
        
        # Add new documents to existing store
        print(f"Adding {len(documents)} new chunks to existing vector store...")
        vectorstore.add_documents(documents)
        
        print("Documents added successfully")
        return vectorstore
    
    def delete_vector_store(self, topic_id: str):
        """
        Deletes a topic's vector store from disk.
        
        Args:
            topic_id: Topic identifier
        """
        persist_directory = self._get_persist_directory(topic_id)
        
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            print(f"Deleted vector store for topic: {topic_id}")
    
    def topic_exists(self, topic_id: str) -> bool:
        """
        Checks if a vector store exists for a given topic.
        
        Args:
            topic_id: Topic identifier
            
        Returns:
            bool: True if topic exists, False otherwise
        """
        persist_directory = self._get_persist_directory(topic_id)
        return os.path.exists(persist_directory)
    
    def get_retriever(self, topic_id: str, k: int = None):
        """
        Gets a retriever object for querying the vector store.
        A retriever finds the most relevant chunks for a question.
        
        Args:
            topic_id: Topic identifier
            k: Number of chunks to retrieve (default from config)
            
        Returns:
            Retriever object, or None if topic doesn't exist
        """
        vectorstore = self.load_vector_store(topic_id)
        
        if vectorstore is None:
            return None
        
        # Create a retriever that will find the top K most similar chunks
        # similarity_score_threshold could be added for filtering
        return vectorstore.as_retriever(
            search_kwargs={"k": k or Config.TOP_K_RESULTS}
        )