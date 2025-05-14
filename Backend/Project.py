import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup  # For better web scraping
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import qdrant_client
import os
import traceback
from typing import List, Dict
from openai import OpenAI
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import json
from typing import Any, Dict
import re
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()


class RAG_ReAct_Agent():


    def __init__(self, groq_api_key=None, collection_name="rag_collection"):
        """Initialize the RAG Project with components"""
        # Set API key if provided, otherwise use env
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key

        # Check if API key is available
        if not os.environ.get("GROQ_API_KEY"):
            raise ValueError("Groq API key is required. Set it as an environment variable or pass it to constructor.")
        
        # Initialzie components
        #Using HuggingFace embeddings instead of OpenAI
        self.embedding_model = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Using Groq's LLM 
        self.llm = ChatGroq(
            model = "llama3-70b-8192",
            temperature=0.6
        )

        #Initialize Qdrant client and collection
        self.collection_name = collection_name
        self.qdrant_client = qdrant_client.QdrantClient(location=":memory:")  # Using in-memory for simplicity
        self.vector_store = None
        self._initialize_vector_store()





    # def _initialize_vector_store():


    # def ingest_documents():


    # def ingest_from_url():


    # def retrieve():


    # def format_docs():


    # def process_query():


    
