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

    def __init__(self, groq_api_key=None, qdrant_url=None, qdrant_api_key=None, collection_name="drugs", qdrant_path=None):
        """Initialize the RAG Project with components"""
        # Set environment variables if provided
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        if qdrant_url:
            os.environ["QDRANTWhenever_URL"] = qdrant_url
        if qdrant_api_key:
            os.environ["QDRANT_API_KEY"] = qdrant_api_key

        # Validate required environment variables
        if not os.environ.get("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY is required.")
        if not os.environ.get("QDRANT_URL") and not qdrant_path:
            raise ValueError("Either QDRANT_URL or qdrant_path is required.")
        if os.environ.get("QDRANT_URL") and not os.environ.get("QDRANT_API_KEY"):
            raise ValueError("QDRANT_API_KEY is required when using QDRANT_URL.")

        # Initialize embedding model (sentence-transformers/all-MiniLM-L6-v2, 384 dimensions)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        # Initialize Groq LLM
        self.llm = ChatGroq(
            model="llama3-70b-8192",
            temperature=0.7,
            max_tokens=4096
        )

        # Initialize Qdrant client
        self.collection_name = collection_name
        if os.environ.get("QDRANT_URL"):
            # Cloud-based Qdrant client
            self.qdrant_client = qdrant_client.QdrantClient(
                url=os.environ["QDRANT_URL"],
                api_key=os.environ["QDRANT_API_KEY"]
            )
        else:
            # Local Qdrant client
            self.qdrant_client = qdrant_client.QdrantClient(path=qdrant_path)

        # Initialize vector store
        self.vector_store = None
        self._initialize_vector_store()


    # def _initialize_vector_store():


    # def ingest_documents():


    # def ingest_from_url():


    # def retrieve():


    # def format_docs():


    # def process_query():


    
