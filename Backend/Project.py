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


# Load environment variables
load_dotenv()


# class RAG_ReAct_Agent():

    # def __init__():


    # def _initialize_vector_store():


    # def ingest_documents():


    # def ingest_from_url():


    # def retrieve():


    # def format_docs():


    # def process_query():


    
