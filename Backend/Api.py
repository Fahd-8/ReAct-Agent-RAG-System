from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from Project import RAG_ReAct_Agent
import os


app = FastAPI(title="RAG with Groq and Qdrant")