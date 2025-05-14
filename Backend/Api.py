from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from Project import RAG_ReAct_Agent
import os
from fastapi.middleware.cors import CORSMiddleware  

app = FastAPI(title="RAG with Groq and Qdrant")



# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"], # Allow all headers
)


# Initialize RAG system
rag = RAG_ReAct_Agent()

class IngestRequest(BaseModel):
    texts: List[str] =None
    url: str =None
    metadata: Dict[str,Any] =None

class QueryRequest(BaseModel):
    query: str


@app.post("/ingest")
async def ingest(request: IngestRequest):
    try:
        if request.texts:
            rag.ingest_documents(request.texts, [request.metadata or {}] * len(request.texts))
        elif request.url:
            rag.ingest_from_url(request.url, request.metadata)
        else:
            raise HTTPException(status_code=400, detail = "Either texts or url must be provided.")
        return {"message":"Documents ingested successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    

@app.post("/query")
async def query(request: QueryRequest):
    try:
        result = rag.process_query(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

