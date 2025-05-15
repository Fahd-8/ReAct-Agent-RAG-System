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
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system with Qdrant Cloud credentials and ReAct_Rag collection
rag = RAG_ReAct_Agent(
    groq_api_key=os.environ["GROQ_API_KEY"],
    qdrant_url=os.environ["QDRANT_URL"],
    qdrant_api_key=os.environ["QDRANT_API_KEY"]
)

# Create or connect to the ReAct_Rag vector store
rag.create_new_vector_store("ReAct_Rag")

class IngestRequest(BaseModel):
    texts: List[str] = None
    url: str = None
    metadata: Dict[str, Any] = None

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
            raise HTTPException(status_code=400, detail="Either texts or url must be provided.")
        return {"message": "Documents ingested successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(request: QueryRequest):
    try:
        result = rag.process_query(request.query)
        return result
    except Exception as e:
        print(f"Error in /query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def get_documents():
    try:
        documents = rag.list_documents()
        return {"documents": documents}
    except Exception as e:
        print(f"Error fetching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)