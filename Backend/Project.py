from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Qdrant
import qdrant_client
from qdrant_client.models import Distance, VectorParams
from langchain_core.documents import Document
from typing import List, Dict, Any
from datetime import datetime
import os

class RAG_ReAct_Agent:
    def __init__(self, groq_api_key=None, qdrant_url=None, qdrant_api_key=None, collection_name=None, qdrant_path=None):
        """Initialize the RAG system without connecting to a default collection."""
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        if qdrant_url:
            os.environ["QDRANT_URL"] = qdrant_url
        if qdrant_api_key:
            os.environ["QDRANT_API_KEY"] = qdrant_api_key

        if not os.environ.get("GROQ_API_KEY"):
            raise ValueError("GROQ_API_KEY is required.")
        if not (os.environ.get("QDRANT_URL") or qdrant_path):
            raise ValueError("Either QDRANT_URL or qdrant_path is required.")

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        self.embedding_dim = 384  # Dimension for sentence-transformers/all-MiniLM-L6-v2
        self.llm = ChatGroq(model="llama3-70b-8192", temperature=0.7, max_tokens=4096)
        self.collection_name = collection_name

        if os.environ.get("QDRANT_URL"):
            self.qdrant_client = qdrant_client.QdrantClient(
                url=os.environ["QDRANT_URL"],
                api_key=os.environ["QDRANT_API_KEY"]
            )
        else:
            self.qdrant_client = qdrant_client.QdrantClient(path=qdrant_path or "./qdrant_data")

        self.vector_store = None

    def _initialize_vector_store(self):
        """Initialize or connect to a specific Qdrant vector store collection."""
        if not self.collection_name:
            raise ValueError("Collection name must be specified to initialize vector store.")
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
                )
                self.vector_store = Qdrant.from_texts(
                    texts=["Initial placeholder document to create collection."],
                    embedding=self.embedding_model,
                    url=os.environ["QDRANT_URL"],
                    api_key=os.environ["QDRANT_API_KEY"],
                    collection_name=self.collection_name
                )
                print(f"Created new Qdrant collection: {self.collection_name}")
            else:
                self.vector_store = Qdrant(
                    client=self.qdrant_client,
                    collection_name=self.collection_name,
                    embeddings=self.embedding_model
                )
                print(f"Connected to existing Qdrant collection: {self.collection_name}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            self.vector_store = None

    def create_new_vector_store(self, new_collection_name: str):
        """Create a new vector store in the Qdrant cluster and set it as the active vector store."""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            if new_collection_name in collection_names:
                print(f"Collection {new_collection_name} already exists. Connecting to it.")
                self.collection_name = new_collection_name
                self._initialize_vector_store()
                return

            self.qdrant_client.create_collection(
                collection_name=new_collection_name,
                vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
            )
            self.collection_name = new_collection_name
            self.vector_store = Qdrant.from_texts(
                texts=["Initial placeholder document for new collection."],
                embedding=self.embedding_model,
                url=os.environ["QDRANT_URL"],
                api_key=os.environ["QDRANT_API_KEY"],
                collection_name=new_collection_name
            )
            print(f"Successfully created new vector store: {new_collection_name}")
        except Exception as e:
            print(f"Error creating new vector store {new_collection_name}: {e}")
            raise

    def ingest_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """Ingest documents into the vector store."""
        if not texts or not isinstance(texts, list):
            raise ValueError("At least one text must be provided as a non-empty list.")
        if metadatas is None:
            metadatas = [{} for _ in texts]
        elif len(metadatas) != len(texts):
            raise ValueError("Number of metadatas must match number of texts.")
        
        documents = [Document(page_content=text.strip() or "Empty document", metadata=metadata or {}) 
                    for text, metadata in zip(texts, metadatas)]
        try:
            if self.vector_store is None:
                self._initialize_vector_store()
            self.vector_store.add_documents(documents)
            print(f"Added {len(documents)} documents to vector store.")
        except Exception as e:
            print(f"Error ingesting documents: {e}")
            raise

    def ingest_from_url(self, url: str, metadata: Dict[str, Any] = None):
        """Fetch and ingest text from a URL."""
        if not url or not isinstance(url, str):
            raise ValueError("A valid URL string is required.")
        try:
            import requests
            from bs4 import BeautifulSoup
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for element in soup(["script", "style", "nav", "footer"]):
                element.decompose()
            paragraphs = soup.find_all(["p", "h1", "h2", "h3", "article"])
            text = " ".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
            if not text.strip():
                raise ValueError("No meaningful text extracted from URL.")
            if metadata is None:
                metadata = {"source": url, "fetched_at": str(datetime.now())}
            else:
                metadata.update({"source": url, "fetched_at": str(datetime.now())})
            self.ingest_documents([text], [metadata])
            print(f"Successfully ingested content from {url}")
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
            raise

    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query using the ReAct pattern."""
        if self.vector_store is None or not self.list_documents():
            raise ValueError("No documents have been ingested yet.")
        try:
            # Retrieve fewer documents to reduce token count
            retrieved_docs = self.vector_store.similarity_search(query, k=1)  # Reduced from k=3 to k=1
            # Truncate each document to reduce token count
            max_chars_per_doc = 6000  # Roughly 1500 tokens per doc (6000 chars / 4)
            truncated_docs = []
            for doc in retrieved_docs:
                if len(doc.page_content) > max_chars_per_doc:
                    doc.page_content = doc.page_content[:max_chars_per_doc] + "..."
                truncated_docs.append(doc)
            context = "\n".join([doc.page_content for doc in truncated_docs])
            # Further truncate the entire context if needed
            max_total_chars = 20000  # Roughly 5000 tokens (20000 chars / 4)
            if len(context) > max_total_chars:
                context = context[:max_total_chars] + "..."
            response = self.llm.invoke(f"Context: {context}\n\nQuestion: {query}\nAnswer:")
            return {
                "answer": response.content,
                "retrieved_docs": [
                    {
                        "id": str(i),
                        "metadata": doc.metadata,
                        "content": doc.page_content
                    } for i, doc in enumerate(truncated_docs)
                ]
            }
        except Exception as e:
            print(f"Error processing query: {e}")
            raise

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the vector store with their metadata."""
        if self.vector_store is None or not self.collection_name:
            return []
        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1000
            )
            points = scroll_result[0]  # Points are in the first element of the tuple
            if points is None or not points:
                return []  # Return empty list if no points are found
            documents = []
            for point in points:
                if not hasattr(point, 'payload') or point.payload is None:
                    continue  # Skip points with no payload
                metadata = point.payload.get("metadata", {})
                if not isinstance(metadata, dict):  # Ensure metadata is a dict
                    continue
                title = metadata.get("source", "Unknown")
                documents.append({
                    "id": str(point.id),
                    "title": title
                })
            return documents
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []

if __name__ == "__main__":
    rag = RAG_ReAct_Agent(
        groq_api_key=os.environ["GROQ_API_KEY"],
        qdrant_url=os.environ["QDRANT_URL"],
        qdrant_api_key=os.environ["QDRANT_API_KEY"]
    )
    
    rag.create_new_vector_store("ReAct_Rag")
    
    rag.ingest_from_url("https://en.wikipedia.org/wiki/Aspirin")
    
    query = "What is aspirin used for?"
    response = rag.process_query(query)
    print(f"Query: {query}")
    print(f"Response: {response}")