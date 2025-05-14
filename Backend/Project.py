from typing import List, Dict, Any
from dotenv import load_dotenv
from bs4 import BeautifulSoup  # For better web scraping
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import qdrant_client,re,json,os, requests, traceback
from langchain_community.vectorstores import Qdrant 
from typing import List, Dict
from openai import OpenAI
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from typing import Any, Dict
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
            model = "mixtral-8x7b-32768",
            temperature=0.6
        )

        #Initialize Qdrant client and collection
        self.collection_name = collection_name
        self.qdrant_client = qdrant_client.QdrantClient(location=":memory:")  # Using in-memory for simplicity
        self.vector_store = None
        self._initialize_vector_store()



    def _initialize_vector_store(self):
        """Initialize the Qdrant vectorstore"""
        from langchain_community.vectorstores import Qdrant  # Import Qdrant vectorstore

        # Check if collection exists
        try:
            # Use get_collection with the collection_name
            collection_info = self.qdrant_client.get_collection(collection_name=self.collection_name)
            # If the collection exists, connect to it
            self.vector_store = Qdrant(
                client=self.qdrant_client,
                collection_name=self.collection_name,  # Corrected parameter name
                embeddings=self.embedding_model
            )
        except Exception as e:
            # Collection doesn't exist or other error (e.g., not found)
            self.vector_store = None


    def ingest_documents(self, texts: List[str], metadatas: List[Dict[str,Any]] = None) -> None:
        """
        Ingest documents into the vector store

        Args:
            text: List of text content
            metadatas: Optional list of metadata dictionaries
        """

        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        documents = [Document(page_content=text, metadata=metadata)
                     for text, metadata in zip(texts,metadatas)]
        
        #create or update vectorstore
        if self.vector_store is None:
            self.vector_store = Qdrant.from_documents(
                documents,
                self.embedding_model,
                location=":memory:",
                collection_name = self.collection_name,
            )
        else:
            self.vector_store.add_documents(documents)
        print(f"Ingested {len(documents)} documents into the Qdrant vector store")



    def ingest_from_url(self, url: str, metadata: Dict[str,Any] = None) -> None:
        """
        Fetch text from a URL and ingest it
        
        Args:
            url: URL to fetch content from
            metadata: Optional metadata to associate with the document
        """

        try:
            response = request.get(url)
            response.raise_for_status()
            text = response.text

            if metadata is None:
                meatadata = {"source": url}
            else:
                metadata["source"] = url
            self.ingest_documents([text], [metadata])
        except Exception as e:
            print(f"Error fetching content from {url}: {e}")

    

    def retrieve(self, query:str, k:int=3) -> List[Document]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: The search query
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        if self.vector_store is None:
            raise ValueError("No documents have been ingested yet")
        
        retrieved_docs = self.vector_store.similarity_search(query, k=k)
        return retrieved_docs
    

    def format_docs(self, docs:List[Document]) -> str:
        """Format documents into a string"""
        return "\n\n".join(f"Document {i+i}: \n {doc.page_content}" for i, doc in enumerate(docs))
    

    def process_query(self,query: str) -> str:
        """
        Process a user query using the ReAct pattern
        
        Args:
            query: User query
            
        Returns:
            Response from the LLM
        """
        if self.vector_store is None:
            raise ValueError("No documents have been ingested yet")
        

        #create retriever
        retriever = self.vector_store.as_retriever(search_kwargs={"k":3})

        # Setup the RAG prompt
        template = """
        You are an assistant that follows the ReAct pattern to answer questions.
        
        First, REASON about the question to understand what is being asked.
        Think step by step:
        1. What is the core information the user is looking for?
        2. What context would be helpful to answer this question?
        3. What specific details should I focus on in my response?
        
        Then, consider the following context retrieved from your knowledge base:
        
        {context}
        
        Based on this context and your reasoning, GENERATE a helpful answer to the question: {question}
        
        If the context doesn't contain enough information, you can say so and provide the best answer 
        based on your general knowledge.
        """

        prompt = ChatPromptTemplate.from_template(template)

        # Create the RAG chain

        rag_chain = (

            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        #Proccess the query
        response =rag_chain.invoke(query)
        return response
    


    def run_interactive(self):
        """Run an interactive session with the RAG system"""
        print("Welcome to RAG Interactive Mode!")
        print("Type 'exit' to quit, 'ingest <url>' to add content, or any question to query." )


        while True:
            user_input = input("\nYou: ").strip()

#             
            if user_input.lower == 'exit':
                print("Goodbye!")
                break

            elif user_input.lower().startswith('ingest '):
                url = user_input[7:].strip()
                print(f"Ingesting content from: {url}")
                self.ingest_from_url(url)

            else:
                try:
                    if self.vector_store is None:
                        print("No documents have been ingested yet. Please ingest some content first.")
                        continue
                    response= self.process_query(user_input)
                    print(f"\nAssistant: {response}")
                except Exception as e:
                    print(f"Error processing query: {e}")





# Example usage
if __name__ == "__main__":
    # Create a RAG project
    rag = RAG_ReAct_Agent()
    
    # Example data - ingest some documents
    example_texts = [
        "RAG stands for Retrieval-Augmented Generation. It's a technique that enhances LLM responses with external knowledge.",
        "The ReAct pattern combines reasoning and acting in AI systems. It involves reasoning about a query, taking actions, and generating responses.",
        "Vector databases store embeddings which are numerical representations of text, images, or other data that capture semantic meaning."
    ]
    
    example_metadata = [
        {"source": "RAG documentation", "topic": "RAG basics"},
        {"source": "ReAct paper", "topic": "AI patterns"},
        {"source": "Vector DB guide", "topic": "Embeddings"}
    ]
    
    # Ingest the example documents
    rag.ingest_documents(example_texts, example_metadata)
    
    # Option 1: Process a single query
    query = "What is RAG and how does it relate to the ReAct pattern?"
    response = rag.process_query(query)
    print(f"Query: {query}")
    print(f"Response: {response}")
    
    # Option 2: Run interactive mode
    # rag.run_interactive()