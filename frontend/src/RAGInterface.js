import React, { useState, useEffect } from 'react';
import { MessageSquare, Search, Database, Send } from 'lucide-react';
import axios from 'axios';

const RAGInterface = () => {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([
    { role: 'system', content: 'Welcome! Ask me anything and I\'ll use RAG to respond.' }
  ]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [newDocUrl, setNewDocUrl] = useState('');

  // Fetch ingested documents (optional, for display)
  const fetchDocuments = async () => {
    try {
      const response = await axios.get('http://localhost:8000/documents'); // Optional endpoint if implemented
      setDocuments(response.data);
    } catch (error) {
      console.error('Error fetching documents:', error);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  // Process a query
  const processQuery = async () => {
    if (!query.trim()) return;

    setIsProcessing(true);
    setMessages(prev => [...prev, { role: 'user', content: query }]);

    try {
      const response = await axios.post('http://localhost:8000/query', { query });
      const { answer, retrieved_docs } = response.data;

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: answer,
        retrievedDocs: retrieved_docs.map(doc => ({
          id: doc.metadata.source || Math.random(),
          title: doc.metadata.source || 'Document',
          content: doc.content
        }))
      }]);
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Error processing query. Please try again.'
      }]);
    }

    setQuery('');
    setIsProcessing(false);
  };

  // Add a document via URL
  const addDocument = async () => {
    if (!newDocUrl.trim()) return;

    try {
      await axios.post('http://localhost:8000/ingest', { url: newDocUrl });
      setDocuments(prev => [...prev, {
        id: Math.random(),
        title: `Document from ${newDocUrl.substring(0, 20)}...`,
        content: `Content from ${newDocUrl}`
      }]);
      setNewDocUrl('');
    } catch (error) {
      console.error('Error ingesting document:', error);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <div className="bg-blue-600 text-white p-4">
        <h1 className="text-xl font-bold">ReAct Pattern RAG Implementation</h1>
      </div>
      <div className="flex flex-1 overflow-hidden">
        <div className="w-64 bg-gray-200 p-4 flex flex-col">
          <h2 className="font-bold flex items-center gap-2 mb-4">
            <Database size={18} /> Knowledge Base
          </h2>
          <div className="mb-4">
            <div className="flex gap-2 mb-2">
              <input 
                type="text" 
                className="flex-1 p-2 text-sm rounded border" 
                placeholder="Enter document URL" 
                value={newDocUrl}
                onChange={(e) => setNewDocUrl(e.target.value)}
              />
              <button 
                className="bg-blue-500 text-white p-2 rounded"
                onClick={addDocument}
              >
                +
              </button>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto">
            {documents.map(doc => (
              <div key={doc.id} className="bg-white p-2 rounded mb-2 text-sm">
                <div className="font-bold">{doc.title}</div>
                <div className="text-gray-600 truncate">{doc.content}</div>
              </div>
            ))}
          </div>
        </div>
        <div className="flex-1 flex flex-col">
          <div className="flex-1 p-4 overflow-y-auto">
            {messages.map((msg, index) => (
              <div key={index} className={`mb-4 ${msg.role === 'user' ? 'text-right' : ''}`}>
                <div className={`inline-block p-3 rounded-lg ${
                  msg.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-300 text-gray-800'
                }`}>
                  {msg.content}
                </div>
                {msg.retrievedDocs && (
                  <div className="mt-2 bg-gray-100 p-2 rounded text-left text-sm">
                    <div className="font-bold text-xs text-gray-500 flex items-center gap-1">
                      <Search size={12} /> Retrieved Documents:
                    </div>
                    {msg.retrievedDocs.map(doc => (
                      <div key={doc.id} className="mt-1">
                        • {doc.title}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
          <div className="p-4 border-t">
            <div className="flex gap-2">
              <input
                type="text"
                className="flex-1 p-2 border rounded"
                placeholder="Ask a question..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && processQuery()}
              />
              <button 
                className="bg-blue-500 text-white p-2 rounded"
                onClick={processQuery}
                disabled={isProcessing}
              >
                {isProcessing ? '...' : <Send size={20} />}
              </button>
            </div>
          </div>
        </div>
        <div className="w-64 bg-gray-200 p-4 overflow-y-auto">
          <h2 className="font-bold flex items-center gap-2 mb-4">
            <MessageSquare size={18} /> ReAct Flow
          </h2>
          <div className="bg-white p-3 rounded mb-3 text-sm">
            <div className="font-bold text-blue-600">1. User Query</div>
            <div className="text-gray-600">User sends a question</div>
          </div>
          <div className="bg-white p-3 rounded mb-3 text-sm">
            <div className="font-bold text-purple-600">2. LLM Reasoning</div>
            <div className="text-gray-600">AI understands the query intent</div>
          </div>
          <div className="bg-white p-3 rounded mb-3 text-sm">
            <div className="font-bold text-green-600">3. Tool Usage</div>
            <div className="text-gray-600">Retrieves relevant documents</div>
          </div>
          <div className="bg-white p-3 rounded mb-3 text-sm">
            <div className="font-bold text-red-600">4. Response Generation</div>
            <div className="text-gray-600">AI creates answer using retrieved info</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RAGInterface;