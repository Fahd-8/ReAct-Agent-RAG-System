import React, { useState, useEffect, useRef } from 'react';
import { MessageSquare, Search, Database, Send, Clock, User, Bot } from 'lucide-react';
import axios from 'axios';

const RAGInterface = () => {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([
    { role: 'system', content: 'Welcome! Ask me anything and I’ll use RAG to respond.', timestamp: new Date() }
  ]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [newDocUrl, setNewDocUrl] = useState('');
  const chatEndRef = useRef(null);

  // Fetch documents (optional, if backend supports listing)
  const fetchDocuments = async () => {
    try {
      const response = await axios.get('http://localhost:8000/documents'); // Optional endpoint
      setDocuments(response.data);
    } catch (error) {
      console.error('Error fetching documents:', error);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const processQuery = async () => {
    if (!query.trim()) return;

    setIsProcessing(true);
    setMessages(prev => [...prev, { role: 'user', content: query, timestamp: new Date() }]);

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
        })),
        timestamp: new Date()
      }]);
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Error processing query. Please try again.',
        timestamp: new Date()
      }]);
    }

    setQuery('');
    setIsProcessing(false);
  };

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

  const formatTimestamp = (date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="rag-container">
      {/* Header */}
      <header className="rag-header">
        <h1>ReAct Pattern RAG Implementation</h1>
      </header>

      <div className="rag-main">
        {/* Knowledge Base Sidebar */}
        <aside className="rag-sidebar knowledge-base">
          <h2>
            <Database className="inline-icon" /> Knowledge Base
          </h2>
          <div className="input-group">
            <input
              type="text"
              value={newDocUrl}
              onChange={(e) => setNewDocUrl(e.target.value)}
              placeholder="Enter document URL"
            />
            <button onClick={addDocument}>+</button>
          </div>
          <div className="documents-list">
            {documents.map(doc => (
              <div key={doc.id} className="document-card">
                <div className="document-title">{doc.title}</div>
                <div className="document-content">{doc.content}</div>
              </div>
            ))}
          </div>
        </aside>

        {/* Chat Area */}
        <main className="rag-chat">
          <div className="chat-messages">
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.role === 'user' ? 'user-message' : 'assistant-message'}`}>
                <div className="message-content">
                  {msg.role === 'assistant' && <Bot className="message-icon" />}
                  <div>
                    <p>{msg.content}</p>
                    <div className="message-timestamp">
                      <Clock className="inline-icon" /> {formatTimestamp(msg.timestamp)}
                    </div>
                  </div>
                  {msg.role === 'user' && <User className="message-icon" />}
                </div>
                {msg.retrievedDocs && (
                  <div className="retrieved-docs">
                    <div className="docs-header">
                      <Search className="inline-icon" /> Retrieved Documents
                    </div>
                    {msg.retrievedDocs.map(doc => (
                      <div key={doc.id} className="doc-item">
                        • <span className="doc-title">{doc.title}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>
          <div className="chat-input">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && processQuery()}
              placeholder="Ask a question..."
            />
            <button onClick={processQuery} disabled={isProcessing}>
              {isProcessing ? '...' : <Send className="inline-icon" />}
            </button>
          </div>
        </main>

        {/* ReAct Flow Sidebar */}
        <aside className="rag-sidebar react-flow">
          <h2>
            <MessageSquare className="inline-icon" /> ReAct Flow
          </h2>
          <div className="flow-timeline">
            <div className="flow-step">
              <span className="step-number">1</span>
              <div className="step-content">
                <div className="step-title">User Query</div>
                <div>User sends a question</div>
              </div>
            </div>
            <div className="flow-step">
              <span className="step-number">2</span>
              <div className="step-content">
                <div className="step-title">LLM Reasoning</div>
                <div>AI understands the query intent</div>
              </div>
            </div>
            <div className="flow-step">
              <span className="step-number">3</span>
              <div className="step-content">
                <div className="step-title">Tool Usage</div>
                <div>Retrieves relevant documents</div>
              </div>
            </div>
            <div className="flow-step">
              <span className="step-number">4</span>
              <div className="step-content">
                <div className="step-title">Response Generation</div>
                <div>AI creates answer using retrieved info</div>
              </div>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
};

export default RAGInterface;