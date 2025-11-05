import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import './react_app.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Component to render markdown text (bold and italics)
function MarkdownText({ text }) {
  if (!text) return null;
  
  // Split by newlines first
  const lines = text.split('\n');
  
  const parseMarkdown = (line) => {
    const parts = [];
    let lastIndex = 0;
    let i = 0;
    
    while (i < line.length) {
      // Check for bold **text**
      if (line[i] === '*' && line[i + 1] === '*' && i + 2 < line.length) {
        const endIndex = line.indexOf('**', i + 2);
        if (endIndex !== -1) {
          // Add text before bold
          if (i > lastIndex) {
            parts.push(parseMarkdownInline(line.substring(lastIndex, i)));
          }
          // Add bold text
          const boldText = line.substring(i + 2, endIndex);
          parts.push(<strong key={`bold-${i}`}>{parseMarkdownInline(boldText)}</strong>);
          lastIndex = endIndex + 2;
          i = endIndex + 2;
          continue;
        }
      }
      // Check for italic *text* (but not **text**)
      else if (line[i] === '*' && (i === 0 || line[i - 1] !== '*') && (i === line.length - 1 || line[i + 1] !== '*')) {
        const endIndex = line.indexOf('*', i + 1);
        if (endIndex !== -1 && (endIndex === line.length - 1 || line[endIndex + 1] !== '*')) {
          // Add text before italic
          if (i > lastIndex) {
            parts.push(parseMarkdownInline(line.substring(lastIndex, i)));
          }
          // Add italic text
          const italicText = line.substring(i + 1, endIndex);
          parts.push(<em key={`italic-${i}`}>{italicText}</em>);
          lastIndex = endIndex + 1;
          i = endIndex + 1;
          continue;
        }
      }
      i++;
    }
    
    // Add remaining text
    if (lastIndex < line.length) {
      parts.push(parseMarkdownInline(line.substring(lastIndex)));
    }
    
    return parts.length > 0 ? parts : [line];
  };
  
  // Helper to parse inline markdown (for nested cases)
  const parseMarkdownInline = (text) => {
    const parts = [];
    let lastIndex = 0;
    let i = 0;
    
    while (i < text.length) {
      // Check for italic *text* (but not **text**)
      if (text[i] === '*' && (i === 0 || text[i - 1] !== '*') && (i === text.length - 1 || text[i + 1] !== '*')) {
        const endIndex = text.indexOf('*', i + 1);
        if (endIndex !== -1 && (endIndex === text.length - 1 || text[endIndex + 1] !== '*')) {
          // Add text before italic
          if (i > lastIndex) {
            parts.push(text.substring(lastIndex, i));
          }
          // Add italic text
          const italicText = text.substring(i + 1, endIndex);
          parts.push(<em key={`inline-italic-${i}`}>{italicText}</em>);
          lastIndex = endIndex + 1;
          i = endIndex + 1;
          continue;
        }
      }
      i++;
    }
    
    // Add remaining text
    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }
    
    return parts.length > 0 ? parts : text;
  };
  
  return (
    <>
      {lines.map((line, lineIdx) => {
        const parsed = parseMarkdown(line);
        return <p key={lineIdx}>{parsed}</p>;
      })}
    </>
  );
}

function ChatUI() {
  const navigate = useNavigate();
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedModel, setSelectedModel] = useState('llama3.2');
  const [availableModels, setAvailableModels] = useState(['llama3.2']);
  const [showContext, setShowContext] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    loadModels();
    // Welcome message
    setMessages([{
      role: 'assistant',
      content: "Hello! I'm your assistant for the Cairo Genizah collection. I can help you learn about historical manuscripts, answer questions about the collection, and provide information from scholarly bibliography references. What would you like to know?",
      bibliography_context: null
    }]);
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadModels = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/chat/models`);
      if (response.ok) {
        const data = await response.json();
        setAvailableModels(data.models || ['llama3.2']);
        if (data.default && !data.models.includes(selectedModel)) {
          setSelectedModel(data.default);
        }
      }
    } catch (err) {
      console.error('Failed to load models:', err);
    }
  };

  const handleSend = async (e) => {
    e.preventDefault();
    
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = inputMessage.trim();
    setInputMessage('');
    setError(null);

    // Add user message to chat
    const newUserMessage = {
      role: 'user',
      content: userMessage,
      bibliography_context: null
    };
    setMessages(prev => [...prev, newUserMessage]);
    setIsLoading(true);

    try {
      // Build conversation history (exclude the welcome message and current user message)
      const conversationHistory = messages
        .filter(msg => msg.role !== 'system')
        .slice(1) // Skip welcome message
        .map(msg => ({
          role: msg.role,
          content: msg.content
        }));

      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          conversation_history: conversationHistory.length > 0 ? conversationHistory : null,
          num_bibliography_results: 5,
          model: selectedModel
        }),
      });

      const data = await response.json();

      if (response.ok) {
        const assistantMessage = {
          role: 'assistant',
          content: data.message,
          bibliography_context: data.bibliography_context,
          model_used: data.model_used
        };
        setMessages(prev => [...prev, assistantMessage]);
      } else {
        setError(data.detail || 'Failed to get response');
        const errorMessage = {
          role: 'assistant',
          content: `Sorry, I encountered an error: ${data.detail || 'Unknown error'}`,
          bibliography_context: null,
          isError: true
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (err) {
      const errorMsg = 'Network error. Please check your connection and try again.';
      setError(errorMsg);
      const errorMessage = {
        role: 'assistant',
        content: `Sorry, I encountered an error: ${errorMsg}`,
        bibliography_context: null,
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearChat = () => {
    setMessages([{
      role: 'assistant',
      content: "Hello! I'm your assistant for the Cairo Genizah collection. I can help you learn about historical manuscripts, answer questions about the collection, and provide information from scholarly bibliography references. What would you like to know?",
      bibliography_context: null
    }]);
    setError(null);
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <div className="chat-header-content">
          <div className="chat-header-title">
            <button
              onClick={() => navigate('/')}
              className="back-to-search-btn"
              title="Back to Search"
            >
              ← Back
            </button>
            <div>
              <h1>Cairo Genizah Chat Assistant</h1>
              <p>Ask questions about the Cairo Genizah collection using RAG with bibliography search</p>
            </div>
          </div>
        </div>
        <div className="chat-header-controls">
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="model-select"
            disabled={isLoading}
          >
            {availableModels.map(model => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
          <button
            onClick={handleClearChat}
            className="clear-chat-btn"
            disabled={isLoading}
          >
            Clear Chat
          </button>
        </div>
      </div>

      {error && (
        <div className="chat-error">
          {error}
          <button onClick={() => setError(null)} className="error-dismiss">×</button>
        </div>
      )}

      <div className="chat-messages">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`chat-message ${message.role === 'user' ? 'user-message' : 'assistant-message'} ${message.isError ? 'error-message' : ''}`}
          >
            <div className="message-header">
              <span className="message-role">
                {message.role === 'user' ? 'You' : 'Assistant'}
              </span>
              {message.model_used && (
                <span className="message-model">({message.model_used})</span>
              )}
            </div>
            <div className="message-content">
              <MarkdownText text={message.content} />
            </div>
            {message.bibliography_context && message.bibliography_context.length > 0 && (
              <div className="message-context">
                <button
                  onClick={() => setShowContext(showContext === index ? null : index)}
                  className="context-toggle"
                >
                  {showContext === index ? '▼' : '▶'} Bibliography References ({message.bibliography_context.length})
                </button>
                {showContext === index && (
                  <div className="context-details">
                    {message.bibliography_context.map((bib, bibIndex) => (
                      <div key={bibIndex} className="context-item">
                        <h4>Reference {bibIndex + 1}</h4>
                        {bib.description && (
                          <p><strong>Description:</strong> {bib.description}</p>
                        )}
                        {bib.full_text && (
                          <p><strong>Text:</strong> {bib.full_text.length > 300 ? bib.full_text.substring(0, 300) + '...' : bib.full_text}</p>
                        )}
                        {bib.shelf_marks_mentioned && bib.shelf_marks_mentioned.length > 0 && (
                          <p><strong>Shelfmarks:</strong> {bib.shelf_marks_mentioned.join(', ')}</p>
                        )}
                        <p className="context-score">Similarity: {bib.similarity_score?.toFixed(3) || 'N/A'}</p>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
        {isLoading && (
          <div className="chat-message assistant-message">
            <div className="message-header">
              <span className="message-role">Assistant</span>
            </div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <form className="chat-input-form" onSubmit={handleSend}>
        <input
          type="text"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          placeholder="Ask about the Cairo Genizah collection..."
          className="chat-input"
          disabled={isLoading}
        />
        <button
          type="submit"
          disabled={isLoading || !inputMessage.trim()}
          className="chat-send-btn"
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>

      <style jsx>{`
        .chat-container {
          display: flex;
          flex-direction: column;
          height: 100vh;
          max-width: 1200px;
          margin: 0 auto;
          background: white;
        }

        .chat-header {
          padding: 20px;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border-bottom: 1px solid #e0e0e0;
        }

        .chat-header-content {
          display: flex;
          flex-direction: column;
          gap: 16px;
        }

        .chat-header-title {
          display: flex;
          align-items: flex-start;
          gap: 16px;
        }

        .back-to-search-btn {
          padding: 8px 16px;
          background: rgba(255, 255, 255, 0.2);
          color: white;
          border: 1px solid rgba(255, 255, 255, 0.3);
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
          transition: background 0.2s;
          white-space: nowrap;
          margin-top: 4px;
        }

        .back-to-search-btn:hover {
          background: rgba(255, 255, 255, 0.3);
        }

        .chat-header-title > div {
          flex: 1;
        }

        .chat-header-content h1 {
          margin: 0 0 8px 0;
          font-size: 28px;
          font-weight: 600;
        }

        .chat-header-content p {
          margin: 0;
          font-size: 14px;
          opacity: 0.9;
        }

        .chat-header-controls {
          display: flex;
          gap: 12px;
          margin-top: 16px;
          align-items: center;
        }

        .model-select {
          padding: 8px 12px;
          border: 1px solid rgba(255, 255, 255, 0.3);
          border-radius: 6px;
          background: rgba(255, 255, 255, 0.2);
          color: white;
          font-size: 14px;
          cursor: pointer;
        }

        .model-select option {
          background: #667eea;
          color: white;
        }

        .clear-chat-btn {
          padding: 8px 16px;
          background: rgba(255, 255, 255, 0.2);
          color: white;
          border: 1px solid rgba(255, 255, 255, 0.3);
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
          transition: background 0.2s;
        }

        .clear-chat-btn:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.3);
        }

        .clear-chat-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .chat-error {
          padding: 12px 20px;
          background: #fee;
          color: #c33;
          border-bottom: 1px solid #fcc;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .error-dismiss {
          background: none;
          border: none;
          color: #c33;
          font-size: 24px;
          cursor: pointer;
          padding: 0 8px;
          line-height: 1;
        }

        .chat-messages {
          flex: 1;
          overflow-y: auto;
          padding: 20px;
          background: #f5f5f5;
        }

        .chat-message {
          margin-bottom: 20px;
          max-width: 80%;
        }

        .user-message {
          margin-left: auto;
        }

        .assistant-message {
          margin-right: auto;
        }

        .message-header {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-bottom: 8px;
          font-size: 12px;
          font-weight: 600;
          color: #666;
        }

        .message-model {
          font-weight: normal;
          color: #999;
          font-size: 11px;
        }

        .message-content {
          padding: 12px 16px;
          border-radius: 12px;
          background: white;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        .user-message .message-content {
          background: #667eea;
          color: white;
        }

        .assistant-message .message-content {
          background: white;
          color: #333;
        }

        .error-message .message-content {
          background: #fee;
          color: #c33;
          border: 1px solid #fcc;
        }

        .message-content p {
          margin: 0 0 8px 0;
          line-height: 1.6;
        }

        .message-content p:last-child {
          margin-bottom: 0;
        }

        .message-content strong {
          font-weight: 700;
          color: inherit;
        }

        .message-content em {
          font-style: italic;
          color: inherit;
        }

        .assistant-message .message-content strong {
          font-weight: 700;
          color: #2c3e50;
        }

        .assistant-message .message-content em {
          font-style: italic;
          color: #7f8c8d;
        }

        .message-context {
          margin-top: 8px;
        }

        .context-toggle {
          background: none;
          border: none;
          color: #667eea;
          cursor: pointer;
          font-size: 12px;
          padding: 4px 8px;
          text-align: left;
        }

        .context-toggle:hover {
          text-decoration: underline;
        }

        .context-details {
          margin-top: 8px;
          padding: 12px;
          background: #f9f9f9;
          border-radius: 8px;
          border: 1px solid #e0e0e0;
        }

        .context-item {
          margin-bottom: 12px;
          padding-bottom: 12px;
          border-bottom: 1px solid #e0e0e0;
        }

        .context-item:last-child {
          border-bottom: none;
          margin-bottom: 0;
          padding-bottom: 0;
        }

        .context-item h4 {
          margin: 0 0 8px 0;
          font-size: 14px;
          color: #667eea;
        }

        .context-item p {
          margin: 4px 0;
          font-size: 13px;
          color: #666;
          line-height: 1.5;
        }

        .context-item strong {
          color: #333;
        }

        .context-score {
          font-size: 11px;
          color: #999;
          font-style: italic;
        }

        .typing-indicator {
          display: flex;
          gap: 4px;
          padding: 8px 0;
        }

        .typing-indicator span {
          width: 8px;
          height: 8px;
          background: #667eea;
          border-radius: 50%;
          animation: typing 1.4s infinite;
        }

        .typing-indicator span:nth-child(2) {
          animation-delay: 0.2s;
        }

        .typing-indicator span:nth-child(3) {
          animation-delay: 0.4s;
        }

        @keyframes typing {
          0%, 60%, 100% {
            transform: translateY(0);
            opacity: 0.7;
          }
          30% {
            transform: translateY(-10px);
            opacity: 1;
          }
        }

        .chat-input-form {
          display: flex;
          padding: 20px;
          background: white;
          border-top: 1px solid #e0e0e0;
          gap: 12px;
        }

        .chat-input {
          flex: 1;
          padding: 12px 16px;
          border: 2px solid #e0e0e0;
          border-radius: 24px;
          font-size: 14px;
          outline: none;
          transition: border-color 0.2s;
        }

        .chat-input:focus {
          border-color: #667eea;
        }

        .chat-input:disabled {
          background: #f5f5f5;
          cursor: not-allowed;
        }

        .chat-send-btn {
          padding: 12px 24px;
          background: #667eea;
          color: white;
          border: none;
          border-radius: 24px;
          font-size: 14px;
          font-weight: 600;
          cursor: pointer;
          transition: background 0.2s;
        }

        .chat-send-btn:hover:not(:disabled) {
          background: #5568d3;
        }

        .chat-send-btn:disabled {
          background: #ccc;
          cursor: not-allowed;
        }

        @media (max-width: 768px) {
          .chat-container {
            height: 100vh;
          }

          .chat-message {
            max-width: 90%;
          }

          .chat-header-controls {
            flex-direction: column;
            align-items: stretch;
          }

          .model-select,
          .clear-chat-btn {
            width: 100%;
          }
        }
      `}</style>
    </div>
  );
}

export default ChatUI;

