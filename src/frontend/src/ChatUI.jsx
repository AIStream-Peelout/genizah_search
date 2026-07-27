import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import './react_app.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
// Shared API key for the chat/LM Studio endpoints. Baked at build time (CRA).
// Empty in dev/local, where the backend leaves the check disabled.
const CHAT_API_KEY = process.env.REACT_APP_CHAT_API_KEY || '';

/**
 * Merge the shared chat API key (when configured) into a fetch headers object.
 * @param {Object} headers Base headers to extend.
 * @returns {Object} Headers including the X-API-Key header if a key is set.
 */
const withApiKey = (headers = {}) =>
  CHAT_API_KEY ? { ...headers, 'X-API-Key': CHAT_API_KEY } : headers;

// Component to render markdown text (bold and italics)
// Helper to escape regex characters
function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Cookie helpers
const setCookie = (name, value, hours) => {
  const expires = new Date();
  expires.setTime(expires.getTime() + (hours * 60 * 60 * 1000));
  document.cookie = `${name}=${value};expires=${expires.toUTCString()};path=/`;
};

const getCookie = (name) => {
  const nameEQ = name + "=";
  const ca = document.cookie.split(';');
  for (let i = 0; i < ca.length; i++) {
    let c = ca[i];
    while (c.charAt(0) === ' ') c = c.substring(1, c.length);
    if (c.indexOf(nameEQ) === 0) return c.substring(nameEQ.length, c.length);
  }
  return null;
};

const CHAT_SESSION_COOKIE = 'genizah_chat_session';
const SESSION_DURATION_HOURS = 4;
const LOCAL_STORAGE_KEY = 'genizah_chat_history';
const DISCLAIMER_SHOWN_KEY = 'genizah_disclaimer_seen';

// Inline flag markers emitted by the backend around claims the verification
// model could not support: ⟦flag:N⟧…⟦/flag⟧. N indexes into flagged_claims.
const FLAG_MARKER_REGEX = /⟦flag:(\d+)⟧([\s\S]*?)⟦\/flag⟧/g;

// A claim the verifier could not support: highlighted, clickable, and showing
// the verifier's exact reasoning in a popover so the user can judge it.
function FlaggedSpan({ flag, children }) {
  const [open, setOpen] = useState(false);

  return (
    <span className="flagged-claim-wrapper" style={{ position: 'relative', display: 'inline' }}>
      <span
        className="flagged-claim"
        onClick={() => setOpen(prev => !prev)}
        title="This claim could not be verified against the retrieved sources — click for details"
        style={{
          backgroundColor: '#fff0f0',
          color: '#b71c1c',
          borderBottom: '2px dotted #d32f2f',
          cursor: 'pointer',
          borderRadius: '2px',
          padding: '0 2px'
        }}
      >
        {children}
      </span>
      {open && (
        <span
          className="flagged-claim-popover"
          style={{
            position: 'absolute',
            zIndex: 30,
            top: '100%',
            left: 0,
            minWidth: '260px',
            maxWidth: '380px',
            background: '#fff',
            border: '1px solid #d32f2f',
            borderRadius: '6px',
            boxShadow: '0 4px 12px rgba(0,0,0,0.18)',
            padding: '10px 12px',
            fontSize: '0.85rem',
            color: '#333',
            display: 'block',
            whiteSpace: 'normal'
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <span style={{ display: 'block', fontWeight: 600, color: '#b71c1c', marginBottom: '4px' }}>
            ⚠ Unverified {String(flag?.claim_type || 'claim').replace(/_/g, ' ')}
          </span>
          {flag?.text && (
            <span style={{ display: 'block', fontStyle: 'italic', marginBottom: '6px' }}>
              “{flag.text}”
            </span>
          )}
          <span style={{ display: 'block', marginBottom: flag?.source_citation ? '6px' : 0 }}>
            <strong>Verifier:</strong> {flag?.reason || 'No reasoning recorded.'}
          </span>
          {flag?.source_citation && (
            <span style={{ display: 'block', color: '#666' }}>
              Cited: {flag.source_citation}
            </span>
          )}
          <button
            onClick={() => setOpen(false)}
            style={{
              marginTop: '8px', border: 'none', background: '#f5f5f5',
              borderRadius: '4px', padding: '3px 10px', cursor: 'pointer', fontSize: '0.8rem'
            }}
          >
            Close
          </button>
        </span>
      )}
    </span>
  );
}

// Session cache for /book-info lookups keyed by normalized title.
const bookInfoCache = new Map();

const normalizeTitle = (value) =>
  String(value || '').toLowerCase().replace(/[^a-z0-9֐-׿]+/g, ' ').trim();

// A cited work title: click for publication details and a WorldCat locator
// link (full text can't be shown for copyright reasons).
function BookTitleSpan({ title, children }) {
  const [open, setOpen] = useState(false);
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [alignRight, setAlignRight] = useState(false);

  const openPopup = async (event) => {
    // Anchor the popover to whichever side has room so it never clips at
    // the edge of the chat column.
    const rect = event?.currentTarget?.getBoundingClientRect?.();
    if (rect) setAlignRight(rect.left > window.innerWidth * 0.55);
    setOpen(prev => !prev);
    if (info || loading) return;
    const key = normalizeTitle(title);
    if (bookInfoCache.has(key)) {
      setInfo(bookInfoCache.get(key));
      return;
    }
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/book-info?title=${encodeURIComponent(title)}`);
      const data = await response.json();
      bookInfoCache.set(key, data);
      setInfo(data);
    } catch (err) {
      setInfo({ title, worldcat_url: `https://search.worldcat.org/search?q=${encodeURIComponent('ti:"' + title + '"')}` });
    } finally {
      setLoading(false);
    }
  };

  return (
    <span className="book-title-wrapper" style={{ position: 'relative', display: 'inline' }}>
      <em
        className="book-title-link"
        onClick={openPopup}
        title="Click for publication details and where to find this work"
        style={{ cursor: 'pointer', textDecorationLine: 'underline', textDecorationStyle: 'dotted', textUnderlineOffset: '3px' }}
      >
        {children}
      </em>
      {open && (
        <span
          className="book-info-popover"
          style={{
            position: 'absolute', zIndex: 30, top: '100%',
            ...(alignRight ? { right: 0 } : { left: 0 }),
            minWidth: '260px', maxWidth: 'min(360px, 86vw)', background: '#fff',
            border: '1px solid #667eea', borderRadius: '8px',
            boxShadow: '0 4px 14px rgba(0,0,0,0.18)', padding: '12px 14px',
            fontSize: '0.85rem', color: '#333', display: 'block', whiteSpace: 'normal'
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {loading && <span>Looking up publication details…</span>}
          {!loading && (
            <>
              <span style={{ display: 'block', fontWeight: 600, marginBottom: '4px' }}>
                📖 {info?.title || title}
              </span>
              {(info?.authors || []).length > 0 && (
                <span style={{ display: 'block', marginBottom: '2px' }}>
                  {info.authors.slice(0, 3).join('; ')}
                </span>
              )}
              <span style={{ display: 'block', color: '#555', marginBottom: '6px' }}>
                {[
                  info?.year,
                  info?.journal ? `Journal article · ${info.journal}` : null,
                  info?.publisher
                ].filter(Boolean).join(' · ') ||
                  'No catalog record in the knowledge graph yet.'}
              </span>
              <span style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                {info?.doi_url && (
                  <a
                    href={info.doi_url} target="_blank" rel="noopener noreferrer"
                    style={{
                      background: '#667eea', color: '#fff', padding: '5px 10px',
                      borderRadius: '5px', textDecoration: 'none', fontSize: '0.8rem'
                    }}
                  >
                    DOI ↗
                  </a>
                )}
                {info?.work_type === 'journal_article' ? (
                  <>
                    {info?.scholar_url && (
                      <a
                        href={info.scholar_url} target="_blank" rel="noopener noreferrer"
                        style={{
                          background: info?.doi_url ? '#f5f5f5' : '#667eea',
                          color: info?.doi_url ? '#333' : '#fff', padding: '5px 10px',
                          borderRadius: '5px', textDecoration: 'none', fontSize: '0.8rem'
                        }}
                      >
                        Google Scholar ↗
                      </a>
                    )}
                    {info?.worldcat_url && (
                      <a
                        href={info.worldcat_url} target="_blank" rel="noopener noreferrer"
                        style={{
                          background: '#f5f5f5', color: '#333', padding: '5px 10px',
                          borderRadius: '5px', textDecoration: 'none', fontSize: '0.8rem'
                        }}
                      >
                        Journal in WorldCat ↗
                      </a>
                    )}
                  </>
                ) : (
                  <>
                    {info?.worldcat_url && (
                      <a
                        href={info.worldcat_url} target="_blank" rel="noopener noreferrer"
                        style={{
                          background: info?.doi_url ? '#f5f5f5' : '#667eea',
                          color: info?.doi_url ? '#333' : '#fff', padding: '5px 10px',
                          borderRadius: '5px', textDecoration: 'none', fontSize: '0.8rem'
                        }}
                      >
                        Find in WorldCat ↗
                      </a>
                    )}
                    {info?.scholar_url && (
                      <a
                        href={info.scholar_url} target="_blank" rel="noopener noreferrer"
                        style={{
                          background: '#f5f5f5', color: '#333', padding: '5px 10px',
                          borderRadius: '5px', textDecoration: 'none', fontSize: '0.8rem'
                        }}
                      >
                        Google Scholar ↗
                      </a>
                    )}
                  </>
                )}
                <button
                  onClick={() => setOpen(false)}
                  style={{
                    border: 'none', background: 'none', color: '#888',
                    cursor: 'pointer', fontSize: '0.8rem', padding: '5px 4px'
                  }}
                >
                  Close
                </button>
              </span>
            </>
          )}
        </span>
      )}
    </span>
  );
}

// Component to render markdown text (bold, italics, and shelfmark links)
// Component to render markdown text (bold, italics, and links)
function MarkdownText({ text, onShelfmarkClick, flaggedClaims, knownTitles }) {
  if (!text) return null;

  // Split by newlines
  const lines = text.split('\n');

  const flagsById = {};
  (flaggedClaims || []).forEach(flag => {
    if (flag && flag.flag_id != null) flagsById[flag.flag_id] = flag;
  });

  // Titles of works actually retrieved as evidence for this message; italic
  // spans matching one become clickable publication-info links.
  const normalizedKnownTitles = (knownTitles || []).filter(Boolean).map(t => ({
    raw: t,
    norm: normalizeTitle(t)
  }));
  const findKnownTitle = (candidate) => {
    const norm = normalizeTitle(candidate);
    if (norm.length < 8) return null;
    const hit = normalizedKnownTitles.find(t =>
      t.norm === norm || t.norm.includes(norm) || norm.includes(t.norm)
    );
    return hit ? hit.raw : null;
  };

  const parseMarkdown = (line) => {
    // 1. Handle Markdown Links: [text](url)
    const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = linkRegex.exec(line)) !== null) {
      const start = match.index;
      const end = match.index + match[0].length;
      const linkText = match[1];
      const linkUrl = match[2];

      // Add preceding styled text
      if (start > lastIndex) {
        parts.push(...parseStyles(line.substring(lastIndex, start)));
      }

      // Handle special doc: links
      if (linkUrl.startsWith('doc:')) {
        const docId = linkUrl.replace('doc:', '');
        parts.push(
          <button
            key={`doc-link-${start}`}
            onClick={() => onShelfmarkClick && onShelfmarkClick(linkText, [docId])}
            className="shelfmark-link"
            title={`View details for ${linkText}`}
          >
            {linkText}
          </button>
        );
      } else {
        // Standard external link
        parts.push(
          <a
            key={`ext-link-${start}`}
            href={linkUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="external-link"
          >
            {linkText}
          </a>
        );
      }

      lastIndex = end;
    }

    // Add remaining styled text
    if (lastIndex < line.length) {
      parts.push(...parseStyles(line.substring(lastIndex)));
    }

    return parts.length > 0 ? parts : parseStyles(line);
  };

  // Helper to parse bold, italic, and highlighting styles
  const parseStyles = (text) => {
    if (typeof text !== 'string') return [text];

    const parts = [];
    let lastIndex = 0;

    // Combined regex for highlighting, bold, and italic
    const styleRegex = /(:::red\[(.*?)\]:::)|(\*\*([^*]+)\*\*)|(\*([^*]+)\*)/g;
    let match;

    while ((match = styleRegex.exec(text)) !== null) {
      const [fullMatch, redWrap, redText, boldWrap, boldText, italicWrap, italicText] = match;
      const start = match.index;

      if (start > lastIndex) {
        parts.push(text.substring(lastIndex, start));
      }

      if (redWrap) {
        parts.push(
          <span key={`red-${start}`} className="highlight-red" style={{ backgroundColor: '#ffe6e6', color: '#d32f2f', padding: '2px 4px', borderRadius: '4px' }}>
            {redText}
          </span>
        );
      } else if (boldWrap) {
        parts.push(<strong key={`bold-${start}`}>{boldText}</strong>);
      } else if (italicWrap) {
        const matchedTitle = findKnownTitle(italicText);
        if (matchedTitle) {
          parts.push(
            <BookTitleSpan key={`book-${start}`} title={matchedTitle}>
              {italicText}
            </BookTitleSpan>
          );
        } else {
          parts.push(<em key={`italic-${start}`}>{italicText}</em>);
        }
      }

      lastIndex = start + fullMatch.length;
    }

    if (lastIndex < text.length) {
      parts.push(text.substring(lastIndex));
    }

    return parts.length > 0 ? parts : [text];
  };

  // Top-level pass: split out ⟦flag:N⟧…⟦/flag⟧ spans before normal markdown
  // parsing so flagged sentences render as clickable highlights.
  const parseFlags = (line) => {
    const parts = [];
    let lastIndex = 0;
    let match;
    FLAG_MARKER_REGEX.lastIndex = 0;
    while ((match = FLAG_MARKER_REGEX.exec(line)) !== null) {
      const [fullMatch, flagId, innerText] = match;
      if (match.index > lastIndex) {
        parts.push(...parseMarkdown(line.substring(lastIndex, match.index)));
      }
      const flag = flagsById[Number(flagId)];
      if (flag) {
        parts.push(
          <FlaggedSpan key={`flag-${flagId}-${match.index}`} flag={flag}>
            {parseMarkdown(innerText)}
          </FlaggedSpan>
        );
      } else {
        // No metadata for this marker — render the inner text unhighlighted.
        parts.push(...parseMarkdown(innerText));
      }
      lastIndex = match.index + fullMatch.length;
    }
    if (lastIndex === 0) {
      return parseMarkdown(line);
    }
    if (lastIndex < line.length) {
      parts.push(...parseMarkdown(line.substring(lastIndex)));
    }
    return parts;
  };

  return (
    <div className="markdown-container">
      {lines.map((line, i) => (
        <React.Fragment key={i}>
          {parseFlags(line)}
          {i < lines.length - 1 && <br />}
        </React.Fragment>
      ))}
    </div>
  );
}

function ChatUI({ onShelfmarkSearch, onPrimarySources, onDocumentClick, onShelfmarkClick, isSidebar = false, examplePrompts = null }) {
  const navigate = useNavigate();
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showContext, setShowContext] = useState(false);
  const [showGraphContext, setShowGraphContext] = useState(false);
  const [autoShowPrimarySources, setAutoShowPrimarySources] = useState(false);
  const [streamingStatus, setStreamingStatus] = useState(null);
  const [expandedClaims, setExpandedClaims] = useState({});
  const [showDisclaimer, setShowDisclaimer] = useState(false);
  // Optional synthesis-model picker (for testing different LM Studio models)
  const [availableModels, setAvailableModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [useCustomModel, setUseCustomModel] = useState(false);
  const [modelLoading, setModelLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const messageRefs = useRef({});

  // Default example prompts if not provided
  const defaultExamplePrompts = [
    { text: "Can you tell me about Ketubah's in the Cairo Genizah", icon: "💍" },
    { text: "Yom Kippur Piyyut Fragments", icon: "📜" },
    { text: "Who is S.D. Goitein", icon: "👤" }
  ];

  // Normalize prompts - handle both string arrays and object arrays
  const normalizePrompts = (prompts) => {
    if (!prompts) return defaultExamplePrompts;
    return prompts.map(p => typeof p === 'string' ? { text: p, icon: "💬" } : p);
  };

  const prompts = normalizePrompts(examplePrompts);

  useEffect(() => {
    // Check for existing session
    const sessionActive = getCookie(CHAT_SESSION_COOKIE);
    const savedHistory = localStorage.getItem(LOCAL_STORAGE_KEY);
    const disclaimerSeen = localStorage.getItem(DISCLAIMER_SHOWN_KEY);

    if (!disclaimerSeen) {
      setShowDisclaimer(true);
    }

    if (sessionActive && savedHistory) {
      try {
        const parsedHistory = JSON.parse(savedHistory);
        if (parsedHistory && parsedHistory.length > 0) {
          setMessages(parsedHistory);
          return;
        }
      } catch (err) {
        console.error('Failed to parse saved chat history:', err);
      }
    }

    // Default welcome message if no session or parse failed
    setMessages([{
      role: 'assistant',
      content: "Hello! I'm your assistant for the Cairo Genizah collection. I can help you learn about historical manuscripts, answer questions about the collection, and provide information from scholarly bibliography references. What would you like to know?",
      bibliography_context: null
    }]);

    // Start a new session if none exists
    if (!sessionActive) {
      setCookie(CHAT_SESSION_COOKIE, 'active', SESSION_DURATION_HOURS);
    }
  }, []);

  // Persist messages to localStorage whenever they change
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(messages));
      // Refresh session cookie on activity
      setCookie(CHAT_SESSION_COOKIE, 'active', SESSION_DURATION_HOURS);
    }
  }, [messages]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Load the list of LM Studio models available for the synthesis-model picker.
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/chat/models`, {
          headers: withApiKey(),
        });
        if (!response.ok) return;
        const data = await response.json();
        const models = data.models || [];
        setAvailableModels(models);
        // Default the picker to the server's default model (or the first available).
        setSelectedModel(data.default || models[0] || '');
      } catch (err) {
        console.warn('Could not load chat models:', err);
      }
    };
    fetchModels();
  }, []);

  const handleModelChange = async (modelId) => {
    setSelectedModel(modelId);
    setModelLoading(true);
    try {
      await fetch(`${API_BASE_URL}/chat/models/load`, {
        method: 'POST',
        headers: withApiKey({ 'Content-Type': 'application/json' }),
        body: JSON.stringify({ model: modelId }),
      });
    } catch (err) {
      console.warn('Model load request failed:', err);
    } finally {
      setModelLoading(false);
    }
  };

  const scrollToBottom = () => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  };

  const scrollToMessageTop = (index) => {
    const element = messageRefs.current[index];
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const handleDismissDisclaimer = () => {
    localStorage.setItem(DISCLAIMER_SHOWN_KEY, 'true');
    setShowDisclaimer(false);
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

      const chatRequestPayload = {
        message: userMessage,
        conversation_history: conversationHistory.length > 0 ? conversationHistory : null,
        num_bibliography_results: 5,
        // Only override the synthesis model when the user has opted in.
        ...(useCustomModel && selectedModel ? { model: selectedModel } : {})
      };

      try {
        const response = await fetch(`${API_BASE_URL}/chat-stream`, {
          method: 'POST',
          headers: withApiKey({
            'Content-Type': 'application/json',
          }),
          body: JSON.stringify(chatRequestPayload),
        });

        if (!response.ok) {
          throw new Error('Stream request failed');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let streamStarted = false;

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          streamStarted = true;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop(); // Keep the last partial line in buffer

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.substring(6);
              try {
                const data = JSON.parse(dataStr);

                if (data.type === 'status') {
                  setStreamingStatus(data);
                } else if (data.type === 'final') {
                  const finalData = data.data;
                  const assistantMessage = {
                    role: 'assistant',
                    content: finalData.answer,
                    resolved_query: finalData.resolved_query,
                    reasoning: finalData.query_plan?.reasoning,
                    verified_claims: finalData.verified_claims,
                    flagged_claims: finalData.flagged_claims,
                    verification_summary: finalData.verification_summary,
                    bibliography_context: finalData.bibliography_results,
                    graph_context: finalData.graph_results,
                    primary_sources: finalData.primary_source_results,
                    model_used: 'Agentic RAG'
                  };
                  setMessages(prev => [...prev, assistantMessage]);
                  setStreamingStatus(null);

                  // Scroll to the top of this new message when it's final
                  setTimeout(() => {
                    scrollToMessageTop(messages.length + 1);
                  }, 100);

                  // Auto-show primary sources if enabled
                  if (finalData.primary_source_results?.length > 0 && onPrimarySources && autoShowPrimarySources) {
                    onPrimarySources(finalData.primary_source_results);
                  }
                } else if (data.type === 'error') {
                  throw new Error(data.detail || 'Stream error');
                }
              } catch (err) {
                console.error('Error parsing stream chunk:', err);
              }
            }
          }
        }
      } catch (streamErr) {
        console.warn('Streaming failed, falling back to standard chat:', streamErr);

        // Fallback to non-streaming chat
        const fallbackResponse = await fetch(`${API_BASE_URL}/chat`, {
          method: 'POST',
          headers: withApiKey({
            'Content-Type': 'application/json',
          }),
          body: JSON.stringify(chatRequestPayload),
        });

        if (!fallbackResponse.ok) {
          const errorData = await fallbackResponse.json();
          throw new Error(errorData.detail || 'Fallback chat failed');
        }

        const finalData = await fallbackResponse.json();
        const assistantMessage = {
          role: 'assistant',
          content: finalData.answer,
          resolved_query: finalData.resolved_query,
          reasoning: finalData.query_plan?.reasoning,
          verified_claims: finalData.verified_claims,
          flagged_claims: finalData.flagged_claims,
          verification_summary: finalData.verification_summary,
          bibliography_context: finalData.bibliography_results,
          graph_context: finalData.graph_results,
          primary_sources: finalData.primary_source_results,
          model_used: 'Agentic RAG (Fallback)'
        };
        setMessages(prev => [...prev, assistantMessage]);
        setStreamingStatus(null);
      }
    } catch (err) {
      const errorMsg = err.message || 'Network error. Please check your connection and try again.';
      setError(errorMsg);
      const errorMessage = {
        role: 'assistant',
        content: `Sorry, I encountered an error: ${errorMsg}`,
        bibliography_context: null,
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
      setStreamingStatus(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearChat = () => {
    if (window.confirm('Are you sure you want to clear the chat history?')) {
      setMessages([{
        role: 'assistant',
        content: "Hello! I'm your assistant for the Cairo Genizah collection. I can help you learn about historical manuscripts, answer questions about the collection, and provide information from scholarly bibliography references. What would you like to know?",
        bibliography_context: null
      }]);
      localStorage.removeItem(LOCAL_STORAGE_KEY);
      setError(null);
    }
  };

  const handleExampleClick = async (promptText) => {
    if (isLoading || !promptText || !promptText.trim()) return;

    const userMessage = promptText.trim();
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

      const chatRequestPayload = {
        message: userMessage,
        conversation_history: conversationHistory.length > 0 ? conversationHistory : null,
        num_bibliography_results: 5,
        // Only override the synthesis model when the user has opted in.
        ...(useCustomModel && selectedModel ? { model: selectedModel } : {})
      };

      try {
        const response = await fetch(`${API_BASE_URL}/chat-stream`, {
          method: 'POST',
          headers: withApiKey({
            'Content-Type': 'application/json',
          }),
          body: JSON.stringify(chatRequestPayload),
        });

        if (!response.ok) {
          throw new Error('Stream request failed');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop();

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const dataStr = line.substring(6);
              try {
                const data = JSON.parse(dataStr);

                if (data.type === 'status') {
                  setStreamingStatus(data);
                } else if (data.type === 'final') {
                  const finalData = data.data;
                  const assistantMessage = {
                    role: 'assistant',
                    content: finalData.answer,
                    resolved_query: finalData.resolved_query,
                    reasoning: finalData.query_plan?.reasoning,
                    verified_claims: finalData.verified_claims,
                    flagged_claims: finalData.flagged_claims,
                    verification_summary: finalData.verification_summary,
                    bibliography_context: finalData.bibliography_results,
                    graph_context: finalData.graph_results,
                    primary_sources: finalData.primary_source_results,
                    model_used: 'Agentic RAG'
                  };
                  setMessages(prev => [...prev, assistantMessage]);
                  setStreamingStatus(null);

                  // Scroll to the top of this new message when it's final
                  setTimeout(() => {
                    scrollToMessageTop(messages.length + 1);
                  }, 100);

                  if (finalData.primary_source_results?.length > 0 && onPrimarySources && autoShowPrimarySources) {
                    onPrimarySources(finalData.primary_source_results);
                  }
                } else if (data.type === 'error') {
                  throw new Error(data.detail || 'Stream error');
                }
              } catch (err) {
                console.error('Error parsing stream chunk:', err);
              }
            }
          }
        }
      } catch (streamErr) {
        console.warn('Streaming failed, falling back to standard chat:', streamErr);

        // Fallback to non-streaming chat
        const fallbackResponse = await fetch(`${API_BASE_URL}/chat`, {
          method: 'POST',
          headers: withApiKey({
            'Content-Type': 'application/json',
          }),
          body: JSON.stringify(chatRequestPayload),
        });

        if (!fallbackResponse.ok) {
          const errorData = await fallbackResponse.json();
          throw new Error(errorData.detail || 'Fallback chat failed');
        }

        const finalData = await fallbackResponse.json();
        const assistantMessage = {
          role: 'assistant',
          content: finalData.answer,
          resolved_query: finalData.resolved_query,
          reasoning: finalData.query_plan?.reasoning,
          verified_claims: finalData.verified_claims,
          flagged_claims: finalData.flagged_claims,
          verification_summary: finalData.verification_summary,
          bibliography_context: finalData.bibliography_results,
          graph_context: finalData.graph_results,
          primary_sources: finalData.primary_source_results,
          model_used: 'Agentic RAG (Fallback)'
        };
        setMessages(prev => [...prev, assistantMessage]);
        setStreamingStatus(null);
      }
    } catch (err) {
      const errorMsg = err.message || 'Network error. Please check your connection and try again.';
      setError(errorMsg);
      const errorMessage = {
        role: 'assistant',
        content: `Sorry, I encountered an error: ${errorMsg}`,
        bibliography_context: null,
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
      setStreamingStatus(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`chat-container ${isSidebar ? 'chat-sidebar' : ''}`}>
      <div className="chat-header">
        <div className="chat-header-content">
          <div className="chat-header-title">
            {!isSidebar && (
              <button
                onClick={() => navigate('/')}
                className="back-to-search-btn"
                title="Back to Search"
              >
                ← Back
              </button>
            )}
            <div>
              <h1>Cairo Genizah Chat Assistant</h1>
              <p>Ask questions about the Cairo Genizah collection using RAG with bibliography search</p>
            </div>
          </div>
        </div>
        <div className="chat-header-controls">
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

      <div className="chat-messages" ref={messagesContainerRef}>
        {messages.map((message, index) => (
          <div
            key={index}
            ref={el => messageRefs.current[index] = el}
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
              {message.reasoning && (
                <div className="agent-reasoning">
                  <div className="reasoning-header">
                    <span className="icon">🧠</span> Thought Process
                  </div>
                  <div className="reasoning-text">{message.reasoning}</div>
                  {message.resolved_query && message.resolved_query !== message.content && (
                    <div className="resolved-query">
                      <span className="label">Understanding:</span> {message.resolved_query}
                    </div>
                  )}
                </div>
              )}

              <MarkdownText
                text={message.content}
                onShelfmarkClick={onShelfmarkClick}
                flaggedClaims={message.flagged_claims}
                knownTitles={[
                  ...(message.bibliography_context?.map(b => b.title) || []),
                  ...(message.graph_context?.flatMap(g => (g.works || []).map(w => w.title)) || [])
                ].filter(Boolean)}
                knownShelfmarks={[
                  ...(message.primary_sources?.map(s => s.shelf_mark || s.matched_shelf_mark) || []),
                  ...(message.bibliography_context?.flatMap(b => b.shelf_marks_mentioned || []) || [])
                ].filter(Boolean)}
                shelfmarkMap={
                  message.primary_sources?.reduce((acc, src) => {
                    if (src.shelf_mark) acc[src.shelf_mark] = src.doc_id;
                    if (src.matched_shelf_mark) acc[src.matched_shelf_mark] = src.doc_id;
                    return acc;
                  }, {}) || {}
                }
              />

              {(message.flagged_claims || []).some(f => f.claim_type === 'verification_error') && (
                <div className="verification-incomplete" style={{
                  marginTop: '10px', padding: '8px 12px', background: '#fff8ec',
                  border: '1px solid #f0d9a8', borderRadius: '6px', fontSize: '0.85rem', color: '#8a6d1a'
                }}>
                  ⚠ Automatic claim verification did not complete for this answer, so its
                  claims were not machine-checked against the sources. Treat citations with
                  the usual scholarly caution.
                </div>
              )}

              {(message.flagged_claims || []).filter(f => !f.answer_span && f.claim_type !== 'verification_error').length > 0 && (
                <div className="unanchored-flags" style={{
                  marginTop: '10px', padding: '8px 12px', background: '#fff7f7',
                  border: '1px solid #f2c1c1', borderRadius: '6px', fontSize: '0.85rem'
                }}>
                  <div style={{ fontWeight: 600, color: '#b71c1c', marginBottom: '4px' }}>
                    ⚠ Additional unverified claims
                  </div>
                  {message.flagged_claims.filter(f => !f.answer_span && f.claim_type !== 'verification_error').map((flag, idx) => (
                    <div key={idx} style={{ marginBottom: '6px' }}>
                      <em>“{flag.text}”</em>
                      <div style={{ color: '#666' }}>Verifier: {flag.reason}</div>
                    </div>
                  ))}
                </div>
              )}

              {message.verified_claims && message.verified_claims.length > 0 && (
                <div className="verification-section">
                  <button
                    className="verification-header-toggle"
                    onClick={() => setExpandedClaims(prev => ({ ...prev, [index]: !prev[index] }))}
                  >
                    <span className="icon">✅</span>
                    Verified Claims ({message.verified_claims.length})
                    <span className="toggle-arrow">{expandedClaims[index] ? '▼' : '▶'}</span>
                  </button>

                  {expandedClaims[index] && (
                    <div className="verified-claims-list">
                      {message.verified_claims.map((claim, idx) => (
                        <div key={idx} className={`verified-claim ${claim.verification_status.toLowerCase()}`}>
                          <div className="claim-text">{claim.claim}</div>
                          <div className="claim-citation">
                            <span className="citation-source">{claim.source_citation}</span>
                            {claim.quote && <span className="citation-quote">"{claim.quote}"</span>}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
            {message.graph_context && message.graph_context.length > 0 && (
              <div className="message-context graph-context">
                <button
                  onClick={() => setShowGraphContext(showGraphContext === index ? null : index)}
                  className="context-toggle"
                >
                  {showGraphContext === index ? '▼' : '▶'} Knowledge Graph Evidence ({message.graph_context.length})
                </button>
                {showGraphContext === index && (
                  <div className="context-details">
                    {message.graph_context.map((evidence, graphIndex) => {
                      const scholar = evidence.scholar || {};
                      const works = evidence.works || [];
                      const relationships = evidence.relationships || [];
                      return (
                        <div key={graphIndex} className="context-item graph-context-item">
                          <h4>{scholar.name || 'Scholar graph record'}</h4>
                          <p>
                            <strong>Graph provenance:</strong>{' '}
                            {(scholar.data_sources || []).join(', ') || 'unspecified'}
                          </p>
                          <p>
                            <strong>Connected works:</strong> {works.length};{' '}
                            <strong>studied fragments:</strong> {evidence.studied_fragment_count || 0}
                          </p>
                          {works.length > 0 && (
                            <ul>
                              {works.slice(0, 10).map((work, workIndex) => (
                                <li key={work.article_id || workIndex}>
                                  {work.title || 'Untitled'}{work.year ? ` (${work.year})` : ''}
                                  {` — ${work.referenced_fragment_count || 0} referenced fragments`}
                                </li>
                              ))}
                            </ul>
                          )}
                          {relationships.length > 0 && (
                            <p>
                              <strong>Relationships:</strong>{' '}
                              {relationships.map(rel => `${rel.relationship}: ${rel.name}`).join('; ')}
                            </p>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}
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
                        <h4>{bib.title || `Reference ${bibIndex + 1}`}</h4>
                        {(bib.authors?.length > 0 || bib.author || bib.extracted_page_number) && (
                          <p>
                            <strong>Author:</strong>{' '}
                            {(bib.authors?.length > 0 ? bib.authors.join(', ') : bib.author) || 'Unknown'}
                            {bib.extracted_page_number ? `, p. ${bib.extracted_page_number}` : ''}
                          </p>
                        )}
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
              {streamingStatus ? (
                <div className="streaming-status">
                  <div className="typing-indicator status-typing">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                  <div className="status-tracker">
                    <p className="status-text">{streamingStatus.status}</p>
                    <div className="status-indicators">
                      {streamingStatus.bibliography_count > 0 && (
                        <span className="status-stat">📚 {streamingStatus.bibliography_count} bibs</span>
                      )}
                      {streamingStatus.graph_count > 0 && (
                        <span className="status-stat">🕸️ {streamingStatus.graph_count} graph</span>
                      )}
                      {streamingStatus.primary_count > 0 && (
                        <span className="status-stat">📜 {streamingStatus.primary_count} manuscripts</span>
                      )}
                      {streamingStatus.verified_claims_count > 0 && (
                        <span className="status-stat">✅ {streamingStatus.verified_claims_count} verified</span>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              )}
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

      <div className="chat-controls-footer">
        <label className="auto-show-toggle" title="Automatically show primary source documents in the main view when mentioned">
          <input
            type="checkbox"
            checked={autoShowPrimarySources}
            onChange={(e) => setAutoShowPrimarySources(e.target.checked)}
          />
          <span className="toggle-label">Show primary sources automatically</span>
        </label>

        <label className="auto-show-toggle" title="Override the model used to write the final answer (experimental — for testing different LM Studio models)">
          <input
            type="checkbox"
            checked={useCustomModel}
            onChange={(e) => setUseCustomModel(e.target.checked)}
            disabled={availableModels.length === 0}
          />
          <span className="toggle-label">Choose synthesis model</span>
        </label>

        {useCustomModel && (
          <select
            className="model-select"
            value={selectedModel}
            onChange={(e) => handleModelChange(e.target.value)}
            disabled={isLoading || modelLoading}
            title="Model used for the final answer synthesis"
          >
            {availableModels.map((model) => (
              <option key={model} value={model}>{model}</option>
            ))}
          </select>
        )}
        {modelLoading && (
          <span className="model-loading-indicator">Loading model…</span>
        )}
      </div>

      {/* Example Prompts Section - At the bottom */}
      {messages.length === 1 && prompts.length > 0 && (
        <div className="chat-examples">
          <div className="examples-header">Try asking:</div>
          <div className="examples-list">
            {prompts.map((prompt, idx) => (
              <button
                key={idx}
                className="example-prompt-btn"
                onClick={() => handleExampleClick(prompt.text)}
                disabled={isLoading}
              >
                <span className="example-icon">{prompt.icon}</span>
                <span className="example-text">{prompt.text}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {showDisclaimer && (
        <div className="chat-disclaimer-overlay">
          <div className="chat-disclaimer-modal">
            <div className="disclaimer-header">
              <span className="experimental-tag">Experimental Feature</span>
              <button className="close-disclaimer" onClick={handleDismissDisclaimer}>×</button>
            </div>
            <h2>Welcome to Genizah Chat</h2>
            <p>
              This is an <strong>experimental</strong> AI-powered assistant for the Cairo Genizah collection.
              Please be aware that while we strive for accuracy:
            </p>
            <ul>
              <li>The assistant may occasionally generate incorrect information (hallucinations).</li>
              <li>Always verify facts and citations against the <strong>primary sources</strong> and bibliography provided.</li>
              <li>The system's performance and accuracy will continue to improve over time.</li>
            </ul>
            <div className="disclaimer-actions">
              <button className="dismiss-disclaimer-btn" onClick={handleDismissDisclaimer}>
                I Understand, Continue to Chat
              </button>
            </div>
          </div>
        </div>
      )}

      <style jsx>{`
        .chat-container {
          display: flex;
          flex-direction: column;
          height: 100%;
          max-width: 100%;
          margin: 0;
          background: white;
          width: 100%;
        }

        .chat-sidebar {
          border-left: 1px solid #e0e0e0;
          height: 100%;
        }

        .chat-examples {
          padding: ${isSidebar ? '10px 12px' : '16px 20px'};
          background: #f8f9fa;
          border-top: 1px solid #e0e0e0;
          flex-shrink: 0;
        }

        .examples-header {
          font-size: ${isSidebar ? '10px' : '12px'};
          font-weight: 600;
          color: #6c757d;
          margin-bottom: ${isSidebar ? '6px' : '8px'};
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .examples-list {
          display: flex;
          flex-direction: column;
          gap: ${isSidebar ? '6px' : '8px'};
        }

        .example-prompt-btn {
          padding: ${isSidebar ? '8px 10px' : '10px 14px'};
          background: white;
          border: 1px solid #e0e0e0;
          border-radius: 8px;
          cursor: pointer;
          font-size: ${isSidebar ? '11px' : '13px'};
          text-align: left;
          color: #495057;
          transition: all 0.2s;
          word-wrap: break-word;
          display: flex;
          align-items: center;
          gap: ${isSidebar ? '6px' : '10px'};
          width: 100%;
          line-height: 1.4;
        }

        .example-icon {
          font-size: ${isSidebar ? '14px' : '16px'};
          flex-shrink: 0;
        }

        .example-text {
          flex: 1;
          text-align: left;
          overflow: hidden;
          text-overflow: ellipsis;
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
        }

        .example-prompt-btn:hover:not(:disabled) {
          background: #667eea;
          color: white;
          border-color: #667eea;
          transform: translateX(4px);
        }

        .example-prompt-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .streaming-status {
          display: flex;
          flex-direction: column;
          gap: 12px;
          padding: 8px 0;
        }

        .status-typing {
          margin-bottom: 4px;
        }

        .status-tracker {
          background: #f8fbff;
          border-left: 4px solid #667eea;
          padding: 12px 16px;
          border-radius: 0 12px 12px 0;
          box-shadow: 0 2px 5px rgba(0,0,0,0.02);
        }

        .status-text {
          font-weight: 500;
          color: #4a5568;
          margin: 0 0 8px 0;
          font-size: 14px;
        }

        .status-indicators {
          display: flex;
          flex-wrap: wrap;
          gap: 8px;
        }

        .status-stat {
          font-size: 12px;
          background: white;
          padding: 2px 8px;
          border-radius: 12px;
          border: 1px solid #e2e8f0;
          color: #718096;
          display: flex;
          align-items: center;
          gap: 4px;
        }

        .verification-header-toggle {
          display: flex;
          align-items: center;
          width: 100%;
          background: #f0fff4;
          border: 1px solid #c6f6d5;
          padding: 8px 12px;
          border-radius: 8px;
          cursor: pointer;
          font-size: 14px;
          font-weight: 600;
          color: #2f855a;
          margin-bottom: 8px;
          transition: background 0.2s;
        }

        .verification-header-toggle:hover {
          background: #e6fffa;
        }

        .toggle-arrow {
          margin-left: auto;
          font-size: 10px;
          color: #48bb78;
        }

        .chat-header {
          padding: ${isSidebar ? '12px' : '20px 30px'};
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border-bottom: 1px solid rgba(0,0,0,0.1);
          flex-shrink: 0;
          display: flex;
          justify-content: space-between;
          align-items: center;
          width: 100%;
          box-shadow: 0 2px 10px rgba(0,0,0,0.1);
          z-index: 10;
        }
        
        ${!isSidebar ? `
        .chat-container {
          height: 100vh;
          max-width: 1200px;
          margin: 0 auto;
        }
        ` : ''}

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
          margin: 0 0 ${isSidebar ? '2px' : '4px'} 0;
          font-size: ${isSidebar ? '16px' : '24px'};
          font-weight: 700;
          line-height: 1.2;
          color: white;
        }

        .chat-header-content p {
          margin: 0;
          font-size: ${isSidebar ? '10px' : '14px'};
          color: rgba(255, 255, 255, 0.9);
          line-height: 1.4;
        }

        /* Agent Reasoning Styles */
        .agent-reasoning {
          background-color: #f0f7ff;
          border-left: 3px solid #667eea;
          padding: 10px 14px;
          margin-bottom: 12px;
          border-radius: 4px;
          font-size: 0.9em;
        }

        .reasoning-header {
          font-weight: 600;
          color: #4a5568;
          margin-bottom: 4px;
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .reasoning-text {
          color: #2d3748;
          line-height: 1.4;
        }

        .resolved-query {
          margin-top: 6px;
          padding-top: 6px;
          border-top: 1px solid #e2e8f0;
          font-style: italic;
          color: #718096;
          font-size: 0.85em;
        }

        /* Verification Styles */
        .verification-section {
          margin-top: 16px;
          border-top: 1px solid #e2e8f0;
          padding-top: 10px;
        }

        .verification-header {
          font-weight: 600;
          color: #2f855a;
          margin-bottom: 8px;
          font-size: 0.9em;
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .verified-claims-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .verified-claim {
          padding: 8px 10px;
          background: #f0fff4;
          border: 1px solid #c6f6d5;
          border-radius: 6px;
          font-size: 0.85em;
        }

        .claim-text {
          font-weight: 500;
          color: #276749;
          margin-bottom: 2px;
        }

        .claim-citation {
          display: block;
          color: #4a5568;
          font-size: 0.9em;
        }

        .citation-source {
          font-weight: 600;
        }

        .citation-quote {
          font-style: italic;
          color: #718096;
          margin-left: 4px;
        }

        /* Agent Reasoning Styles */
        .agent-reasoning {
          background-color: #f0f7ff;
          border-left: 3px solid #667eea;
          padding: 10px 14px;
          margin-bottom: 12px;
          border-radius: 4px;
          font-size: 0.9em;
        }

        .reasoning-header {
          font-weight: 600;
          color: #4a5568;
          margin-bottom: 4px;
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .reasoning-text {
          color: #2d3748;
          line-height: 1.4;
        }

        .resolved-query {
          margin-top: 6px;
          padding-top: 6px;
          border-top: 1px solid #e2e8f0;
          font-style: italic;
          color: #718096;
          font-size: 0.85em;
        }

        /* Verification Styles */
        .verification-section {
          margin-top: 16px;
          border-top: 1px solid #e2e8f0;
          padding-top: 10px;
        }

        .verification-header {
          font-weight: 600;
          color: #2f855a;
          margin-bottom: 8px;
          font-size: 0.9em;
          display: flex;
          align-items: center;
          gap: 6px;
        }

        .verified-claims-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .verified-claim {
          padding: 8px 10px;
          background: #f0fff4;
          border: 1px solid #c6f6d5;
          border-radius: 6px;
          font-size: 0.85em;
        }

        .claim-text {
          font-weight: 500;
          color: #276749;
          margin-bottom: 2px;
        }

        .claim-citation {
          display: block;
          color: #4a5568;
          font-size: 0.9em;
        }

        .citation-source {
          font-weight: 600;
        }

        .citation-quote {
          font-style: italic;
          color: #718096;
          margin-left: 4px;
        }

        .chat-header-controls {
          display: flex;
          gap: ${isSidebar ? '6px' : '12px'};
          margin-top: ${isSidebar ? '6px' : '16px'};
          align-items: center;
          flex-wrap: ${isSidebar ? 'wrap' : 'nowrap'};
        }

        .model-select {
          margin-top: 6px;
          padding: ${isSidebar ? '6px 10px' : '8px 12px'};
          border: 1px solid #ccc;
          border-radius: 6px;
          background: white;
          color: #333;
          font-size: ${isSidebar ? '12px' : '14px'};
          cursor: pointer;
          max-width: 100%;
        }

        .model-select option {
          background: white;
          color: #333;
        }

        .model-select:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .model-loading-indicator {
          margin-left: 8px;
          font-size: 12px;
          color: #667eea;
          font-style: italic;
        }

        .clear-chat-btn {
          padding: ${isSidebar ? '6px 12px' : '8px 16px'};
          background: rgba(255, 255, 255, 0.2);
          color: white;
          border: 1px solid rgba(255, 255, 255, 0.3);
          border-radius: 6px;
          cursor: pointer;
          font-size: ${isSidebar ? '12px' : '14px'};
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
          padding: ${isSidebar ? '8px 12px' : '12px 20px'};
          background: #fee;
          color: #c33;
          border-bottom: 1px solid #fcc;
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: ${isSidebar ? '11px' : '14px'};
          flex-shrink: 0;
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
          padding: ${isSidebar ? '12px 14px' : '20px'};
          background: #f5f5f5;
          min-height: 0;
        }

        .chat-message {
          margin-bottom: ${isSidebar ? '14px' : '20px'};
          max-width: ${isSidebar ? '95%' : '80%'};
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
          margin-bottom: ${isSidebar ? '6px' : '8px'};
          font-size: ${isSidebar ? '10px' : '12px'};
          font-weight: 600;
          color: #666;
        }

        .message-model {
          font-weight: normal;
          color: #999;
          font-size: ${isSidebar ? '9px' : '11px'};
        }

        .message-content {
          padding: ${isSidebar ? '10px 12px' : '12px 16px'};
          border-radius: 12px;
          background: white;
          box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
          word-wrap: break-word;
          overflow-wrap: break-word;
        }

        .user-message .message-content {
          background: #667eea;
          color: white;
        }

        .assistant-message .message-content {
          background: white;
          color: #333;
        }

        .shelfmark-link {
          background: rgba(102, 126, 234, 0.1);
          color: #667eea;
          border: none;
          padding: 0 4px;
          margin: 0 1px;
          border-radius: 4px;
          cursor: pointer;
          font-weight: 500;
          font-size: inherit;
          text-decoration: none;
          transition: all 0.2s;
          display: inline-block;
        }

        .shelfmark-link:hover {
          background: rgba(102, 126, 234, 0.2);
          text-decoration: underline;
        }

        .chat-controls-footer {
          padding: 0 ${isSidebar ? '12px' : '20px'} 10px;
          background: white;
        }

        .auto-show-toggle {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: ${isSidebar ? '11px' : '13px'};
          color: #666;
          cursor: pointer;
          user-select: none;
        }

        .auto-show-toggle input {
          cursor: pointer;
        }

        .error-message .message-content {
          background: #fee;
          color: #c33;
          border: 1px solid #fcc;
        }

        .message-content p {
          margin: 0 0 ${isSidebar ? '6px' : '8px'} 0;
          line-height: ${isSidebar ? '1.5' : '1.6'};
          font-size: ${isSidebar ? '13px' : '15px'};
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
          font-size: ${isSidebar ? '10px' : '12px'};
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
          font-size: ${isSidebar ? '12px' : '14px'};
          color: #667eea;
        }

        .context-item p {
          margin: 4px 0;
          font-size: ${isSidebar ? '11px' : '13px'};
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
          padding: ${isSidebar ? '12px 14px' : '20px'};
          background: white;
          border-top: 1px solid #e0e0e0;
          gap: ${isSidebar ? '8px' : '12px'};
          flex-shrink: 0;
        }

        .chat-input {
          flex: 1;
          padding: ${isSidebar ? '10px 12px' : '12px 16px'};
          border: 2px solid #e0e0e0;
          border-radius: 24px;
          font-size: ${isSidebar ? '13px' : '14px'};
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
          padding: ${isSidebar ? '10px 16px' : '12px 24px'};
          background: #667eea;
          color: white;
          border: none;
          border-radius: 24px;
          font-size: ${isSidebar ? '13px' : '14px'};
          font-weight: 600;
          cursor: pointer;
          transition: background 0.2s;
          white-space: nowrap;
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

          .message-content p {
            font-size: 14px;
          }

          .example-prompt-btn {
            font-size: 12px;
            padding: 8px 10px;
          }
        }

        @media (max-width: 480px) {
          .message-content p {
            font-size: 13px;
          }

          .example-prompt-btn {
            font-size: 11px;
            padding: 6px 8px;
          }

          .chat-messages {
            padding: 10px 12px;
          }

          .chat-input-form {
            padding: 10px 12px;
          }
        }

        .chat-disclaimer-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(0, 0, 0, 0.6);
          backdrop-filter: blur(4px);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
          padding: 20px;
        }

        .chat-disclaimer-modal {
          background: white;
          border-radius: 16px;
          max-width: 500px;
          width: 100%;
          padding: 30px;
          box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
          animation: modalSlideUp 0.3s ease-out;
        }

        @keyframes modalSlideUp {
          from { transform: translateY(20px); opacity: 0; }
          to { transform: translateY(0); opacity: 1; }
        }

        .disclaimer-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }

        .experimental-tag {
          background: #fff4e5;
          color: #b95d00;
          padding: 4px 12px;
          border-radius: 20px;
          font-size: 12px;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .close-disclaimer {
          background: none;
          border: none;
          font-size: 24px;
          color: #999;
          cursor: pointer;
          line-height: 1;
        }

        .chat-disclaimer-modal h2 {
          margin: 0 0 15px 0;
          color: #1a202c;
          font-size: 24px;
        }

        .chat-disclaimer-modal p {
          color: #4a5568;
          line-height: 1.6;
          margin-bottom: 20px;
        }

        .chat-disclaimer-modal ul {
          margin: 0 0 25px 0;
          padding-left: 20px;
          color: #4a5568;
        }

        .chat-disclaimer-modal li {
          margin-bottom: 10px;
          line-height: 1.5;
        }

        .disclaimer-actions {
          display: flex;
          justify-content: flex-end;
        }

        .dismiss-disclaimer-btn {
          background: #667eea;
          color: white;
          border: none;
          padding: 12px 24px;
          border-radius: 8px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
        }

        .dismiss-disclaimer-btn:hover {
          background: #5a67d8;
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(90, 103, 216, 0.3);
        }
      `}</style>
    </div>
  );
}

export default ChatUI;
