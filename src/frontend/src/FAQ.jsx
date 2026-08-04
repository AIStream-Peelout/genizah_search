import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './react_app.css';

// Render answer text supporting [label](url) links and **bold**. Prefer
// labelled links in answers: a raw URL in running prose is hard to read and
// looks unpolished. A bare URL still becomes a link as a fallback.
const renderAnswerText = (text) => {
  const tokens = String(text).split(
    /(\[[^\]]+\]\([^)]+\)|https?:\/\/[^\s<>()]+[^\s<>().,;:!?]|\*\*[^*]+\*\*)/g
  );
  return tokens.map((token, index) => {
    const markdownLink = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(token);
    if (markdownLink) {
      const [, label, url] = markdownLink;
      return (
        <a
          key={index}
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: '#667eea' }}
        >
          {label}
        </a>
      );
    }
    if (/^https?:\/\//.test(token)) {
      return (
        <a
          key={index}
          href={token}
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: '#667eea', wordBreak: 'break-word' }}
        >
          {token.replace(/^https?:\/\//, '')}
        </a>
      );
    }
    if (/^\*\*[^*]+\*\*$/.test(token)) {
      return <strong key={index}>{token.slice(2, -2)}</strong>;
    }
    return token;
  });
};

const FAQ = () => {
  const navigate = useNavigate();
  const [openIndex, setOpenIndex] = useState(null);
  const itemRefs = useRef({});

  // Support deep links such as /faq#hardware, used by the chat assistant when
  // it has to report that the local model is unavailable.
  useEffect(() => {
    const anchor = window.location.hash.replace('#', '');
    if (!anchor) return;
    const index = faqData.findIndex(item => item.id === anchor);
    if (index === -1) return;
    setOpenIndex(index);
    setTimeout(() => {
      itemRefs.current[index]?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 100);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // FAQ data - user will edit these manually
  const faqData = [
    {
      question: "What is the Cairo Genizah?",
      answer: "The Cairo Genizah is a collection of over 400,000 Jewish manuscript fragments found in the Ben Ezra Synagogue in Fustat, Egypt. These documents date from the 9th to 19th centuries and provide invaluable insights into medieval Jewish life, commerce, law, and culture."
    },
    {
      question: "How does semantic search work?",
      answer: "Semantic search uses AI-powered embeddings to understand the meaning and context of your query, not just keyword matching. This allows you to find documents that are conceptually related to your search, even if they don't contain the exact words you used."
    },
    {
      question: "What types of documents can I search and how many shelf-marks have you indexed?",
      answer: "You can search through various types of documents including legal documents, liturgical texts, literary works, commercial records, and personal correspondence. The collection includes documents in Hebrew, Arabic, Aramaic, and Judeo-Arabic. Currently, we have indexed about 49,000 of the 200k+ Genizah documents in our system. We are working on indexing the rest of the documents."
    },
    {
      question: "How do I use the advanced search features?",
      answer: "The advanced search allows you to combine semantic search with keyword matching, search by shelfmark, and apply filters for language, document type, institution, and collection. You can also adjust the balance between semantic and keyword search using the hybrid search option."
    },
    {
      question: "What is the visualization feature?",
      answer: "The visualization feature uses dimensionality reduction techniques (PCA, UMap or t-SNE) to create a 2D representation of document embeddings. This helps you see relationships between documents and understand how your search results are distributed in the semantic space."
    },
    {
      question: "Can I download or export search results?",
      answer: "Currently, you can view document details and metadata through the interface. Export functionality may be available in future updates. For now, you can use the browser's print or save functionality to preserve search results."
    },
    {
      question: "Will you open source the data?",
      answer: "We hope to open source the Cairo Genizah data soon, however we have to be careful about licensing issues. Once we do open source it will likely be posted to HuggingFace."
    },

    {
      question: "Will you open source the code?",
      answer: "Currently, all the code is open-sourced except for some of the pipelines/data-engineering code. The search and chat interface is on GitHub [here](https://github.com/AIStream-Peelout/genizah_search), and the indexing and knowledge-graph code is [here](https://github.com/AIStream-Peelout/historical-document-analysis)."
    },

    {
      question: "How accurate are the search results?",
      answer: "Search results are ranked by similarity scores, with higher scores indicating closer matches to your query. The semantic search is powered by state-of-the-art embedding models trained on historical documents, providing high-quality relevance ranking."
    },
    {
      question: "What sources are used for the bibliography?",
      answer: "The bibliography feature draws from scholarly sources and references related to the Cairo Genizah documents. When you search, the system can identify relevant secondary sources and primary documents mentioned in those sources. We have currently indexed about 1,000 pages of scholary literature and working on adding more sources."
    },
    {
      id: "hardware",
      question: "Why is the search and AI Assistant so slow — and why does it sometimes fail entirely?",
      answer: "Everything you see here runs on a single Mac Studio sitting on a desk: the website, the search indexes, the knowledge graph, and the language models that write the AI Assistant's answers. That one machine is also used for the project's research work, including training models on Genizah handwriting. When a training job is running, it competes with the website for memory and compute.\n\nThat is the honest explanation behind almost every rough edge you may hit: answers that take a few minutes, requests that queue behind one another, and occasional periods where the Assistant reports it is unavailable while search and browsing keep working. The software is designed to degrade gracefully — it queues requests, tells you where you are in line, and says plainly when a model cannot be reached rather than failing silently — but no amount of software can make one shared workstation behave like dedicated hardware.\n\nNone of these would be issues with funding. A dedicated inference workstation (or modest cloud GPU budget) would separate serving from research, cut answer times from minutes to seconds, allow several people to use the Assistant at once, and remove the outages entirely. Everything is already containerized, so scaling is a question of budget rather than engineering. This is an independent project without institutional compute funding; if you or your institution would like to support it, that is where support would go first."
    },
    {
      question: "The AI Assistant said something incorrect...",
      answer: "The AI Assistant right now is more a proof of concept. We have not conducted extensive tests on its accuracy. Therefore all answers should be double checked"
    }

  ];

  const toggleQuestion = (index) => {
    setOpenIndex(openIndex === index ? null : index);
  };

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <div className="header-left">
            <h1>Cairo Genizah Search</h1>
            <p>Frequently Asked Questions</p>
          </div>
          <div className="header-right">
            <button
              onClick={() => navigate('/')}
              className="browser-btn"
              style={{ marginRight: 0 }}
            >
              ← Back to Search
            </button>
          </div>
        </div>
      </header>

      <main className="main-content" style={{ maxWidth: '900px', margin: '2rem auto' }}>
        <div className="faq-container">
          <div className="faq-intro">
            <h2>Frequently Asked Questions</h2>
            <p>
              Find answers to common questions about using the Cairo Genizah Search platform.
              If you have additional questions, please contact us through the contact information
              provided in the footer.
            </p>
          </div>

          <div className="faq-list">
            {faqData.map((item, index) => (
              <div
                key={index}
                id={item.id}
                ref={el => { itemRefs.current[index] = el; }}
                className="faq-item"
              >
                <button
                  className={`faq-question ${openIndex === index ? 'open' : ''}`}
                  onClick={() => toggleQuestion(index)}
                >
                  <span className="faq-question-text">{item.question}</span>
                  <span className="faq-icon">{openIndex === index ? '−' : '+'}</span>
                </button>
                {openIndex === index && (
                  <div className="faq-answer">
                    {String(item.answer).split('\n\n').map((paragraph, i) => (
                      <p key={i}>{renderAnswerText(paragraph)}</p>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <div className="footer-content">
          <p>
            Cairo Genizah Search Demo • Powered by AI and historical scholarship
          </p>
          <p>
            Special thanks to the <a href="https://geniza.princeton.edu/en/"> Princeton Cairo Genizah Project</a> (PGP)
          </p>
          <div className="footer-links">
            <a href="/docs" target="_blank" rel="noopener noreferrer">API Documentation</a>
            <a href="https://github.com/your-repo" target="_blank" rel="noopener noreferrer">GitHub</a>
            <a href="mailto:contact@example.com">Contact</a>
          </div>
        </div>
      </footer>

      <style jsx>{`
        .faq-container {
          background: white;
          border-radius: 12px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
          padding: 2rem;
          margin: 2rem 0;
        }

        .faq-intro {
          margin-bottom: 2rem;
          padding-bottom: 1.5rem;
          border-bottom: 2px solid #e9ecef;
        }

        .faq-intro h2 {
          font-size: 2rem;
          color: #333;
          margin: 0 0 1rem 0;
        }

        .faq-intro p {
          font-size: 1rem;
          color: #666;
          line-height: 1.6;
          margin: 0;
        }

        .faq-list {
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .faq-item {
          border: 1px solid #e9ecef;
          border-radius: 8px;
          overflow: hidden;
          transition: box-shadow 0.2s ease;
        }

        .faq-item:hover {
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .faq-question {
          width: 100%;
          padding: 1.25rem 1.5rem;
          background: white;
          border: none;
          text-align: left;
          cursor: pointer;
          display: flex;
          justify-content: space-between;
          align-items: center;
          font-size: 1.1rem;
          font-weight: 500;
          color: #333;
          transition: background-color 0.2s ease;
        }

        .faq-question:hover {
          background-color: #f8f9fa;
        }

        .faq-question.open {
          background-color: #f8f9fa;
          border-bottom: 1px solid #e9ecef;
        }

        .faq-question-text {
          flex: 1;
          padding-right: 1rem;
        }

        .faq-icon {
          font-size: 1.5rem;
          font-weight: 300;
          color: #667eea;
          min-width: 24px;
          text-align: center;
        }

        .faq-answer {
          padding: 1.5rem;
          background-color: #f8f9fa;
          border-top: 1px solid #e9ecef;
          animation: slideDown 0.3s ease;
        }

        .faq-answer p {
          margin: 0;
          line-height: 1.8;
          color: #555;
          font-size: 1rem;
        }

        @keyframes slideDown {
          from {
            opacity: 0;
            max-height: 0;
          }
          to {
            opacity: 1;
            max-height: 500px;
          }
        }

        @media (max-width: 768px) {
          .faq-container {
            padding: 1.5rem;
            margin: 1rem;
          }

          .faq-intro h2 {
            font-size: 1.5rem;
          }

          .faq-question {
            padding: 1rem;
            font-size: 1rem;
          }

          .faq-answer {
            padding: 1rem;
          }
        }
      `}</style>
    </div>
  );
};

export default FAQ;


