import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './react_app.css';

const FAQ = () => {
  const navigate = useNavigate();
  const [openIndex, setOpenIndex] = useState(null);

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
      question: "What types of documents can I search?",
      answer: "You can search through various types of documents including legal documents, liturgical texts, literary works, commercial records, and personal correspondence. The collection includes documents in Hebrew, Arabic, Aramaic, and Judeo-Arabic."
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
      answer: "Currently, all the code is open-sourced except for some of pipelines/data-engineering code. You can find the UI on GitHub here and the indexing code here"
    },

    {
      question: "How accurate are the search results?",
      answer: "Search results are ranked by similarity scores, with higher scores indicating closer matches to your query. The semantic search is powered by state-of-the-art embedding models trained on historical documents, providing high-quality relevance ranking."
    },
    {
      question: "What sources are used for the bibliography?",
      answer: "The bibliography feature draws from scholarly sources and references related to the Cairo Genizah documents. When you search, the system can identify relevant secondary sources and primary documents mentioned in those sources."
    },
    {
      question: "Why is the search and AI Assisant so slow?",
      answer: "Currently, we are hosting everything on a single Mac Studio. Once we obtain more funding we will scale up to a cloud-based solution. All of our code is Dockerized so we can scale easily we just would need funding."
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
              <div key={index} className="faq-item">
                <button
                  className={`faq-question ${openIndex === index ? 'open' : ''}`}
                  onClick={() => toggleQuestion(index)}
                >
                  <span className="faq-question-text">{item.question}</span>
                  <span className="faq-icon">{openIndex === index ? '−' : '+'}</span>
                </button>
                {openIndex === index && (
                  <div className="faq-answer">
                    <p>{item.answer}</p>
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

