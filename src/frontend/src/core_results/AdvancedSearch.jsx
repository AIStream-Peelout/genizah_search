import React, { useState } from 'react';

const AdvancedSearch = ({ onSearch, loading }) => {
  const [searchMode, setSearchMode] = useState('semantic'); // 'semantic' or 'shelfmark'
  const [shelfMarkQuery, setShelfMarkQuery] = useState('');
  const [semanticQuery, setSemanticQuery] = useState('');
  const [exactMatch, setExactMatch] = useState(false);

  const handleShelfMarkSearch = (e) => {
    e.preventDefault();
    if (!shelfMarkQuery.trim()) return;
    
    onSearch({
      mode: 'shelfmark',
      query: shelfMarkQuery.trim(),
      exactMatch: exactMatch
    });
  };

  const handleSemanticSearch = (e) => {
    e.preventDefault();
    if (!semanticQuery.trim()) return;
    
    onSearch({
      mode: 'semantic',
      query: semanticQuery.trim()
    });
  };

  const handleModeChange = (mode) => {
    setSearchMode(mode);
    // Clear the other query when switching modes
    if (mode === 'shelfmark') {
      setSemanticQuery('');
    } else {
      setShelfMarkQuery('');
    }
  };

  return (
    <div className="advanced-search">
      <div className="search-mode-selector">
        <h3>Advanced Search</h3>
        <div className="mode-tabs">
          <button
            className={`mode-tab ${searchMode === 'shelfmark' ? 'active' : ''}`}
            onClick={() => handleModeChange('shelfmark')}
          >
            📚 Shelf Mark Search
          </button>
          <button
            className={`mode-tab ${searchMode === 'semantic' ? 'active' : ''}`}
            onClick={() => handleModeChange('semantic')}
          >
            🔍 Semantic Search
          </button>
        </div>
      </div>

      {searchMode === 'shelfmark' && (
        <div className="shelfmark-search">
          <div className="search-description">
            <p>
              <strong>Find documents by their exact shelf mark or catalog number.</strong>
              <br />
              Examples: T-S 8J5.1, MS-TS-NS-144.1, Cambridge Or.1080 J2
            </p>
          </div>
          
          <form onSubmit={handleShelfMarkSearch} className="shelfmark-form">
            <div className="input-group">
              <input
                type="text"
                value={shelfMarkQuery}
                onChange={(e) => setShelfMarkQuery(e.target.value)}
                placeholder="Enter shelf mark (e.g., T-S 8J5.1)"
                className="shelfmark-input"
                disabled={loading}
              />
              <button
                type="submit"
                disabled={loading || !shelfMarkQuery.trim()}
                className="search-button primary"
              >
                {loading ? 'Searching...' : 'Find Document'}
              </button>
            </div>
            
            <div className="search-options">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={exactMatch}
                  onChange={(e) => setExactMatch(e.target.checked)}
                />
                <span className="checkmark"></span>
                Exact match only
              </label>
              <div className="help-text">
                {exactMatch 
                  ? "Only documents with exactly this shelf mark will be returned"
                  : "Documents containing this shelf mark (partial matches) will be returned"
                }
              </div>
            </div>
          </form>
        </div>
      )}

      {searchMode === 'semantic' && (
        <div className="semantic-search">
          <div className="search-description">
            <p>
              <strong>Search by meaning and content.</strong>
              <br />
              Find documents based on their content, themes, or concepts using AI-powered semantic search.
            </p>
          </div>
          
          <form onSubmit={handleSemanticSearch} className="semantic-form">
            <div className="input-group">
              <input
                type="text"
                value={semanticQuery}
                onChange={(e) => setSemanticQuery(e.target.value)}
                placeholder="Search for Hebrew manuscripts, marriage contracts, religious texts, responsa..."
                className="semantic-input"
                disabled={loading}
              />
              <button
                type="submit"
                disabled={loading || !semanticQuery.trim()}
                className="search-button primary"
              >
                {loading ? 'Searching...' : 'Search'}
              </button>
            </div>
          </form>
        </div>
      )}

      <style jsx>{`
        .advanced-search {
          background: #fff;
          border-radius: 12px;
          padding: 24px;
          margin: 20px 0;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
          border: 1px solid #e1e5e9;
        }

        .search-mode-selector h3 {
          margin: 0 0 16px 0;
          color: #2c3e50;
          font-size: 1.5rem;
          font-weight: 600;
        }

        .mode-tabs {
          display: flex;
          gap: 8px;
          margin-bottom: 24px;
          border-bottom: 2px solid #f1f3f4;
        }

        .mode-tab {
          padding: 12px 20px;
          border: none;
          background: transparent;
          color: #6c757d;
          font-size: 14px;
          font-weight: 500;
          cursor: pointer;
          border-radius: 8px 8px 0 0;
          transition: all 0.2s ease;
          position: relative;
        }

        .mode-tab:hover {
          background: #f8f9fa;
          color: #495057;
        }

        .mode-tab.active {
          background: #007bff;
          color: white;
          box-shadow: 0 2px 4px rgba(0, 123, 255, 0.3);
        }

        .search-description {
          background: #f8f9fa;
          padding: 16px;
          border-radius: 8px;
          margin-bottom: 20px;
          border-left: 4px solid #007bff;
        }

        .search-description p {
          margin: 0;
          color: #495057;
          line-height: 1.5;
        }

        .input-group {
          display: flex;
          gap: 12px;
          margin-bottom: 16px;
        }

        .shelfmark-input,
        .semantic-input {
          flex: 1;
          padding: 12px 16px;
          border: 2px solid #e1e5e9;
          border-radius: 8px;
          font-size: 16px;
          transition: border-color 0.2s ease;
        }

        .shelfmark-input:focus,
        .semantic-input:focus {
          outline: none;
          border-color: #007bff;
          box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1);
        }

        .search-button {
          padding: 12px 24px;
          border: none;
          border-radius: 8px;
          font-size: 16px;
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s ease;
          white-space: nowrap;
        }

        .search-button.primary {
          background: #007bff;
          color: white;
        }

        .search-button.primary:hover:not(:disabled) {
          background: #0056b3;
          transform: translateY(-1px);
        }

        .search-button:disabled {
          background: #6c757d;
          cursor: not-allowed;
          transform: none;
        }

        .search-options {
          display: flex;
          align-items: center;
          gap: 12px;
          flex-wrap: wrap;
        }

        .checkbox-label {
          display: flex;
          align-items: center;
          gap: 8px;
          cursor: pointer;
          font-size: 14px;
          color: #495057;
          font-weight: 500;
        }

        .checkbox-label input[type="checkbox"] {
          display: none;
        }

        .checkmark {
          width: 18px;
          height: 18px;
          background-color: #fff;
          border: 2px solid #dee2e6;
          border-radius: 3px;
          position: relative;
          transition: all 0.2s;
        }

        .checkbox-label input[type="checkbox"]:checked + .checkmark {
          background-color: #007bff;
          border-color: #007bff;
        }

        .checkbox-label input[type="checkbox"]:checked + .checkmark::after {
          content: '';
          position: absolute;
          left: 5px;
          top: 2px;
          width: 4px;
          height: 8px;
          border: solid white;
          border-width: 0 2px 2px 0;
          transform: rotate(45deg);
        }

        .help-text {
          font-size: 12px;
          color: #6c757d;
          font-style: italic;
          max-width: 300px;
        }

        @media (max-width: 768px) {
          .advanced-search {
            padding: 16px;
            margin: 16px 0;
          }

          .input-group {
            flex-direction: column;
          }

          .search-button {
            width: 100%;
          }

          .mode-tabs {
            flex-direction: column;
            gap: 4px;
          }

          .mode-tab {
            border-radius: 8px;
            margin-bottom: 4px;
          }
        }
      `}</style>
    </div>
  );
};

export default AdvancedSearch;
