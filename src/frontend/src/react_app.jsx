// Updated App.js - Main application with t-SNE visualization integration
import React, { useState, useEffect } from 'react';
import './react_app.css';
import StatsCard from './core_results/StatsCard';
import SearchFilters from './core_results/SearchFilters';
import SearchResults from './core_results/SearchResults';
import DocumentModal from './core_results/DocumentModel';
import ErrorMessage from './core_results/ErrorMessage';
import TSNEVisualization from './TSNEVisualization'; // Import our new component

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [query, setQuery] = useState('');
  const [filters, setFilters] = useState({});
  const [filterOptions, setFilterOptions] = useState({
    languages: ['Hebrew', 'Arabic', 'Aramaic', 'Judeo-Arabic'],
    document_types: ['legal', 'liturgical', 'literary', 'commercial', 'personal'],
    institutions: ['cambridge', 'jewish_theological_seminary', 'princeton'],
    collections: ['taylor_schechter', 'adler', 'gottheil_worrell']
  });
  const [results, setResults] = useState(null);
  const [page, setPage] = useState(1);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({
    global_queries_today: 1247,
    global_limit: 10000,
    your_queries_hour: 8,
    hourly_limit: 100,
    your_queries_today: 23,
    daily_limit: 500,
    estimated_cost_today: 0.0456,
    budget_cap: 10.00,
    remaining_queries_today: 477
  });
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  
  // New state for visualization settings
  const [showVisualization, setShowVisualization] = useState(true);
  const [visualizationMethod, setVisualizationMethod] = useState('tsne');
  const [includeEmbeddings, setIncludeEmbeddings] = useState(true);

  useEffect(() => {
    fetchStats();
    fetchFilterOptions();

    const interval = setInterval(fetchStats, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/stats`);
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (err) {
      console.error('Failed to fetch stats:', err);
    }
  };

  const fetchFilterOptions = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/filters`);
      if (response.ok) {
        const data = await response.json();
        setFilterOptions(data);
      }
    } catch (err) {
      console.error('Failed to fetch filter options:', err);
    }
  };

  const handleSearch = async (e) => {
    if (e) e.preventDefault();

    if (!query.trim()) {
      setError({ message: 'Please enter a search query', type: 'validation' });
      return;
    }

    setLoading(true);
    setError(null);
    setPage(1);

    try {
      const requestBody = {
        query: query.trim(),
        filters: Object.fromEntries(
            Object.entries(filters).filter(([_, value]) => value !== null && value !== '')
        ),
        num_results: 10,
        page: 1,
        // Include embeddings if visualization is enabled
        include_embeddings: showVisualization && includeEmbeddings
      };

      const response = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      const data = await response.json();

      if (response.ok) {
        setResults(data);
        setPage(1);
        fetchStats();
      } else {
        setError({
          message: data.detail || 'Search failed',
          type: response.status === 429 ? 'rate_limit' : 'api'
        });
      }
    } catch (err) {
      setError({
        message: 'Network error. Please check your connection and try again.',
        type: 'network'
      });
    } finally {
      setLoading(false);
    }
  };

  const loadMore = async () => {
    if (!results || !results.has_more) return;
    const nextPage = (page || 1) + 1;
    setIsLoadingMore(true);
    setError(null);
    try {
      const requestBody = {
        query: query.trim(),
        filters: Object.fromEntries(
            Object.entries(filters).filter(([_, value]) => value !== null && value !== '')
        ),
        num_results: results.page_size || 10,
        page: nextPage,
        include_embeddings: showVisualization && includeEmbeddings
      };

      const response = await fetch(`${API_BASE_URL}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      const data = await response.json();
      if (response.ok) {
        setResults(prev => ({
          ...data,
          results: [...(prev?.results || []), ...(data?.results || [])]
        }));
        setPage(nextPage);
      } else {
        setError({
          message: data.detail || 'Failed to load more results',
          type: response.status === 429 ? 'rate_limit' : 'api'
        });
      }
    } catch (err) {
      setError({
        message: 'Network error while loading more results',
        type: 'network'
      });
    } finally {
      setIsLoadingMore(false);
    }
  };

  const handleFilterChange = (filterKey, value) => {
    setFilters(prev => ({
      ...prev,
      [filterKey]: value
    }));
  };

  const clearFilters = () => {
    setFilters({});
  };

  const handleDocumentClick = (document) => {
    setSelectedDocument(document);
    setIsModalOpen(true);
  };

  const activeFiltersCount = Object.keys(filters).filter(key => filters[key]).length;

  return (
      <div className="App">
        <header className="app-header">
          <h1>Cairo Genizah Search</h1>
          <p>AI-powered semantic search through historical manuscripts from the Cairo Genizah collection</p>
        </header>

        {stats && <StatsCard stats={stats} />}

        {error && (
            <ErrorMessage
                error={error}
                onDismiss={() => setError(null)}
            />
        )}

        <main className="main-content">
          <div className="search-form">
            <div className="search-input-group">
              <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSearch(e)}
                  placeholder="Search for Hebrew manuscripts, marriage contracts, religious texts, responsa..."
                  className="search-input"
                  disabled={loading}
              />
              <button
                  onClick={handleSearch}
                  disabled={loading || !query.trim()}
                  className="search-button"
              >
                {loading ? 'Searching...' : 'Search'}
              </button>
            </div>
          </div>

          <SearchFilters
              filters={filters}
              filterOptions={filterOptions}
              onFilterChange={handleFilterChange}
          />

          <div className="search-options">
            <div className="filter-actions">
              <button onClick={clearFilters} className="clear-filters-btn">
                Clear All Filters
              </button>
              <span className="active-filters">
                {activeFiltersCount} filter{activeFiltersCount !== 1 ? 's' : ''} active
              </span>
            </div>
            
            {/* Visualization Controls */}
            <div className="visualization-controls">
              <label className="visualization-toggle">
                <input
                  type="checkbox"
                  checked={showVisualization}
                  onChange={(e) => setShowVisualization(e.target.checked)}
                />
                <span className="checkmark"></span>
                Show Embedding Visualization
              </label>
              
              {showVisualization && (
                <div className="visualization-options">
                  <select 
                    value={visualizationMethod} 
                    onChange={(e) => setVisualizationMethod(e.target.value)}
                    className="method-select"
                  >
                    <option value="pca">PCA (Fast)</option>
                    <option value="tsne">t-SNE (Detailed)</option>
                  </select>
                  
                  <label className="embeddings-toggle">
                    <input
                      type="checkbox"
                      checked={includeEmbeddings}
                      onChange={(e) => setIncludeEmbeddings(e.target.checked)}
                    />
                    <span className="checkmark small"></span>
                    Include embeddings in search
                  </label>
                </div>
              )}
            </div>
          </div>


          <SearchResults
              results={results}
              loading={loading}
              query={query}
              processingTime={results?.processing_time_ms}
              onDocumentClick={handleDocumentClick}
              onLoadMore={loadMore}
              isLoadingMore={isLoadingMore}
          />

          <DocumentModal
              document={selectedDocument}
              isOpen={isModalOpen}
              onClose={() => setIsModalOpen(false)}
          />

          {/* t-SNE Visualization */}
          {showVisualization && results && results.embedding_data && (
            <TSNEVisualization
              results={results}
              query={query}
              embeddingData={results.embedding_data}
              method={visualizationMethod}
              className="search-visualization"
            />
          )}
        </main>

        <footer className="app-footer">
          <div className="footer-content">
            <p>
              Cairo Genizah Search Demo • Powered by AI and historical scholarship
            </p>
            <div className="footer-links">
              <a href="/docs" target="_blank" rel="noopener noreferrer">API Documentation</a>
              <a href="https://github.com/your-repo" target="_blank" rel="noopener noreferrer">GitHub</a>
              <a href="mailto:contact@example.com">Contact</a>
            </div>
          </div>
        </footer>

        {/* Additional CSS for new components */}
        <style jsx>{`
          .search-options {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin: 20px 0;
            padding: 16px;
            background: #F8F9FA;
            border-radius: 8px;
            border: 1px solid #E9ECEF;
            flex-wrap: wrap;
            gap: 20px;
          }

          .filter-actions {
            display: flex;
            align-items: center;
            gap: 16px;
          }

          .clear-filters-btn {
            padding: 8px 16px;
            background: #6C757D;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.2s;
          }

          .clear-filters-btn:hover {
            background: #5A6268;
          }

          .active-filters {
            font-size: 14px;
            color: #6C757D;
          }

          .visualization-controls {
            display: flex;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
          }

          .visualization-toggle {
            display: flex;
            align-items: center;
            gap: 8px;
            cursor: pointer;
            font-size: 14px;
            color: #495057;
            font-weight: 500;
            position: relative;
          }

          .visualization-toggle input[type="checkbox"] {
            display: none;
          }

          .checkmark {
            width: 18px;
            height: 18px;
            background-color: #fff;
            border: 2px solid #DEE2E6;
            border-radius: 3px;
            position: relative;
            transition: all 0.2s;
          }

          .checkmark.small {
            width: 14px;
            height: 14px;
          }

          .visualization-toggle input[type="checkbox"]:checked + .checkmark {
            background-color: #007BFF;
            border-color: #007BFF;
          }

          .visualization-toggle input[type="checkbox"]:checked + .checkmark::after {
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

          .checkmark.small::after {
            left: 3px !important;
            top: 1px !important;
            width: 3px !important;
            height: 6px !important;
          }

          .visualization-options {
            display: flex;
            align-items: center;
            gap: 12px;
            padding-left: 12px;
            border-left: 2px solid #DEE2E6;
          }

          .method-select {
            padding: 6px 12px;
            border: 1px solid #CED4DA;
            border-radius: 4px;
            background: white;
            font-size: 13px;
            color: #495057;
            cursor: pointer;
          }

          .method-select:focus {
            outline: none;
            border-color: #007BFF;
            box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
          }

          .embeddings-toggle {
            display: flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            font-size: 13px;
            color: #6C757D;
          }

          .search-visualization {
            margin: 24px 0;
          }

          @media (max-width: 768px) {
            .search-options {
              flex-direction: column;
              align-items: stretch;
            }

            .visualization-controls {
              justify-content: flex-start;
            }

            .visualization-options {
              padding-left: 0;
              border-left: none;
              border-top: 1px solid #DEE2E6;
              padding-top: 12px;
              margin-top: 12px;
            }
          }
        `}</style>
      </div>
  );
}

export default App;