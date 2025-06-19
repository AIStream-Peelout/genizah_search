import React, { useState, useEffect } from 'react';
import './react_app.css';
import StatsCard from './core_results/StatsCard';
import SearchFilters from './core_results/SearchFilters';
import SearchResults from './core_results/SearchResults';
import DocumentModal from './core_results/DocumentModel';
import ErrorMessage from './core_results/ErrorMessage';

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

    try {
      const requestBody = {
        query: query.trim(),
        filters: Object.fromEntries(
            Object.entries(filters).filter(([_, value]) => value !== null && value !== '')
        ),
        num_results: 10
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

          <div className="filter-actions">
            <button onClick={clearFilters} className="clear-filters-btn">
              Clear All Filters
            </button>
            <span className="active-filters">
            {activeFiltersCount} filter{activeFiltersCount !== 1 ? 's' : ''} active
          </span>
          </div>

          <SearchResults
              results={results}
              loading={loading}
              query={query}
              processingTime={results?.processing_time_ms}
              onDocumentClick={handleDocumentClick}
          />

          <DocumentModal
              document={selectedDocument}
              isOpen={isModalOpen}
              onClose={() => setIsModalOpen(false)}
          />
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
      </div>
  );
}

export default App;