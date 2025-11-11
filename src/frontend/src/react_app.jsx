// Updated App.js - Main application with routing and visualization explorer
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import './react_app.css';
import SearchFilters from './core_results/SearchFilters';
import SearchResults from './core_results/SearchResults';
import DocumentModal from './core_results/DocumentModel';
import ErrorMessage from './core_results/ErrorMessage';
import AdvancedSearch from './core_results/AdvancedSearch';
import TSNEVisualization from './TSNEVisualization';
import VisualizationExplorer from './VisualizationExplorer';
import CollectionBrowser from './CollectionBrowser';
import ChatUI from './ChatUI';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Search Page Component
function SearchPage() {
  const navigate = useNavigate();
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
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  
  // New state for visualization settings
  const [showVisualization, setShowVisualization] = useState(true);
  const [visualizationMethod, setVisualizationMethod] = useState('tsne');
  const [includeEmbeddings, setIncludeEmbeddings] = useState(true);
  
  // Advanced search state
  const [showAdvancedSearch, setShowAdvancedSearch] = useState(false);
  
  // Collection browser state
  const [showCollectionBrowser, setShowCollectionBrowser] = useState(false);
  const [browsedDocuments, setBrowsedDocuments] = useState(null);
  const [currentSearchMode, setCurrentSearchMode] = useState('semantic'); // Track current search mode
  const [currentSearchParams, setCurrentSearchParams] = useState(null); // Store search parameters for pagination
  const [selectedIndex, setSelectedIndex] = useState(''); // Track selected index
  const [availableIndices, setAvailableIndices] = useState([]); // Available indices

  useEffect(() => {
    fetchFilterOptions();
    loadIndices();
  }, []);

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

  const loadIndices = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/indices`);
      if (response.ok) {
        const data = await response.json();
        setAvailableIndices(data.indices || []);
        // Set default index if available
        if (data.indices && data.indices.length > 0) {
          const defaultIndex = data.indices.find(idx => idx.is_default);
          if (defaultIndex) {
            setSelectedIndex(defaultIndex.name);
          } else {
            setSelectedIndex(data.indices[0].name);
          }
        }
      }
    } catch (error) {
      console.error('Failed to load indices:', error);
    }
  };

  const handleAdvancedSearch = async (searchParams) => {
    setLoading(true);
    setError(null);
    setPage(1);
    
    // Store search mode and parameters for pagination
    setCurrentSearchMode(searchParams.mode);
    setCurrentSearchParams(searchParams);

    try {
      let response;
      
      if (searchParams.mode === 'shelfmark') {
        // Shelf mark search
        const requestBody = {
          shelf_mark: searchParams.query,
          exact_match: searchParams.exactMatch,
          num_results: 10,
          index_name: searchParams.indexName
        };

        response = await fetch(`${API_BASE_URL}/search-shelfmark`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
        });
      } else if (searchParams.mode === 'keyword') {
        // Keyword search
        const requestBody = {
          query: searchParams.query,
          num_results: 10,
          page: 1,
          index_name: searchParams.indexName
        };

        response = await fetch(`${API_BASE_URL}/search-keyword`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
        });
      } else if (searchParams.mode === 'hybrid') {
        // Hybrid search
        const requestBody = {
          query: searchParams.query,
          semanticWeight: searchParams.semanticWeight,
          keywordWeight: searchParams.keywordWeight,
          filters: Object.fromEntries(
            Object.entries(filters).filter(([_, value]) => value !== null && value !== '')
          ),
          num_results: 10,
          page: 1,
          include_embeddings: showVisualization && includeEmbeddings,
          index_name: searchParams.indexName
        };

        response = await fetch(`${API_BASE_URL}/search-hybrid`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
        });
      } else {
        // Semantic search (existing functionality)
        const requestBody = {
          query: searchParams.query,
          filters: Object.fromEntries(
              Object.entries(filters).filter(([_, value]) => value !== null && value !== '')
          ),
          num_results: 10,
          page: 1,
          include_embeddings: showVisualization && includeEmbeddings,
          index_name: searchParams.indexName
        };

        response = await fetch(`${API_BASE_URL}/search`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
        });
      }

      const data = await response.json();

      if (response.ok) {
        setResults(data);
        setPage(1);
      } else {
        setError({
          message: data.detail || 'Search failed',
          type: 'api'
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

  const handleSearch = async (e) => {
    if (e) e.preventDefault();

    if (!query.trim()) {
      setError({ message: 'Please enter a search query', type: 'validation' });
      return;
    }

    setLoading(true);
    setError(null);
    setPage(1);
    
    // Store search mode and parameters for pagination
    setCurrentSearchMode('semantic');
    setCurrentSearchParams({
      mode: 'semantic',
      query: query.trim()
    });

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
      } else {
        setError({
          message: data.detail || 'Search failed',
          type: 'api'
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
    if (!results || !results.has_more || !currentSearchParams) return;
    const nextPage = (page || 1) + 1;
    setIsLoadingMore(true);
    setError(null);
    
    try {
      let response;
      let requestBody;
      
      // Determine which endpoint to use based on current search mode
      if (currentSearchMode === 'shelfmark') {
        // Shelf mark search doesn't support pagination, so we'll skip load more
        setIsLoadingMore(false);
        return;
      } else if (currentSearchMode === 'keyword') {
        requestBody = {
          query: currentSearchParams.query,
          num_results: results.page_size || 10,
          page: nextPage,
          index_name: currentSearchParams.indexName
        };
        
        response = await fetch(`${API_BASE_URL}/search-keyword`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
        });
      } else if (currentSearchMode === 'hybrid') {
        requestBody = {
          query: currentSearchParams.query,
          semanticWeight: currentSearchParams.semanticWeight,
          keywordWeight: currentSearchParams.keywordWeight,
          filters: Object.fromEntries(
            Object.entries(filters).filter(([_, value]) => value !== null && value !== '')
          ),
          num_results: results.page_size || 10,
          page: nextPage,
          include_embeddings: showVisualization && includeEmbeddings,
          index_name: currentSearchParams.indexName
        };
        
        response = await fetch(`${API_BASE_URL}/search-hybrid`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
        });
      } else {
        // Default to semantic search
        requestBody = {
          query: currentSearchParams.query,
          filters: Object.fromEntries(
              Object.entries(filters).filter(([_, value]) => value !== null && value !== '')
          ),
          num_results: results.page_size || 10,
          page: nextPage,
          include_embeddings: showVisualization && includeEmbeddings,
          index_name: currentSearchParams.indexName
        };
        
        response = await fetch(`${API_BASE_URL}/search`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
        });
      }

      const data = await response.json();
      if (response.ok) {
        setResults(prev => ({
          ...data,
          results: [...(prev?.results || []), ...(data?.results || [])],
          embedding_data: (prev?.embedding_data && data?.embedding_data)
            ? {
                query_embedding: prev.embedding_data.query_embedding,
                result_embeddings: [
                  ...(prev.embedding_data.result_embeddings || []),
                  ...(data.embedding_data.result_embeddings || [])
                ],
                dimension: prev.embedding_data.dimension || data.embedding_data.dimension
              }
            : (data?.embedding_data || prev?.embedding_data)
        }));
        setPage(nextPage);
      } else {
        setError({
          message: data.detail || 'Failed to load more results',
          type: 'api'
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

  const handleMultipleShelfmarkSearch = async (shelfMarks) => {
    if (!shelfMarks || shelfMarks.length === 0) return;
    
    setLoading(true);
    setError(null);
    setPage(1);
    
    try {
      // Search for all shelf marks and combine results
      const allResults = [];
      const allEmbeddings = [];
      
      for (const shelfMark of shelfMarks) {
        try {
          const effectiveIndex = selectedIndex;
          const idxParam = effectiveIndex ? `&index_name=${encodeURIComponent(effectiveIndex)}` : '';
          const response = await fetch(`${API_BASE_URL}/shelfmark/${encodeURIComponent(shelfMark)}/documents?include_embeddings=true${idxParam}`);
          
          if (response.ok) {
            const data = await response.json();
            if (data.documents && data.documents.length > 0) {
              allResults.push(...data.documents.map(doc => ({
                doc_id: doc.doc_id,
                similarity_score: doc.similarity_score || 1.0,
                metadata: doc.metadata,
                embedding: doc.embedding
              })));
              
              if (data.documents[0].embedding) {
                allEmbeddings.push(...data.documents.map(doc => doc.embedding).filter(Boolean));
              }
            }
          }
        } catch (err) {
          console.error(`Failed to search for shelfmark ${shelfMark}:`, err);
        }
      }
      
      // Create a combined results structure
      if (allResults.length > 0) {
        const combinedResults = {
          results: allResults,
          count: allResults.length,
          query: `Shelfmarks: ${shelfMarks.join(', ')}`,
          processing_time_ms: 0,
          embedding_data: allEmbeddings.length > 0 ? {
            query_embedding: null,
            result_embeddings: allEmbeddings,
            dimension: allEmbeddings[0]?.length || 768
          } : null
        };
        
        setResults(combinedResults);
        setCurrentSearchMode('shelfmark');
        setCurrentSearchParams({
          mode: 'shelfmark',
          query: shelfMarks.join(', ')
        });
      } else {
        setError({
          message: `No documents found for shelfmarks: ${shelfMarks.join(', ')}`,
          type: 'api'
        });
      }
    } catch (err) {
      setError({
        message: 'Network error while loading shelfmark documents',
        type: 'network'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleShelfmarkSelect = async (shelfmark, docIds = [], indexOverride = null) => {
    try {
      setLoading(true);
      
      // If we were provided docIds for this shelfmark, open the first doc immediately
      if (docIds && docIds.length > 0) {
        try {
          const effectiveIndex = indexOverride || selectedIndex;
          const idxParamDoc = effectiveIndex ? `?index_name=${encodeURIComponent(effectiveIndex)}` : '';
          const docResp = await fetch(`${API_BASE_URL}/document/${encodeURIComponent(docIds[0])}${idxParamDoc}`);
          if (docResp.ok) {
            const docMeta = await docResp.json();
            const m = docMeta || {};
            const displayData = {
              title: m.title || `Document ${docIds[0]}`,
              description: m.description || "Historical manuscript from the Cairo Genizah collection.",
              image_url: (m.actual_image_url || (m.image_urls && m.image_urls[0]) || m.image_url || m.thumbnail_url || "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=400&h=300&fit=crop"),
              language: m.language || m.main_language,
              material: m.material,
              institution: m.institution,
              collection: m.collection,
              shelfmark: m.shelf_mark || m.shelfmark || m.classmark,
              transcription: m.transcription_full_text,
              translation: m.translation_full_text,
              tags: m.tags,
              period: m.period,
              location: m.location,
              dimensions: m.dimensions,
              document_type: m.document_type,
              doc_id: docIds[0]
            };
            setSelectedDocument(displayData);
            setIsModalOpen(true);
          }
        } catch {}
      }

      // Fetch documents for this shelfmark with embeddings
      const effectiveIndexForShelf = indexOverride || selectedIndex;
      const idxParam = effectiveIndexForShelf ? `&index_name=${encodeURIComponent(effectiveIndexForShelf)}` : '';
      const response = await fetch(`${API_BASE_URL}/shelfmark/${encodeURIComponent(shelfmark)}/documents?include_embeddings=true${idxParam}`);
      if (response.ok) {
        const data = await response.json();
        
        // Create a results-like structure for the visualization
        const browsedResults = {
          results: data.documents.map(doc => ({
            doc_id: doc.doc_id,
            similarity_score: doc.similarity_score,
            metadata: doc.metadata,
            embedding: doc.embedding
          })),
          count: data.count,
          query: `Shelfmark: ${shelfmark}`,
          processing_time_ms: 0,
          embedding_data: data.documents.length > 0 && data.documents[0].embedding ? {
            query_embedding: null,
            result_embeddings: data.documents.map(doc => doc.embedding).filter(Boolean),
            dimension: data.documents[0].embedding.length
          } : null
        };
        
        setBrowsedDocuments(browsedResults);
        setResults(browsedResults);

        // Also open the stats/document card for the first document in this shelfmark
        if (browsedResults.results && browsedResults.results.length > 0) {
          const r = browsedResults.results[0];
          const m = r.metadata || {};
          const displayData = {
            title: m.title || `Document ${r.doc_id}`,
            description: m.description || "Historical manuscript from the Cairo Genizah collection.",
            image_url: (() => {
              if (m.actual_image_url) return m.actual_image_url;
              if (m.image_urls && m.image_urls.length > 0) {
                const valid = m.image_urls.filter(u => u && u.trim());
                if (valid.length > 0) return valid[0];
              }
              if (m.image_url) return m.image_url;
              if (m.thumbnail_url) return m.thumbnail_url;
              return "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=400&h=300&fit=crop";
            })(),
            date: m.date || "Unknown",
            language: m.language || m.main_language || "Hebrew",
            material: m.material || "Parchment",
            institution: m.institution,
            collection: m.collection,
            shelfmark: m.shelf_mark || m.shelfmark || m.classmark,
            transcription: m.transcription_full_text,
            translation: m.translation_full_text,
            tags: m.tags,
            period: m.period,
            location: m.location,
            dimensions: m.dimensions,
            document_type: m.document_type,
            doc_id: r.doc_id,
            similarity_score: r.similarity_score,
            ...r
          };
          setSelectedDocument(displayData);
          setIsModalOpen(true);
        }
      } else {
        setError({
          message: 'Failed to load shelfmark documents',
          type: 'api'
        });
      }
    } catch (err) {
      setError({
        message: 'Network error while loading shelfmark documents',
        type: 'network'
      });
    } finally {
      setLoading(false);
    }
  };

  return (
      <div className="App">
        <header className="app-header">
          <div className="header-content">
            <div className="header-left">
              <h1>Cairo Genizah Search</h1>
              <p>AI-powered semantic search through historical manuscripts from the Cairo Genizah collection</p>
            </div>
            <div className="header-right">
              <button 
                onClick={() => setShowCollectionBrowser(!showCollectionBrowser)} 
                className={`browser-btn ${showCollectionBrowser ? 'active' : ''}`}
              >
                {showCollectionBrowser ? '✕ Close Browser' : '📚 Browse by Collection'}
              </button>
              <button 
                onClick={() => navigate('/explorer')} 
                className="explorer-btn"
              >
                🗺️ Collection Explorer
              </button>
            </div>
          </div>
        </header>

        {error && (
            <ErrorMessage
                error={error}
                onDismiss={() => setError(null)}
            />
        )}

        <div className="main-layout">
          <main className="main-content">
            {/* Collection Browser Toggle */}
            {showCollectionBrowser && (
              <CollectionBrowser
                onSelectShelfmark={handleShelfmarkSelect}
                isVisible={showCollectionBrowser}
              />
            )}

            <div className="search-form">
            <div className="search-toggle">
              <button
                className={`search-mode-btn ${!showAdvancedSearch ? 'active' : ''}`}
                onClick={() => setShowAdvancedSearch(false)}
              >
                🔍 Basic Search
              </button>
              <button
                className={`search-mode-btn ${showAdvancedSearch ? 'active' : ''}`}
                onClick={() => setShowAdvancedSearch(true)}
              >
                ⚡ Advanced Search
              </button>
            </div>

            {!showAdvancedSearch ? (
              <div className="basic-search">
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
            ) : (
              <AdvancedSearch onSearch={handleAdvancedSearch} loading={loading} />
            )}
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
              query={results?.query || query || currentSearchParams?.query || 'Search'}
              processingTime={results?.processing_time_ms}
              onDocumentClick={handleDocumentClick}
              onLoadMore={loadMore}
              isLoadingMore={isLoadingMore}
              currentSearchMode={currentSearchMode}
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
              onDocumentClick={handleDocumentClick}
            />
          )}
          </main>

          {/* Right Sidebar with Chat Assistant */}
          <aside className="chat-sidebar-container">
            <ChatUI 
              onShelfmarkSearch={handleMultipleShelfmarkSearch}
              isSidebar={true}
              examplePrompts={[
                { text: "Can you tell me about Ketubah's in the Cairo Genizah", icon: "💍" },
                { text: "Yom Kippur Piyyut Fragments", icon: "📜" },
                { text: "Who is S.D. Goitein", icon: "👤" }
              ]}
            />
          </aside>
        </div>

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

        {/* Additional CSS for new components */}
        <style jsx>{`
          .search-toggle {
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
            border-bottom: 2px solid #f1f3f4;
          }

          .search-mode-btn {
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

          .search-mode-btn:hover {
            background: #f8f9fa;
            color: #495057;
          }

          .search-mode-btn.active {
            background: #007bff;
            color: white;
            box-shadow: 0 2px 4px rgba(0, 123, 255, 0.3);
          }

          .basic-search {
            margin-top: 16px;
          }

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

          .main-layout {
            display: flex;
            width: 100%;
            max-width: 100%;
            margin: 0 auto;
            min-height: calc(100vh - 200px);
          }

          .main-content {
            flex: 1;
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            width: 100%;
            box-sizing: border-box;
            min-width: 0; /* Allow flexbox to shrink */
          }

          .chat-sidebar-container {
            width: 400px;
            min-width: 400px;
            max-width: 400px;
            background: white;
            border-left: 1px solid #e0e0e0;
            display: flex;
            flex-direction: column;
            height: calc(100vh - 200px);
            position: sticky;
            top: 0;
            overflow: hidden;
          }

          .header-content {
            display: flex;
            justify-content: space-between;
            align-items: center;
            max-width: 100%;
            margin: 0 auto;
            padding: 0 20px;
          }

          .header-left h1 {
            margin: 0 0 8px 0;
            font-size: 32px;
            font-weight: 600;
          }

          .header-left p {
            margin: 0;
            font-size: 16px;
            opacity: 0.9;
          }

          .explorer-btn {
            padding: 12px 24px;
            background: #27AE60;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: background-color 0.2s;
            text-decoration: none;
            display: inline-block;
          }

          .explorer-btn:hover {
            background: #229954;
          }

          .browser-btn {
            padding: 12px 24px;
            background: #9B59B6;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: background-color 0.2s;
            margin-right: 12px;
          }

          .browser-btn:hover {
            background: #8E44AD;
          }

          .browser-btn.active {
            background: #7D3C98;
          }

          .chat-btn {
            padding: 12px 24px;
            background: #E67E22;
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: background-color 0.2s;
            margin-left: 12px;
          }

          .chat-btn:hover {
            background: #D35400;
          }

          @media (max-width: 1200px) {
            .chat-sidebar-container {
              display: none;
            }

            .main-content {
              max-width: 100%;
            }
          }

          @media (max-width: 768px) {
            .main-layout {
              flex-direction: column;
            }

            .chat-sidebar-container {
              width: 100%;
              min-width: 100%;
              max-width: 100%;
              height: 50vh;
              position: relative;
              border-left: none;
              border-top: 1px solid #e0e0e0;
            }

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

// Main App Component with Routing
function App() {
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const handleDocumentClick = (document) => {
    setSelectedDocument(document);
    setIsModalOpen(true);
  };

  const handleBackToSearch = () => {
    // This will be handled by the router
  };

  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<SearchPage />} />
          <Route 
            path="/explorer" 
            element={
              <VisualizationExplorer 
                onDocumentClick={handleDocumentClick}
              />
            } 
          />
          <Route 
            path="/chat" 
            element={<ChatUI />} 
          />
        </Routes>
        
        <DocumentModal
          document={selectedDocument}
          isOpen={isModalOpen}
          onClose={() => setIsModalOpen(false)}
        />
      </div>
    </Router>
  );
}

export default App;