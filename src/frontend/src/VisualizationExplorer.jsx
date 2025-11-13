// VisualizationExplorer.jsx - Full-page visualization explorer for the Cairo Genizah collection
// Refactored to use backend Python libraries (scikit-learn, umap-learn) instead of JavaScript implementations
import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import Plot from 'react-plotly.js';

const VisualizationExplorer = ({ onDocumentClick = null }) => {
  const navigate = useNavigate();
  const [documents, setDocuments] = useState(null);
  const [plotData, setPlotData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isCalculating, setIsCalculating] = useState(false);
  const [error, setError] = useState(null);
  const [method, setMethod] = useState('tsne');
  const [colorBy, setColorBy] = useState('language');
  const [numDocuments, setNumDocuments] = useState(1000);
  const [loadFullIndex, setLoadFullIndex] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [availableIndices, setAvailableIndices] = useState([]);
  const [umapParams, setUmapParams] = useState({
    nNeighbors: 15,
    minDist: 0.1,
    iterations: 300
  });
  const [selectedDocuments, setSelectedDocuments] = useState([]);
  const [similarityMatrix, setSimilarityMatrix] = useState(null);
  const [showSimilarityMatrix, setShowSimilarityMatrix] = useState(false);
  const plotRef = useRef(null);
  
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  
  // Compute cosine similarity between two vectors
  const cosineSimilarity = (a, b) => {
    if (a.length !== b.length) return 0;
    
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    if (normA === 0 || normB === 0) return 0;
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  };
  
  // Compute similarity matrix for selected documents.
  const computeSimilarityMatrix = () => {
    if (selectedDocuments.length < 2) return;
    
    const embeddings = selectedDocuments.map(doc => doc.embedding).filter(Boolean);
    if (embeddings.length !== selectedDocuments.length) {
      alert('Some selected documents are missing embeddings');
      return;
    }
    
    const matrix = [];
    for (let i = 0; i < embeddings.length; i++) {
      const row = [];
      for (let j = 0; j < embeddings.length; j++) {
        if (i === j) {
          row.push(1.0); // Self-similarity
        } else {
          row.push(cosineSimilarity(embeddings[i], embeddings[j]));
        }
      }
      matrix.push(row);
    }
    
    setSimilarityMatrix(matrix);
    setShowSimilarityMatrix(true);
  };
  
  // Handle document selection from plot
  const handlePlotSelection = (event) => {
    if (!event.points || event.points.length === 0) return;
    
    const selectedIndices = event.points.map(point => {
      const traceIndex = point.curveNumber;
      const pointIndex = point.pointIndex;
      
      if (plotData && plotData[traceIndex] && plotData[traceIndex].customdata) {
        return plotData[traceIndex].customdata[pointIndex];
      }
      return pointIndex;
    });
    
    const selectedDocs = selectedIndices.map(index => documents.results[index]).filter(Boolean);
    setSelectedDocuments(selectedDocs);
  };
  
  // Clear selection
  const clearSelection = () => {
    setSelectedDocuments([]);
    setSimilarityMatrix(null);
    setShowSimilarityMatrix(false);
  };
  
  // Fetch available indices on component mount
  useEffect(() => {
    const fetchIndices = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/indices`);
        const data = await response.json();
        
        if (data.indices && data.indices.length > 0) {
          setAvailableIndices(data.indices);
          // Set default index if available
          if (data.default_index && data.indices.some(idx => idx.name === data.default_index)) {
            setSelectedIndex(data.default_index);
          } else if (data.indices.length > 0) {
            setSelectedIndex(data.indices[0].name);
          }
        }
      } catch (err) {
        console.error('Failed to fetch indices:', err);
      }
    };
    
    fetchIndices();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  
  const loadDocuments = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const requestBody = {
        num_documents: numDocuments,
        load_full_index: loadFullIndex,
        include_embeddings: true,
        index_name: selectedIndex || undefined
      };
      
      const response = await fetch(`${API_BASE_URL}/visualization-explorer`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setDocuments(data);
        calculateVisualization(data);
      } else {
        setError({
          message: data.detail || 'Failed to load documents',
          type: 'api'
        });
      }
    } catch (err) {
      setError({
        message: 'Network error. Please check your connection and try again.',
        type: 'network'
      });
    } finally {
      setIsLoading(false);
    }
  };
  
  const calculateVisualization = async (data = documents) => {
    if (!data || !data.results.length) return;
    
    setIsCalculating(true);
    setError(null);
    
    try {
      const embeddings = data.results.map(r => r.embedding).filter(Boolean);
      
      if (!embeddings.length) {
        throw new Error('No embeddings available for visualization');
      }
      
      // Prepare request body based on method
      const requestBody = {
        embeddings: embeddings,
        method: method,
        random_state: 42
      };
      
      // Add method-specific parameters
      if (method === 'pca') {
        requestBody.n_components = 2;
      } else if (method === 'tsne') {
        // Auto-calculate perplexity if not set
        const autoPerplexity = Math.min(30, Math.floor(embeddings.length / 3));
        requestBody.perplexity = autoPerplexity;
        requestBody.n_iter = 1000;
        requestBody.learning_rate = 200.0;
        requestBody.early_exaggeration = 12.0;
      } else if (method === 'umap') {
        requestBody.n_neighbors = Math.min(umapParams.nNeighbors, Math.floor(embeddings.length / 3));
        requestBody.min_dist = umapParams.minDist;
        requestBody.n_components = 2;
        requestBody.metric = 'cosine';
      }
      
      // Call backend endpoint for visualization calculation
      const response = await fetch(`${API_BASE_URL}/visualization-explorer/calculate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to calculate visualization');
      }
      
      const result = await response.json();
      const coords = result.coordinates;
      
      if (!coords || coords.length !== embeddings.length) {
        throw new Error('Invalid coordinates returned from backend');
      }
      
      // Generate color mapping based on selected attribute
      const colorMapping = generateColorMapping(data.results, colorBy);
      
      // Create plot traces
      const plotTraces = [];
      
      Object.entries(colorMapping).forEach(([category, indices]) => {
        if (indices.length > 0) {
          const trace = {
            x: indices.map(i => coords[i][0]),
            y: indices.map(i => coords[i][1]),
            mode: 'markers',
            type: 'scatter',
            name: category,
            marker: {
              size: 6,
              color: getColorForCategory(category, colorBy),
              opacity: 0.7,
              line: { width: 0.5, color: '#FFF' }
            },
            text: indices.map(i => {
              const doc = data.results[i];
              const metadata = doc.metadata || {};
              return `<b>${metadata.title || 'Document'}</b><br>` +
                     `Language: ${metadata.language || metadata.main_language || 'Unknown'}<br>` +
                     `Type: ${metadata.document_type || 'Unknown'}<br>` +
                     `Collection: ${metadata.collection || 'Unknown'}<br>` +
                     `ID: ${doc.doc_id}`;
            }),
            hovertemplate: '%{text}<extra></extra>',
            showlegend: true,
            customdata: indices
          };
          plotTraces.push(trace);
        }
      });
      
      setPlotData(plotTraces);
      
    } catch (err) {
      console.error('Visualization calculation failed:', err);
      setError({
        message: err.message || 'Failed to generate visualization',
        type: 'calculation'
      });
    } finally {
      setIsCalculating(false);
    }
  };
  
  const generateColorMapping = (results, attribute) => {
    const mapping = {};
    
    results.forEach((result, index) => {
      let value = 'Unknown';
      
      switch (attribute) {
        case 'language':
          value = result.metadata?.language || result.metadata?.main_language || 'Unknown';
          break;
        case 'document_type':
          value = result.metadata?.document_type || 'Unknown';
          break;
        case 'collection':
          value = result.metadata?.collection || 'Unknown';
          break;
        case 'institution':
          value = result.metadata?.institution || 'Unknown';
          break;
        case 'period':
          value = result.metadata?.period || 'Unknown';
          break;
        case 'material':
          value = result.metadata?.material || 'Unknown';
          break;
        default:
          value = 'Unknown';
      }
      
      // Handle multi-value strings (e.g., "Hebrew; Arabic")
      const primaryValue = value.split(';')[0].trim();
      
      if (!mapping[primaryValue]) {
        mapping[primaryValue] = [];
      }
      mapping[primaryValue].push(index);
    });
    
    return mapping;
  };
  
  // Generate a consistent color from a string using hash
  const generateColorFromString = (str) => {
    // Simple hash function
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }
    
    // Use HSL color space for better color distribution
    // Hue: 0-360 (full spectrum)
    // Saturation: 60-90% (vibrant but not too intense)
    // Lightness: 45-65% (visible but not too dark/light)
    const hue = Math.abs(hash) % 360;
    const saturation = 60 + (Math.abs(hash * 7) % 30); // 60-90%
    const lightness = 45 + (Math.abs(hash * 11) % 20); // 45-65%
    
    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
  };

  const getColorForCategory = (category, attribute) => {
    const colorPalettes = {
      language: {
        'Hebrew': '#2E8B57',
        'Arabic': '#8B008B',
        'Aramaic': '#4169E1',
        'Judeo-Arabic': '#DAA520',
        'Judaeo-Arabic': '#DAA520', // Alternative spelling
        'Greek': '#9932CC',
        'Latin': '#FF8C00',
        'Persian': '#20B2AA',
        'Syriac': '#696969',
        'Coptic': '#4B0082',
        'Unknown': '#A9A9A9'
      },
      document_type: {
        'legal': '#E74C3C',
        'liturgical': '#3498DB',
        'literary': '#2ECC71',
        'commercial': '#F39C12',
        'personal': '#9B59B6',
        'religious': '#1ABC9C',
        'administrative': '#34495E',
        'Unknown': '#A9A9A9'
      },
      collection: {
        'taylor_schechter': '#E74C3C',
        'adler': '#3498DB',
        'gottheil_worrell': '#2ECC71',
        'cambridge': '#F39C12',
        'princeton': '#9B59B6',
        'jts': '#1ABC9C',
        'Unknown': '#A9A9A9'
      },
      institution: {
        'cambridge': '#E74C3C',
        'jewish_theological_seminary': '#3498DB',
        'princeton': '#2ECC71',
        'oxford': '#F39C12',
        'manchester': '#9B59B6',
        'Unknown': '#A9A9A9'
      },
      period: {
        'medieval': '#E74C3C',
        'early_medieval': '#3498DB',
        'late_medieval': '#2ECC71',
        'early_modern': '#F39C12',
        'modern': '#9B59B6',
        'Unknown': '#A9A9A9'
      },
      material: {
        'parchment': '#8B4513',
        'paper': '#F5DEB3',
        'papyrus': '#D2B48C',
        'vellum': '#DEB887',
        'Unknown': '#A9A9A9'
      }
    };
    
    const palette = colorPalettes[attribute] || colorPalettes.language;
    
    // Check if category exists in predefined palette
    if (palette[category]) {
      return palette[category];
    }
    
    // For "Unknown", return gray
    if (category === 'Unknown') {
      return palette['Unknown'];
    }
    
    // Generate a vibrant color for unmapped categories
    // Include attribute in hash to ensure different colors for same category name across attributes
    return generateColorFromString(`${attribute}:${category}`);
  };
  
  const handlePlotClick = (event) => {
    if (!onDocumentClick || !event.points || event.points.length === 0) return;
    
    const point = event.points[0];
    const pointIndex = point.pointIndex;
    const traceIndex = point.curveNumber;
    
    // Get the actual result index from customdata
    let resultIndex = pointIndex;
    
    if (plotData && plotData[traceIndex] && plotData[traceIndex].customdata) {
      resultIndex = plotData[traceIndex].customdata[pointIndex];
    }
    
    if (resultIndex >= 0 && resultIndex < documents.results.length) {
      const result = documents.results[resultIndex];
      const metadata = result.metadata || {};
      
      const displayData = {
        title: metadata.title || `Document ${result.doc_id}`,
        description: metadata.description || "Historical manuscript from the Cairo Genizah collection.",
        image_url: (() => {
          // First priority: actual_image_url (best quality)
          if (metadata.actual_image_url) {
            return metadata.actual_image_url;
          }
          // Second priority: image_urls array
          if (metadata.image_urls && metadata.image_urls.length > 0) {
            const validUrls = metadata.image_urls.filter(url => url && url.trim());
            if (validUrls.length > 0) {
              return validUrls[0];
            }
          }
          // Third priority: image_url
          if (metadata.image_url) {
            return metadata.image_url;
          }
          // Fourth priority: thumbnail_url
          if (metadata.thumbnail_url) {
            return metadata.thumbnail_url;
          }
          // Fallback
          return "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=400&h=300&fit=crop";
        })(),
        date: metadata.date || "Unknown",
        language: metadata.language || metadata.main_language || "Hebrew",
        material: metadata.material || "Parchment",
        institution: metadata.institution,
        collection: metadata.collection,
        shelfmark: metadata.shelf_mark,
        transcription: metadata.transcription_full_text,
        translation: metadata.translation_full_text,
        tags: metadata.tags,
        period: metadata.period,
        location: metadata.location,
        dimensions: metadata.dimensions,
        document_type: metadata.document_type,
        doc_id: result.doc_id,
        similarity_score: result.similarity_score,
        ...result
      };
      
      onDocumentClick(displayData);
    }
  };
  
  useEffect(() => {
    if (documents) {
      calculateVisualization();
    }
  }, [method, colorBy]);

  // Auto-reload documents when switching index after initial fetch
  useEffect(() => {
    // Only trigger reload if we already showed setup indices and have a selection
    if (selectedIndex && (documents || availableIndices.length > 0)) {
      // Reset existing visualization state and reload from new index
      if (documents) {
        setDocuments(null);
        setPlotData(null);
      }
      // Load from the newly selected index
      // Debounce slightly to avoid double fires on rapid changes
      const t = setTimeout(() => {
        loadDocuments();
      }, 50);
      return () => clearTimeout(t);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedIndex]);
  
  const layout = {
    title: {
      text: `Cairo Genizah Collection Explorer (${method.toUpperCase()})`,
      font: { size: 18, family: 'Arial, sans-serif' }
    },
    xaxis: {
      title: `${method.toUpperCase()} Dimension 1`,
      showgrid: true,
      gridcolor: '#E5E5E5',
      zeroline: false,
      showticklabels: false
    },
    yaxis: {
      title: `${method.toUpperCase()} Dimension 2`,
      showgrid: true,
      gridcolor: '#E5E5E5',
      zeroline: false,
      showticklabels: false
    },
    plot_bgcolor: '#FAFAFA',
    paper_bgcolor: '#FFFFFF',
    margin: { l: 60, r: 200, t: 80, b: 60 },
    hovermode: 'closest',
    showlegend: true,
    legend: {
      x: 1.02,
      y: 1,
      xanchor: 'left',
      yanchor: 'top',
      bgcolor: 'rgba(255,255,255,0.9)',
      bordercolor: '#CCC',
      borderwidth: 1,
      font: { size: 12 },
      title: {
        text: `<b>${colorBy.replace('_', ' ').toUpperCase()}</b>`,
        font: { size: 13 }
      }
    },
    annotations: documents ? [{
      text: `${documents.count} documents visualized`,
      showarrow: false,
      x: 0.02,
      y: 0.02,
      xref: 'paper',
      yref: 'paper',
      xanchor: 'left',
      yanchor: 'bottom',
      font: { size: 12, color: '#666' }
    }] : []
  };
  
  const config = {
    displayModeBar: true,
    modeBarButtonsToRemove: [
      'pan2d', 'autoScale2d', 
      'hoverClosestCartesian', 'hoverCompareCartesian'
    ],
    displaylogo: false,
    responsive: true,
    toImageButtonOptions: {
      format: 'png',
      filename: `genizah_explorer_${method}_${colorBy}`,
      width: 1200,
      height: 800
    }
  };
  
  if (error) {
    return (
      <div className="visualization-explorer error">
        <div className="error-message">
          <span>⚠️ {error.message || error}</span>
          <button onClick={loadDocuments} className="retry-btn">
            Retry
          </button>
        </div>
      </div>
    );
  }
  
  if (isLoading) {
    return (
      <div className="visualization-explorer loading">
        <div className="loading-content">
          <div className="spinner"></div>
          <p>Loading documents from the Cairo Genizah collection...</p>
          <p className="loading-details">
            {loadFullIndex ? 'Loading entire collection' : `Loading ${numDocuments} documents`}
          </p>
        </div>
      </div>
    );
  }
  
  if (!documents) {
    return (
      <div className="visualization-explorer setup">
        <div className="setup-content">
          <h2>Collection Explorer Setup</h2>
          <p>Configure how many documents to load for visualization:</p>
          
          <div className="setup-controls">
            <div className="control-group">
              <label>
                Select Index:
                <select
                  value={selectedIndex || ''}
                  onChange={(e) => setSelectedIndex(e.target.value)}
                  className="index-select"
                >
                  {availableIndices.map((idx) => (
                    <option key={idx.name} value={idx.name}>
                      {idx.name} ({idx.document_count.toLocaleString()} documents{idx.is_default ? ' - default' : ''})
                    </option>
                  ))}
                </select>
              </label>
              {selectedIndex && (
                <small style={{ color: '#666', marginTop: '4px', display: 'block' }}>
                  {availableIndices.find(idx => idx.name === selectedIndex)?.description || ''}
                </small>
              )}
            </div>
            
            <div className="control-group">
              <label>
                <input
                  type="checkbox"
                  checked={loadFullIndex}
                  onChange={(e) => setLoadFullIndex(e.target.checked)}
                />
                Load entire collection
              </label>
            </div>
            
            {!loadFullIndex && (
              <div className="control-group">
                <label>
                  Number of documents:
                  <input
                    type="number"
                    value={numDocuments}
                    onChange={(e) => setNumDocuments(parseInt(e.target.value) || 1000)}
                    min="10"
                    max="10000"
                  />
                </label>
              </div>
            )}
          </div>
          
          <button onClick={loadDocuments} className="load-btn" disabled={!selectedIndex}>
            Load Documents
          </button>
        </div>
      </div>
    );
  }
  
  return (
    <div className="visualization-explorer">
      <div className="explorer-header">
        <div className="header-left">
          <h1>Cairo Genizah Collection Explorer</h1>
          <p>Explore the semantic relationships in the collection</p>
        </div>
        
        <div className="header-right">
          {availableIndices && availableIndices.length > 0 && (
            <div className="index-switcher">
              <label>
                Index
                <select
                  value={selectedIndex || ''}
                  onChange={(e) => setSelectedIndex(e.target.value)}
                  className="header-index-select"
                >
                  {availableIndices.map((idx) => (
                    <option key={idx.name} value={idx.name}>
                      {idx.name}
                    </option>
                  ))}
                </select>
              </label>
            </div>
          )}
          <button onClick={() => navigate('/')} className="back-btn">
            ← Back to Search
          </button>
        </div>
      </div>
      
      <div className="explorer-controls">
        <div className="control-group">
          <label>
            Visualization Method:
            <select 
              value={method} 
              onChange={(e) => setMethod(e.target.value)}
              disabled={isCalculating}
            >
              <option value="pca">PCA (Fast)</option>
              <option value="tsne">t-SNE (Detailed)</option>
              <option value="umap">UMAP (Balanced)</option>
            </select>
          </label>
        </div>
        
        <div className="control-group">
          <label>
            Color By:
            <select 
              value={colorBy} 
              onChange={(e) => setColorBy(e.target.value)}
              disabled={isCalculating}
            >
              <option value="language">Language</option>
              <option value="document_type">Document Type</option>
              <option value="collection">Collection</option>
              <option value="institution">Institution</option>
              <option value="period">Period</option>
              <option value="material">Material</option>
            </select>
          </label>
        </div>
        
        <div className="control-group">
          <button 
            onClick={() => calculateVisualization()} 
            disabled={isCalculating}
            className="recalculate-btn"
          >
            {isCalculating ? 'Calculating...' : 'Recalculate'}
          </button>
        </div>
        
        <div className="control-group">
          <div className="selection-controls">
            <button 
              onClick={computeSimilarityMatrix}
              disabled={selectedDocuments.length < 2}
              className="similarity-btn"
            >
              Compute Similarities ({selectedDocuments.length} selected)
            </button>
            <button 
              onClick={clearSelection}
              disabled={selectedDocuments.length === 0}
              className="clear-btn"
            >
              Clear Selection
            </button>
          </div>
        </div>
      </div>
      
      {method === 'umap' && (
        <div className="umap-params">
          <h4>UMAP Parameters:</h4>
          <div className="param-controls">
            <div className="param-group">
              <label>
                nNeighbors: {umapParams.nNeighbors}
                <input
                  type="range"
                  min="5"
                  max="50"
                  value={umapParams.nNeighbors}
                  onChange={(e) => setUmapParams(prev => ({ ...prev, nNeighbors: parseInt(e.target.value) }))}
                  disabled={isCalculating}
                />
              </label>
            </div>
            <div className="param-group">
              <label>
                minDist: {umapParams.minDist}
                <input
                  type="range"
                  min="0.01"
                  max="1.0"
                  step="0.01"
                  value={umapParams.minDist}
                  onChange={(e) => setUmapParams(prev => ({ ...prev, minDist: parseFloat(e.target.value) }))}
                  disabled={isCalculating}
                />
              </label>
            </div>
            <div className="param-group">
              <label>
                iterations: {umapParams.iterations}
                <input
                  type="range"
                  min="100"
                  max="1000"
                  step="50"
                  value={umapParams.iterations}
                  onChange={(e) => setUmapParams(prev => ({ ...prev, iterations: parseInt(e.target.value) }))}
                  disabled={isCalculating}
                />
              </label>
            </div>
          </div>
        </div>
      )}
      
      {isCalculating && (
        <div className="calculation-overlay">
          <div className="calculation-content">
            <div className="spinner"></div>
            <p>Calculating {method.toUpperCase()} visualization...</p>
          </div>
        </div>
      )}
      
      <div className="plot-container" ref={plotRef}>
        {plotData && (
          <Plot
            data={plotData}
            layout={layout}
            config={config}
            style={{ width: '100%', height: '70vh' }}
            useResizeHandler={true}
            onClick={handlePlotClick}
            onSelected={handlePlotSelection}
          />
        )}
      </div>
      
      <div className="explorer-info">
        <p>
          <strong>What this shows:</strong> This {method.toUpperCase()} plot visualizes semantic relationships 
          between documents in the Cairo Genizah collection. Documents closer together are more semantically similar. 
          Colors represent different {colorBy.replace('_', ' ')} categories. Click on any point to view document details.
        </p>
        
        <div className="debug-info">
          <h4>Debug Information:</h4>
          <p><strong>Method:</strong> {method.toUpperCase()}</p>
          <p><strong>Documents:</strong> {documents?.count || 0}</p>
          <p><strong>Embedding Dimension:</strong> {documents?.embedding_data?.dimension || 'Unknown'}</p>
          <p><strong>Color Categories:</strong> {plotData ? Object.keys(generateColorMapping(documents.results, colorBy)).length : 0}</p>
          
          <div className="coordinate-stats">
            <h5>Coordinate Statistics:</h5>
            {plotData && plotData.length > 0 && (
              <div>
                <p>X Range: {Math.min(...plotData.flatMap(trace => trace.x)).toFixed(3)} to {Math.max(...plotData.flatMap(trace => trace.x)).toFixed(3)}</p>
                <p>Y Range: {Math.min(...plotData.flatMap(trace => trace.y)).toFixed(3)} to {Math.max(...plotData.flatMap(trace => trace.y)).toFixed(3)}</p>
                <p>Total Points: {plotData.reduce((sum, trace) => sum + trace.x.length, 0)}</p>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {showSimilarityMatrix && similarityMatrix && (
        <div className="similarity-matrix">
          <h3>Cosine Similarity Matrix</h3>
          <p>Selected {selectedDocuments.length} documents. Values range from -1 (completely dissimilar) to 1 (identical).</p>
          
          <div className="matrix-container">
            <table className="similarity-table">
              <thead>
                <tr>
                  <th></th>
                  {selectedDocuments.map((doc, i) => (
                    <th key={i} title={doc.metadata?.title || doc.doc_id}>
                      Doc {i + 1}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {similarityMatrix.map((row, i) => (
                  <tr key={i}>
                    <th title={selectedDocuments[i]?.metadata?.title || selectedDocuments[i]?.doc_id}>
                      Doc {i + 1}
                    </th>
                    {row.map((value, j) => (
                      <td 
                        key={j} 
                        className={`similarity-cell ${i === j ? 'diagonal' : ''}`}
                        style={{
                          backgroundColor: i === j 
                            ? '#E8F4FD' 
                            : `rgba(52, 152, 219, ${Math.max(0, value)})`,
                          color: value < 0.3 ? '#666' : '#000'
                        }}
                        title={`${selectedDocuments[i]?.metadata?.title || selectedDocuments[i]?.doc_id} ↔ ${selectedDocuments[j]?.metadata?.title || selectedDocuments[j]?.doc_id}: ${value.toFixed(3)}`}
                      >
                        {value.toFixed(3)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          <div className="selected-docs-info">
            <h4>Selected Documents:</h4>
            <p className="click-hint">💡 Click on any document below to view its details</p>
            <div className="doc-list">
              {selectedDocuments.map((doc, i) => (
                <div 
                  key={i} 
                  className="doc-item clickable"
                  onClick={() => onDocumentClick && onDocumentClick(doc)}
                  title="Click to view document details"
                >
                  <div className="doc-header">
                    <strong>Doc {i + 1}:</strong> {doc.metadata?.title || doc.doc_id}
                    <span className="click-icon">👆</span>
                  </div>
                  <small>
                    Language: {doc.metadata?.language || doc.metadata?.main_language || 'Unknown'} | 
                    Type: {doc.metadata?.document_type || 'Unknown'} | 
                    ID: {doc.doc_id}
                  </small>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
      
      <style jsx>{`
        .visualization-explorer {
          min-height: 100vh;
          background: #FAFAFA;
          padding: 0;
        }
        
        .explorer-header {
          background: #2C3E50;
          color: white;
          padding: 20px 40px;
          display: flex;
          justify-content: space-between;
          align-items: center;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .header-left h1 {
          margin: 0 0 8px 0;
          font-size: 28px;
          font-weight: 600;
        }
        
        .header-left p {
          margin: 0;
          font-size: 16px;
          opacity: 0.9;
        }
        
        .back-btn {
          padding: 12px 24px;
          background: #3498DB;
          color: white;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
          font-weight: 500;
          transition: background-color 0.2s;
        }
        
        .back-btn:hover {
          background: #2980B9;
        }
        
        .index-switcher {
          display: flex;
          align-items: center;
          gap: 8px;
          margin-right: 12px;
        }
        
        .index-switcher label {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
        }
        
        .header-index-select {
          padding: 8px 12px;
          border: 1px solid #DDD;
          border-radius: 4px;
          background: white;
          font-size: 14px;
          cursor: pointer;
        }
        
        .header-index-select:focus {
          outline: none;
          border-color: #3498DB;
          box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.25);
        }
        
        .explorer-controls {
          background: white;
          padding: 20px 40px;
          border-bottom: 1px solid #E5E5E5;
          display: flex;
          gap: 30px;
          align-items: center;
          flex-wrap: wrap;
        }
        
        .control-group {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        
        .control-group label {
          font-size: 14px;
          font-weight: 500;
          color: #2C3E50;
        }
        
        .control-group select {
          padding: 8px 12px;
          border: 1px solid #DDD;
          border-radius: 4px;
          background: white;
          font-size: 14px;
          cursor: pointer;
        }
        
        .control-group select:focus {
          outline: none;
          border-color: #3498DB;
          box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.25);
        }
        
        .recalculate-btn {
          padding: 8px 16px;
          background: #27AE60;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 14px;
          font-weight: 500;
          transition: background-color 0.2s;
        }
        
        .recalculate-btn:hover:not(:disabled) {
          background: #229954;
        }
        
        .recalculate-btn:disabled {
          background: #95A5A6;
          cursor: not-allowed;
        }
        
        .plot-container {
          padding: 20px 40px;
          background: white;
          margin: 0;
        }
        
        .explorer-info {
          background: #F8F9FA;
          padding: 20px 40px;
          border-top: 1px solid #E5E5E5;
        }
        
        .explorer-info p {
          margin: 0;
          font-size: 14px;
          color: #666;
          line-height: 1.5;
        }
        
        .debug-info {
          margin-top: 20px;
          padding: 16px;
          background: #F8F9FA;
          border-radius: 6px;
          border: 1px solid #E9ECEF;
        }
        
        .debug-info h4 {
          margin: 0 0 12px 0;
          font-size: 16px;
          color: #2C3E50;
        }
        
        .debug-info h5 {
          margin: 12px 0 8px 0;
          font-size: 14px;
          color: #34495E;
        }
        
        .debug-info p {
          margin: 4px 0;
          font-size: 13px;
          color: #666;
        }
        
        .coordinate-stats {
          margin-top: 12px;
          padding-top: 12px;
          border-top: 1px solid #DEE2E6;
        }
        
        .umap-params {
          background: white;
          padding: 20px 40px;
          border-bottom: 1px solid #E5E5E5;
        }
        
        .umap-params h4 {
          margin: 0 0 16px 0;
          color: #2C3E50;
          font-size: 16px;
        }
        
        .param-controls {
          display: flex;
          gap: 30px;
          flex-wrap: wrap;
        }
        
        .param-group {
          display: flex;
          flex-direction: column;
          gap: 8px;
          min-width: 200px;
        }
        
        .param-group label {
          font-size: 14px;
          font-weight: 500;
          color: #2C3E50;
        }
        
        .param-group input[type="range"] {
          width: 100%;
          margin-top: 4px;
        }
        
        .calculation-overlay {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: rgba(255, 255, 255, 0.9);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
        }
        
        .calculation-content {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 20px;
          background: white;
          padding: 40px;
          border-radius: 8px;
          box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        .loading, .error, .setup {
          display: flex;
          align-items: center;
          justify-content: center;
          min-height: 100vh;
          background: #FAFAFA;
        }
        
        .loading-content, .setup-content {
          text-align: center;
          background: white;
          padding: 40px;
          border-radius: 8px;
          box-shadow: 0 4px 12px rgba(0,0,0,0.1);
          max-width: 500px;
        }
        
        .setup-content h2 {
          margin: 0 0 16px 0;
          color: #2C3E50;
        }
        
        .setup-controls {
          margin: 24px 0;
          text-align: left;
        }
        
        .setup-controls .control-group {
          margin: 16px 0;
        }
        
        .setup-controls label {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
          color: #2C3E50;
          flex-direction: column;
          align-items: flex-start;
        }
        
        .setup-controls .index-select {
          padding: 8px 12px;
          border: 1px solid #DDD;
          border-radius: 4px;
          background: white;
          font-size: 14px;
          cursor: pointer;
          width: 100%;
          margin-top: 8px;
        }
        
        .setup-controls .index-select:focus {
          outline: none;
          border-color: #3498DB;
          box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.25);
        }
        
        .setup-controls input[type="number"] {
          padding: 8px 12px;
          border: 1px solid #DDD;
          border-radius: 4px;
          width: 120px;
          margin-left: 8px;
        }
        
        .load-btn {
          padding: 12px 24px;
          background: #3498DB;
          color: white;
          border: none;
          border-radius: 6px;
          cursor: pointer;
          font-size: 16px;
          font-weight: 500;
          transition: background-color 0.2s;
        }
        
        .load-btn:hover {
          background: #2980B9;
        }
        
        .loading-details {
          font-size: 14px;
          color: #666;
          margin-top: 8px;
        }
        
        .spinner {
          width: 40px;
          height: 40px;
          border: 4px solid #E5E5E5;
          border-top: 4px solid #3498DB;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }
        
        .error-message {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 16px;
          color: #E74C3C;
          font-size: 16px;
        }
        
        .retry-btn {
          padding: 8px 16px;
          background: #E74C3C;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 14px;
        }
        
        .retry-btn:hover {
          background: #C0392B;
        }
        
        .selection-controls {
          display: flex;
          gap: 12px;
          align-items: center;
        }
        
        .similarity-btn {
          padding: 8px 16px;
          background: #3498DB;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 14px;
          font-weight: 500;
          transition: background-color 0.2s;
        }
        
        .similarity-btn:hover:not(:disabled) {
          background: #2980B9;
        }
        
        .similarity-btn:disabled {
          background: #95A5A6;
          cursor: not-allowed;
        }
        
        .clear-btn {
          padding: 8px 16px;
          background: #E74C3C;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 14px;
          font-weight: 500;
          transition: background-color 0.2s;
        }
        
        .clear-btn:hover:not(:disabled) {
          background: #C0392B;
        }
        
        .clear-btn:disabled {
          background: #95A5A6;
          cursor: not-allowed;
        }
        
        .similarity-matrix {
          background: white;
          padding: 20px 40px;
          border-top: 1px solid #E5E5E5;
        }
        
        .similarity-matrix h3 {
          margin: 0 0 8px 0;
          color: #2C3E50;
          font-size: 20px;
        }
        
        .similarity-matrix p {
          margin: 0 0 20px 0;
          color: #666;
          font-size: 14px;
        }
        
        .matrix-container {
          overflow-x: auto;
          margin-bottom: 20px;
        }
        
        .similarity-table {
          border-collapse: collapse;
          width: 100%;
          min-width: 300px;
        }
        
        .similarity-table th,
        .similarity-table td {
          border: 1px solid #DDD;
          padding: 8px;
          text-align: center;
          font-size: 12px;
        }
        
        .similarity-table th {
          background: #F8F9FA;
          font-weight: 600;
          color: #2C3E50;
        }
        
        .similarity-cell {
          font-weight: 500;
          min-width: 60px;
        }
        
        .similarity-cell.diagonal {
          font-weight: 700;
        }
        
        .selected-docs-info {
          background: #F8F9FA;
          padding: 16px;
          border-radius: 6px;
          border: 1px solid #E9ECEF;
        }
        
        .selected-docs-info h4 {
          margin: 0 0 12px 0;
          color: #2C3E50;
          font-size: 16px;
        }
        
        .click-hint {
          margin: 0 0 12px 0;
          color: #3498DB;
          font-size: 13px;
          font-style: italic;
          background: #E8F4FD;
          padding: 8px 12px;
          border-radius: 4px;
          border-left: 3px solid #3498DB;
        }
        
        .doc-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }
        
        .doc-item {
          padding: 8px;
          background: white;
          border-radius: 4px;
          border: 1px solid #E9ECEF;
          font-size: 13px;
        }
        
        .doc-item.clickable {
          cursor: pointer;
          transition: all 0.2s ease;
        }
        
        .doc-item.clickable:hover {
          background: #F8F9FA;
          border-color: #3498DB;
          box-shadow: 0 2px 4px rgba(52, 152, 219, 0.1);
          transform: translateY(-1px);
        }
        
        .doc-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 4px;
        }
        
        .click-icon {
          font-size: 12px;
          opacity: 0.6;
          transition: opacity 0.2s;
        }
        
        .doc-item.clickable:hover .click-icon {
          opacity: 1;
        }
        
        .doc-item strong {
          color: #2C3E50;
        }
        
        .doc-item small {
          color: #666;
        }
        
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
          .explorer-header {
            padding: 16px 20px;
            flex-direction: column;
            gap: 16px;
            text-align: center;
          }
          
          .explorer-controls {
            padding: 16px 20px;
            flex-direction: column;
            align-items: stretch;
            gap: 16px;
          }
          
          .plot-container {
            padding: 16px 20px;
          }
          
          .explorer-info {
            padding: 16px 20px;
          }
        }
      `}</style>
    </div>
  );
};

export default VisualizationExplorer;
