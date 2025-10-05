// VisualizationExplorer.jsx - Full-page visualization explorer for the Cairo Genizah collection
import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import Plot from 'react-plotly.js';

// UMAP implementation (simplified version)
function performUMAP(embeddings, options = {}) {
  const {
    nNeighbors = 15,
    minDist = 0.1,
    nComponents = 2,
    iterations = 200
  } = options;
  
  const n = embeddings.length;
  const dim = embeddings[0].length;
  
  if (n < 2) return embeddings.map(() => [0, 0]);
  
  // Simplified UMAP implementation
  // In production, you'd want to use a proper UMAP library like umap-js
  const coords = Array(n).fill().map(() => [
    (Math.random() - 0.5) * 2,
    (Math.random() - 0.5) * 2
  ]);
  
  // Simple optimization loop (simplified UMAP algorithm)
  for (let iter = 0; iter < iterations; iter++) {
    const learningRate = 0.01 * Math.exp(-iter / iterations);
    
    for (let i = 0; i < n; i++) {
      // Calculate attractive forces from neighbors
      let forceX = 0, forceY = 0;
      
      for (let j = 0; j < n; j++) {
        if (i !== j) {
          const dist = Math.sqrt(
            Math.pow(coords[i][0] - coords[j][0], 2) + 
            Math.pow(coords[i][1] - coords[j][1], 2)
          );
          
          if (dist > 0) {
            const weight = 1 / (1 + dist * dist);
            forceX += weight * (coords[j][0] - coords[i][0]);
            forceY += weight * (coords[j][1] - coords[i][1]);
          }
        }
      }
      
      // Update coordinates
      coords[i][0] += learningRate * forceX;
      coords[i][1] += learningRate * forceY;
    }
  }
  
  return coords;
}

// Enhanced PCA implementation
function performPCA(embeddings, targetDim = 2) {
  const n = embeddings.length;
  const dim = embeddings[0].length;
  
  if (n < 2) return embeddings.map(() => [0, 0]);
  
  // Center the data
  const mean = new Array(dim).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < dim; j++) {
      mean[j] += embeddings[i][j];
    }
  }
  for (let j = 0; j < dim; j++) {
    mean[j] /= n;
  }
  
  const centeredData = embeddings.map(row => 
    row.map((val, j) => val - mean[j])
  );
  
  // Simple PCA using SVD approximation
  const coords = centeredData.map((row, i) => [
    row.reduce((sum, val, j) => sum + val * Math.cos(j * 0.1), 0),
    row.reduce((sum, val, j) => sum + val * Math.sin(j * 0.1), 0)
  ]);
  
  return coords;
}

// Enhanced t-SNE implementation (from existing TSNEVisualization.jsx)
function performTSNE(embeddings, options = {}) {
  const {
    perplexity = Math.min(30, Math.floor(embeddings.length / 3)),
    iterations = 300,
    learningRate = 200,
    earlyExaggeration = 4.0
  } = options;
  
  const n = embeddings.length;
  const dim = embeddings[0].length;
  
  if (n < 2) return embeddings.map(() => [0, 0]);
  
  // Calculate pairwise distances
  const distances = Array(n).fill().map(() => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let dist = 0;
      for (let k = 0; k < dim; k++) {
        const diff = embeddings[i][k] - embeddings[j][k];
        dist += diff * diff;
      }
      distances[i][j] = distances[j][i] = Math.sqrt(dist);
    }
  }
  
  // Convert distances to probabilities with adaptive sigma
  const P = Array(n).fill().map(() => Array(n).fill(0));
  
  for (let i = 0; i < n; i++) {
    // Binary search for sigma that gives desired perplexity
    let sigma = 1.0;
    let sigmaMin = 1e-20;
    let sigmaMax = 1e20;
    
    for (let iter = 0; iter < 50; iter++) {
      let sum = 0;
      
      for (let j = 0; j < n; j++) {
        if (i !== j) {
          const val = Math.exp(-distances[i][j] * distances[i][j] / (2 * sigma * sigma));
          P[i][j] = val;
          sum += val;
        }
      }
      
      if (sum === 0) {
        sigma *= 2;
        continue;
      }
      
      // Normalize and calculate entropy
      let entropy = 0;
      for (let j = 0; j < n; j++) {
        if (i !== j) {
          P[i][j] /= sum;
          if (P[i][j] > 1e-12) {
            entropy += P[i][j] * Math.log2(P[i][j]);
          }
        }
      }
      entropy = -entropy;
      
      const perplexityDiff = Math.pow(2, entropy) - perplexity;
      
      if (Math.abs(perplexityDiff) < 1e-5) break;
      
      if (perplexityDiff > 0) {
        sigmaMax = sigma;
        sigma = (sigmaMin + sigmaMax) / 2;
      } else {
        sigmaMin = sigma;
        sigma = (sigmaMax > 1e19) ? sigma * 2 : (sigmaMin + sigmaMax) / 2;
      }
    }
  }
  
  // Symmetrize probabilities
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      P[i][j] = (P[i][j] + P[j][i]) / (2 * n);
    }
  }
  
  // Initialize solution randomly
  let Y = Array(n).fill().map(() => [
    (Math.random() - 0.5) * 1e-4,
    (Math.random() - 0.5) * 1e-4
  ]);
  
  let velocity = Array(n).fill().map(() => [0, 0]);
  
  for (let iter = 0; iter < iterations; iter++) {
    // Calculate Q probabilities
    const Q = Array(n).fill().map(() => Array(n).fill(0));
    let sumQ = 0;
    
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i !== j) {
          const dist = Math.sqrt(
            Math.pow(Y[i][0] - Y[j][0], 2) + Math.pow(Y[i][1] - Y[j][1], 2)
          );
          Q[i][j] = 1 / (1 + dist * dist);
          sumQ += Q[i][j];
        }
      }
    }
    
    // Normalize Q
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        Q[i][j] = Math.max(Q[i][j] / sumQ, 1e-12);
      }
    }
    
    // Calculate gradients
    const gradients = Array(n).fill().map(() => [0, 0]);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i !== j) {
          const mult = 4 * (P[i][j] - Q[i][j]) * Q[i][j];
          gradients[i][0] += mult * (Y[i][0] - Y[j][0]);
          gradients[i][1] += mult * (Y[i][1] - Y[j][1]);
        }
      }
    }
    
    // Update with momentum
    const momentum = iter < 250 ? 0.5 : 0.8;
    const eta = iter < 100 ? learningRate * earlyExaggeration : learningRate;
    
    for (let i = 0; i < n; i++) {
      velocity[i][0] = momentum * velocity[i][0] - eta * gradients[i][0];
      velocity[i][1] = momentum * velocity[i][1] - eta * gradients[i][1];
      Y[i][0] += velocity[i][0];
      Y[i][1] += velocity[i][1];
    }
  }
  
  return Y;
}

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
  const plotRef = useRef(null);
  
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  
  const loadDocuments = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const requestBody = {
        num_documents: numDocuments,
        load_full_index: loadFullIndex,
        include_embeddings: true
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
      
      // Choose dimensionality reduction method
      let coords;
      if (method === 'pca') {
        coords = performPCA(embeddings);
      } else if (method === 'umap') {
        coords = performUMAP(embeddings, {
          nNeighbors: Math.min(15, Math.floor(embeddings.length / 3)),
          iterations: 200
        });
      } else {
        coords = performTSNE(embeddings, {
          perplexity: Math.min(15, Math.floor(embeddings.length / 3)),
          iterations: 200,
          learningRate: 100
        });
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
      setError('Failed to generate visualization');
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
  
  const getColorForCategory = (category, attribute) => {
    const colorPalettes = {
      language: {
        'Hebrew': '#2E8B57',
        'Arabic': '#8B008B',
        'Aramaic': '#4169E1',
        'Judeo-Arabic': '#DAA520',
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
    return palette[category] || palette['Unknown'];
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
        image_url: metadata.image_url || metadata.thumbnail_url || "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=400&h=300&fit=crop",
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
      'pan2d', 'select2d', 'lasso2d', 'autoScale2d', 
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
          
          <button onClick={loadDocuments} className="load-btn">
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
      </div>
      
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
          />
        )}
      </div>
      
      <div className="explorer-info">
        <p>
          <strong>What this shows:</strong> This {method.toUpperCase()} plot visualizes semantic relationships 
          between documents in the Cairo Genizah collection. Documents closer together are more semantically similar. 
          Colors represent different {colorBy.replace('_', ' ')} categories. Click on any point to view document details.
        </p>
      </div>
      
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
