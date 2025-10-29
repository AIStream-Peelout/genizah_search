// CollectionBrowser.jsx - Browse documents by collection hierarchy
import React, { useState, useEffect } from 'react';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const CollectionBrowser = ({ onSelectShelfmark, isVisible }) => {
  const [hierarchy, setHierarchy] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedCollections, setExpandedCollections] = useState(new Set());
  const [expandedSubCollections, setExpandedSubCollections] = useState(new Set());
  const [selectedShelfmark, setSelectedShelfmark] = useState(null);
  const [selectedIndex, setSelectedIndex] = useState('');
  const [availableIndices, setAvailableIndices] = useState([]);

  // Load indices on mount.
  useEffect(() => {
    loadIndices();
  }, []);

  // Load hierarchy when index changes
  useEffect(() => {
    if (selectedIndex) {
      loadHierarchy();
    }
  }, [selectedIndex]);

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

  const loadHierarchy = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/collection-hierarchy?index_name=${selectedIndex}`);
      if (response.ok) {
        const data = await response.json();
        console.log('Collection hierarchy data:', data);
        const hierarchyData = data.hierarchy || {};
        
        // Debug: log hierarchy structure
        Object.entries(hierarchyData).forEach(([colName, col]) => {
          console.log(`Collection: ${colName}`, {
            count: col.count,
            sub_collections: Object.keys(col.sub_collections || {}),
            sub_collections_count: Object.keys(col.sub_collections || {}).length
          });
        });
        
        setHierarchy(hierarchyData);
      } else {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        setError(errorData.detail || 'Failed to load collection hierarchy');
      }
    } catch (err) {
      console.error('Failed to load hierarchy:', err);
      setError('Network error while loading collection hierarchy');
    } finally {
      setLoading(false);
    }
  };

  const toggleCollection = (collectionName) => {
    const newExpanded = new Set(expandedCollections);
    if (newExpanded.has(collectionName)) {
      newExpanded.delete(collectionName);
    } else {
      newExpanded.add(collectionName);
    }
    setExpandedCollections(newExpanded);
  };

  const toggleSubCollection = (collectionName, subCollectionName) => {
    const key = `${collectionName}:${subCollectionName}`;
    const newExpanded = new Set(expandedSubCollections);
    if (newExpanded.has(key)) {
      newExpanded.delete(key);
    } else {
      newExpanded.add(key);
      // If shelfmarks for this sub-collection are not loaded yet, fetch them
      const sub = hierarchy[collectionName]?.sub_collections?.[subCollectionName];
      if (sub && (!sub.shelfmarks || sub.shelfmarks.length === 0)) {
        fetchShelfmarks(collectionName, subCollectionName);
      }
    }
    setExpandedSubCollections(newExpanded);
  };

  const fetchShelfmarks = async (collectionName, subCollectionName) => {
    try {
      const params = new URLSearchParams({
        collection: collectionName,
        sub_collection: subCollectionName,
      });
      if (selectedIndex) params.append('index_name', selectedIndex);
      const resp = await fetch(`${API_BASE_URL}/collection-shelfmarks?${params.toString()}`);
      if (!resp.ok) return;
      const data = await resp.json();
      const shelfmarks = data.shelfmarks || [];
      // Inject into local hierarchy state
      setHierarchy(prev => {
        const copy = { ...prev };
        if (!copy[collectionName] || !copy[collectionName].sub_collections) return prev;
        const sub = copy[collectionName].sub_collections[subCollectionName];
        if (sub) {
          sub.shelfmarks = shelfmarks;
        }
        return copy;
      });
    } catch (e) {
      console.error('Failed to fetch shelfmarks:', e);
    }
  };

  const handleShelfmarkClick = (shelfmark) => {
    setSelectedShelfmark(shelfmark);
    if (onSelectShelfmark) {
      onSelectShelfmark(shelfmark);
    }
  };

  if (!isVisible) {
    return null;
  }

  return (
    <div className="collection-browser">
      <div className="browser-header">
        <h3>📚 Browse by Collection</h3>
        <div className="index-selector">
          <label>
            Index:
            <select
              value={selectedIndex}
              onChange={(e) => setSelectedIndex(e.target.value)}
              className="index-select"
            >
              {availableIndices.map((idx) => (
                <option key={idx.name} value={idx.name}>
                  {idx.name}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      {loading && (
        <div className="loading-message">
          <div className="spinner"></div>
          <p>Loading collections...</p>
        </div>
      )}

      {error && (
        <div className="error-message">
          <span>⚠️ {error}</span>
          <button onClick={loadHierarchy} className="retry-btn">
            Retry
          </button>
        </div>
      )}

      {!loading && !error && Object.keys(hierarchy).length > 0 && (
        <div className="hierarchy-tree">
          {Object.entries(hierarchy).map(([collectionName, collection]) => (
            <div key={collectionName} className="collection-group">
              <div
                className="collection-header clickable"
                onClick={() => toggleCollection(collectionName)}
              >
                <span className="toggle-icon">
                  {expandedCollections.has(collectionName) ? '▼' : '▶'}
                </span>
                <span className="collection-name">{collectionName}</span>
                <span className="doc-count">({collection.count} docs)</span>
              </div>

              {expandedCollections.has(collectionName) && (
                <div className="sub-collections">
                  {Object.keys(collection.sub_collections || {}).length > 0 ? (
                    Object.entries(collection.sub_collections || {}).map(([subCollectionName, subCollection]) => (
                      <div key={subCollectionName} className="sub-collection-group">
                        <div
                          className="sub-collection-header clickable"
                          onClick={() => toggleSubCollection(collectionName, subCollectionName)}
                        >
                          <span className="toggle-icon">
                            {expandedSubCollections.has(`${collectionName}:${subCollectionName}`) ? '▼' : '▶'}
                          </span>
                          <span className="sub-collection-name">
                            {subCollection.name || subCollectionName}
                          </span>
                          <span className="doc-count">({subCollection.count} docs)</span>
                        </div>

                        {expandedSubCollections.has(`${collectionName}:${subCollectionName}`) && (
                          <div className="shelfmarks">
                            {subCollection.shelfmarks && subCollection.shelfmarks.length > 0 ? (
                              subCollection.shelfmarks.map((shelfmark, idx) => (
                                <div
                                  key={idx}
                                  className={`shelfmark-item clickable ${selectedShelfmark === shelfmark.name ? 'selected' : ''}`}
                                  onClick={() => handleShelfmarkClick(shelfmark.name)}
                                  title={`Click to add ${shelfmark.count} document(s) to visualization`}
                                >
                                  <span className="shelfmark-name">{shelfmark.name}</span>
                                  <span className="doc-count">({shelfmark.count} docs)</span>
                                </div>
                              ))
                            ) : (
                              <div className="empty-state-small">
                                <p>No shelfmarks found</p>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    ))
                  ) : (
                    <div className="empty-state-small">
                      <p>No sub-collections found for this collection.</p>
                      <p className="help-text">This collection may not have sub-collection metadata, or the data may be structured differently.</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {!loading && !error && Object.keys(hierarchy).length === 0 && (
        <div className="empty-state">
          <p>No collections found. Try selecting a different index.</p>
        </div>
      )}

      <style jsx>{`
        .collection-browser {
          background: white;
          border-radius: 8px;
          border: 1px solid #E5E5E5;
          overflow: hidden;
          margin: 20px 0;
        }

        .browser-header {
          padding: 16px;
          background: #F8F9FA;
          border-bottom: 1px solid #E5E5E5;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .browser-header h3 {
          margin: 0;
          font-size: 18px;
          color: #2C3E50;
        }

        .index-selector {
          display: flex;
          align-items: center;
          gap: 8px;
        }

        .index-selector label {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 14px;
          color: #2C3E50;
        }

        .index-select {
          padding: 6px 12px;
          border: 1px solid #DDD;
          border-radius: 4px;
          background: white;
          font-size: 13px;
          cursor: pointer;
        }

        .loading-message,
        .error-message {
          padding: 40px;
          text-align: center;
        }

        .loading-message {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 16px;
        }

        .spinner {
          width: 32px;
          height: 32px;
          border: 3px solid #E5E5E5;
          border-top: 3px solid #3498DB;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }

        .error-message {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 12px;
          color: #E74C3C;
        }

        .retry-btn {
          padding: 4px 8px;
          background: #E74C3C;
          color: white;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 12px;
        }

        .retry-btn:hover {
          background: #C0392B;
        }

        .hierarchy-tree {
          max-height: 600px;
          overflow-y: auto;
          padding: 16px;
        }

        .collection-group {
          margin-bottom: 8px;
        }

        .collection-header,
        .sub-collection-header,
        .shelfmark-item {
          padding: 8px 12px;
          display: flex;
          align-items: center;
          gap: 8px;
          border-radius: 4px;
          transition: background-color 0.2s;
        }

        .clickable {
          cursor: pointer;
        }

        .collection-header.clickable:hover,
        .sub-collection-header.clickable:hover {
          background: #F8F9FA;
        }

        .shelfmark-item {
          margin-left: 20px;
          padding: 6px 12px;
          font-size: 13px;
        }

        .shelfmark-item.clickable:hover {
          background: #E8F4FD;
        }

        .shelfmark-item.selected {
          background: #D6EAF8;
          border-left: 3px solid #3498DB;
        }

        .toggle-icon {
          color: #3498DB;
          font-size: 10px;
          width: 16px;
          display: inline-block;
        }

        .collection-name {
          font-weight: 600;
          color: #2C3E50;
          font-size: 15px;
        }

        .sub-collection-name {
          font-weight: 500;
          color: #34495E;
          font-size: 14px;
        }

        .shelfmark-name {
          color: #555;
          font-family: monospace;
          font-size: 13px;
        }

        .doc-count {
          color: #7F8C8D;
          font-size: 12px;
        }

        .sub-collections {
          margin-left: 24px;
          margin-top: 4px;
        }

        .sub-collection-group {
          margin-bottom: 4px;
        }

        .shelfmarks {
          margin-left: 24px;
          margin-top: 4px;
          max-height: 260px; /* show roughly first ~10 items before scroll */
          overflow-y: auto;
          border-left: 2px solid #EEF2F4;
          padding-left: 12px;
          background: #FBFCFD;
        }

        .empty-state {
          padding: 40px;
          text-align: center;
          color: #7F8C8D;
        }

        .empty-state-small {
          padding: 16px;
          margin-left: 24px;
          background: #F8F9FA;
          border-radius: 4px;
          color: #7F8C8D;
          font-size: 13px;
        }

        .empty-state-small p {
          margin: 4px 0;
        }

        .help-text {
          font-size: 12px;
          font-style: italic;
          color: #95A5A6;
        }

        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default CollectionBrowser;


