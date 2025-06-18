import React from 'react';

const SearchResults = ({ results, loading, query, processingTime, onDocumentClick }) => {
    if (loading) {
        return (
            <div className="results-section">
                <div className="loading">
                    <div className="spinner"></div>
                    <span>Searching through historical manuscripts...</span>
                </div>
            </div>
        );
    }

    if (!results) return null;

    return (
        <div className="results-section">
            <div className="results-header">
                <h3>Search Results for: "{query}"</h3>
                <div className="results-meta">
                    <span>{results.count} results found</span>
                    {processingTime && <span>({processingTime}ms)</span>}
                </div>
            </div>

            {results.count === 0 ? (
                <div className="no-results">
                    <p>No results found. Try different search terms or adjust your filters.</p>
                    <div className="search-tips">
                        <h4>Search Tips:</h4>
                        <ul>
                            <li>Try broader terms like "marriage", "religious", or "legal"</li>
                            <li>Use different language filters</li>
                            <li>Remove some filters to expand results</li>
                            <li>Try historical terms like "ketubah", "responsa", or "halakhic"</li>
                        </ul>
                    </div>
                </div>
            ) : (
                <div className="results-grid">
                    {results.results.map((result, index) => {
                        const metadata = result.metadata || {};
                        const fallbackData = {
                            title: `Document ${result.doc_id}`,
                            description: "Historical manuscript from the Cairo Genizah collection.",
                            image_url: "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=400&h=300&fit=crop",
                            date: "Unknown",
                            language: "Hebrew",
                            material: "Parchment"
                        };

                        const displayData = {
                            title: metadata.title || fallbackData.title,
                            description: metadata.description || fallbackData.description,
                            image_url: metadata.image_url || metadata.thumbnail_url || fallbackData.image_url,
                            date: metadata.date || fallbackData.date,
                            language: metadata.language || fallbackData.language,
                            material: metadata.material || fallbackData.material,
                            institution: metadata.institution,
                            collection: metadata.collection,
                            shelfmark: metadata.shelfmark,
                            transcription: metadata.transcription,
                            translation: metadata.translation,
                            tags: metadata.tags,
                            period: metadata.period,
                            location: metadata.location,
                            dimensions: metadata.dimensions,
                            document_type: metadata.document_type
                        };

                        return (
                            <div
                                key={result.doc_id}
                                className="result-item enhanced-result-item"
                                onClick={() => onDocumentClick({ ...displayData, doc_id: result.doc_id, ...result })}
                            >
                                <div className="result-image">
                                    <img
                                        src={displayData.image_url}
                                        alt={displayData.title}
                                        onError={(e) => {
                                            e.target.src = "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=400&h=300&fit=crop";
                                        }}
                                    />
                                    <div className="similarity-badge">
                                        {(result.similarity_score * 100).toFixed(1)}% match
                                    </div>
                                </div>

                                <div className="result-content">
                                    <h4 className="result-title">{displayData.title}</h4>
                                    <p className="result-description">{displayData.description}</p>

                                    <div className="result-metadata">
                                        <span className="metadata-item">{displayData.date}</span>
                                        <span className="metadata-item">{displayData.language}</span>
                                        <span className="metadata-item">{displayData.material}</span>
                                    </div>

                                    {displayData.shelfmark && (
                                        <div className="result-shelfmark">{displayData.shelfmark}</div>
                                    )}

                                    <div className="result-footer">
                                        <div className="result-id">ID: {result.doc_id}</div>
                                        <div className="view-details">View Details →</div>
                                    </div>

                                    <div className="similarity-progress">
                                        <div
                                            className="progress-fill"
                                            style={{ width: `${result.similarity_score * 100}%` }}
                                        ></div>
                                    </div>
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
};

export default SearchResults;