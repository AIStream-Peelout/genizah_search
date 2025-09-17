import React from 'react';

const DocumentModal = ({ document, isOpen, onClose }) => {
    if (!isOpen || !document) return null;

    // Helper function to format transcriptions properly (handles arrays and strings)
    const formatTranscription = (transcription) => {
        if (!transcription) return null;
        
        // If it's an array, join with proper formatting
        if (Array.isArray(transcription)) {
            return transcription.map((text, index) => (
                <div key={index} className="transcription-section">
                    {transcription.length > 1 && <h6>Transcription {index + 1}</h6>}
                    <div className="transcription-text" dir="auto">{text}</div>
                    {index < transcription.length - 1 && <hr className="transcription-separator" />}
                </div>
            ));
        }
        
        // If it's a string, preserve line breaks and handle RTL text
        return (
            <div className="transcription-text" dir="auto">
                {transcription.split('\n').map((line, index) => (
                    <div key={index} className="transcription-line">{line || '\u00A0'}</div>
                ))}
            </div>
        );
    };

    // Helper function to format bibliography
    const formatBibliography = (bibliography) => {
        if (!bibliography || bibliography.length === 0) return null;
        
        return (
            <div className="bibliography-list">
                {bibliography.map((item, index) => (
                    <div key={index} className="bibliography-item">
                        <span className="bibliography-number">{index + 1}.</span>
                        <span className="bibliography-text">{item}</span>
                    </div>
                ))}
            </div>
        );
    };

    // Get metadata from document
    const metadata = document.metadata || document;

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                <div className="modal-header">
                    <div>
                        <h2>{document.title}</h2>
                        {/* Enhanced shelf mark display */}
                        {(metadata.shelf_mark || document.shelfmark) && (
                            <div className="modal-shelf-mark">
                                <strong>Shelf Mark:</strong> {metadata.shelf_mark || document.shelfmark}
                            </div>
                        )}
                        {/* Original source link */}
                        {metadata.original_url && (
                            <div className="modal-source-link">
                                <a 
                                    href={metadata.original_url} 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                    className="original-source-btn"
                                >
                                    🔗 View Original Source
                                </a>
                            </div>
                        )}
                    </div>
                    <button className="modal-close" onClick={onClose}>×</button>
                </div>

                <div className="modal-body">
                    <div className="modal-image-section">
                        <img
                            src={metadata.actual_image_url || document.image_url}
                            alt={document.title}
                            className="modal-image"
                            onError={(e) => {
                                e.target.src = "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=800&h=600&fit=crop";
                            }}
                        />
                        
                        {/* Enhanced document details */}
                        <div className="document-details">
                            <h4>Document Details</h4>
                            <div className="details-grid">
                                {document.date && <div><strong>Date:</strong> {document.date}</div>}
                                {(metadata.language || document.language) && (
                                    <div><strong>Language:</strong> {metadata.language || document.language}</div>
                                )}
                                {metadata.main_language && metadata.main_language !== metadata.language && (
                                    <div><strong>Main Language:</strong> {metadata.main_language}</div>
                                )}
                                {metadata.script_type && (
                                    <div><strong>Script:</strong> {metadata.script_type}</div>
                                )}
                                {(metadata.material || document.material) && (
                                    <div><strong>Material:</strong> {metadata.material || document.material}</div>
                                )}
                                {(metadata.dimensions || document.dimensions) && (
                                    <div><strong>Dimensions:</strong> {metadata.dimensions || document.dimensions}</div>
                                )}
                                {metadata.condition && (
                                    <div><strong>Condition:</strong> {metadata.condition}</div>
                                )}
                                {metadata.extent && (
                                    <div><strong>Extent:</strong> {metadata.extent}</div>
                                )}
                                {(metadata.location || document.location) && (
                                    <div><strong>Location:</strong> {metadata.location || document.location}</div>
                                )}
                                {(metadata.period || document.period) && (
                                    <div><strong>Period:</strong> {metadata.period || document.period}</div>
                                )}
                                {metadata.date_certainty && (
                                    <div><strong>Date Certainty:</strong> {metadata.date_certainty}</div>
                                )}
                                {metadata.document_type && (
                                    <div><strong>Document Type:</strong> {metadata.document_type}</div>
                                )}
                            </div>

                            {/* Quality indicators */}
                            {(metadata.completeness_score || metadata.content_quality) && (
                                <div className="quality-section">
                                    <h5>Quality Indicators</h5>
                                    {metadata.completeness_score && (
                                        <div className="quality-item">
                                            <strong>Completeness:</strong> {(metadata.completeness_score * 100).toFixed(0)}%
                                        </div>
                                    )}
                                    {metadata.content_quality && (
                                        <div className="quality-item">
                                            <strong>Content Quality:</strong> 
                                            <span className={`quality-badge quality-${metadata.content_quality}`}>
                                                {metadata.content_quality}
                                            </span>
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="modal-text-section">
                        {document.description && (
                            <div className="modal-section">
                                <h4>Description</h4>
                                <p>{document.description}</p>
                            </div>
                        )}

                        {/* Enhanced institution & collection */}
                        {(document.institution || document.collection || metadata.repository || metadata.source_collection) && (
                            <div className="modal-section">
                                <h4>Institution & Collection</h4>
                                <div className="institution-details">
                                    {(metadata.repository || document.institution) && (
                                        <div><strong>Institution:</strong> {(metadata.repository || document.institution).replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</div>
                                    )}
                                    {metadata.library && (
                                        <div><strong>Library:</strong> {metadata.library}</div>
                                    )}
                                    {(metadata.source_collection || document.collection) && (
                                        <div><strong>Collection:</strong> {(metadata.source_collection || document.collection).replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</div>
                                    )}
                                    {metadata.collection_type && (
                                        <div><strong>Collection Type:</strong> {metadata.collection_type}</div>
                                    )}
                                    {metadata.provenance && (
                                        <div><strong>Provenance:</strong> {metadata.provenance}</div>
                                    )}
                                </div>
                            </div>
                        )}

                        {/* Enhanced transcription display */}
                        {(document.metadata?.transcription_full_text || document.transcription_full_text || document.transcription || document.transcription_text || metadata.transcriptions) && (
                            <div className="modal-section">
                                <h4>Transcription</h4>
                                <div className="transcription-container">
                                    {formatTranscription(
                                        metadata.transcriptions ||
                                        document.metadata?.transcription_full_text || 
                                        document.transcription_full_text || 
                                        document.transcription || 
                                        document.transcription_text
                                    )}
                                </div>
                            </div>
                        )}

                        {/* Enhanced translation display */}
                        {(document.metadata?.translation_full_text || document.translation_full_text || document.translation || document.translation_text || metadata.translations) && (
                            <div className="modal-section">
                                <h4>Translation</h4>
                                <div className="translation-container">
                                    {formatTranscription(
                                        metadata.translations ||
                                        document.metadata?.translation_full_text || 
                                        document.translation_full_text || 
                                        document.translation || 
                                        document.translation_text
                                    )}
                                </div>
                            </div>
                        )}

                        {/* NEW: Bibliography section */}
                        {metadata.bibliography && metadata.bibliography.length > 0 && (
                            <div className="modal-section bibliography-section">
                                <h4>Bibliography</h4>
                                {formatBibliography(metadata.bibliography)}
                            </div>
                        )}

                        {/* NEW: Named entities section */}
                        {metadata.named_entities && (
                            <div className="modal-section">
                                <h4>Named Entities</h4>
                                <div className="named-entities-container">
                                    {metadata.named_entities.persons && metadata.named_entities.persons.length > 0 && (
                                        <div className="entity-group">
                                            <strong>Persons:</strong>
                                            <div className="entity-tags">
                                                {metadata.named_entities.persons.map((person, index) => (
                                                    <span key={index} className="entity-tag person-tag">{person}</span>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                    {metadata.named_entities.places && metadata.named_entities.places.length > 0 && (
                                        <div className="entity-group">
                                            <strong>Places:</strong>
                                            <div className="entity-tags">
                                                {metadata.named_entities.places.map((place, index) => (
                                                    <span key={index} className="entity-tag place-tag">{place}</span>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                    {metadata.named_entities.organizations && metadata.named_entities.organizations.length > 0 && (
                                        <div className="entity-group">
                                            <strong>Organizations:</strong>
                                            <div className="entity-tags">
                                                {metadata.named_entities.organizations.map((org, index) => (
                                                    <span key={index} className="entity-tag org-tag">{org}</span>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                    {metadata.named_entities.dates && metadata.named_entities.dates.length > 0 && (
                                        <div className="entity-group">
                                            <strong>Dates:</strong>
                                            <div className="entity-tags">
                                                {metadata.named_entities.dates.map((date, index) => (
                                                    <span key={index} className="entity-tag date-tag">{date}</span>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}

                        {/* Enhanced tags section */}
                        {document.tags && document.tags.length > 0 && (
                            <div className="modal-section">
                                <h4>Tags</h4>
                                <div className="tags-container">
                                    {document.tags.map(tag => (
                                        <span key={tag} className="tag">
                                            {tag}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Enhanced search match section */}
                        <div className="modal-section">
                            <h4>Search Match</h4>
                            <div className="match-score">
                                <span>Relevance Score: </span>
                                <span className="score-value">
                                    {document.similarity_score ? (document.similarity_score * 100).toFixed(1) : 'N/A'}%
                                </span>
                                <div className="score-bar">
                                    <div
                                        className="score-fill"
                                        style={{ width: `${document.similarity_score ? document.similarity_score * 100 : 0}%` }}
                                    ></div>
                                </div>
                            </div>
                        </div>

                        {/* Technical metadata */}
                        {(metadata.indexed_at || metadata.transcription_count || metadata.translation_count) && (
                            <div className="modal-section technical-section">
                                <h4>Technical Information</h4>
                                <div className="technical-details">
                                    {metadata.transcription_count && (
                                        <div><strong>Transcription Count:</strong> {metadata.transcription_count}</div>
                                    )}
                                    {metadata.translation_count && (
                                        <div><strong>Translation Count:</strong> {metadata.translation_count}</div>
                                    )}
                                    {metadata.total_transcription_lines && (
                                        <div><strong>Total Lines:</strong> {metadata.total_transcription_lines}</div>
                                    )}
                                    {metadata.indexed_at && (
                                        <div><strong>Indexed:</strong> {new Date(metadata.indexed_at).toLocaleDateString()}</div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default DocumentModal;