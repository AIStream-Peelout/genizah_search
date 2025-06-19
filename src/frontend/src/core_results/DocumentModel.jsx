import React from 'react';

const DocumentModal = ({ document, isOpen, onClose }) => {
    if (!isOpen || !document) return null;

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                <div className="modal-header">
                    <div>
                        <h2>{document.title}</h2>
                        {document.shelfmark && <p className="modal-subtitle">{document.shelfmark}</p>}
                    </div>
                    <button className="modal-close" onClick={onClose}>×</button>
                </div>

                <div className="modal-body">
                    <div className="modal-image-section">
                        <img
                            src={document.image_url}
                            alt={document.title}
                            className="modal-image"
                            onError={(e) => {
                                e.target.src = "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=800&h=600&fit=crop";
                            }}
                        />
                        <div className="document-details">
                            <h4>Document Details</h4>
                            <div className="details-grid">
                                {document.date && <div><strong>Date:</strong> {document.date}</div>}
                                {document.language && <div><strong>Language:</strong> {document.language}</div>}
                                {document.material && <div><strong>Material:</strong> {document.material}</div>}
                                {document.dimensions && <div><strong>Dimensions:</strong> {document.dimensions}</div>}
                                {document.location && <div><strong>Location:</strong> {document.location}</div>}
                                {document.period && <div><strong>Period:</strong> {document.period}</div>}
                            </div>
                        </div>
                    </div>

                    <div className="modal-text-section">
                        {document.description && (
                            <div className="modal-section">
                                <h4>Description</h4>
                                <p>{document.description}</p>
                            </div>
                        )}

                        {(document.institution || document.collection) && (
                            <div className="modal-section">
                                <h4>Institution & Collection</h4>
                                <p>
                                    {document.institution && <><strong>{document.institution.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</strong><br /></>}
                                    {document.collection && document.collection.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                </p>
                            </div>
                        )}

                        {/* DEBUG - Show what fields we have */}


                        {/* Show actual transcription from metadata */}
                        {(document.metadata?.transcription_full_text || document.transcription_full_text || document.transcription || document.transcription_text) && (
                            <div className="modal-section">
                                <h4>Transcription</h4>
                                <div className="transcription-text">
                                    {document.metadata?.transcription_full_text || document.transcription_full_text || document.transcription || document.transcription_text}
                                </div>
                            </div>
                        )}

                        {/* Show actual translation from metadata */}
                        {(document.metadata?.translation_full_text || document.translation_full_text || document.translation || document.translation_text) && (
                            <div className="modal-section">
                                <h4>Translation</h4>
                                <div className="translation-text">
                                    {document.metadata?.translation_full_text || document.translation_full_text || document.translation || document.translation_text}
                                </div>
                            </div>
                        )}

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
                    </div>
                </div>
            </div>
        </div>
    );
};

export default DocumentModal;