import React from 'react';

const DocumentModal = ({ document, isOpen, onClose }) => {
    if (!isOpen || !document) return null;

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-xl max-w-4xl max-h-[90vh] overflow-y-auto w-full">
                <div className="sticky top-0 bg-white border-b border-gray-200 p-6 flex justify-between items-start">
                    <div>
                        <h2 className="text-2xl font-bold text-gray-900">{document.title}</h2>
                        {document.shelfmark && <p className="text-gray-600 mt-1">{document.shelfmark}</p>}
                    </div>
                    <button
                        onClick={onClose}
                        className="text-gray-400 hover:text-gray-600 text-xl font-bold"
                    >
                        ×
                    </button>
                </div>

                <div className="p-6">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                        <div>
                            <img
                                src={document.image_url}
                                alt={document.title}
                                className="w-full h-auto rounded-lg shadow-lg"
                                onError={(e) => {
                                    e.target.src = "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=800&h=600&fit=crop";
                                }}
                            />
                            <div className="mt-4 bg-gray-50 rounded-lg p-4">
                                <h4 className="font-semibold text-gray-900 mb-2">Document Details</h4>
                                <div className="grid grid-cols-2 gap-2 text-sm">
                                    {document.date && <div><strong>Date:</strong> {document.date}</div>}
                                    {document.language && <div><strong>Language:</strong> {document.language}</div>}
                                    {document.material && <div><strong>Material:</strong> {document.material}</div>}
                                    {document.dimensions && <div><strong>Dimensions:</strong> {document.dimensions}</div>}
                                    {document.location && <div><strong>Location:</strong> {document.location}</div>}
                                    {document.period && <div><strong>Period:</strong> {document.period}</div>}
                                </div>
                            </div>
                        </div>

                        <div className="space-y-6">
                            {document.description && (
                                <div>
                                    <h4 className="font-semibold text-gray-900 mb-2">Description</h4>
                                    <p className="text-gray-700 leading-relaxed">{document.description}</p>
                                </div>
                            )}

                            {(document.institution || document.collection) && (
                                <div>
                                    <h4 className="font-semibold text-gray-900 mb-2">Institution & Collection</h4>
                                    <p className="text-gray-700">
                                        {document.institution && <><strong>{document.institution.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</strong><br /></>}
                                        {document.collection && document.collection.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                    </p>
                                </div>
                            )}

                            {document.transcription && (
                                <div>
                                    <h4 className="font-semibold text-gray-900 mb-2">Transcription</h4>
                                    <div className="bg-gray-50 rounded-lg p-4 font-mono text-sm text-gray-800">
                                        {document.transcription}
                                    </div>
                                </div>
                            )}

                            {document.translation && (
                                <div>
                                    <h4 className="font-semibold text-gray-900 mb-2">Translation</h4>
                                    <div className="bg-blue-50 rounded-lg p-4 text-gray-800">
                                        {document.translation}
                                    </div>
                                </div>
                            )}

                            {document.tags && document.tags.length > 0 && (
                                <div>
                                    <h4 className="font-semibold text-gray-900 mb-2">Tags</h4>
                                    <div className="flex flex-wrap gap-2">
                                        {document.tags.map(tag => (
                                            <span key={tag} className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                        {tag}
                      </span>
                                        ))}
                                    </div>
                                </div>
                            )}

                            <div>
                                <h4 className="font-semibold text-gray-900 mb-2">Search Match</h4>
                                <div className="bg-green-50 rounded-lg p-4">
                                    <div className="flex justify-between items-center mb-2">
                                        <span className="text-sm text-gray-700">Relevance Score:</span>
                                        <span className="font-semibold text-green-700">
                      {document.similarity_score ? (document.similarity_score * 100).toFixed(1) : 'N/A'}%
                    </span>
                                    </div>
                                    <div className="bg-gray-200 rounded-full h-2">
                                        <div
                                            className="bg-green-500 h-2 rounded-full"
                                            style={{ width: `${document.similarity_score ? document.similarity_score * 100 : 0}%` }}
                                        ></div>
                                    </div>
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