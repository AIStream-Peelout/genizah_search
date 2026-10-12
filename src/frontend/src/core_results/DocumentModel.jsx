import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { normalizeDocId } from '../utils';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
import DocumentDetailView from './DocumentDetailView';
import SecondarySourceView from './SecondarySourceView';
import BibliographyDetail from './BibliographyDetail';

// Helper function to format transcriptions properly (handles arrays, strings, and objects)
const formatTranscription = (transcription) => {
    if (!transcription) return null;

    // DEBUG: Log transcription data structure
    console.log('=== TRANSCRIPTION DEBUG ===');
    console.log('Type:', typeof transcription);
    console.log('Is Array:', Array.isArray(transcription));
    console.log('Data:', transcription);
    console.log('Keys (if object):', typeof transcription === 'object' ? Object.keys(transcription) : 'N/A');
    console.log('========================');

    // If it's an array, handle each item
    if (Array.isArray(transcription)) {
        return transcription.map((item, index) => {
            let text = item;

            console.log(`Array item ${index}:`, typeof item, item);

            // If array item is an object, extract the text
            if (typeof item === 'object' && item !== null) {
                text = item.text || item.content || item.transcription || JSON.stringify(item);
                console.log(`Extracted text from object:`, text);
            }

            return (
                <div key={index} className="transcription-section">
                    {transcription.length > 1 && <h6>Transcription {index + 1}</h6>}
                    <div className="transcription-text" dir="auto">{String(text)}</div>
                    {index < transcription.length - 1 && <hr className="transcription-separator" />}
                </div>
            );
        });
    }

    // If it's an object, extract the text property
    if (typeof transcription === 'object' && transcription !== null) {
        const text = transcription.text || transcription.content || transcription.transcription || JSON.stringify(transcription);
        console.log('Extracted text from single object:', text);
        return (
            <div className="transcription-text" dir="auto">
                {String(text).split('\n').map((line, index) => (
                    <div key={index} className="transcription-line">{line || '\u00A0'}</div>
                ))}
            </div>
        );
    }

    // If it's a string, preserve line breaks and handle RTL text
    console.log('Processing as string:', transcription);
    return (
        <div className="transcription-text" dir="auto">
            {String(transcription).split('\n').map((line, index) => (
                <div key={index} className="transcription-line">{line || '\u00A0'}</div>
            ))}
        </div>
    );
};

/**
 * Render a catalogue bibliography string, honouring its <em>/<i> italics and
 * dropping any other markup, so entries like "<em>Fatimid Decrees</em>" show
 * as italics instead of literal tags. No HTML is injected into the DOM.
 * @param {string} text - Bibliography entry as stored in the index.
 * @returns {React.ReactNode[]} Text and <em> nodes.
 */
const renderBibliographyText = (text) => {
    const s = String(text ?? '');
    const parts = s.split(/(<\/?(?:em|i)>)/i);
    const out = [];
    let italic = false;
    parts.forEach((part, i) => {
        if (/^<(?:em|i)>$/i.test(part)) { italic = true; return; }
        if (/^<\/(?:em|i)>$/i.test(part)) { italic = false; return; }
        const clean = part.replace(/<[^>]+>/g, '');
        if (!clean) return;
        out.push(italic ? <em key={i}>{clean}</em> : <React.Fragment key={i}>{clean}</React.Fragment>);
    });
    return out;
};

/**
 * Providers attributed on the page, in display order. The backend only ever
 * sends ``ktiv`` or ``pgp`` as a source; every other citation arrives with no
 * source and is listed under a neutral heading with no badge.
 */
const BIBLIOGRAPHY_SOURCES = [
    { key: 'ktiv', label: 'KTIV', full: 'KTIV, National Library of Israel' },
    { key: 'pgp', label: 'PGP', full: 'Princeton Geniza Project' },
    { key: 'other', label: null, full: 'Further references' },
];

/** How a citation relates to the fragment (KTIV vocabulary; other entries use free strings). */
const RELATION_HINTS = {
    Mention: 'The work mentions this fragment',
    Discussion: 'The work discusses this fragment',
    Image: 'The work reproduces an image of this fragment',
};

/**
 * One structured citation: authors, italic title, year, pages, relation tags.
 * Falls back to the raw citation string when the index has no parsed title.
 * When the entry has a ``title``, the citation itself becomes a button that
 * opens the "work detail" panel (``onOpenWork``) for it; a safe ``url`` gets
 * its own small external-link icon alongside, rather than wrapping the whole
 * citation, since the two actions (view details vs. leave the site) differ.
 * @param {{entry: object, onOpenWork?: (entry: object) => void}} props -
 *   Entry from ``metadata.bibliography_entries``, and the handler that opens
 *   its work-detail panel.
 */
function BibliographyEntry({ entry, onOpenWork }) {
    const authors = (entry.authors || []).join('; ');
    // Some providers store the location with its own "p." / "pp." prefix already.
    const pages = entry.location
        ? (/^\s*(p{1,2}\.|pages?\b|עמ)/i.test(entry.location) ? entry.location.trim() : `p. ${entry.location}`)
        : null;
    const body = entry.title ? (
        <>
            {authors && <span className="bib-authors">{authors}. </span>}
            <em className="bib-title">{renderBibliographyText(entry.title)}</em>
            {entry.year && <span className="bib-year"> ({entry.year})</span>}
            {pages && <span className="bib-pages">, {pages}</span>}
        </>
    ) : (
        renderBibliographyText(entry.citation)
    );
    const canOpenWork = Boolean(entry.title && onOpenWork);
    return (
        <li className="bib-entry" dir="auto">
            <span className="bib-entry-text">
                {canOpenWork ? (
                    <button
                        type="button"
                        className="bib-entry-link"
                        onClick={() => onOpenWork(entry)}
                        title="View publication details and citing fragments"
                    >
                        {body}
                    </button>
                ) : body}
                {entry.url && (
                    <a
                        className="bib-external-link"
                        href={entry.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        title="Open source link"
                        aria-label="Open source link"
                    >
                        ↗
                    </a>
                )}
            </span>
            {(entry.relations || []).map((rel) => (
                <span key={rel} className="bib-relation" title={RELATION_HINTS[rel] || rel}>{rel}</span>
            ))}
        </li>
    );
}

/**
 * Scholarship panel: catalogue citations grouped by the project that recorded
 * them, each group under a source badge.
 * @param {object[]} entries - ``metadata.bibliography_entries`` from the backend.
 * @param {(entry: object) => void} [onOpenWork] - Opens the work-detail panel
 *   for a clicked citation; forwarded to each ``BibliographyEntry``.
 * @returns {React.ReactNode|null} The grouped list, or null when empty.
 */
const formatBibliographyEntries = (entries, onOpenWork) => {
    if (!entries || entries.length === 0) return null;
    const groups = {};
    entries.forEach((e) => {
        const key = BIBLIOGRAPHY_SOURCES.some((s) => s.key === e.source) ? e.source : 'other';
        (groups[key] = groups[key] || []).push(e);
    });
    return (
        <div className="bib-groups">
            {BIBLIOGRAPHY_SOURCES.filter((s) => groups[s.key]).map((s) => (
                <div key={s.key} className="bib-group">
                    <div className="bib-group-header">
                        {s.label && <span className={`bib-badge bib-badge-${s.key}`}>{s.label}</span>}
                        <span className="bib-group-name">{s.full}</span>
                        <span className="bib-group-count">{groups[s.key].length}</span>
                    </div>
                    <ol className="bib-entries">
                        {groups[s.key].map((e, i) => <BibliographyEntry key={i} entry={e} onOpenWork={onOpenWork} />)}
                    </ol>
                </div>
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
                    <span className="bibliography-text">{renderBibliographyText(item)}</span>
                </div>
            ))}
        </div>
    );
};

const DocumentModal = ({ document, isOpen, onClose, onShelfmarkClick }) => {
    // Image navigation state - MUST be called before any early returns
    const [currentImageIndex, setCurrentImageIndex] = useState(0);

    // Bibliography "work detail" panel: the clicked citation entry, or null
    // when closed. See handleOpenWork / handleOpenFragment below.
    const [activeBibEntry, setActiveBibEntry] = useState(null);

    // Memoize the image list to prevent recalculation on every render
    const allImages = React.useMemo(() => {
        if (!document) return [];

        const metadata = document.metadata || document;

        // Determine index based on index_name field (preferred) or fall back to field detection
        const isLegacyIndex = document.index_name === 'cairo_genizah_text_only_v1.0.6';

        console.log('=== MODAL IMAGE DEBUG ===');
        console.log('document.index_name:', document.index_name);
        console.log('Is legacy index:', isLegacyIndex);
        console.log('Has actual_image_url:', !!metadata.actual_image_url);
        console.log('Has image_urls:', !!metadata.image_urls);
        console.log('image_urls value:', metadata.image_urls);

        // Check if this is a legacy index (cairo_genizah_text_only_v1.0.6)
        // For legacy index, use ONLY actual_image_url
        // For new indices, use image_urls array
        if (isLegacyIndex && metadata.actual_image_url) {
            console.log('✅ Using actual_image_url for legacy index');
            return [metadata.actual_image_url];
        }

        // This is a new index - use image_urls array
        if (metadata.image_urls && Array.isArray(metadata.image_urls) && metadata.image_urls.length > 0) {
            console.log('✅ Processing image_urls array, length:', metadata.image_urls.length);
            // Clean URLs by removing srcset descriptors like "1440w"
            const validUrls = metadata.image_urls
                .filter(url => url && typeof url === 'string' && url.trim())
                .map(url => {
                    const cleaned = url.split(/\s+/)[0]; // Remove width descriptors
                    console.log('Cleaned URL:', url, '->', cleaned);
                    return cleaned;
                })
                .filter(url => url && url.trim() && !url.endsWith('w'));

            console.log('✅ Returning cleaned URLs:', validUrls.length, 'images');
            return validUrls;
        }

        // Fallback for legacy documents without index_name set
        if (metadata.actual_image_url && typeof metadata.actual_image_url === 'string' && metadata.actual_image_url.trim()) {
            console.log('✅ Fallback: Using actual_image_url');
            return [metadata.actual_image_url];
        }

        console.log('❌ No images found');
        return [];
    }, [document]);

    const currentImage = allImages[currentImageIndex] || "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=800&h=600&fit=crop";

    // Reset image index and any open work-detail panel when document changes
    useEffect(() => {
        setCurrentImageIndex(0);
        setActiveBibEntry(null);
    }, [document?.doc_id]);

    /**
     * Open the "work detail" panel for a clicked bibliography citation.
     * @param {object} entry - Entry from ``metadata.bibliography_entries``.
     */
    const handleOpenWork = (entry) => {
        setActiveBibEntry(entry);
    };

    /**
     * Open a fragment cited by the current work-detail panel. Reuses the
     * document modal's existing ``onShelfmarkClick`` wiring (already passed
     * down from react_app.jsx): passing the fragment's own ``doc_id`` as the
     * ``docIds`` hint makes it fetch that exact document instead of running
     * a shelf-mark search. This needs no changes to react_app.jsx and no
     * extra document-fetching logic here.
     * @param {string} docId - ``doc_id`` of the fragment to open.
     */
    const handleOpenFragment = (docId) => {
        if (onShelfmarkClick) {
            onShelfmarkClick(docId, [docId], document?.index_name);
        }
        setActiveBibEntry(null);
    };

    // Offline AI transcription availability (drives the "Transcribe with AI" button).
    const [aiStatus, setAiStatus] = useState(null);
    useEffect(() => {
        setAiStatus(null);
        if (!isOpen || !document?.doc_id) return undefined;
        const controller = new AbortController();
        fetch(`${API_BASE_URL}/ai-transcriptions/${encodeURIComponent(document.doc_id)}`, { signal: controller.signal })
            .then((r) => (r.ok ? r.json() : null))
            .then((s) => setAiStatus(s))
            .catch(() => setAiStatus(null));
        return () => controller.abort();
    }, [isOpen, document?.doc_id]);

    // Navigation functions
    const goToPreviousImage = () => {
        setCurrentImageIndex(prev => prev > 0 ? prev - 1 : allImages.length - 1);
    };

    const goToNextImage = () => {
        setCurrentImageIndex(prev => prev < allImages.length - 1 ? prev + 1 : 0);
    };

    // Keyboard navigation
    useEffect(() => {
        const handleKeyPress = (e) => {
            if (!isOpen) return;
            if (e.key === 'ArrowLeft') {
                goToPreviousImage();
            } else if (e.key === 'ArrowRight') {
                goToNextImage();
            }
        };

        window.addEventListener('keydown', handleKeyPress);
        return () => window.removeEventListener('keydown', handleKeyPress);
    }, [isOpen, allImages.length, currentImageIndex]);

    if (!isOpen || !document) return null;

    // Check if this is a secondary source / bibliography item
    const isSecondarySource = document.index_name && document.index_name.includes('bibliography');

    if (isSecondarySource) {
        return <SecondarySourceView document={document} onClose={onClose} />;
    }

    // Get metadata from document
    const metadata = document.metadata || document;

    // Check if transcription data exists
    const hasTranscription = !!(
        metadata.transcriptions ||
        document.metadata?.transcription_full_text ||
        document.transcription_full_text ||
        document.transcription ||
        document.transcription_text
    );

    // Get the transcription data
    const transcriptionData = metadata.transcriptions ||
        document.metadata?.transcription_full_text ||
        document.transcription_full_text ||
        document.transcription ||
        document.transcription_text;

    // Check if translation data exists
    const hasTranslation = !!(
        metadata.translations ||
        document.metadata?.translation_full_text ||
        document.translation_full_text ||
        document.translation ||
        document.translation_text
    );

    // Get the translation data
    const translationData = metadata.translations ||
        document.metadata?.translation_full_text ||
        document.translation_full_text ||
        document.translation ||
        document.translation_text;

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                {activeBibEntry && (
                    <BibliographyDetail
                        entry={activeBibEntry}
                        indexName={document.index_name}
                        onClose={() => setActiveBibEntry(null)}
                        onOpenDocument={handleOpenFragment}
                    />
                )}
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
                        {aiStatus?.available && (
                            <div className="modal-source-link">
                                <Link
                                    to={`/read?doc=${encodeURIComponent(document.doc_id)}&image=${aiStatus.items[0].image_index}${document.index_name ? `&index=${encodeURIComponent(document.index_name)}` : ''}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="ai-transcribe-btn"
                                    title="Machine transcription with line boxes. Beta: not checked by a person."
                                >
                                    ✨ Transcribe with AI (beta)
                                </Link>
                            </div>
                        )}
                    </div>
                    <button className="modal-close" onClick={onClose}>×</button>
                </div>

                <div className="modal-body">
                    <div className="modal-image-section">
                        <div className="image-container">
                            {/* Use Mirador for advanced viewing if we have a manifest URL (which we construct) */}
                            {/* We construct the manifest URL based on doc_id */}
                            {/* Note: In a real app, we might want to check if the manifest endpoint actually returns 200 first, 
                                but Mirador handles errors gracefully usually. */}

                            <div className="document-viewer-frame">
                                <DocumentDetailView
                                    docId={document.doc_id}
                                    manifestUrl={`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/document/${normalizeDocId(document.doc_id)}/manifest${document.index_name ? `?index_name=${encodeURIComponent(document.index_name)}` : ''}`}
                                />
                            </div>
                        </div>

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
                        {(document.institution || document.collection || metadata.repository || metadata.collection) && (
                            <div className="modal-section">
                                <h4>Institution & Collection</h4>
                                <div className="institution-details">
                                    {(metadata.repository || document.institution) && (
                                        <div><strong>Institution:</strong> {(metadata.repository || document.institution).replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</div>
                                    )}
                                    {metadata.library && (
                                        <div><strong>Library:</strong> {metadata.library}</div>
                                    )}
                                    {(metadata.collection || document.collection) && (
                                        <div><strong>Collection:</strong> {(metadata.collection || document.collection).replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</div>
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

                        {/* Enhanced transcription display - only show if transcription exists */}
                        {hasTranscription && (
                            <div className="modal-section">
                                <h4>Transcription</h4>
                                <div className="transcription-container">
                                    {formatTranscription(transcriptionData)}
                                </div>
                            </div>
                        )}

                        {/* Enhanced translation display - only show if translation exists */}
                        {hasTranslation && (
                            <div className="modal-section">
                                <h4>Translation</h4>
                                <div className="translation-container">
                                    {formatTranscription(translationData)}
                                </div>
                            </div>
                        )}

                        {/* NEW: Bibliography section */}
                        {metadata.bibliography && metadata.bibliography.length > 0 && (
                            <div className="modal-section bibliography-section">
                                <h4>Bibliography</h4>
                                {formatBibliographyEntries(metadata.bibliography_entries, handleOpenWork) || formatBibliography(metadata.bibliography)}
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

                        {/* Search match: only meaningful when the document came from a
                            search; documents opened from the map or a link have no score. */}
                        {document.similarity_score != null && (
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
                        )}

                        {/* Technical metadata */}
                        {(metadata.indexed_at || metadata.transcription_count || metadata.translation_count || metadata.joins_data) && (
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
                                    {metadata.joins_data && (
                                        <div className="joins-data-section">
                                            <h5 className="joins-data-heading">Joins Data</h5>
                                            <div className="joins-data-content">
                                                {metadata.joins_data.mainShelfmark && (
                                                    <div className="joins-main-shelfmark">
                                                        <strong>Main Shelfmark:</strong> {
                                                            onShelfmarkClick ? (
                                                                <span
                                                                    className="join-shelfmark-link"
                                                                    onClick={(e) => {
                                                                        e.stopPropagation();
                                                                        onShelfmarkClick(metadata.joins_data.mainShelfmark);
                                                                    }}
                                                                    title="Click to view this shelfmark"
                                                                >
                                                                    {metadata.joins_data.mainShelfmark}
                                                                </span>
                                                            ) : (
                                                                metadata.joins_data.mainShelfmark
                                                            )
                                                        }
                                                    </div>
                                                )}
                                                {metadata.joins_data.joinedManuscripts && metadata.joins_data.joinedManuscripts.length > 0 && (
                                                    <div className="joins-manuscripts">
                                                        <strong>Joined Manuscripts ({metadata.joins_data.joinedManuscripts.length}):</strong>
                                                        <ul className="joins-list">
                                                            {metadata.joins_data.joinedManuscripts.map((join, index) => (
                                                                <li key={index}>
                                                                    {onShelfmarkClick ? (
                                                                        <span
                                                                            className="join-shelfmark join-shelfmark-link"
                                                                            onClick={(e) => {
                                                                                e.stopPropagation();
                                                                                onShelfmarkClick(join.shelfmark);
                                                                            }}
                                                                            title="Click to view this shelfmark"
                                                                        >
                                                                            {join.shelfmark}
                                                                        </span>
                                                                    ) : (
                                                                        <span className="join-shelfmark">{join.shelfmark}</span>
                                                                    )}
                                                                    {join.source && <span className="join-source"> ({join.source})</span>}
                                                                </li>
                                                            ))}
                                                        </ul>
                                                    </div>
                                                )}
                                            </div>
                                        </div>
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