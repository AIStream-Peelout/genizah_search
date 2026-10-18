import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { normalizeDocId, fragmentPath, hasFragmentPage, toCatalogueDocId } from '../utils';
import { getStoredToken } from '../read/advancedAccess';
import {
    altTitles as recordAltTitles,
    documentTypeLabel,
    needsEnrichedFields,
    recordDate,
    recordTitle,
    sourceLinkRel,
    sourceLinks as recordSourceLinks,
    subjectAndGenreChips,
    textList,
    withEnrichedFields,
} from './recordDisplay';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
import DocumentDetailView from './DocumentDetailView';
import SecondarySourceView from './SecondarySourceView';
import BibliographyDetail from './BibliographyDetail';

// Helper function to format transcriptions properly (handles arrays, strings, and objects)
const formatTranscription = (transcription) => {
    if (!transcription) return null;

    // If it's an array, handle each item
    if (Array.isArray(transcription)) {
        return transcription.map((item, index) => {
            let text = item;

            // If array item is an object, extract the text
            if (typeof item === 'object' && item !== null) {
                text = item.text || item.content || item.transcription || JSON.stringify(item);
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
        return (
            <div className="transcription-text" dir="auto">
                {String(text).split('\n').map((line, index) => (
                    <div key={index} className="transcription-line">{line || '\u00A0'}</div>
                ))}
            </div>
        );
    }

    // If it's a string, preserve line breaks and handle RTL text
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

/**
 * Alternative titles under the modal heading: the best one always, the rest
 * behind a "+N more" toggle.
 * @param {{titles: string[]}} props - ``alt_titles`` (non-empty), best first.
 * @returns {JSX.Element} The subtitle block.
 */
function AltTitles({ titles }) {
    const [expanded, setExpanded] = useState(false);
    const rest = titles.slice(1);
    return (
        <div className="modal-alt-titles">
            <p className="modal-alt-title" dir="auto">{titles[0]}</p>
            {expanded && rest.length > 0 && (
                <div className="modal-alt-titles-more">
                    {rest.map((title, index) => (
                        <p key={index} className="modal-alt-title" dir="auto">{title}</p>
                    ))}
                </div>
            )}
            {rest.length > 0 && (
                <button
                    type="button"
                    className="alt-titles-toggle"
                    aria-expanded={expanded}
                    onClick={() => setExpanded(!expanded)}
                >
                    {expanded ? 'Show fewer titles' : `+${rest.length} more`}
                </button>
            )}
        </div>
    );
}

/**
 * "View at the source" row: one outline button per holding institution or
 * project page; each button names the source, which is also its attribution.
 * @param {{links: object[]}} props - Links from ``recordSourceLinks`` (non-empty).
 * @returns {JSX.Element} The row.
 */
function SourceLinks({ links }) {
    return (
        <div className="modal-source-link source-links-row">
            <span className="source-links-label">View at the source:</span>
            {links.map((link) => (
                <a
                    key={link.url}
                    href={link.url}
                    target="_blank"
                    rel={sourceLinkRel(link.kind)}
                    className="original-source-btn source-link-btn"
                >
                    {link.label}
                    <span aria-hidden="true"> ↗</span>
                </a>
            ))}
        </div>
    );
}

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

        // Check if this is a legacy index (cairo_genizah_text_only_v1.0.6)
        // For legacy index, use ONLY actual_image_url
        // For new indices, use image_urls array
        if (isLegacyIndex && metadata.actual_image_url) {
            return [metadata.actual_image_url];
        }

        // This is a new index - use image_urls array
        if (metadata.image_urls && Array.isArray(metadata.image_urls) && metadata.image_urls.length > 0) {
            // Clean URLs by removing srcset descriptors like "1440w"
            return metadata.image_urls
                .filter(url => url && typeof url === 'string' && url.trim())
                .map(url => url.split(/\s+/)[0])
                .filter(url => url && url.trim() && !url.endsWith('w'));
        }

        // Fallback for legacy documents without index_name set
        if (metadata.actual_image_url && typeof metadata.actual_image_url === 'string' && metadata.actual_image_url.trim()) {
            return [metadata.actual_image_url];
        }

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

    // Enriched display fields (alt titles, subjects, genres, catalogue
    // headings): GET /document reads them from the enrichment index, search
    // hits do not carry them. One request per modal open, none per result
    // card. Kept only when GET /document answered with this very record (its
    // shelf-mark fallback can find another), and with its doc_id so a late
    // answer never lands on the next record opened.
    const [enriched, setEnriched] = useState(null);
    const enrichDocId = (isOpen && document?.doc_id
        && needsEnrichedFields(document.index_name, document.metadata || document))
        ? document.doc_id
        : null;
    // The record's own doc_id field (a hit's doc_id can be the ES _id).
    const enrichRecordId = enrichDocId ? (document.metadata || document).doc_id : null;
    useEffect(() => {
        setEnriched(null);
        if (!enrichDocId) return undefined;
        const controller = new AbortController();
        const q = document?.index_name ? `?index_name=${encodeURIComponent(document.index_name)}` : '';
        fetch(`${API_BASE_URL}/document/${encodeURIComponent(normalizeDocId(enrichDocId))}${q}`, { signal: controller.signal })
            .then((r) => (r.ok ? r.json() : null))
            .then((details) => {
                const sameRecord = details && (details.doc_id === enrichDocId || details.doc_id === enrichRecordId);
                if (sameRecord) setEnriched({ docId: enrichDocId, details });
            })
            .catch(() => {});
        return () => controller.abort();
    }, [enrichDocId, enrichRecordId, document?.index_name]);

    // "Copy link" feedback: true for a moment after the permalink was copied.
    const [linkCopied, setLinkCopied] = useState(false);
    useEffect(() => {
        setLinkCopied(false);
    }, [document?.doc_id]);
    useEffect(() => {
        if (!linkCopied) return undefined;
        const timer = setTimeout(() => setLinkCopied(false), 1800);
        return () => clearTimeout(timer);
    }, [linkCopied]);

    /**
     * Copy the absolute permalink of the open record to the clipboard. Where
     * the Clipboard API is unavailable (non-HTTPS origins), show the link in a
     * prompt so it can be copied by hand.
     * @param {string} path - Site path from ``fragmentPath``.
     */
    const handleCopyPermalink = (path) => {
        const url = window.location.origin + path;
        if (!navigator.clipboard) {
            window.prompt('Copy this link:', url);
            return;
        }
        navigator.clipboard.writeText(url)
            .then(() => setLinkCopied(true))
            .catch(() => window.prompt('Copy this link:', url));
    };

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

    // Get metadata from document, with the enriched fields once they arrive.
    const metadata = withEnrichedFields(
        document.metadata || document,
        enriched && enriched.docId === document.doc_id ? enriched.details : null
    );

    // Public page of this record: catalogue records only (never bibliography
    // works, which returned above, nor documents of other indices; a beta
    // record only when the backend found it in the catalogue).
    const permalinkPath = document.doc_id && hasFragmentPage(document.index_name, document)
        ? fragmentPath(toCatalogueDocId(document.doc_id))
        : null;

    // Catalogue display fields, read the same way as on the result cards.
    const heading = recordTitle(metadata, document.shelfmark || document.title || document.doc_id);
    const titles = recordAltTitles(metadata);
    const links = recordSourceLinks(metadata);
    const dateText = recordDate(metadata);
    const typeText = documentTypeLabel(metadata);
    const languageText = metadata.language || document.language || '';
    const mainLanguage = typeof metadata.main_language === 'string' ? metadata.main_language.trim() : '';
    // main_language is stored lower case ("hebrew"); only show it when it says more.
    const showMainLanguage = Boolean(mainLanguage) && mainLanguage.toLowerCase() !== String(languageText).toLowerCase();
    const topicChips = subjectAndGenreChips(metadata);
    const catalogueHeadings = textList(metadata.scholarly_frames);
    const institutionName = metadata.repository || document.institution;
    const collectionName = metadata.collection || document.collection;
    // Researchers with an advanced-view token get a link to the Readers view
    // (a localStorage read, no request); everyone else sees nothing new.
    const hasAdvancedToken = Boolean(getStoredToken());

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
                        <h2>{heading}</h2>
                        {titles.length > 0 && <AltTitles key={document.doc_id} titles={titles} />}
                        {permalinkPath && (
                            <div className="modal-source-link modal-permalink">
                                <a
                                    href={permalinkPath}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="permalink-btn"
                                    title="Public page of this record, with an address you can share or cite"
                                >
                                    🔗 Permalink
                                </a>
                                <button
                                    type="button"
                                    className="copy-link-btn"
                                    onClick={() => handleCopyPermalink(permalinkPath)}
                                    title="Copy the permalink"
                                    aria-label="Copy the permalink"
                                >
                                    {linkCopied ? 'Copied' : 'Copy link'}
                                </button>
                                <span className="sr-only" role="status" aria-live="polite">
                                    {linkCopied ? 'Link copied to the clipboard' : ''}
                                </span>
                            </div>
                        )}
                        {links.length > 0 && <SourceLinks links={links} />}
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
                        {hasAdvancedToken && document.doc_id && (
                            <div className="modal-source-link">
                                <Link
                                    to={`/read?doc=${encodeURIComponent(document.doc_id)}&image=${currentImageIndex}&view=readers${document.index_name ? `&index=${encodeURIComponent(document.index_name)}` : ''}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="copy-link-btn"
                                    title="Advanced view for researchers: every reader's lines, the decoding policy and suggested readings"
                                >
                                    Readers (advanced)
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
                                    manifestUrl={`${process.env.REACT_APP_API_URL || 'http://localhost:8000'}/document/${encodeURIComponent(normalizeDocId(document.doc_id))}/manifest${document.index_name ? `?index_name=${encodeURIComponent(document.index_name)}` : ''}`}
                                />
                            </div>
                        </div>

                        {/* Enhanced document details */}
                        <div className="document-details">
                            <h4>Document Details</h4>
                            <div className="details-grid">
                                {dateText && <div><strong>Date:</strong> <span dir="auto">{dateText}</span></div>}
                                {languageText && (
                                    <div><strong>Language:</strong> {languageText}</div>
                                )}
                                {showMainLanguage && (
                                    <div><strong>Main Language:</strong> {mainLanguage}</div>
                                )}
                                {typeText && (
                                    <div><strong>Document Type:</strong> {typeText}</div>
                                )}
                                {/* Catalogue value only: some callers pass a placeholder material. */}
                                {metadata.material && (
                                    <div><strong>Material:</strong> {metadata.material}</div>
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

                        {/* Enhanced institution & collection */}
                        {(document.institution || document.collection || metadata.repository || metadata.collection) && (
                            <div className="modal-section">
                                <h4>Institution & Collection</h4>
                                <div className="institution-details">
                                    {/* Free-text names: shown exactly as catalogued. */}
                                    {institutionName && (
                                        <div><strong>Institution:</strong> {institutionName}</div>
                                    )}
                                    {metadata.library && (
                                        <div><strong>Library:</strong> {metadata.library}</div>
                                    )}
                                    {collectionName && (
                                        <div><strong>Collection:</strong> {collectionName}</div>
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

                        {/* Subjects and genres (catalogue fields; genres first) */}
                        {topicChips.length > 0 && (
                            <div className="modal-section">
                                <h4>Subjects & genres</h4>
                                <div className="tags-container">
                                    {topicChips.map((chip) => (
                                        <span
                                            key={`${chip.kind}:${chip.value}`}
                                            className={`tag tag-${chip.kind}`}
                                            title={chip.kind === 'genre' ? 'Genre' : 'Subject'}
                                            dir="auto"
                                        >
                                            {chip.value}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Scholarly catalogue headings, collapsed by default */}
                        {catalogueHeadings.length > 0 && (
                            <details className="modal-section catalogue-headings">
                                <summary>Catalogue headings ({catalogueHeadings.length})</summary>
                                <ul className="catalogue-headings-list">
                                    {catalogueHeadings.map((entry, index) => (
                                        <li key={index} dir="auto">{entry}</li>
                                    ))}
                                </ul>
                            </details>
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

                        {/* Joins: fragments of the same manuscript held under other shelf marks */}
                        {metadata.joins_data && (
                            <div className="modal-section technical-section">
                                <h4>Joins</h4>
                                <div className="technical-details">
                                    <div className="joins-data-section">
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