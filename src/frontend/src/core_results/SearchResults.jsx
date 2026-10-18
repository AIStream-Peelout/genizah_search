import React from 'react';
import { Link } from 'react-router-dom';
import { fragmentPath, hasFragmentPage, isPlainLeftClick } from '../utils';
import { ROUTED_TO_SHELFMARK } from './semanticBeta';
import { cardChips, cardSubtitle, recordTitle } from './recordDisplay';

/** Index whose records keep their image in ``actual_image_url`` only. */
const LEGACY_INDEX = 'cairo_genizah_text_only_v1.0.6';

/** Card image when a record has none, or when its image fails to load. */
const PLACEHOLDER_IMAGE_URL = 'https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=400&h=300&fit=crop';

/** Card text when a record has no description. */
const FALLBACK_DESCRIPTION = 'Historical manuscript from the Cairo Genizah collection.';

/**
 * Image shown on a result card. Legacy-index records use ``actual_image_url``;
 * newer ones the first usable ``image_urls`` entry (srcset width descriptors
 * such as "1440w" removed), then ``actual_image_url``, ``image_url``,
 * ``thumbnail_url`` and finally a placeholder.
 * @param {object} metadata - ``result.metadata`` from the backend.
 * @param {boolean} isLegacyIndex - Whether the result comes from the legacy index.
 * @returns {string} The image URL.
 */
const cardImageUrl = (metadata, isLegacyIndex) => {
    if (isLegacyIndex && metadata.actual_image_url) return metadata.actual_image_url;
    if (Array.isArray(metadata.image_urls)) {
        const validUrls = metadata.image_urls
            .filter((url) => url && typeof url === 'string' && url.trim())
            .map((url) => url.split(/\s+/)[0])
            .filter((url) => url && url.trim() && !url.endsWith('w'));
        if (validUrls.length > 0) return validUrls[0];
    }
    return metadata.actual_image_url || metadata.image_url || metadata.thumbnail_url || PLACEHOLDER_IMAGE_URL;
};

/** Label on results that matched only in a machine reading, never in the catalogue. */
export const MACHINE_READING_LABEL =
    'Matched in machine reading (two machine readers agree; not checked by a person)';

/**
 * Whether a result is an appended machine-reading match (search result
 * contract: ``match_source === 'machine_reading'`` plus a ``machine_reading``
 * object); catalogue hits have no ``match_source``.
 * @param {object} result - One entry of ``results.results``.
 * @returns {boolean} True for a machine-reading match.
 */
const isMachineReadingHit = (result) =>
    result.match_source === 'machine_reading' && Boolean(result.machine_reading);

/**
 * Viewer link of a machine-reading match, only when it is a site-relative
 * /read address (the backend builds it; anything else is not rendered).
 * @param {object} machineReading - ``result.machine_reading``.
 * @returns {string|null} The /read route, or null.
 */
const machineReadingHref = (machineReading) => {
    const href = machineReading && machineReading.read_url;
    return typeof href === 'string' && href.startsWith('/read?') ? href : null;
};

/**
 * Body of a machine-reading match: the label, the matched agreed line and a
 * link to see it on the manuscript image.
 * @param {{machineReading: object}} props - ``result.machine_reading``.
 * @returns {JSX.Element} The block.
 */
function MachineReadingMatch({ machineReading }) {
    const href = machineReadingHref(machineReading);
    return (
        <div className="mr-match" data-nosnippet="">
            <div className="mr-match-badge">{MACHINE_READING_LABEL}</div>
            {machineReading.snippet && (
                <p className="mr-match-snippet" dir="rtl" lang="he">{machineReading.snippet}</p>
            )}
            {href && (
                <Link
                    to={href}
                    className="mr-match-link"
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                >
                    See it on the manuscript →
                </Link>
            )}
        </div>
    );
}

const SearchResults = ({ results, loading, query, processingTime, onDocumentClick, onLoadMore, isLoadingMore, currentSearchMode }) => {
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
                    {results.index_name && (
                        <span className="index-info">
                            📊 Index: {results.index_name}
                        </span>
                    )}
                    {results.semantic_beta && (
                        <span
                            className="semantic-beta-badge"
                            title="Fine-tuned embedder over the new index (beta)"
                            data-testid="semantic-beta-badge"
                        >
                            🧪 v10 beta
                        </span>
                    )}
                </div>
                {results.routed_to === ROUTED_TO_SHELFMARK && (
                    <p className="routed-shelfmark-note" data-testid="routed-shelfmark-note">
                        Shelf mark detected: showing shelf-mark matches.
                    </p>
                )}
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
                        // Titled by shelf mark; the best alt title and the
                        // chips only show what the catalogue actually holds.
                        const subtitle = cardSubtitle(metadata);
                        const chips = cardChips(metadata);

                        const displayData = {
                            title: recordTitle(metadata, result.doc_id),
                            description: metadata.description || FALLBACK_DESCRIPTION,
                            image_url: cardImageUrl(metadata, results.index_name === LEGACY_INDEX),
                            language: metadata.language,
                            institution: metadata.institution,
                            collection: metadata.collection,
                            shelfmark: metadata.shelf_mark,
                            transcription: metadata.transcription,
                            translation: metadata.translation,
                            tags: metadata.tags,
                            period: metadata.period,
                            location: metadata.location,
                            dimensions: metadata.dimensions,
                            document_type: metadata.document_type
                        };

                        const machineHit = isMachineReadingHit(result);
                        const mr = result.machine_reading;
                        const openResult = () => onDocumentClick({ ...displayData, doc_id: result.doc_id, ...result, index_name: results.index_name });
                        // Catalogue records have a public page; link to it so the
                        // address is real, but keep a plain click opening the modal.
                        const recordHref = result.doc_id && hasFragmentPage(results.index_name, result)
                            ? fragmentPath(result.doc_id)
                            : null;
                        const handleRecordLinkClick = (e) => {
                            if (isPlainLeftClick(e)) {
                                e.preventDefault();
                                openResult();
                            }
                            e.stopPropagation();
                        };
                        // Machine-reading matches arrive after every catalogue
                        // result; a heading marks where they start.
                        const firstMachineHit = machineHit && (index === 0 || !isMachineReadingHit(results.results[index - 1]));

                        return (
                            <React.Fragment
                                key={machineHit ? `mr:${result.doc_id}:${mr.image_index}:${mr.line_index}` : result.doc_id}
                            >
                            {firstMachineHit && (
                                <div className="mr-results-divider" data-nosnippet="">
                                    <h4>Matches in machine readings</h4>
                                    <p>
                                        These fragments matched only in text read by machine: two automatic readers
                                        agree on the line, but no person has checked it. They come after every
                                        catalogue result.
                                    </p>
                                </div>
                            )}
                            <div
                                className={`result-item enhanced-result-item${machineHit ? ' result-item--machine' : ''}`}
                                onClick={openResult}
                            >
                                <div className="result-image">
                                    <img
                                        src={displayData.image_url}
                                        alt={displayData.title}
                                        onError={(e) => {
                                            e.target.src = PLACEHOLDER_IMAGE_URL;
                                        }}
                                    />
                                    <div className={`similarity-badge${machineHit ? ' similarity-badge--machine' : ''}`}>
                                        {machineHit ? 'Machine reading' : `${(result.similarity_score * 100).toFixed(1)}% match`}
                                    </div>
                                </div>

                                <div className="result-content">
                                    <h4 className="result-title">
                                        {recordHref ? (
                                            <a href={recordHref} className="result-title-link" onClick={handleRecordLinkClick}>
                                                {displayData.title}
                                            </a>
                                        ) : displayData.title}
                                    </h4>
                                    {subtitle && (
                                        <p className="result-subtitle" dir="auto" title={subtitle}>{subtitle}</p>
                                    )}
                                    {machineHit && <MachineReadingMatch machineReading={mr} />}
                                    <p className="result-description">{displayData.description}</p>

                                    {chips.length > 0 && (
                                        <div className="result-metadata">
                                            {chips.map((chip, chipIndex) => (
                                                <span key={chipIndex} className="metadata-item" dir="auto">{chip}</span>
                                            ))}
                                        </div>
                                    )}

                                    <div className="result-footer">
                                        <div className="result-id">ID: {result.doc_id}</div>
                                        {recordHref ? (
                                            <a href={recordHref} className="view-details" onClick={handleRecordLinkClick}>
                                                View Details →
                                            </a>
                                        ) : (
                                            <div className="view-details">View Details →</div>
                                        )}
                                    </div>

                                    {!machineHit && (
                                        <div className="similarity-progress">
                                            <div
                                                className="progress-fill"
                                                style={{ width: `${result.similarity_score * 100}%` }}
                                            ></div>
                                        </div>
                                    )}
                                </div>
                            </div>
                            </React.Fragment>
                        );
                    })}
                </div>
            )}

            {/* Temporary: Always show load more button for testing */}
            {results && currentSearchMode !== 'shelfmark' && (
                <div className="pagination-controls">
                    <button className="load-more-button" onClick={onLoadMore} disabled={isLoadingMore}>
                        {isLoadingMore ? 'Loading...' : 'Load more results'}
                    </button>
                    <div className="pagination-meta">
                        <span>Page {results.page} of {results.total_pages}</span>
                        <span> • {results.total} total matches</span>
                        <span> • Mode: {currentSearchMode}</span>
                        <span> • Has more: {String(results.has_more)}</span>
                    </div>
                </div>
            )}
        </div>
    );
};

export default SearchResults;