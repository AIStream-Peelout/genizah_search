import React from 'react';
import { Link } from 'react-router-dom';
import { fragmentPath, isCatalogueIndex, isPlainLeftClick } from '../utils';
import { ROUTED_TO_SHELFMARK } from './semanticBeta';

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
                            🧪 v9 beta
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
                        const fallbackData = {
                            title: `Document ${result.doc_id}`,
                            description: "Historical manuscript from the Cairo Genizah collection.",
                            image_url: "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=400&h=300&fit=crop",
                            date: "Unknown",
                            language: "Hebrew",
                            material: "Parchment"
                        };

                        // Determine if this is a legacy index
                        const isLegacyIndex = results.index_name === 'cairo_genizah_text_only_v1.0.6';

                        const displayData = {
                            title: metadata.title || fallbackData.title,
                            description: metadata.description || fallbackData.description,
                            image_url: (() => {
                                console.log(`[SearchResults] Image selection for doc ${result.doc_id}:`);
                                console.log(`  - Index: ${results.index_name}`);
                                console.log(`  - Is legacy: ${isLegacyIndex}`);
                                console.log(`  - Has actual_image_url: ${!!metadata.actual_image_url}`);
                                console.log(`  - Has image_urls: ${!!metadata.image_urls}`);
                                console.log(`  - image_urls value:`, metadata.image_urls);
                                
                                // If legacy index: use actual_image_url
                                if (isLegacyIndex && metadata.actual_image_url) {
                                    console.log(`  ✅ Using actual_image_url for legacy index`);
                                    return metadata.actual_image_url;
                                }
                                
                                // For new indices or if legacy check fails: use image_urls array
                                if (metadata.image_urls && Array.isArray(metadata.image_urls) && metadata.image_urls.length > 0) {
                                    const validUrls = metadata.image_urls
                                        .filter(url => url && typeof url === 'string' && url.trim())
                                        .map(url => {
                                            // Clean URLs by removing srcset descriptors like "1440w"
                                            const cleaned = url.split(/\s+/)[0];
                                            return cleaned;
                                        })
                                        .filter(url => url && url.trim() && !url.endsWith('w'));
                                    
                                    if (validUrls.length > 0) {
                                        console.log(`  ✅ Using image_urls array, returning first of ${validUrls.length} URLs`);
                                        return validUrls[0];
                                    }
                                    console.log(`  ⚠️ image_urls array exists but no valid URLs after filtering`);
                                }
                                
                                // Fallback: actual_image_url (in case metadata isn't set properly)
                                if (metadata.actual_image_url) {
                                    console.log(`  ✅ Fallback: Using actual_image_url`);
                                    return metadata.actual_image_url;
                                }
                                
                                // Priority 3: image_url
                                if (metadata.image_url) {
                                    console.log(`  ✅ Using image_url`);
                                    return metadata.image_url;
                                }
                                
                                // Priority 4: thumbnail_url
                                if (metadata.thumbnail_url) {
                                    console.log(`  ✅ Using thumbnail_url`);
                                    return metadata.thumbnail_url;
                                }
                                
                                // Fallback to placeholder
                                console.log(`  ❌ No valid image found, using fallback unsplash`);
                                return fallbackData.image_url;
                            })(),
                            date: metadata.date || fallbackData.date,
                            language: metadata.language || fallbackData.language,
                            material: metadata.material || fallbackData.material,
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
                        const recordHref = result.doc_id && isCatalogueIndex(results.index_name)
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
                                            e.target.src = "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?w=400&h=300&fit=crop";
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
                                    {machineHit && <MachineReadingMatch machineReading={mr} />}
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