import React from 'react';
import { Link } from 'react-router-dom';
import { AGREEMENT_CAVEAT, UNCONFIRMED_TOOLTIP, agreedTooltip } from '../read/caveat';
import './AiTranscriptionResults.css';

/**
 * Render an Elasticsearch highlight string (`<em>` around matched terms) as
 * React nodes without injecting HTML: the text is machine output, so it is
 * split on the tags and rendered as plain text nodes.
 * @param {string} marked - Highlighted line, e.g. "abc <em>def</em> ghi".
 * @returns {React.ReactNode[]} Text and <mark> nodes.
 */
export const renderHighlight = (marked) =>
    marked.split(/(<em>.*?<\/em>)/g).filter(Boolean).map((part, i) => {
        const m = part.match(/^<em>(.*)<\/em>$/);
        return m ? <mark key={i}>{m[1]}</mark> : <React.Fragment key={i}>{part}</React.Fragment>;
    });

/**
 * Deep link into the /read viewer at one line.
 * @param {object} hit - Search hit.
 * @param {number} lineIndex - 0-based line index.
 * @returns {string} Route with query string.
 */
export const readLink = (hit, lineIndex) => {
    const q = new URLSearchParams({ doc: hit.doc_id, image: String(hit.image_index) });
    if (hit.source_index) q.set('index', hit.source_index);
    if (hit.vlm_model) q.set('model', hit.vlm_model);
    q.set('line', String(lineIndex));
    return `/read?${q.toString()}`;
};

/** Persistent beta caveat shown above AI-transcription results. */
export function AiSearchBanner() {
    return (
        <div className="ai-search-banner" role="note">
            <strong>Beta: searching machine reads, not checked by a person.</strong>{' '}
            Two automatic readers (a fine-tuned vision-language model and the Kraken HTR model) read each page
            separately, offline. {AGREEMENT_CAVEAT} Matches are shown on the line that matched; open the viewer
            to see it on the image. Do not cite this text.
        </div>
    );
}

/**
 * One matched line: number, highlighted text, agreement badge as in the viewer, deep link.
 */
function MatchedLine({ hit, line }) {
    const agreed = line.status === 'agreed';
    return (
        <li className={`ai-line ${line.status}`} title={agreed ? agreedTooltip(line) : UNCONFIRMED_TOOLTIP}>
            <Link to={readLink(hit, line.index)} className="ai-line-number" dir="ltr" title="Open this line in the viewer">
                {line.index + 1}
            </Link>
            <span className="ai-line-text" dir="rtl">
                {line.highlight ? renderHighlight(line.highlight) : line.text}
            </span>
            {agreed ? (
                <span className="ai-badge agreed" dir="ltr">✓ 2 readers · {Number(line.agreement).toFixed(2)}</span>
            ) : (
                <span className="ai-badge unconfirmed" dir="ltr">single reader</span>
            )}
        </li>
    );
}

/**
 * Results of POST /search-ai-transcriptions: one card per image with its matched lines.
 * Separate from the catalogue results on purpose: nothing here is ranked against them.
 */
export default function AiTranscriptionResults({ results, loading, query, onLoadMore, isLoadingMore, onDocumentClick }) {
    if (loading) {
        return (
            <div className="results-section">
                <div className="loading">
                    <div className="spinner"></div>
                    <span>Searching the AI line reads...</span>
                </div>
            </div>
        );
    }
    if (!results) return null;

    if (!results.enabled) {
        return (
            <div className="results-section">
                <AiSearchBanner />
                <div className="no-results"><p>AI transcription search is switched off.</p></div>
            </div>
        );
    }
    if (!results.available) {
        return (
            <div className="results-section">
                <AiSearchBanner />
                <div className="no-results">
                    <p>AI transcription search is not available yet{results.message ? ` (${results.message})` : ''}.</p>
                </div>
            </div>
        );
    }

    return (
        <div className="results-section ai-results">
            <div className="results-header">
                <h3>AI transcription matches for: "{query}"</h3>
                <div className="results-meta">
                    <span className="ai-results-reminder">machine reads, unchecked; two readers agreeing is not verification</span>
                    <span>{results.total} image{results.total === 1 ? '' : 's'} with a matching line</span>
                    {results.processing_time_ms != null && <span>({results.processing_time_ms}ms)</span>}
                    <span className="index-info">
                        {results.include_unconfirmed ? 'confirmed + unconfirmed lines' : 'confirmed lines only'}
                    </span>
                    {results.models?.length > 1 && (
                        <span className="index-info">
                            readers: {results.models.map((m) => `${m.vlm_model} (${m.count})`).join(', ')}
                        </span>
                    )}
                </div>
            </div>

            {results.total === 0 ? (
                <div className="no-results">
                    <p>No AI-read line matches. The reads cover only a small part of the collection.</p>
                    <div className="search-tips">
                        <h4>Tips:</h4>
                        <ul>
                            <li>Search in Hebrew letters; points and final forms are ignored.</li>
                            <li>Try one or two words rather than a long phrase.</li>
                            <li>Tick “include unconfirmed lines” to widen the search (more noise).</li>
                        </ul>
                    </div>
                </div>
            ) : (
                <div className="ai-results-list">
                    {results.results.map((hit) => {
                        const label = hit.shelf_mark || hit.doc_id;
                        return (
                            <article key={`${hit.doc_id}__${hit.image_index}__${hit.vlm_model}`} className="ai-result-card">
                                <Link to={readLink(hit, hit.lines[0]?.index ?? 0)} className="ai-result-thumb" title="Open in the viewer">
                                    <img src={hit.image_url} alt={`${label}, image ${hit.image_index + 1}`} loading="lazy" />
                                </Link>
                                <div className="ai-result-body">
                                    <div className="ai-result-head">
                                        <h4 className="ai-result-shelfmark">
                                            <button
                                                type="button"
                                                className="ai-result-doc-link"
                                                onClick={() => onDocumentClick && onDocumentClick(hit)}
                                                title="Open the catalogue record"
                                            >
                                                {label}
                                            </button>
                                            <span className="ai-result-image">image {hit.image_index + 1}</span>
                                        </h4>
                                        <span className="ai-result-confirmed">
                                            {hit.n_agreed} of {hit.n_lines} lines confirmed
                                        </span>
                                    </div>
                                    {(hit.title || hit.description) && (
                                        <p className="ai-result-catalogue">
                                            {hit.title && <strong>{hit.title}</strong>}
                                            {hit.title && hit.description ? ' — ' : ''}
                                            {hit.description}
                                        </p>
                                    )}
                                    <ol className="ai-lines">
                                        {hit.lines.map((line) => (
                                            <MatchedLine key={line.index} hit={hit} line={line} />
                                        ))}
                                    </ol>
                                    <div className="ai-result-footer">
                                        <Link to={readLink(hit, hit.lines[0]?.index ?? 0)} className="ai-result-open">
                                            Open in viewer at line {(hit.lines[0]?.index ?? 0) + 1} →
                                        </Link>
                                        <span className="ai-result-model" title="Vision-language checkpoint that produced this read">
                                            {hit.vlm_model}
                                        </span>
                                    </div>
                                </div>
                            </article>
                        );
                    })}
                </div>
            )}

            {results.has_more && (
                <div className="pagination-controls">
                    <button className="load-more-button" onClick={onLoadMore} disabled={isLoadingMore}>
                        {isLoadingMore ? 'Loading...' : 'Load more matches'}
                    </button>
                    <div className="pagination-meta">
                        <span>{results.results.length} of {results.total} images shown</span>
                    </div>
                </div>
            )}
        </div>
    );
}
