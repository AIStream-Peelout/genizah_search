import React, { useCallback, useEffect, useState } from 'react';
import { buildWorldcatSearchUrl, buildGoogleScholarUrl } from '../bibliographyLinks';
import './BibliographyDetail.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/** Public providers this panel may badge, mirroring the backend's allowlist. */
const PROVIDER_LABELS = {
    ktiv: { label: 'KTIV', full: 'KTIV, National Library of Israel' },
    pgp: { label: 'PGP', full: 'Princeton Geniza Project' },
};

const PAGE_SIZE = 50;

/**
 * Prefix a raw ``location`` (pages) value with "p." unless it already reads
 * like a page reference. Mirrors the formatting `BibliographyEntry` in
 * DocumentModel.jsx applies to the same field.
 * @param {string|null|undefined} location - Raw ``location``/pages value.
 * @returns {string|null} Formatted pages string, or null when empty.
 */
const formatPages = (location) => {
    if (!location) return null;
    const trimmed = String(location).trim();
    if (!trimmed) return null;
    return /^\s*(p{1,2}\.|pages?\b|עמ)/i.test(trimmed) ? trimmed : `p. ${trimmed}`;
};

/**
 * "Work detail" card for one bibliography citation: the publication's own
 * details, links to find it externally, and every other Genizah fragment in
 * the index that cites the same work (via ``GET /bibliography/work``).
 *
 * Renders as a full-viewport overlay so it works both inside the desktop
 * document modal and on the full-screen mobile modal (<768px).
 *
 * @param {object} props
 * @param {{title: string, authors?: string[], year?: string, source?: string|null}} props.entry -
 *   The clicked citation from ``metadata.bibliography_entries``. Only ``title`` is required;
 *   its ``authors``/``year``/``source`` seed the header before the fetch resolves.
 * @param {string} [props.indexName] - Index the parent document came from; passed through
 *   so citing fragments are looked up in the same index.
 * @param {() => void} props.onClose - Closes the panel.
 * @param {(docId: string) => void} props.onOpenDocument - Opens a citing fragment by its ``doc_id``.
 * @returns {React.ReactElement|null}
 */
function BibliographyDetail({ entry, indexName, onClose, onOpenDocument }) {
    const title = entry && entry.title;
    const [status, setStatus] = useState('loading'); // 'loading' | 'ready' | 'error'
    const [data, setData] = useState(null);
    const [error, setError] = useState(null);
    const [limit, setLimit] = useState(PAGE_SIZE);

    const load = useCallback((nextLimit) => {
        if (!title) return;
        setStatus('loading');
        setError(null);
        const params = new URLSearchParams({ title, limit: String(nextLimit) });
        if (indexName) params.set('index_name', indexName);
        fetch(`${API_BASE_URL}/bibliography/work?${params.toString()}`)
            .then(async (res) => {
                if (!res.ok) {
                    const body = await res.json().catch(() => ({}));
                    throw new Error(body.detail || `Request failed (${res.status})`);
                }
                return res.json();
            })
            .then((body) => {
                setData(body);
                setStatus('ready');
            })
            .catch((err) => {
                setError(err.message || 'Failed to load work details');
                setStatus('error');
            });
    }, [title, indexName]);

    useEffect(() => {
        setLimit(PAGE_SIZE);
        load(PAGE_SIZE);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [title, indexName]);

    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.key === 'Escape') onClose();
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [onClose]);

    if (!entry || !title) return null;

    const showMore = () => {
        const nextLimit = limit + PAGE_SIZE;
        setLimit(nextLimit);
        load(nextLimit);
    };

    const authors = (data && data.authors && data.authors.length ? data.authors : entry.authors) || [];
    const years = (data && data.years && data.years.length)
        ? data.years
        : (entry.year ? [entry.year] : []);
    // The provider badge reflects the clicked citation itself; fall back to
    // the first fragment's source if the entry (legacy string citation) had none.
    const provider = entry.source || (data && data.fragments && data.fragments.find((f) => f.source)?.source) || null;
    const providerInfo = provider && PROVIDER_LABELS[provider];
    const primaryAuthor = authors[0];
    const worldcatUrl = buildWorldcatSearchUrl(title, primaryAuthor);
    const scholarUrl = buildGoogleScholarUrl(title, primaryAuthor);

    return (
        <div className="bib-detail-overlay" onClick={onClose}>
            <div
                className="bib-detail-panel"
                role="dialog"
                aria-modal="true"
                aria-label={`Work details: ${title}`}
                onClick={(e) => e.stopPropagation()}
            >
                <div className="bib-detail-header">
                    <button
                        type="button"
                        className="bib-detail-close"
                        onClick={onClose}
                        aria-label="Close work details"
                    >
                        ×
                    </button>
                </div>

                <div className="bib-detail-body">
                    <em className="bib-detail-title" dir="auto">{title}</em>
                    {authors.length > 0 && (
                        <div className="bib-detail-authors">{authors.join('; ')}</div>
                    )}
                    {years.length > 0 && (
                        <div className="bib-detail-years">{years.join(', ')}</div>
                    )}
                    {providerInfo && (
                        <span
                            className={`bib-badge bib-badge-${provider}`}
                            title={providerInfo.full}
                        >
                            {providerInfo.label}
                        </span>
                    )}

                    <div className="bib-detail-links">
                        <a
                            className="bib-detail-link bib-detail-link-primary"
                            href={worldcatUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            Find in a library (WorldCat) ↗
                        </a>
                        <a
                            className="bib-detail-link"
                            href={scholarUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            Search Google Scholar ↗
                        </a>
                    </div>

                    <div className="bib-detail-fragments">
                        {status === 'loading' && !data && (
                            <p className="bib-detail-status">Loading citing fragments…</p>
                        )}
                        {status === 'error' && (
                            <p className="bib-detail-status bib-detail-error">{error}</p>
                        )}
                        {status === 'ready' && data && data.fragments.length === 0 && (
                            <p className="bib-detail-status">No other Genizah fragments cite this work.</p>
                        )}
                        {data && data.fragments.length > 0 && (
                            <>
                                <h5 className="bib-detail-fragments-heading">
                                    Genizah fragments citing this work ({data.total_fragments})
                                </h5>
                                <ul className="bib-detail-fragments-list">
                                    {data.fragments.map((fragment) => (
                                        <li key={fragment.doc_id} className="bib-detail-fragment">
                                            <button
                                                type="button"
                                                className="bib-detail-fragment-btn"
                                                onClick={() => onOpenDocument(fragment.doc_id)}
                                            >
                                                <span className="bib-detail-fragment-shelfmark">
                                                    {fragment.shelf_mark || fragment.doc_id}
                                                </span>
                                                {formatPages(fragment.location) && (
                                                    <span className="bib-detail-fragment-pages">
                                                        {formatPages(fragment.location)}
                                                    </span>
                                                )}
                                                {(fragment.relations || []).map((rel) => (
                                                    <span key={rel} className="bib-relation">{rel}</span>
                                                ))}
                                            </button>
                                        </li>
                                    ))}
                                </ul>
                                {data.total_fragments > data.fragments.length && (
                                    <button type="button" className="bib-detail-show-more" onClick={showMore}>
                                        {status === 'loading'
                                            ? 'Loading…'
                                            : `Show more (${data.fragments.length} of ${data.total_fragments})`}
                                    </button>
                                )}
                            </>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}

export default BibliographyDetail;
