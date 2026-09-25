import React, { useState } from 'react';
import { Link } from 'react-router-dom';

/**
 * DOM id of a fragment card, the target of strip chips and ``?f=`` deep links.
 * @param {string} docId - Catalogue doc_id.
 * @returns {string} Element id.
 */
export const cardAnchorId = (docId) => `sk-card-${docId}`;

/**
 * CSS custom property holding a genre's colour, with a neutral fallback.
 * @param {string|undefined} genre - Genre id, e.g. ``hoshanot``.
 * @returns {string} A ``var(...)`` expression.
 */
export const genreColor = (genre) => `var(--sk-g-${genre || 'none'}, var(--sk-g-default))`;

/**
 * The one-line text status of a fragment: a human transcription takes
 * precedence, then a machine reading (only present when lines were agreed),
 * otherwise none.
 * @param {object} f - Fragment record.
 * @returns {string} Status text.
 */
export function textStatusLine(f) {
    if (f.text_status === 'human') return 'Human transcription available';
    if (f.machine_read && f.machine_read.label) return f.machine_read.label;
    return 'No transcription yet';
}

/**
 * Which of the three text statuses applies, for styling; mirrors ``textStatusLine``.
 * @param {object} f - Fragment record.
 * @returns {'human'|'machine'|'none'} Status kind.
 */
const statusKind = (f) => {
    if (f.text_status === 'human') return 'human';
    return f.machine_read && f.machine_read.label ? 'machine' : 'none';
};

/**
 * Date label, with "(computed)" appended when the caveat says the CE date is
 * our own conversion and the display text does not already say so.
 * @param {object|null} date - ``fragment.date``.
 * @returns {string|null} Label, or null when there is no date.
 */
export function dateLabel(date) {
    if (!date || !date.display) return null;
    const computed = /computed/i.test(date.caveat || '') && !/computed/i.test(date.display);
    return computed ? `${date.display} (computed)` : date.display;
}

/**
 * One fragment card: thumbnail, normalised shelfmark, human-sourced card line
 * with its source, date, place, text status, catalogue-record button, optional
 * machine-reading link, bibliography line and image credit.
 * @param {{fragment: object, highlighted: boolean, onOpenEsDocument: function(string): void,
 *   clusters: Object<string, object>, dayLabels: Object<string, string>, genre: string}} props - Card props.
 * @returns {JSX.Element} The card.
 */
export default function FragmentCard({ fragment: f, highlighted, onOpenEsDocument, clusters, dayLabels, genre }) {
    const [showDateNote, setShowDateNote] = useState(false);
    const img = f.image && (f.image.thumb || f.image.url);
    const date = dateLabel(f.date);
    const clusterList = (Array.isArray(f.clusters) ? f.clusters : [])
        .map((id) => clusters[id])
        .filter(Boolean)
        .slice(0, 2);
    const day = f.festival_day && f.festival_day.day ? dayLabels[f.festival_day.day] : null;
    const mr = f.machine_read;

    return (
        <article
            id={cardAnchorId(f.doc_id)}
            className={`sk-card${highlighted ? ' sk-card--highlight' : ''}`}
            style={{ '--sk-accent': genreColor(f.genre || genre) }}
            tabIndex={-1}
            aria-labelledby={`${cardAnchorId(f.doc_id)}-h`}
        >
            <div className="sk-thumb">
                {img ? (
                    <img src={img} alt={`${f.shelfmark}, manuscript image`} loading="lazy" decoding="async" />
                ) : (
                    <span className="sk-thumb-none">No image on the site</span>
                )}
            </div>
            <div className="sk-card-body">
                <h3 className="sk-card-title" id={`${cardAnchorId(f.doc_id)}-h`}>{f.shelfmark}</h3>
                <p className="sk-card-line" dir="auto">{f.card_line}</p>
                <p className="sk-meta">
                    {f.card_source_public && (
                        <span className="sk-badge" title={f.card_source_note || undefined}>
                            Source: {f.card_source_public}
                        </span>
                    )}
                    {date && (
                        <span className="sk-date">
                            {date}
                            {f.date.caveat && (
                                <button
                                    type="button"
                                    className="sk-info"
                                    aria-expanded={showDateNote}
                                    aria-label="About this date"
                                    title={f.date.caveat}
                                    onClick={() => setShowDateNote((v) => !v)}
                                >
                                    ⓘ
                                </button>
                            )}
                        </span>
                    )}
                    {f.place && <span className="sk-place">{f.place}</span>}
                    {f.contested && <span className="sk-flag">Contested</span>}
                </p>
                {showDateNote && f.date && f.date.caveat && (
                    <p className="sk-small sk-date-note" dir="auto">{f.date.caveat}</p>
                )}
                {day && (
                    <p className="sk-small sk-dayline">
                        Festival day: {day}
                        {f.festival_day.basis && f.festival_day.basis !== 'contested' && ` (${f.festival_day.basis})`}
                        {f.festival_day.basis === 'contested' && ' (contested)'}
                    </p>
                )}
                {clusterList.length > 0 && (
                    <p className="sk-clusters">
                        {clusterList.map((c) => (
                            <span key={c.id} className="sk-cluster" title={c.basis || undefined}>{c.title}</span>
                        ))}
                    </p>
                )}
                <p className={`sk-status sk-status--${statusKind(f)}`}>{textStatusLine(f)}</p>
                <div className="sk-actions">
                    <button type="button" className="sk-btn" onClick={() => onOpenEsDocument && onOpenEsDocument(f.doc_id)}>
                        Catalogue record
                    </button>
                    {mr && mr.href && (
                        <Link to={mr.href} className="sk-readlink" target="_blank" rel="noopener noreferrer">
                            Read the machine text on the manuscript →
                        </Link>
                    )}
                </div>
                {mr && mr.note && <p className="sk-small sk-mr-note" dir="auto">{mr.note}</p>}
                {f.bibliography_line && <p className="sk-small sk-biblio" dir="auto">{f.bibliography_line}</p>}
                {f.holding_credit && <p className="sk-credit">{f.holding_credit}</p>}
            </div>
        </article>
    );
}
