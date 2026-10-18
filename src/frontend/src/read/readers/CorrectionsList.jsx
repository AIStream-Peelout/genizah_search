import React from 'react';
import { FINAL_KEY, findCorrectionLine, formatTime, sameInstant } from './readersModel';

/**
 * Why a stored suggestion may not match what is on screen now, if it may not.
 * @param {object} correction - Correction record.
 * @param {object} sidecar - Sidecar being shown.
 * @returns {string} Short note, or ''.
 */
function staleNote(correction, sidecar) {
    if (correction.page_hash && sidecar.image_sha256 && correction.page_hash !== sidecar.image_sha256) {
        return 'made on a different image file';
    }
    if (correction.vlm_model && sidecar.vlm_model && correction.vlm_model !== sidecar.vlm_model) {
        return `made on ${correction.vlm_model}`;
    }
    if (correction.decoded_at && sidecar.decoded_at && !sameInstant(correction.decoded_at, sidecar.decoded_at)) {
        return 'made on an earlier decode';
    }
    return '';
}

/**
 * Suggested readings for this page, newest first. Clicking one selects its
 * final line when that line still exists: by index for a suggestion made on
 * this very read, else by box overlap (see ``findCorrectionLine``).
 * @param {object} props - Component props.
 * @param {Array<object>} props.corrections - Correction records.
 * @param {boolean} props.loading - Still fetching.
 * @param {string} props.error - Fetch error, or ''.
 * @param {Array<object>} props.finalLines - Normalised final lines.
 * @param {object} props.sidecar - Sidecar being shown.
 * @param {function(number): void} props.onSelect - Select (and centre) a final line by index.
 */
export default function CorrectionsList({ corrections, loading, error, finalLines, sidecar, onSelect }) {
    return (
        <section className="readers-section readers-corrections">
            <h3>Suggested readings{corrections.length > 0 ? ` (${corrections.length})` : ''}</h3>
            {loading && <p className="readers-muted">Loading suggestions…</p>}
            {error && <p className="readers-error">{error}</p>}
            {!loading && !error && corrections.length === 0 && (
                <p className="readers-muted">
                    No suggestions for this page yet.
                    {finalLines.length > 0 && ' Select a final line and use “Suggest a reading”.'}
                </p>
            )}
            <ol className="readers-correction-list">
                {corrections.map((correction, k) => {
                    const line = findCorrectionLine(correction, finalLines, sidecar);
                    const stale = staleNote(correction, sidecar);
                    const number = Number.isFinite(correction.line_index) ? correction.line_index + 1 : '?';
                    return (
                        <li
                            key={correction.id || k}
                            className={`readers-correction${line ? ' clickable' : ''}`}
                            onClick={line ? () => onSelect(line.index) : undefined}
                        >
                            <div className="readers-correction-head">
                                {line ? (
                                    <button type="button" className="readers-link-button" title="Show this line">
                                        Line {number}
                                    </button>
                                ) : (
                                    <span title="This line is not in the current reading">Line {number}</span>
                                )}
                                {correction.reader_id && correction.reader_id !== FINAL_KEY && (
                                    <code className="readers-id">{correction.reader_id}</code>
                                )}
                                <span className="readers-correction-time">{formatTime(correction.created_at)}</span>
                            </div>
                            <div className="readers-correction-text readers-hebrew" dir="auto">
                                {correction.corrected_text}
                            </div>
                            {correction.original_text && (
                                <del className="readers-correction-original readers-hebrew" dir="auto">
                                    {correction.original_text}
                                </del>
                            )}
                            {correction.note && (
                                <div className="readers-correction-note" dir="auto">{correction.note}</div>
                            )}
                            <div className="readers-correction-by">
                                — <span dir="auto">{correction.contributor || 'anonymous'}</span>
                                {stale && <span className="readers-correction-stale"> · {stale}</span>}
                                {!line && <span className="readers-correction-stale"> · line not in this reading</span>}
                            </div>
                        </li>
                    );
                })}
            </ol>
        </section>
    );
}
