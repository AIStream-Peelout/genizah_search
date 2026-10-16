import React, { useMemo, useState } from 'react';
import { HealthBadge, Swatch } from './ReaderList';
import { getStoredContributor, storeContributor } from './readingsApi';
import { bestMatches, formatNumber, isDegenerate, noLinesReason } from './readersModel';

/**
 * Kraken confidence of a line, when the reader gives one.
 * @param {object|null} line - Reader line.
 * @returns {string} Formatted confidence, or ''.
 */
const confText = (line) => (line && typeof line.conf === 'number' ? formatNumber(line.conf, 2) : '');

/**
 * Form for suggesting a better reading of the pinned final line. The text
 * starts as the machine reading; the contributor name is remembered.
 * @param {object} props - Component props.
 * @param {object} props.line - Pinned final line.
 * @param {function(object): Promise<{ok: boolean, message?: string}>} props.onSubmit -
 *     Saves ``{correctedText, note, contributor}``; the parent closes the form on success.
 * @param {function(): void} props.onCancel - Closes the form.
 */
function CorrectionForm({ line, onSubmit, onCancel }) {
    const [text, setText] = useState(line.text || '');
    const [note, setNote] = useState('');
    const [contributor, setContributor] = useState(getStoredContributor);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState('');

    const submit = (e) => {
        e.preventDefault();
        const name = contributor.trim();
        const corrected = text.trim();
        if (!name) {
            setError('Please give your name; it is shown with the suggestion.');
            return;
        }
        if (!corrected) {
            setError('The suggested reading is empty.');
            return;
        }
        if (corrected === (line.text || '').trim()) {
            setError('The suggestion is the same as the current reading.');
            return;
        }
        storeContributor(name);
        setBusy(true);
        setError('');
        onSubmit({ correctedText: corrected, note: note.trim(), contributor: name }).then((result) => {
            if (result.ok) return;
            setBusy(false);
            setError(result.message || 'The suggestion was not saved.');
        });
    };

    return (
        <form className="readers-form" onSubmit={submit}>
            <label className="readers-form-field">
                <span>Suggested reading</span>
                <textarea
                    className="readers-hebrew"
                    dir="auto"
                    rows={3}
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    disabled={busy}
                />
            </label>
            <label className="readers-form-field">
                <span>Note (optional)</span>
                <input type="text" dir="auto" value={note} onChange={(e) => setNote(e.target.value)} disabled={busy} />
            </label>
            <label className="readers-form-field">
                <span>Your name</span>
                <input
                    type="text"
                    dir="auto"
                    value={contributor}
                    onChange={(e) => setContributor(e.target.value)}
                    autoComplete="name"
                    required
                    disabled={busy}
                />
            </label>
            {error && <p className="readers-error">{error}</p>}
            <div className="readers-form-actions">
                <button type="submit" disabled={busy}>{busy ? 'Saving…' : 'Submit suggestion'}</button>
                <button type="button" className="readers-link-button" onClick={onCancel} disabled={busy}>
                    cancel
                </button>
            </div>
        </form>
    );
}

/**
 * Details of the pinned final line: its text, agreement, status and Kraken
 * reading, then, for every reader (shown or not), the line whose box overlaps
 * it most, with the IoU and Kraken confidence. A reader without lines says
 * why (its note, an unparsed reply, or nothing found).
 * @param {object} props - Component props.
 * @param {object} props.line - Pinned final line (normalised).
 * @param {Array<object>} props.entries - Entries from ``buildEntries``.
 * @param {boolean} props.formOpen - Whether the suggestion form is open.
 * @param {function(boolean): void} props.onFormOpen - Opens or closes the form.
 * @param {function(object): Promise<object>} props.onSubmitCorrection - Saves a suggestion.
 * @param {function(): void} props.onUnpin - Clears the pin.
 */
export default function LineDetail({ line, entries, formOpen, onFormOpen, onSubmitCorrection, onUnpin }) {
    const readers = useMemo(() => entries.filter((entry) => !entry.isFinal), [entries]);
    const matches = useMemo(() => bestMatches(line, readers), [line, readers]);
    const differs = line.htr_text && line.htr_text !== line.text;

    return (
        <section className="readers-section readers-detail">
            <div className="readers-section-head">
                <h3>Line {line.index + 1} (final)</h3>
                <button type="button" className="readers-link-button" onClick={onUnpin}>unpin</button>
            </div>
            <div className="readers-detail-text readers-hebrew" dir="auto">{line.text || '∅'}</div>
            <dl className="readers-detail-facts">
                <dt>Status</dt>
                <dd>{line.status || '—'}</dd>
                <dt>Agreement</dt>
                <dd>{formatNumber(line.agreement, 2)}</dd>
                <dt>Kraken</dt>
                <dd className={differs ? 'readers-differs' : ''}>
                    {line.htr_text ? (
                        <span className="readers-hebrew" dir="auto">{line.htr_text}</span>
                    ) : (
                        <em>no Kraken reading</em>
                    )}
                </dd>
                {Array.isArray(line.htr_fragments) && (
                    <>
                        <dt>Kraken fragments</dt>
                        <dd>{line.htr_fragments.length}</dd>
                    </>
                )}
                {isDegenerate(line.box) && (
                    <>
                        <dt>Box</dt>
                        <dd>zero height or width: the pipeline clamped a box that ran past the page edge (often a looping line)</dd>
                    </>
                )}
            </dl>

            {formOpen ? (
                <CorrectionForm
                    key={line.index}
                    line={line}
                    onSubmit={onSubmitCorrection}
                    onCancel={() => onFormOpen(false)}
                />
            ) : (
                <button type="button" className="readers-suggest" onClick={() => onFormOpen(true)}>
                    Suggest a reading
                </button>
            )}

            <h4 className="readers-subhead">Every reader on this line</h4>
            {matches.length === 0 && <p className="readers-muted">This page has no other readers.</p>}
            <ul className="readers-matches">
                {matches.map(({ reader: entry, line: match, iou: score }) => {
                    const reason = noLinesReason(entry);
                    return (
                        <li key={entry.key} className="readers-match">
                            <div className="readers-match-head">
                                <Swatch entry={entry} />
                                <span className="readers-row-label">{entry.label}</span>
                                <HealthBadge label={entry.health} />
                                {match && <span className="readers-match-score">IoU {formatNumber(score, 2)}</span>}
                                {confText(match) && <span className="readers-match-score">conf {confText(match)}</span>}
                            </div>
                            <code className="readers-id">{entry.id}</code>
                            {reason && <div className="readers-muted" dir="auto">{reason}</div>}
                            {!reason && !match && <div className="readers-muted">no overlapping line</div>}
                            {match && (
                                <div className="readers-match-text">
                                    <span className="readers-line-no">{match.index + 1}</span>
                                    <span className="readers-hebrew" dir="auto">{match.text || '∅'}</span>
                                </div>
                            )}
                        </li>
                    );
                })}
            </ul>
        </section>
    );
}
