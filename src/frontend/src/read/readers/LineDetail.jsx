import React, { useMemo, useState } from 'react';
import { HealthBadge, Swatch } from './ReaderList';
import { getStoredContributor, storeContributor } from './readingsApi';
import { bestMatches, formatNumber, isDegenerate, lineColour, noLinesReason } from './readersModel';

/**
 * Kraken confidence of a line, when the reader gives one.
 * @param {object|null} line - Reader line.
 * @returns {string} Formatted confidence, or ''.
 */
const confText = (line) => (line && typeof line.conf === 'number' ? formatNumber(line.conf, 2) : '');

/**
 * Why a line's box looks odd on the image, when it does.
 * @param {object} line - Normalised line.
 * @returns {string} A note for a missing or zero-area box, else ''.
 */
const boxNote = (line) => {
    if (!line.box) return 'none: this line has no box on the image';
    if (isDegenerate(line.box)) {
        return 'zero height or width: the pipeline clamped a box that ran past the page edge (often a looping line)';
    }
    return '';
};

/**
 * Form for suggesting a better reading of the selected final line. The text
 * starts as the machine reading; the contributor name is remembered.
 * @param {object} props - Component props.
 * @param {object} props.line - Selected final line.
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
 * Facts of a selected final line: status, agreement, Kraken reading and
 * fragments, and why its box looks odd when it does.
 * @param {{line: object}} props - Selected final line (normalised).
 */
function FinalFacts({ line }) {
    const differs = line.htr_text && line.htr_text !== line.text;
    const note = boxNote(line);
    return (
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
            {note && (
                <>
                    <dt>Box</dt>
                    <dd>{note}</dd>
                </>
            )}
        </dl>
    );
}

/**
 * Facts of a selected reader line: the reader's raw id and health, the
 * line's confidence when the reader gives one, the reader's note, and why
 * the box looks odd when it does.
 * @param {object} props - Component props.
 * @param {object} props.entry - The reader's entry from ``buildEntries``.
 * @param {object} props.line - Selected line (normalised).
 */
function ReaderFacts({ entry, line }) {
    const conf = confText(line);
    const note = boxNote(line);
    return (
        <dl className="readers-detail-facts">
            <dt>Reader</dt>
            <dd>
                <code className="readers-id">{entry.id}</code>
            </dd>
            <dt>Health</dt>
            <dd>{entry.health ? <HealthBadge label={entry.health} /> : '—'}</dd>
            {conf && (
                <>
                    <dt>{entry.reader && entry.reader.kind === 'htr' ? 'Kraken conf' : 'Conf'}</dt>
                    <dd>{conf}</dd>
                </>
            )}
            {entry.note && (
                <>
                    <dt>Note</dt>
                    <dd dir="auto">{entry.note}</dd>
                </>
            )}
            {note && (
                <>
                    <dt>Box</dt>
                    <dd>{note}</dd>
                </>
            )}
        </dl>
    );
}

/**
 * Details of the selected line, of the final row or of any reader: its text
 * and facts, then, for every other entry (shown or not), the line whose box
 * overlaps it most, with the IoU and Kraken confidence; each of those lines
 * selects itself when clicked. A final line can get a suggested reading; a
 * reader line offers the final line overlapping it most instead. An entry
 * without lines says why (its note, an unparsed reply, or nothing found).
 * @param {object} props - Component props.
 * @param {object} props.entry - Entry of the selected line (``buildEntries``).
 * @param {object} props.line - Selected line (normalised).
 * @param {Array<object>} props.entries - Entries from ``buildEntries``.
 * @param {boolean} props.formOpen - Whether the suggestion form is open.
 * @param {function(boolean): void} props.onFormOpen - Opens or closes the form.
 * @param {function(object): Promise<object>} props.onSubmitCorrection - Saves a suggestion.
 * @param {function(string, number): void} props.onSelect - Select (and centre) a line by entry key and index.
 * @param {Array<{entry: object, line: object}>} [props.alsoHere] - The other boxes under the click
 *     that selected this line, in click order.
 * @param {function(string, number): void} [props.onSelectHere] - Switch to one of those boxes.
 * @param {function(): void} props.onClear - Clears the selection.
 */
export default function LineDetail({
    entry,
    line,
    entries,
    formOpen,
    onFormOpen,
    onSubmitCorrection,
    onSelect,
    alsoHere,
    onSelectHere,
    onClear,
}) {
    const others = useMemo(() => entries.filter((other) => other.key !== entry.key), [entries, entry]);
    const matches = useMemo(() => bestMatches(line, others), [line, others]);
    // For a reader line, the final line overlapping it most (others holds the final row then).
    const finalMatch = entry.isFinal ? null : matches.find((m) => m.reader.isFinal && m.line) || null;

    return (
        <section className="readers-section readers-detail">
            <div className="readers-section-head">
                <h3 className="readers-detail-title">
                    <Swatch entry={entry} />
                    <span>
                        Line {line.index + 1} · {entry.label}
                    </span>
                </h3>
                <button type="button" className="readers-link-button" onClick={onClear} title="Clear the selection (Esc)">
                    unselect
                </button>
            </div>
            <div
                className="readers-detail-text readers-hebrew"
                dir="auto"
                style={{ borderRightColor: lineColour(entry, line) }}
            >
                {line.text || '∅'}
            </div>
            {alsoHere && alsoHere.length > 0 && onSelectHere && (
                <div className="readers-also-here">
                    <span className="readers-muted">Also under this click:</span>
                    {alsoHere.map((hit) => (
                        <button
                            type="button"
                            key={`${hit.entry.key}:${hit.line.index}`}
                            className="readers-also-here-item"
                            title={hit.line.text || ''}
                            onClick={() => onSelectHere(hit.entry.key, hit.line.index)}
                        >
                            <Swatch entry={hit.entry} />
                            <span>
                                {hit.entry.label} · line {hit.line.index + 1}
                            </span>
                        </button>
                    ))}
                </div>
            )}

            {entry.isFinal ? (
                <>
                    <FinalFacts line={line} />
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
                </>
            ) : (
                <>
                    <ReaderFacts entry={entry} line={line} />
                    <p className="readers-muted readers-detail-note">
                        Suggestions are made on final lines.
                        {!finalMatch && ' No final line overlaps this one.'}
                    </p>
                    {finalMatch && (
                        <button
                            type="button"
                            className="readers-goto-final"
                            title={`Final line ${finalMatch.line.index + 1}`}
                            onClick={() => onSelect(finalMatch.reader.key, finalMatch.line.index)}
                        >
                            Go to the final line
                        </button>
                    )}
                </>
            )}

            <h4 className="readers-subhead">Every reader on this line</h4>
            {matches.length === 0 && <p className="readers-muted">This page has no other readers.</p>}
            <ul className="readers-matches">
                {matches.map(({ reader: other, line: match, iou: score }) => {
                    const reason = noLinesReason(other);
                    return (
                        <li key={other.key} className="readers-match">
                            <div className="readers-match-head">
                                <Swatch entry={other} />
                                <span className="readers-row-label">{other.label}</span>
                                <HealthBadge label={other.health} />
                                {match && <span className="readers-match-score">IoU {formatNumber(score, 2)}</span>}
                                {confText(match) && <span className="readers-match-score">conf {confText(match)}</span>}
                            </div>
                            {!other.isFinal && <code className="readers-id">{other.id}</code>}
                            {reason && <div className="readers-muted" dir="auto">{reason}</div>}
                            {!reason && !match && <div className="readers-muted">no overlapping line</div>}
                            {match && (
                                <button
                                    type="button"
                                    className="readers-match-text readers-match-line"
                                    title="Select this line"
                                    onClick={() => onSelect(other.key, match.index)}
                                >
                                    <span className="readers-line-no">{match.index + 1}</span>
                                    <span className="readers-hebrew" dir="auto">{match.text || '∅'}</span>
                                </button>
                            )}
                        </li>
                    );
                })}
            </ul>
        </section>
    );
}
