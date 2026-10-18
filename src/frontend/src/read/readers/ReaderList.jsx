import React, { useEffect, useState } from 'react';
import { healthTone, lineCountLabel, noLinesReason, swatchBackground } from './readersModel';

/**
 * Health label as a coloured badge: ok green, loop/overgen red, the known
 * soft failures amber, anything new grey.
 * @param {{label: string|null}} props - ``health_label`` of a reader or unit.
 */
export function HealthBadge({ label }) {
    if (!label) return null;
    return <span className={`readers-badge ${healthTone(label)}`}>{label}</span>;
}

/**
 * Colour swatch of a reader (the final row is half green, half amber).
 * @param {{entry: object}} props - Entry from ``buildEntries``.
 */
export function Swatch({ entry }) {
    return <span className="readers-swatch" style={{ background: swatchBackground(entry) }} aria-hidden="true" />;
}

/**
 * The lines of one entry as a clickable list, opened under its row. An entry
 * without lines says why instead.
 * @param {object} props - Component props.
 * @param {object} props.entry - Entry from ``buildEntries``.
 * @param {number|null} props.selectedIndex - Selected line of this entry, else null.
 * @param {function(string, number): void} props.onSelect - Select (and centre) a line by entry key and index.
 */
function EntryLines({ entry, selectedIndex, onSelect }) {
    const reason = noLinesReason(entry);
    if (reason) {
        return (
            <div className="readers-row-lines-empty readers-muted" dir="auto">
                No lines to list: {reason}
            </div>
        );
    }
    return (
        <ol className="readers-row-lines">
            {entry.lines.map((line, i) => (
                <li key={`${line.index}-${i}`} className={line.index === selectedIndex ? 'selected' : undefined}>
                    <button type="button" onClick={() => onSelect(entry.key, line.index)}>
                        <span className="readers-line-no">{line.index + 1}</span>
                        <span className="readers-hebrew" dir="auto">{line.text || '∅'}</span>
                    </button>
                </li>
            ))}
        </ol>
    );
}

/**
 * One row per reader with a show/hide checkbox. The final consensus is the
 * first row; every row shows its generic label, line count and health, and
 * readers also show their raw id and, when the sidecar gives one, the note
 * on why their lines are missing. Clicking a row's name opens the list of
 * its lines (one row at a time); clicking a line there selects it.
 * @param {object} props - Component props.
 * @param {Array<object>} props.entries - Entries from ``buildEntries``.
 * @param {Object<string, boolean>} props.visible - Visibility per entry key.
 * @param {{entry: object, line: object}|null} props.selected - Selected line and its entry.
 * @param {function(string, boolean): void} props.onToggle - Show or hide one entry.
 * @param {function(boolean): void} props.onToggleAll - Show (true) or hide (false) all.
 * @param {function(string, number): void} props.onSelect - Select (and centre) a line by entry key and index.
 */
export default function ReaderList({ entries, visible, selected, onToggle, onToggleAll, onSelect }) {
    const [expandedKey, setExpandedKey] = useState(null);

    // A new sidecar (page or model) closes the open line list.
    useEffect(() => {
        setExpandedKey(null);
    }, [entries]);

    return (
        <section className="readers-section">
            <div className="readers-section-head">
                <h3>Readers</h3>
                <span className="readers-section-actions">
                    <button type="button" className="readers-link-button" onClick={() => onToggleAll(true)}>
                        show all
                    </button>
                    {' · '}
                    <button type="button" className="readers-link-button" onClick={() => onToggleAll(false)}>
                        hide all
                    </button>
                </span>
            </div>
            <ul className="readers-list">
                {entries.map((entry) => {
                    const expanded = entry.key === expandedKey;
                    const selectedIndex = selected && selected.entry.key === entry.key ? selected.line.index : null;
                    return (
                        <li key={entry.key} className={`readers-row${entry.isFinal ? ' final' : ''}`}>
                            <span className="readers-row-main">
                                <label
                                    className="readers-row-toggle"
                                    title={
                                        entry.isFinal
                                            ? 'Green: two readers agree. Amber: single reader, unconfirmed.'
                                            : 'Show or hide on the image'
                                    }
                                >
                                    <input
                                        type="checkbox"
                                        checked={Boolean(visible[entry.key])}
                                        onChange={(e) => onToggle(entry.key, e.target.checked)}
                                        aria-label={`Show ${entry.label} on the image`}
                                    />
                                    <Swatch entry={entry} />
                                </label>
                                <button
                                    type="button"
                                    className="readers-row-name"
                                    aria-expanded={expanded}
                                    title={expanded ? 'Hide its lines' : 'List its lines'}
                                    onClick={() => setExpandedKey((key) => (key === entry.key ? null : entry.key))}
                                >
                                    <span className="readers-row-caret" aria-hidden="true">{expanded ? '▾' : '▸'}</span>
                                    <span className="readers-row-label">{entry.label}</span>
                                </button>
                            </span>
                            <span
                                className={`readers-row-count${
                                    entry.lines === null || entry.parsed === false ? ' unattributed' : ''
                                }`}
                            >
                                {lineCountLabel(entry.lines, entry.parsed)}
                            </span>
                            <HealthBadge label={entry.health} />
                            {!entry.isFinal && <code className="readers-id">{entry.id}</code>}
                            {!entry.isFinal && entry.note && (
                                <div className="readers-row-note readers-muted" dir="auto">{entry.note}</div>
                            )}
                            {expanded && <EntryLines entry={entry} selectedIndex={selectedIndex} onSelect={onSelect} />}
                        </li>
                    );
                })}
            </ul>
        </section>
    );
}
