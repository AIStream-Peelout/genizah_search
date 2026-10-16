import React from 'react';
import { healthTone, lineCountLabel, swatchBackground } from './readersModel';

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
 * One row per reader with a show/hide checkbox. The final consensus is the
 * first row; every row shows its generic label, line count and health, and
 * readers also show their raw id and, when the sidecar gives one, the note
 * on why their lines are missing.
 * @param {object} props - Component props.
 * @param {Array<object>} props.entries - Entries from ``buildEntries``.
 * @param {Object<string, boolean>} props.visible - Visibility per entry key.
 * @param {function(string, boolean): void} props.onToggle - Show or hide one entry.
 * @param {function(boolean): void} props.onToggleAll - Show (true) or hide (false) all.
 */
export default function ReaderList({ entries, visible, onToggle, onToggleAll }) {
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
                {entries.map((entry) => (
                    <li key={entry.key} className={`readers-row${entry.isFinal ? ' final' : ''}`}>
                        <label
                            className="readers-row-main"
                            title={entry.isFinal ? 'Green: two readers agree. Amber: single reader, unconfirmed.' : undefined}
                        >
                            <input
                                type="checkbox"
                                checked={Boolean(visible[entry.key])}
                                onChange={(e) => onToggle(entry.key, e.target.checked)}
                            />
                            <Swatch entry={entry} />
                            <span className="readers-row-label">{entry.label}</span>
                        </label>
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
                    </li>
                ))}
            </ul>
        </section>
    );
}
