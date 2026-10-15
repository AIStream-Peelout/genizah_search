import React, { useMemo, useState } from 'react';
import { FINAL_KEY, alignLines, charDiff, formatNumber, lineCountLabel } from './readersModel';

/**
 * One side of a diffed line: shared characters plain, this side's own
 * characters highlighted (deletions on the left, insertions on the right).
 * @param {object} props - Component props.
 * @param {Array<{type: string, text: string}>} props.segments - ``charDiff`` output.
 * @param {'a'|'b'} props.side - Which side to render.
 */
function DiffText({ segments, side }) {
    return (
        <span className="readers-hebrew" dir="auto">
            {segments.map((segment, k) => {
                if (segment.type === 'same') return <span key={k}>{segment.text}</span>;
                if (segment.type === 'del' && side === 'a') {
                    return <del key={k} className="readers-del">{segment.text}</del>;
                }
                if (segment.type === 'ins' && side === 'b') {
                    return <ins key={k} className="readers-ins">{segment.text}</ins>;
                }
                return null;
            })}
        </span>
    );
}

/**
 * One aligned row: both lines with a character diff, or a lone line.
 * @param {{row: {a: object|null, b: object|null, iou: number}}} props - Row from ``alignLines``.
 */
function CompareRow({ row }) {
    const segments = row.a && row.b ? charDiff(row.a.text, row.b.text) : null;
    const identical = segments && segments.every((s) => s.type === 'same');
    return (
        <li className={`readers-compare-row${identical ? ' identical' : ''}`}>
            <div className="readers-compare-cell">
                {row.a ? (
                    <>
                        <span className="readers-line-no">{row.a.index + 1}</span>
                        {segments ? (
                            <DiffText segments={segments} side="a" />
                        ) : (
                            <span className="readers-hebrew readers-alone" dir="auto">{row.a.text || '∅'}</span>
                        )}
                    </>
                ) : (
                    <span className="readers-muted">—</span>
                )}
            </div>
            <div className="readers-compare-iou">{row.a && row.b ? formatNumber(row.iou, 2) : ''}</div>
            <div className="readers-compare-cell">
                {row.b ? (
                    <>
                        <span className="readers-line-no">{row.b.index + 1}</span>
                        {segments ? (
                            <DiffText segments={segments} side="b" />
                        ) : (
                            <span className="readers-hebrew readers-alone" dir="auto">{row.b.text || '∅'}</span>
                        )}
                    </>
                ) : (
                    <span className="readers-muted">—</span>
                )}
            </div>
        </li>
    );
}

/**
 * Select one reader (or the final row) for a side of the comparison.
 * @param {object} props - Component props.
 * @param {string} props.label - Field label.
 * @param {string} props.value - Selected entry key.
 * @param {Array<object>} props.entries - Entries from ``buildEntries``.
 * @param {function(string): void} props.onChange - New entry key.
 */
function ReaderSelect({ label, value, entries, onChange }) {
    return (
        <label className="readers-compare-select">
            <span>{label}</span>
            <select value={value} onChange={(e) => onChange(e.target.value)}>
                {entries.map((entry) => (
                    <option key={entry.key} value={entry.key}>
                        {entry.label} · {lineCountLabel(entry.lines)}
                        {entry.isFinal ? '' : ` · ${entry.id}`}
                    </option>
                ))}
            </select>
        </label>
    );
}

/**
 * Compare two readers line by line: lines are paired by box overlap (greedy
 * best IoU, one-to-one), ordered top to bottom, and each pair is diffed
 * character by character. Work happens only while the section is open.
 * @param {{entries: Array<object>}} props - Entries from ``buildEntries``.
 */
export default function CompareReaders({ entries }) {
    const [open, setOpen] = useState(false);
    const [aKey, setAKey] = useState(FINAL_KEY);
    const [bKey, setBKey] = useState('');

    const a = entries.find((entry) => entry.key === aKey) || entries[0] || null;
    const b = entries.find((entry) => entry.key === bKey) || entries.find((entry) => entry !== a) || null;

    const rows = useMemo(() => {
        if (!open || !a || !b || !a.lines || !b.lines) return [];
        return alignLines(a.lines, b.lines);
    }, [open, a, b]);

    /**
     * Why a side has nothing to align, if so.
     * @param {object|null} entry - Selected entry.
     * @returns {string} Message, or ''.
     */
    const emptyReason = (entry) => {
        if (!entry) return '';
        if (entry.lines === null) return `${entry.label}: reply could not be attributed to lines.`;
        if (entry.lines.length === 0) return `${entry.label}: no lines.`;
        return '';
    };

    return (
        <details className="readers-section readers-compare" open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
            <summary>Compare two readers</summary>
            {entries.length < 2 ? (
                <p className="readers-muted">This page has only one reader.</p>
            ) : (
                <>
                    <div className="readers-compare-controls">
                        <ReaderSelect label="Left" value={a ? a.key : ''} entries={entries} onChange={setAKey} />
                        <ReaderSelect label="Right" value={b ? b.key : ''} entries={entries} onChange={setBKey} />
                    </div>
                    {[emptyReason(a), emptyReason(b)].filter(Boolean).map((reason) => (
                        <p key={reason} className="readers-muted">{reason}</p>
                    ))}
                    {rows.length > 0 && (
                        <>
                            <p className="readers-muted readers-compare-legend">
                                Lines paired by box overlap (IoU in the middle).{' '}
                                <del className="readers-del">only left</del>{' '}
                                <ins className="readers-ins">only right</ins>
                            </p>
                            <ol className="readers-compare-rows">
                                {rows.map((row, k) => (
                                    <CompareRow key={k} row={row} />
                                ))}
                            </ol>
                        </>
                    )}
                </>
            )}
        </details>
    );
}
