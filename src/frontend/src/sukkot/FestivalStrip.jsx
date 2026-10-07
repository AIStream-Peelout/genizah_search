import React from 'react';
import { genreColor } from './FragmentCard';

/**
 * Split fragments into per-day chips and contested brackets. A fragment whose
 * basis is "contested" and whose ``span`` names two known days becomes one
 * bracket; any other fragment with a known day becomes a chip in that column.
 * @param {object[]} days - ``data.festival.days``.
 * @param {object[]} fragments - Visible fragments.
 * @returns {{byDay: Object<string, object[]>, brackets: {f: object, from: number, to: number, other: number}[]}}
 *   Chips by day id; brackets with their first/last column and the column other than the fragment's own day.
 */
export function placeOnStrip(days, fragments) {
    const index = {};
    days.forEach((d, i) => {
        index[d.id] = i;
    });
    const byDay = {};
    days.forEach((d) => {
        byDay[d.id] = [];
    });
    const brackets = [];
    fragments.forEach((f) => {
        const fd = f.festival_day;
        if (!fd || !(fd.day in index)) return;
        const span = Array.isArray(fd.span) ? fd.span.filter((id) => id in index) : [];
        if (fd.basis === 'contested' && span.length >= 2) {
            const cols = span.map((id) => index[id]);
            const others = cols.filter((c) => c !== index[fd.day]);
            brackets.push({
                f,
                from: Math.min(...cols),
                to: Math.max(...cols),
                other: others.length > 0 ? others[others.length - 1] : Math.max(...cols),
            });
        }
        byDay[fd.day].push(f);
    });
    return { byDay, brackets };
}

/**
 * One chip button for a fragment in the strip.
 * @param {{f: object, onSelect: function(string): void, extraLabel: string|null, className: string}} props - Chip props.
 * @returns {JSX.Element} The chip.
 */
function Chip({ f, onSelect, extraLabel = null, className = '' }) {
    const basis = f.festival_day && f.festival_day.basis;
    const kind = basis === 'placed by theme' ? 'theme' : basis === 'contested' ? 'contested' : 'named';
    return (
        <button
            type="button"
            className={`sk-chip sk-chip--${kind} ${className}`}
            style={{ '--sk-accent': genreColor(f.genre) }}
            title={f.card_line || undefined}
            onClick={() => onSelect(f.doc_id)}
        >
            <span className="sk-chip-shelf">{f.shelfmark}</span>
            {extraLabel && <span className="sk-chip-extra">{extraLabel}</span>}
        </button>
    );
}

/**
 * The festival-days strip: one column per day from the preparations to Simhat
 * Torah, with a chip for each fragment that concerns that day. Contested
 * placements are drawn once, as a bracket across the columns in dispute (on
 * phones, as a labelled chip in the first of them).
 * @param {{days: object[], fragments: object[], onSelect: function(string): void}} props - Strip props.
 * @returns {JSX.Element|null} The strip, or null when there are no days.
 */
export default function FestivalStrip({ days, fragments, onSelect }) {
    if (!Array.isArray(days) || days.length === 0) return null;
    const { byDay, brackets } = placeOnStrip(days, fragments);
    const bracketOf = {};
    brackets.forEach((b) => {
        bracketOf[b.f.doc_id] = b;
    });
    const labelOf = (i) => (days[i] && days[i].label) || '';

    return (
        <section className="sk-strip-section" aria-labelledby="sk-strip-h">
            <h2 id="sk-strip-h" className="sk-text">The festival, day by day</h2>
            <p className="sk-legend sk-text">
                <span className="sk-legend-item">
                    <span className="sk-chip-sample sk-chip--named" aria-hidden="true" /> Named in the source: the
                    fragment itself names the day or is dated by it.
                </span>
                <span className="sk-legend-item">
                    <span className="sk-chip-sample sk-chip--theme" aria-hidden="true" /> Placed by theme: the day its
                    liturgy or subject belongs to, not a date.
                </span>
            </p>
            <div className="sk-strip-scroll" tabIndex={0} role="region" aria-label="Festival days (scrolls sideways on small screens)">
                <div className="sk-strip" style={{ '--sk-ncols': days.length }}>
                    {days.map((d, i) => (
                        <div key={d.id} className="sk-day" style={{ gridColumn: i + 1, gridRow: 1 }}>
                            <div className="sk-day-head">
                                <strong>{d.label}</strong>
                                {d.sublabel && <span>{d.sublabel}</span>}
                                {d.gregorian && <span className="sk-day-greg">{d.gregorian}</span>}
                            </div>
                            <ul className="sk-day-chips">
                                {byDay[d.id].map((f) =>
                                    bracketOf[f.doc_id] ? (
                                        <li key={f.doc_id} className="sk-phone-only">
                                            <Chip
                                                f={f}
                                                onSelect={onSelect}
                                                extraLabel={`contested: also placed on ${labelOf(bracketOf[f.doc_id].other)}`}
                                            />
                                        </li>
                                    ) : (
                                        <li key={f.doc_id}>
                                            <Chip f={f} onSelect={onSelect} />
                                        </li>
                                    )
                                )}
                                {byDay[d.id].length === 0 && <li className="sk-day-empty">None shown yet</li>}
                            </ul>
                        </div>
                    ))}
                    {brackets.map((b, k) => (
                        <div
                            key={b.f.doc_id}
                            className="sk-bracket sk-desktop-only"
                            style={{ gridColumn: `${b.from + 1} / ${b.to + 2}`, gridRow: k + 2 }}
                        >
                            <div className="sk-bracket-bar" aria-hidden="true">
                                <span className="sk-bracket-end">{labelOf(b.from)}</span>
                                <span className="sk-bracket-end">{labelOf(b.to)}</span>
                            </div>
                            <div className="sk-bracket-body">
                                <Chip f={b.f} onSelect={onSelect} />
                                <span className="sk-bracket-label" dir="auto">
                                    {b.f.festival_day.label || `Contested: ${labelOf(b.from)} or ${labelOf(b.to)}`}
                                </span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
}
