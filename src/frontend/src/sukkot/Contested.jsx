import React from 'react';
import { RefMarks } from './Refs';

/**
 * Normalise one contested item. The data may give ``{claim, positions}`` where
 * each position is ``{view, refs}`` or a plain string, or the whole item may
 * be a plain string.
 * @param {object|string} item - One entry of ``theme.contested``.
 * @returns {{claim: string, positions: {view: string, refs: string[]}[]}} Normalised item.
 */
export function normaliseContested(item) {
    if (typeof item === 'string') return { claim: item, positions: [] };
    if (!item || typeof item !== 'object') return { claim: '', positions: [] };
    const positions = (Array.isArray(item.positions) ? item.positions : [])
        .map((p) => {
            if (typeof p === 'string') return { view: p, refs: [] };
            if (!p || typeof p !== 'object') return null;
            return { view: String(p.view || p.position || ''), refs: Array.isArray(p.refs) ? p.refs : [] };
        })
        .filter((p) => p && p.view);
    return { claim: String(item.claim || item.question || ''), positions };
}

/**
 * A contested point set out with both (or all) scholarly positions, each with
 * its references as numbered links.
 * @param {{item: object|string, numbers: Object<string, number>}} props - The contested item and reference numbers.
 * @returns {JSX.Element|null} The callout, or null for an empty item.
 */
export default function Contested({ item, numbers }) {
    const { claim, positions } = normaliseContested(item);
    if (!claim && positions.length === 0) return null;
    return (
        <aside className="sk-contested" aria-label={claim ? `Contested: ${claim}` : 'Contested point'}>
            <p className="sk-contested-claim">
                <span className="sk-contested-tag">Contested</span>
                {claim && <span dir="auto">{claim}</span>}
            </p>
            {positions.length > 0 && (
                <ul className="sk-contested-positions">
                    {positions.map((p, i) => (
                        <li key={i} dir="auto">
                            {p.view}
                            <RefMarks keys={p.refs} numbers={numbers} />
                        </li>
                    ))}
                </ul>
            )}
        </aside>
    );
}
