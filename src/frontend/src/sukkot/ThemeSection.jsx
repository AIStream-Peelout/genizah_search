import React from 'react';
import FragmentCard, { genreColor } from './FragmentCard';
import Contested from './Contested';
import { RefMarks } from './Refs';

/**
 * DOM id of a theme section, the target of the themes index.
 * @param {string} themeId - Theme id.
 * @returns {string} Element id.
 */
export const themeAnchorId = (themeId) => `sk-theme-${themeId}`;

/**
 * One theme: title, intro paragraph with its references, open questions
 * (hedges), contested points, then its fragment cards in a grid.
 * @param {{theme: object, fragments: object[], genreLabel: string|null, numbers: Object<string, number>,
 *   highlightId: string|null, onOpenEsDocument: function(string): void,
 *   clusters: Object<string, object>, dayLabels: Object<string, string>}} props - Section props.
 * @returns {JSX.Element} The section.
 */
export default function ThemeSection({
    theme,
    fragments,
    genreLabel,
    numbers,
    highlightId,
    onOpenEsDocument,
    clusters,
    dayLabels,
}) {
    const hedges = (Array.isArray(theme.hedges) ? theme.hedges : []).filter((h) => typeof h === 'string' && h);
    const contested = Array.isArray(theme.contested) ? theme.contested : [];
    const headingId = `${themeAnchorId(theme.id)}-h`;

    return (
        <section
            className="sk-theme"
            id={themeAnchorId(theme.id)}
            aria-labelledby={headingId}
            style={{ '--sk-accent': genreColor(theme.genre) }}
        >
            <div className="sk-text">
                {genreLabel && <p className="sk-genre-label">{genreLabel}</p>}
                <h2 id={headingId}>{theme.title}</h2>
                {theme.intro && (
                    <p className="sk-intro" dir="auto">
                        {theme.intro}
                        <RefMarks keys={theme.refs} numbers={numbers} />
                    </p>
                )}
                {hedges.length > 0 && (
                    <div className="sk-hedges">
                        <h3>Open questions</h3>
                        <ul>
                            {hedges.map((h, i) => (
                                <li key={i} dir="auto">{h}</li>
                            ))}
                        </ul>
                    </div>
                )}
                {contested.map((item, i) => (
                    <Contested key={i} item={item} numbers={numbers} />
                ))}
            </div>
            {fragments.length > 0 ? (
                <div className="sk-grid">
                    {fragments.map((f) => (
                        <FragmentCard
                            key={f.doc_id}
                            fragment={f}
                            genre={theme.genre}
                            highlighted={highlightId === f.doc_id}
                            onOpenEsDocument={onOpenEsDocument}
                            clusters={clusters}
                            dayLabels={dayLabels}
                        />
                    ))}
                </div>
            ) : (
                <p className="sk-text sk-small sk-empty">No fragments for this theme are shown yet.</p>
            )}
        </section>
    );
}
