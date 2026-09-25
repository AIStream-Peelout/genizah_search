import React from 'react';

/**
 * DOM id of one entry in the references list.
 * @param {string} key - Reference key, e.g. ``stern2019``.
 * @returns {string} Element id.
 */
export const refAnchorId = (key) => `sk-ref-${key}`;

/**
 * Normalise ``data.refs`` into a key → citation map. The contract is an object
 * of strings; an array of ``{id, citation}`` is accepted too.
 * @param {object|Array|undefined} refs - ``data.refs``.
 * @returns {Object<string, string>} Citation text by key.
 */
export function normaliseRefs(refs) {
    if (Array.isArray(refs)) {
        const out = {};
        refs.forEach((r) => {
            if (r && r.id) out[r.id] = r.citation || r.text || String(r.id);
        });
        return out;
    }
    return refs && typeof refs === 'object' ? refs : {};
}

/**
 * Superscript-style numbered links to the references list. Unknown keys
 * (not in ``numbers``) are skipped rather than shown as broken links.
 * @param {{keys: string[], numbers: Object<string, number>}} props - Reference keys and their list numbers.
 * @returns {JSX.Element|null} The marks, or null when none resolve.
 */
export function RefMarks({ keys, numbers }) {
    const known = (Array.isArray(keys) ? keys : []).filter((k) => numbers[k]);
    if (known.length === 0) return null;
    return (
        <sup className="sk-refmarks">
            {known.map((k, i) => (
                <React.Fragment key={k}>
                    {i > 0 && ','}
                    <a href={`#${refAnchorId(k)}`} className="sk-refmark" aria-label={`Reference ${numbers[k]}`}>
                        {numbers[k]}
                    </a>
                </React.Fragment>
            ))}
        </sup>
    );
}

/**
 * Numbered list of the references cited on the page, in order of first use.
 * @param {{order: string[], refs: Object<string, string>}} props - Keys in citation order and the citation map.
 * @returns {JSX.Element} The references section.
 */
export default function Refs({ order, refs }) {
    return (
        <section className="sk-refs sk-text" id="sk-references" aria-labelledby="sk-references-h">
            <h2 id="sk-references-h">References</h2>
            <ol>
                {order.map((key) => (
                    <li key={key} id={refAnchorId(key)} dir="auto">
                        {refs[key]}
                    </li>
                ))}
            </ol>
        </section>
    );
}
