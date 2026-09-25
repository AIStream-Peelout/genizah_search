import { normaliseRefs } from './Refs';
import { normaliseContested } from './Contested';

/*
 * Pure helpers that turn the static page JSON (data/sukkot_5787.json) into the
 * lists and lookups the page renders. No React, no network.
 */

/**
 * Accept a list of ``{id, ...}`` objects or an object keyed by id, and return
 * a list sorted by ``order`` (stable for missing orders).
 * @param {Array|object|undefined} x - ``data.themes``, ``data.genres`` or ``data.clusters``.
 * @returns {object[]} Records with an ``id``.
 */
export function asOrderedList(x) {
    const list = Array.isArray(x)
        ? x.filter((r) => r && r.id)
        : Object.entries(x || {}).map(([id, v]) => ({ id, ...(v || {}) }));
    return list
        .map((r, i) => ({ r, i }))
        .sort((a, b) => (a.r.order ?? 1e9) - (b.r.order ?? 1e9) || a.i - b.i)
        .map(({ r }) => r);
}

/**
 * Give each theme its fragments, each fragment appearing once (in the first
 * theme, by order, that lists it). A theme's ``fragment_ids`` sets the order;
 * without it, fragments whose ``theme`` matches are used.
 * @param {object[]} themes - Ordered themes.
 * @param {object[]} fragments - Visible fragments.
 * @returns {Object<string, object[]>} Fragments by theme id.
 */
export function assignFragments(themes, fragments) {
    const byId = {};
    fragments.forEach((f) => {
        byId[f.doc_id] = f;
    });
    const used = new Set();
    const out = {};
    themes.forEach((t) => {
        const ids = Array.isArray(t.fragment_ids) && t.fragment_ids.length > 0
            ? t.fragment_ids
            : fragments.filter((f) => f.theme === t.id).map((f) => f.doc_id);
        out[t.id] = ids.filter((id) => byId[id] && !used.has(id)).map((id) => byId[id]);
        out[t.id].forEach((f) => used.add(f.doc_id));
    });
    return out;
}

/**
 * Reference keys in order of first citation (theme refs, then contested
 * positions), keeping only keys present in the references map.
 * @param {object[]} themes - Ordered themes.
 * @param {Object<string, string>} refs - Citation map.
 * @returns {{order: string[], numbers: Object<string, number>}} Citation order and 1-based numbers.
 */
export function collectRefs(themes, refs) {
    const order = [];
    const add = (k) => {
        if (typeof k === 'string' && refs[k] && !order.includes(k)) order.push(k);
    };
    themes.forEach((t) => {
        (Array.isArray(t.refs) ? t.refs : []).forEach(add);
        (Array.isArray(t.contested) ? t.contested : []).forEach((c) =>
            normaliseContested(c).positions.forEach((p) => p.refs.forEach(add))
        );
    });
    const numbers = {};
    order.forEach((k, i) => {
        numbers[k] = i + 1;
    });
    return { order, numbers };
}

/**
 * Build the page model once from the static JSON.
 * @param {object} d - The imported page data.
 * @returns {object} Themes, fragments, lookups and reference numbering.
 */
export function buildModel(d) {
    const themes = asOrderedList(d.themes);
    const genres = asOrderedList(d.genres);
    const themeGenre = {};
    themes.forEach((t) => {
        themeGenre[t.id] = t.genre;
    });
    const fragments = (Array.isArray(d.fragments) ? d.fragments : [])
        .filter((f) => f && f.doc_id && f.usable_now !== false)
        .map((f) => ({ ...f, genre: f.genre || themeGenre[f.theme] }));
    const clusters = {};
    asOrderedList(d.clusters).forEach((c) => {
        clusters[c.id] = c;
    });
    const genreLabels = {};
    genres.forEach((g) => {
        genreLabels[g.id] = g.label;
    });
    const days = (d.festival && Array.isArray(d.festival.days) ? d.festival.days : []).filter((x) => x && x.id);
    const dayLabels = {};
    days.forEach((x) => {
        dayLabels[x.id] = x.label;
    });
    const refs = normaliseRefs(d.refs);
    const byTheme = assignFragments(themes, fragments);
    const onPage = new Set(Object.values(byTheme).flat().map((f) => f.doc_id));
    return { themes, fragments, clusters, genreLabels, days, dayLabels, refs, byTheme, onPage, ...collectRefs(themes, refs) };
}
