/**
 * Display helpers shared by the search result cards and the document modal,
 * so both read the backend's DocumentMetadata fields the same way:
 * ``shelf_mark``, ``alt_titles``, ``date_display`` / ``century``,
 * ``document_type``, ``genres`` / ``subjects`` and ``source_links``.
 *
 * The backend already cleans these fields (placeholders, provider mentions,
 * blocked hosts, caps); nothing here invents a value. Pure functions only.
 */

import { isCatalogueIndex } from '../utils';

/** A description shorter than this (in characters) is too thin to stand alone. */
export const THIN_DESCRIPTION_CHARS = 40;

/** Link kinds the backend sends in ``source_links``. */
const SOURCE_LINK_KINDS = ['item', 'catalogue', 'search'];

/**
 * Trimmed text of a string field.
 * @param {unknown} value - A metadata value.
 * @returns {string} The trimmed string, or '' when it is not a string.
 */
const text = (value) => (typeof value === 'string' ? value.trim() : '');

/**
 * Lower-cased text with runs of whitespace collapsed, for loose comparisons.
 * @param {string} value - Text to fold.
 * @returns {string} The folded text.
 */
const foldText = (value) => value.toLowerCase().replace(/\s+/g, ' ');

/**
 * Trimmed, non-empty strings of a list field.
 * @param {unknown} value - A metadata list such as ``alt_titles``.
 * @returns {string[]} The usable entries in order; [] when it is not a list.
 */
export const textList = (value) => (Array.isArray(value) ? value : [])
    .filter((item) => typeof item === 'string' && item.trim())
    .map((item) => item.trim());

/**
 * Heading of a record: its shelf mark, else its title, else the fallback.
 * Never an invented "Document <id>" label.
 * @param {object|null|undefined} metadata - DocumentMetadata from the backend.
 * @param {string} [fallback] - Used when the record has neither, e.g. its doc_id.
 * @returns {string} The heading ('' when nothing is known).
 */
export const recordTitle = (metadata, fallback) => {
    const m = metadata || {};
    const last = fallback == null ? '' : String(fallback).trim();
    return text(m.shelf_mark) || text(m.shelfmark) || text(m.title) || last;
};

/**
 * Alternative titles of a record; the first one is the best alt title.
 * @param {object|null|undefined} metadata - DocumentMetadata from the backend.
 * @returns {string[]} The alt titles, in the backend's order.
 */
export const altTitles = (metadata) => textList((metadata || {}).alt_titles);

/**
 * One-line card subtitle: the best alt title, but only when it adds
 * something, i.e. the description is thin or does not already contain it.
 * @param {object|null|undefined} metadata - DocumentMetadata from the backend.
 * @returns {string|null} The subtitle, or null when there is none to show.
 */
export const cardSubtitle = (metadata) => {
    const best = altTitles(metadata)[0];
    if (!best) return null;
    const description = text((metadata || {}).description);
    if (description.length < THIN_DESCRIPTION_CHARS) return best;
    return foldText(description).includes(foldText(best)) ? null : best;
};

/**
 * Date of a record as catalogued: ``date_display`` (the backend's visible
 * form, an ISO interval already written "a – b" as on the fragment page),
 * else ``century``.
 * @param {object|null|undefined} metadata - DocumentMetadata from the backend.
 * @returns {string|null} The date text, or null when the record has none.
 */
export const recordDate = (metadata) => {
    const m = metadata || {};
    return text(m.date_display) || text(m.century) || null;
};

/**
 * Text with its first character upper-cased ("legal" -> "Legal").
 * @param {unknown} value - Text to capitalise.
 * @returns {string} The capitalised text, or '' when it is not a string.
 */
export const capitalize = (value) => {
    const s = text(value);
    return s ? s.charAt(0).toUpperCase() + s.slice(1) : '';
};

/**
 * Document type for display (stored lower case, e.g. "legal").
 * @param {object|null|undefined} metadata - DocumentMetadata from the backend.
 * @returns {string|null} The capitalised type, or null.
 */
export const documentTypeLabel = (metadata) => capitalize((metadata || {}).document_type) || null;

/**
 * Chips of a result card, in order: date, language, then the document type
 * (or the first genre when there is no type). Missing values are left out.
 * @param {object|null|undefined} metadata - DocumentMetadata from the backend.
 * @returns {string[]} Up to three chip texts.
 */
export const cardChips = (metadata) => {
    const m = metadata || {};
    const kind = documentTypeLabel(m) || textList(m.genres)[0] || null;
    return [recordDate(m), text(m.language), kind].filter(Boolean);
};

/**
 * Genre and subject chips for the modal: genres first, then subjects that
 * do not repeat a genre (compared case-insensitively).
 * @param {object|null|undefined} metadata - DocumentMetadata from the backend.
 * @returns {{value: string, kind: string}[]} Chips; ``kind`` is "genre" or "subject".
 */
export const subjectAndGenreChips = (metadata) => {
    const m = metadata || {};
    const seen = new Set();
    const chips = [];
    const add = (value, kind) => {
        const key = foldText(value);
        if (seen.has(key)) return;
        seen.add(key);
        chips.push({ value, kind });
    };
    textList(m.genres).forEach((value) => add(value, 'genre'));
    textList(m.subjects).forEach((value) => add(value, 'subject'));
    return chips;
};

/**
 * "View at the source" links of a record, as the backend built them
 * (holding institution, then KTIV, then PGP). Entries without a label or an
 * http(s) URL are skipped and repeated URLs dropped.
 * @param {object|null|undefined} metadata - DocumentMetadata from the backend.
 * @returns {{source: string, label: string, url: string, kind: string}[]} The links.
 */
export const sourceLinks = (metadata) => {
    const raw = (metadata || {}).source_links;
    const seen = new Set();
    const links = [];
    (Array.isArray(raw) ? raw : []).forEach((link) => {
        const url = text(link && link.url);
        const label = text(link && link.label);
        if (!label || !/^https?:\/\//i.test(url) || seen.has(url)) return;
        seen.add(url);
        links.push({
            source: text(link.source),
            label,
            url,
            kind: SOURCE_LINK_KINDS.includes(link.kind) ? link.kind : 'item',
        });
    });
    return links;
};

/**
 * ``rel`` of an outbound source link, as on the public fragment page: item
 * and catalogue pages are followed, search-result pages are not.
 * @param {string} kind - ``item``, ``catalogue`` or ``search``.
 * @returns {string} The rel attribute value.
 */
export const sourceLinkRel = (kind) => (kind === 'search' ? 'nofollow noopener' : 'noopener');

/**
 * Display fields that GET /document completes from the enrichment index
 * (genizah_merged_v10) for a catalogue record. Search hits never carry the
 * enriched values: result cards show what their hit has.
 */
export const ENRICHED_FIELDS = ['alt_titles', 'subjects', 'genres', 'scholarly_frames'];

/**
 * Whether the document modal should read GET /document once for the enriched
 * fields: a catalogue-index record whose metadata has no subjects, genres or
 * catalogue headings. That is every search hit (the catalogue index holds
 * none of the three); a record already read from GET /document has them
 * whenever the enrichment index does.
 * @param {string|null|undefined} indexName - Index the record came from.
 * @param {object|null|undefined} metadata - DocumentMetadata the modal was opened with.
 * @returns {boolean} True when one GET /document can add display fields.
 */
export const needsEnrichedFields = (indexName, metadata) => {
    const m = metadata || {};
    return isCatalogueIndex(indexName)
        && textList(m.subjects).length === 0
        && textList(m.genres).length === 0
        && textList(m.scholarly_frames).length === 0;
};

/**
 * Metadata with the enriched display fields of a GET /document response.
 * @param {object} metadata - DocumentMetadata the modal was opened with.
 * @param {object|null|undefined} details - The GET /document response, or null.
 * @returns {object} ``metadata`` with each ENRICHED_FIELDS list the response
 *   carries replacing its own; ``metadata`` itself when there is no response.
 */
export const withEnrichedFields = (metadata, details) => {
    if (!details) return metadata;
    const merged = { ...metadata };
    ENRICHED_FIELDS.forEach((field) => {
        if (Array.isArray(details[field])) merged[field] = details[field];
    });
    return merged;
};
