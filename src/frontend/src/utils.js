/**
 * Normalizes a document ID or shelfmark by replacing slashes with underscores.
 * This is required because the backend expects underscores in URLs for these identifiers.
 *
 * @param {string} id - The document ID or shelfmark to normalize.
 * @returns {string} The normalized ID.
 */
export const normalizeDocId = (id) => {
    if (!id) return '';
    return id.replace(/\//g, '_');
};

/**
 * Catalogue index the public /fragment/ pages are built from. Permalinks are
 * only offered for documents of this index (or of the backend default, when a
 * document carries no index name).
 */
export const CATALOGUE_INDEX = process.env.REACT_APP_KG_ES_INDEX || 'genizah_merged_v8';

/**
 * Whether a document from ``indexName`` has a public /fragment/ page.
 * @param {string|null|undefined} indexName - Index the document came from; empty means the backend default.
 * @returns {boolean} True for the catalogue index or no index at all.
 */
export const isCatalogueIndex = (indexName) => !indexName || indexName === CATALOGUE_INDEX;

const FRAGMENT_PREFIX = '/fragment/';

/**
 * Site path of a fragment's public page. Doc ids can hold characters such as
 * ``?``, ``*``, ``°`` or ``=``, so the id is always percent-encoded. Beyond
 * encodeURIComponent, ``! ' ( ) *`` are encoded too, so the path is exactly the
 * backend's canonical one (Python ``quote(doc_id, safe="")``).
 * @param {string} id - Catalogue doc_id.
 * @returns {string} Path such as ``/fragment/Cambridge_CUL_T_S_1%2A``.
 */
export const fragmentPath = (id) =>
    FRAGMENT_PREFIX + encodeURIComponent(id).replace(
        /[!'()*]/g,
        (c) => `%${c.charCodeAt(0).toString(16).toUpperCase()}`
    );

/**
 * Doc id named by a /fragment/<id> path, or null for any other path.
 * @param {string} pathname - Encoded location pathname.
 * @returns {string|null} The decoded doc_id, or null when the path is not a
 *   fragment page or its id is not valid percent-encoding.
 */
export const fragmentDocIdFromPath = (pathname) => {
    if (!pathname || !pathname.startsWith(FRAGMENT_PREFIX)) return null;
    const raw = pathname.slice(FRAGMENT_PREFIX.length);
    if (!raw) return null;
    try {
        return decodeURIComponent(raw);
    } catch (err) {
        // Malformed percent-encoding (URIError): treat as no fragment id.
        return null;
    }
};

/**
 * True for an unmodified primary-button click, the only click a link handler
 * should turn into an in-app action. Middle clicks and modifier clicks keep
 * the browser's own behaviour (open in a new tab or window).
 * @param {MouseEvent|React.MouseEvent} e - Click event.
 * @returns {boolean} Whether the click is a plain left click.
 */
export const isPlainLeftClick = (e) =>
    e.button === 0 && !e.metaKey && !e.ctrlKey && !e.shiftKey && !e.altKey;

const OXFORD_OLD_PREFIX = 'Oxford_Bodleian_MS_heb_';
const OXFORD_V8_PREFIX = 'Oxford_Bodleian_Bodl_MS_heb_';

/**
 * Catalogue (v8) doc_id for an id from the AI transcription index. Older AI
 * reads name Bodleian fragments ``Oxford_Bodleian_MS_heb_X``; the catalogue
 * calls them ``Oxford_Bodleian_Bodl_MS_heb_X``. Every other id is unchanged.
 * @param {string} id - Doc id as stored on an AI read.
 * @returns {string} The catalogue doc_id.
 */
export const toCatalogueDocId = (id) => {
    if (!id) return '';
    return id.startsWith(OXFORD_OLD_PREFIX) ? OXFORD_V8_PREFIX + id.slice(OXFORD_OLD_PREFIX.length) : id;
};
