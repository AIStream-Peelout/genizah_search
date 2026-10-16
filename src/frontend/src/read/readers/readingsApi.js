/**
 * Requests of the advanced Readers panel. Every call sends the researcher's
 * token in ``X-Advanced-Token``; nothing here is used by the public pages.
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/** localStorage key remembering the name a researcher signs suggestions with. */
export const CONTRIBUTOR_KEY = 'cg.readingsContributor';

/**
 * Read the remembered contributor name. Blocked storage means none.
 * @returns {string} The name, or ''.
 */
export const getStoredContributor = () => {
    try {
        return window.localStorage.getItem(CONTRIBUTOR_KEY) || '';
    } catch (err) {
        return '';
    }
};

/**
 * Remember the contributor name for the next suggestion.
 * @param {string} name - Name to keep; empty removes it.
 */
export const storeContributor = (name) => {
    try {
        if (name) {
            window.localStorage.setItem(CONTRIBUTOR_KEY, name);
        } else {
            window.localStorage.removeItem(CONTRIBUTOR_KEY);
        }
    } catch (err) {
        // Storage unavailable; the form simply starts empty next time.
    }
};

/**
 * Request headers carrying the access token.
 * @param {string} token - Advanced-view token.
 * @param {object} [extra] - Further headers.
 * @returns {object} Headers object for fetch.
 */
const authHeaders = (token, extra) => ({ 'X-Advanced-Token': token || '', ...(extra || {}) });

/**
 * Base URL of one document's readings endpoints.
 * @param {string} docId - Document id (encoded here).
 * @returns {string} URL without a trailing slash.
 */
const documentUrl = (docId) => `${API_BASE_URL}/documents/${encodeURIComponent(docId)}`;

/**
 * Parse a JSON body, tolerating empty or non-JSON bodies.
 * @param {Response} response - Fetch response.
 * @returns {Promise<object|null>} Parsed body or null.
 */
const readJson = (response) => response.json().catch(() => null);

/**
 * Turn a FastAPI ``detail`` (a string, or a list of validation errors) into text.
 * @param {*} detail - ``detail`` field of an error body.
 * @returns {string} Readable message, '' when there is none.
 */
export function describeDetail(detail) {
    if (!detail) return '';
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
        return detail
            .map((item) => {
                if (!item || typeof item !== 'object') return String(item);
                const where = Array.isArray(item.loc) ? item.loc.filter((part) => part !== 'body').join('.') : '';
                const what = item.msg ? String(item.msg) : JSON.stringify(item);
                return where ? `${where}: ${what}` : what;
            })
            .join('; ');
    }
    if (typeof detail === 'object' && detail.msg) return String(detail.msg);
    return JSON.stringify(detail);
}

/**
 * Message for a failed request.
 * @param {number} status - HTTP status.
 * @param {object|null} body - Parsed error body.
 * @returns {string} Message for the UI.
 */
function failureMessage(status, body) {
    if (status === 401) return 'The access token was not accepted. Sign out and enter it again.';
    if (status === 429) return 'Too many suggestions, try again later.';
    const detail = describeDetail(body && body.detail);
    return detail || `Request failed (status ${status}).`;
}

/**
 * List the page images of a document that have a readings sidecar.
 * @param {string} docId - Document id.
 * @param {string} token - Advanced-view token.
 * @returns {Promise<Array<object>>} ``pages`` sorted by image_index; rejects on failure.
 */
export async function fetchReadingsPages(docId, token) {
    // Always revalidate (a 304 when unchanged): a browser-held list could hide pages pushed since.
    const response = await fetch(`${documentUrl(docId)}/readings/pages`, { headers: authHeaders(token), cache: 'no-cache' });
    const body = await readJson(response);
    if (!response.ok) throw new Error(failureMessage(response.status, body));
    return body && Array.isArray(body.pages) ? body.pages : [];
}

/**
 * Fetch the readings sidecar of one page image. Without a model the server
 * answers 409 with every model's sidecar when there is more than one.
 * @param {string} docId - Document id.
 * @param {number} imageIndex - 0-based image index.
 * @param {string} token - Advanced-view token.
 * @param {{fresh?: boolean}} [options] - ``fresh`` skips the browser cache
 *     entirely (after the server said the loaded copy is out of date);
 *     otherwise the browser revalidates its copy (a 304 when unchanged), so a
 *     second model's sidecar or a re-decode shows up at once.
 * @returns {Promise<object>} ``{status: 'ok', sidecar}``, ``{status: 'missing', message}``,
 *     ``{status: 'choose', options: {model: sidecar}}`` or ``{status: 'error', message}``.
 *     Rejects only on a network failure.
 */
export async function fetchSidecar(docId, imageIndex, token, options) {
    const url = `${documentUrl(docId)}/readings?image_index=${encodeURIComponent(imageIndex)}`;
    const init = { headers: authHeaders(token), cache: options && options.fresh ? 'no-store' : 'no-cache' };
    const response = await fetch(url, init);
    const body = await readJson(response);
    if (response.ok && body) return { status: 'ok', sidecar: body };
    if (response.status === 404) return { status: 'missing', message: describeDetail(body && body.detail) };
    if (response.status === 409 && body && body.readings && typeof body.readings === 'object') {
        return { status: 'choose', options: body.readings, message: describeDetail(body.detail) };
    }
    return { status: 'error', message: failureMessage(response.status, body) };
}

/**
 * Fetch the suggested readings of one page image, newest first.
 * @param {string} docId - Document id.
 * @param {number} imageIndex - 0-based image index.
 * @param {string} token - Advanced-view token.
 * @returns {Promise<Array<object>>} Corrections; rejects on failure.
 */
export async function fetchCorrections(docId, imageIndex, token) {
    const url = `${documentUrl(docId)}/readings/corrections?image_index=${encodeURIComponent(imageIndex)}`;
    const response = await fetch(url, { headers: authHeaders(token) });
    const body = await readJson(response);
    if (!response.ok) throw new Error(failureMessage(response.status, body));
    return body && Array.isArray(body.corrections) ? body.corrections : [];
}

/**
 * Submit a suggested reading for one line.
 * @param {string} docId - Document id.
 * @param {object} payload - ``{image_index, line_index, reader_id, bbox, original_text,
 *     corrected_text, note, contributor, vlm_model, decoded_at}``.
 * @param {string} token - Advanced-view token.
 * @returns {Promise<{ok: boolean, correction?: object, message?: string, stale?: boolean}>}
 *     The stored correction, or a message to show; ``stale`` (409) means the page was
 *     re-decoded since the sidecar on screen was fetched. Rejects only on a network failure.
 */
export async function postCorrection(docId, payload, token) {
    const response = await fetch(`${documentUrl(docId)}/readings/corrections`, {
        method: 'POST',
        headers: authHeaders(token, { 'Content-Type': 'application/json' }),
        body: JSON.stringify(payload),
    });
    const body = await readJson(response);
    if (response.ok && body && body.correction) return { ok: true, correction: body.correction };
    if (response.status === 404) {
        return { ok: false, message: 'This page has no readings sidecar any more; the suggestion was not saved.' };
    }
    if (response.status === 409) {
        const detail = describeDetail(body && body.detail);
        return {
            ok: false,
            stale: true,
            message: detail || 'This page was re-decoded; reload the readings and pin the line again.',
        };
    }
    return { ok: false, message: failureMessage(response.status, body) };
}
