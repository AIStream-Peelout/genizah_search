/**
 * Opt-in "v9 beta" semantic search: the fine-tuned embedder over its own index.
 *
 * The default search (index and embedder) is untouched; a beta search is the
 * normal semantic request with ``index_name`` set to the beta index that
 * GET /semantic-beta reports, and only once that endpoint says it is available.
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/** localStorage key remembering this browser's v9 beta switch (default off). */
export const SEMANTIC_BETA_KEY = 'genizah_semantic_beta';

/** Value of ``routed_to`` when the backend answered a query as a shelf mark. */
export const ROUTED_TO_SHELFMARK = 'shelfmark';

/**
 * Read the remembered v9 beta switch. Storage can be blocked or empty
 * (private windows, cleared site data), so any failure means off.
 * @returns {boolean} True only when this browser turned the beta on before.
 */
export const readBetaPreference = () => {
    try {
        return window.localStorage.getItem(SEMANTIC_BETA_KEY) === 'true';
    } catch (err) {
        return false;
    }
};

/**
 * Remember the v9 beta switch for this browser. Blocked storage is ignored:
 * the switch still works for the current page view.
 * @param {boolean} enabled - New switch state.
 */
export const writeBetaPreference = (enabled) => {
    try {
        if (enabled) {
            window.localStorage.setItem(SEMANTIC_BETA_KEY, 'true');
        } else {
            window.localStorage.removeItem(SEMANTIC_BETA_KEY);
        }
    } catch (err) {
        // Storage unavailable; nothing to persist.
    }
};

/**
 * Normalise a GET /semantic-beta body to the beta status the UI may act on.
 * @param {object|null|undefined} data - Parsed response body.
 * @returns {object|null} The status when ``available`` is true and an index is named, else null.
 */
export const parseSemanticBeta = (data) => {
    if (!data || data.available !== true || typeof data.index !== 'string' || !data.index) {
        return null;
    }
    return data;
};

/**
 * Fetch the v9 beta status. A network error, a 404 from a backend without the
 * endpoint, or a malformed body all mean the beta is unavailable.
 * @returns {Promise<object|null>} The available beta status, or null.
 */
export const fetchSemanticBeta = async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/semantic-beta`);
        if (!response.ok) return null;
        return parseSemanticBeta(await response.json());
    } catch (err) {
        return null;
    }
};

/**
 * Index a semantic search should use.
 * @param {object} options - Inputs.
 * @param {string} options.selectedIndex - Index from Index Selection (the default unless changed).
 * @param {boolean} options.explicitIndex - True when the user picked a non-default index there.
 * @param {boolean} options.betaEnabled - The v9 beta switch.
 * @param {object|null} options.beta - Available beta status from GET /semantic-beta.
 * @returns {{indexName: string, semanticBeta: boolean}} The index to send and whether it is the beta.
 */
export const resolveSemanticIndex = ({ selectedIndex, explicitIndex, betaEnabled, beta }) => {
    if (beta && betaEnabled && !explicitIndex) {
        return { indexName: beta.index, semanticBeta: true };
    }
    return { indexName: selectedIndex, semanticBeta: false };
};

/**
 * Tag a beta search response so the results header shows the badge and the
 * result cards open records from the index that was searched.
 * @param {object} data - Response body of POST /search.
 * @param {string} indexName - Beta index the request was sent to.
 * @returns {object} The response with ``semantic_beta`` set and ``index_name`` filled in.
 */
export const markSemanticBetaResults = (data, indexName) => ({
    ...data,
    semantic_beta: true,
    index_name: data.index_name || indexName,
});
