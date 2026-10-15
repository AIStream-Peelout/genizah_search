/**
 * Access gate for the advanced "Readers" view on /read (researchers only).
 *
 * This module is in the main bundle, so it stays tiny: token storage, one
 * check against GET /readings/access, and a hook. The prompt and the panel
 * themselves are lazy chunks that ordinary visitors never download. No token
 * value is ever bundled; the only token is the one a researcher types.
 */
import { useCallback, useEffect, useState } from 'react';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/** localStorage key holding the researcher's advanced-view token. */
export const ADVANCED_TOKEN_KEY = 'cg.advancedToken';

/**
 * Read the stored token. Storage can be blocked or empty (private windows,
 * cleared site data), so any failure means no token.
 * @returns {string} The token, or ''.
 */
export const getStoredToken = () => {
    try {
        return window.localStorage.getItem(ADVANCED_TOKEN_KEY) || '';
    } catch (err) {
        return '';
    }
};

/**
 * Remember a token for this browser. Blocked storage is ignored: access then
 * lasts for the current page view only.
 * @param {string} token - Token accepted by the server.
 */
export const storeToken = (token) => {
    try {
        window.localStorage.setItem(ADVANCED_TOKEN_KEY, token);
    } catch (err) {
        // Storage unavailable; nothing to persist.
    }
};

/**
 * Forget the stored token.
 */
export const clearToken = () => {
    try {
        window.localStorage.removeItem(ADVANCED_TOKEN_KEY);
    } catch (err) {
        // Storage unavailable; nothing stored to remove.
    }
};

/**
 * Ask the server whether a token is valid.
 * @param {string} token - Token to check.
 * @returns {Promise<'ok'|'invalid'|'disabled'|'error'>} 'ok' (200), 'invalid'
 *     (401/403), 'disabled' (404: feature off on this server) or 'error'
 *     (anything else, including a network failure).
 */
export const checkToken = (token) =>
    fetch(`${API_BASE_URL}/readings/access`, { headers: { 'X-Advanced-Token': token } })
        .then((response) => {
            if (response.ok) return 'ok';
            if (response.status === 401 || response.status === 403) return 'invalid';
            if (response.status === 404) return 'disabled';
            return 'error';
        })
        .catch(() => 'error');

/** Pending checks per token: a stored token is validated once per page load. */
const checks = new Map();
/** Settled results per token, so a later mount starts in the right state. */
const results = new Map();

/**
 * Check a token at most once per page load (React StrictMode mounts twice).
 * @param {string} token - Token to check.
 * @returns {Promise<string>} The result of ``checkToken``.
 */
const checkOnce = (token) => {
    if (!checks.has(token)) {
        checks.set(
            token,
            checkToken(token).then((result) => {
                results.set(token, result);
                return result;
            })
        );
    }
    return checks.get(token);
};

/**
 * Advanced-view access for the current page.
 *
 * ``state`` is 'off' (and no request is made) when no token is stored; a
 * stored token is 'checking' until the server answers, then 'ok', 'invalid'
 * (the token is cleared), 'disabled' or 'error'. ``promptOpen`` is true when
 * the URL asks for the view (``?advanced=1``) or a stored token was just
 * rejected, and access is not granted.
 * @param {URLSearchParams} params - Current query string.
 * @returns {{state: string, token: string, promptOpen: boolean, rejected: boolean,
 *     submitToken: function(string): Promise<string>, signOut: function(): void,
 *     dismissPrompt: function(): void}} Access state and actions.
 */
export function useAdvancedAccess(params) {
    const requested = Boolean(params && params.get('advanced') === '1');
    const [storedAtLoad] = useState(getStoredToken);
    const [token, setToken] = useState(storedAtLoad);
    const [state, setState] = useState(() => (storedAtLoad ? results.get(storedAtLoad) || 'checking' : 'off'));
    const [rejected, setRejected] = useState(false);
    const [dismissed, setDismissed] = useState(false);

    // Validate the token that was stored when the page opened.
    useEffect(() => {
        if (!storedAtLoad) return undefined;
        let active = true;
        checkOnce(storedAtLoad).then((result) => {
            if (!active) return;
            if (result === 'invalid') {
                clearToken();
                setToken('');
                setRejected(true);
            }
            setState(result);
        });
        return () => {
            active = false;
        };
    }, [storedAtLoad]);

    // Asking again with ?advanced=1 (after a dismiss or sign-out) reopens the prompt.
    useEffect(() => {
        if (requested) setDismissed(false);
    }, [requested]);

    const submitToken = useCallback((value) => {
        const candidate = String(value || '').trim();
        if (!candidate) return Promise.resolve('invalid');
        return checkToken(candidate).then((result) => {
            if (result === 'ok') {
                storeToken(candidate);
                checks.set(candidate, Promise.resolve('ok'));
                results.set(candidate, 'ok');
                setToken(candidate);
                setRejected(false);
            }
            setState(result);
            return result;
        });
    }, []);

    const signOut = useCallback(() => {
        clearToken();
        setToken('');
        setRejected(false);
        setDismissed(true);
        setState('off');
    }, []);

    const dismissPrompt = useCallback(() => setDismissed(true), []);

    const promptOpen = !dismissed && state !== 'ok' && state !== 'checking' && (requested || rejected);

    return { state, token, promptOpen, rejected, submitToken, signOut, dismissPrompt };
}
