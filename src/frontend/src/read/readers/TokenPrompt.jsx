import React, { useState } from 'react';
import './Readers.css';

/** What to tell the researcher after each check result. */
const RESULT_MESSAGES = {
    invalid: 'That token was not accepted.',
    disabled: 'The advanced view is not enabled on this server.',
    error: 'Could not reach the server. Try again in a moment.',
};

/**
 * Small panel asking for the advanced-view token (lazy chunk; ordinary
 * visitors never load it). The token is only what the researcher types.
 * @param {object} props - Component props.
 * @param {string} props.state - Access state from ``useAdvancedAccess``.
 * @param {boolean} props.rejected - True when a stored token was just refused.
 * @param {function(string): Promise<string>} props.onSubmit - Checks and stores a token.
 * @param {function(): void} props.onClose - Hides the panel.
 */
export default function TokenPrompt({ state, rejected, onSubmit, onClose }) {
    const [value, setValue] = useState('');
    const [busy, setBusy] = useState(false);
    const [message, setMessage] = useState(() => {
        if (rejected) return 'Your saved access token is no longer valid.';
        return RESULT_MESSAGES[state] || '';
    });

    const submit = (e) => {
        e.preventDefault();
        if (!value.trim()) {
            setMessage('Enter the token first.');
            return;
        }
        setBusy(true);
        setMessage('');
        onSubmit(value).then((result) => {
            // On success the page closes this panel.
            if (result === 'ok') return;
            setBusy(false);
            setMessage(RESULT_MESSAGES[result] || RESULT_MESSAGES.error);
        });
    };

    return (
        <div className="readers-token" role="dialog" aria-labelledby="readers-token-title">
            <div className="readers-token-head">
                <strong id="readers-token-title">Advanced view</strong>
                <button type="button" className="readers-token-close" onClick={onClose} aria-label="Close">
                    ×
                </button>
            </div>
            <p className="readers-token-text">
                Advanced view for researchers; enter the access token you were given.
            </p>
            <form className="readers-token-form" onSubmit={submit}>
                <input
                    type="password"
                    name="advanced-token"
                    autoComplete="off"
                    value={value}
                    onChange={(e) => setValue(e.target.value)}
                    placeholder="Access token"
                    aria-label="Access token"
                    disabled={busy}
                />
                <button type="submit" disabled={busy}>
                    {busy ? 'Checking…' : 'Enter'}
                </button>
            </form>
            {message && <p className="readers-token-message">{message}</p>}
        </div>
    );
}
