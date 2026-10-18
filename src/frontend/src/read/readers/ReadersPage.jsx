import React, { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import ReadersPanel from './ReadersPanel';
import { fetchReadingsPages } from './readingsApi';
import { groupPages, pageLabel } from './readersModel';
import './Readers.css';

/**
 * Readers-only page (advanced view, lazy chunk): shown instead of the public
 * read when this document or image has no public AI read. Same header as
 * /read, with an image picker built from the pages that have readings.
 * @param {object} props - Component props.
 * @param {string} props.docId - Document id.
 * @param {string} props.title - Title or shelf mark to show.
 * @param {string} [props.shelfmark] - Shelf mark, shown under a different title.
 * @param {number|null} props.imageIndex - Image from ``?image=``, or null for the first page with readings.
 * @param {string} [props.vlmModel] - Preferred model (``?model=``).
 * @param {string} props.token - Advanced-view token.
 * @param {function(number): void} props.onImageChange - Switch image (updates the URL).
 * @param {function(): void} props.onSignOut - Forget the token.
 */
export default function ReadersPage({ docId, title, shelfmark, imageIndex, vlmModel, token, onImageChange, onSignOut }) {
    const [pages, setPages] = useState(null);
    const [error, setError] = useState('');

    useEffect(() => {
        let active = true;
        setPages(null);
        setError('');
        fetchReadingsPages(docId, token)
            .then((list) => {
                if (active) setPages(list);
            })
            .catch((err) => {
                if (active) setError(err.message || 'Could not list the pages with readings.');
            });
        return () => {
            active = false;
        };
    }, [docId, token]);

    const choices = useMemo(() => groupPages(pages || []), [pages]);
    let current = imageIndex;
    if (current === null || current === undefined || Number.isNaN(current)) {
        current = choices.length ? choices[0].image_index : null;
    }
    const currentListed = choices.some((page) => page.image_index === current);

    let body;
    if (current !== null) {
        body = <ReadersPanel docId={docId} imageIndex={current} vlmModel={vlmModel} token={token} />;
    } else if (pages === null && !error) {
        body = <div className="readers-state">Looking for pages with readings…</div>;
    } else if (error) {
        body = <div className="readers-state readers-error">{error}</div>;
    } else {
        body = <div className="readers-state">No readings sidecars for this document yet.</div>;
    }

    return (
        <div className="read-page readers-page">
            <header className="read-header">
                <div>
                    <Link to="/" className="read-back">← Search</Link>
                    <h1>{title}</h1>
                    {shelfmark && shelfmark !== title && <div className="read-shelfmark">{shelfmark}</div>}
                    <div className="readers-page-note">Readers view (advanced). No public AI read is shown for this image.</div>
                </div>
                <div className="read-header-right">
                    <span className="readers-advanced-status">
                        Advanced ·{' '}
                        <button type="button" className="readers-link-button" onClick={onSignOut}>sign out</button>
                    </span>
                    {(choices.length > 0 || current !== null) && (
                        <label className="read-image-picker">
                            Image{' '}
                            <select
                                value={current === null ? '' : String(current)}
                                onChange={(e) => onImageChange(Number(e.target.value))}
                            >
                                {current !== null && !currentListed && (
                                    <option value={String(current)}>
                                        {pages === null ? current + 1 : `${current + 1} · no readings`}
                                    </option>
                                )}
                                {choices.map((page) => (
                                    <option key={page.image_index} value={String(page.image_index)}>
                                        {pageLabel(page)}
                                    </option>
                                ))}
                            </select>
                        </label>
                    )}
                    {error && current !== null && <span className="readers-error">{error}</span>}
                </div>
            </header>
            {body}
        </div>
    );
}
