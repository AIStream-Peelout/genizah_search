import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { AGREEMENT_CAVEAT, READERS_DESCRIPTION } from './read/caveat';
import './YomKippur.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/** Catalogue index the thirteen fragments live in; passed through to /read. */
const SOURCE_INDEX = 'genizah_merged_v6';

/**
 * The thirteen Yom Kippur liturgy fragments read for Yom Kippur 5787.
 * A fixed list: this is a one-time page, not a search.
 */
const YOM_KIPPUR_DOC_IDS = [
    'Cambridge_Lewis_Gibson_L_G_Bib_VI_29',
    'Cambridge_CUL_Or_1080_7_5',
    'Cambridge_CUL_Or_1080_15_21',
    'Cambridge_CUL_Or_1080_1_55',
    'Cambridge_CUL_T_S_NS_200_51',
    'Cambridge_CUL_Or_1080_15_5',
    'Cambridge_CUL_T_S_NS_200_5',
    'Cambridge_CUL_T_S_NS_200_11',
    'Cambridge_CUL_T_S_24_10',
    'Manchester_JRL_A_217',
    'Philadelphia_CAJS_Halper_204',
    'Philadelphia_CAJS_Halper_233',
    'Philadelphia_CAJS_Halper_234',
];

/**
 * Share of lines confirmed by the second reader.
 * @param {{n_agreed: number, n_lines: number}} read - Read summary.
 * @returns {number} 0-1, 0 when the read has no lines.
 */
const confirmedShare = (read) => (read.n_lines > 0 ? read.n_agreed / read.n_lines : 0);

/**
 * Pick the image with the highest confirmed share (ties: more confirmed lines).
 * @param {object[]} items - ``items`` of the availability endpoint.
 * @returns {object|null} The best item, or null when there are none.
 */
const bestImage = (items) =>
    items.reduce((best, item) => {
        if (!best) return item;
        const a = confirmedShare(item.ai_read);
        const b = confirmedShare(best.ai_read);
        if (a !== b) return a > b ? item : best;
        return item.ai_read.n_agreed > best.ai_read.n_agreed ? item : best;
    }, null);

/**
 * Human-readable fallback for a document id when the catalogue record is unavailable.
 * @param {string} docId - Elasticsearch document id.
 * @returns {string} The id with underscores as spaces.
 */
const fallbackLabel = (docId) => docId.replace(/_/g, ' ');

/**
 * Link to the viewer for one image of one fragment.
 * @param {string} docId - Document id.
 * @param {number} imageIndex - 0-based image index.
 * @returns {string} Route with query string.
 */
const readLink = (docId, imageIndex) => {
    const q = new URLSearchParams({ doc: docId, image: String(imageIndex), index: SOURCE_INDEX });
    return `/read?${q.toString()}`;
};

/**
 * One fragment card: thumbnail, shelfmark, catalogue title, confirmed count,
 * link to the viewer at its best image. The catalogue record is fetched per
 * card so the list renders as soon as the counts are known.
 * @param {{docId: string, item: object, imageCount: number}} props - Best image and its document.
 */
function FragmentCard({ docId, item, imageCount }) {
    const [meta, setMeta] = useState(null);

    useEffect(() => {
        let cancelled = false;
        fetch(`${API_BASE_URL}/document/${encodeURIComponent(docId)}?index_name=${encodeURIComponent(SOURCE_INDEX)}`)
            .then((r) => (r.ok ? r.json() : null))
            .then((m) => {
                if (!cancelled) setMeta(m);
            })
            .catch(() => {
                if (!cancelled) setMeta(null);
            });
        return () => {
            cancelled = true;
        };
    }, [docId]);

    const shelfmark = meta?.shelf_mark || meta?.shelfmark || fallbackLabel(docId);
    const title = meta?.title && meta.title !== shelfmark ? meta.title : null;
    const read = item.ai_read;
    const href = readLink(docId, item.image_index);

    return (
        <li className="yk-card">
            <Link to={href} className="yk-thumb" title="Open in the viewer">
                <img src={item.image_url} alt={`${shelfmark}, image ${item.image_index + 1}`} loading="lazy" />
            </Link>
            <div className="yk-card-body">
                <h3 className="yk-shelfmark">
                    <Link to={href}>{shelfmark}</Link>
                </h3>
                {title && <p className="yk-title">{title}</p>}
                <p className="yk-count">
                    <strong>{read.n_agreed} of {read.n_lines} lines</strong> confirmed by a second reader
                    {imageCount > 1 && <span className="yk-image-note"> · image {item.image_index + 1} of {imageCount}</span>}
                </p>
                <Link to={href} className="yk-open">Read it on the manuscript →</Link>
            </div>
        </li>
    );
}

/**
 * One-time page for Yom Kippur 5787 (21-22 September 2026): the thirteen
 * liturgical fragments with machine reads, ordered by confirmed share.
 * Route: /yom-kippur (and /yk).
 */
export default function YomKippur() {
    const [cards, setCards] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        document.title = 'Yom Kippur in the Cairo Genizah';
        let cancelled = false;
        Promise.all(
            YOM_KIPPUR_DOC_IDS.map((docId) =>
                fetch(`${API_BASE_URL}/ai-transcriptions/${encodeURIComponent(docId)}`)
                    .then((r) => (r.ok ? r.json() : null))
                    .catch(() => null)
            )
        ).then((statuses) => {
            if (cancelled) return;
            const ready = [];
            statuses.forEach((status, i) => {
                if (!status || !status.available) return;
                const item = bestImage(status.items);
                if (item) ready.push({ docId: YOM_KIPPUR_DOC_IDS[i], item, imageCount: status.items.length });
            });
            ready.sort((a, b) => {
                const d = confirmedShare(b.item.ai_read) - confirmedShare(a.item.ai_read);
                return d !== 0 ? d : b.item.ai_read.n_agreed - a.item.ai_read.n_agreed;
            });
            if (ready.length === 0) setError('The reads for these fragments are not available right now.');
            setCards(ready);
        });
        return () => {
            cancelled = true;
            document.title = 'Cairo Genizah AI';
        };
    }, []);

    return (
        <div className="yk-page">
            <header className="yk-header">
                <Link to="/" className="yk-back">← Cairo Genizah AI</Link>
                <h1>Yom Kippur in the Cairo Genizah</h1>
                <p className="yk-intro">
                    Fragments of Yom Kippur liturgy from the Cairo Genizah, read by machine and checked by a second
                    reader. A one-time page for Yom Kippur 5787, 21–22 September 2026.
                </p>
                <details className="yk-caveat">
                    <summary>
                        <strong>Beta: machine reading, not checked by a person.</strong> Read the full caveat
                    </summary>
                    <p>
                        {READERS_DESCRIPTION} {AGREEMENT_CAVEAT} Unconfirmed lines show a yellow box and grey text in
                        the viewer: the box shows where the model read, but nothing checked the text. A second run of the
                        same page moves individual lines. Do not cite this text; use it to explore the fragment against
                        the image.
                    </p>
                </details>
            </header>

            {error && <p className="yk-error">{error}</p>}
            {!cards && !error && <p className="yk-loading">Loading the fragments…</p>}
            {cards && cards.length > 0 && (
                <ol className="yk-list">
                    {cards.map((c) => (
                        <FragmentCard key={c.docId} docId={c.docId} item={c.item} imageCount={c.imageCount} />
                    ))}
                </ol>
            )}

            <footer className="yk-footer">
                Models trained on Cairo Genizah fragments with Princeton Geniza Project transcriptions and on Hebrew
                manuscript pages with transcriptions from the National Library of Israel’s KTIV project. Images courtesy
                of the holding libraries. Ordered by the share of lines the two readers agree on.
            </footer>
        </div>
    );
}
