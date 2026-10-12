import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { AGREEMENT_CAVEAT, READERS_DESCRIPTION, UNCONFIRMED_TOOLTIP, agreedTooltip } from './caveat';
import './ReadFragment.css';

/** How long the deep-linked line pulses, in ms. */
const PULSE_MS = 2600;

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

/** Phone-sized viewport: the banner collapses and the stage opens on the first line. */
const SMALL_VIEWPORT = '(max-width: 599px)';
/** Touch device: pinch replaces the wheel for zooming. */
const COARSE_POINTER = '(pointer: coarse)';
/** Below this zoom the overlay row numbers overlap on a phone, so they are hidden. */
const MIN_ZOOM_FOR_NUMBERS = 0.15;
/** Zoom bounds shared by the wheel, the buttons and the pinch gesture. */
const MIN_ZOOM = 0.02;
const MAX_ZOOM = 12;

/**
 * Track whether a CSS media query matches.
 * @param {string} query - Media query, e.g. '(max-width: 599px)'.
 * @returns {boolean} True while the query matches.
 */
function useMediaQuery(query) {
    const [matches, setMatches] = useState(() =>
        typeof window !== 'undefined' && window.matchMedia ? window.matchMedia(query).matches : false
    );
    useEffect(() => {
        if (typeof window === 'undefined' || !window.matchMedia) return undefined;
        const mq = window.matchMedia(query);
        const onChange = (e) => setMatches(e.matches);
        setMatches(mq.matches);
        mq.addEventListener('change', onChange);
        return () => mq.removeEventListener('change', onChange);
    }, [query]);
    return matches;
}

/**
 * Convert a 0-1000 normalised box to pixel coordinates of the natural image.
 * @param {number[]} bbox - [x1, y1, x2, y2] normalised to 0-1000.
 * @param {number} w - Natural image width in pixels.
 * @param {number} h - Natural image height in pixels.
 * @returns {number[]} [x0, y0, x1, y1] in pixels.
 */
const toPx = (bbox, w, h) => [
    (bbox[0] / 1000) * w,
    (bbox[1] / 1000) * h,
    (bbox[2] / 1000) * w,
    (bbox[3] / 1000) * h,
];

/**
 * Beta warning shown on first load. Copy is deliberately blunt: nothing here
 * has been checked by a person. On a phone (``compact``) only the first
 * sentence shows until the visitor taps "read more", so the manuscript stays
 * above the fold; desktop always shows the full text.
 * @param {{compact: boolean}} props - Whether the viewport is phone-sized.
 */
function BetaBanner({ compact }) {
    const [expanded, setExpanded] = useState(false);
    const collapsed = compact && !expanded;
    return (
        <div className={`read-banner${collapsed ? ' compact' : ''}`} role="alert">
            <strong>Beta: machine reading, not checked by a person.</strong>{' '}
            {collapsed ? (
                <button type="button" className="read-banner-toggle" onClick={() => setExpanded(true)}>
                    read more
                </button>
            ) : (
                <>
                    {READERS_DESCRIPTION} {AGREEMENT_CAVEAT} Unconfirmed lines get a yellow box and grey
                    text: the box shows where the model read, but nothing checked the text. A second run of the same page moves individual lines. Do not cite this text; use it to explore the
                    fragment against the image.
                    {compact && (
                        <>
                            {' '}
                            <button type="button" className="read-banner-toggle" onClick={() => setExpanded(false)}>
                                show less
                            </button>
                        </>
                    )}
                </>
            )}
        </div>
    );
}

/**
 * Availability and error states for the page.
 */
function ReadMessage({ title, children }) {
    return (
        <div className="read-message">
            <h2>{title}</h2>
            <div>{children}</div>
            <p>
                <Link to="/">Back to search</Link>
            </p>
        </div>
    );
}

/**
 * Clamp a zoom factor to the stage's bounds.
 * @param {number} z - Requested zoom.
 * @returns {number} Zoom within [MIN_ZOOM, MAX_ZOOM].
 */
const clampZoom = (z) => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z));

/**
 * Image stage with an SVG overlay of confirmed-line boxes plus wheel zoom,
 * pinch zoom (two pointers) and drag pan.
 * @param {object} props - See the JSX below; ``compact`` is a phone-sized
 *     viewport (row numbers hide at low zoom), ``coarse`` a touch pointer
 *     (the hint says pinch instead of scroll).
 */
function ImageStage({ record, natural, onNatural, hovered, setHovered, showBoxes, showNumbers, showFragments, focusRequest, pulsed, compact, coarse }) {
    const viewportRef = useRef(null);
    const [view, setView] = useState({ z: 1, tx: 0, ty: 0 });
    const dragRef = useRef(null);
    /** Active pointers by id, in viewport coordinates. */
    const pointersRef = useRef(new Map());
    /** Geometry captured when the second pointer went down; null when not pinching. */
    const pinchRef = useRef(null);
    const W = natural.w;
    const H = natural.h;

    const fit = useCallback(() => {
        const vp = viewportRef.current;
        if (!vp || !W || !H) return;
        const r = vp.getBoundingClientRect();
        if (r.width < 20 || r.height < 20) return;
        const z = Math.max(MIN_ZOOM, Math.min(r.width / W, r.height / H) * 0.96);
        setView({ z, tx: (r.width - W * z) / 2, ty: (r.height - H * z) / 2 });
    }, [W, H]);

    useEffect(() => {
        fit();
        let lastWidth = viewportRef.current ? viewportRef.current.getBoundingClientRect().width : 0;
        // Mobile browsers fire resize whenever the address bar shows or hides;
        // only refit when the stage actually changed width (rotation, window resize),
        // otherwise a scroll would throw away the visitor's zoom.
        const onResize = () => {
            const vp = viewportRef.current;
            if (!vp) return;
            const width = vp.getBoundingClientRect().width;
            if (Math.abs(width - lastWidth) < 1) return;
            lastWidth = width;
            fit();
        };
        window.addEventListener('resize', onResize);
        return () => window.removeEventListener('resize', onResize);
    }, [fit]);

    const zoomAt = useCallback((factor, mx, my) => {
        setView((v) => {
            const nz = clampZoom(v.z * factor);
            return { z: nz, tx: mx - (mx - v.tx) * (nz / v.z), ty: my - (my - v.ty) * (nz / v.z) };
        });
    }, []);

    // Wheel zoom needs a non-passive listener so preventDefault works.
    useEffect(() => {
        const vp = viewportRef.current;
        if (!vp) return undefined;
        const onWheel = (e) => {
            e.preventDefault();
            const r = vp.getBoundingClientRect();
            zoomAt(Math.exp(-e.deltaY * 0.0016), e.clientX - r.left, e.clientY - r.top);
        };
        vp.addEventListener('wheel', onWheel, { passive: false });
        return () => vp.removeEventListener('wheel', onWheel);
    }, [zoomAt]);

    // Centre a requested line (from a click in the text panel, or the ?line=
    // deep link, which also zooms so the box fills most of the panel).
    // ``zoom: 'page'`` is the phone's opening view: fit the width of the page
    // that holds the line (half the image on a two-page opening) and centre
    // on the line, so the visitor lands on legible text instead of a 3 % fit.
    useEffect(() => {
        if (!focusRequest || !W || !H) return;
        const vp = viewportRef.current;
        if (!vp) return;
        const r = vp.getBoundingClientRect();
        const [x0, y0, x1, y1] = toPx(focusRequest.bbox, W, H);
        const cx = (x0 + x1) / 2;
        const cy = (y0 + y1) / 2;
        setView((v) => {
            if (focusRequest.zoom === 'page') {
                const twoPage = W / H > 1.15;
                const pageWidth = twoPage ? W / 2 : W;
                const pageCentre = twoPage ? (cx < W / 2 ? W / 4 : (3 * W) / 4) : W / 2;
                const z = Math.min(8, Math.max(0.05, (r.width * 0.98) / pageWidth));
                let ty = r.height / 2 - cy * z;
                if (H * z <= r.height) {
                    ty = (r.height - H * z) / 2;
                } else {
                    ty = Math.min(0, Math.max(r.height - H * z, ty));
                }
                return { z, tx: r.width / 2 - pageCentre * z, ty };
            }
            let z = v.z;
            if (focusRequest.zoom) {
                const zw = (r.width * 0.7) / Math.max(1, x1 - x0);
                const zh = (r.height * 0.35) / Math.max(1, y1 - y0);
                z = Math.min(8, Math.max(0.05, Math.min(zw, zh)));
            }
            return { z, tx: r.width / 2 - cx * z, ty: r.height / 2 - cy * z };
        });
    }, [focusRequest, W, H]);

    /**
     * Pointer position relative to the viewport's top-left corner.
     * @param {PointerEvent} e - Pointer event.
     * @returns {{x: number, y: number}} Viewport coordinates.
     */
    const viewportPoint = (e) => {
        const r = viewportRef.current.getBoundingClientRect();
        return { x: e.clientX - r.left, y: e.clientY - r.top };
    };

    /**
     * Record the geometry of the two active pointers so later moves can be
     * applied relative to it (zoom by distance ratio, pan by midpoint shift).
     * @param {{z: number, tx: number, ty: number}} current - View at pinch start.
     */
    const startPinch = (current) => {
        const [a, b] = Array.from(pointersRef.current.values());
        pinchRef.current = {
            d0: Math.hypot(a.x - b.x, a.y - b.y) || 1,
            mid0: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 },
            z0: current.z,
            tx0: current.tx,
            ty0: current.ty,
        };
        dragRef.current = null;
    };

    const onPointerDown = (e) => {
        if (e.pointerType === 'mouse' && e.button !== 0) return;
        const p = viewportPoint(e);
        pointersRef.current.set(e.pointerId, p);
        e.currentTarget.setPointerCapture(e.pointerId);
        if (pointersRef.current.size >= 2) {
            startPinch(view);
        } else {
            dragRef.current = { x: p.x, y: p.y, tx: view.tx, ty: view.ty };
        }
    };
    const onPointerMove = (e) => {
        if (!pointersRef.current.has(e.pointerId)) return;
        const p = viewportPoint(e);
        pointersRef.current.set(e.pointerId, p);
        const pinch = pinchRef.current;
        if (pinch && pointersRef.current.size >= 2) {
            const [a, b] = Array.from(pointersRef.current.values());
            const d = Math.hypot(a.x - b.x, a.y - b.y);
            const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
            const nz = clampZoom(pinch.z0 * (d / pinch.d0));
            // Keep the image point that sat under the first midpoint under the current one.
            const px = (pinch.mid0.x - pinch.tx0) / pinch.z0;
            const py = (pinch.mid0.y - pinch.ty0) / pinch.z0;
            setView({ z: nz, tx: mid.x - px * nz, ty: mid.y - py * nz });
            return;
        }
        const d = dragRef.current;
        if (!d) return;
        setView((v) => ({ z: v.z, tx: d.tx + (p.x - d.x), ty: d.ty + (p.y - d.y) }));
    };
    const onPointerUp = (e) => {
        pointersRef.current.delete(e.pointerId);
        if (pointersRef.current.size >= 2) {
            startPinch(view);
            return;
        }
        pinchRef.current = null;
        if (pointersRef.current.size === 1) {
            // One finger left after a pinch: carry on as a plain drag from here, without a jump.
            const [p] = Array.from(pointersRef.current.values());
            dragRef.current = { x: p.x, y: p.y, tx: view.tx, ty: view.ty };
        } else {
            dragRef.current = null;
        }
    };

    const fontSize = 13 / view.z;
    const numbersVisible = showNumbers && !(compact && view.z < MIN_ZOOM_FOR_NUMBERS);

    return (
        <div className="read-stage">
            <div className="read-stage-toolbar">
                <button type="button" onClick={fit} title="Fit image to the panel">Fit</button>
                <button
                    type="button"
                    onClick={() => {
                        const r = viewportRef.current.getBoundingClientRect();
                        zoomAt(1.25, r.width / 2, r.height / 2);
                    }}
                    title="Zoom in"
                >
                    +
                </button>
                <button
                    type="button"
                    onClick={() => {
                        const r = viewportRef.current.getBoundingClientRect();
                        zoomAt(0.8, r.width / 2, r.height / 2);
                    }}
                    title="Zoom out"
                >
                    −
                </button>
                <span className="read-zoom-level">{Math.round(view.z * 100)}%</span>
                <span className="read-stage-hint">
                    {coarse ? 'Pinch to zoom, drag to pan.' : 'Scroll to zoom, drag to pan.'} Green: two readers agree. Yellow: single reader, caution.
                </span>
            </div>
            <div
                className="read-viewport"
                ref={viewportRef}
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                onPointerCancel={onPointerUp}
            >
                <div
                    className="read-canvas"
                    style={{
                        width: W ? W * view.z : 'auto',
                        height: H ? H * view.z : 'auto',
                        left: view.tx,
                        top: view.ty,
                    }}
                >
                    <img
                        src={record.image_url}
                        alt={`Fragment ${record.doc_id}, image ${record.image_index + 1}`}
                        draggable={false}
                        onLoad={(e) => onNatural({ w: e.target.naturalWidth, h: e.target.naturalHeight })}
                    />
                    {W > 0 && H > 0 && (
                        <svg className="read-overlay" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
                            {record.ai_read.lines.map((line) => {
                                const [x0, y0, x1, y1] = toPx(line.bbox, W, H);
                                const isHover = hovered === line.index;
                                const isPulse = pulsed === line.index;
                                return (
                                    <g
                                        key={line.index}
                                        onMouseEnter={() => setHovered(line.index)}
                                        onMouseLeave={() => setHovered(null)}
                                    >
                                        {showFragments &&
                                            line.htr_fragments.map((frag, k) => {
                                                const [fx0, fy0, fx1, fy1] = toPx(frag, W, H);
                                                return (
                                                    <rect
                                                        key={k}
                                                        className="read-frag"
                                                        x={fx0}
                                                        y={fy0}
                                                        width={Math.max(1, fx1 - fx0)}
                                                        height={Math.max(1, fy1 - fy0)}
                                                    />
                                                );
                                            })}
                                        {showBoxes && (
                                            <rect
                                                className={`read-box ${line.status}${isHover ? ' hover' : ''}${isPulse ? ' pulse' : ''}`}
                                                data-line={line.index}
                                                x={x0}
                                                y={y0}
                                                width={Math.max(1, x1 - x0)}
                                                height={Math.max(1, y1 - y0)}
                                            >
                                                <title>{`${line.index + 1}: ${line.text}`}</title>
                                            </rect>
                                        )}
                                        {numbersVisible && (
                                            <text
                                                className="read-box-label"
                                                x={x1 + 4 / view.z}
                                                y={y0 + fontSize}
                                                style={{ fontSize }}
                                            >
                                                {line.index + 1}
                                            </text>
                                        )}
                                    </g>
                                );
                            })}
                        </svg>
                    )}
                </div>
            </div>
        </div>
    );
}

/**
 * Right-to-left text panel: confirmed lines carry a badge and link to their
 * box; unconfirmed lines are grey and have no box.
 */
function TextPanel({ record, hovered, setHovered, showHtr, onFocusLine, pulsed }) {
    return (
        <ol className="read-lines" dir="rtl">
            {record.ai_read.lines.map((line) => {
                const agreed = line.status === 'agreed';
                const isHover = hovered === line.index;
                const differs = line.htr_text && line.htr_text !== line.text;
                const isPulse = pulsed === line.index;
                return (
                    <li
                        key={line.index}
                        id={`read-line-${line.index}`}
                        className={`read-line ${line.status}${isHover ? ' hover' : ''}${isPulse ? ' pulse' : ''}`}
                        onMouseEnter={() => setHovered(line.index)}
                        onMouseLeave={() => setHovered(null)}
                        onClick={() => onFocusLine(line)}
                        title={agreed ? agreedTooltip(line) : UNCONFIRMED_TOOLTIP}
                    >
                        <span className="read-line-number" dir="ltr">{line.index + 1}</span>
                        <span className="read-line-text">{line.text || ' '}</span>
                        {agreed && <span className="read-badge" dir="ltr">✓ 2 readers</span>}
                        {showHtr && (
                            <span className={`read-line-htr${differs ? ' differs' : ''}`}>
                                {line.htr_text ? line.htr_text : <em dir="ltr">no Kraken reading</em>}
                            </span>
                        )}
                    </li>
                );
            })}
        </ol>
    );
}

/**
 * Public read-only viewer for offline AI reads.
 * Route: /read?doc=<es doc id>&image=<image index>&index=<source index>&model=<vlm key>&line=<0-based line>
 * ``line`` (from AI-transcription search) zooms the stage to that line's box,
 * pulses it, and scrolls the text panel to it.
 */
export default function ReadFragment() {
    const [params, setParams] = useSearchParams();
    const docId = params.get('doc') || '';
    const requestedImage = params.get('image');
    const sourceIndex = params.get('index') || '';
    const requestedModel = params.get('model') || '';
    const requestedLine = params.get('line');

    const [status, setStatus] = useState(null);
    const [record, setRecord] = useState(null);
    const [docMeta, setDocMeta] = useState(null);
    const [error, setError] = useState(null);
    const [natural, setNatural] = useState({ w: 0, h: 0 });
    const [hovered, setHovered] = useState(null);
    const [showBoxes, setShowBoxes] = useState(true);
    const [showNumbers, setShowNumbers] = useState(true);
    const [showFragments, setShowFragments] = useState(false);
    const [showHtr, setShowHtr] = useState(false);
    const [focusRequest, setFocusRequest] = useState(null);
    const [pulsed, setPulsed] = useState(null);
    const compact = useMediaQuery(SMALL_VIEWPORT);
    const coarse = useMediaQuery(COARSE_POINTER);

    // 1. Which images have reads?
    useEffect(() => {
        if (!docId) return;
        setStatus(null);
        setRecord(null);
        setError(null);
        fetch(`${API_BASE_URL}/ai-transcriptions/${encodeURIComponent(docId)}`)
            .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`status ${r.status}`))))
            .then(setStatus)
            .catch((e) => setError(`Could not check AI read availability (${e.message}).`));
    }, [docId]);

    // 2. Fetch the chosen record.
    const imageIndex = useMemo(() => {
        if (requestedImage != null && requestedImage !== '') return Number(requestedImage);
        return status?.items?.[0]?.image_index ?? 0;
    }, [requestedImage, status]);

    useEffect(() => {
        if (!status?.available) return;
        setNatural({ w: 0, h: 0 });
        const model = requestedModel ? `?model=${encodeURIComponent(requestedModel)}` : '';
        fetch(`${API_BASE_URL}/ai-transcriptions/${encodeURIComponent(docId)}/${imageIndex}${model}`)
            .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`status ${r.status}`))))
            .then(setRecord)
            .catch((e) => setError(`Could not load the read (${e.message}).`));
    }, [status, docId, imageIndex, requestedModel]);

    // 2b. Deep link to one line: zoom/centre its box, pulse it, scroll the text panel.
    useEffect(() => {
        if (!record || requestedLine == null || requestedLine === '') return undefined;
        const index = Number(requestedLine);
        const line = record.ai_read.lines.find((l) => l.index === index);
        if (!line) return undefined;
        setFocusRequest({ bbox: line.bbox, key: `deeplink-${index}`, zoom: true });
        setPulsed(index);
        const el = document.getElementById(`read-line-${index}`);
        if (el && typeof el.scrollIntoView === 'function') el.scrollIntoView({ block: 'center' });
        const timer = setTimeout(() => setPulsed(null), PULSE_MS);
        return () => clearTimeout(timer);
    }, [record, requestedLine]);

    // 2c. Phones: fitting the whole image renders a two-page opening at ~3 %
    // with forty boxes as hatching. Open on the first confirmed line instead,
    // at the width of its page. A ?line= deep link already handles its own zoom.
    useEffect(() => {
        if (!compact || !record || natural.w === 0) return;
        if (requestedLine != null && requestedLine !== '') return;
        const lines = record.ai_read.lines;
        const first = lines.find((l) => l.status === 'agreed') || lines[0];
        if (!first) return;
        setFocusRequest({ bbox: first.bbox, key: `initial-${record.doc_id}-${record.image_index}`, zoom: 'page' });
    }, [compact, record, natural.w, requestedLine]);

    // 3. Document title/shelfmark (best effort; the page works without it).
    useEffect(() => {
        if (!docId) return;
        const q = sourceIndex ? `?index_name=${encodeURIComponent(sourceIndex)}` : '';
        fetch(`${API_BASE_URL}/document/${encodeURIComponent(docId)}${q}`)
            .then((r) => (r.ok ? r.json() : null))
            .then((m) => setDocMeta(m))
            .catch(() => setDocMeta(null));
    }, [docId, sourceIndex]);

    if (!docId) {
        return (
            <ReadMessage title="Read a fragment with AI">
                Open a document from search and use its “Transcribe with AI (beta)” button.
            </ReadMessage>
        );
    }
    if (error) {
        return <ReadMessage title="Something went wrong">{error}</ReadMessage>;
    }
    if (!status) {
        return <ReadMessage title="Checking availability…">Looking for an AI read of {docId}.</ReadMessage>;
    }
    if (!status.enabled) {
        return <ReadMessage title="AI reads are switched off">This feature is currently disabled.</ReadMessage>;
    }
    if (!status.available) {
        return (
            <ReadMessage title="No AI read yet">
                This fragment has not been read by the offline pipeline, or too few of its lines were confirmed
                by a second reader to show. Only fragments with a published read show the button.
            </ReadMessage>
        );
    }
    if (!record) {
        return <ReadMessage title="Loading read…">Fetching the read for image {imageIndex + 1}.</ReadMessage>;
    }

    const read = record.ai_read;
    const title = docMeta?.title || docMeta?.shelf_mark || docMeta?.shelfmark || docId;
    const shelfmark = docMeta?.shelf_mark || docMeta?.shelfmark;
    const aspectMismatch =
        natural.w > 0 &&
        Math.abs(natural.w / natural.h - record.image_width / record.image_height) > 0.01;

    return (
        <div className="read-page">
            <header className="read-header">
                <div>
                    <Link to="/" className="read-back">← Search</Link>
                    <h1>{title}</h1>
                    {shelfmark && shelfmark !== title && <div className="read-shelfmark">{shelfmark}</div>}
                </div>
                <div className="read-header-right">
                    <div className="read-confirmed-count">
                        <strong>{read.n_agreed} of {read.n_lines} lines</strong> confirmed by a second reader (Kraken)
                    </div>
                    {status.items.length > 1 && (
                        <label className="read-image-picker">
                            Image{' '}
                            <select
                                value={imageIndex}
                                onChange={(e) => {
                                    const next = new URLSearchParams(params);
                                    next.set('image', e.target.value);
                                    setParams(next);
                                }}
                            >
                                {status.items.map((item) => (
                                    <option key={item.image_index} value={item.image_index}>
                                        {item.image_index + 1} · {item.ai_read.n_agreed}/{item.ai_read.n_lines} confirmed
                                    </option>
                                ))}
                            </select>
                        </label>
                    )}
                </div>
            </header>

            <BetaBanner compact={compact} />

            <div className="read-controls">
                <label><input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} /> Line boxes</label>
                <label><input type="checkbox" checked={showNumbers} onChange={(e) => setShowNumbers(e.target.checked)} /> Numbers</label>
                <label><input type="checkbox" checked={showFragments} onChange={(e) => setShowFragments(e.target.checked)} /> Kraken fragments</label>
                <label><input type="checkbox" checked={showHtr} onChange={(e) => setShowHtr(e.target.checked)} /> Show Kraken reading</label>
                <div className="read-legend">
                    <span className="read-legend-item agreed"><i /> Confirmed by two readers ({read.n_agreed})</span>
                    <span className="read-legend-item unconfirmed"><i /> Single reader, caution ({read.n_lines - read.n_agreed})</span>
                    <span className="read-legend-item fragments"><i /> Kraken fragments</span>
                </div>
            </div>

            <div className="read-body">
                <ImageStage
                    record={record}
                    natural={natural}
                    onNatural={setNatural}
                    hovered={hovered}
                    setHovered={setHovered}
                    showBoxes={showBoxes}
                    showNumbers={showNumbers}
                    showFragments={showFragments}
                    focusRequest={focusRequest}
                    pulsed={pulsed}
                    compact={compact}
                    coarse={coarse}
                />
                <aside className="read-text">
                    <TextPanel
                        record={record}
                        hovered={hovered}
                        setHovered={setHovered}
                        showHtr={showHtr}
                        pulsed={pulsed}
                        onFocusLine={(line) => setFocusRequest({ bbox: line.bbox, key: Date.now() })}
                    />
                </aside>
            </div>

            <footer className="read-footer">
                {aspectMismatch && (
                    <div className="read-warning">
                        The displayed image has a different aspect ratio from the one that was read; boxes may be offset.
                    </div>
                )}
                <div className="read-credits">
                    Models trained on Cairo Genizah fragments with Princeton Geniza Project transcriptions and on
                    Hebrew manuscript pages with transcriptions from the National Library of Israel’s KTIV project.
                    Images courtesy of the holding libraries.
                </div>
                <details className="read-raw">
                    <summary>How this was produced (models, rule, raw record)</summary>
                    <dl className="read-provenance">
                        <dt>Vision-language reader</dt>
                        <dd><code>{read.vlm_model}</code>{read.vlm_revision && <> (revision <code>{read.vlm_revision}</code>)</>}</dd>
                        <dt>Second reader (HTR)</dt>
                        <dd><code>{read.htr_model}</code> via the Kraken line service</dd>
                        <dt>Matching rule</dt>
                        <dd><code>{read.rule_version}</code>: Kraken fragments are assigned to a line by its vertical band, one column at a time; a line is confirmed when the two readings agree at 0.8 or above on Hebrew letters only. The drawn box is the union of the model’s line box and the Kraken fragments assigned to it.</dd>
                        <dt>Decoded</dt>
                        <dd>{new Date(read.decoded_at).toLocaleString()} (single decode; a rerun moves individual lines)</dd>
                    </dl>
                    <pre dir="ltr">{JSON.stringify(record, null, 2)}</pre>
                </details>
            </footer>
        </div>
    );
}
