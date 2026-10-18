import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Swatch } from './ReaderList';
import { hitStack, lineColour, nextHit, normaliseBox, paddedBox } from './readersModel';

/** Zoom bounds shared by the wheel and the buttons. */
const MIN_ZOOM = 0.02;
const MAX_ZOOM = 12;
/** A press that moves less than this (px) is a click, not a drag. */
const DRAG_THRESHOLD = 4;
/** Most rows the hover readout lists at once. */
const MAX_READOUT = 8;

/**
 * Convert a 0-1000 normalised box to pixel coordinates of the natural image.
 * @param {number[]} box - [x1, y1, x2, y2] normalised to 0-1000.
 * @param {number} w - Natural image width in pixels.
 * @param {number} h - Natural image height in pixels.
 * @returns {number[]} [x0, y0, x1, y1] in pixels.
 */
const toPx = (box, w, h) => [(box[0] / 1000) * w, (box[1] / 1000) * h, (box[2] / 1000) * w, (box[3] / 1000) * h];

/**
 * Clamp a zoom factor to the stage's bounds.
 * @param {number} z - Requested zoom.
 * @returns {number} Zoom within [MIN_ZOOM, MAX_ZOOM].
 */
const clampZoom = (z) => Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, z));

/**
 * Stable key of one hovered line.
 * @param {{entry: object, line: object}} hit - Hit from ``hitStack``.
 * @returns {string} Key.
 */
const hitKey = (hit) => `${hit.entry.key}:${hit.line.index}`;

/**
 * Whether a hit is the selected line.
 * @param {{entry: object, line: object}} hit - Hit from ``hitStack``.
 * @param {{entry: object, line: object}|null} selected - Selected line and its entry.
 * @returns {boolean} True for the selected line.
 */
const isSelectedHit = (hit, selected) =>
    Boolean(selected) && hit.entry.key === selected.entry.key && hit.line.index === selected.line.index;

/**
 * Page image with every visible reader's boxes, the policy units as dashed
 * rectangles, wheel/button zoom and drag pan. A visible unit reader whose
 * unit is not a policy unit (the halves of a failed column, ``col0.top``)
 * also shows the crop the model saw, as a thin dashed rectangle in its colour.
 * Readers past the tenth repeat a colour with a dash pattern (``entry.dash``).
 * Hover and click are hit-tested in page coordinates (not on SVG elements),
 * so a drag never selects a line. A click selects the topmost visible box
 * under the pointer, of any reader; clicking the same spot again steps to
 * the box below it (``hitStack``, ``nextHit``).
 * @param {object} props - Component props.
 * @param {string} props.imageUrl - ``sidecar.image_url``.
 * @param {string} props.altText - Image alt text.
 * @param {number} props.imageWidth - ``sidecar.image_width`` (the decoded image).
 * @param {number} props.imageHeight - ``sidecar.image_height``.
 * @param {Array<object>} props.entries - Entries from ``buildEntries``.
 * @param {Object<string, boolean>} props.visible - Visibility per entry key.
 * @param {{entry: object, line: object}|null} props.selected - Selected line and its entry
 *     (``findSelection``), drawn last and thicker even when its entry is hidden.
 * @param {Array<object>} props.units - ``policy.units`` (may be empty).
 * @param {object|null} props.focusRequest - ``{box, key}``: centre this box.
 * @param {Object<string, number>} [props.recency] - When each entry was last shown; the reader
 *     shown most recently wins a click where boxes overlap (``hitStack``).
 * @param {function(string, number, Array<object>): void} props.onSelect - Select a line by entry
 *     key and line index; the third argument is every hit under the click (``hitStack``).
 */
export default function ReadersStage({
    imageUrl,
    altText,
    imageWidth,
    imageHeight,
    entries,
    visible,
    selected,
    units,
    focusRequest,
    recency,
    onSelect,
}) {
    const viewportRef = useRef(null);
    // Natural size is stored with its URL so a new page never inherits the old size.
    const [natural, setNatural] = useState({ url: '', w: 0, h: 0 });
    const [failedUrl, setFailedUrl] = useState('');
    const [view, setView] = useState({ z: 1, tx: 0, ty: 0 });
    const [hits, setHits] = useState([]);
    const hitSignatureRef = useRef('');
    const dragRef = useRef(null);
    const loaded = natural.url === imageUrl;
    const W = loaded ? natural.w : 0;
    const H = loaded ? natural.h : 0;

    const fit = useCallback(() => {
        const vp = viewportRef.current;
        if (!vp || !W || !H) return;
        const r = vp.getBoundingClientRect();
        if (r.width < 20 || r.height < 20) return;
        const z = clampZoom(Math.min(r.width / W, r.height / H) * 0.96);
        setView({ z, tx: (r.width - W * z) / 2, ty: (r.height - H * z) / 2 });
    }, [W, H]);

    // Fit when the image loads, and again when the stage changes width.
    useEffect(() => {
        fit();
        let lastWidth = viewportRef.current ? viewportRef.current.getBoundingClientRect().width : 0;
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

    // Bring a requested line (picked from a list) into view, keeping the zoom.
    // A line already fully on screen stays put: centring it at fit-to-panel
    // zoom would shove the whole page aside for nothing.
    useEffect(() => {
        if (!focusRequest || !focusRequest.box || !W || !H) return;
        const vp = viewportRef.current;
        if (!vp) return;
        const r = vp.getBoundingClientRect();
        const [x0, y0, x1, y1] = toPx(paddedBox(focusRequest.box), W, H);
        setView((v) => {
            const margin = 8;
            const onScreen =
                x0 * v.z + v.tx >= margin &&
                y0 * v.z + v.ty >= margin &&
                x1 * v.z + v.tx <= r.width - margin &&
                y1 * v.z + v.ty <= r.height - margin;
            if (onScreen) return v;
            return { z: v.z, tx: r.width / 2 - ((x0 + x1) / 2) * v.z, ty: r.height / 2 - ((y0 + y1) / 2) * v.z };
        });
    }, [focusRequest, W, H]);

    // A new sidecar (page or model) clears the hover readout.
    useEffect(() => {
        hitSignatureRef.current = '';
        setHits([]);
    }, [entries]);

    /**
     * Pointer position in viewport pixels and in 0-1000 page coordinates.
     * @param {PointerEvent} e - Pointer event.
     * @returns {{vx: number, vy: number, x: number, y: number}} Both positions.
     */
    const pointerPosition = (e) => {
        const r = viewportRef.current.getBoundingClientRect();
        const vx = e.clientX - r.left;
        const vy = e.clientY - r.top;
        return {
            vx,
            vy,
            x: W ? (((vx - view.tx) / view.z) / W) * 1000 : -1,
            y: H ? (((vy - view.ty) / view.z) / H) * 1000 : -1,
        };
    };

    /**
     * Update the hover readout with every visible box under the pointer.
     * @param {PointerEvent} e - Pointer event.
     */
    const updateHover = (e) => {
        if (!W || !H) return;
        const p = pointerPosition(e);
        const found = hitStack(entries, visible, p.x, p.y, recency);
        const signature = found.map(hitKey).join('|');
        if (signature === hitSignatureRef.current) return;
        hitSignatureRef.current = signature;
        setHits(found);
    };

    const onPointerDown = (e) => {
        if (e.pointerType === 'mouse' && e.button !== 0) return;
        const p = pointerPosition(e);
        dragRef.current = { id: e.pointerId, x: p.vx, y: p.vy, tx: view.tx, ty: view.ty, moved: false };
    };

    const onPointerMove = (e) => {
        const drag = dragRef.current;
        if (drag && drag.id === e.pointerId) {
            const p = pointerPosition(e);
            const dx = p.vx - drag.x;
            const dy = p.vy - drag.y;
            if (!drag.moved && Math.hypot(dx, dy) < DRAG_THRESHOLD) return;
            if (!drag.moved) {
                // Capture only once it is a real drag, so plain clicks stay clicks.
                drag.moved = true;
                e.currentTarget.setPointerCapture(e.pointerId);
            }
            setView((v) => ({ z: v.z, tx: drag.tx + dx, ty: drag.ty + dy }));
            return;
        }
        if (e.pointerType === 'mouse') updateHover(e);
    };

    const onPointerUp = (e) => {
        const drag = dragRef.current;
        dragRef.current = null;
        if (!drag || drag.id !== e.pointerId || drag.moved || !W || !H) return;
        // A click: select the topmost box under the pointer, or the one below the
        // selected line when that is already under it; on touch show what is there.
        // A click on bare image keeps the selection.
        const p = pointerPosition(e);
        const current = selected ? { key: selected.entry.key, index: selected.line.index } : null;
        const stack = hitStack(entries, visible, p.x, p.y, recency);
        const hit = nextHit(stack, current);
        if (hit) onSelect(hit.entry.key, hit.line.index, stack);
        if (e.pointerType !== 'mouse') updateHover(e);
    };

    const onPointerCancel = () => {
        dragRef.current = null;
    };

    const onPointerLeave = (e) => {
        if (e.pointerType !== 'mouse' || dragRef.current) return;
        hitSignatureRef.current = '';
        setHits([]);
    };

    const hoverKeys = new Set(hits.map(hitKey));
    // The selected line is drawn last and thicker in its own colour, even when its entry is hidden.
    let selectedRect = null;
    if (W && H && selected && selected.line.box) {
        // Padded so a zero-height line clamped to the page edge stays visible.
        const [x0, y0, x1, y1] = toPx(paddedBox(selected.line.box), W, H);
        selectedRect = { x: x0, y: y0, width: Math.max(1, x1 - x0), height: Math.max(1, y1 - y0) };
    }
    const selectedColour = selected ? lineColour(selected.entry, selected.line) : '';
    const fontSize = 12 / view.z;
    const canSelect = entries.some(
        (entry) => visible[entry.key] && Array.isArray(entry.lines) && entry.lines.length > 0
    );
    // Crops of visible unit readers, unless the same unit is already drawn from policy.units.
    const drawnUnits = new Set(
        units.filter((unit) => unit && unit.name && normaliseBox(unit.crop_box_1000)).map((unit) => String(unit.name))
    );
    const readerCrops = [];
    if (W && H) {
        entries.forEach((entry) => {
            const reader = entry.reader;
            if (!visible[entry.key] || entry.isFinal || !reader || reader.scope !== 'unit') return;
            if (reader.unit && drawnUnits.has(String(reader.unit))) return;
            const box = normaliseBox(reader.crop_box_1000);
            if (!box) return;
            const [x0, y0, x1, y1] = toPx(box, W, H);
            readerCrops.push({
                key: entry.key,
                colour: entry.colour,
                name: reader.unit || entry.label,
                x0,
                y0,
                x1,
                y1,
            });
        });
    }
    const aspectMismatch =
        loaded &&
        imageWidth > 0 &&
        imageHeight > 0 &&
        Math.abs(W / H / (imageWidth / imageHeight) - 1) > 0.01;

    return (
        <div className="readers-stage">
            <div className="readers-stage-toolbar">
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
                <span className="readers-zoom-level">{Math.round(view.z * 100)}%</span>
                <span className="readers-stage-hint">
                    {canSelect
                        ? 'Scroll to zoom, drag to pan, click any box to see that line in every reader.'
                        : 'Scroll to zoom, drag to pan.'}
                </span>
            </div>
            <div
                className="readers-viewport"
                ref={viewportRef}
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                onPointerCancel={onPointerCancel}
                onPointerLeave={onPointerLeave}
            >
                {failedUrl === imageUrl && <div className="readers-stage-message">Could not load the page image.</div>}
                <div
                    className="readers-canvas"
                    style={{
                        width: W ? W * view.z : 'auto',
                        height: H ? H * view.z : 'auto',
                        left: view.tx,
                        top: view.ty,
                    }}
                >
                    <img
                        src={imageUrl}
                        alt={altText}
                        draggable={false}
                        onLoad={(e) => setNatural({ url: imageUrl, w: e.target.naturalWidth, h: e.target.naturalHeight })}
                        onError={() => setFailedUrl(imageUrl)}
                    />
                    {W > 0 && H > 0 && (
                        <svg className="readers-overlay" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
                            {units.map((unit, k) => {
                                const box = normaliseBox(unit && unit.crop_box_1000);
                                if (!box) return null;
                                const [x0, y0, x1, y1] = toPx(box, W, H);
                                return (
                                    <g key={`unit-${unit.name || k}`}>
                                        <rect
                                            className="readers-unit"
                                            x={x0}
                                            y={y0}
                                            width={Math.max(1, x1 - x0)}
                                            height={Math.max(1, y1 - y0)}
                                        />
                                        <text
                                            className="readers-unit-label"
                                            x={x0 + 4 / view.z}
                                            y={y0 + fontSize * 1.2}
                                            style={{ fontSize }}
                                        >
                                            {`${unit.name || 'unit'}: ${unit.health_label || '?'} → ${unit.resolved_by || '?'}`}
                                        </text>
                                    </g>
                                );
                            })}
                            {readerCrops.map((crop) => (
                                <g key={`crop-${crop.key}`}>
                                    <rect
                                        className="readers-reader-crop"
                                        x={crop.x0}
                                        y={crop.y0}
                                        width={Math.max(1, crop.x1 - crop.x0)}
                                        height={Math.max(1, crop.y1 - crop.y0)}
                                        style={{ stroke: crop.colour }}
                                    />
                                    <text
                                        className="readers-reader-crop-label"
                                        x={crop.x0 + 4 / view.z}
                                        y={crop.y1 - 4 / view.z}
                                        style={{ fontSize: fontSize * 0.9, fill: crop.colour }}
                                    >
                                        {crop.name}
                                    </text>
                                </g>
                            ))}
                            {entries
                                .filter((entry) => visible[entry.key] && entry.lines)
                                .map((entry) => (
                                    <g key={entry.key}>
                                        {entry.lines.map((line, i) => {
                                            if (!line.box) return null;
                                            const [x0, y0, x1, y1] = toPx(paddedBox(line.box), W, H);
                                            const colour = lineColour(entry, line);
                                            const hovered = hoverKeys.has(`${entry.key}:${line.index}`);
                                            return (
                                                <rect
                                                    key={`${line.index}-${i}`}
                                                    className={`readers-box${hovered ? ' hover' : ''}`}
                                                    x={x0}
                                                    y={y0}
                                                    width={Math.max(1, x1 - x0)}
                                                    height={Math.max(1, y1 - y0)}
                                                    style={{
                                                        stroke: colour,
                                                        fill: colour,
                                                        strokeDasharray: entry.dash || undefined,
                                                    }}
                                                />
                                            );
                                        })}
                                    </g>
                                ))}
                            {selectedRect && (
                                <g>
                                    <rect
                                        className="readers-selected-halo"
                                        {...selectedRect}
                                        style={{ stroke: selectedColour }}
                                    />
                                    <rect
                                        className="readers-selected"
                                        {...selectedRect}
                                        style={{
                                            stroke: selectedColour,
                                            fill: selectedColour,
                                            strokeDasharray: selected.entry.dash || undefined,
                                        }}
                                    />
                                </g>
                            )}
                        </svg>
                    )}
                </div>
            </div>
            <div className="readers-readout" aria-live="polite">
                {hits.length === 0 && !selected && <span className="readers-muted">Hover a box to read its line.</span>}
                {hits.length === 0 && selected && (
                    <div className="readers-readout-row selected">
                        <Swatch entry={selected.entry} />
                        <span className="readers-readout-label">
                            Selected: {selected.entry.label} · line {selected.line.index + 1}
                        </span>
                        <span className="readers-hebrew" dir="auto">{selected.line.text || '∅'}</span>
                    </div>
                )}
                {hits.slice(0, MAX_READOUT).map((hit) => (
                    <div
                        key={hitKey(hit)}
                        className={`readers-readout-row${isSelectedHit(hit, selected) ? ' selected' : ''}`}
                    >
                        <Swatch entry={hit.entry} />
                        <span className="readers-readout-label">
                            {hit.entry.label} · line {hit.line.index + 1}
                        </span>
                        <span className="readers-hebrew" dir="auto">{hit.line.text || '∅'}</span>
                    </div>
                ))}
                {hits.length > MAX_READOUT && (
                    <span className="readers-muted">and {hits.length - MAX_READOUT} more</span>
                )}
                {hits.length > 1 && (
                    <div className="readers-muted readers-readout-tip">
                        Overlapping boxes, topmost first: each click selects the next one.
                    </div>
                )}
            </div>
            {aspectMismatch && (
                <div className="readers-warning">
                    The displayed image has a different aspect ratio from the one that was read; boxes may be offset.
                </div>
            )}
        </div>
    );
}
