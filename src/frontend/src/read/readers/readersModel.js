/**
 * Pure helpers for the advanced Readers panel on /read (readings-v1 sidecars).
 *
 * Every box is [x1, y1, x2, y2] in 0-1000 of the oriented page, the same frame
 * as the public overlay. Reader lines carry ``box``; final (consensus) lines
 * carry ``bbox``; ``normaliseLines`` gives both a ``box``.
 *
 * Nothing here knows today's reader ids: labels come from ``kind``, ``scope``,
 * ``role`` and ``unit`` so new reader kinds (more VLM checkpoints, a vote)
 * render without code changes.
 */

/** Key of the merged consensus row; also the ``reader_id`` sent with corrections. */
export const FINAL_KEY = 'final';

/** Minimum IoU for two lines to be paired in compare mode. */
export const ALIGN_MIN_IOU = 0.1;

/** Above this many LCS cells a line diff falls back to "all deleted, all inserted". */
const MAX_DIFF_CELLS = 250000;

/**
 * Reader colours: distinct on a light parchment background, and none of them
 * the green/amber that the final line boxes use for agreed/unconfirmed.
 */
export const PALETTE = [
    '#1f77b4', // blue
    '#d62728', // red
    '#9467bd', // purple
    '#17becf', // cyan
    '#c2185b', // magenta
    '#8c564b', // brown
    '#283593', // indigo
    '#00796b', // teal
    '#827717', // olive
    '#546e7a', // slate
];

/** Final line colours by status, matching the public Read view. */
export const FINAL_COLOURS = {
    agreed: '#2e7d32',
    unconfirmed: '#f9a825',
};

/** Badge tone per known health label; anything else is neutral grey. */
const HEALTH_TONES = {
    ok: 'good',
    loop: 'bad',
    overgen: 'bad',
    capped: 'warn',
    near_empty: 'warn',
    skipped_text: 'warn',
    parse_loss: 'warn',
};

/** Short names for the known reader kinds; unknown kinds are capitalised. */
const KIND_LABELS = {
    vlm: 'VLM',
    htr: 'HTR',
};

/** Human names for ``final.source``. */
const FINAL_SOURCE_LABELS = {
    whole_page: 'whole page',
    assembled: 'assembled',
    assembled_partial: 'assembled, partial',
    legacy: 'legacy',
};

/**
 * Upper-case the first letter of a string.
 * @param {string} text - Any text.
 * @returns {string} The text with its first character capitalised.
 */
const capitalise = (text) => (text ? text.charAt(0).toUpperCase() + text.slice(1) : text);

/**
 * Validate a box and order its corners.
 * @param {Array<number>|null|undefined} box - [x1, y1, x2, y2] in 0-1000.
 * @returns {number[]|null} [left, top, right, bottom], or null when the box is
 *     missing, too short or holds a non-number.
 */
export function normaliseBox(box) {
    if (!Array.isArray(box) || box.length < 4) return null;
    const raw = box.slice(0, 4);
    if (raw.some((v) => v === null || v === undefined || v === '')) return null;
    const v = raw.map(Number);
    if (v.some((n) => !Number.isFinite(n))) return null;
    return [Math.min(v[0], v[2]), Math.min(v[1], v[3]), Math.max(v[0], v[2]), Math.max(v[1], v[3])];
}

/**
 * Area of a normalised box.
 * @param {number[]} b - [left, top, right, bottom].
 * @returns {number} Area in 0-1000 units squared.
 */
const area = (b) => (b[2] - b[0]) * (b[3] - b[1]);

/**
 * Intersection over union of two boxes.
 * @param {Array<number>|null} a - First box.
 * @param {Array<number>|null} b - Second box.
 * @returns {number} IoU in [0, 1]; 0 when either box is invalid or they do not overlap.
 */
export function iou(a, b) {
    const A = normaliseBox(a);
    const B = normaliseBox(b);
    if (!A || !B) return 0;
    const ix = Math.max(0, Math.min(A[2], B[2]) - Math.max(A[0], B[0]));
    const iy = Math.max(0, Math.min(A[3], B[3]) - Math.max(A[1], B[1]));
    const inter = ix * iy;
    if (inter <= 0) return 0;
    const union = area(A) + area(B) - inter;
    return union > 0 ? inter / union : 0;
}

/**
 * Whether a point lies inside a box (edges included).
 * @param {number[]|null} box - Normalised box.
 * @param {number} x - X in 0-1000.
 * @param {number} y - Y in 0-1000.
 * @returns {boolean} True when inside.
 */
export function boxContains(box, x, y) {
    return Boolean(box) && x >= box[0] && x <= box[2] && y >= box[1] && y <= box[3];
}

/**
 * Give every line a normalised ``box`` (from ``box`` or ``bbox``), a numeric
 * ``index`` and string ``text``. ``null`` stays ``null`` (unattributed reply).
 * @param {Array<object>|null|undefined} lines - Reader or final lines.
 * @returns {Array<object>|null} Normalised copies, or null.
 */
export function normaliseLines(lines) {
    if (!Array.isArray(lines)) return null;
    return lines.map((line, i) => {
        const raw = line || {};
        return {
            ...raw,
            index: Number.isFinite(raw.index) ? raw.index : i,
            text: typeof raw.text === 'string' ? raw.text : '',
            box: normaliseBox(raw.box || raw.bbox),
        };
    });
}

/**
 * Generic label for a reader from its kind, scope, unit and role.
 * Examples: "VLM · whole page", "VLM · unit col1.top · half", "Vote · whole page".
 * @param {object|null} reader - Reader entry from ``sidecar.readers``.
 * @returns {string} Display label (the raw id is shown separately).
 */
export function readerLabel(reader) {
    if (!reader) return 'Reader';
    const kind = reader.kind ? String(reader.kind) : '';
    const parts = [KIND_LABELS[kind] || (kind ? capitalise(kind.replace(/_/g, ' ')) : 'Reader')];
    if (reader.scope === 'unit' || (reader.unit && reader.scope !== 'page')) {
        parts.push(reader.unit ? `unit ${reader.unit}` : 'unit');
    } else if (reader.scope) {
        parts.push(reader.scope === 'page' ? 'whole page' : String(reader.scope).replace(/_/g, ' '));
    }
    if (reader.role && reader.role !== reader.scope) {
        parts.push(String(reader.role).replace(/_/g, ' '));
    }
    // Future readers may name their own checkpoint; show it when they do.
    const model = [reader.model, reader.vlm_model, reader.checkpoint].find((m) => typeof m === 'string' && m);
    if (model) parts.push(model);
    return parts.join(' · ');
}

/**
 * Label of the final (consensus) row.
 * @param {string|null|undefined} source - ``final.source``.
 * @returns {string} E.g. "Final (assembled)".
 */
export function finalLabel(source) {
    if (!source) return 'Final';
    const name = FINAL_SOURCE_LABELS[source] || String(source).replace(/_/g, ' ');
    return `Final (${name})`;
}

/**
 * Assign each reader a palette colour in reader order (cycling past ten).
 * The same sidecar always gives the same colours.
 * @param {Array<object>} readers - ``sidecar.readers``.
 * @returns {Object<string, string>} Colour per reader id.
 */
export function assignColours(readers) {
    const colours = {};
    let next = 0;
    (readers || []).forEach((reader) => {
        if (!reader || reader.id === null || reader.id === undefined || colours[reader.id]) return;
        colours[reader.id] = PALETTE[next % PALETTE.length];
        next += 1;
    });
    return colours;
}

/**
 * Build the rows of the reader list: the final consensus first, then every
 * reader in sidecar order, each with a unique ``key``, label, colour and
 * normalised lines (``null`` when unattributed).
 * @param {object} sidecar - readings-v1 sidecar.
 * @returns {Array<object>} Entries ``{key, id, isFinal, label, colour, lines, health, reader}``.
 */
export function buildEntries(sidecar) {
    const readers = sidecar && Array.isArray(sidecar.readers) ? sidecar.readers.filter(Boolean) : [];
    const colours = assignColours(readers);
    const entries = [];
    const final = sidecar ? sidecar.final : null;
    if (final) {
        entries.push({
            key: FINAL_KEY,
            id: FINAL_KEY,
            isFinal: true,
            label: finalLabel(final.source),
            colour: FINAL_COLOURS.agreed,
            lines: normaliseLines(final.lines) || [],
            health: null,
            reader: null,
        });
    }
    const seen = new Set([FINAL_KEY]);
    readers.forEach((reader, i) => {
        const id = reader.id !== null && reader.id !== undefined ? String(reader.id) : `reader-${i}`;
        const key = seen.has(id) ? `${id}#${i}` : id;
        seen.add(key);
        entries.push({
            key,
            id,
            isFinal: false,
            label: readerLabel(reader),
            colour: colours[reader.id] || PALETTE[i % PALETTE.length],
            lines: normaliseLines(reader.lines),
            health: reader.health_label || null,
            reader,
        });
    });
    return entries;
}

/**
 * Stroke colour of one line: final lines by status, readers by their colour.
 * @param {object} entry - Entry from ``buildEntries``.
 * @param {object} line - One of its lines.
 * @returns {string} CSS colour.
 */
export function lineColour(entry, line) {
    if (!entry.isFinal) return entry.colour;
    return line && line.status === 'agreed' ? FINAL_COLOURS.agreed : FINAL_COLOURS.unconfirmed;
}

/**
 * CSS background of an entry's swatch (the final row is half green, half amber).
 * @param {object} entry - Entry from ``buildEntries``.
 * @returns {string} CSS background value.
 */
export function swatchBackground(entry) {
    if (!entry.isFinal) return entry.colour;
    return `linear-gradient(135deg, ${FINAL_COLOURS.agreed} 50%, ${FINAL_COLOURS.unconfirmed} 50%)`;
}

/**
 * Line count for the reader list.
 * @param {Array<object>|null} lines - Normalised lines.
 * @returns {string} "unattributed" for null, "0 lines" for [], else "N line(s)".
 */
export function lineCountLabel(lines) {
    if (lines === null || lines === undefined) return 'unattributed';
    return lines.length === 1 ? '1 line' : `${lines.length} lines`;
}

/**
 * For every reader, the line whose box overlaps the final line's box most.
 * @param {object} finalLine - The pinned final line (``box`` or ``bbox``).
 * @param {Array<object>} readers - Readers or entries, each with ``lines``.
 * @returns {Array<{reader: object, line: object|null, iou: number, unattributed: boolean}>}
 *     One result per reader, in input order; ``line`` is null when nothing overlaps.
 */
export function bestMatches(finalLine, readers) {
    const target = finalLine ? normaliseBox(finalLine.box || finalLine.bbox) : null;
    return (readers || []).map((reader) => {
        const lines = normaliseLines(reader ? reader.lines : null);
        if (lines === null) return { reader, line: null, iou: 0, unattributed: true };
        let best = null;
        let bestIou = 0;
        lines.forEach((line) => {
            const score = target ? iou(target, line.box) : 0;
            if (score > bestIou) {
                best = line;
                bestIou = score;
            }
        });
        return { reader, line: best, iou: bestIou, unattributed: false };
    });
}

/**
 * Vertical position used to order compare rows: mean of the boxes' centres.
 * @param {{a: object|null, b: object|null}} row - Aligned row.
 * @returns {number} Y in 0-1000, or a large number when neither side has a box.
 */
const rowTop = (row) => {
    const ys = [row.a, row.b]
        .filter((line) => line && line.box)
        .map((line) => (line.box[1] + line.box[3]) / 2);
    return ys.length ? ys.reduce((s, y) => s + y, 0) / ys.length : 1e9;
};

/**
 * Align two readers' lines one-to-one by box overlap: greedy, highest IoU
 * first, pairs below ``minIou`` left unmatched. Unmatched lines get a row of
 * their own. Rows are ordered top to bottom.
 * @param {Array<object>} linesA - First reader's lines.
 * @param {Array<object>} linesB - Second reader's lines.
 * @param {number} [minIou] - Smallest IoU that still pairs two lines.
 * @returns {Array<{a: object|null, b: object|null, iou: number}>} Aligned rows.
 */
export function alignLines(linesA, linesB, minIou = ALIGN_MIN_IOU) {
    const a = normaliseLines(linesA) || [];
    const b = normaliseLines(linesB) || [];
    const pairs = [];
    a.forEach((la, i) => {
        b.forEach((lb, j) => {
            const score = iou(la.box, lb.box);
            if (score > 0 && score >= minIou) pairs.push({ i, j, score });
        });
    });
    pairs.sort((p, q) => q.score - p.score || p.i - q.i || p.j - q.j);

    const usedA = new Set();
    const usedB = new Set();
    const rows = [];
    pairs.forEach(({ i, j, score }) => {
        if (usedA.has(i) || usedB.has(j)) return;
        usedA.add(i);
        usedB.add(j);
        rows.push({ a: a[i], b: b[j], iou: score, order: i });
    });
    a.forEach((line, i) => {
        if (!usedA.has(i)) rows.push({ a: line, b: null, iou: 0, order: i });
    });
    b.forEach((line, j) => {
        if (!usedB.has(j)) rows.push({ a: null, b: line, iou: 0, order: a.length + j });
    });
    rows.sort((r, s) => rowTop(r) - rowTop(s) || r.order - s.order);
    return rows.map(({ a: la, b: lb, iou: score }) => ({ a: la, b: lb, iou: score }));
}

/** A combining mark (nikud, cantillation, Arabic harakat). */
const COMBINING_MARK = /^\p{M}$/u;

/**
 * Split text into characters: code points (so astral characters stay whole),
 * with combining marks kept on their base letter so a pointed letter is never
 * split across two diff spans.
 * @param {string} text - Any text.
 * @returns {string[]} Characters.
 */
export function splitChars(text) {
    const out = [];
    Array.from(text || '').forEach((ch) => {
        if (out.length && COMBINING_MARK.test(ch)) {
            out[out.length - 1] += ch;
        } else {
            out.push(ch);
        }
    });
    return out;
}

/**
 * Character-level diff of two strings via a longest common subsequence.
 * @param {string} a - Old text (left side).
 * @param {string} b - New text (right side).
 * @returns {Array<{type: 'same'|'del'|'ins', text: string}>} Segments in reading
 *     order; joining ``same`` + ``del`` gives ``a``, ``same`` + ``ins`` gives ``b``.
 */
export function charDiff(a, b) {
    const x = splitChars(a);
    const y = splitChars(b);
    const n = x.length;
    const m = y.length;
    const segments = [];
    const push = (type, text) => {
        const last = segments[segments.length - 1];
        if (last && last.type === type) {
            last.text += text;
        } else {
            segments.push({ type, text });
        }
    };

    if (n * m > MAX_DIFF_CELLS) {
        if (n) push('del', x.join(''));
        if (m) push('ins', y.join(''));
        return segments;
    }

    // lcs[i][j] = length of the LCS of x[i..] and y[j..].
    const lcs = [];
    for (let i = 0; i <= n; i += 1) lcs.push(new Int32Array(m + 1));
    for (let i = n - 1; i >= 0; i -= 1) {
        for (let j = m - 1; j >= 0; j -= 1) {
            lcs[i][j] = x[i] === y[j] ? lcs[i + 1][j + 1] + 1 : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
        }
    }

    let i = 0;
    let j = 0;
    while (i < n && j < m) {
        if (x[i] === y[j]) {
            push('same', x[i]);
            i += 1;
            j += 1;
        } else if (lcs[i + 1][j] >= lcs[i][j + 1]) {
            push('del', x[i]);
            i += 1;
        } else {
            push('ins', y[j]);
            j += 1;
        }
    }
    while (i < n) {
        push('del', x[i]);
        i += 1;
    }
    while (j < m) {
        push('ins', y[j]);
        j += 1;
    }
    return segments;
}

/**
 * One-line history of a policy unit: each attempt as "source (label)", then
 * how it was resolved. E.g. "col1: crop (loop) → halves (loop) → share".
 * @param {object} unit - Entry of ``policy.units``.
 * @returns {string} Summary.
 */
export function unitSummary(unit) {
    if (!unit) return '';
    const attempts = Array.isArray(unit.attempts) ? unit.attempts : [];
    const steps = attempts.map((attempt) => {
        const source = attempt && attempt.source ? String(attempt.source) : '?';
        return attempt && attempt.label ? `${source} (${attempt.label})` : source;
    });
    if (unit.resolved_by) steps.push(String(unit.resolved_by));
    const name = unit.name ? String(unit.name) : 'unit';
    return steps.length ? `${name}: ${steps.join(' → ')}` : name;
}

/**
 * Badge tone of a health label.
 * @param {string|null|undefined} label - ``health_label``.
 * @returns {'good'|'warn'|'bad'|'neutral'} Tone; unknown labels are neutral.
 */
export function healthTone(label) {
    return (label && HEALTH_TONES[label]) || 'neutral';
}

/**
 * All visible lines whose box contains a point, for the hover readout.
 * @param {Array<object>} entries - Entries from ``buildEntries``.
 * @param {Object<string, boolean>} visible - Visibility per entry key.
 * @param {number} x - X in 0-1000.
 * @param {number} y - Y in 0-1000.
 * @returns {Array<{entry: object, line: object}>} Hits in entry order.
 */
export function linesAtPoint(entries, visible, x, y) {
    const hits = [];
    entries.forEach((entry) => {
        if (!visible[entry.key] || !entry.lines) return;
        entry.lines.forEach((line) => {
            if (boxContains(line.box, x, y)) hits.push({ entry, line });
        });
    });
    return hits;
}

/**
 * The final line under a point; the smallest box wins where boxes overlap.
 * @param {Array<object>} lines - Normalised final lines.
 * @param {number} x - X in 0-1000.
 * @param {number} y - Y in 0-1000.
 * @returns {object|null} The line, or null.
 */
export function pickLineAt(lines, x, y) {
    let best = null;
    (lines || []).forEach((line) => {
        if (!boxContains(line.box, x, y)) return;
        if (!best || area(line.box) < area(best.box)) best = line;
    });
    return best;
}

/**
 * The final line a stored correction refers to: same index when it was made
 * on the final row, else the final line overlapping its box most.
 * @param {object} correction - Correction record.
 * @param {Array<object>} finalLines - Normalised final lines.
 * @returns {object|null} The line, or null when it no longer exists.
 */
export function findCorrectionLine(correction, finalLines) {
    if (!correction || !finalLines || !finalLines.length) return null;
    if (correction.reader_id === FINAL_KEY) {
        const same = finalLines.find((line) => line.index === correction.line_index);
        if (same) return same;
    }
    const [match] = bestMatches({ bbox: correction.bbox }, [{ lines: finalLines }]);
    return match && match.iou > 0 ? match.line : null;
}

/**
 * Group the /readings/pages list by image (a page can have one sidecar per model).
 * @param {Array<object>} pages - ``pages`` from GET /documents/{id}/readings/pages.
 * @returns {Array<object>} One item per image, sorted by index, with ``models``.
 */
export function groupPages(pages) {
    const byImage = new Map();
    (pages || []).forEach((page) => {
        if (!page || !Number.isFinite(page.image_index)) return;
        const existing = byImage.get(page.image_index);
        if (existing) {
            if (page.vlm_model) existing.models.push(page.vlm_model);
        } else {
            byImage.set(page.image_index, { ...page, models: page.vlm_model ? [page.vlm_model] : [] });
        }
    });
    return Array.from(byImage.values()).sort((p, q) => p.image_index - q.image_index);
}

/**
 * Format a number to fixed digits, or a dash when it is not a number.
 * @param {*} value - Value to show.
 * @param {number} [digits] - Decimal places.
 * @returns {string} Formatted value.
 */
export function formatNumber(value, digits = 2) {
    return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
}

/**
 * Display a policy value of any type.
 * @param {*} value - String, number, boolean or object.
 * @returns {string} Text.
 */
export function formatValue(value) {
    if (value !== null && typeof value === 'object') return JSON.stringify(value);
    return String(value);
}

/**
 * Local date and time of an ISO timestamp.
 * @param {string|null|undefined} value - ISO timestamp.
 * @returns {string} Localised text, the raw value when unparseable, '' when empty.
 */
export function formatTime(value) {
    if (!value) return '';
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
}

/**
 * Shorten a hash for display.
 * @param {string|null|undefined} hash - Hex digest.
 * @returns {string} First 12 characters plus an ellipsis, or '' when empty.
 */
export function shortHash(hash) {
    if (!hash) return '';
    const text = String(hash);
    return text.length > 12 ? `${text.slice(0, 12)}…` : text;
}
