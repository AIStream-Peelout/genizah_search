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

/**
 * Box dash pattern per pass through the palette (SVG ``stroke-dasharray``):
 * the first ten readers draw solid boxes, the next ten dashed, and so on, so
 * two readers that share a colour still look different.
 */
export const DASHES = [null, '6 3', '2 2', '8 3 2 3'];

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

/** Smallest extent (0-1000 units) a box gets for matching, hit-testing and drawing. */
export const MIN_BOX_EXTENT = 4;

/**
 * Whether a box has zero width or height. The pipeline clamps a model box
 * that runs past the page edge, so looping lines often end up as
 * [x1, 1000, x2, 1000] (about 3 % of all final lines in the corpus).
 * @param {Array<number>|null|undefined} box - Box in 0-1000.
 * @returns {boolean} True for a valid box with zero width or height.
 */
export function isDegenerate(box) {
    const b = normaliseBox(box);
    return Boolean(b) && (b[2] - b[0] <= 0 || b[3] - b[1] <= 0);
}

/**
 * A box widened to at least ``min`` in each direction, centred and kept on
 * the page, so a zero-area box can still be seen, hovered and matched. The
 * stored box is never changed.
 * @param {Array<number>|null|undefined} box - Box in 0-1000.
 * @param {number} [min] - Smallest width and height.
 * @returns {number[]|null} [left, top, right, bottom], or null for an invalid box.
 */
export function paddedBox(box, min = MIN_BOX_EXTENT) {
    const b = normaliseBox(box);
    if (!b) return null;
    const pad = (lo, hi) => {
        if (hi - lo >= min) return [lo, hi];
        const mid = (lo + hi) / 2;
        const start = Math.min(Math.max(0, mid - min / 2), 1000 - min);
        return [start, start + min];
    };
    const [x1, x2] = pad(b[0], b[2]);
    const [y1, y2] = pad(b[1], b[3]);
    return [x1, y1, x2, y2];
}

/**
 * Intersection over union of two boxes. Zero-area boxes are padded first
 * (``paddedBox``), so two readers that both put a line on the page edge
 * still match.
 * @param {Array<number>|null} a - First box.
 * @param {Array<number>|null} b - Second box.
 * @returns {number} IoU in [0, 1]; 0 when either box is invalid or they do not overlap.
 */
export function iou(a, b) {
    const A = paddedBox(a);
    const B = paddedBox(b);
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
 * Colour and dash pattern of the n-th reader: colours cycle through the
 * palette, and each pass through it uses the next dash pattern.
 * @param {number} n - 0-based position among the readers.
 * @returns {{colour: string, dash: string|null}} Style for that reader.
 */
export function paletteStyle(n) {
    const cycle = Math.floor(n / PALETTE.length);
    return { colour: PALETTE[n % PALETTE.length], dash: DASHES[cycle % DASHES.length] };
}

/**
 * Assign each reader a palette colour in reader order; past ten readers the
 * colours repeat with a dash pattern. The same sidecar always gives the same styles.
 * @param {Array<object>} readers - ``sidecar.readers``.
 * @returns {Object<string, {colour: string, dash: string|null}>} Style per reader id.
 */
export function assignColours(readers) {
    const styles = {};
    let next = 0;
    (readers || []).forEach((reader) => {
        if (!reader || reader.id === null || reader.id === undefined || styles[reader.id]) return;
        styles[reader.id] = paletteStyle(next);
        next += 1;
    });
    return styles;
}

/**
 * Build the rows of the reader list: the final consensus first, then every
 * reader in sidecar order, each with a unique ``key``, label, colour, dash
 * pattern and normalised lines (``null`` when unattributed). ``parsed`` is the
 * reader's flag (null when it does not say) and ``note`` why its lines are null.
 * @param {object} sidecar - readings-v1 sidecar.
 * @returns {Array<object>} Entries ``{key, id, isFinal, label, colour, dash, lines,
 *     health, parsed, note, reader}``.
 */
export function buildEntries(sidecar) {
    const readers = sidecar && Array.isArray(sidecar.readers) ? sidecar.readers.filter(Boolean) : [];
    const styles = assignColours(readers);
    const entries = [];
    const final = sidecar ? sidecar.final : null;
    if (final) {
        entries.push({
            key: FINAL_KEY,
            id: FINAL_KEY,
            isFinal: true,
            label: finalLabel(final.source),
            colour: FINAL_COLOURS.agreed,
            dash: null,
            lines: normaliseLines(final.lines) || [],
            health: null,
            parsed: null,
            note: null,
            reader: null,
        });
    }
    const seen = new Set([FINAL_KEY]);
    readers.forEach((reader, i) => {
        const id = reader.id !== null && reader.id !== undefined ? String(reader.id) : `reader-${i}`;
        const key = seen.has(id) ? `${id}#${i}` : id;
        seen.add(key);
        const style = styles[reader.id] || paletteStyle(i);
        let parsed = null;
        if (reader.parsed === true || reader.parsed === false) parsed = reader.parsed;
        entries.push({
            key,
            id,
            isFinal: false,
            label: readerLabel(reader),
            colour: style.colour,
            dash: style.dash,
            lines: normaliseLines(reader.lines),
            health: reader.health_label || null,
            parsed,
            note: typeof reader.note === 'string' && reader.note ? reader.note : null,
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
 * CSS background of an entry's swatch (the final row is half green, half
 * amber). A reader with a dash pattern gets stripes that follow the pattern:
 * its colour for each dash, white for each gap.
 * @param {object} entry - Entry from ``buildEntries``.
 * @returns {string} CSS background value.
 */
export function swatchBackground(entry) {
    if (entry.isFinal) {
        return `linear-gradient(135deg, ${FINAL_COLOURS.agreed} 50%, ${FINAL_COLOURS.unconfirmed} 50%)`;
    }
    if (!entry.dash) return entry.colour;
    const parts = String(entry.dash)
        .split(/[\s,]+/)
        .map(Number)
        .filter((n) => Number.isFinite(n) && n > 0);
    if (!parts.length) return entry.colour;
    // SVG repeats an odd-length dash list twice; do the same so dashes and gaps alternate.
    const pattern = parts.length % 2 ? parts.concat(parts) : parts;
    let at = 0;
    const stops = pattern.map((length, k) => {
        const from = at;
        at += length;
        return `${k % 2 ? '#fff' : entry.colour} ${from}px ${at}px`;
    });
    return `repeating-linear-gradient(135deg, ${stops.join(', ')})`;
}

/**
 * Line count for the reader list.
 * @param {Array<object>|null} lines - Normalised lines.
 * @param {boolean|null} [parsed] - The reader's ``parsed`` flag (``entry.parsed``).
 * @returns {string} "reply did not parse" when parsed is false and there are
 *     no lines, "unattributed" for null, "0 lines" for [], else "N line(s)".
 */
export function lineCountLabel(lines, parsed) {
    const none = lines === null || lines === undefined;
    if (parsed === false && (none || lines.length === 0)) return 'reply did not parse';
    if (none) return 'unattributed';
    return lines.length === 1 ? '1 line' : `${lines.length} lines`;
}

/**
 * Why an entry has no lines to show, when it has none.
 * @param {object|null} entry - Entry from ``buildEntries``.
 * @returns {string} For null lines the reader's note, else "reply did not parse"
 *     or "reply not attributed"; for [] "reply did not parse" or "found no lines";
 *     '' when the entry has lines.
 */
export function noLinesReason(entry) {
    if (!entry) return '';
    const { lines } = entry;
    if (lines === null || lines === undefined) {
        if (entry.note) return entry.note;
        return entry.parsed === false ? 'reply did not parse' : 'reply not attributed';
    }
    if (lines.length > 0) return '';
    return entry.parsed === false ? 'reply did not parse' : 'found no lines';
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
 * Whether a policy unit was resolved: ``resolved`` is true, or ``resolved_by``
 * names a source rather than "unresolved".
 * @param {object|null} unit - Entry of ``policy.units``.
 * @returns {boolean|null} True or false, or null when the unit says neither.
 */
export function isUnitResolved(unit) {
    if (!unit) return null;
    const by = unit.resolved_by ? String(unit.resolved_by) : '';
    if (unit.resolved === true || (by && by !== 'unresolved')) return true;
    if (unit.resolved === false || by === 'unresolved') return false;
    return null;
}

/**
 * One-line history of a policy unit: each attempt as "source (label)" in the
 * order tried, then the outcome as a separate tag (never as one more step).
 * E.g. "col1: crop (loop) → halves (loop) → share (ok) · resolved by share",
 * "col0: crop (loop) → halves (loop) · unresolved".
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
    const by = unit.resolved_by ? String(unit.resolved_by) : '';
    const resolved = isUnitResolved(unit);
    let outcome = '';
    if (resolved === true) outcome = by && by !== 'unresolved' ? `resolved by ${by}` : 'resolved';
    else if (resolved === false) outcome = 'unresolved';
    const name = unit.name ? String(unit.name) : 'unit';
    const head = steps.length ? `${name}: ${steps.join(' → ')}` : name;
    return outcome ? `${head} · ${outcome}` : head;
}

/**
 * The evidence of one attempt on a unit, in one line: source, label, flags,
 * letters, line count and the reply (or replies) it came from. The attempt's
 * ``note`` is left to the caller.
 * @param {object|null} attempt - Entry of ``unit.attempts``.
 * @returns {string} E.g. "halves · loop · flags loop, capped · 1822 letters · 98 lines · replies 4, 5".
 */
export function attemptDetail(attempt) {
    if (!attempt) return '';
    const parts = [attempt.source ? String(attempt.source) : '?'];
    if (attempt.label) parts.push(String(attempt.label));
    if (Array.isArray(attempt.flags)) {
        parts.push(attempt.flags.length ? `flags ${attempt.flags.join(', ')}` : 'no flags');
    }
    if (Number.isFinite(attempt.letters)) parts.push(`${attempt.letters} letters`);
    if (Number.isFinite(attempt.n_lines)) parts.push(attempt.n_lines === 1 ? '1 line' : `${attempt.n_lines} lines`);
    if (Array.isArray(attempt.reply_indices) && attempt.reply_indices.length) {
        parts.push(`replies ${attempt.reply_indices.join(', ')}`);
    } else if (Number.isFinite(attempt.reply_index)) {
        parts.push(`reply ${attempt.reply_index}`);
    }
    return parts.join(' · ');
}

/** Order in which an unresolved unit's best effort is taken on an assembled_partial page. */
const BEST_EFFORT_ORDER = ['halves', 'crop', 'share'];

/**
 * What an unresolved unit gave the final lines: on an ``assembled_partial``
 * page its best effort (halves, else crop, else share, among the attempts it
 * has), and nothing when the page fell back to the whole-page read.
 * @param {object|null} unit - Entry of ``policy.units``.
 * @param {object|null} policy - ``sidecar.policy``.
 * @returns {string} A short sentence, or '' for a resolved unit or when the policy does not say.
 */
export function unitContribution(unit, policy) {
    if (!unit || !policy || isUnitResolved(unit) !== false) return '';
    if (policy.final_source === 'assembled_partial') {
        const sources = (Array.isArray(unit.attempts) ? unit.attempts : []).map((a) => a && a.source);
        const best = BEST_EFFORT_ORDER.find((source) => sources.includes(source));
        return best
            ? `Best effort in the final lines: the ${best} attempt.`
            : 'No attempt to take a best effort from.';
    }
    if (policy.final_source === 'whole_page' && policy.fallback) {
        return `Contributes nothing: the page fell back to the whole-page read (${formatValue(policy.fallback)}).`;
    }
    return '';
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
            if (boxContains(paddedBox(line.box), x, y)) hits.push({ entry, line });
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
    let bestArea = 0;
    (lines || []).forEach((line) => {
        const box = paddedBox(line.box);
        if (!boxContains(box, x, y)) return;
        if (!best || area(box) < bestArea) {
            best = line;
            bestArea = area(box);
        }
    });
    return best;
}

/**
 * Whether two timestamps are the same instant ("+00:00" and "Z" forms match).
 * Unparseable values are compared as text.
 * @param {string|null|undefined} a - ISO timestamp.
 * @param {string|null|undefined} b - ISO timestamp.
 * @returns {boolean} True when both are set and name the same instant.
 */
export function sameInstant(a, b) {
    if (!a || !b) return false;
    const ta = Date.parse(a);
    const tb = Date.parse(b);
    if (Number.isNaN(ta) || Number.isNaN(tb)) return String(a) === String(b);
    return ta === tb;
}

/**
 * The final line a stored correction refers to. Its line index is trusted only
 * when it was made on the final row of this very read (same model, same
 * decode and, when both are known, the same image file); any other suggestion
 * is matched to the final line overlapping its box most (boxes share the page
 * frame across models). A suggestion made on a different image file matches nothing.
 * @param {object} correction - Correction record.
 * @param {Array<object>} finalLines - Normalised final lines.
 * @param {object} [sidecar] - Sidecar being shown.
 * @returns {object|null} The line, or null when it no longer exists.
 */
export function findCorrectionLine(correction, finalLines, sidecar) {
    if (!correction || !finalLines || !finalLines.length) return null;
    const read = sidecar || {};
    if (correction.page_hash && read.image_sha256 && correction.page_hash !== read.image_sha256) return null;
    const sameRead =
        Boolean(correction.vlm_model) &&
        correction.vlm_model === read.vlm_model &&
        sameInstant(correction.decoded_at, read.decoded_at);
    if (correction.reader_id === FINAL_KEY && sameRead) {
        const same = finalLines.find((line) => line.index === correction.line_index);
        if (same) return same;
    }
    const [match] = bestMatches({ bbox: correction.bbox }, [{ lines: finalLines }]);
    return match && match.iou > 0 ? match.line : null;
}

/**
 * The model whose sidecar has the newest ``decoded_at``, among the sidecars a
 * 409 offered. Sidecars without a parseable time lose to any that have one.
 * @param {Object<string, object>|null} options - Sidecar per model.
 * @returns {string} The model, the first one when no time parses, or '' when there are none.
 */
export function newestModel(options) {
    let best = '';
    let bestTime = -Infinity;
    Object.keys(options || {}).forEach((model) => {
        const sidecar = options[model];
        const parsed = sidecar ? Date.parse(sidecar.decoded_at) : NaN;
        const time = Number.isNaN(parsed) ? -Infinity : parsed;
        if (!best || time > bestTime) {
            best = model;
            bestTime = time;
        }
    });
    return best;
}

/**
 * Warning text when the sidecar is not the read shown on the Read tab: another
 * model, or another decode (``decoded_at`` compared as instants).
 * @param {object|null} sidecar - Sidecar being shown.
 * @param {{vlm_model?: string, decoded_at?: string}|null|undefined} publicRead - The Read tab's read.
 * @returns {string} The warning, or '' when they match or there is no public read.
 */
export function publicReadMismatch(sidecar, publicRead) {
    if (!sidecar || !publicRead) return '';
    const otherModel = Boolean(publicRead.vlm_model) && sidecar.vlm_model !== publicRead.vlm_model;
    const otherDecode =
        Boolean(sidecar.decoded_at && publicRead.decoded_at) && !sameInstant(sidecar.decoded_at, publicRead.decoded_at);
    if (!otherModel && !otherDecode) return '';
    let what = 'decode';
    if (otherModel) what = otherDecode ? 'model and decode' : 'model';
    /**
     * One side of the comparison as text.
     * @param {{vlm_model?: string, decoded_at?: string}} read - Sidecar or public read.
     * @returns {string} "model, decoded <time>".
     */
    const describe = (read) =>
        [read.vlm_model || 'unknown model', read.decoded_at ? `decoded ${formatTime(read.decoded_at)}` : '']
            .filter(Boolean)
            .join(', ');
    const sides = `readings: ${describe(sidecar)}; Read tab: ${describe(publicRead)}`;
    return `These readings come from another ${what} than the Read tab (${sides}); line numbers may not match.`;
}

/**
 * Picker option text for one page image with readings.
 * @param {object} page - Grouped page from ``groupPages``.
 * @returns {string} E.g. "2 · columns · 6 readers".
 */
export function pageLabel(page) {
    const parts = [String(page.image_index + 1), page.policy_action || 'legacy'];
    if (Number.isFinite(page.n_readers)) parts.push(`${page.n_readers} readers`);
    if (Array.isArray(page.models) && page.models.length > 1) parts.push(`${page.models.length} models`);
    return parts.join(' · ');
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
