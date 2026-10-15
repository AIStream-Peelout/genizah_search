import {
    FINAL_KEY,
    PALETTE,
    alignLines,
    assignColours,
    bestMatches,
    buildEntries,
    charDiff,
    findCorrectionLine,
    groupPages,
    healthTone,
    iou,
    lineCountLabel,
    normaliseBox,
    normaliseLines,
    readerLabel,
    splitChars,
    unitSummary,
} from './readersModel';

/**
 * Rebuild one side of a diff from its segments.
 * @param {Array<{type: string, text: string}>} segments - charDiff output.
 * @param {string} side - 'a' keeps deletions, 'b' keeps insertions.
 * @returns {string} The reconstructed text.
 */
const rebuild = (segments, side) =>
    segments
        .filter((s) => s.type === 'same' || (side === 'a' ? s.type === 'del' : s.type === 'ins'))
        .map((s) => s.text)
        .join('');

describe('normaliseBox', () => {
    it('orders corners', () => {
        expect(normaliseBox([300, 400, 100, 200])).toEqual([100, 200, 300, 400]);
    });

    it('rejects malformed boxes', () => {
        expect(normaliseBox(null)).toBeNull();
        expect(normaliseBox([1, 2, 3])).toBeNull();
        expect(normaliseBox([1, null, 3, 4])).toBeNull();
        expect(normaliseBox([1, 'x', 3, 4])).toBeNull();
    });
});

describe('iou', () => {
    it('is 1 for identical boxes', () => {
        expect(iou([0, 0, 100, 100], [0, 0, 100, 100])).toBe(1);
    });

    it('is 0 for disjoint and edge-touching boxes', () => {
        expect(iou([0, 0, 100, 100], [200, 200, 300, 300])).toBe(0);
        expect(iou([0, 0, 100, 100], [100, 0, 200, 100])).toBe(0);
    });

    it('computes partial overlap', () => {
        // Intersection 50x100 = 5000; union 10000 + 10000 - 5000 = 15000.
        expect(iou([0, 0, 100, 100], [50, 0, 150, 100])).toBeCloseTo(1 / 3, 6);
    });

    it('handles containment', () => {
        expect(iou([0, 0, 100, 100], [25, 25, 75, 75])).toBeCloseTo(0.25, 6);
    });

    it('is 0 for zero-area and invalid boxes', () => {
        expect(iou([10, 10, 10, 50], [10, 10, 10, 50])).toBe(0);
        expect(iou(null, [0, 0, 1, 1])).toBe(0);
        expect(iou([0, 0, 1, 1], undefined)).toBe(0);
    });

    it('accepts reversed corners', () => {
        expect(iou([100, 100, 0, 0], [0, 0, 100, 100])).toBe(1);
    });
});

describe('normaliseLines', () => {
    it('reads bbox for final lines and keeps null', () => {
        const [line] = normaliseLines([{ index: 3, text: 'א', bbox: [1, 2, 3, 4] }]);
        expect(line.box).toEqual([1, 2, 3, 4]);
        expect(line.index).toBe(3);
        expect(normaliseLines(null)).toBeNull();
        expect(normaliseLines([])).toEqual([]);
    });
});

describe('readerLabel', () => {
    it('labels a whole-page VLM reader', () => {
        expect(readerLabel({ id: 'vlm:m:page', kind: 'vlm', scope: 'page' })).toBe('VLM · whole page');
    });

    it('labels a unit crop reader with its role', () => {
        expect(readerLabel({ kind: 'vlm', scope: 'unit', unit: 'col1.top', role: 'half' })).toBe(
            'VLM · unit col1.top · half'
        );
    });

    it('labels an unknown reader kind generically', () => {
        expect(readerLabel({ id: 'vote:3x:page', kind: 'vote', scope: 'page' })).toBe('Vote · whole page');
        expect(readerLabel({ kind: 'ensemble_vote', scope: 'region' })).toBe('Ensemble vote · region');
        expect(readerLabel({})).toBe('Reader');
    });
});

describe('assignColours and buildEntries', () => {
    it('gives distinct colours in order and cycles past the palette', () => {
        const readers = Array.from({ length: PALETTE.length + 2 }, (_, i) => ({ id: `r${i}` }));
        const colours = assignColours(readers);
        expect(colours.r0).toBe(PALETTE[0]);
        expect(colours.r1).toBe(PALETTE[1]);
        expect(colours[`r${PALETTE.length}`]).toBe(PALETTE[0]);
    });

    it('puts the final row first and keeps unattributed readers as null', () => {
        const entries = buildEntries({
            final: { source: 'assembled', lines: [{ index: 0, text: 'x', bbox: [0, 0, 10, 10] }] },
            readers: [
                { id: 'a', kind: 'vlm', scope: 'page', lines: null },
                { id: 'a', kind: 'htr', scope: 'page', lines: [] },
            ],
        });
        expect(entries.map((e) => e.key)).toEqual([FINAL_KEY, 'a', 'a#1']);
        expect(entries[0].label).toBe('Final (assembled)');
        expect(lineCountLabel(entries[1].lines)).toBe('unattributed');
        expect(lineCountLabel(entries[2].lines)).toBe('0 lines');
    });
});

describe('bestMatches', () => {
    it('picks the most overlapping line per reader', () => {
        const final = { index: 0, bbox: [0, 0, 100, 20] };
        const readers = [
            { id: 'r1', lines: [{ index: 0, text: 'far', box: [0, 500, 100, 520] }, { index: 1, text: 'near', box: [0, 2, 100, 22] }] },
            { id: 'r2', lines: [{ index: 0, text: 'none', box: [500, 500, 600, 520] }] },
            { id: 'r3', lines: null },
        ];
        const [m1, m2, m3] = bestMatches(final, readers);
        expect(m1.line.text).toBe('near');
        expect(m1.iou).toBeGreaterThan(0.8);
        expect(m2.line).toBeNull();
        expect(m2.iou).toBe(0);
        expect(m3.unattributed).toBe(true);
    });
});

describe('alignLines', () => {
    it('matches one-to-one by overlap and orders rows top to bottom', () => {
        const a = [
            { index: 0, text: 'a0', box: [0, 100, 100, 120] },
            { index: 1, text: 'a1', box: [0, 0, 100, 20] },
            { index: 2, text: 'a2', box: [0, 300, 100, 320] },
        ];
        const b = [
            { index: 0, text: 'b0', box: [0, 2, 100, 22] },
            { index: 1, text: 'b1', box: [0, 101, 100, 121] },
            { index: 2, text: 'b2', box: [0, 200, 100, 220] },
        ];
        const rows = alignLines(a, b);
        expect(rows.map((r) => [r.a && r.a.text, r.b && r.b.text])).toEqual([
            ['a1', 'b0'],
            ['a0', 'b1'],
            [null, 'b2'],
            ['a2', null],
        ]);
        expect(rows[0].iou).toBeGreaterThan(0.5);
        expect(rows[2].iou).toBe(0);
    });

    it('does not pair one line twice when two candidates overlap it', () => {
        const a = [{ index: 0, text: 'wide', box: [0, 0, 200, 20] }];
        const b = [
            { index: 0, text: 'left', box: [0, 0, 100, 20] },
            { index: 1, text: 'right', box: [100, 0, 200, 20] },
        ];
        const rows = alignLines(a, b);
        expect(rows.filter((r) => r.a && r.b)).toHaveLength(1);
        expect(rows).toHaveLength(2);
    });

    it('ignores pairs below the threshold', () => {
        const rows = alignLines([{ text: 'a', box: [0, 0, 100, 100] }], [{ text: 'b', box: [95, 95, 200, 200] }]);
        expect(rows).toHaveLength(2);
    });
});

describe('charDiff', () => {
    it('finds a deleted Hebrew letter', () => {
        expect(charDiff('שלום', 'שלם')).toEqual([
            { type: 'same', text: 'של' },
            { type: 'del', text: 'ו' },
            { type: 'same', text: 'ם' },
        ]);
    });

    it('rebuilds both sides for a ד/ר confusion', () => {
        const a = 'ברוך אתה ה׳ אלהינו';
        const b = 'בדוך אתה ה׳ אלהינו מלך';
        const segments = charDiff(a, b);
        expect(rebuild(segments, 'a')).toBe(a);
        expect(rebuild(segments, 'b')).toBe(b);
        expect(segments).toContainEqual({ type: 'del', text: 'ר' });
        expect(segments).toContainEqual({ type: 'ins', text: 'ד' });
    });

    it('keeps nikud on its letter', () => {
        expect(splitChars('שָׁלוֹם')).toHaveLength(4);
        const segments = charDiff('שָׁלוֹם', 'שָׁלם');
        expect(segments).toContainEqual({ type: 'del', text: 'וֹ' });
        expect(rebuild(segments, 'a')).toBe('שָׁלוֹם');
    });

    it('handles empty strings', () => {
        expect(charDiff('', '')).toEqual([]);
        expect(charDiff('אב', '')).toEqual([{ type: 'del', text: 'אב' }]);
        expect(charDiff('', 'אב')).toEqual([{ type: 'ins', text: 'אב' }]);
    });
});

describe('unitSummary', () => {
    it('lists attempts then the resolution', () => {
        expect(
            unitSummary({
                name: 'col1',
                resolved_by: 'share',
                attempts: [
                    { source: 'crop', label: 'loop' },
                    { source: 'halves', label: 'loop' },
                ],
            })
        ).toBe('col1: crop (loop) → halves (loop) → share');
    });

    it('copes with missing attempts and labels', () => {
        expect(unitSummary({ name: 'col0', resolved_by: 'crop', attempts: [] })).toBe('col0: crop');
        expect(unitSummary({ name: 'col0', attempts: [{ source: 'crop' }] })).toBe('col0: crop');
        expect(unitSummary({ name: 'col0' })).toBe('col0');
    });
});

describe('healthTone', () => {
    it('maps known labels and greys out unknown ones', () => {
        expect(healthTone('ok')).toBe('good');
        expect(healthTone('loop')).toBe('bad');
        expect(healthTone('parse_loss')).toBe('warn');
        expect(healthTone('something_new')).toBe('neutral');
        expect(healthTone(null)).toBe('neutral');
    });
});

describe('findCorrectionLine and groupPages', () => {
    const finalLines = normaliseLines([
        { index: 0, text: 'a', bbox: [0, 0, 100, 20] },
        { index: 1, text: 'b', bbox: [0, 40, 100, 60] },
    ]);

    it('finds a final-row correction by index, else by box', () => {
        expect(findCorrectionLine({ reader_id: FINAL_KEY, line_index: 1 }, finalLines).text).toBe('b');
        expect(findCorrectionLine({ reader_id: 'x', line_index: 9, bbox: [0, 1, 100, 21] }, finalLines).text).toBe('a');
        expect(findCorrectionLine({ reader_id: 'x', line_index: 9, bbox: [500, 500, 600, 600] }, finalLines)).toBeNull();
    });

    it('groups sidecars of several models under one image', () => {
        const grouped = groupPages([
            { image_index: 2, vlm_model: 'm1' },
            { image_index: 0, vlm_model: 'm1' },
            { image_index: 2, vlm_model: 'm2' },
        ]);
        expect(grouped.map((p) => p.image_index)).toEqual([0, 2]);
        expect(grouped[1].models).toEqual(['m1', 'm2']);
    });
});
