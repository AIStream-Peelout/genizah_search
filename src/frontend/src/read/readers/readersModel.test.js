import realSidecar from './__fixtures__/C_131__2.json';
import {
    DASHES,
    FINAL_KEY,
    PALETTE,
    alignLines,
    assignColours,
    attemptDetail,
    bestMatches,
    buildEntries,
    charDiff,
    findCorrectionLine,
    groupPages,
    healthTone,
    iou,
    isDegenerate,
    lineCountLabel,
    newestModel,
    noLinesReason,
    normaliseBox,
    normaliseLines,
    paddedBox,
    pageLabel,
    pickLineAt,
    publicReadMismatch,
    readerLabel,
    sameInstant,
    splitChars,
    swatchBackground,
    unitContribution,
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

    it('pads zero-area boxes so identical edge lines still match, and is 0 for invalid boxes', () => {
        expect(iou([10, 10, 10, 50], [10, 10, 10, 50])).toBe(1);
        expect(iou([481, 1000, 730, 1000], [481, 1000, 730, 1000])).toBe(1);
        expect(iou([481, 1000, 730, 1000], [100, 100, 200, 200])).toBe(0);
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
    it('gives distinct colours in order and cycles past the palette with a dash per cycle', () => {
        const readers = Array.from({ length: PALETTE.length * 2 + 2 }, (_, i) => ({ id: `r${i}` }));
        const styles = assignColours(readers);
        expect(styles.r0).toEqual({ colour: PALETTE[0], dash: null });
        expect(styles.r1).toEqual({ colour: PALETTE[1], dash: null });
        expect(styles[`r${PALETTE.length - 1}`].dash).toBeNull();
        expect(styles[`r${PALETTE.length}`]).toEqual({ colour: PALETTE[0], dash: DASHES[1] });
        expect(styles[`r${PALETTE.length * 2}`]).toEqual({ colour: PALETTE[0], dash: DASHES[2] });
        expect(DASHES[1]).not.toBeNull();
        expect(DASHES[2]).not.toBe(DASHES[1]);
    });

    it('stores the dash on entries past the tenth reader and stripes their swatch', () => {
        const readers = Array.from({ length: PALETTE.length + 1 }, (_, i) => ({ id: `r${i}`, kind: 'vlm', lines: [] }));
        const entries = buildEntries({ final: { source: 'whole_page', lines: [] }, readers });
        expect(entries).toHaveLength(PALETTE.length + 2);
        expect(entries[0].dash).toBeNull();
        expect(entries[1].dash).toBeNull();
        expect(swatchBackground(entries[1])).toBe(PALETTE[0]);
        const eleventh = entries[PALETTE.length + 1];
        expect(eleventh.colour).toBe(PALETTE[0]);
        expect(eleventh.dash).toBe(DASHES[1]);
        expect(swatchBackground(eleventh)).toContain('repeating-linear-gradient');
        expect(swatchBackground(eleventh)).toContain(PALETTE[0]);
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

    it('carries the parsed flag and the note of each reader', () => {
        const entries = buildEntries({
            final: { source: 'assembled_partial', lines: [] },
            readers: [
                { id: 'p', kind: 'vlm', scope: 'page', parsed: true, lines: [] },
                { id: 'u', kind: 'vlm', scope: 'unit', unit: 'col1', lines: null, note: 'reply 3 does not match request col1' },
                { id: 'x', kind: 'vlm', scope: 'page', parsed: false, lines: [] },
            ],
        });
        expect(entries.map((e) => e.parsed)).toEqual([null, true, null, false]);
        expect(entries[2].note).toBe('reply 3 does not match request col1');
        expect(noLinesReason(entries[0])).toBe('found no lines');
        expect(noLinesReason(entries[1])).toBe('found no lines');
        expect(noLinesReason(entries[2])).toBe('reply 3 does not match request col1');
        expect(noLinesReason(entries[3])).toBe('reply did not parse');
        expect(noLinesReason({ lines: null, parsed: null, note: null })).toBe('reply not attributed');
        expect(noLinesReason({ lines: [{ index: 0 }], parsed: false })).toBe('');
    });
});

describe('lineCountLabel', () => {
    it('tells an unparsed reply from an unattributed one and from an empty one', () => {
        expect(lineCountLabel([], false)).toBe('reply did not parse');
        expect(lineCountLabel(null, false)).toBe('reply did not parse');
        expect(lineCountLabel(null)).toBe('unattributed');
        expect(lineCountLabel(null, true)).toBe('unattributed');
        expect(lineCountLabel([])).toBe('0 lines');
        expect(lineCountLabel([], true)).toBe('0 lines');
        expect(lineCountLabel([{}], false)).toBe('1 line');
        expect(lineCountLabel([{}, {}])).toBe('2 lines');
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
    // The contract's example unit: the share attempt is in the list and resolves it.
    const contractUnit = {
        name: 'col1',
        role: 'column',
        resolved_by: 'share',
        resolved: true,
        attempts: [
            { source: 'crop', label: 'loop', flags: ['loop'], letters: 175, n_lines: 30, reply_index: 3 },
            {
                source: 'halves',
                label: 'loop',
                flags: ['loop', 'capped'],
                letters: 1822,
                n_lines: 98,
                reply_index: null,
                reply_indices: [4, 5],
                note: 'joined from the replies in reply_indices (top half, bottom half)',
            },
            { source: 'share', label: 'ok', flags: [], letters: 540, n_lines: 22, reply_index: 6 },
        ],
    };

    it('lists the attempts once, then the resolution as a tag', () => {
        const summary = unitSummary(contractUnit);
        expect(summary).toBe('col1: crop (loop) → halves (loop) → share (ok) · resolved by share');
        expect(summary.match(/share/g)).toHaveLength(2);
    });

    it('does not read a single resolving attempt as two tries', () => {
        expect(
            unitSummary({ name: 'top', role: 'half', resolved_by: 'crop', resolved: true, attempts: [{ source: 'crop', label: 'ok' }] })
        ).toBe('top: crop (ok) · resolved by crop');
    });

    it('marks an unresolved unit', () => {
        expect(
            unitSummary({
                name: 'col1',
                resolved_by: 'unresolved',
                resolved: false,
                attempts: [
                    { source: 'crop', label: 'loop' },
                    { source: 'halves', label: 'loop' },
                    { source: 'share', label: 'loop' },
                ],
            })
        ).toBe('col1: crop (loop) → halves (loop) → share (loop) · unresolved');
    });

    it('copes with missing attempts and labels', () => {
        expect(unitSummary({ name: 'col0', resolved_by: 'crop', attempts: [] })).toBe('col0 · resolved by crop');
        expect(unitSummary({ name: 'col0', attempts: [{ source: 'crop' }] })).toBe('col0: crop');
        expect(unitSummary({ name: 'col0' })).toBe('col0');
    });

    it('details each attempt with its evidence', () => {
        expect(attemptDetail(contractUnit.attempts[0])).toBe('crop · loop · flags loop · 175 letters · 30 lines · reply 3');
        expect(attemptDetail(contractUnit.attempts[1])).toBe(
            'halves · loop · flags loop, capped · 1822 letters · 98 lines · replies 4, 5'
        );
        expect(attemptDetail(contractUnit.attempts[2])).toBe('share · ok · no flags · 540 letters · 22 lines · reply 6');
        expect(attemptDetail(null)).toBe('');
    });

    it('says what an unresolved unit gave the final lines', () => {
        const unresolved = {
            name: 'col0',
            resolved_by: 'unresolved',
            resolved: false,
            attempts: [{ source: 'crop' }, { source: 'halves' }],
        };
        expect(unitContribution(unresolved, { final_source: 'assembled_partial', fallback: null })).toContain('halves');
        expect(
            unitContribution({ ...unresolved, attempts: [{ source: 'crop' }, { source: 'share' }] }, { final_source: 'assembled_partial' })
        ).toContain('crop');
        expect(
            unitContribution(unresolved, { final_source: 'whole_page', fallback: 'all_units_unresolved' })
        ).toContain('nothing');
        expect(unitContribution(contractUnit, { final_source: 'assembled_partial' })).toBe('');
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
    const sidecar = {
        vlm_model: 'm1',
        decoded_at: '2026-10-13T05:30:25+00:00',
        image_sha256: 'hash-1',
    };
    // Made on final line 1 of this read, but its box overlaps line 0.
    const sameRead = {
        reader_id: FINAL_KEY,
        line_index: 1,
        bbox: [0, 1, 100, 21],
        vlm_model: 'm1',
        decoded_at: '2026-10-13T05:30:25Z',
        page_hash: 'hash-1',
    };

    it('finds a correction made on this read by index', () => {
        expect(findCorrectionLine(sameRead, finalLines, sidecar).text).toBe('b');
        expect(findCorrectionLine({ ...sameRead, page_hash: null }, finalLines, sidecar).text).toBe('b');
    });

    it('matches a correction from another model or decode by box, not by index', () => {
        expect(findCorrectionLine({ ...sameRead, vlm_model: 'm2' }, finalLines, sidecar).text).toBe('a');
        expect(findCorrectionLine({ ...sameRead, decoded_at: '2026-09-01T00:00:00Z' }, finalLines, sidecar).text).toBe('a');
        expect(findCorrectionLine({ ...sameRead, decoded_at: null }, finalLines, sidecar).text).toBe('a');
        expect(findCorrectionLine(sameRead, finalLines).text).toBe('a');
    });

    it('matches nothing for a correction made on another image file', () => {
        expect(findCorrectionLine({ ...sameRead, page_hash: 'hash-2' }, finalLines, sidecar)).toBeNull();
    });

    it('matches other readers by box and gives up without overlap', () => {
        expect(findCorrectionLine({ reader_id: 'x', line_index: 9, bbox: [0, 41, 100, 61] }, finalLines, sidecar).text).toBe('b');
        expect(
            findCorrectionLine({ reader_id: 'x', line_index: 9, bbox: [500, 500, 600, 600] }, finalLines, sidecar)
        ).toBeNull();
    });

    it('compares timestamps as instants', () => {
        expect(sameInstant('2026-10-13T05:30:25+00:00', '2026-10-13T05:30:25Z')).toBe(true);
        expect(sameInstant('2026-10-13T05:30:25+00:00', '2026-10-13T05:30:26Z')).toBe(false);
        expect(sameInstant(null, null)).toBe(false);
        expect(sameInstant('not a date', 'not a date')).toBe(true);
    });

    it('defaults to the newest decode among several models', () => {
        expect(
            newestModel({
                old: { decoded_at: '2026-09-28T20:52:04+00:00' },
                new: { decoded_at: '2026-10-13T05:30:25+00:00' },
                none: { decoded_at: null },
            })
        ).toBe('new');
        expect(newestModel({ a: {}, b: {} })).toBe('a');
        expect(newestModel({})).toBe('');
    });

    it('warns when the readings are not the Read tab read', () => {
        expect(publicReadMismatch(sidecar, { vlm_model: 'm1', decoded_at: '2026-10-13T05:30:25Z' })).toBe('');
        expect(publicReadMismatch(sidecar, undefined)).toBe('');
        expect(publicReadMismatch(sidecar, { vlm_model: 'm2', decoded_at: sidecar.decoded_at })).toContain(
            'another model than'
        );
        expect(publicReadMismatch(sidecar, { vlm_model: 'm1', decoded_at: '2026-10-14T00:00:00Z' })).toContain(
            'another decode than'
        );
    });

    it('groups sidecars of several models under one image', () => {
        const grouped = groupPages([
            { image_index: 2, vlm_model: 'm1' },
            { image_index: 0, vlm_model: 'm1' },
            { image_index: 2, vlm_model: 'm2' },
        ]);
        expect(grouped.map((p) => p.image_index)).toEqual([0, 2]);
        expect(grouped[1].models).toEqual(['m1', 'm2']);
        expect(pageLabel({ ...grouped[1], policy_action: 'columns', n_readers: 6 })).toBe('3 · columns · 6 readers · 2 models');
        expect(pageLabel(grouped[0])).toBe('1 · legacy');
    });
});

describe('a real sidecar (Manchester_JRL_C_131 image 2: columns page that fell back)', () => {
    const entries = buildEntries(realSidecar);

    it('lists the final row first, then the 8 readers in sidecar order', () => {
        expect(realSidecar.readers).toHaveLength(8);
        expect(entries).toHaveLength(9);
        expect(entries[0].isFinal).toBe(true);
        expect(entries[0].label).toBe('Final (whole page)');
        expect(entries.slice(1).map((e) => e.id)).toEqual(realSidecar.readers.map((r) => r.id));
        expect(new Set(entries.map((e) => e.key)).size).toBe(9);
    });

    it('labels the page reader as a whole-page VLM reader', () => {
        const page = entries.find((e) => e.reader && e.reader.role === 'page');
        expect(page.label).toBe('VLM · whole page');
        expect(page.parsed).toBe(true);
    });

    it('gives the Kraken reader no verdict', () => {
        const kraken = entries.find((e) => e.reader && e.reader.kind === 'htr');
        expect(kraken.label).toBe('HTR · whole page');
        expect(healthTone(kraken.health)).toBe('neutral');
    });

    it('summarises both column units as unresolved', () => {
        const { policy } = realSidecar;
        expect(policy.units).toHaveLength(2);
        policy.units.forEach((unit) => {
            expect(unitSummary(unit).endsWith(' · unresolved')).toBe(true);
            expect(unitContribution(unit, policy)).toContain('nothing');
        });
        expect(unitSummary(policy.units[1])).toBe('col1: crop (loop) → halves (loop) → share (loop) · unresolved');
    });

    it('matches a suggestion on this read by index', () => {
        const finalLines = entries[0].lines;
        const correction = {
            reader_id: FINAL_KEY,
            line_index: 2,
            bbox: finalLines[0].box,
            vlm_model: realSidecar.vlm_model,
            decoded_at: realSidecar.decoded_at,
            page_hash: realSidecar.image_sha256,
        };
        expect(findCorrectionLine(correction, finalLines, realSidecar).index).toBe(2);
    });
});

describe('zero-area boxes (lines clamped to the page edge)', () => {
    it('recognises and pads them without moving real boxes', () => {
        expect(isDegenerate([481, 1000, 730, 1000])).toBe(true);
        expect(isDegenerate([10, 10, 10, 50])).toBe(true);
        expect(isDegenerate([10, 10, 20, 50])).toBe(false);
        expect(isDegenerate(null)).toBe(false);
        expect(paddedBox([481, 1000, 730, 1000])).toEqual([481, 996, 730, 1000]);
        expect(paddedBox([0, 0, 0, 0])).toEqual([0, 0, 4, 4]);
        expect(paddedBox([10, 10, 20, 50])).toEqual([10, 10, 20, 50]);
    });

    it('can still be picked from the image', () => {
        const lines = normaliseLines([
            { index: 0, text: 'a', bbox: [100, 100, 900, 150] },
            { index: 1, text: 'b', bbox: [481, 1000, 730, 1000] },
        ]);
        expect(pickLineAt(lines, 600, 998).index).toBe(1);
        expect(pickLineAt(lines, 600, 120).index).toBe(0);
    });
});

