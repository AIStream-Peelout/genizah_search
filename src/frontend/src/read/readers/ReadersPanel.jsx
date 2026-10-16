import React, { useEffect, useMemo, useState } from 'react';
import CompareReaders from './CompareReaders';
import CorrectionsList from './CorrectionsList';
import LineDetail from './LineDetail';
import ReaderList, { HealthBadge } from './ReaderList';
import ReadersStage from './ReadersStage';
import { fetchCorrections, fetchReadingsPages, fetchSidecar, postCorrection } from './readingsApi';
import {
    FINAL_KEY,
    attemptDetail,
    buildEntries,
    formatTime,
    formatValue,
    groupPages,
    newestModel,
    pageLabel,
    publicReadMismatch,
    shortHash,
    unitContribution,
    unitSummary,
} from './readersModel';
import './Readers.css';

/**
 * Header line of the panel: which decode this is, plus provenance as JSON.
 * When several models read the page, a select switches between them.
 * @param {object} props - Component props.
 * @param {object} props.sidecar - Sidecar being shown.
 * @param {string[]} props.models - Models with a sidecar for this page (409), else [].
 * @param {string} props.activeModel - Model picked among ``models``, or ''.
 * @param {function(string): void} props.onModel - Switch model.
 */
function SidecarInfo({ sidecar, models, activeModel, onModel }) {
    const final = sidecar.final || {};
    return (
        <div className="readers-info">
            <span>
                Model <code>{sidecar.vlm_model || '—'}</code>
                {sidecar.vlm_revision && <> rev <code>{sidecar.vlm_revision}</code></>}
            </span>
            {models.length > 1 && (
                <label>
                    Switch model{' '}
                    <select value={activeModel || sidecar.vlm_model || ''} onChange={(e) => onModel(e.target.value)}>
                        {models.map((model) => (
                            <option key={model} value={model}>{model}</option>
                        ))}
                    </select>
                </label>
            )}
            {sidecar.decoded_at && <span>Decoded {formatTime(sidecar.decoded_at)}</span>}
            {sidecar.rule_version && <span>Rule <code>{sidecar.rule_version}</code></span>}
            {sidecar.image_sha256 && (
                <span title={sidecar.image_sha256}>
                    Image <code>{shortHash(sidecar.image_sha256)}</code>
                </span>
            )}
            {final.source && <span>Final <code>{final.source}</code></span>}
            {sidecar.provenance && (
                <details className="readers-provenance">
                    <summary>Provenance</summary>
                    <pre dir="ltr">{JSON.stringify(sidecar.provenance, null, 2)}</pre>
                </details>
            )}
        </div>
    );
}

/**
 * Warnings under the header: a refused suggestion (the page was re-decoded),
 * readings that are not the read on the Read tab, and a partial reply mapping
 * with the builder's notes.
 * @param {object} props - Component props.
 * @param {object} props.sidecar - Sidecar being shown.
 * @param {{vlm_model?: string, decoded_at?: string}} [props.publicRead] - The Read tab's read.
 * @param {string} props.notice - Message of a refused suggestion, or ''.
 * @param {function(): void} props.onDismissNotice - Hide the notice.
 */
function PanelWarnings({ sidecar, publicRead, notice, onDismissNotice }) {
    const provenance = sidecar.provenance || {};
    const notes = Array.isArray(provenance.notes) ? provenance.notes.filter(Boolean) : [];
    const partial = provenance.reply_mapping === 'partial';
    const mismatch = publicReadMismatch(sidecar, publicRead);
    if (!notice && !mismatch && !partial) return null;
    return (
        <div className="readers-warnings">
            {notice && (
                <div className="readers-warning" role="status">
                    <span dir="auto">{notice}</span>{' '}
                    <button type="button" className="readers-link-button" onClick={onDismissNotice}>
                        dismiss
                    </button>
                </div>
            )}
            {mismatch && <div className="readers-warning">{mismatch}</div>}
            {partial && (
                <div className="readers-warning">
                    Partial reply mapping: some unit reads could not be attributed.
                    {notes.length > 0 && (
                        <ul>
                            {notes.map((note, k) => (
                                <li key={k} dir="auto">{formatValue(note)}</li>
                            ))}
                        </ul>
                    )}
                </div>
            )}
        </div>
    );
}

/**
 * Image picker over the pages that have readings, for the Read page's Readers
 * tab. Shown when there is more than one such page, or when the current image
 * is not among them.
 * @param {object} props - Component props.
 * @param {Array<object>|null} props.pages - ``pages`` from /readings/pages; null while unknown.
 * @param {number} props.imageIndex - Current image.
 * @param {function(number): void} [props.onImageChange] - Switch image; no picker without it.
 */
function PagePicker({ pages, imageIndex, onImageChange }) {
    const choices = useMemo(() => groupPages(pages || []), [pages]);
    if (!onImageChange || pages === null) return null;
    const current = Number(imageIndex);
    const listed = choices.some((page) => page.image_index === current);
    if (choices.length === 0 || (choices.length === 1 && listed)) return null;
    return (
        <label className="readers-page-picker">
            Pages with readings{' '}
            <select value={String(current)} onChange={(e) => onImageChange(Number(e.target.value))}>
                {!listed && <option value={String(current)}>{`${current + 1} · no readings`}</option>}
                {choices.map((page) => (
                    <option key={page.image_index} value={String(page.image_index)}>
                        {pageLabel(page)}
                    </option>
                ))}
            </select>
        </label>
    );
}

/**
 * How the decoding policy handled this page: one line of facts, then each
 * unit's attempts and resolution, with every attempt's evidence one click
 * away. Legacy pages (policy null) show nothing.
 * @param {{policy: object}} props - ``sidecar.policy``.
 */
function PolicyStrip({ policy }) {
    const facts = [
        ['version', policy.version],
        ['action', policy.action],
        ['reason', policy.reason],
        ['final source', policy.final_source],
        ['calls', policy.calls],
        ['fallback', policy.fallback],
        ['mode', policy.mode],
        ['decided', formatTime(policy.decided_at)],
    ].filter(([, value]) => value !== undefined && value !== null && value !== '');
    const units = Array.isArray(policy.units) ? policy.units : [];
    return (
        <div className="readers-policy">
            <div className="readers-policy-facts">
                <strong>Policy</strong>
                {facts.map(([name, value]) => (
                    <span key={name} className="readers-policy-fact">
                        <span className="readers-muted">{name}</span> {formatValue(value)}
                    </span>
                ))}
            </div>
            {units.length > 0 && (
                <ul className="readers-policy-units">
                    {units.map((unit, k) => {
                        const attempts = Array.isArray(unit.attempts) ? unit.attempts : [];
                        const contribution = unitContribution(unit, policy);
                        return (
                            <li key={unit.name || k}>
                                <details className="readers-unit-details">
                                    <summary>
                                        <code>{unitSummary(unit)}</code>
                                        {unit.role && <span className="readers-muted"> {unit.role}</span>}{' '}
                                        <HealthBadge label={unit.health_label} />
                                    </summary>
                                    {attempts.length > 0 ? (
                                        <ol className="readers-attempts">
                                            {attempts.map((attempt, j) => (
                                                <li key={j}>
                                                    <code>{attemptDetail(attempt)}</code>
                                                    {attempt && attempt.note && (
                                                        <div className="readers-muted" dir="auto">{attempt.note}</div>
                                                    )}
                                                </li>
                                            ))}
                                        </ol>
                                    ) : (
                                        <div className="readers-muted readers-unit-contribution">
                                            No attempts recorded.
                                        </div>
                                    )}
                                    {contribution && <div className="readers-unit-contribution">{contribution}</div>}
                                </details>
                            </li>
                        );
                    })}
                </ul>
            )}
        </div>
    );
}

/**
 * The final lines as a clickable list (pinning works without hitting a box).
 * @param {object} props - Component props.
 * @param {Array<object>} props.lines - Normalised final lines.
 * @param {number|null} props.pinnedIndex - Pinned line index.
 * @param {function(number): void} props.onPin - Pin (and centre) a line.
 */
function FinalLines({ lines, pinnedIndex, onPin }) {
    return (
        <details className="readers-section readers-final" open>
            <summary>Final lines ({lines.length})</summary>
            <ol className="readers-final-list">
                {lines.map((line) => (
                    <li
                        key={line.index}
                        className={`readers-final-line ${line.status || ''}${line.index === pinnedIndex ? ' pinned' : ''}`}
                    >
                        <button type="button" onClick={() => onPin(line.index)}>
                            <span className="readers-line-no">{line.index + 1}</span>
                            <span className="readers-hebrew" dir="auto">{line.text || '∅'}</span>
                        </button>
                    </li>
                ))}
            </ol>
        </details>
    );
}

/**
 * Advanced Readers panel (lazy chunk): every reader's lines over the page
 * image, the decoding policy, line-by-line comparison and suggested readings.
 * Data comes from the page's readings-v1 sidecar.
 * @param {object} props - Component props.
 * @param {string} props.docId - Document id.
 * @param {number} props.imageIndex - 0-based page image index.
 * @param {string} [props.vlmModel] - Model of the public read on screen; picked when two models
 *     have sidecars (else the newest decode is shown first).
 * @param {string} props.token - Advanced-view token.
 * @param {{vlm_model?: string, decoded_at?: string}} [props.publicRead] - The Read tab's read;
 *     the panel warns when the sidecar comes from another model or decode.
 * @param {function(number): void} [props.onImageChange] - Switch image; when given, the panel
 *     shows a picker over the pages that have readings.
 */
export default function ReadersPanel({ docId, imageIndex, vlmModel, token, publicRead, onImageChange }) {
    const [load, setLoad] = useState({ status: 'loading' });
    const [reloadKey, setReloadKey] = useState(0);
    const [chosenModel, setChosenModel] = useState('');
    const [corrections, setCorrections] = useState([]);
    const [correctionsState, setCorrectionsState] = useState({ loading: true, error: '' });
    const [pages, setPages] = useState(null);
    const [visible, setVisible] = useState({ [FINAL_KEY]: true });
    const [pinnedIndex, setPinnedIndex] = useState(null);
    const [focusRequest, setFocusRequest] = useState(null);
    const [formOpen, setFormOpen] = useState(false);
    const [notice, setNotice] = useState('');
    const wantsPages = Boolean(onImageChange);

    // 1. The sidecar of this page (409: one per model, chosen below). After the
    // server refused a suggestion as made on an old decode, skip the browser cache.
    useEffect(() => {
        let active = true;
        setLoad({ status: 'loading' });
        fetchSidecar(docId, imageIndex, token, { fresh: reloadKey > 0 })
            .then((result) => {
                if (active) setLoad(result);
            })
            .catch(() => {
                if (active) setLoad({ status: 'error', message: 'Could not reach the server.' });
            });
        return () => {
            active = false;
        };
    }, [docId, imageIndex, token, reloadKey]);

    // 2. Suggested readings of this page.
    useEffect(() => {
        let active = true;
        setCorrections([]);
        setCorrectionsState({ loading: true, error: '' });
        fetchCorrections(docId, imageIndex, token)
            .then((list) => {
                if (!active) return;
                setCorrections(list);
                setCorrectionsState({ loading: false, error: '' });
            })
            .catch((err) => {
                if (active) setCorrectionsState({ loading: false, error: err.message || 'Could not load suggestions.' });
            });
        return () => {
            active = false;
        };
    }, [docId, imageIndex, token]);

    // 3. Pages with readings, for the image picker (only when the host can switch images).
    useEffect(() => {
        if (!wantsPages) return undefined;
        let active = true;
        setPages(null);
        fetchReadingsPages(docId, token)
            .then((list) => {
                if (active) setPages(list);
            })
            .catch(() => {
                // No picker then; the host's own image picker still works.
                if (active) setPages(null);
            });
        return () => {
            active = false;
        };
    }, [docId, token, wantsPages]);

    // A refused-suggestion notice belongs to the page it was made on.
    useEffect(() => {
        setNotice('');
    }, [docId, imageIndex]);

    // Several models: the visitor's choice, else the model of the public read,
    // else the newest decode (the "Switch model" select offers the others).
    const options = load.status === 'choose' ? load.options : null;
    const models = options ? Object.keys(options) : [];
    const activeModel = options ? [chosenModel, vlmModel].find((m) => m && options[m]) || newestModel(options) : '';
    let sidecar = null;
    if (load.status === 'ok') sidecar = load.sidecar;
    else if (activeModel) sidecar = options[activeModel];

    const entries = useMemo(() => (sidecar ? buildEntries(sidecar) : []), [sidecar]);
    const finalEntry = entries.find((entry) => entry.isFinal) || null;
    const finalLines = useMemo(() => (finalEntry ? finalEntry.lines : []), [finalEntry]);
    const pinnedLine = finalLines.find((line) => line.index === pinnedIndex) || null;

    // A different page or model drops the pin, the pending focus and closes the form.
    useEffect(() => {
        setPinnedIndex(null);
        setFocusRequest(null);
        setFormOpen(false);
    }, [sidecar]);

    /**
     * Pin a final line; pinning another line closes an open suggestion form.
     * @param {number} index - Final line index.
     */
    const pinLine = (index) => {
        if (index !== pinnedIndex) setFormOpen(false);
        setPinnedIndex(index);
    };

    /**
     * Pin a line from a list and centre it on the image.
     * @param {number} index - Final line index.
     */
    const pinAndFocus = (index) => {
        pinLine(index);
        const line = finalLines.find((l) => l.index === index);
        if (line && line.box) setFocusRequest({ box: line.box, key: Date.now() });
    };

    const toggleEntry = (key, on) => setVisible((v) => ({ ...v, [key]: on }));
    const toggleAll = (on) =>
        setVisible(entries.reduce((acc, entry) => ({ ...acc, [entry.key]: on }), {}));

    /**
     * Save a suggested reading of the pinned line and add it to the list. When
     * the server answers that the page was re-decoded (409), show its message,
     * drop the pin and reload the sidecar past the browser cache.
     * @param {{correctedText: string, note: string, contributor: string}} form - Form values.
     * @returns {Promise<{ok: boolean, message?: string}>} Result for the form.
     */
    const submitCorrection = (form) => {
        const payload = {
            image_index: Number(imageIndex),
            line_index: pinnedLine.index,
            reader_id: FINAL_KEY,
            bbox: pinnedLine.bbox || pinnedLine.box,
            original_text: pinnedLine.text,
            corrected_text: form.correctedText,
            note: form.note,
            contributor: form.contributor,
            vlm_model: sidecar.vlm_model,
            decoded_at: sidecar.decoded_at,
        };
        return postCorrection(docId, payload, token)
            .then((result) => {
                if (result.ok) {
                    setCorrections((list) => [result.correction, ...list]);
                    setFormOpen(false);
                } else if (result.stale) {
                    setNotice(result.message);
                    setPinnedIndex(null);
                    setFocusRequest(null);
                    setFormOpen(false);
                    setReloadKey((k) => k + 1);
                }
                return result;
            })
            .catch(() => ({ ok: false, message: 'Could not reach the server; the suggestion was not saved.' }));
    };

    const picker = <PagePicker pages={pages} imageIndex={imageIndex} onImageChange={onImageChange} />;

    /**
     * A panel that shows only a state (loading, missing, error, choose), with
     * the refused-suggestion notice and the image picker.
     * @param {React.ReactNode} children - The state message.
     * @returns {JSX.Element} The panel.
     */
    const statePanel = (children) => (
        <div className="readers-panel readers-state">
            {children}
            {notice && <div className="readers-warning" dir="auto">{notice}</div>}
            {picker}
        </div>
    );

    if (load.status === 'loading') {
        return statePanel('Loading readings…');
    }
    if (load.status === 'missing') {
        return statePanel(
            <>
                <strong>No readings sidecar for this page yet.</strong>
                {load.message && <div className="readers-muted">{load.message}</div>}
            </>
        );
    }
    if (load.status === 'error') {
        return statePanel(
            <>
                <strong>Could not load the readings.</strong>
                <div className="readers-error">{load.message}</div>
            </>
        );
    }
    if (!sidecar) {
        // Only reachable when a 409 listed no sidecar at all.
        return statePanel(
            <>
                <strong>Several models read this page.</strong> Choose one:
                <div className="readers-model-choices">
                    {models.map((model) => (
                        <button key={model} type="button" onClick={() => setChosenModel(model)}>
                            {model}
                        </button>
                    ))}
                </div>
            </>
        );
    }

    const policy = sidecar.policy || null;
    const units = policy && Array.isArray(policy.units) ? policy.units : [];

    return (
        <div className="readers-panel">
            {picker}
            <SidecarInfo sidecar={sidecar} models={models} activeModel={activeModel} onModel={setChosenModel} />
            <PanelWarnings
                sidecar={sidecar}
                publicRead={publicRead}
                notice={notice}
                onDismissNotice={() => setNotice('')}
            />
            <div className="readers-grid">
                <div className="readers-col readers-col-image">
                    {policy && <PolicyStrip policy={policy} />}
                    <ReadersStage
                        imageUrl={sidecar.image_url}
                        altText={`Fragment ${docId}, image ${Number(imageIndex) + 1}`}
                        imageWidth={sidecar.image_width}
                        imageHeight={sidecar.image_height}
                        entries={entries}
                        visible={visible}
                        finalEntry={finalEntry}
                        pinnedLine={pinnedLine}
                        units={units}
                        focusRequest={focusRequest}
                        onPin={pinLine}
                    />
                </div>
                <div className="readers-col readers-col-middle">
                    <ReaderList entries={entries} visible={visible} onToggle={toggleEntry} onToggleAll={toggleAll} />
                    {pinnedLine && (
                        <LineDetail
                            line={pinnedLine}
                            entries={entries}
                            formOpen={formOpen}
                            onFormOpen={setFormOpen}
                            onSubmitCorrection={submitCorrection}
                            onUnpin={() => setPinnedIndex(null)}
                        />
                    )}
                    {!pinnedLine && finalLines.length > 0 && (
                        <p className="readers-hint">
                            Click a final line on the image or in the list below to see every reader on it and to
                            suggest a reading.
                        </p>
                    )}
                    {!pinnedLine && finalLines.length === 0 && (
                        <p className="readers-hint">This page has no final lines; nothing to pin.</p>
                    )}
                    {finalLines.length > 0 && (
                        <FinalLines lines={finalLines} pinnedIndex={pinnedIndex} onPin={pinAndFocus} />
                    )}
                    <CompareReaders entries={entries} />
                </div>
                <div className="readers-col readers-col-corrections">
                    <CorrectionsList
                        corrections={corrections}
                        loading={correctionsState.loading}
                        error={correctionsState.error}
                        finalLines={finalLines}
                        sidecar={sidecar}
                        onPin={pinAndFocus}
                    />
                </div>
            </div>
        </div>
    );
}
