import React, { useEffect, useMemo, useState } from 'react';
import CompareReaders from './CompareReaders';
import CorrectionsList from './CorrectionsList';
import LineDetail from './LineDetail';
import ReaderList, { HealthBadge } from './ReaderList';
import ReadersStage from './ReadersStage';
import { fetchCorrections, fetchSidecar, postCorrection } from './readingsApi';
import { FINAL_KEY, buildEntries, formatTime, formatValue, shortHash, unitSummary } from './readersModel';
import './Readers.css';

/**
 * Header line of the panel: which decode this is, plus provenance as JSON.
 * When several models read the page, a select switches between them.
 * @param {object} props - Component props.
 * @param {object} props.sidecar - Sidecar being shown.
 * @param {string[]} props.models - Models with a sidecar for this page (409), else [].
 * @param {function(string): void} props.onModel - Switch model.
 */
function SidecarInfo({ sidecar, models, onModel }) {
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
                    <select value={sidecar.vlm_model || ''} onChange={(e) => onModel(e.target.value)}>
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
 * How the decoding policy handled this page: one line of facts, then each
 * unit's attempts and resolution. Legacy pages (policy null) show nothing.
 * @param {{policy: object}} props - ``sidecar.policy``.
 */
function PolicyStrip({ policy }) {
    const facts = [
        ['action', policy.action],
        ['reason', policy.reason],
        ['final source', policy.final_source],
        ['calls', policy.calls],
        ['fallback', policy.fallback],
        ['mode', policy.mode],
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
                    {units.map((unit, k) => (
                        <li key={unit.name || k}>
                            <code>{unitSummary(unit)}</code>
                            {unit.role && <span className="readers-muted"> {unit.role}</span>}{' '}
                            <HealthBadge label={unit.health_label} />
                        </li>
                    ))}
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
 * @param {string} [props.vlmModel] - Model of the public read on screen; picked when two models have sidecars.
 * @param {string} props.token - Advanced-view token.
 */
export default function ReadersPanel({ docId, imageIndex, vlmModel, token }) {
    const [load, setLoad] = useState({ status: 'loading' });
    const [chosenModel, setChosenModel] = useState('');
    const [corrections, setCorrections] = useState([]);
    const [correctionsState, setCorrectionsState] = useState({ loading: true, error: '' });
    const [visible, setVisible] = useState({ [FINAL_KEY]: true });
    const [pinnedIndex, setPinnedIndex] = useState(null);
    const [focusRequest, setFocusRequest] = useState(null);
    const [formOpen, setFormOpen] = useState(false);

    // 1. The sidecar of this page (409: one per model, chosen below).
    useEffect(() => {
        let active = true;
        setLoad({ status: 'loading' });
        fetchSidecar(docId, imageIndex, token)
            .then((result) => {
                if (active) setLoad(result);
            })
            .catch(() => {
                if (active) setLoad({ status: 'error', message: 'Could not reach the server.' });
            });
        return () => {
            active = false;
        };
    }, [docId, imageIndex, token]);

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

    // Several models: the visitor's choice, else the model of the public read.
    const options = load.status === 'choose' ? load.options : null;
    const models = options ? Object.keys(options) : [];
    const activeModel = options ? [chosenModel, vlmModel].find((m) => m && options[m]) || '' : '';
    let sidecar = null;
    if (load.status === 'ok') sidecar = load.sidecar;
    else if (activeModel) sidecar = options[activeModel];

    const entries = useMemo(() => (sidecar ? buildEntries(sidecar) : []), [sidecar]);
    const finalEntry = entries.find((entry) => entry.isFinal) || null;
    const finalLines = useMemo(() => (finalEntry ? finalEntry.lines : []), [finalEntry]);
    const pinnedLine = finalLines.find((line) => line.index === pinnedIndex) || null;

    // A different page or model drops the pin and closes the form.
    useEffect(() => {
        setPinnedIndex(null);
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
     * Save a suggested reading of the pinned line and add it to the list.
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
        };
        return postCorrection(docId, payload, token)
            .then((result) => {
                if (result.ok) {
                    setCorrections((list) => [result.correction, ...list]);
                    setFormOpen(false);
                }
                return result;
            })
            .catch(() => ({ ok: false, message: 'Could not reach the server; the suggestion was not saved.' }));
    };

    if (load.status === 'loading') {
        return <div className="readers-panel readers-state">Loading readings…</div>;
    }
    if (load.status === 'missing') {
        return (
            <div className="readers-panel readers-state">
                <strong>No readings sidecar for this page yet.</strong>
                {load.message && <div className="readers-muted">{load.message}</div>}
            </div>
        );
    }
    if (load.status === 'error') {
        return (
            <div className="readers-panel readers-state">
                <strong>Could not load the readings.</strong>
                <div className="readers-error">{load.message}</div>
            </div>
        );
    }
    if (!sidecar) {
        return (
            <div className="readers-panel readers-state">
                <strong>Several models read this page.</strong> Choose one:
                <div className="readers-model-choices">
                    {models.map((model) => (
                        <button key={model} type="button" onClick={() => setChosenModel(model)}>
                            {model}
                        </button>
                    ))}
                </div>
            </div>
        );
    }

    const policy = sidecar.policy || null;
    const units = policy && Array.isArray(policy.units) ? policy.units : [];

    return (
        <div className="readers-panel">
            <SidecarInfo sidecar={sidecar} models={models} onModel={setChosenModel} />
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
                    {pinnedLine ? (
                        <LineDetail
                            line={pinnedLine}
                            entries={entries}
                            formOpen={formOpen}
                            onFormOpen={setFormOpen}
                            onSubmitCorrection={submitCorrection}
                            onUnpin={() => setPinnedIndex(null)}
                        />
                    ) : (
                        <p className="readers-hint">
                            Click a final line on the image or in the list below to see every reader on it and to
                            suggest a reading.
                        </p>
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
