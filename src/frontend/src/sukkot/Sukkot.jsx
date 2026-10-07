import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { AGREEMENT_CAVEAT, READERS_DESCRIPTION } from '../read/caveat';
import data from './data/sukkot_5787.json';
import FestivalStrip from './FestivalStrip';
import ThemeSection, { themeAnchorId } from './ThemeSection';
import Refs from './Refs';
import { buildModel } from './model';
import { cardAnchorId } from './FragmentCard';
import './Sukkot.css';

const PAGE_TITLE = 'Exploring Sukkot through the Cairo Genizah';

/**
 * Whether the visitor asked the OS for reduced motion.
 * @returns {boolean} True when smooth scrolling should be avoided.
 */
const prefersReducedMotion = () =>
    typeof window !== 'undefined' && !!window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

/**
 * Scroll a fragment card into view and move focus to it.
 * @param {string} docId - Catalogue doc_id.
 * @returns {void}
 */
function scrollToCard(docId) {
    const el = document.getElementById(cardAnchorId(docId));
    if (!el) return;
    el.scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth', block: 'start' });
    el.focus({ preventScroll: true });
}

/**
 * The permanent festival page "Exploring Sukkot through the Cairo Genizah"
 * (route /sukkot; /sk redirects here). All content comes from one prebuilt
 * JSON; the page makes no API calls on load. ``?f=<doc_id>`` scrolls to and
 * highlights one card.
 * @param {{onOpenEsDocument: function(string): void}} props - Opens the site's catalogue-record modal.
 * @returns {JSX.Element} The page.
 */
export default function Sukkot({ onOpenEsDocument }) {
    const model = useMemo(() => buildModel(data), []);
    const [searchParams, setSearchParams] = useSearchParams();
    const focusId = searchParams.get('f');
    const [highlightId, setHighlightId] = useState(null);
    const caveatRef = useRef(null);

    useEffect(() => {
        document.title = PAGE_TITLE;
        return () => {
            document.title = 'Cairo Genizah AI';
        };
    }, []);

    useEffect(() => {
        if (!focusId || !model.onPage.has(focusId)) return;
        setHighlightId(focusId);
        scrollToCard(focusId);
    }, [focusId, model]);

    const selectFragment = useCallback(
        (docId) => {
            if (docId === focusId) {
                scrollToCard(docId);
            } else {
                setSearchParams({ f: docId }, { replace: true });
            }
        },
        [focusId, setSearchParams]
    );

    const openCaveat = () => {
        const el = caveatRef.current;
        if (!el) return;
        el.open = true;
        el.scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth', block: 'center' });
    };

    const stripFragments = model.fragments.filter((f) => model.onPage.has(f.doc_id));
    const generated = typeof data.generated_at === 'string' ? data.generated_at.slice(0, 10) : null;

    return (
        <div className="sk-page">
            <header className="sk-hero sk-text">
                <Link to="/" className="sk-back">← Cairo Genizah AI</Link>
                <h1>{PAGE_TITLE}</h1>
                <p className="sk-lead">
                    Medieval manuscript fragments that concern Sukkot: its prayers and poems, its laws, its calendar,
                    and the letters and accounts of the people who kept it.
                </p>
                {data.edition && <p className="sk-edition">{data.edition}</p>}
                <details className="sk-caveat" id="sk-caveat" ref={caveatRef}>
                    <summary>
                        Cards are drawn from catalogue records and published scholarship; machine readings, where
                        offered, are not checked by a person. <span className="sk-caveat-more">Full wording</span>
                    </summary>
                    <p>
                        Each card’s description comes from a catalogue record or a published study, named on the card.
                        Where a card offers a machine reading: {READERS_DESCRIPTION} {AGREEMENT_CAVEAT} The card says how
                        many lines the two readers agreed on; a card with no agreed lines offers no machine reading. Do
                        not cite machine text; use it to explore the fragment against the image.
                    </p>
                </details>
            </header>

            <FestivalStrip days={model.days} fragments={stripFragments} onSelect={selectFragment} />

            <nav className="sk-index sk-text" aria-labelledby="sk-index-h">
                <h2 id="sk-index-h">Themes</h2>
                <ol>
                    {model.themes.map((t) => (
                        <li key={t.id}>
                            <a href={`#${themeAnchorId(t.id)}`}>{t.title}</a>{' '}
                            <span className="sk-count">({(model.byTheme[t.id] || []).length})</span>
                        </li>
                    ))}
                </ol>
            </nav>

            {model.themes.map((t) => (
                <ThemeSection
                    key={t.id}
                    theme={t}
                    fragments={model.byTheme[t.id] || []}
                    genreLabel={model.genreLabels[t.genre] || null}
                    numbers={model.numbers}
                    highlightId={highlightId}
                    onOpenEsDocument={onOpenEsDocument}
                    clusters={model.clusters}
                    dayLabels={model.dayLabels}
                />
            ))}

            {model.order.length > 0 && <Refs order={model.order} refs={model.refs} />}

            <footer className="sk-credits sk-text">
                <h2>Credits</h2>
                <p>
                    Catalogue data: National Library of Israel (KTIV) and the Princeton Geniza Project. Images courtesy
                    of the holding libraries; OPenn items are public domain.
                </p>
                <p>
                    Machine readings, where a card offers one, are made automatically and have not been checked by a
                    person.{' '}
                    <button type="button" className="sk-linkbtn" onClick={openCaveat}>
                        Read the full wording
                    </button>
                </p>
                {generated && <p className="sk-small">Page data compiled {generated}.</p>}
            </footer>
        </div>
    );
}
