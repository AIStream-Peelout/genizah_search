import React, { useState, useEffect, useRef, useCallback } from 'react';

// localStorage key marking that the user has seen (or skipped) the tour.
export const TOUR_SEEN_KEY = 'genizah_tour_seen';

// Below this width the step card renders as a bottom sheet instead of being
// anchored beside the highlighted element.
const PHONE_BREAKPOINT = 640;
const SPOTLIGHT_PADDING = 8;
const CARD_MARGIN = 12;
const CARD_WIDTH = 360;

/**
 * Ordered tour steps. `selectors` are tried in order and the first visible
 * match becomes the highlighted element; steps with no selectors render as a
 * centered card over a dimmed page.
 */
const TOUR_STEPS = [
  {
    id: 'welcome',
    title: 'Welcome to Cairo Genizah Search',
    body: (
      <>
        <p>
          Search tens of thousands of medieval manuscript fragments, chat with a
          research assistant grounded in real scholarship, and explore the
          collection through maps and visualizations.
        </p>
        <p>This one-minute tour shows you the essentials.</p>
      </>
    ),
    selectors: [],
  },
  {
    id: 'search-modes',
    title: 'Four ways to search',
    body: (
      <ul>
        <li><strong>Semantic</strong> — search by meaning, e.g. “dowry disputes in Fustat”.</li>
        <li><strong>Keyword</strong> — exact words and phrases.</li>
        <li><strong>Shelf Mark</strong> — look up a fragment by its catalog reference, e.g. T-S 8J5.14.</li>
        <li><strong>Hybrid</strong> — blends meaning with exact matches.</li>
      </ul>
    ),
    selectors: ['[data-tour="search-modes"]'],
  },
  {
    id: 'search-box',
    title: 'Start with a question or phrase',
    body: (
      <p>
        Type anything from a broad theme to a precise phrase and hit Search.
        The filters below narrow results by language, document type, and
        collection.
      </p>
    ),
    selectors: ['[data-tour="search-box"]'],
  },
  {
    id: 'chat',
    title: 'Chat with the research assistant',
    body: (
      <>
        <p>
          Ask questions in plain language. Answers draw on scholarly
          bibliography and a knowledge graph, cite authors and page numbers,
          and link straight to the manuscripts they mention.
        </p>
        <p className="gt-note">
          Experimental: the assistant can make mistakes — verify claims against
          the cited sources.
        </p>
      </>
    ),
    selectors: ['[data-tour="chat"]', '[data-tour="chat-fab"]'],
  },
  {
    id: 'map',
    title: 'Explore places on the map',
    body: (
      <p>
        An interactive map of the Genizah world — historical places, holding
        institutions, scholars, and the connections tying fragments to
        geography.
      </p>
    ),
    selectors: ['[data-tour="map-button"]'],
  },
  {
    id: 'visualizations',
    title: 'See the collection as a whole',
    body: (
      <>
        <p>
          The Collection Explorer lays thousands of fragments out on an
          interactive grid where similar documents cluster together — related
          material sits side by side, so whole genres and themes emerge at a
          glance.
        </p>
        <p>
          Click points to inspect and compare documents, or project your own
          query into the space to see where it lands. Search results include
          their own mini-visualization too.
        </p>
      </>
    ),
    selectors: ['[data-tour="explorer-button"]'],
  },
  {
    id: 'done',
    title: 'You’re all set',
    body: (
      <p>
        Replay this walkthrough anytime with the 🎓 Tour button at the top of
        the page. Happy exploring!
      </p>
    ),
    selectors: [],
  },
];

/**
 * Return the first visible element matching one of the given selectors.
 * @param {string[]} selectors CSS selectors in priority order.
 * @returns {?Element} The matched element, or null when none is visible.
 */
function findTarget(selectors) {
  for (const selector of selectors || []) {
    const el = document.querySelector(selector);
    if (!el) continue;
    const rect = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    if (rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden') {
      return el;
    }
  }
  return null;
}

/**
 * First-run guided tour: dims the page, spotlights one feature at a time, and
 * explains it in a step card (anchored on desktop, bottom sheet on phones).
 * @param {boolean} open Whether the tour is showing.
 * @param {Function} onClose Called with `true` when the user finishes the
 *   final step, `false` when they skip or dismiss early.
 */
function GuidedTour({ open, onClose }) {
  const [stepIndex, setStepIndex] = useState(0);
  const [layout, setLayout] = useState({ spotlight: null, card: null, phone: false });
  const cardRef = useRef(null);
  const nextBtnRef = useRef(null);

  const step = TOUR_STEPS[stepIndex];
  const isLast = stepIndex === TOUR_STEPS.length - 1;

  // Restart from the first step each time the tour is reopened.
  useEffect(() => {
    if (open) setStepIndex(0);
  }, [open]);

  const measure = useCallback(() => {
    const phone = window.innerWidth <= PHONE_BREAKPOINT;
    const target = findTarget(TOUR_STEPS[stepIndex].selectors);
    let spotlight = null;
    let card = null;
    if (target) {
      const rect = target.getBoundingClientRect();
      spotlight = {
        top: rect.top - SPOTLIGHT_PADDING,
        left: rect.left - SPOTLIGHT_PADDING,
        width: rect.width + SPOTLIGHT_PADDING * 2,
        height: rect.height + SPOTLIGHT_PADDING * 2,
      };
      if (!phone) {
        const cardHeight = cardRef.current?.offsetHeight || 240;
        const below = spotlight.top + spotlight.height + CARD_MARGIN;
        const top = below + cardHeight <= window.innerHeight - CARD_MARGIN
          ? below
          : Math.max(spotlight.top - CARD_MARGIN - cardHeight, CARD_MARGIN);
        const left = Math.min(
          Math.max(spotlight.left, CARD_MARGIN),
          Math.max(window.innerWidth - CARD_WIDTH - CARD_MARGIN, CARD_MARGIN)
        );
        card = { top, left };
      }
    }
    // Skip the re-render when nothing moved (this runs on a polling interval).
    setLayout(prev => {
      const next = { spotlight, card, phone };
      return JSON.stringify(prev) === JSON.stringify(next) ? prev : next;
    });
  }, [stepIndex]);

  // Scroll the step's target into view, then keep the spotlight glued to it:
  // remeasure on resize/scroll and on a slow poll, so late layout shifts
  // (fonts, images, plots loading) can't strand the highlight.
  useEffect(() => {
    if (!open) return;
    const target = findTarget(TOUR_STEPS[stepIndex].selectors);
    if (target) target.scrollIntoView({ block: 'center', behavior: 'auto' });
    const raf = requestAnimationFrame(measure);
    const poll = setInterval(measure, 400);
    window.addEventListener('resize', measure);
    window.addEventListener('scroll', measure, true);
    return () => {
      cancelAnimationFrame(raf);
      clearInterval(poll);
      window.removeEventListener('resize', measure);
      window.removeEventListener('scroll', measure, true);
    };
  }, [open, stepIndex, measure]);

  // Keep keyboard users moving: arrows navigate, Escape dismisses.
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose(false);
      } else if (e.key === 'ArrowRight') {
        setStepIndex(i => Math.min(i + 1, TOUR_STEPS.length - 1));
      } else if (e.key === 'ArrowLeft') {
        setStepIndex(i => Math.max(i - 1, 0));
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  useEffect(() => {
    if (open) nextBtnRef.current?.focus({ preventScroll: true });
  }, [open, stepIndex]);

  if (!open) return null;

  const goNext = () => (isLast ? onClose(true) : setStepIndex(stepIndex + 1));
  const goBack = () => setStepIndex(Math.max(stepIndex - 1, 0));

  const { spotlight, card, phone } = layout;
  const cardStyle = phone
    ? undefined // bottom-sheet position comes from CSS
    : card
      ? { top: card.top, left: card.left, width: CARD_WIDTH }
      : { top: '50%', left: '50%', transform: 'translate(-50%, -50%)', width: CARD_WIDTH };

  return (
    <div className="gt-root" role="dialog" aria-modal="true" aria-label="Site tour">
      <div
        className="gt-blocker"
        style={{ background: spotlight ? 'transparent' : 'rgba(17, 24, 39, 0.62)' }}
      />
      {spotlight && (
        <div
          className="gt-spotlight"
          style={{
            top: spotlight.top,
            left: spotlight.left,
            width: spotlight.width,
            height: spotlight.height,
          }}
        />
      )}
      <div className={`gt-card ${phone ? 'gt-card-phone' : ''}`} ref={cardRef} style={cardStyle}>
        <button className="gt-close" onClick={() => onClose(false)} aria-label="Close tour">×</button>
        <div className="gt-progress">Step {stepIndex + 1} of {TOUR_STEPS.length}</div>
        <h3 className="gt-title">{step.title}</h3>
        <div className="gt-body">{step.body}</div>
        <div className="gt-actions">
          <button className="gt-skip" onClick={() => onClose(false)}>Skip tour</button>
          <div className="gt-nav">
            {stepIndex > 0 && (
              <button className="gt-back" onClick={goBack}>Back</button>
            )}
            <button className="gt-next" onClick={goNext} ref={nextBtnRef}>
              {isLast ? 'Finish' : 'Next'}
            </button>
          </div>
        </div>
        <div className="gt-dots">
          {TOUR_STEPS.map((s, i) => (
            <button
              key={s.id}
              className={`gt-dot ${i === stepIndex ? 'gt-dot-active' : ''}`}
              onClick={() => setStepIndex(i)}
              aria-label={`Go to step ${i + 1}`}
            />
          ))}
        </div>
      </div>

      <style jsx>{`
        .gt-blocker {
          position: fixed;
          inset: 0;
          z-index: 10500;
        }

        .gt-spotlight {
          position: fixed;
          z-index: 10501;
          border-radius: 12px;
          border: 2px solid rgba(255, 255, 255, 0.9);
          box-shadow: 0 0 0 200vmax rgba(17, 24, 39, 0.62);
          pointer-events: none;
          transition: top 0.25s ease, left 0.25s ease, width 0.25s ease, height 0.25s ease;
        }

        .gt-card {
          position: fixed;
          z-index: 10502;
          background: white;
          border-radius: 14px;
          box-shadow: 0 18px 45px rgba(0, 0, 0, 0.35);
          padding: 18px 20px 14px;
          box-sizing: border-box;
          max-width: calc(100vw - 24px);
          animation: gt-card-in 0.25s ease-out;
        }

        .gt-card-phone {
          left: 10px;
          right: 10px;
          bottom: calc(10px + env(safe-area-inset-bottom, 0px));
          top: auto;
          width: auto;
        }

        @keyframes gt-card-in {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: none; }
        }

        .gt-card[style*="translate(-50%"] {
          animation: none;
        }

        .gt-close {
          position: absolute;
          top: 8px;
          right: 10px;
          background: none;
          border: none;
          font-size: 22px;
          line-height: 1;
          color: #9ca3af;
          cursor: pointer;
          padding: 4px;
        }

        .gt-progress {
          font-size: 11px;
          font-weight: 700;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          color: #667eea;
          margin-bottom: 6px;
        }

        .gt-title {
          margin: 0 0 8px;
          font-size: 18px;
          color: #1a202c;
        }

        .gt-body {
          font-size: 14px;
          line-height: 1.55;
          color: #4a5568;
        }

        .gt-body p {
          margin: 0 0 8px;
        }

        .gt-body ul {
          margin: 0 0 8px;
          padding-left: 18px;
        }

        .gt-body li {
          margin-bottom: 5px;
        }

        .gt-note {
          font-size: 12.5px;
          color: #8a6d1a;
          background: #fff8ec;
          border: 1px solid #f0d9a8;
          border-radius: 6px;
          padding: 6px 9px;
        }

        .gt-actions {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-top: 12px;
          gap: 10px;
        }

        .gt-skip {
          background: none;
          border: none;
          color: #718096;
          font-size: 13px;
          cursor: pointer;
          padding: 6px 4px;
          text-decoration: underline;
        }

        .gt-nav {
          display: flex;
          gap: 8px;
        }

        .gt-back {
          background: #edf2f7;
          border: none;
          color: #4a5568;
          border-radius: 8px;
          padding: 8px 14px;
          font-size: 13px;
          font-weight: 600;
          cursor: pointer;
        }

        .gt-next {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          border: none;
          color: white;
          border-radius: 8px;
          padding: 8px 18px;
          font-size: 13px;
          font-weight: 700;
          cursor: pointer;
        }

        .gt-next:focus-visible,
        .gt-back:focus-visible {
          outline: 2px solid #667eea;
          outline-offset: 2px;
        }

        .gt-dots {
          display: flex;
          justify-content: center;
          gap: 6px;
          margin-top: 12px;
        }

        .gt-dot {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          border: none;
          padding: 0;
          background: #d7dcf5;
          cursor: pointer;
        }

        .gt-dot-active {
          background: #667eea;
          transform: scale(1.25);
        }
      `}</style>
    </div>
  );
}

export default GuidedTour;
