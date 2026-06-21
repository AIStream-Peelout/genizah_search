import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { MapContainer, TileLayer, CircleMarker, Polyline, Popup, useMap } from 'react-leaflet';
import { useNavigate } from 'react-router-dom';
import 'leaflet/dist/leaflet.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// ─── Colour palette ──────────────────────────────────────────────────────────

const SCHOLAR_COLOR = '#7EB8D4';

const C = {
  joinedFragments:       '#B07FE0',
  joinedFragmentsBorder: '#7C4BAE',
  place:            '#C8962A',
  placeBorder:      '#8B6200',
  institution:      '#4FC3A1',
  institutionBorder:'#2A7A62',
  connection:       '#C8962A',
  traveledTo:       '#E07A5F',
  livedIn:          '#81B29A',
  person:           '#A78BD4',
  personBorder:     '#6B4FAE',
  personArc:        '#C9B8E8',
  highlight:        'rgba(200,150,42,0.28)',
};

// ─── Utilities ───────────────────────────────────────────────────────────────

function markerRadius(count) {
  if (!count) return 5;
  return Math.min(5 + Math.log(count + 1) * 3.5, 28);
}

function connOpacity(count, max) {
  if (!max) return 0.2;
  return 0.15 + (count / max) * 0.55;
}

// ─── FitBounds ───────────────────────────────────────────────────────────────

function FitBounds({ places, institutions }) {
  const map = useMap();
  useEffect(() => {
    const all = [...places, ...institutions];
    if (!all.length) return;
    const lats = all.map(p => p.lat);
    const lngs = all.map(p => p.lng);
    map.fitBounds(
      [[Math.min(...lats), Math.min(...lngs)], [Math.max(...lats), Math.max(...lngs)]],
      { padding: [40, 40] }
    );
  }, [places, institutions, map]);
  return null;
}

// ─── Relation metadata ───────────────────────────────────────────────────────

const RELATION_INFO = {
  ORIGINATED_FROM: {
    label:       'Originated here',
    explanation: 'This document was produced or issued at this location.',
    color:       '#C8962A',
  },
  MENTIONS_PLACE: {
    label:       'Mentions this place',
    explanation: 'The text of this document references this location by name.',
    color:       '#7B9FD4',
  },
  WRITTEN_AT: {
    label:       'Written here',
    explanation: 'This document was physically written at this location.',
    color:       '#81B29A',
  },
  SENT_TO: {
    label:       'Sent to this place',
    explanation: 'This document was addressed or dispatched to this location.',
    color:       '#E07A5F',
  },
};

function RelationBadge({ relation }) {
  const info = RELATION_INFO[relation] || {
    label: relation?.replace(/_/g, ' ').toLowerCase() || 'connected',
    explanation: '',
    color: '#6b7280',
  };
  return (
    <span title={info.explanation} style={{
      background: `${info.color}22`,
      color: info.color,
      border: `1px solid ${info.color}55`,
      borderRadius: 10,
      padding: '2px 8px',
      fontSize: 11,
      fontWeight: 600,
      letterSpacing: '0.03em',
      cursor: 'help',
      whiteSpace: 'nowrap',
    }}>
      {info.label}
    </span>
  );
}

// ─── Text highlighter ────────────────────────────────────────────────────────

function HighlightedText({ text, placeName, nameVariants }) {
  if (!text) return null;
  const terms = [placeName, ...(nameVariants ? nameVariants.split(',').map(s => s.trim()) : [])]
    .filter(Boolean);
  if (!terms.length) return <>{text}</>;
  const pattern = terms.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|');
  const regex = new RegExp(`(${pattern})`, 'gi');
  const parts = text.split(regex);
  // split() with a capturing group puts matched terms at odd indices (1, 3, 5…)
  // Using index parity avoids the lastIndex mutation bug from regex.test() with /g
  return (
    <>
      {parts.map((part, i) =>
        i % 2 === 1
          ? <mark key={i} style={{ background: C.highlight, color: '#C8962A', borderRadius: 2, padding: '0 2px' }}>{part}</mark>
          : part
      )}
    </>
  );
}

// ─── Scholar panel ────────────────────────────────────────────────────────────

function ScholarPanel({ scholarName, detail, onFragmentClick, onBack, onClose }) {
  if (!scholarName) return null;

  const books = (detail?.books || []).filter(
    b => b?.title && !b.title.toLowerCase().includes('unpublished')
  );

  return (
    <div style={styles.panel}>
      <button style={styles.panelClose} onClick={onClose}>✕</button>

      {/* Back to institution */}
      {onBack && (
        <button style={styles.backLink} onClick={onBack}>← Back to institution</button>
      )}

      <div style={{ ...styles.panelTypeBadge, background: `${SCHOLAR_COLOR}22`, color: SCHOLAR_COLOR, borderColor: `${SCHOLAR_COLOR}55` }}>
        Scholar
      </div>
      <h2 style={{ ...styles.panelTitle, color: SCHOLAR_COLOR }}>{scholarName}</h2>

      <div style={styles.statRow}>
        <span style={styles.stat}><strong>{books.length}</strong> published work{books.length !== 1 ? 's' : ''}</span>
        {detail && <span style={styles.stat}><strong>{detail.fragment_count}</strong> fragments studied</span>}
      </div>

      {!detail && <p style={styles.panelLoading}>Loading details…</p>}

      {detail && (
        <>
          {/* Publications */}
          {books.length > 0 && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>Publications</h3>
              {books.map((b, i) => (
                <div key={i} style={styles.bookCard}>
                  <p style={styles.bookTitle}>{b.title}</p>
                  <div style={styles.bookMeta}>
                    {b.year    && <span style={styles.metaChip}>{b.year}</span>}
                    {b.journal && <span style={styles.metaChip}>{b.journal}</span>}
                    {b.publisher && !b.journal && <span style={styles.metaChip}>{b.publisher}</span>}
                    {b.has_local_copy && (
                      <span
                        title="This publication has been indexed and short excerpts are available via the chatbot and knowledge graph."
                        style={{ ...styles.metaChip, background: '#1a3a2a', color: '#6ee7a0', border: '1px solid #2a5a3a', cursor: 'help' }}
                      >
                        Indexed
                      </span>
                    )}
                  </div>
                  {b.doi && (
                    <span style={styles.doiText}>DOI: {b.doi}</span>
                  )}
                </div>
              ))}
            </section>
          )}

          {/* Fragments they've written about */}
          {detail.fragments?.filter(f => f.shelfmark).length > 0 && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>Fragments they've studied</h3>
              {detail.fragments.filter(f => f.shelfmark).map((f, i) => (
                <button key={i} style={styles.fragmentCard} onClick={() => onFragmentClick(f)}>
                  <div style={styles.fragmentCardTop}>
                    <span style={styles.shelfmark}>{f.shelfmark}</span>
                    {f.date_range && <span style={styles.fragmentMetaChip}>{f.date_range}</span>}
                  </div>
                  {f.description && (
                    <p style={styles.description}>
                      {f.description.length > 120 ? f.description.slice(0, 120) + '…' : f.description}
                    </p>
                  )}
                  <span style={styles.clickHint}>Click to view →</span>
                </button>
              ))}
            </section>
          )}

          {/* Places covered in their work */}
          {detail.places?.filter(Boolean).length > 0 && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>Places covered in their work</h3>
              <div style={styles.tagsRow}>
                {detail.places.filter(Boolean).map((p, i) => (
                  <span key={i} style={styles.tag}>{p}</span>
                ))}
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
}

// ─── Institution panel ───────────────────────────────────────────────────────

function InstitutionPanel({ institution, detail, onFragmentClick, onScholarClick, onClose }) {
  if (!institution) return null;
  return (
    <div style={styles.panel}>
      <button style={styles.panelClose} onClick={onClose}>✕</button>
      <div style={{ ...styles.panelTypeBadge, background: `${C.institution}22`, color: C.institution, borderColor: `${C.institution}55` }}>
        Institution
      </div>
      <h2 style={{ ...styles.panelTitle, color: C.institution }}>{institution.name}</h2>
      {institution.country && <p style={styles.panelMeta}>{institution.country}</p>}

      <div style={styles.statRow}>
        <span style={styles.stat}><strong>{detail?.fragment_count ?? institution.fragment_count}</strong> fragments</span>
      </div>

      {!detail && <p style={styles.panelLoading}>Loading details…</p>}

      {detail && (
        <>
          {detail.languages?.filter(Boolean).length > 0 && (
            <div style={styles.tagsRow}>
              {detail.languages.filter(Boolean).map((l, i) => (
                <span key={i} style={styles.tag}>{l}</span>
              ))}
            </div>
          )}

          {detail.scholars?.filter(Boolean).length > 0 && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>Scholars who studied this collection</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {detail.scholars.filter(Boolean).map((s, i) => (
                  <button key={i} style={styles.scholarBtn} onClick={() => onScholarClick(s)}>
                    <span style={styles.scholarName}>{s}</span>
                    <span style={styles.scholarArrow}>→</span>
                  </button>
                ))}
              </div>
            </section>
          )}

          {detail.people?.filter(Boolean).length > 0 && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>People mentioned in fragments here</h3>
              <ul style={styles.list}>
                {detail.people.filter(Boolean).map((p, i) => (
                  <li key={i} style={styles.listItem}>{p}</li>
                ))}
              </ul>
            </section>
          )}

          {detail.fragments?.filter(f => f.shelfmark).length > 0 && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>Sample fragments held here</h3>
              {detail.fragments.filter(f => f.shelfmark).map((f, i) => (
                <button key={i} style={styles.fragmentCard} onClick={() => onFragmentClick(f)}>
                  <div style={styles.fragmentCardTop}>
                    <span style={styles.shelfmark}>{f.shelfmark}</span>
                    {f.date_range && <span style={styles.fragmentMetaChip}>{f.date_range}</span>}
                  </div>
                  {f.description && (
                    <p style={styles.description}>
                      {f.description.length > 120 ? f.description.slice(0, 120) + '…' : f.description}
                    </p>
                  )}
                  <span style={styles.clickHint}>Click to view →</span>
                </button>
              ))}
            </section>
          )}
        </>
      )}
    </div>
  );
}

// ─── Person panel ─────────────────────────────────────────────────────────────

function PersonPanel({ personName, detail, onFragmentClick, onClose }) {
  if (!personName) return null;

  const livedIn    = (detail?.places || []).filter(p => p.relation === 'LIVED_IN' || p.relation === 'ORIGINATED_FROM');
  const traveledTo = (detail?.places || []).filter(p => p.relation === 'TRAVELED_TO');

  return (
    <div style={styles.panel}>
      <button style={styles.panelClose} onClick={onClose}>✕</button>
      <div style={{ ...styles.panelTypeBadge, background: `${C.person}22`, color: C.person, borderColor: `${C.person}55` }}>
        Person
      </div>
      <h2 style={{ ...styles.panelTitle, color: C.person }}>{personName}</h2>
      {detail?.role && <p style={styles.panelMeta}>{detail.role}{detail.home_base ? ` · ${detail.home_base}` : ''}</p>}
      {detail?.description && <p style={styles.personDescription}>{detail.description}</p>}

      {detail && (
        <div style={styles.statRow}>
          <span style={styles.stat}><strong>{detail.fragment_count}</strong> fragments</span>
          <span style={styles.stat}><strong>{livedIn.length}</strong> home{livedIn.length !== 1 ? 's' : ''}</span>
          <span style={styles.stat}><strong>{traveledTo.length}</strong> destinations</span>
        </div>
      )}

      {!detail && <p style={styles.panelLoading}>Loading details…</p>}

      {detail && (
        <>
          {livedIn.length > 0 && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>Home / Origin</h3>
              <ul style={styles.list}>
                {livedIn.map((p, i) => (
                  <li key={i} style={styles.listItem}>
                    {p.place}{p.country ? ` — ${p.country}` : ''}
                  </li>
                ))}
              </ul>
            </section>
          )}

          {traveledTo.length > 0 && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>Traveled to</h3>
              <ul style={styles.list}>
                {traveledTo.map((p, i) => (
                  <li key={i} style={styles.listItem}>
                    {p.place}{p.country ? ` — ${p.country}` : ''}
                  </li>
                ))}
              </ul>
            </section>
          )}

          {detail.fragments?.filter(f => f.shelfmark).length > 0 && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>Mentioned in fragments</h3>
              {detail.fragments.filter(f => f.shelfmark).map((f, i) => (
                <button key={i} style={styles.fragmentCard} onClick={() => onFragmentClick(f)}>
                  <div style={styles.fragmentCardTop}>
                    <span style={styles.shelfmark}>{f.shelfmark}</span>
                    {f.date_range && <span style={styles.fragmentMetaChip}>{f.date_range}</span>}
                  </div>
                  {f.description && (
                    <p style={styles.description}>
                      {f.description.length > 120 ? f.description.slice(0, 120) + '…' : f.description}
                    </p>
                  )}
                  <span style={styles.clickHint}>Click to view →</span>
                </button>
              ))}
            </section>
          )}

          {detail.books?.filter(Boolean).length > 0 && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>Scholarship</h3>
              <ul style={styles.list}>
                {detail.books.filter(Boolean).map((b, i) => (
                  <li key={i} style={styles.listItem}>{b}</li>
                ))}
              </ul>
            </section>
          )}
        </>
      )}
    </div>
  );
}

// ─── People-at-place popup ─────────────────────────────────────────────────────

/**
 * Lists every historical person anchored to a single place, since many people
 * share the same coordinates (e.g. 150+ at Fustat). Clicking a name selects
 * that person to trace their journeys.
 */
function PeoplePlacePopup({ group, selectedPerson, onSelect }) {
  const people = group.people;
  return (
    <div style={{ minWidth: 200, maxWidth: 260, fontFamily: 'sans-serif' }}>
      <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 2, color: '#1a1a2e' }}>
        {group.place}
      </div>
      <div style={{ fontSize: 12, color: '#666', marginBottom: 8 }}>
        <strong>{people.length}</strong> {people.length === 1 ? 'person' : 'people'} connected to this place
      </div>
      <div style={{ maxHeight: 220, overflowY: 'auto' }}>
        {people.map((pm, i) => (
          <button
            key={i}
            onClick={() => onSelect(pm.person)}
            style={{
              display: 'block', width: '100%', textAlign: 'left',
              background: selectedPerson === pm.person ? '#efe7fb' : (i % 2 === 0 ? '#faf8ff' : '#fff'),
              border: 'none', borderTop: '1px solid #ede8f5',
              padding: '5px 6px', cursor: 'pointer', fontSize: 12,
            }}
          >
            <span style={{ color: '#6B4FAE', fontWeight: 600 }}>{pm.person}</span>
            {pm.role && <span style={{ color: '#999', fontSize: 11 }}> · {pm.role}</span>}
          </button>
        ))}
      </div>
      <div style={{ fontSize: 10, color: '#999', padding: '6px 2px 0' }}>
        Click a name to trace their journeys
      </div>
    </div>
  );
}

// ─── Joined-fragments popup ───────────────────────────────────────────────────

function JoinedFragmentsPopup({ join }) {
  const samples = (join.sample_joins || []).filter(j => j.shelfmark1);
  return (
    <div style={{ minWidth: 220, maxWidth: 300, fontFamily: 'sans-serif' }}>
      <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 4, color: '#1a1a2e' }}>
        Joined fragments
      </div>
      <div style={{ fontSize: 12, color: '#666', marginBottom: samples.length ? 10 : 0 }}>
        <strong>{join.join_count}</strong> fragment pair{join.join_count !== 1 ? 's' : ''} split
        across <strong>{join.source}</strong> and <strong>{join.target}</strong>
      </div>
      {samples.map((j, i) => (
        <div key={i} style={{
          background: i % 2 === 0 ? '#f8f6ff' : '#fff',
          borderTop: '1px solid #ede8f5',
          padding: '6px 4px',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
            <span style={{ color: '#7C4BAE', fontFamily: 'monospace', fontSize: 12, fontWeight: 600 }}>
              {j.shelfmark1}
            </span>
            <span style={{ color: '#aaa', fontSize: 11 }}>↔</span>
            <span style={{ color: '#7C4BAE', fontFamily: 'monospace', fontSize: 12, fontWeight: 600 }}>
              {j.shelfmark2}
            </span>
            {j.date_range && <span style={{ color: '#999', fontSize: 10 }}>{j.date_range}</span>}
          </div>
          {j.description && (
            <p style={{ margin: '3px 0 0', fontSize: 11, color: '#666', lineHeight: 1.4 }}>
              {j.description.length > 90 ? j.description.slice(0, 90) + '…' : j.description}
            </p>
          )}
        </div>
      ))}
      {join.join_count > 5 && (
        <div style={{ fontSize: 11, color: '#999', padding: '4px 4px 0', textAlign: 'right' }}>
          +{join.join_count - 5} more pairs
        </div>
      )}
    </div>
  );
}

// ─── Connection popup ────────────────────────────────────────────────────────

function ConnectionPopup({ conn, onFragmentClick }) {
  const frags = (conn.sample_fragments || []).filter(f => f.shelfmark);
  return (
    <div style={{ minWidth: 220, maxWidth: 290, fontFamily: 'sans-serif' }}>
      {/* Header */}
      <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 4, color: '#1a1a2e' }}>
        {conn.source} → {conn.target}
      </div>
      <div style={{ fontSize: 12, color: '#666', marginBottom: frags.length ? 10 : 0 }}>
        {conn.connections} fragment{conn.connections !== 1 ? 's' : ''} originated in{' '}
        <strong>{conn.source}</strong> and reference <strong>{conn.target}</strong>
      </div>

      {/* Sample fragments */}
      {frags.map((f, i) => (
        <button
          key={i}
          onClick={() => onFragmentClick(f, conn.source)}
          style={{
            display: 'block', width: '100%', textAlign: 'left',
            background: i % 2 === 0 ? '#f8f6f0' : '#fff',
            border: 'none', borderTop: '1px solid #e5e0d5',
            padding: '6px 4px', cursor: 'pointer',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 2 }}>
            <span style={{ color: '#8B6200', fontFamily: 'monospace', fontSize: 12, fontWeight: 600 }}>
              {f.shelfmark}
            </span>
            {f.date_range && (
              <span style={{ color: '#999', fontSize: 10 }}>{f.date_range}</span>
            )}
            {(f.relations || []).map((rel, ri) => {
              const info = RELATION_INFO[rel];
              return info ? (
                <span key={ri} style={{
                  background: `${info.color}22`, color: info.color,
                  border: `1px solid ${info.color}55`,
                  borderRadius: 8, padding: '1px 5px', fontSize: 10, fontWeight: 600,
                }}>
                  {info.label}
                </span>
              ) : null;
            })}
          </div>
          {f.description && (
            <p style={{ margin: 0, fontSize: 11, color: '#555', lineHeight: 1.4 }}>
              {f.description.length > 90 ? f.description.slice(0, 90) + '…' : f.description}
            </p>
          )}
        </button>
      ))}

      {conn.connections > 5 && (
        <div style={{ fontSize: 11, color: '#999', padding: '4px 4px 0', textAlign: 'right' }}>
          +{conn.connections - 5} more — click either place to see all
        </div>
      )}
    </div>
  );
}

// ─── Fragment modal ──────────────────────────────────────────────────────────

function FragmentModal({ fragment, placeName, nameVariants, onClose }) {
  if (!fragment) return null;

  const relations = fragment.relations || [];

  return (
    <div style={styles.modalOverlay} onClick={e => e.target === e.currentTarget && onClose()}>
      <div style={styles.modal}>
        <button style={styles.modalClose} onClick={onClose}>✕</button>

        {/* Header */}
        <div style={styles.modalHeader}>
          <span style={styles.modalShelfmark}>{fragment.shelfmark}</span>
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
            {relations.map((rel, i) => (
              <RelationBadge key={i} relation={rel?.type || rel} />
            ))}
            {relations.some(r => (r?.certainty) === 'possible') && (
              <span style={styles.possibleBadge}>possible</span>
            )}
          </div>
        </div>

        {/* Meta chips */}
        <div style={styles.metaRow}>
          {(fragment.period || fragment.date_range) &&
            <span style={styles.metaChip}>{fragment.period || fragment.date_range}</span>}
          {fragment.type && <span style={styles.metaChip}>{fragment.type}</span>}
          {fragment.has_transcription && <span style={styles.metaChip}>📜 Transcription available</span>}
          {fragment.has_translation  && <span style={styles.metaChip}>🔤 Translation available</span>}
          {fragment.data_sources && !fragment.data_sources.includes('pgp') &&
            <span style={{ ...styles.metaChip, ...styles.academicBadge }}>academic source</span>}
        </div>

        {/* Description with place-name highlighting */}
        {fragment.description ? (
          <div style={styles.modalDescription}>
            <p style={styles.modalDescLabel}>Description</p>
            <p style={styles.modalDescText}>
              <HighlightedText
                text={fragment.description}
                placeName={placeName}
                nameVariants={nameVariants}
              />
            </p>
          </div>
        ) : (
          <p style={styles.modalLoading}>No description available.</p>
        )}

        {fragment.pgpid && (
          <p style={styles.modalDescLabel} >PGP ID: {fragment.pgpid}</p>
        )}
      </div>
    </div>
  );
}

// ─── Place panel ─────────────────────────────────────────────────────────────

function PlacePanel({ place, detail, onFragmentClick, onClose }) {
  if (!place) return null;

  return (
    <div style={styles.panel}>
      <button style={styles.panelClose} onClick={onClose}>✕</button>

      <h2 style={styles.panelTitle}>{place.name}</h2>
      {place.country && (
        <p style={styles.panelMeta}>
          {place.country}{place.region ? ` · ${place.region}` : ''}
        </p>
      )}

      <div style={styles.statRow}>
        <span style={styles.stat}><strong>{place.fragment_count}</strong> fragments</span>
        <span style={styles.stat}><strong>{place.person_count}</strong> people</span>
      </div>

      {place.name_variants && (
        <p style={styles.variants}>Also known as: {place.name_variants}</p>
      )}

      {!detail && <p style={styles.panelLoading}>Loading details…</p>}

      {detail && (
        <>
          {detail.people?.filter(p => p?.name).length > 0 && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>People connected here</h3>
              <ul style={styles.list}>
                {detail.people.filter(p => p?.name).map((p, i) => (
                  <li key={i} style={styles.listItem}>
                    {p.name}
                    {p.role && <span style={styles.personRole}> · {p.role}</span>}
                  </li>
                ))}
              </ul>
            </section>
          )}

          {detail.fragments?.filter(f => f.shelfmark).length > 0 && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>Fragments</h3>
              {detail.fragments.filter(f => f.shelfmark).map((f, i) => (
                <React.Fragment key={i}>
                <button
                  style={styles.fragmentCard}
                  onClick={() => onFragmentClick(f)}
                >
                  {/* Top row: shelfmark + relation badges */}
                  <div style={styles.fragmentCardTop}>
                    <span style={styles.shelfmark}>{f.shelfmark}</span>
                    <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', alignItems: 'center' }}>
                      {(f.relations || []).map((rel, ri) => {
                        const relType = rel?.type || rel;
                        return <RelationBadge key={ri} relation={relType} />;
                      })}
                      {/* Possible certainty warning */}
                      {(f.relations || []).some(r => (r?.certainty || r) === 'possible') && (
                        <span style={styles.possibleBadge} title="This connection is uncertain">possible</span>
                      )}
                      {/* Provenance badge for non-PGP */}
                      {f.data_sources && !f.data_sources.includes('pgp') && (
                        <span style={styles.academicBadge} title="From academic literature extraction">academic</span>
                      )}
                    </div>
                  </div>

                  {/* Explanation lines */}
                  {(f.relations || []).map((rel, ri) => {
                    const relType = rel?.type || rel;
                    return RELATION_INFO[relType] ? (
                      <p key={ri} style={styles.relationExplanation}>{RELATION_INFO[relType].explanation}</p>
                    ) : null;
                  })}

                  {/* Meta row */}
                  <div style={styles.fragmentMeta}>
                    {(f.period || f.date_range) && (
                      <span style={styles.fragmentMetaChip}>{f.period || f.date_range}</span>
                    )}
                    {f.type && <span style={styles.fragmentMetaChip}>{f.type}</span>}
                    {f.has_transcription && <span style={styles.fragmentMetaChip}>📜 transcription</span>}
                    {f.has_translation   && <span style={styles.fragmentMetaChip}>🔤 translation</span>}
                  </div>

                  {/* Description preview */}
                  {f.description && (
                    <p style={styles.description}>
                      {f.description.length > 120
                        ? f.description.slice(0, 120) + '…'
                        : f.description}
                    </p>
                  )}

                  <span style={styles.clickHint}>Click to view full fragment →</span>
                </button>
                {f.pgpid && (
                  <span style={styles.pgpIdText}>PGP ID: {f.pgpid}</span>
                )}
                </React.Fragment>
              ))}
            </section>
          )}

          {detail.books?.filter(Boolean).length > 0 && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>Scholarship</h3>
              <ul style={styles.list}>
                {detail.books.filter(Boolean).map((b, i) => (
                  <li key={i} style={styles.listItem}>{b}</li>
                ))}
              </ul>
            </section>
          )}
        </>
      )}
    </div>
  );
}

// ─── People search ─────────────────────────────────────────────────────────────

/**
 * Type-ahead search over all mappable people. Since a single place can anchor
 * 150+ people, this lets the user jump straight to a person by name; selecting
 * one opens their card and traces their journeys.
 */
function PeopleSearch({ people, onSelect }) {
  const [query, setQuery] = useState('');
  const [open, setOpen]   = useState(false);

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return [];
    return people.filter(pm => pm.person.toLowerCase().includes(q)).slice(0, 8);
  }, [query, people]);

  const choose = person => { onSelect(person); setQuery(''); setOpen(false); };

  return (
    <div style={styles.peopleSearchWrap}>
      <input
        type="text"
        value={query}
        placeholder="🔍 Search people…"
        onChange={e => { setQuery(e.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        onKeyDown={e => {
          if (e.key === 'Enter' && results.length) choose(results[0].person);
          else if (e.key === 'Escape') { setQuery(''); setOpen(false); }
        }}
        style={styles.peopleSearchInput}
      />
      {open && query.trim() && (
        <div style={styles.peopleSearchDropdown}>
          {results.length === 0 && (
            <div style={styles.peopleSearchEmpty}>No mappable people match “{query.trim()}”</div>
          )}
          {results.map((pm, i) => (
            <button key={i} onMouseDown={() => choose(pm.person)} style={styles.peopleSearchResult}>
              <span style={{ color: C.person, fontWeight: 600, fontSize: 12 }}>{pm.person}</span>
              {pm.role && <span style={{ color: '#888', fontSize: 11 }}> · {pm.role}</span>}
              <span style={{ color: '#6b7280', fontSize: 11, display: 'block' }}>{pm.place}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Pan-to-person ───────────────────────────────────────────────────────────

/**
 * Frames the selected person's journey on the map — fits the bounds of all their
 * arc endpoints, or recentres on their single anchor when they have no routes.
 */
function PanToPerson({ markers, arcs, selectedPerson }) {
  const map = useMap();
  useEffect(() => {
    if (!selectedPerson) return;
    const pts = [];
    arcs.forEach(a => { if (a.person === selectedPerson) { pts.push(a.from); pts.push(a.to); } });
    if (pts.length) {
      map.fitBounds(pts, { padding: [80, 80], maxZoom: 8, animate: true });
    } else {
      const pm = markers.find(m => m.person === selectedPerson);
      if (pm) map.setView([pm.lat, pm.lng], Math.max(map.getZoom(), 6), { animate: true });
    }
  }, [selectedPerson, markers, arcs, map]);
  return null;
}

// ─── Main component ──────────────────────────────────────────────────────────

export default function MapView() {
  const navigate = useNavigate();
  const [places,      setPlaces]      = useState([]);
  const [institutions,setInstitutions]= useState([]);
  const [connections, setConnections] = useState([]);
  const [journeys,    setJourneys]    = useState([]);
  const [journeysLoaded, setJourneysLoaded] = useState(false);

  const [selectedPlace,       setSelectedPlace]       = useState(null);
  const [placeDetail,         setPlaceDetail]         = useState(null);
  const [selectedInstitution, setSelectedInstitution] = useState(null);
  const [institutionDetail,   setInstitutionDetail]   = useState(null);
  const [selectedScholar,     setSelectedScholar]     = useState(null);
  const [scholarDetail,       setScholarDetail]       = useState(null);
  const [personDetail,        setPersonDetail]        = useState(null);
  const [fragmentModal,       setFragmentModal]       = useState(null);

  const [selectedPerson,   setSelectedPerson]   = useState(null);

  const [includeAcademic,     setIncludeAcademic]     = useState(false);
  const [showPlaces,          setShowPlaces]          = useState(true);
  const [showConnections,     setShowConnections]     = useState(true);
  const [showInstitutions,    setShowInstitutions]    = useState(true);
  const [showJoinedFragments, setShowJoinedFragments] = useState(false);
  const [joinedFragments,     setJoinedFragments]     = useState([]);
  const [joinedLoaded,        setJoinedLoaded]        = useState(false);
  const [showPeople,          setShowPeople]          = useState(false);
  const [showJourneys,        setShowJourneys]        = useState(false);

  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  // Re-fetch places + connections whenever the academic toggle changes
  useEffect(() => {
    setLoading(true);
    const qs = `include_academic=${includeAcademic}`;
    async function load() {
      try {
        const [pRes, iRes, cRes] = await Promise.all([
          fetch(`${API_BASE_URL}/map/places?${qs}`),
          fetch(`${API_BASE_URL}/map/institutions`),
          fetch(`${API_BASE_URL}/map/connections?min_connections=3&${qs}`),
        ]);
        const [pData, iData, cData] = await Promise.all([pRes.json(), iRes.json(), cRes.json()]);
        setPlaces(pData.places || []);
        setInstitutions((iData.institutions || []).filter(i => i.fragment_count > 0));
        setConnections(cData.connections || []);
        // Reset lazy-loaded layers so they re-fetch with new source filter
        setJourneys([]); setJourneysLoaded(false);
      } catch (e) {
        setError(e.message);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [includeAcademic]);

  // Load cross-institution joins lazily
  useEffect(() => {
    if (!showJoinedFragments || joinedLoaded) return;
    fetch(`${API_BASE_URL}/map/joined-fragments`)
      .then(r => r.json())
      .then(data => { setJoinedFragments(data.joins || []); setJoinedLoaded(true); })
      .catch(console.error);
  }, [showJoinedFragments, joinedLoaded]);

  // Load journeys lazily — triggered by either People or Journeys toggle
  useEffect(() => {
    if ((!showJourneys && !showPeople) || journeysLoaded) return;
    fetch(`${API_BASE_URL}/map/journeys?include_academic=${includeAcademic}`)
      .then(r => r.json())
      .then(data => { setJourneys(data.journeys || []); setJourneysLoaded(true); })
      .catch(console.error);
  }, [showJourneys, showPeople, journeysLoaded, includeAcademic]);

  // Load place detail when selection or source filter changes
  useEffect(() => {
    if (!selectedPlace) { setPlaceDetail(null); return; }
    setPlaceDetail(null);
    fetch(`${API_BASE_URL}/map/place/${encodeURIComponent(selectedPlace.name)}?include_academic=${includeAcademic}`)
      .then(r => r.json()).then(setPlaceDetail).catch(console.error);
  }, [selectedPlace, includeAcademic]);

  // Load institution detail
  useEffect(() => {
    if (!selectedInstitution) { setInstitutionDetail(null); return; }
    setInstitutionDetail(null);
    fetch(`${API_BASE_URL}/map/institution/${encodeURIComponent(selectedInstitution.name)}`)
      .then(r => r.json()).then(setInstitutionDetail).catch(console.error);
  }, [selectedInstitution]);

  // Load scholar detail
  useEffect(() => {
    if (!selectedScholar) { setScholarDetail(null); return; }
    setScholarDetail(null);
    fetch(`${API_BASE_URL}/map/scholar/${encodeURIComponent(selectedScholar)}`)
      .then(r => r.json()).then(setScholarDetail).catch(console.error);
  }, [selectedScholar]);

  // Load person detail when selected person or source filter changes
  useEffect(() => {
    if (!selectedPerson) { setPersonDetail(null); return; }
    setPersonDetail(null);
    fetch(`${API_BASE_URL}/map/person/${encodeURIComponent(selectedPerson)}?include_academic=${includeAcademic}`)
      .then(r => r.json()).then(setPersonDetail).catch(console.error);
  }, [selectedPerson, includeAcademic]);

  // People markers — prefer LIVED_IN, then ORIGINATED_FROM, fall back to first TRAVELED_TO
  const peopleMarkers = useMemo(() => {
    const homes = {};      // LIVED_IN
    const origins = {};    // ORIGINATED_FROM
    const fallbacks = {};  // TRAVELED_TO
    journeys.forEach(j => {
      if (j.relation_type === 'LIVED_IN' && !homes[j.person]) {
        homes[j.person] = { person: j.person, lat: j.lat, lng: j.lng, place: j.place, role: j.role };
      }
      if (j.relation_type === 'ORIGINATED_FROM' && !origins[j.person]) {
        origins[j.person] = { person: j.person, lat: j.lat, lng: j.lng, place: j.place, role: j.role };
      }
      if (j.relation_type === 'TRAVELED_TO' && !fallbacks[j.person]) {
        fallbacks[j.person] = { person: j.person, lat: j.lat, lng: j.lng, place: j.place, role: j.role };
      }
    });
    const all = { ...fallbacks, ...origins, ...homes };
    return Object.values(all);
  }, [journeys]);

  // Aggregate people by their anchor coordinates — one marker per place sized by
  // headcount, since hundreds of people stack on the same ~112 geocoded points.
  const peopleByPlace = useMemo(() => {
    const groups = {};
    peopleMarkers.forEach(pm => {
      const key = `${pm.lat},${pm.lng}`;
      if (!groups[key]) groups[key] = { lat: pm.lat, lng: pm.lng, place: pm.place, people: [] };
      groups[key].people.push(pm);
    });
    // Sort each group's people alphabetically for a stable, scannable popup list
    Object.values(groups).forEach(g => g.people.sort((a, b) => a.person.localeCompare(b.person)));
    return Object.values(groups);
  }, [peopleMarkers]);

  // Journey arcs — home (LIVED_IN > ORIGINATED_FROM) → destinations; else chain TRAVELED_TO
  const allJourneyArcs = useMemo(() => {
    const byPerson = {};
    journeys.forEach(j => {
      if (!byPerson[j.person]) byPerson[j.person] = { home: null, dests: [] };
      const rec = byPerson[j.person];
      if (j.relation_type === 'LIVED_IN') {
        // LIVED_IN always wins as home
        if (!rec.home || rec.home.relation_type === 'ORIGINATED_FROM') rec.home = j;
      } else if (j.relation_type === 'ORIGINATED_FROM' && !rec.home) {
        rec.home = j;
      } else if (j.relation_type === 'TRAVELED_TO') {
        rec.dests.push(j);
      }
    });
    const arcs = [];
    Object.entries(byPerson).forEach(([person, { home, dests }]) => {
      if (home && dests.length) {
        // Draw from known home to each destination
        dests.forEach(dest => arcs.push({
          person, role: home.role,
          from: [home.lat, home.lng], to: [dest.lat, dest.lng],
          fromPlace: home.place, toPlace: dest.place,
        }));
      } else if (!home && dests.length > 1) {
        // No home — chain traveled places together to show movement
        for (let i = 0; i < dests.length - 1; i++) {
          arcs.push({
            person, role: dests[i].role,
            from: [dests[i].lat, dests[i].lng], to: [dests[i + 1].lat, dests[i + 1].lng],
            fromPlace: dests[i].place, toPlace: dests[i + 1].place,
          });
        }
      }
    });
    return arcs;
  }, [journeys]);

  // Which arcs to actually render: selected person takes priority, else global toggle
  const visibleArcs = useMemo(() => {
    if (selectedPerson) return allJourneyArcs.filter(a => a.person === selectedPerson);
    if (showJourneys)   return allJourneyArcs;
    return [];
  }, [allJourneyArcs, selectedPerson, showJourneys]);

  const maxConnections = useMemo(
    () => connections.length ? Math.max(...connections.map(c => c.connections)) : 1,
    [connections]
  );

  const handleMarkerClick = useCallback(place => {
    setSelectedPlace(place);
    setSelectedInstitution(null);
    setSelectedPerson(null);
  }, []);

  const handleInstitutionClick = useCallback(inst => {
    setSelectedInstitution(inst);
    setSelectedPlace(null);
    setSelectedPerson(null);
  }, []);

  // Called from place panel — has full place context for text highlighting
  const handleFragmentClick = useCallback((fragment) => {
    setFragmentModal({
      fragment,
      placeName:    selectedPlace?.name || '',
      nameVariants: selectedPlace?.name_variants || '',
    });
  }, [selectedPlace]);

  // Called from connection popup — uses source place name as highlight context
  const handleConnectionFragmentClick = useCallback((fragment, sourceName) => {
    setFragmentModal({ fragment, placeName: sourceName || '', nameVariants: '' });
  }, []);


  if (loading) return (
    <div style={styles.centred}>
      <div style={styles.spinner} />
      <p style={{ color: C.place, marginTop: 16 }}>Loading map data…</p>
    </div>
  );

  if (error) return (
    <div style={styles.centred}>
      <p style={{ color: '#e55' }}>Failed to load map: {error}</p>
    </div>
  );

  return (
    <div style={styles.wrapper}>

      {/* ── Toolbar ── */}
      <div style={styles.toolbar}>

        {/* Left: nav + title */}
        <button style={styles.backBtn} onClick={() => navigate('/')}>← Search</button>
        <span style={styles.toolbarTitle}>Cairo Genizah — Places</span>

        {/* Centre: layer toggles */}
        <div style={styles.toggleGroup}>
          <span style={styles.toggleGroupLabel}>Layers</span>
          <label style={styles.toggle}>
            <input type="checkbox" checked={showPlaces}
              onChange={e => setShowPlaces(e.target.checked)} style={{ marginRight: 4 }} />
            Places
          </label>
          <label style={styles.toggle}>
            <input type="checkbox" checked={showConnections}
              onChange={e => setShowConnections(e.target.checked)} style={{ marginRight: 4 }} />
            Connections
          </label>
          <label style={styles.toggle}>
            <input type="checkbox" checked={showInstitutions}
              onChange={e => setShowInstitutions(e.target.checked)} style={{ marginRight: 4 }} />
            Institutions
          </label>
          <label style={styles.toggle}>
            <input type="checkbox" checked={showJoinedFragments}
              onChange={e => setShowJoinedFragments(e.target.checked)} style={{ marginRight: 4 }} />
            Joins
          </label>
          <label style={styles.toggle}>
            <input type="checkbox" checked={showPeople}
              onChange={e => { setShowPeople(e.target.checked); if (!e.target.checked) setSelectedPerson(null); }}
              style={{ marginRight: 4 }} />
            People
          </label>
          <label style={styles.toggle}>
            <input type="checkbox" checked={showJourneys}
              onChange={e => setShowJourneys(e.target.checked)} style={{ marginRight: 4 }} />
            Journeys
          </label>
        </div>

        {showPeople && (
          <PeopleSearch
            people={peopleMarkers}
            onSelect={person => {
              setSelectedPerson(person);
              setSelectedPlace(null);
              setSelectedInstitution(null);
            }}
          />
        )}

        {selectedPerson && (
          <span style={styles.selectedPersonChip}>
            {selectedPerson}
            <button onClick={() => setSelectedPerson(null)} style={styles.clearPerson}>✕</button>
          </span>
        )}

        {/* Right: source filter — visually separated, always visible */}
        <div
          style={{
            ...styles.academicToggleBox,
            ...(includeAcademic ? styles.academicToggleBoxOn : {}),
          }}
          title="When on, includes LLM-extracted relationships from academic literature. Off shows only Princeton Genizah Project (PGP) verified data."
        >
          <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', margin: 0 }}>
            <input
              type="checkbox"
              checked={includeAcademic}
              onChange={e => setIncludeAcademic(e.target.checked)}
            />
            <span>
              <span style={{ fontWeight: 700, fontSize: 12 }}>
                {includeAcademic ? '📚 Academic sources on' : '📚 Academic sources'}
              </span>
              <span style={{ display: 'block', fontSize: 10, opacity: 0.7, lineHeight: 1.3, marginTop: 1 }}>
                {includeAcademic ? 'PGP + literature (LLM-inferred)' : 'PGP only (authoritative)'}
              </span>
            </span>
          </label>
        </div>

      </div>

      {/* ── Map ── */}
      <MapContainer center={[30, 35]} zoom={5} style={styles.map} zoomControl>
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>'
          subdomains="abcd"
          maxZoom={19}
        />

        <FitBounds places={places} institutions={institutions} />
        <PanToPerson markers={peopleMarkers} arcs={allJourneyArcs} selectedPerson={selectedPerson} />

        {/* Fragment-connection lines between places */}
        {showConnections && connections.map((c, i) => (
          <Polyline key={`conn-${i}`}
            positions={[[c.source_lat, c.source_lng], [c.target_lat, c.target_lng]]}
            pathOptions={{
              color: C.connection, weight: 1.5,
              opacity: connOpacity(c.connections, maxConnections),
            }}
            eventHandlers={{ click: () => {} }}
          >
            <Popup maxWidth={300} autoPan={false}>
              <ConnectionPopup
                conn={c}
                onFragmentClick={handleConnectionFragmentClick}
              />
            </Popup>
          </Polyline>
        ))}

        {/* Journey arcs — home → destination, shown for selected person or all when toggled */}
        {visibleArcs.map((a, i) => (
          <Polyline key={`arc-${i}`}
            positions={[a.from, a.to]}
            pathOptions={{
              color:   selectedPerson ? C.personArc : C.traveledTo,
              weight:  selectedPerson ? 2 : 1.5,
              opacity: selectedPerson ? 0.75 : 0.45,
            }}
          >
            <Popup>
              <strong>{a.person}</strong><br />
              {a.fromPlace} → {a.toPlace}
            </Popup>
          </Polyline>
        ))}

        {/* People markers — one per place, sized by headcount; popup lists people */}
        {showPeople && peopleByPlace.map((grp, i) => {
          const hasSelected = selectedPerson && grp.people.some(pm => pm.person === selectedPerson);
          return (
            <CircleMarker key={`people-${i}`}
              center={[grp.lat, grp.lng]}
              radius={markerRadius(grp.people.length)}
              pathOptions={{
                fillColor:   C.person,
                fillOpacity: 0.85,
                color:       hasSelected ? '#fff' : C.personBorder,
                weight:      hasSelected ? 2.5 : 1,
              }}
            >
              <Popup maxWidth={260} autoPan={false}>
                <PeoplePlacePopup
                  group={grp}
                  selectedPerson={selectedPerson}
                  onSelect={person => {
                    const next = selectedPerson === person ? null : person;
                    setSelectedPerson(next);
                    if (next) { setSelectedPlace(null); setSelectedInstitution(null); }
                  }}
                />
              </Popup>
            </CircleMarker>
          );
        })}

        {/* Cross-institution joined-fragment lines — purple dashed */}
        {showJoinedFragments && joinedFragments.map((j, i) => (
          <Polyline key={`join-${i}`}
            positions={[[j.source_lat, j.source_lng], [j.target_lat, j.target_lng]]}
            pathOptions={{
              color:     C.joinedFragments,
              weight:    2,
              opacity:   0.7,
              dashArray: '6 4',
            }}
          >
            <Popup maxWidth={320} autoPan={false}>
              <JoinedFragmentsPopup join={j} />
            </Popup>
          </Polyline>
        ))}

        {/* Institution markers — teal */}
        {showInstitutions && institutions.map((inst, i) => (
          <CircleMarker key={`inst-${i}`}
            center={[inst.lat, inst.lng]}
            radius={markerRadius(inst.fragment_count)}
            pathOptions={{
              fillColor: C.institution, fillOpacity: 0.85,
              color: selectedInstitution?.name === inst.name ? '#fff' : C.institutionBorder,
              weight: selectedInstitution?.name === inst.name ? 2.5 : 1.5,
            }}
            eventHandlers={{ click: () => handleInstitutionClick(inst) }}
          >
            <Popup>
              <strong>{inst.name}</strong><br />
              {inst.country && <span>{inst.country}</span>}<br />
              {inst.fragment_count} fragments held here
            </Popup>
          </CircleMarker>
        ))}

        {/* Historical place markers — amber */}
        {showPlaces && places.map((place, i) => (
          <CircleMarker key={`place-${i}`}
            center={[place.lat, place.lng]}
            radius={markerRadius(place.fragment_count)}
            pathOptions={{ fillColor: C.place, fillOpacity: 0.82,
              color: C.placeBorder, weight: 1 }}
            eventHandlers={{ click: () => handleMarkerClick(place) }}
          >
            <Popup>
              <strong>{place.name}</strong><br />
              {place.country && <span>{place.country}</span>}<br />
              {place.fragment_count} fragments · {place.person_count} people
            </Popup>
          </CircleMarker>
        ))}
      </MapContainer>

      {/* ── Legend ── */}
      <div style={styles.legend}>
        <div style={styles.legendRow}>
          <span style={{ ...styles.legendDot, background: C.place }} />
          Historical place
        </div>
        <div style={styles.legendRow}>
          <span style={{ ...styles.legendDot, background: C.institution }} />
          Institution
        </div>
        {showJoinedFragments && (
          <div style={styles.legendRow}>
            <span style={{ ...styles.legendLine, background: C.joinedFragments,
              backgroundImage: `repeating-linear-gradient(to right, ${C.joinedFragments} 0 6px, transparent 6px 10px)`,
              background: 'none', borderTop: `2px dashed ${C.joinedFragments}` }} />
            Joined fragments (cross-institution)
          </div>
        )}
        {showPeople && (
          <div style={styles.legendRow}>
            <span style={{ ...styles.legendDot, background: C.person }} />
            People (size = headcount)
          </div>
        )}
        {(showJourneys || selectedPerson) && (
          <div style={styles.legendRow}>
            <span style={{ ...styles.legendLine, background: C.traveledTo }} />
            Journey route
          </div>
        )}
        {showConnections && (
          <div style={styles.legendRow}>
            <span style={{ ...styles.legendLine, background: C.connection, opacity: 0.5 }} />
            Fragment connection
          </div>
        )}
      </div>

      {/* ── Side panels — only one visible at a time ── */}
      <PlacePanel
        place={selectedPlace}
        detail={placeDetail}
        onFragmentClick={handleFragmentClick}
        onClose={() => setSelectedPlace(null)}
      />
      <InstitutionPanel
        institution={selectedInstitution}
        detail={institutionDetail}
        onFragmentClick={f => setFragmentModal({ fragment: f, placeName: selectedInstitution?.name || '', nameVariants: '' })}
        onScholarClick={s => setSelectedScholar(s)}
        onClose={() => { setSelectedInstitution(null); setSelectedScholar(null); }}
      />
      <ScholarPanel
        scholarName={selectedScholar}
        detail={scholarDetail}
        onFragmentClick={f => setFragmentModal({ fragment: f, placeName: '', nameVariants: '' })}
        onBack={() => setSelectedScholar(null)}
        onClose={() => { setSelectedScholar(null); setSelectedInstitution(null); }}
      />
      <PersonPanel
        personName={selectedPerson}
        detail={personDetail}
        onFragmentClick={f => setFragmentModal({ fragment: f, placeName: '', nameVariants: '' })}
        onClose={() => { setSelectedPerson(null); }}
      />

      {/* ── Fragment modal ── */}
      {fragmentModal && (
        <FragmentModal
          fragment={fragmentModal.fragment}
          placeName={fragmentModal.placeName}
          nameVariants={fragmentModal.nameVariants}
          onClose={() => setFragmentModal(null)}
        />
      )}
    </div>
  );
}

// ─── Styles ──────────────────────────────────────────────────────────────────

const styles = {
  wrapper: {
    position: 'relative', width: '100%', height: '100vh',
    background: '#1a1a2e', display: 'flex', flexDirection: 'column',
  },
  map: { flex: 1, width: '100%' },

  // Toolbar
  toolbar: {
    display: 'flex', alignItems: 'center', gap: 12, padding: '8px 16px',
    background: '#111827', borderBottom: '1px solid #2a2a4a',
    zIndex: 1000, flexShrink: 0, overflowX: 'auto',
  },
  toolbarTitle: {
    color: C.place, fontWeight: 700, fontSize: 15,
    letterSpacing: '0.03em', marginRight: 'auto',
  },
  toggle: { color: '#ccc', fontSize: 13, cursor: 'pointer', display: 'flex', alignItems: 'center' },
  count:  { color: '#555', fontSize: 12, marginLeft: 'auto' },

  // Legend
  legend: {
    position: 'absolute', bottom: 28, left: 12,
    background: 'rgba(17,24,39,0.9)', border: '1px solid #2a2a4a',
    borderRadius: 6, padding: '8px 12px', zIndex: 1000,
    fontSize: 12, color: '#ccc',
  },
  legendRow:  { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 },
  legendDot:  { width: 11, height: 11, borderRadius: '50%', flexShrink: 0 },
  legendLine: { width: 20, height: 2, flexShrink: 0 },

  // Side panel
  panel: {
    position: 'absolute', top: 44, right: 0, width: 340,
    height: 'calc(100% - 44px)', background: '#111827',
    borderLeft: '1px solid #2a2a4a', overflowY: 'auto',
    padding: '20px 18px', zIndex: 1000, color: '#ddd',
  },
  panelClose:     { position: 'absolute', top: 12, right: 14, background: 'none', border: 'none', color: '#555', fontSize: 16, cursor: 'pointer' },
  panelTypeBadge: { display: 'inline-block', borderRadius: 10, border: '1px solid', padding: '2px 9px', fontSize: 11, fontWeight: 700, letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 8 },
  panelTitle:   { color: C.place, fontSize: 20, fontWeight: 700, margin: '0 0 4px' },
  tagsRow:      { display: 'flex', gap: 6, flexWrap: 'wrap', margin: '8px 0 4px' },
  tag:          { background: '#0f172a', border: '1px solid #2a3a5a', borderRadius: 10, padding: '2px 9px', fontSize: 11, color: '#94a3b8' },
  panelMeta:    { color: '#888', fontSize: 13, margin: '0 0 12px' },
  panelLoading: { color: '#555', fontSize: 13, marginTop: 16 },
  statRow:      { display: 'flex', gap: 12, marginBottom: 12 },
  stat:         { background: '#1e293b', borderRadius: 6, padding: '5px 11px', fontSize: 13, color: '#aaa' },
  variants:     { color: '#555', fontSize: 12, fontStyle: 'italic', marginBottom: 14 },
  section:      { marginTop: 18, borderTop: '1px solid #1e293b', paddingTop: 14 },
  sectionTitle: { color: '#9ca3af', fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', margin: '0 0 8px' },
  list:         { margin: 0, paddingLeft: 16, color: '#bbb', fontSize: 13, lineHeight: 1.7 },
  listItem:     { marginBottom: 2 },

  // Fragment cards (clickable)
  fragmentCard: {
    width: '100%', textAlign: 'left', background: '#1e293b',
    border: '1px solid transparent', borderRadius: 6, padding: '8px 10px',
    marginBottom: 8, cursor: 'pointer', transition: 'border-color 0.15s',
  },
  fragmentCardTop: { display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 },
  shelfmark:          { color: C.place, fontSize: 13, fontWeight: 600, fontFamily: 'monospace' },
  relation:           { color: '#6b7280', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.05em' },
  relationExplanation:{ color: '#6b7280', fontSize: 11, fontStyle: 'italic', margin: '3px 0 4px', lineHeight: 1.4 },
  fragmentMeta:       { display: 'flex', gap: 5, flexWrap: 'wrap', marginBottom: 5 },
  fragmentMetaChip:   { background: '#0f172a', borderRadius: 8, padding: '1px 7px', fontSize: 11, color: '#94a3b8' },
  sourceChip:         { background: '#1e3a2a', borderRadius: 8, padding: '1px 7px', fontSize: 10, color: '#6ee7a0', border: '1px solid #2a5a3a' },
  description:        { color: '#9ca3af', fontSize: 12, margin: '2px 0 4px', lineHeight: 1.5 },
  clickHint:          { color: '#374151', fontSize: 11, display: 'block', textAlign: 'right', marginTop: 2 },

  // Fragment modal
  modalOverlay: {
    position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.65)',
    zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center',
  },
  modal: {
    background: '#1a2236', border: '1px solid #2a3a5a', borderRadius: 10,
    padding: '24px 26px', width: 520, maxWidth: '90vw',
    maxHeight: '80vh', overflowY: 'auto', position: 'relative', color: '#ddd',
  },
  modalClose:     { position: 'absolute', top: 14, right: 16, background: 'none', border: 'none', color: '#555', fontSize: 18, cursor: 'pointer' },
  modalHeader:    { display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 },
  modalShelfmark: { color: C.place, fontSize: 18, fontWeight: 700, fontFamily: 'monospace' },
  modalRelation:  { color: '#6b7280', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.07em' },
  metaRow:        { display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 16 },
  metaChip:       { background: '#1e293b', borderRadius: 12, padding: '3px 10px', fontSize: 12, color: '#9ca3af' },
  modalLoading:   { color: '#555', fontSize: 13 },
  modalDescription: { marginBottom: 16 },
  modalDescLabel: { color: '#6b7280', fontSize: 10, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.08em', margin: '0 0 6px' },
  modalDescText:  { color: '#ccc', fontSize: 14, lineHeight: 1.7, margin: 0 },
  tagsRow:        { display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 16 },
  tag:            { background: '#0f172a', border: '1px solid #2a3a5a', borderRadius: 10, padding: '2px 9px', fontSize: 11, color: '#94a3b8' },
  modalActions:   { borderTop: '1px solid #2a3a5a', paddingTop: 14, display: 'flex', justifyContent: 'flex-end' },
  actionBtn:      { background: C.place, color: '#111', border: 'none', borderRadius: 6, padding: '7px 16px', fontSize: 13, fontWeight: 600, cursor: 'pointer' },

  toolbarDivider: { width: 1, height: 20, background: '#2a2a4a', flexShrink: 0 },
  toggleGroup: {
    display: 'flex', alignItems: 'center', gap: 12,
    background: '#1e293b', borderRadius: 6, padding: '4px 10px',
  },
  toggleGroupLabel: {
    color: '#4b5563', fontSize: 10, fontWeight: 700,
    textTransform: 'uppercase', letterSpacing: '0.08em', marginRight: 2,
  },
  academicToggleBox: {
    marginLeft: 'auto',
    background: '#1e293b',
    border: '1px solid #2a3a5a',
    borderRadius: 6,
    padding: '5px 10px',
    color: '#9ca3af',
    flexShrink: 0,
  },
  academicToggleBoxOn: {
    background: '#2a1f08',
    border: '1px solid #7a5a10',
    color: '#f0c060',
  },
  academicToggle: { color: '#9ca3af' },
  academicToggleOn: { color: '#f0c060', fontWeight: 600 },
  possibleBadge: {
    background: '#2a1a0a', color: '#f0a040', border: '1px solid #6b3a10',
    borderRadius: 8, padding: '1px 6px', fontSize: 10, fontStyle: 'italic',
  },
  academicBadge: {
    background: '#0f1a2a', color: '#6b9ed4', border: '1px solid #1a3a5a',
    borderRadius: 8, padding: '1px 6px', fontSize: 10,
  },
  pgpLink: {
    color: C.place, fontSize: 11, textDecoration: 'none', fontWeight: 600,
  },
  pgpLinkStandalone: {
    display: 'block', color: C.place, fontSize: 11, textDecoration: 'none',
    fontWeight: 600, textAlign: 'right', padding: '2px 0 6px',
    opacity: 0.8,
  },
  personRole:        { color: '#6b7280', fontSize: 12, fontStyle: 'italic' },
  personDescription: { color: '#9ca3af', fontSize: 12, lineHeight: 1.5, margin: '4px 0 12px', fontStyle: 'italic' },
  backLink: {
    background: 'none', border: 'none', color: '#6b7280', fontSize: 12,
    cursor: 'pointer', padding: '0 0 12px', display: 'block',
  },
  scholarBtn: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    width: '100%', background: '#1e293b', border: '1px solid #2a3a5a',
    borderRadius: 6, padding: '7px 10px', cursor: 'pointer',
    transition: 'border-color 0.15s',
  },
  scholarName:  { color: SCHOLAR_COLOR, fontSize: 13, fontWeight: 500 },
  scholarArrow: { color: '#374151', fontSize: 13 },
  bookCard: {
    background: '#1e293b', borderRadius: 6, padding: '10px 12px', marginBottom: 8,
  },
  bookTitle:  { color: '#e2e8f0', fontSize: 13, fontWeight: 600, margin: '0 0 6px', lineHeight: 1.4 },
  bookMeta:   { display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 4 },
  metaChip:   { background: '#0f172a', borderRadius: 8, padding: '2px 8px', fontSize: 11, color: '#94a3b8' },
  doiLink:    { color: '#7EB8D4', fontSize: 11, textDecoration: 'none' },
  doiText:    { color: '#6b7280', fontSize: 11 },
  pgpIdText:  { color: '#6b7280', fontSize: 11, display: 'block', textAlign: 'right', padding: '2px 0 4px' },
  backBtn: {
    background: 'none', border: '1px solid #2a2a4a', borderRadius: 6,
    color: '#9ca3af', fontSize: 13, padding: '4px 12px', cursor: 'pointer',
    flexShrink: 0,
  },
  peopleSearchWrap: { position: 'relative', flexShrink: 0 },
  peopleSearchInput: {
    background: '#1e293b', border: '1px solid #2a3a5a', borderRadius: 6,
    color: '#ddd', fontSize: 12, padding: '5px 9px', width: 160, outline: 'none',
  },
  peopleSearchDropdown: {
    position: 'absolute', top: 'calc(100% + 4px)', left: 0, width: 240,
    background: '#111827', border: '1px solid #2a3a5a', borderRadius: 6,
    maxHeight: 300, overflowY: 'auto', zIndex: 1100,
    boxShadow: '0 6px 20px rgba(0,0,0,0.45)',
  },
  peopleSearchResult: {
    display: 'block', width: '100%', textAlign: 'left',
    background: 'none', border: 'none', borderBottom: '1px solid #1e293b',
    padding: '6px 9px', cursor: 'pointer',
  },
  peopleSearchEmpty: { color: '#6b7280', fontSize: 12, padding: '8px 9px' },
  selectedPersonChip: {
    display: 'flex', alignItems: 'center', gap: 5,
    background: `${C.person}33`, border: `1px solid ${C.personBorder}`,
    borderRadius: 12, padding: '2px 8px 2px 10px',
    color: C.person, fontSize: 12, fontWeight: 600,
  },
  clearPerson: { background: 'none', border: 'none', color: C.personBorder, cursor: 'pointer', fontSize: 13, padding: 0 },

  centred: { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh', background: '#111827' },
  spinner: { width: 36, height: 36, border: '3px solid #2a2a4a', borderTop: `3px solid ${C.place}`, borderRadius: '50%', animation: 'spin 0.9s linear infinite' },
};
