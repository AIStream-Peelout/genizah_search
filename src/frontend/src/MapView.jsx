import React, { useState, useEffect, useCallback } from 'react';
import { MapContainer, TileLayer, CircleMarker, Polyline, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// ─── Styling constants ───────────────────────────────────────────────────────

const PLACE_COLOR        = '#C8962A'; // amber — historical places
const PLACE_COLOR_BORDER = '#8B6200';
const CONNECTION_COLOR   = '#C8962A';

/** Log-scale radius so high-count places don't dwarf everything else */
function markerRadius(fragmentCount) {
  if (!fragmentCount || fragmentCount === 0) return 5;
  return Math.min(5 + Math.log(fragmentCount + 1) * 3.5, 28);
}

/** Connection opacity scaled to connection strength */
function connectionOpacity(count, max) {
  if (!max) return 0.2;
  return 0.15 + (count / max) * 0.55;
}

// ─── Fit-bounds helper ───────────────────────────────────────────────────────

function FitBounds({ places }) {
  const map = useMap();
  useEffect(() => {
    if (!places.length) return;
    const lats = places.map(p => p.lat);
    const lngs = places.map(p => p.lng);
    map.fitBounds([
      [Math.min(...lats), Math.min(...lngs)],
      [Math.max(...lats), Math.max(...lngs)],
    ], { padding: [40, 40] });
  }, [places, map]);
  return null;
}

// ─── Place detail panel ──────────────────────────────────────────────────────

function PlacePanel({ place, detail, onClose }) {
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
        <span style={styles.stat}>
          <strong>{place.fragment_count}</strong> fragments
        </span>
        <span style={styles.stat}>
          <strong>{place.person_count}</strong> people
        </span>
      </div>

      {place.name_variants && (
        <p style={styles.variants}>
          Also known as: {place.name_variants}
        </p>
      )}

      {detail ? (
        <>
          {detail.people?.filter(Boolean).length > 0 && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>People connected here</h3>
              <ul style={styles.list}>
                {detail.people.filter(Boolean).map((p, i) => (
                  <li key={i} style={styles.listItem}>{p}</li>
                ))}
              </ul>
            </section>
          )}

          {detail.fragments?.filter(f => f.shelfmark).length > 0 && (
            <section style={styles.section}>
              <h3 style={styles.sectionTitle}>Fragments</h3>
              {detail.fragments.filter(f => f.shelfmark).map((f, i) => (
                <div key={i} style={styles.fragmentCard}>
                  <span style={styles.shelfmark}>{f.shelfmark}</span>
                  {f.relation && (
                    <span style={styles.relation}>
                      {f.relation.replace(/_/g, ' ').toLowerCase()}
                    </span>
                  )}
                  {f.description && (
                    <p style={styles.description}>
                      {f.description.length > 120
                        ? f.description.slice(0, 120) + '…'
                        : f.description}
                    </p>
                  )}
                </div>
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
      ) : (
        <p style={styles.loading}>Loading details…</p>
      )}
    </div>
  );
}

// ─── Main component ──────────────────────────────────────────────────────────

export default function MapView() {
  const [places, setPlaces]           = useState([]);
  const [connections, setConnections] = useState([]);
  const [selectedPlace, setSelectedPlace] = useState(null);
  const [placeDetail, setPlaceDetail]     = useState(null);
  const [showConnections, setShowConnections] = useState(true);
  const [loading, setLoading]         = useState(true);
  const [error, setError]             = useState(null);

  // Load all places + connections on mount
  useEffect(() => {
    async function loadMapData() {
      try {
        const [placesRes, connectionsRes] = await Promise.all([
          fetch(`${API_BASE_URL}/map/places`),
          fetch(`${API_BASE_URL}/map/connections?min_connections=3`),
        ]);

        if (!placesRes.ok) throw new Error(`Places API error: ${placesRes.status}`);
        if (!connectionsRes.ok) throw new Error(`Connections API error: ${connectionsRes.status}`);

        const placesData      = await placesRes.json();
        const connectionsData = await connectionsRes.json();

        setPlaces(placesData.places || []);
        setConnections(connectionsData.connections || []);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    loadMapData();
  }, []);

  // Load detail when a place is selected
  useEffect(() => {
    if (!selectedPlace) { setPlaceDetail(null); return; }
    setPlaceDetail(null);
    fetch(`${API_BASE_URL}/map/place/${encodeURIComponent(selectedPlace.name)}`)
      .then(r => r.json())
      .then(setPlaceDetail)
      .catch(console.error);
  }, [selectedPlace]);

  const handleMarkerClick = useCallback((place) => {
    setSelectedPlace(place);
  }, []);

  const maxConnections = connections.length
    ? Math.max(...connections.map(c => c.connections))
    : 1;

  if (loading) {
    return (
      <div style={styles.centred}>
        <div style={styles.spinner} />
        <p style={{ color: '#C8962A', marginTop: 16 }}>Loading map data…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={styles.centred}>
        <p style={{ color: '#e55' }}>Failed to load map: {error}</p>
      </div>
    );
  }

  return (
    <div style={styles.wrapper}>
      {/* Toolbar */}
      <div style={styles.toolbar}>
        <span style={styles.toolbarTitle}>Cairo Genizah — Places</span>
        <label style={styles.toggle}>
          <input
            type="checkbox"
            checked={showConnections}
            onChange={e => setShowConnections(e.target.checked)}
            style={{ marginRight: 6 }}
          />
          Show connections
        </label>
        <span style={styles.count}>{places.length} places · {connections.length} connections</span>
      </div>

      {/* Legend */}
      <div style={styles.legend}>
        <div style={styles.legendRow}>
          <span style={{ ...styles.legendDot, background: PLACE_COLOR }} />
          Historical place (sized by fragment count)
        </div>
        <div style={styles.legendRow}>
          <span style={{ ...styles.legendLine, background: CONNECTION_COLOR }} />
          Fragment connection (origin → mention)
        </div>
      </div>

      <MapContainer
        center={[30, 35]}
        zoom={5}
        style={styles.map}
        zoomControl={true}
      >
        <TileLayer
          url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          subdomains="abcd"
          maxZoom={19}
        />

        <FitBounds places={places} />

        {/* Connections layer */}
        {showConnections && connections.map((c, i) => (
          <Polyline
            key={i}
            positions={[
              [c.source_lat, c.source_lng],
              [c.target_lat, c.target_lng],
            ]}
            pathOptions={{
              color: CONNECTION_COLOR,
              weight: 1.2,
              opacity: connectionOpacity(c.connections, maxConnections),
            }}
          />
        ))}

        {/* Place markers */}
        {places.map((place, i) => (
          <CircleMarker
            key={i}
            center={[place.lat, place.lng]}
            radius={markerRadius(place.fragment_count)}
            pathOptions={{
              fillColor: PLACE_COLOR,
              fillOpacity: 0.82,
              color: PLACE_COLOR_BORDER,
              weight: 1,
            }}
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

      {/* Side panel */}
      <PlacePanel
        place={selectedPlace}
        detail={placeDetail}
        onClose={() => setSelectedPlace(null)}
      />
    </div>
  );
}

// ─── Styles ──────────────────────────────────────────────────────────────────

const styles = {
  wrapper: {
    position: 'relative',
    width: '100%',
    height: '100vh',
    background: '#1a1a2e',
    display: 'flex',
    flexDirection: 'column',
  },
  map: {
    flex: 1,
    width: '100%',
  },
  toolbar: {
    display: 'flex',
    alignItems: 'center',
    gap: 20,
    padding: '10px 18px',
    background: '#111827',
    borderBottom: '1px solid #2a2a4a',
    zIndex: 1000,
    flexShrink: 0,
  },
  toolbarTitle: {
    color: '#C8962A',
    fontWeight: 700,
    fontSize: 16,
    letterSpacing: '0.03em',
    marginRight: 'auto',
  },
  toggle: {
    color: '#ccc',
    fontSize: 13,
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
  },
  count: {
    color: '#666',
    fontSize: 12,
  },
  legend: {
    position: 'absolute',
    bottom: 28,
    left: 12,
    background: 'rgba(17,24,39,0.88)',
    border: '1px solid #2a2a4a',
    borderRadius: 6,
    padding: '8px 12px',
    zIndex: 1000,
    fontSize: 12,
    color: '#ccc',
  },
  legendRow: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    marginBottom: 4,
  },
  legendDot: {
    width: 12,
    height: 12,
    borderRadius: '50%',
    flexShrink: 0,
  },
  legendLine: {
    width: 20,
    height: 2,
    flexShrink: 0,
    opacity: 0.7,
  },
  panel: {
    position: 'absolute',
    top: 52,
    right: 0,
    width: 340,
    height: 'calc(100% - 52px)',
    background: '#111827',
    borderLeft: '1px solid #2a2a4a',
    overflowY: 'auto',
    padding: '20px 18px',
    zIndex: 1000,
    color: '#ddd',
  },
  panelClose: {
    position: 'absolute',
    top: 12,
    right: 14,
    background: 'none',
    border: 'none',
    color: '#666',
    fontSize: 16,
    cursor: 'pointer',
  },
  panelTitle: {
    color: '#C8962A',
    fontSize: 20,
    fontWeight: 700,
    margin: '0 0 4px',
  },
  panelMeta: {
    color: '#888',
    fontSize: 13,
    margin: '0 0 12px',
  },
  statRow: {
    display: 'flex',
    gap: 16,
    marginBottom: 12,
  },
  stat: {
    background: '#1e293b',
    borderRadius: 6,
    padding: '6px 12px',
    fontSize: 13,
    color: '#aaa',
  },
  variants: {
    color: '#666',
    fontSize: 12,
    fontStyle: 'italic',
    marginBottom: 16,
  },
  section: {
    marginTop: 20,
    borderTop: '1px solid #1e293b',
    paddingTop: 16,
  },
  sectionTitle: {
    color: '#9ca3af',
    fontSize: 11,
    fontWeight: 700,
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
    margin: '0 0 10px',
  },
  list: {
    margin: 0,
    paddingLeft: 16,
    color: '#bbb',
    fontSize: 13,
    lineHeight: 1.7,
  },
  listItem: {
    marginBottom: 2,
  },
  fragmentCard: {
    background: '#1e293b',
    borderRadius: 6,
    padding: '8px 10px',
    marginBottom: 8,
  },
  shelfmark: {
    color: '#C8962A',
    fontSize: 13,
    fontWeight: 600,
    fontFamily: 'monospace',
  },
  relation: {
    marginLeft: 8,
    color: '#6b7280',
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  description: {
    color: '#9ca3af',
    fontSize: 12,
    margin: '6px 0 0',
    lineHeight: 1.5,
  },
  loading: {
    color: '#555',
    fontSize: 13,
    marginTop: 20,
  },
  centred: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    height: '100vh',
    background: '#111827',
  },
  spinner: {
    width: 36,
    height: 36,
    border: '3px solid #2a2a4a',
    borderTop: '3px solid #C8962A',
    borderRadius: '50%',
    animation: 'spin 0.9s linear infinite',
  },
};
