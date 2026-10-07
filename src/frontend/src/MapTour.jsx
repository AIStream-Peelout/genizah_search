import React from 'react';

// localStorage key marking that the user has seen (or skipped) the map tour.
export const MAP_TOUR_SEEN_KEY = 'genizah_map_tour_seen';

/**
 * Steps for the Places Map walkthrough, rendered by GuidedTour. Selectors
 * point at `data-tour` anchors in MapView; steps without selectors show a
 * centered card.
 */
export const MAP_TOUR_STEPS = [
  {
    id: 'map-welcome',
    title: 'The Genizah world on a map',
    body: (
      <>
        <p>
          Every gold dot is a historical place named in the fragments — Fustat,
          Alexandria, Qayrawan, Aden — sized by how many documents mention it.
          Green dots are the libraries that hold the fragments today.
        </p>
        <p>This short tour shows what the map can do.</p>
      </>
    ),
    selectors: [],
  },
  {
    id: 'map-layers',
    title: 'Switch layers on and off',
    body: (
      <ul>
        <li><strong>Places</strong> and <strong>Institutions</strong> — the dots.</li>
        <li><strong>Connections</strong> — lines between places that share fragments.</li>
        <li><strong>Joins</strong> — dashed lines linking pieces of one document held in different libraries.</li>
        <li><strong>People</strong> and <strong>Journeys</strong> — individuals and the routes they lived along or travelled.</li>
      </ul>
    ),
    selectors: ['[data-tour="map-layers"]'],
  },
  {
    id: 'map-people',
    title: 'Find a person',
    body: (
      <p>
        Turn on <strong>People</strong>, then search a name to see where that
        person lived and travelled. Pick one and the map pans to them; add
        <strong> Journeys</strong> to draw their movements as arcs.
      </p>
    ),
    selectors: ['[data-tour="map-people"]', '[data-tour="map-layers"]'],
  },
  {
    id: 'map-click',
    title: 'Click anything for details',
    body: (
      <p>
        Click a place, institution, or person to open a side panel listing the
        fragments behind it. Every fragment links to its full record, and
        connection lines explain which documents tie two places together.
      </p>
    ),
    selectors: ['[data-tour="map-legend"]'],
  },
  {
    id: 'map-sources',
    title: 'PGP only vs. academic sources',
    body: (
      <>
        <p>
          <strong>PGP only</strong> (default): everything on the map comes from
          the Princeton Geniza Project’s catalogue — places, people, and links
          recorded by its editors from the documents themselves. Authoritative,
          but only as complete as the catalogue.
        </p>
        <p>
          <strong>Academic sources on</strong>: adds relationships a language
          model extracted from published scholarship (Goitein and later
          studies). This surfaces many more places and people, at the cost of
          being machine-inferred: entries carry an <em>academic</em> badge and
          uncertain links are marked <em>possible</em>. Counts and connections
          change when you toggle it.
        </p>
        <p className="gt-note">
          Use PGP only when you need citable, verified data; switch academic
          sources on to explore, then check claims against the linked fragments.
        </p>
      </>
    ),
    selectors: ['[data-tour="map-sources"]'],
  },
  {
    id: 'map-done',
    title: 'Start exploring',
    body: (
      <p>
        Replay this walkthrough anytime with the 🎓 Tour button in the toolbar,
        or go back to Search with the arrow on the left.
      </p>
    ),
    selectors: ['[data-tour="map-tour-button"]'],
  },
];
