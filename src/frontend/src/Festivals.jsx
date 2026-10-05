import React, { useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import './Festivals.css';

/**
 * Festival pages kept online after their festival, newest first.
 * Each entry links to a route that still exists in react_app.jsx.
 */
const FESTIVAL_PAGES = [
  {
    href: '/sukkot',
    title: 'Exploring Sukkot through the Cairo Genizah',
    when: 'Sukkot 5787, 26 Sep – 4 Oct 2026',
    summary:
      '73 fragments across 16 themes: hoshanot and the Day of the Willow, the Mount of Olives assembly, ' +
      'palm branches and citrons, the synagogue sukkah, Simhat Torah and the calendar dispute of 921.',
  },
  {
    href: '/yom-kippur',
    title: 'Yom Kippur in the Cairo Genizah',
    when: 'Yom Kippur 5787, 20–21 Sep 2026',
    summary: 'Thirteen fragments of Yom Kippur liturgy with machine readings shown on the manuscript.',
  },
];

/**
 * One card in the festival archive, linking to a festival page.
 * @param {{href: string, title: string, when: string, summary: string}} props - The page to link to.
 * @returns {JSX.Element} A linked card with title, date line and summary.
 */
const FestivalCard = ({ href, title, when, summary }) => (
  <li className="fv-card">
    <h2 className="fv-card-title">
      <Link to={href}>{title}</Link>
    </h2>
    <p className="fv-card-when">{when}</p>
    <p className="fv-card-summary">{summary}</p>
    <Link to={href} className="fv-card-link">Open the page →</Link>
  </li>
);

/**
 * Festival archive at /festivals: an index of the festival pages built in
 * past years, which stay online as a record after the festival ends.
 * @returns {JSX.Element} The rendered festival archive page.
 */
const Festivals = () => {
  const navigate = useNavigate();

  // Set the tab title while this page is mounted, restore it on the way out.
  useEffect(() => {
    document.title = 'Festival archive · Cairo Genizah AI';
    return () => {
      document.title = 'Cairo Genizah AI';
    };
  }, []);

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <div className="header-left">
            <h1>Cairo Genizah AI</h1>
            <p>Festival archive</p>
          </div>
          <div className="header-right">
            <button
              onClick={() => navigate('/')}
              className="browser-btn"
              style={{ marginRight: 0 }}
            >
              ← Back to Search
            </button>
          </div>
        </div>
      </header>

      <main className="main-content fv-main">
        <div className="fv-container">
          <h2 className="fv-title">Festival pages from past years</h2>
          <p className="fv-lead">
            Pages built for a festival stay online as a record; each gathers Genizah fragments on that
            festival&apos;s prayers, customs and documents.
          </p>
          <ul className="fv-list">
            {FESTIVAL_PAGES.map((page) => (
              <FestivalCard key={page.href} {...page} />
            ))}
          </ul>
        </div>
      </main>

      <footer className="app-footer">
        <div className="footer-content">
          <p>
            Cairo Genizah AI • cairogenizah.ai • Built on AI and historical scholarship
          </p>
          <p>
            Special thanks to the <a href="https://geniza.princeton.edu/en/"> Princeton Cairo Genizah Project</a> (PGP)
          </p>
          <div className="footer-links">
            <a href="/blog" onClick={(e) => { e.preventDefault(); navigate('/blog'); }}>Blog</a>
            <a href="/faq" onClick={(e) => { e.preventDefault(); navigate('/faq'); }}>FAQ</a>
            <a href="/about" onClick={(e) => { e.preventDefault(); navigate('/about'); }}>About</a>
            <a href="https://github.com/AIStream-Peelout/genizah_search" target="_blank" rel="noopener noreferrer">GitHub</a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Festivals;
