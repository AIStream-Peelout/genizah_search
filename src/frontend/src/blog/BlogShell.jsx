import React from 'react';
import { useNavigate } from 'react-router-dom';
import './Blog.css';

/**
 * Page frame shared by the blog index and the post pages: the site header
 * (same markup as the About page) and footer around a reading column. The
 * site name in the header is not an <h1>, so each page's own title is the
 * only <h1> on the page.
 * @param {{subtitle: string, children: React.ReactNode}} props - Header subtitle and page body.
 * @returns {JSX.Element} The framed page.
 */
const BlogShell = ({ subtitle, children }) => {
  const navigate = useNavigate();

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <div className="header-left">
            <p className="bl-site-name">Cairo Genizah AI</p>
            <p>{subtitle}</p>
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

      <main className="main-content bl-main">
        <div className="bl-container">{children}</div>
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
            <a href="/holidays" onClick={(e) => { e.preventDefault(); navigate('/holidays'); }}>Holiday archive</a>
            <a href="/faq" onClick={(e) => { e.preventDefault(); navigate('/faq'); }}>FAQ</a>
            <a href="/about" onClick={(e) => { e.preventDefault(); navigate('/about'); }}>About</a>
            <a href="https://github.com/AIStream-Peelout/genizah_search" target="_blank" rel="noopener noreferrer">GitHub</a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default BlogShell;
