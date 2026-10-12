import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './About.css';

/**
 * Placeholder copy for the About page, kept together in one place so the
 * owner can rewrite the real text without touching any JSX below. Every
 * placeholder string starts with "[Placeholder]" and is rendered inside an
 * element with className "about-placeholder" so it is visually obvious
 * (and easy to grep for) until it is replaced.
 */
const SECTIONS = {
  hero: {
    title: 'From Medieval Documents to 21st-Century Language Models',
    subtitle:
      '[Placeholder] One sentence framing the project for a first-time visitor — what it is and who it is for.',
  },
  whyThisProject: {
    heading: 'Why this project',
    body:
      '[Placeholder] Describe the original motivation for the project — how the owner came to the Cairo Genizah, ' +
      'what problem search and AI were meant to solve, and why now. A paragraph or two.',
  },
  aiForGenizah: {
    heading: 'What AI can do for the Cairo Genizah',
    intro:
      '[Placeholder] One or two sentences introducing this half of the story: applying modern AI to a historical archive.',
    blocks: [
      {
        heading: 'Finding: search by meaning',
        body:
          '[Placeholder] Explain semantic search over the Genizah collection — how embeddings let readers find ' +
          'documents by meaning rather than exact keywords, and why that matters for fragmentary medieval text.',
      },
      {
        heading: 'Reading: machine transcription',
        body:
          '[Placeholder] Explain the handwriting-recognition / transcription pipeline — what scripts and languages ' +
          'it covers, how confident it is, and how it is checked against human readers.',
      },
      {
        heading: 'Connecting: people, places and joins',
        body:
          '[Placeholder] Explain the knowledge graph — how it links people, places and related fragments (joins), ' +
          'and what questions it lets a researcher answer that keyword search alone could not.',
      },
      {
        heading: 'Asking: a research assistant grounded in scholarship',
        body:
          '[Placeholder] Explain the AI research assistant — what it is grounded in, its current limitations, and ' +
          'how it should and should not be used (see also the FAQ).',
      },
    ],
  },
  genizahForAi: {
    heading: 'What the Cairo Genizah can do for AI',
    intro:
      '[Placeholder] One or two sentences introducing the reverse direction: what this archive offers AI research.',
    blocks: [
      {
        heading: 'A hard, real benchmark for low-resource scripts and languages',
        body:
          '[Placeholder] Describe why Genizah documents (Hebrew, Arabic, Aramaic, Judeo-Arabic) are a demanding, ' +
          'real-world benchmark for low-resource script and language models.',
      },
      {
        heading: 'Damaged, fragmentary, multilingual data',
        body:
          '[Placeholder] Describe the challenges of damaged, torn, multilingual and code-switched manuscript pages, ' +
          'and why models that handle this data well are more broadly useful.',
      },
      {
        heading: 'Ground truth from a century of scholarship',
        body:
          '[Placeholder] Describe the scholarly transcriptions, editions and catalogue data that provide ground ' +
          'truth for training and evaluation, and how they were assembled.',
      },
      {
        heading: 'Honest evaluation: what models get wrong',
        body:
          '[Placeholder] Describe the project\'s approach to evaluation — reporting where models fail, not just ' +
          'where they succeed, and why that honesty matters for the field.',
      },
    ],
  },
  howItWorks: {
    heading: 'How it works',
    body:
      '[Placeholder] Short, high-level description of the data sources, the models used (embeddings, OCR/HTR, ' +
      'language models), and the pipeline that turns a manuscript image into a searchable, readable document.',
  },
  credits: {
    heading: 'Data, credits and acknowledgements',
    body:
      '[Placeholder] Credit named data partners and sources, including the Princeton Geniza Project (PGP), the ' +
      "National Library of Israel's KTIV project, and the holding libraries that provide manuscript images. " +
      'Add any other acknowledgements here.',
  },
  limitations: {
    heading: 'Limitations and how to cite',
    body:
      '[Placeholder] Describe known limitations (coverage, transcription accuracy, AI assistant reliability) and ' +
      'give a preferred citation format for referencing this project in scholarly work.',
  },
  contact: {
    heading: 'Contact',
    body:
      '[Placeholder] Give the preferred way to reach the owner (email, form, or social) for questions, ' +
      'corrections or collaboration proposals.',
  },
};

/**
 * In-page table of contents linking to each section by its anchor id.
 * @param {{items: Array<{id: string, label: string}>}} props - Sections to link to.
 * @returns {JSX.Element} A small nav listing anchor links.
 */
const TableOfContents = ({ items }) => (
  <nav className="about-toc" aria-label="On this page">
    <span className="about-toc-label">On this page:</span>
    <ul>
      {items.map((item) => (
        <li key={item.id}>
          <a href={`#${item.id}`}>{item.label}</a>
        </li>
      ))}
    </ul>
  </nav>
);

/**
 * One placeholder paragraph, visually marked as owner-editable copy.
 * @param {{children: React.ReactNode}} props - The placeholder text.
 * @returns {JSX.Element} A paragraph with className "about-placeholder".
 */
const Placeholder = ({ children }) => <p className="about-placeholder">{children}</p>;

/**
 * A sub-headed block used inside the "What AI can do for the Cairo Genizah"
 * and "What the Cairo Genizah can do for AI" sections.
 * @param {{heading: string, body: string}} props - Block heading and placeholder body text.
 * @returns {JSX.Element} A titled block with placeholder copy.
 */
const SubBlock = ({ heading, body }) => (
  <div className="about-subblock">
    <h3>{heading}</h3>
    <Placeholder>{body}</Placeholder>
  </div>
);

/**
 * About page for Cairo Genizah AI: explains the project from two angles —
 * what AI can do for the Cairo Genizah, and what the Cairo Genizah can do
 * for AI. All body copy is placeholder text for the owner to replace; see
 * the SECTIONS constant above.
 * @returns {JSX.Element} The rendered About page.
 */
const About = () => {
  const navigate = useNavigate();

  // Set the tab title while this page is mounted, restore it on the way out.
  useEffect(() => {
    document.title = 'About · Cairo Genizah AI';
    return () => {
      document.title = 'Cairo Genizah AI';
    };
  }, []);

  const tocItems = [
    { id: 'why-this-project', label: 'Why this project' },
    { id: 'ai-for-genizah', label: 'What AI can do for the Cairo Genizah' },
    { id: 'genizah-for-ai', label: 'What the Cairo Genizah can do for AI' },
    { id: 'how-it-works', label: 'How it works' },
    { id: 'credits', label: 'Data, credits and acknowledgements' },
    { id: 'limitations', label: 'Limitations and how to cite' },
    { id: 'contact', label: 'Contact' },
  ];

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <div className="header-left">
            <h1>Cairo Genizah AI</h1>
            <p>About this project</p>
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

      <main className="main-content about-main">
        <div className="about-container">
          <section id="hero" className="about-hero">
            <h2>{SECTIONS.hero.title}</h2>
            <p className="about-placeholder about-subtitle">{SECTIONS.hero.subtitle}</p>
          </section>

          <TableOfContents items={tocItems} />

          <section id="why-this-project" className="about-section">
            <h2>{SECTIONS.whyThisProject.heading}</h2>
            <Placeholder>{SECTIONS.whyThisProject.body}</Placeholder>
          </section>

          <section id="ai-for-genizah" className="about-section">
            <h2>{SECTIONS.aiForGenizah.heading}</h2>
            <Placeholder>{SECTIONS.aiForGenizah.intro}</Placeholder>
            <div className="about-subblock-grid">
              {SECTIONS.aiForGenizah.blocks.map((block) => (
                <SubBlock key={block.heading} heading={block.heading} body={block.body} />
              ))}
            </div>
          </section>

          <section id="genizah-for-ai" className="about-section">
            <h2>{SECTIONS.genizahForAi.heading}</h2>
            <Placeholder>{SECTIONS.genizahForAi.intro}</Placeholder>
            <div className="about-subblock-grid">
              {SECTIONS.genizahForAi.blocks.map((block) => (
                <SubBlock key={block.heading} heading={block.heading} body={block.body} />
              ))}
            </div>
          </section>

          <section id="how-it-works" className="about-section">
            <h2>{SECTIONS.howItWorks.heading}</h2>
            <Placeholder>{SECTIONS.howItWorks.body}</Placeholder>
          </section>

          <section id="credits" className="about-section">
            <h2>{SECTIONS.credits.heading}</h2>
            <Placeholder>{SECTIONS.credits.body}</Placeholder>
          </section>

          <section id="limitations" className="about-section">
            <h2>{SECTIONS.limitations.heading}</h2>
            <Placeholder>{SECTIONS.limitations.body}</Placeholder>
          </section>

          <section id="contact" className="about-section">
            <h2>{SECTIONS.contact.heading}</h2>
            <Placeholder>{SECTIONS.contact.body}</Placeholder>
          </section>
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
            <a href="/faq" onClick={(e) => { e.preventDefault(); navigate('/faq'); }}>FAQ</a>
            <a href="/docs" target="_blank" rel="noopener noreferrer">API Documentation</a>
            <a href="https://github.com/your-repo" target="_blank" rel="noopener noreferrer">GitHub</a>
            <a href="mailto:contact@example.com">Contact</a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default About;
