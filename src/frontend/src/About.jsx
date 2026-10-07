import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './About.css';

/**
 * Copy for the About page, kept together in one place so the owner can edit
 * the text without touching any JSX below. A body string may hold several
 * paragraphs separated by a blank line ("\n\n").
 */
const SECTIONS = {
  hero: {
    title: 'From Medieval Documents to 21st-Century Language Models',
    subtitle:
      'Cairo Genizah AI is an independent website for searching, reading and exploring more than 75,000 ' +
      'Cairo Genizah fragments with AI, for scholars, rabbis, students and curious readers alike.',
  },
  whyThisProject: {
    heading: 'Why this project',
    body:
      'The Cairo Genizah is one of the largest bodies of medieval Jewish writing to survive: over 400,000 ' +
      'fragments, now spread across many libraries, ranging from liturgy and literature to legal documents, ' +
      'commercial records and personal letters. Excellent catalogues exist, but finding material still ' +
      'usually means knowing the right shelfmark, keyword or catalogue.' +
      '\n\n' +
      'The idea behind this site is a single, general-purpose search across that catalogue data, where a ' +
      'question such as "dowry disputes in Fustat" finds relevant fragments whichever library holds them. AI ' +
      'makes that possible, and raises further questions worth testing in public: can a machine help read the ' +
      'handwriting, connect fragments to people and places, and answer questions from the published ' +
      'scholarship? This site is a working attempt to find out, and to say plainly where the answer is still no.',
  },
  aiForGenizah: {
    heading: 'What AI can do for the Cairo Genizah',
    intro:
      'Applying AI to a historical archive is not about replacing scholars. It is about making the existing ' +
      'catalogues, images and scholarship easier to find, read and connect.',
    blocks: [
      {
        heading: 'Finding: search by meaning',
        body:
          'Semantic search compares the meaning of your query with the meaning of each catalogue record, using ' +
          'text embeddings from the Qwen3-Embedding-0.6B model, so a theme can surface records that describe it ' +
          'in other words. Keyword, shelfmark and hybrid modes cover exact phrases and known fragments. The ' +
          'embeddings are made from the catalogue text, not from the handwriting on the page.',
      },
      {
        heading: 'Reading: machine transcription',
        body:
          'Two automatic readers, a fine-tuned vision-language model and the Kraken HTR model, read each page ' +
          'image separately, and the result can be searched and shown line by line on the manuscript. A line is ' +
          'marked as agreed only when both produce the same text, which means the readers concur, not that the ' +
          'line is right. These AI transcriptions are a beta, cover only a small part of the collection so far, ' +
          'and have not been checked by a person.',
      },
      {
        heading: 'Connecting: people, places and joins',
        body:
          'A knowledge graph links fragments to the people, places, works and institutions they are associated ' +
          'with. It drives the places map, which ties fragments to historical places and holding institutions ' +
          'and shows joins: pieces of one original manuscript now held in different libraries.',
      },
      {
        heading: 'Asking: a research assistant grounded in scholarship',
        body:
          'The research assistant answers questions in plain language. It first retrieves passages from an ' +
          'index of published scholarship and entries from the knowledge graph, then writes an answer that ' +
          'cites its sources and links to the manuscripts it mentions, and finally checks its own claims ' +
          'against what it retrieved, flagging any it cannot support. It is experimental and can be wrong: ' +
          'verify every claim against the cited source.',
      },
    ],
  },
  genizahForAi: {
    heading: 'What the Cairo Genizah can do for AI',
    intro:
      'The traffic runs both ways. For AI research, the Genizah combines real, messy, historically important ' +
      'documents with generations of expert scholarship to check the machines against.',
    blocks: [
      {
        heading: 'A hard, real benchmark for low-resource scripts and languages',
        body:
          'Genizah fragments are written in Hebrew, Aramaic, Arabic and Judeo-Arabic (Arabic in Hebrew script), ' +
          'in many hands over many centuries. Current models see little material like this in training, which ' +
          'makes the Genizah a demanding test drawn from real documents rather than a synthetic one.',
      },
      {
        heading: 'Damaged, fragmentary, multilingual data',
        body:
          'Pages are torn, stained, faded and trimmed; lines break off mid-word, and one page can switch ' +
          'between languages, scripts and hands. A model has to work with missing context and know when not to ' +
          'guess, skills that carry over to other damaged, under-resourced archives.',
      },
      {
        heading: 'Ground truth from a century of scholarship',
        body:
          'Generations of scholars have catalogued, transcribed, edited and studied Genizah fragments. ' +
          'Their catalogue records, transcriptions, editions and studies give reference points for training and ' +
          'evaluating models, and for checking what an AI assistant says.',
      },
      {
        heading: 'Honest evaluation: what models get wrong',
        body:
          'Knowing where models fail matters as much as knowing where they succeed. Here, both machine readers ' +
          'share small letter confusions (ד/ר, ב/כ, ם/ס), the vision model invents plausible words where the ' +
          'page is damaged, and the search embeddings reflect genre and language more than subject, so a ' +
          'festival poem whose catalogue record never names the festival will not cluster with the rest. The ' +
          'site states these limits next to the results instead of hiding them.',
      },
    ],
  },
  howItWorks: {
    heading: 'How it works',
    body:
      "Catalogue records from the Princeton Geniza Project and the National Library of Israel's KTIV project " +
      'are indexed in Elasticsearch with Qwen3-Embedding-0.6B embeddings; the same embeddings drive the ' +
      'Collection Explorer (a t-SNE or UMAP map of the collection), and festival pages such as Yom Kippur and ' +
      'Sukkot are built on the same records. Machine readings are made offline and kept in a separate index, ' +
      'so they never change the ranking of catalogue results. A Neo4j knowledge graph of people, places, works ' +
      'and fragments feeds the map and the research assistant. No commercial AI service is involved: the ' +
      'assistant uses open-weight models on a single Mac Studio, and the rest of the site runs on the ' +
      "project's own local hardware, which is also why answers can be slow at busy times (see the FAQ).",
  },
  credits: {
    heading: 'Data, credits and acknowledgements',
    body:
      "Catalogue data comes from the Princeton Geniza Project (PGP) and the National Library of Israel's KTIV " +
      'project; this site would not exist without their work. Manuscript images are courtesy of the holding ' +
      'libraries, among them Cambridge University Library, the Jewish Theological Seminary, the Bodleian ' +
      'Libraries, the Library at the Katz Center (University of Pennsylvania), the John Rylands Library in ' +
      'Manchester and the Library of the Alliance Israélite Universelle in Paris. Images of items published ' +
      'through OPenn (Penn Libraries) are in the public domain. The published scholarship the assistant ' +
      'draws on belongs to its authors.' +
      '\n\n' +
      'The site also relies on open-source software and open-weight models, including Kraken, the Qwen ' +
      'models, Elasticsearch, Neo4j, Leaflet and the Mirador viewer. Its own code is open source at ' +
      'https://github.com/AIStream-Peelout/genizah_search, with the indexing and knowledge-graph code at ' +
      'https://github.com/AIStream-Peelout/historical-document-analysis.' +
      '\n\n' +
      'Cairo Genizah AI is an independent project run by one person. It is not affiliated with or endorsed by ' +
      'the Princeton Geniza Project, the National Library of Israel or any holding library.',
  },
  limitations: {
    heading: 'Limitations and how to cite',
    body:
      'Coverage is partial: the site indexes more than 70,000 catalogue records, not every Genizah fragment, ' +
      'and search is only as good as the catalogue description behind each record. Machine transcriptions ' +
      'cover a small share of images and have not been checked by a person. The research assistant can be ' +
      'wrong or incomplete even when it cites a source.' +
      '\n\n' +
      'To cite this site: Cairo Genizah AI, cairogenizah.ai, accessed <date>. For a fragment itself, cite the ' +
      "holding library's shelfmark and the relevant PGP or KTIV catalogue record rather than this site, and " +
      'if you quote a machine transcription, label it as an unchecked machine reading..',
  },
  contact: {
    heading: 'Contact',
    /* OWNER: add a contact email address (or form) to this paragraph, and to the footer's Contact link. */
    body:
      'Questions, corrections and collaboration proposals are welcome. For a bug, a problem with a record on ' +
      'this site or a feature request, please open an issue on GitHub at ' +
      'https://github.com/AIStream-Peelout/genizah_search/issues. Corrections to the underlying catalogue ' +
      'data are best sent to the catalogue that publishes the record (PGP or KTIV).',
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
 * Body copy for one section or sub-block. The text is split on blank lines
 * ("\n\n") into separate paragraphs.
 * @param {{children: string}} props - The body text.
 * @returns {JSX.Element} One or more paragraphs.
 */
const Prose = ({ children }) => (
  <>
    {String(children).split('\n\n').map((paragraph, i) => (
      <p key={i} style={{ lineHeight: 1.7, marginTop: i > 0 ? '1rem' : 0 }}>
        {paragraph}
      </p>
    ))}
  </>
);

/**
 * A sub-headed block used inside the "What AI can do for the Cairo Genizah"
 * and "What the Cairo Genizah can do for AI" sections.
 * @param {{heading: string, body: string}} props - Block heading and body text.
 * @returns {JSX.Element} A titled block of body copy.
 */
const SubBlock = ({ heading, body }) => (
  <div className="about-subblock">
    <h3>{heading}</h3>
    <Prose>{body}</Prose>
  </div>
);

/**
 * About page for Cairo Genizah AI: explains the project from two angles —
 * what AI can do for the Cairo Genizah, and what the Cairo Genizah can do
 * for AI. All body copy lives in the SECTIONS constant above.
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
            <p className="about-subtitle">{SECTIONS.hero.subtitle}</p>
          </section>

          <TableOfContents items={tocItems} />

          <section id="why-this-project" className="about-section">
            <h2>{SECTIONS.whyThisProject.heading}</h2>
            <Prose>{SECTIONS.whyThisProject.body}</Prose>
          </section>

          <section id="ai-for-genizah" className="about-section">
            <h2>{SECTIONS.aiForGenizah.heading}</h2>
            <Prose>{SECTIONS.aiForGenizah.intro}</Prose>
            <div className="about-subblock-grid">
              {SECTIONS.aiForGenizah.blocks.map((block) => (
                <SubBlock key={block.heading} heading={block.heading} body={block.body} />
              ))}
            </div>
          </section>

          <section id="genizah-for-ai" className="about-section">
            <h2>{SECTIONS.genizahForAi.heading}</h2>
            <Prose>{SECTIONS.genizahForAi.intro}</Prose>
            <div className="about-subblock-grid">
              {SECTIONS.genizahForAi.blocks.map((block) => (
                <SubBlock key={block.heading} heading={block.heading} body={block.body} />
              ))}
            </div>
          </section>

          <section id="how-it-works" className="about-section">
            <h2>{SECTIONS.howItWorks.heading}</h2>
            <Prose>{SECTIONS.howItWorks.body}</Prose>
          </section>

          <section id="credits" className="about-section">
            <h2>{SECTIONS.credits.heading}</h2>
            <Prose>{SECTIONS.credits.body}</Prose>
          </section>

          <section id="limitations" className="about-section">
            <h2>{SECTIONS.limitations.heading}</h2>
            <Prose>{SECTIONS.limitations.body}</Prose>
          </section>

          <section id="contact" className="about-section">
            <h2>{SECTIONS.contact.heading}</h2>
            <Prose>{SECTIONS.contact.body}</Prose>
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
            <a href="https://github.com/AIStream-Peelout/genizah_search" target="_blank" rel="noopener noreferrer">GitHub</a>
            {/* OWNER: swap for a mailto: once there is a public contact address. */}
            <a href="https://github.com/AIStream-Peelout/genizah_search/issues" target="_blank" rel="noopener noreferrer">Contact</a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default About;
