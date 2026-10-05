/**
 * Blog posts: SEO landing pages that summarise articles published on Medium
 * and link to the full text. Newest first. The metaTitle and description of
 * each post are mirrored in nginx.conf (the $page_title / $page_desc maps)
 * so crawlers see them without running JavaScript; keep the two in step.
 *
 * Fields: slug, title, metaTitle, description (meta description, at most
 * 160 characters), mediumUrl, publication, author, datePublished (ISO date),
 * readTime, keywords, lead, sections ({heading, paragraphs}), keyTerms,
 * links ({label, href, external}) and related ({label, href}, internal).
 */

const PUBLICATION = "Deep Data Science on Medium";
const AUTHOR = "Isaac Godfried";

const TRANSCRIBING_URL =
  "https://medium.com/deep-data-science/transcribing-the-cairo-genizah-with-multi-modal-ai-ad4cd9cbe980";
const DISCOVERY_URL =
  "https://medium.com/deep-data-science/multi-modal-ai-for-manuscript-discovery-analysis-searching-and-clustering-the-cairo-genizah-7879e6167374";

export const POSTS = [
  {
    slug: "transcribing-the-cairo-genizah-with-multimodal-ai",
    title: "Transcribing the Cairo Genizah with Multimodal AI",
    metaTitle: "Transcribing the Cairo Genizah with Multimodal AI: AI transcription of medieval manuscripts",
    description:
      "How Cairo Genizah fragments are transcribed with multimodal AI: fine-tuned vision-language models for Hebrew and Judeo-Arabic, Kraken HTR, two-reader consensus.",
    mediumUrl: TRANSCRIBING_URL,
    publication: PUBLICATION,
    author: AUTHOR,
    datePublished: "2026-10-05",
    readTime: "19 min read",
    keywords: [
      "Cairo Genizah",
      "AI transcription",
      "transcribing the Cairo Genizah",
      "multimodal AI",
      "vision-language models",
      "Hebrew OCR",
      "Judeo-Arabic",
      "Kraken HTR",
      "Qwen3-VL",
      "handwritten text recognition",
    ],
    lead:
      "Only about 15% of the Cairo Genizah has ever been transcribed. This article asks whether multimodal AI can read the rest, what goes wrong when vision-language models meet medieval Hebrew, and how a two-reader pipeline turns imperfect machine transcription into something a search engine can use.",
    sections: [
      {
        heading: "Why transcription is the bottleneck",
        paragraphs: [
          "The Cairo Genizah is one of the richest collections of right-to-left medieval documents in existence, but without transcriptions, keyword and semantic search are limited to catalogue descriptions and text-only language models cannot cite the source text of a fragment. For non-specialists, even reading the characters on a damaged page is hard. Transcribing the Cairo Genizah is therefore the single biggest lever for making the collection searchable.",
          "From the AI side the collection is a demanding test bed: how much training data does a vision-language model (VLM) need to generalise to scripts and languages it has effectively never seen, such as Judeo-Arabic and variants of Rashi script, and how deep must fine-tuning go when the usual recipe of freezing the vision tower is not enough?",
        ],
      },
      {
        heading: "OCR models versus vision-language models",
        paragraphs: [
          "Classic optical character recognition classifies glyphs line by line. Modern OCR models such as the Kraken HTR system use convolutional and recurrent networks and a narrow language prior, so they may garble characters but they do not invent plausible prose. Vision-language models understand document layout and can return structured output, which has made them the default for document transcription in industry, but their strong language prior has side effects on historical text.",
          "There are almost no benchmarks for right-to-left documents: OCRBench v2 is English and Chinese, MDPBench includes a little modern Arabic, and KITAB-Bench is modern Arabic only. So the article first builds a benchmark on the Vilna edition of the Talmud, whose pages combine block letters and Rashi script in a demanding multi-column layout, before moving to the Genizah itself.",
        ],
      },
      {
        heading: "Four ways vision-language models fail on Hebrew manuscripts",
        paragraphs: [
          "Abstention: the model declines to read a page it cannot handle, the safest failure. Loop collapse: open models such as Qwen-VL and Gemma repeat the same token or line until the token limit, a form of neural text degeneration that is obvious to a reader. Defensive hallucination: an earlier Gemini model, asked for the Rashi column of Berakhot 2a, composed a plausible but invented commentary from other parts of the page, then defended it when shown the real text. Canonical interpolation: the subtlest failure, where the model fills in what the text should say from memory. On the divorce deed T-S J2.3, written in al-Ramla in 1026, a model transcribed the Babylonian formula כדת משה וישראל where the page carries the Palestinian ויהודאיי, erasing exactly the scribal variation that historians study.",
          "The article argues that canonical interpolation is far more dangerous than loops or refusals, because it produces fluent, genre-appropriate text that is wrong in ways only an expert would catch.",
        ],
      },
      {
        heading: "Fine-tuning Qwen3-VL for the Genizah",
        paragraphs: [
          "A LoRA on the language model alone was enough to make Qwen read the printed Talmud well, cutting character error rates sharply and beating Gemini, Claude and other frontier models on the Talmud benchmark. On the Genizah that recipe failed: the pages were too damaged and too unlike anything the frozen vision encoder had seen. Applying LoRA to the vision tower as well (from model v19 onward) was the change that let the fine-tuned model outperform the Kraken family of HTR models. Later versions add bounding-box training so the model locates each line on the image, and section-based prompting for visual question answering.",
          "A small specialist model of around 8 to 9 billion parameters outperformed much larger generalist models on both benchmarks and runs on a MacBook Pro with 32 GB of memory. Training mixtures matter: more Judeo-Arabic improves letters and business documents but hurts religious texts, and dropping the Talmud replay data hurts everything, so the pipeline now routes religious and documentary fragments to different checkpoints.",
        ],
      },
      {
        heading: "Measuring transcription honestly",
        paragraphs: [
          "Character and word error rates stop making sense on pages where children's writing exercises sit on top of a marriage contract or where scholars themselves disagree on the reading, and they penalise a model that reads half a page correctly before looping more than one that refuses outright. The article adds n-gram precision, which rewards correctly read phrases regardless of order; a substantive-reading rate that separates real attempts from abstention, hallucination and loops; and aligned precision, recall and F1, which respect reading order. Kraken and Google Cloud Vision OCR are benchmarked alongside the VLMs, with Kraken clearly stronger on Rashi script.",
        ],
      },
      {
        heading: "The two-reader consensus pipeline",
        paragraphs: [
          "Even the best fine-tuned model has a character error rate of about 16% on the Genizah, not good enough for a scholarly edition. But Kraken and the VLM are trained independently, so when both produce the same text for a line the reading is far more likely to be right. The production pipeline runs both readers on every image, aligns the lines and marks a line as agreed when the two concur. About 16% of lines currently reach agreement, up from 11% with the previous segmenter. Those agreed lines are what the AI transcription search on this site indexes, and what the manuscript reader shows in green, with single-reader lines in yellow. The models and a Kraken segmenter for the Genizah are released on Hugging Face.",
        ],
      },
    ],
    keyTerms: [
      "Cairo Genizah",
      "AI transcription",
      "multimodal AI",
      "vision-language model (VLM)",
      "Kraken HTR",
      "Qwen3-VL fine-tuning",
      "LoRA on the vision tower",
      "character error rate (CER)",
      "canonical interpolation",
      "loop collapse",
      "two-reader consensus",
      "Judeo-Arabic",
      "Rashi script",
      "Vilna Talmud benchmark",
    ],
    links: [
      { label: "Full article on Medium", href: TRANSCRIBING_URL, external: true },
      {
        label: "Qwen3-VL 8B Hebrew v21b (religious texts flagship) on Hugging Face",
        href: "https://huggingface.co/isaacmg/qwen3-vl-8b-hebrew-v21b-ckpt",
        external: true,
      },
      {
        label: "Qwen3-VL 8B Hebrew v22a pilot on Hugging Face",
        href: "https://huggingface.co/isaacmg/qwen3-vl-8b-hebrew-v22a-ckpt",
        external: true,
      },
      {
        label: "Kraken Cairo Genizah segmenter on Hugging Face",
        href: "https://huggingface.co/isaacmg/kraken-genizah-segmenter",
        external: true,
      },
      { label: "Datasets on Hugging Face", href: "https://huggingface.co/isaacmg/datasets", external: true },
    ],
    related: [
      { label: "Search the AI transcriptions (beta)", href: "/" },
      { label: "See machine readings on a manuscript: Sukkot fragments", href: "/sukkot" },
      { label: "Yom Kippur fragments with machine readings", href: "/yom-kippur" },
      { label: "About the project", href: "/about" },
    ],
  },
  {
    slug: "multimodal-ai-for-manuscript-discovery-searching-and-clustering-the-cairo-genizah",
    title: "Multimodal AI for Manuscript Discovery: Searching and Clustering the Cairo Genizah",
    metaTitle: "Multimodal AI for Medieval Manuscript Discovery: Searching and Clustering the Cairo Genizah",
    description:
      "How multimodal embeddings (CLIP, ColPali, Nomic) search and cluster Cairo Genizah fragments, and how the first version of Cairo Genizah AI was built.",
    mediumUrl: DISCOVERY_URL,
    publication: PUBLICATION,
    author: AUTHOR,
    datePublished: "2025-07-07",
    readTime: "28 min read",
    keywords: [
      "Cairo Genizah",
      "multimodal AI",
      "manuscript search",
      "document embeddings",
      "CLIP",
      "ColPali",
      "semantic search",
      "clustering medieval manuscripts",
      "digital humanities",
      "retrieval-augmented generation",
    ],
    lead:
      "The first article in the series: how multimodal deep learning can help scholars search a 400,000-piece archive, what CLIP and ColPali embeddings actually learn about Hebrew and Arabic documents, and how the first version of this website was built.",
    sections: [
      {
        heading: "A puzzle with 400,000 pieces",
        paragraphs: [
          "The Cairo Genizah holds more than 400,000 fragments from the 6th to the 18th centuries in Hebrew, Aramaic, Arabic and Judeo-Arabic, scattered across libraries on several continents and mostly untranscribed. The article sets three goals: multilingual, multimodal search and question answering over the fragments; direct transcription of untranscribed images; and a better understanding of how text and image embeddings are aligned so the lessons transfer to other archives.",
        ],
      },
      {
        heading: "Where AI can help Genizah scholarship",
        paragraphs: [
          "Identifying joins, pieces of one manuscript now held in different libraries, where earlier image-only methods confirmed only about a quarter of their suggestions. The lack of transcriptions, which limits every search. The composition of inks and writing materials, studied by Zina Cohen, which could become training signal for visual embeddings. Authorship attribution through clustering. And the absence of a single search across collections and the scholarship about them, which the site set out to provide.",
        ],
      },
      {
        heading: "How CLIP and ColPali embeddings work",
        paragraphs: [
          "A reader-friendly walk through contrastive pre-training: how CLIP projects images and captions into one space and pulls matching pairs together, why it works for short captions but not for whole documents, and how ColPali and Nomic's document models instead match query tokens against image patches using hard negatives. Experiments embed Jewish terms and images with CLIP, and Hebrew, Arabic and English word pairs with Nomic, showing that the multilingual encoder places Hebrew and Arabic close together and understands most terms, with instructive exceptions.",
        ],
      },
      {
        heading: "Clustering the Princeton Geniza Project documents",
        paragraphs: [
          "Embedding 5,000 transcribed Princeton Geniza Project documents produces clusters that match scholarly categories: Talmud and piyyut on one side, letters and legal documents on the other, with cluster purity of 0.832 against the broad categories. Letters and legal documents overlap because many medieval letters carried contracts, a historically meaningful blur that standard metrics such as the silhouette score (0.206) cannot see. The article argues that evaluating AI on the humanities needs domain expertise alongside numbers.",
        ],
      },
      {
        heading: "Building the first search system",
        paragraphs: [
          "The first version of Cairo Genizah AI: a streaming document processor that embeds images and text, extracts metadata, and indexes everything into Elasticsearch for vector search; a FastAPI backend and a React frontend with semantic search and metadata filters; Terraform for cloud deployment, with a plan to serve the models from Apple-silicon machines instead because unified memory makes multimodal models far cheaper to run on-premises. The conclusion points to the next step, transcription, because without transcriptions the embeddings see only the catalogue metadata.",
        ],
      },
    ],
    keyTerms: [
      "Cairo Genizah",
      "multimodal deep learning",
      "CLIP",
      "ColPali",
      "Nomic embeddings",
      "contrastive learning",
      "semantic search",
      "document clustering",
      "join identification",
      "Princeton Geniza Project",
      "Elasticsearch vector search",
      "retrieval-augmented generation",
    ],
    links: [
      { label: "Full article on Medium", href: DISCOVERY_URL, external: true },
      {
        label: "Cairo Genizah AI source code on GitHub",
        href: "https://github.com/AIStream-Peelout/genizah_search",
        external: true,
      },
    ],
    related: [
      { label: "Try semantic search", href: "/" },
      { label: "Collection Explorer: the collection mapped by similarity", href: "/explorer" },
      { label: "Places map", href: "/map" },
      {
        label: "Part 2: Transcribing the Cairo Genizah with Multimodal AI",
        href: "/blog/transcribing-the-cairo-genizah-with-multimodal-ai",
      },
    ],
  },
];

/**
 * Look up a post by its URL slug.
 * @param {string} slug - The slug from the /blog/:slug route.
 * @returns {Object|undefined} The matching post, or undefined if there is none.
 */
export const findPost = (slug) => POSTS.find((post) => post.slug === slug);

/**
 * Format an ISO date (YYYY-MM-DD) for display, e.g. "5 October 2026".
 * Parsed as UTC so the day never shifts with the reader's time zone.
 * @param {string} isoDate - The date in ISO 8601 form.
 * @returns {string} The date in long British form.
 */
export const formatPostDate = (isoDate) =>
  new Date(`${isoDate}T00:00:00Z`).toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
    timeZone: 'UTC',
  });
