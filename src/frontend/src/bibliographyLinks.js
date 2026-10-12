/**
 * Shared URL builders for locating a cited work outside the site (WorldCat,
 * Google Scholar). Both ChatUI.jsx's cited-work popover and the document
 * modal's bibliography "work detail" panel (core_results/BibliographyDetail.jsx)
 * use these, so the query shape stays identical wherever a title is linked out.
 *
 * The WorldCat construction mirrors the backend's own fallback in
 * `GET /book-info` (src/backend/app.py, `worldcat_url`): a title phrase query,
 * optionally narrowed by an author surname.
 */

/**
 * Build a WorldCat catalog search URL for a work title.
 * @param {string} title - Work title as cited.
 * @param {string} [author] - Optional author name/surname to narrow the search.
 * @returns {string} A `search.worldcat.org` search URL.
 */
export const buildWorldcatSearchUrl = (title, author) => {
    let query = `ti:"${title}"`;
    if (author) query += ` AND au:"${author}"`;
    return `https://search.worldcat.org/search?q=${encodeURIComponent(query)}`;
};

/**
 * Build a Google Scholar search URL for a work title.
 * @param {string} title - Work title as cited.
 * @param {string} [author] - Optional author name/surname to narrow the search.
 * @returns {string} A `scholar.google.com` search URL.
 */
export const buildGoogleScholarUrl = (title, author) => {
    const query = `"${title}"` + (author ? ` ${author}` : '');
    return `https://scholar.google.com/scholar?q=${encodeURIComponent(query)}`;
};
