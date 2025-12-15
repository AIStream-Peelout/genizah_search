/**
 * Normalizes a document ID or shelfmark by replacing slashes with underscores.
 * This is required because the backend expects underscores in URLs for these identifiers.
 * 
 * @param {string} id - The document ID or shelfmark to normalize.
 * @returns {string} The normalized ID.
 */
export const normalizeDocId = (id) => {
    if (!id) return '';
    return id.replace(/\//g, '_');
};
