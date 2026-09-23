/**
 * Wording shared by the /read viewer's banner and the AI-transcription search
 * banner. Kept in one place so both say exactly the same thing about what
 * "two readers agree" does and does not mean.
 */

/** What confirmation means, and the letter confusions both readers share. */
export const AGREEMENT_CAVEAT =
    'A line is marked confirmed when both produced the same text. That means the readers concur, not that ' +
    'the line is right: both share small letter confusions (ד/ר, ב/כ, ם/ס), and the vision model invents ' +
    'plausible words where the page is damaged.';

/** The two readers, in one clause. */
export const READERS_DESCRIPTION =
    'Two automatic readers, a fine-tuned vision-language model and the Kraken HTR model, read this image ' +
    'separately, offline.';

/**
 * Tooltip for an agreed line.
 * @param {{agreement: number}} line - Line record.
 * @returns {string} Tooltip text.
 */
export const agreedTooltip = (line) =>
    `Two independent readers produced the same text (agreement ${Number(line.agreement).toFixed(2)}). ` +
    'Both can still share small letter confusions; not a scholarly transcription.';

/** Tooltip for an unconfirmed line. */
export const UNCONFIRMED_TOOLTIP =
    'Single reader: only the vision model read this line, or the two readers disagree. ' +
    'The yellow box shows where it read; the text is unchecked.';
