/**
 * One slug function, used by both the anchor plugin and the table of contents.
 * If these ever disagree the contents links point at nothing, so there is
 * deliberately only one of them.
 */
export function slugify(text) {
  return String(text)
    .trim()
    .toLowerCase()
    .replace(/`/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'section';
}

export default slugify;
