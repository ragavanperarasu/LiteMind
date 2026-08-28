/** Everything that changes if the project moves, in one place. */
export const site = {
  name: 'LiteMind',
  tagline: 'Mixture-of-experts inference on the CPU',
  repository: 'https://github.com/ragavanperarasu/LiteMind',
  branch: 'main',
  model: 'https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite',
  license: 'MIT',
};

/**
 * A repo-relative path such as `src/Gemm.cpp` as a link to the source on
 * GitHub. Directories need /tree/ rather than /blob/, which GitHub does not
 * redirect between.
 */
export function sourceUrl(path) {
  const clean = path.replace(/^\.?\/*/, '');
  const isDirectory = clean.endsWith('/') || !/\.[a-z0-9]+$/i.test(clean);
  return `${site.repository}/${isDirectory ? 'tree' : 'blob'}/${site.branch}/${clean.replace(/\/$/, '')}`;
}

export default site;
