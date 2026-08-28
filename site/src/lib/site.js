/** Everything that changes if the project moves, in one place. */
export const site = {
  name: 'LiteMind',
  tagline: 'Mixture-of-experts inference on the CPU',
  repository: 'https://github.com/ragavanperarasu/LiteMind',
  branch: 'main',
  model: 'https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite',
  license: 'MIT',
};

/** Who built it, and what else they have shipped. */
export const developer = {
  name: 'Ragavan M',
  role: 'Full-stack and mobile app developer',
  degree: 'B.E. Computer Science and Engineering',
  college: 'Government College of Technology, Coimbatore',
  collegeUrl: 'https://gct.ac.in/',
  note: 'LiteMind is my final-year project: a mixture-of-experts language model running on an ordinary CPU, written from scratch in C++20 with nothing but the standard library.',
  profiles: [
    { label: 'Portfolio', value: 'ragavan.vercel.app', href: 'https://ragavan.vercel.app/', icon: 'portfolio' },
    { label: 'LinkedIn', value: 'in/ragavandevp', href: 'https://www.linkedin.com/in/ragavandevp', icon: 'linkedin' },
    { label: 'GitHub', value: 'ragavanperarasu', href: 'https://github.com/ragavanperarasu', icon: 'github' },
    { label: 'College', value: 'gct.ac.in', href: 'https://gct.ac.in/', icon: 'college' },
  ],
  projects: [
    {
      name: 'My GCT Hub',
      kind: 'Android app',
      href: 'https://play.google.com/store/apps/details?id=com.mygcthub',
      icon: 'android',
      body: 'Question papers, notes and syllabus for Government College of Technology students, on Google Play.',
    },
    {
      name: 'My GCT',
      kind: 'Web',
      href: 'https://mygct.org/',
      icon: 'web',
      body: 'The home on the web for My GCT Hub — the same past papers, study material and curriculum, in a browser.',
    },
    {
      name: 'My GCT Store',
      kind: 'Web',
      href: 'https://store.mygct.org/',
      icon: 'store',
      body: 'A marketplace where GCT students publish and download each other’s work: mobile apps, websites, CLI tools and open-source projects.',
    },
    {
      name: 'My GCT Slides',
      kind: 'Web',
      href: 'https://slides.mygct.org/',
      icon: 'slides',
      body: 'Share slides from My GCT Hub with a temporary four-digit code. Viewers open them on any device — no app, no login.',
    },
  ],
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
