import { Box, Button, Card, Chip, Divider, Link, Stack, Typography } from '@mui/material';
import { alpha } from '@mui/material/styles';
import ArrowForwardRoundedIcon from '@mui/icons-material/ArrowForwardRounded';
import GitHubIcon from '@mui/icons-material/GitHub';
import { docs, sections } from '../lib/content.js';
import { routes } from '../lib/router.js';
import { site } from '../lib/site.js';
import Markdown from '../components/Markdown.jsx';

/** Every figure here comes from docs/model-info.json or docs/12-performance.md. */
const headline = [
  { value: '15.7 B', label: 'parameters in the checkpoint' },
  { value: '2.45 B', label: 'active per token (15.6%)' },
  { value: '29.3 GiB', label: 'read straight from the SSD' },
  { value: '0 GPUs', label: 'required, by design' },
];

const flow = [
  {
    step: '01',
    title: 'Tokenize',
    body: 'Byte-level BPE, read from the checkpoint’s own tokenizer.json.',
  },
  {
    step: '02',
    title: 'Prefill',
    body: 'One position at a time, filling the KV cache. Only the last position needs logits.',
  },
  {
    step: '03',
    title: 'Route',
    body: 'Each of the 26 MoE layers scores 64 experts and keeps the top 6.',
  },
  {
    step: '04',
    title: 'Stream and execute',
    body: 'The six chosen experts arrive from the mapping, 16.5 MiB each, and run as SwiGLU.',
  },
  {
    step: '05',
    title: 'Sample',
    body: 'Temperature, top-k, top-p and a repetition penalty, then decode and print.',
  },
];

const quickStart = `\`\`\`powershell
git clone ${site.repository}.git
cd LiteMind

.\\scripts\\check_environment.ps1      # what is missing, and how to get it
.\\scripts\\build.ps1 -RunTests        # compile and prove the kernels
.\\scripts\\download_model.ps1         # 29.3 GiB, resumable
.\\scripts\\run.ps1 -Prompt "Explain mixture of experts in one paragraph."
\`\`\``;

function Stat({ value, label }) {
  return (
    <Card
      variant="outlined"
      sx={{ p: 2.5, borderColor: 'divider', bgcolor: 'background.paper', height: '100%' }}
    >
      <Typography
        sx={{
          fontSize: '1.7rem',
          fontWeight: 700,
          letterSpacing: '-0.03em',
          fontVariantNumeric: 'tabular-nums',
          color: 'primary.main',
        }}
      >
        {value}
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
        {label}
      </Typography>
    </Card>
  );
}

export default function Home() {
  return (
    <Box>
      {/* Hero */}
      <Box sx={{ pt: { xs: 2, md: 5 }, pb: { xs: 5, md: 7 } }}>
        <Chip
          label="C++20 · CPU only · no third-party libraries"
          size="small"
          sx={{ mb: 3, fontWeight: 600, fontSize: '0.72rem' }}
        />
        <Typography
          component="h1"
          sx={{
            fontSize: 'clamp(2.3rem, 1.5rem + 3.4vw, 4rem)',
            fontWeight: 700,
            letterSpacing: '-0.035em',
            lineHeight: 1.05,
            maxWidth: 900,
          }}
        >
          A 15.7-billion-parameter model,
          <Box component="span" sx={{ color: 'primary.main' }}> on a laptop with no GPU.</Box>
        </Typography>
        <Typography
          sx={{ mt: 3, fontSize: '1.12rem', lineHeight: 1.65, maxWidth: '62ch', color: 'text.secondary' }}
        >
          LiteMind runs DeepSeek-V2-Lite on the CPU. The checkpoint stays on the SSD as a memory
          mapping, and the mixture-of-experts router decides what is pulled into RAM — so a token
          pays for six experts instead of for the whole model.
        </Typography>

        <Stack direction="row" spacing={1.5} sx={{ mt: 4, flexWrap: 'wrap', gap: 1.5 }}>
          <Button
            component="a"
            href={routes.doc('00-setup')}
            variant="contained"
            size="large"
            endIcon={<ArrowForwardRoundedIcon />}
            sx={{ px: 3 }}
          >
            Setup guide
          </Button>
          <Button
            component="a"
            href={routes.doc('01-overview')}
            variant="outlined"
            size="large"
            color="inherit"
            sx={{ px: 3, borderColor: 'divider' }}
          >
            How it works
          </Button>
          <Button
            component="a"
            href={site.repository}
            target="_blank"
            rel="noopener noreferrer"
            variant="text"
            size="large"
            color="inherit"
            startIcon={<GitHubIcon />}
          >
            Source
          </Button>
        </Stack>
      </Box>

      {/* Headline numbers */}
      <Box
        sx={{
          display: 'grid',
          gap: 2,
          gridTemplateColumns: { xs: '1fr 1fr', md: 'repeat(4, 1fr)' },
        }}
      >
        {headline.map((item) => (
          <Stat key={item.label} {...item} />
        ))}
      </Box>

      {/* The idea */}
      <Box
        sx={{
          mt: { xs: 7, md: 10 },
          display: 'grid',
          gap: { xs: 4, md: 6 },
          gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' },
          alignItems: 'center',
        }}
      >
        <Box>
          <Typography variant="overline" color="text.secondary">
            The idea
          </Typography>
          <Typography variant="h4" sx={{ mt: 1, mb: 2, fontWeight: 700, letterSpacing: '-0.02em' }}>
            Residency follows the working set, not the file size.
          </Typography>
          <Typography color="text.secondary" sx={{ lineHeight: 1.7, mb: 2 }}>
            Every other runtime assumes the whole model fits in fast memory at once. It does not
            have to. Each of the 26 mixture-of-experts layers holds 64 independent feed-forward
            experts and a router that picks six of them, so roughly a tenth of the expert weights
            decide any single token.
          </Typography>
          <Typography color="text.secondary" sx={{ lineHeight: 1.7 }}>
            LiteMind memory-maps the checkpoint and lets that routing decision drive paging. Opening
            29.3 GiB takes <b>0.2 seconds</b>, because nothing is copied.
          </Typography>
        </Box>

        <Card variant="outlined" sx={{ borderColor: 'divider' }}>
          <Box sx={{ px: 3, py: 2, borderBottom: 1, borderColor: 'divider' }}>
            <Typography variant="overline" color="text.secondary">
              Where the 29.3 GiB sits
            </Typography>
          </Box>
          <Stack divider={<Divider />}>
            {[
              {
                name: 'Always-hot weights',
                size: '2.44 GiB',
                note: 'Attention, norms, embeddings, shared experts — touched every step',
                weight: 0.09,
              },
              {
                name: 'Routed experts',
                size: '26.81 GiB',
                note: '1,664 of them; about 10% are touched per token',
                weight: 1,
              },
            ].map((row) => (
              <Box key={row.name} sx={{ px: 3, py: 2.25 }}>
                <Stack direction="row" justifyContent="space-between" alignItems="baseline">
                  <Typography sx={{ fontWeight: 600 }}>{row.name}</Typography>
                  <Typography
                    sx={{ fontWeight: 700, fontVariantNumeric: 'tabular-nums', color: 'primary.main' }}
                  >
                    {row.size}
                  </Typography>
                </Stack>
                <Box
                  sx={{
                    mt: 1.25,
                    height: 6,
                    borderRadius: 3,
                    bgcolor: (t) => alpha(t.palette.primary.main, 0.14),
                  }}
                >
                  <Box
                    sx={{
                      width: `${row.weight * 100}%`,
                      height: '100%',
                      borderRadius: 3,
                      bgcolor: 'primary.main',
                    }}
                  />
                </Box>
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                  {row.note}
                </Typography>
              </Box>
            ))}
          </Stack>
        </Card>
      </Box>

      {/* Flow */}
      <Box sx={{ mt: { xs: 7, md: 10 } }}>
        <Typography variant="overline" color="text.secondary">
          What happens to a prompt
        </Typography>
        <Typography variant="h4" sx={{ mt: 1, mb: 3, fontWeight: 700, letterSpacing: '-0.02em' }}>
          Five stages, twenty-seven layers deep.
        </Typography>
        <Box
          sx={{
            display: 'grid',
            gap: 2,
            gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', lg: 'repeat(5, 1fr)' },
          }}
        >
          {flow.map((item) => (
            <Card
              key={item.step}
              variant="outlined"
              sx={{ p: 2.5, borderColor: 'divider', height: '100%' }}
            >
              <Typography
                sx={{ fontWeight: 700, color: 'secondary.main', fontSize: '0.8rem', letterSpacing: '0.08em' }}
              >
                {item.step}
              </Typography>
              <Typography sx={{ mt: 1, mb: 0.75, fontWeight: 650 }}>{item.title}</Typography>
              <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.6 }}>
                {item.body}
              </Typography>
            </Card>
          ))}
        </Box>
      </Box>

      {/* Quick start */}
      <Box sx={{ mt: { xs: 7, md: 10 } }}>
        <Typography variant="overline" color="text.secondary">
          Quick start
        </Typography>
        <Typography variant="h4" sx={{ mt: 1, mb: 1, fontWeight: 700, letterSpacing: '-0.02em' }}>
          Four scripts, in order.
        </Typography>
        <Typography color="text.secondary" sx={{ mb: 3, maxWidth: '62ch' }}>
          Windows with PowerShell is the tested path; Linux and macOS build with plain CMake. The{' '}
          <Link href={routes.doc('00-setup')}>setup guide</Link> covers versions, disk space and
          what to do when a step fails.
        </Typography>
        <Markdown source={quickStart} />
      </Box>

      {/* Documentation map */}
      <Box sx={{ mt: { xs: 5, md: 8 } }}>
        <Typography variant="overline" color="text.secondary">
          The documentation
        </Typography>
        <Typography variant="h4" sx={{ mt: 1, mb: 1, fontWeight: 700, letterSpacing: '-0.02em' }}>
          {docs.length} pages, one per part of the system.
        </Typography>
        <Typography color="text.secondary" sx={{ mb: 3, maxWidth: '62ch' }}>
          Read them in order the first time; after that each stands on its own.
        </Typography>

        <Box
          sx={{
            display: 'grid',
            gap: 2.5,
            gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', lg: 'repeat(3, 1fr)' },
          }}
        >
          {sections.map((section) => (
            <Card key={section.label} variant="outlined" sx={{ p: 2.5, borderColor: 'divider' }}>
              <Typography variant="overline" color="text.secondary">
                {section.label}
              </Typography>
              <Stack sx={{ mt: 1 }} spacing={0.75}>
                {section.pages.map((page) => (
                  <Link
                    key={page.id}
                    href={routes.doc(page.id)}
                    underline="none"
                    sx={{
                      display: 'flex',
                      gap: 1.25,
                      color: 'text.primary',
                      fontSize: '0.9rem',
                      '&:hover': { color: 'primary.main' },
                    }}
                  >
                    <Box
                      component="span"
                      sx={{
                        color: 'text.secondary',
                        fontVariantNumeric: 'tabular-nums',
                        fontSize: '0.8rem',
                        pt: '2px',
                      }}
                    >
                      {page.number}
                    </Box>
                    {page.title}
                  </Link>
                ))}
              </Stack>
            </Card>
          ))}
        </Box>
      </Box>
    </Box>
  );
}
