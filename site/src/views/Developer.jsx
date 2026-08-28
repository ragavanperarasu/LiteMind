import { Avatar, Box, Card, Chip, Link, Stack, Typography } from '@mui/material';
import { alpha } from '@mui/material/styles';
import GitHubIcon from '@mui/icons-material/GitHub';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import LanguageRoundedIcon from '@mui/icons-material/LanguageRounded';
import SchoolRoundedIcon from '@mui/icons-material/SchoolRounded';
import AndroidRoundedIcon from '@mui/icons-material/AndroidRounded';
import StorefrontRoundedIcon from '@mui/icons-material/StorefrontRounded';
import SlideshowRoundedIcon from '@mui/icons-material/SlideshowRounded';
import PublicRoundedIcon from '@mui/icons-material/PublicRounded';
import OpenInNewRoundedIcon from '@mui/icons-material/OpenInNewRounded';
import { developer, site } from '../lib/site.js';

/** Icon names live in site.js as strings so that file stays free of imports. */
const icons = {
  portfolio: LanguageRoundedIcon,
  linkedin: LinkedInIcon,
  github: GitHubIcon,
  college: SchoolRoundedIcon,
  android: AndroidRoundedIcon,
  web: PublicRoundedIcon,
  store: StorefrontRoundedIcon,
  slides: SlideshowRoundedIcon,
};

const initials = developer.name
  .split(/\s+/)
  .map((part) => part[0])
  .join('')
  .slice(0, 2)
  .toUpperCase();

function ProfileLink({ label, value, href, icon }) {
  const Icon = icons[icon] || LanguageRoundedIcon;
  return (
    <Card
      component="a"
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      variant="outlined"
      sx={{
        p: 2,
        display: 'flex',
        alignItems: 'center',
        gap: 1.5,
        textDecoration: 'none',
        borderColor: 'divider',
        transition: 'border-color 120ms, background-color 120ms',
        '&:hover': {
          borderColor: 'primary.main',
          bgcolor: (t) => alpha(t.palette.primary.main, 0.06),
        },
      }}
    >
      <Icon fontSize="small" sx={{ color: 'primary.main' }} />
      <Box sx={{ minWidth: 0, flex: 1 }}>
        <Typography variant="body2" sx={{ fontWeight: 650, color: 'text.primary' }}>
          {label}
        </Typography>
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ display: 'block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}
        >
          {value}
        </Typography>
      </Box>
      <OpenInNewRoundedIcon sx={{ fontSize: 15, color: 'text.secondary' }} />
    </Card>
  );
}

function ProjectCard({ name, kind, body, href, icon }) {
  const Icon = icons[icon] || PublicRoundedIcon;
  return (
    <Card
      component="a"
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      variant="outlined"
      sx={{
        p: 2.5,
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        textDecoration: 'none',
        borderColor: 'divider',
        transition: 'border-color 120ms, background-color 120ms',
        '&:hover': {
          borderColor: 'primary.main',
          bgcolor: (t) => alpha(t.palette.primary.main, 0.05),
        },
      }}
    >
      <Stack direction="row" alignItems="center" spacing={1.25} sx={{ mb: 1.25 }}>
        <Box
          sx={{
            width: 34,
            height: 34,
            borderRadius: 2,
            display: 'grid',
            placeItems: 'center',
            bgcolor: (t) => alpha(t.palette.primary.main, 0.12),
          }}
        >
          <Icon sx={{ fontSize: 19, color: 'primary.main' }} />
        </Box>
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Typography sx={{ fontWeight: 650, color: 'text.primary', lineHeight: 1.3 }}>
            {name}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {kind}
          </Typography>
        </Box>
        <OpenInNewRoundedIcon sx={{ fontSize: 15, color: 'text.secondary' }} />
      </Stack>
      <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.6 }}>
        {body}
      </Typography>
    </Card>
  );
}

export default function Developer() {
  return (
    <Box>
      <Typography variant="overline" color="text.secondary">
        Developer
      </Typography>

      <Stack
        direction={{ xs: 'column', sm: 'row' }}
        spacing={{ xs: 2.5, sm: 3 }}
        alignItems={{ xs: 'flex-start', sm: 'center' }}
        sx={{ mt: 1, mb: 4 }}
      >
        <Avatar
          sx={{
            width: 76,
            height: 76,
            fontSize: '1.6rem',
            fontWeight: 700,
            letterSpacing: '-0.02em',
            bgcolor: (t) => alpha(t.palette.primary.main, 0.15),
            color: 'primary.main',
          }}
        >
          {initials}
        </Avatar>
        <Box>
          <Typography
            component="h1"
            sx={{
              fontSize: 'clamp(1.9rem, 1.4rem + 2vw, 2.8rem)',
              fontWeight: 700,
              letterSpacing: '-0.03em',
              lineHeight: 1.1,
            }}
          >
            {developer.name}
          </Typography>
          <Typography color="text.secondary" sx={{ mt: 0.75, fontSize: '1.05rem' }}>
            {developer.role}
          </Typography>
          <Stack direction="row" spacing={1} sx={{ mt: 1.5, flexWrap: 'wrap', gap: 1 }}>
            <Chip label={developer.degree} size="small" sx={{ fontWeight: 600, fontSize: '0.72rem' }} />
            <Chip
              component="a"
              href={developer.collegeUrl}
              target="_blank"
              rel="noopener noreferrer"
              clickable
              label={developer.college}
              size="small"
              variant="outlined"
              sx={{ fontWeight: 500, fontSize: '0.72rem' }}
            />
          </Stack>
        </Box>
      </Stack>

      <Typography color="text.secondary" sx={{ maxWidth: '68ch', lineHeight: 1.7, mb: 4 }}>
        {developer.note}
      </Typography>

      <Box
        sx={{
          display: 'grid',
          gap: 2,
          gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', lg: 'repeat(4, 1fr)' },
        }}
      >
        {developer.profiles.map((profile) => (
          <ProfileLink key={profile.href} {...profile} />
        ))}
      </Box>

      <Box sx={{ mt: { xs: 6, md: 8 } }}>
        <Typography variant="overline" color="text.secondary">
          Other projects
        </Typography>
        <Typography variant="h4" sx={{ mt: 1, mb: 1, fontWeight: 700, letterSpacing: '-0.02em' }}>
          Built for my college, and shipped.
        </Typography>
        <Typography color="text.secondary" sx={{ mb: 3, maxWidth: '62ch' }}>
          My GCT is a set of tools for students at Government College of Technology — one Android
          app and three web services that share it.
        </Typography>

        <Box
          sx={{
            display: 'grid',
            gap: 2,
            gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' },
          }}
        >
          {developer.projects.map((project) => (
            <ProjectCard key={project.href} {...project} />
          ))}
        </Box>
      </Box>

      <Typography variant="body2" color="text.secondary" sx={{ mt: 6 }}>
        The source for this site and for the engine it documents is on{' '}
        <Link href={site.repository} target="_blank" rel="noopener noreferrer">
          GitHub
        </Link>
        .
      </Typography>
    </Box>
  );
}
