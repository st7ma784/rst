import { useEffect, useState } from 'react';
import {
  Container, Typography, Grid, Paper, Box, Card, Button, Chip, CircularProgress,
} from '@mui/material';
import {
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  Cloud as CloudIcon,
  Timeline as TimelineIcon,
  CheckCircle as CheckIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

interface BackendInfo { id: string; name: string; available: boolean; gpu: boolean; active: boolean; }
interface HealthInfo  { status: string; gpu_available: boolean; gpu_count: number; version: string; }

export default function HomePage() {
  const navigate = useNavigate();
  const [backends, setBackends]   = useState<BackendInfo[]>([]);
  const [health, setHealth]       = useState<HealthInfo | null>(null);
  const [loadingHW, setLoadingHW] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch('/api/health').then(r => r.json()).catch(() => null),
      fetch('/api/processing/backends').then(r => r.json()).catch(() => ({ backends: [] })),
    ]).then(([h, b]) => {
      setHealth(h);
      setBackends(b.backends ?? []);
      setLoadingHW(false);
    });
  }, []);

  const availableCount = backends.filter(b => b.available).length;
  const gpuBackend     = backends.find(b => b.available && b.gpu);

  const features = [
    {
      title: 'GPU Acceleration',
      description: gpuBackend
        ? `${gpuBackend.id} backend ready with GPU`
        : 'CPU mode — no GPU detected',
      icon: <SpeedIcon sx={{ fontSize: 48, color: gpuBackend ? 'success.main' : 'primary.main' }} />,
    },
    {
      title: 'Real-time Processing',
      description: 'Interactive parameter tuning with WebSocket live progress',
      icon: <TimelineIcon sx={{ fontSize: 48, color: 'primary.main' }} />,
    },
    {
      title: 'Remote Compute',
      description: 'Submit jobs to Slurm clusters or SSH servers',
      icon: <CloudIcon sx={{ fontSize: 48, color: 'primary.main' }} />,
    },
    {
      title: 'Three Backends',
      description: `${availableCount} of 3 backends available — switch per-job`,
      icon: <MemoryIcon sx={{ fontSize: 48, color: availableCount > 0 ? 'success.main' : 'error.main' }} />,
    },
  ];

  return (
    <Container maxWidth="lg">
      <Box sx={{ textAlign: 'center', mb: 6 }}>
        <Typography variant="h2" component="h1" gutterBottom sx={{ fontWeight: 'bold' }}>
          SuperDARN Interactive Workbench
        </Typography>
        <Typography variant="h5" color="text.secondary" paragraph>
          CUDA-Accelerated SuperDARN Data Processing
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Transform your SuperDARN research with GPU acceleration. Process data faster
          with real-time parameter tuning and interactive visualizations.
        </Typography>
        <Box sx={{ mt: 4, display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
          <Button variant="contained" size="large" onClick={() => navigate('/processing')}>
            Start Processing
          </Button>
          <Button variant="outlined" size="large" onClick={() => navigate('/jobs')}>
            View Jobs
          </Button>
          <Button variant="outlined" size="large" onClick={() => navigate('/compare')}>
            Compare Backends
          </Button>
        </Box>
      </Box>

      <Grid container spacing={4}>
        {features.map((feature, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card sx={{
              height: '100%', display: 'flex', flexDirection: 'column',
              alignItems: 'center', p: 3,
              transition: 'transform 0.2s',
              '&:hover': { transform: 'translateY(-4px)' },
            }}>
              <Box sx={{ mb: 2 }}>{feature.icon}</Box>
              <Typography variant="h6" component="h3" gutterBottom align="center">
                {feature.title}
              </Typography>
              <Typography variant="body2" color="text.secondary" align="center">
                {feature.description}
              </Typography>
            </Card>
          </Grid>
        ))}
      </Grid>

      {/* Live system status */}
      <Box sx={{ mt: 6 }}>
        <Paper sx={{ p: 4 }}>
          <Typography variant="h5" gutterBottom>System Status</Typography>
          {loadingHW ? (
            <CircularProgress size={24} />
          ) : (
            <Grid container spacing={3} sx={{ mt: 1 }}>
              {/* Health */}
              <Grid item xs={12} md={4}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                  {health?.status === 'healthy'
                    ? <CheckIcon color="success" />
                    : <ErrorIcon color="error" />}
                  <Typography variant="subtitle1">API</Typography>
                  <Chip label={health?.status ?? 'unreachable'} size="small"
                    color={health?.status === 'healthy' ? 'success' : 'error'} />
                </Box>
                <Typography variant="body2" color="text.secondary">
                  {health?.gpu_available
                    ? `CUDA: ${health.gpu_count} GPU(s) detected`
                    : 'CUDA: not available (CPU mode)'}
                </Typography>
              </Grid>

              {/* Backends */}
              <Grid item xs={12} md={8}>
                <Typography variant="subtitle1" gutterBottom>Backends</Typography>
                <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                  {backends.map(b => (
                    <Chip key={b.id}
                      label={`${b.id}${b.gpu ? ' (GPU)' : ''}${b.active ? ' ★' : ''}`}
                      color={b.available ? 'success' : 'default'}
                      variant={b.available ? 'filled' : 'outlined'}
                      size="small"
                      icon={b.available ? <CheckIcon /> : <ErrorIcon />}
                    />
                  ))}
                  {backends.length === 0 && (
                    <Typography variant="body2" color="text.secondary">
                      No backend info available
                    </Typography>
                  )}
                </Box>
              </Grid>
            </Grid>
          )}
        </Paper>
      </Box>

      {/* Quick Start */}
      <Box sx={{ mt: 4 }}>
        <Paper sx={{ p: 4 }}>
          <Typography variant="h5" gutterBottom>Quick Start</Typography>
          <Typography variant="body1" paragraph>
            <strong>1. Upload Data:</strong> Drag and drop RAWACF files on the Processing page
          </Typography>
          <Typography variant="body1" paragraph>
            <strong>2. Configure Parameters:</strong> Adjust min power, phase tolerance, and batch size
          </Typography>
          <Typography variant="body1" paragraph>
            <strong>3. Select Backend:</strong> Choose pythonv2, CUDArst, or RST reference — all RST v3.0-compatible
          </Typography>
          <Typography variant="body1" paragraph>
            <strong>4. Visualize:</strong> Explore RTI heatmaps, range profiles, grid distribution, and convection maps
          </Typography>
        </Paper>
      </Box>
    </Container>
  );
}
