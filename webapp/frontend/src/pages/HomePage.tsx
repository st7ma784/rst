import {
  Container,
  Typography,
  Grid,
  Paper,
  Box,
  Card,
  CardContent,
  Button,
} from '@mui/material';
import {
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  Cloud as CloudIcon,
  Timeline as TimelineIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

export default function HomePage() {
  const navigate = useNavigate();

  const features = [
    {
      title: 'GPU Acceleration',
      description: '10-100x speedup with CUDA processing',
      icon: <SpeedIcon sx={{ fontSize: 48, color: 'primary.main' }} />,
    },
    {
      title: 'Real-time Processing',
      description: 'Interactive parameter tuning with instant feedback',
      icon: <TimelineIcon sx={{ fontSize: 48, color: 'primary.main' }} />,
    },
    {
      title: 'Remote Compute',
      description: 'Submit jobs to Slurm clusters or SSH servers',
      icon: <CloudIcon sx={{ fontSize: 48, color: 'primary.main' }} />,
    },
    {
      title: 'Efficient Memory',
      description: 'Optimized data structures for large datasets',
      icon: <MemoryIcon sx={{ fontSize: 48, color: 'primary.main' }} />,
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
          Transform your SuperDARN research with GPU acceleration. Process data 10-100x faster
          with real-time parameter tuning and interactive visualizations.
        </Typography>
        <Box sx={{ mt: 4 }}>
          <Button
            variant="contained"
            size="large"
            onClick={() => navigate('/processing')}
            sx={{ mr: 2 }}
          >
            Start Processing
          </Button>
          <Button
            variant="outlined"
            size="large"
            onClick={() => navigate('/visualization/demo')}
          >
            View Demo
          </Button>
        </Box>
      </Box>

      <Grid container spacing={4}>
        {features.map((feature, index) => (
          <Grid item xs={12} sm={6} md={3} key={index}>
            <Card
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                p: 3,
                transition: 'transform 0.2s',
                '&:hover': {
                  transform: 'translateY(-4px)',
                },
              }}
            >
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

      <Box sx={{ mt: 6 }}>
        <Paper sx={{ p: 4 }}>
          <Typography variant="h4" gutterBottom>
            Quick Start
          </Typography>
          <Typography variant="body1" paragraph>
            <strong>1. Upload Data:</strong> Drag and drop RAWACF or FITACF files
          </Typography>
          <Typography variant="body1" paragraph>
            <strong>2. Configure Parameters:</strong> Adjust processing settings in real-time
          </Typography>
          <Typography variant="body1" paragraph>
            <strong>3. Process:</strong> Choose CPU, GPU, or auto-detect mode
          </Typography>
          <Typography variant="body1" paragraph>
            <strong>4. Visualize:</strong> Explore interactive 3D convection maps and plots
          </Typography>
        </Paper>
      </Box>

      <Box sx={{ mt: 4 }}>
        <Paper sx={{ p: 4, bgcolor: 'background.default' }}>
          <Typography variant="h5" gutterBottom>
            Performance Highlights
          </Typography>
          <Grid container spacing={3} sx={{ mt: 2 }}>
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h3" color="primary.main">
                  100x
                </Typography>
                <Typography variant="body1">Faster Processing</Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h3" color="primary.main">
                  49
                </Typography>
                <Typography variant="body1">CUDA Kernels</Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h3" color="primary.main">
                  100%
                </Typography>
                <Typography variant="body1">Backward Compatible</Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </Box>
    </Container>
  );
}
