import { useParams } from 'react-router-dom';
import {
  Container,
  Typography,
  Grid,
  Paper,
  Box,
  Tabs,
  Tab,
} from '@mui/material';
import { useState } from 'react';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export default function VisualizationPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const [currentTab, setCurrentTab] = useState(0);

  const handleTabChange = (_: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  return (
    <Container maxWidth="xl">
      <Typography variant="h3" component="h1" gutterBottom>
        Results Visualization
      </Typography>
      <Typography variant="body1" color="text.secondary" gutterBottom>
        Job ID: {jobId}
      </Typography>

      <Paper sx={{ mt: 3 }}>
        <Tabs value={currentTab} onChange={handleTabChange}>
          <Tab label="Range-Time Plots" />
          <Tab label="3D Convection Map" />
          <Tab label="Grid View" />
          <Tab label="Performance Metrics" />
        </Tabs>

        <TabPanel value={currentTab} index={0}>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Box
                sx={{
                  height: 400,
                  bgcolor: 'background.default',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderRadius: 1,
                }}
              >
                <Typography color="text.secondary">
                  Range-Time Plot Visualization (Velocity, Power, Width)
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </TabPanel>

        <TabPanel value={currentTab} index={1}>
          <Box
            sx={{
              height: 600,
              bgcolor: 'background.default',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              borderRadius: 1,
            }}
          >
            <Typography color="text.secondary">
              3D Globe with Convection Vector Overlays (Three.js)
            </Typography>
          </Box>
        </TabPanel>

        <TabPanel value={currentTab} index={2}>
          <Box
            sx={{
              height: 600,
              bgcolor: 'background.default',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              borderRadius: 1,
            }}
          >
            <Typography color="text.secondary">
              2D Grid View with Interpolated Data
            </Typography>
          </Box>
        </TabPanel>

        <TabPanel value={currentTab} index={3}>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Processing Time Breakdown
                </Typography>
                <Box sx={{ mt: 2 }}>
                  <Typography>ACF: 0.5s</Typography>
                  <Typography>FITACF: 1.0s</Typography>
                  <Typography>Grid: 0.6s</Typography>
                  <Typography fontWeight="bold" sx={{ mt: 1 }}>
                    Total: 2.1s (GPU)
                  </Typography>
                </Box>
              </Paper>
            </Grid>
            <Grid item xs={12} md={6}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Performance Comparison
                </Typography>
                <Box sx={{ mt: 2 }}>
                  <Typography>CPU Time: 18.5s</Typography>
                  <Typography>GPU Time: 2.1s</Typography>
                  <Typography color="primary.main" fontWeight="bold" sx={{ mt: 1 }}>
                    Speedup: 8.8x
                  </Typography>
                </Box>
              </Paper>
            </Grid>
          </Grid>
        </TabPanel>
      </Paper>
    </Container>
  );
}
