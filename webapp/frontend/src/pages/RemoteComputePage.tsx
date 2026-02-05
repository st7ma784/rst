import {
  Container,
  Typography,
  Grid,
  Paper,
  Box,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
} from '@mui/material';
import { useState } from 'react';

interface RemoteConfig {
  computeType: string;
  host: string;
  username: string;
  partition: string;
  account: string;
  nodes: number;
  gpus: number;
  timeLimit: string;
}

export default function RemoteComputePage() {
  const [config, setConfig] = useState<RemoteConfig>({
    computeType: 'slurm',
    host: '',
    username: '',
    partition: 'gpu',
    account: '',
    nodes: 1,
    gpus: 1,
    timeLimit: '01:00:00',
  });
  const [testStatus, setTestStatus] = useState<'idle' | 'testing' | 'success' | 'error'>('idle');

  const handleTestConnection = async () => {
    setTestStatus('testing');
    
    try {
      const response = await fetch('/api/remote/test-connection', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });
      
      if (response.ok) {
        setTestStatus('success');
      } else {
        setTestStatus('error');
      }
    } catch (error) {
      setTestStatus('error');
    }
  };

  return (
    <Container maxWidth="lg">
      <Typography variant="h3" component="h1" gutterBottom>
        Remote Compute Configuration
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Configure connection to Slurm clusters or SSH servers for remote processing
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              Connection Settings
            </Typography>

            <FormControl fullWidth sx={{ mt: 2 }}>
              <InputLabel>Compute Type</InputLabel>
              <Select
                value={config.computeType}
                onChange={(e) => setConfig({ ...config, computeType: e.target.value })}
                label="Compute Type"
              >
                <MenuItem value="slurm">Slurm Cluster</MenuItem>
                <MenuItem value="ssh">Direct SSH</MenuItem>
              </Select>
            </FormControl>

            <TextField
              fullWidth
              label="Host"
              value={config.host}
              onChange={(e) => setConfig({ ...config, host: e.target.value })}
              placeholder="cluster.university.edu"
              sx={{ mt: 2 }}
            />

            <TextField
              fullWidth
              label="Username"
              value={config.username}
              onChange={(e) => setConfig({ ...config, username: e.target.value })}
              sx={{ mt: 2 }}
            />

            {config.computeType === 'slurm' && (
              <>
                <TextField
                  fullWidth
                  label="Partition"
                  value={config.partition}
                  onChange={(e) => setConfig({ ...config, partition: e.target.value })}
                  sx={{ mt: 2 }}
                />

                <TextField
                  fullWidth
                  label="Account"
                  value={config.account}
                  onChange={(e) => setConfig({ ...config, account: e.target.value })}
                  sx={{ mt: 2 }}
                />

                <Grid container spacing={2} sx={{ mt: 1 }}>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      type="number"
                      label="Nodes"
                      value={config.nodes}
                      onChange={(e) => setConfig({ ...config, nodes: parseInt(e.target.value) })}
                    />
                  </Grid>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      type="number"
                      label="GPUs per Node"
                      value={config.gpus}
                      onChange={(e) => setConfig({ ...config, gpus: parseInt(e.target.value) })}
                    />
                  </Grid>
                </Grid>

                <TextField
                  fullWidth
                  label="Time Limit"
                  value={config.timeLimit}
                  onChange={(e) => setConfig({ ...config, timeLimit: e.target.value })}
                  placeholder="HH:MM:SS"
                  sx={{ mt: 2 }}
                />
              </>
            )}

            <Box sx={{ mt: 3 }}>
              <Button
                variant="contained"
                onClick={handleTestConnection}
                disabled={testStatus === 'testing' || !config.host || !config.username}
              >
                {testStatus === 'testing' ? 'Testing...' : 'Test Connection'}
              </Button>
            </Box>

            {testStatus === 'success' && (
              <Alert severity="success" sx={{ mt: 2 }}>
                Connection successful! Configuration saved.
              </Alert>
            )}

            {testStatus === 'error' && (
              <Alert severity="error" sx={{ mt: 2 }}>
                Connection failed. Please check your settings.
              </Alert>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              Remote Job Submission
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Once connected, you can submit processing jobs to the remote compute resource.
              Jobs will be monitored and results automatically retrieved when complete.
            </Typography>

            <Box sx={{ mt: 3 }}>
              <Typography variant="h6" gutterBottom>
                Features:
              </Typography>
              <Typography variant="body2" component="ul" sx={{ pl: 2 }}>
                <li>Automatic job script generation</li>
                <li>Real-time job status monitoring</li>
                <li>Automatic result retrieval</li>
                <li>Support for batch job submission</li>
                <li>Queue management</li>
              </Typography>
            </Box>

            <Box sx={{ mt: 3, p: 2, bgcolor: 'background.default', borderRadius: 1 }}>
              <Typography variant="body2" fontFamily="monospace">
                Example Slurm Script:
              </Typography>
              <Typography
                variant="body2"
                fontFamily="monospace"
                component="pre"
                sx={{ mt: 1, fontSize: '0.75rem' }}
              >
{`#!/bin/bash
#SBATCH --partition=${config.partition}
#SBATCH --nodes=${config.nodes}
#SBATCH --gres=gpu:${config.gpus}
#SBATCH --time=${config.timeLimit}

module load cuda
cudarst_fitacf --input data.rawacf`}
              </Typography>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
}
