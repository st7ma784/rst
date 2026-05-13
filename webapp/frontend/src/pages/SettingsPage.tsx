import { useState, useEffect } from 'react';
import {
  Container, Typography, Box, Paper, Grid, Button,
  Switch, FormControlLabel, Slider, Alert, Snackbar,
  Divider, Chip, Table, TableBody, TableRow, TableCell,
  CircularProgress,
} from '@mui/material';
import SaveIcon from '@mui/icons-material/Save';
import RestoreIcon from '@mui/icons-material/Restore';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';

interface Settings {
  processing:    { default_mode: string; max_batch_size: number; enable_gpu: boolean };
  visualization: { default_colormap: string; enable_3d: boolean; refresh_rate: number };
  remote:        { timeout: number; retry_attempts: number };
}

interface SysInfo {
  gpu_available:  boolean;
  gpu_count:      number;
  gpus:           { id: number; name: string; compute_capability: string; total_memory_gb: number }[];
  python_version: string;
  os:             string;
  cpu_count:      number;
  db_path:        string;
}

export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings | null>(null);
  const [sysInfo, setSysInfo]   = useState<SysInfo | null>(null);
  const [saved, setSaved]       = useState(false);
  const [error, setError]       = useState<string | null>(null);
  const [sysLoading, setSysLoading] = useState(true);

  useEffect(() => {
    fetch('/api/settings/')
      .then(r => r.json())
      .then(setSettings)
      .catch(e => setError(e.message));

    fetch('/api/settings/system-info')
      .then(r => r.json())
      .then(d => { setSysInfo(d); setSysLoading(false); })
      .catch(() => setSysLoading(false));
  }, []);

  const save = async () => {
    try {
      await fetch('/api/settings/', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings),
      });
      setSaved(true);
    } catch (e: any) {
      setError(e.message);
    }
  };

  const reset = async () => {
    try {
      const r = await fetch('/api/settings/reset', { method: 'POST' });
      const d = await r.json();
      setSettings(d.settings);
      setSaved(true);
    } catch (e: any) {
      setError(e.message);
    }
  };

  if (!settings) return <Box sx={{ display: 'flex', justifyContent: 'center', mt: 10 }}><CircularProgress /></Box>;

  const set = (section: keyof Settings, key: string, value: unknown) =>
    setSettings(s => s ? { ...s, [section]: { ...s[section], [key]: value } } : s);

  return (
    <Container maxWidth="md">
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3, gap: 2 }}>
        <Typography variant="h4" sx={{ flexGrow: 1 }}>Settings</Typography>
        <Button variant="outlined" startIcon={<RestoreIcon />} onClick={reset}>
          Reset defaults
        </Button>
        <Button variant="contained" startIcon={<SaveIcon />} onClick={save}>
          Save
        </Button>
      </Box>

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      <Grid container spacing={3}>
        {/* Processing */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>Processing</Typography>
            <Divider sx={{ mb: 2 }} />

            <FormControlLabel
              control={
                <Switch
                  checked={settings.processing.enable_gpu}
                  onChange={e => set('processing', 'enable_gpu', e.target.checked)}
                />
              }
              label="Enable GPU acceleration (requires CUDA)"
            />

            <Box sx={{ mt: 3 }}>
              <Typography gutterBottom>
                Max batch size: {settings.processing.max_batch_size} ranges
              </Typography>
              <Slider
                value={settings.processing.max_batch_size}
                onChange={(_, v) => set('processing', 'max_batch_size', v)}
                min={32} max={1024} step={32} marks valueLabelDisplay="auto"
              />
            </Box>

            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Default mode
              </Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                {['auto', 'cpu', 'cuda'].map(m => (
                  <Chip key={m} label={m} clickable
                    color={settings.processing.default_mode === m ? 'primary' : 'default'}
                    onClick={() => set('processing', 'default_mode', m)}
                  />
                ))}
              </Box>
            </Box>
          </Paper>
        </Grid>

        {/* Visualization */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>Visualization</Typography>
            <Divider sx={{ mb: 2 }} />

            <FormControlLabel
              control={
                <Switch
                  checked={settings.visualization.enable_3d}
                  onChange={e => set('visualization', 'enable_3d', e.target.checked)}
                />
              }
              label="Enable 3D views"
            />

            <Box sx={{ mt: 2 }}>
              <Typography gutterBottom>
                Refresh rate: {settings.visualization.refresh_rate} fps
              </Typography>
              <Slider
                value={settings.visualization.refresh_rate}
                onChange={(_, v) => set('visualization', 'refresh_rate', v)}
                min={1} max={60} step={1} valueLabelDisplay="auto"
              />
            </Box>
          </Paper>
        </Grid>

        {/* Remote */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>Remote Compute</Typography>
            <Divider sx={{ mb: 2 }} />

            <Box sx={{ mt: 1 }}>
              <Typography gutterBottom>
                Job timeout: {settings.remote.timeout}s
              </Typography>
              <Slider
                value={settings.remote.timeout}
                onChange={(_, v) => set('remote', 'timeout', v)}
                min={60} max={3600} step={60} valueLabelDisplay="auto"
              />
            </Box>

            <Box sx={{ mt: 2 }}>
              <Typography gutterBottom>
                Retry attempts: {settings.remote.retry_attempts}
              </Typography>
              <Slider
                value={settings.remote.retry_attempts}
                onChange={(_, v) => set('remote', 'retry_attempts', v)}
                min={0} max={10} step={1} marks valueLabelDisplay="auto"
              />
            </Box>
          </Paper>
        </Grid>

        {/* System Info */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>System Info</Typography>
            <Divider sx={{ mb: 2 }} />
            {sysLoading ? (
              <CircularProgress size={20} />
            ) : sysInfo ? (
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Table size="small">
                    <TableBody>
                      <TableRow>
                        <TableCell>GPU</TableCell>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            {sysInfo.gpu_available
                              ? <><CheckCircleIcon color="success" fontSize="small" /> {sysInfo.gpu_count} device(s)</>
                              : <><ErrorIcon color="disabled" fontSize="small" /> not available</>}
                          </Box>
                        </TableCell>
                      </TableRow>
                      {sysInfo.gpus.map(g => (
                        <TableRow key={g.id}>
                          <TableCell sx={{ pl: 4 }}>GPU {g.id}</TableCell>
                          <TableCell>
                            {g.name} · CC {g.compute_capability} · {g.total_memory_gb} GB
                          </TableCell>
                        </TableRow>
                      ))}
                      <TableRow>
                        <TableCell>Python</TableCell>
                        <TableCell>{sysInfo.python_version}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>CPU cores</TableCell>
                        <TableCell>{sysInfo.cpu_count}</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Table size="small">
                    <TableBody>
                      <TableRow>
                        <TableCell>OS</TableCell>
                        <TableCell>{sysInfo.os}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Database</TableCell>
                        <TableCell sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                          {sysInfo.db_path}
                        </TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Upload retention</TableCell>
                        <TableCell>24 hours (auto-swept)</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>API docs</TableCell>
                        <TableCell>
                          <a href="/docs" target="_blank" rel="noreferrer"
                             style={{ color: '#90caf9' }}>/docs</a>
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </Grid>
              </Grid>
            ) : (
              <Typography color="text.secondary">System info unavailable</Typography>
            )}
          </Paper>
        </Grid>
      </Grid>

      <Snackbar open={saved} autoHideDuration={2500} onClose={() => setSaved(false)}>
        <Alert severity="success" onClose={() => setSaved(false)}>Settings saved</Alert>
      </Snackbar>
    </Container>
  );
}
