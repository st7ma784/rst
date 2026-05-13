/**
 * Backend comparison page.
 * Upload one file, run it on all available backends simultaneously,
 * then show side-by-side range profiles and a metrics summary table.
 */
import { useState, useCallback } from 'react';
import {
  Container, Typography, Box, Paper, Button, Alert,
  Grid, Chip, CircularProgress, Table, TableHead,
  TableBody, TableRow, TableCell, LinearProgress,
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, ReferenceLine,
} from 'recharts';

const BACKEND_COLORS: Record<string, string> = {
  pythonv2: '#2196f3',
  cuda:     '#4caf50',
  rst:      '#ff9800',
};

interface RangeRow { range: number; [key: string]: number | null | undefined; }

function buildComparisonData(results: Record<string, Record<string, (number|null)[]>>): RangeRow[] {
  const nranges = Math.max(...Object.values(results).map(r => r.velocity?.length ?? 0));
  return Array.from({ length: nranges }, (_, i) => {
    const row: RangeRow = { range: i };
    for (const [backend, r] of Object.entries(results)) {
      row[`vel_${backend}`]   = r.velocity?.[i]       ?? null;
      row[`width_${backend}`] = r.spectral_width?.[i] ?? null;
      row[`pwr_${backend}`]   = r.power?.[i]          ?? null;
    }
    return row;
  });
}

export default function ComparisonPage() {
  const [file, setFile]         = useState<File | null>(null);
  const [running, setRunning]   = useState(false);
  const [error, setError]       = useState<string | null>(null);
  const [done, setDone]         = useState<Record<string, Record<string,any>>>({});
  const [progress, setProgress] = useState<Record<string, number>>({});

  const onDrop = useCallback((accepted: File[]) => {
    if (accepted[0]) { setFile(accepted[0]); setDone({}); setError(null); }
  }, []);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/octet-stream': ['.rawacf', '.fitacf'] },
    multiple: false,
  });

  const runAll = async () => {
    if (!file) return;
    setRunning(true); setError(null); setDone({}); setProgress({});
    try {
      // Upload once
      const fd = new FormData();
      fd.append('file', file);
      const upResp = await fetch('/api/upload/', { method: 'POST', body: fd });
      const { file_id } = await upResp.json();

      // Fetch available backends
      const bResp   = await fetch('/api/processing/backends');
      const bData   = await bResp.json();
      const backends: string[] = bData.backends
        .filter((b: any) => b.available)
        .map((b: any) => b.id);

      // Start all jobs in parallel
      const jobIds: Record<string, string> = {};
      await Promise.all(backends.map(async bid => {
        const r = await fetch('/api/processing/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            file_id, backend: bid,
            stages: ['acf', 'fitacf'],
            parameters: { min_power: 3.0, xcf_enabled: true, elevation_enabled: true,
                           phase_tolerance: 25.0, batch_size: 64 },
            mode: 'auto',
          }),
        });
        const { job_id } = await r.json();
        jobIds[bid] = job_id;
        setProgress(p => ({ ...p, [bid]: 0 }));
      }));

      // Poll all jobs until done
      const remaining = new Set(backends);
      while (remaining.size > 0) {
        await new Promise(r => setTimeout(r, 800));
        for (const bid of [...remaining]) {
          const s = await fetch(`/api/processing/status/${jobIds[bid]}`).then(r => r.json());
          setProgress(p => ({ ...p, [bid]: s.progress }));
          if (s.status === 'completed') {
            const res = await fetch(`/api/results/${jobIds[bid]}`).then(r => r.json());
            setDone(d => ({ ...d, [bid]: res.stages?.fitacf ?? {} }));
            remaining.delete(bid);
          } else if (s.status === 'failed') {
            setDone(d => ({ ...d, [bid]: { _error: s.error } }));
            remaining.delete(bid);
          }
        }
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setRunning(false);
    }
  };

  const backends = Object.keys(done).filter(b => !done[b]._error);
  const faData: Record<string, Record<string, (number|null)[]>> = {};
  for (const b of backends) {
    faData[b] = done[b] as Record<string, (number|null)[]>;
  }
  const rows = backends.length > 0 ? buildComparisonData(faData) : [];

  return (
    <Container maxWidth="xl">
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
        <CompareArrowsIcon color="primary" sx={{ fontSize: 36 }} />
        <Typography variant="h4">Backend Comparison</Typography>
      </Box>

      <Grid container spacing={3}>
        {/* Upload */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>1. Upload File</Typography>
            <Box {...getRootProps()} sx={{
              border: '2px dashed', borderColor: isDragActive ? 'primary.main' : 'grey.600',
              borderRadius: 2, p: 3, textAlign: 'center', cursor: 'pointer',
              bgcolor: isDragActive ? 'action.hover' : 'transparent',
            }}>
              <input {...getInputProps()} />
              <CloudUploadIcon sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
              <Typography variant="body2">
                {file ? file.name : isDragActive ? 'Drop here…' : 'Drag rawacf file or click'}
              </Typography>
            </Box>
            <Button variant="contained" fullWidth sx={{ mt: 2 }}
              onClick={runAll} disabled={!file || running}
              startIcon={running ? <CircularProgress size={18} /> : <CompareArrowsIcon />}>
              {running ? 'Running…' : 'Run All Backends'}
            </Button>
            {error && <Alert severity="error" sx={{ mt: 1 }}>{error}</Alert>}
          </Paper>

          {/* Progress */}
          {Object.entries(progress).map(([bid, pct]) => (
            <Paper key={bid} sx={{ p: 2, mt: 1 }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                <Chip label={bid} size="small"
                  sx={{ bgcolor: BACKEND_COLORS[bid], color: '#fff' }} />
                <Typography variant="caption">
                  {done[bid] ? (done[bid]._error ? 'failed' : 'done') : `${pct}%`}
                </Typography>
              </Box>
              <LinearProgress variant="determinate" value={pct}
                color={done[bid]?._error ? 'error' : 'primary'} />
            </Paper>
          ))}

          {/* Metrics table */}
          {backends.length > 0 && (
            <Paper sx={{ p: 2, mt: 2 }}>
              <Typography variant="subtitle2" gutterBottom>Summary</Typography>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Backend</TableCell>
                    <TableCell align="right">Good</TableCell>
                    <TableCell align="right">Avg|vel|</TableCell>
                    <TableCell align="right">Avg W</TableCell>
                    <TableCell align="right">Δvel vs pv2</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {(() => {
                    const baseline = faData['pythonv2'];
                    const baseVel  = baseline
                      ? (baseline.velocity ?? []).filter((v): v is number => v !== null)
                      : [];
                    const avgNum = (arr: number[]) =>
                      arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : NaN;
                    const baseAvgVel = avgNum(baseVel.map(Math.abs));

                    return backends.map(bid => {
                      const fa  = faData[bid];
                      const vel = (fa.velocity ?? []).filter((v): v is number => v !== null);
                      const wid = (fa.spectral_width ?? []).filter((v): v is number => v !== null);
                      const avgV = avgNum(vel.map(Math.abs));
                      const delta = bid !== 'pythonv2' && !isNaN(baseAvgVel) && !isNaN(avgV)
                        ? (avgV - baseAvgVel).toFixed(0)
                        : '—';
                      const fmt = (n: number) => isNaN(n) ? '—' : n.toFixed(0);
                      return (
                        <TableRow key={bid}>
                          <TableCell>
                            <Chip label={bid} size="small"
                              sx={{ bgcolor: BACKEND_COLORS[bid], color: '#fff' }} />
                          </TableCell>
                          <TableCell align="right">{fa.good_ranges ?? '?'}</TableCell>
                          <TableCell align="right">{fmt(avgV)}</TableCell>
                          <TableCell align="right">{fmt(avgNum(wid))}</TableCell>
                          <TableCell align="right"
                            sx={{ color: delta !== '—' && parseFloat(delta) !== 0 ? 'warning.main' : 'text.secondary' }}>
                            {delta !== '—' ? `${parseFloat(delta) >= 0 ? '+' : ''}${delta}` : '—'}
                          </TableCell>
                        </TableRow>
                      );
                    });
                  })()}
                </TableBody>
              </Table>
            </Paper>
          )}
        </Grid>

        {/* Charts */}
        <Grid item xs={12} md={8}>
          {backends.length === 0 && !running && (
            <Paper sx={{ p: 6, textAlign: 'center' }}>
              <Typography color="text.secondary">
                Upload a file and click "Run All Backends" to compare outputs.
              </Typography>
            </Paper>
          )}

          {backends.length > 0 && (
            <Paper sx={{ p: 3 }}>
              <Typography variant="subtitle1" gutterBottom>Velocity (m/s) vs Range Gate</Typography>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={rows} margin={{ left: 10, right: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis dataKey="range" label={{ value: 'Range gate', position: 'insideBottom', offset: -4 }} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <ReferenceLine y={0} stroke="#555" />
                  {backends.map(bid => (
                    <Line key={bid} type="monotone" dataKey={`vel_${bid}`}
                      stroke={BACKEND_COLORS[bid]} dot={false} connectNulls={false}
                      strokeWidth={1.5} name={bid} />
                  ))}
                </LineChart>
              </ResponsiveContainer>

              <Typography variant="subtitle1" gutterBottom sx={{ mt: 3 }}>Spectral Width (m/s)</Typography>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={rows} margin={{ left: 10, right: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis dataKey="range" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  {backends.map(bid => (
                    <Line key={bid} type="monotone" dataKey={`width_${bid}`}
                      stroke={BACKEND_COLORS[bid]} dot={false} connectNulls={false}
                      strokeWidth={1.5} name={bid} />
                  ))}
                </LineChart>
              </ResponsiveContainer>

              <Typography variant="subtitle1" gutterBottom sx={{ mt: 3 }}>Power</Typography>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={rows} margin={{ left: 10, right: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                  <XAxis dataKey="range" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  {backends.map(bid => (
                    <Line key={bid} type="monotone" dataKey={`pwr_${bid}`}
                      stroke={BACKEND_COLORS[bid]} dot={false} connectNulls={false}
                      strokeWidth={1.5} name={bid} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </Paper>
          )}
        </Grid>
      </Grid>
    </Container>
  );
}
