import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container, Typography, Box, Paper, Button, Chip, Alert,
  Table, TableHead, TableBody, TableRow, TableCell,
  TableContainer, Checkbox, IconButton, Tooltip, LinearProgress,
  Dialog, DialogTitle, DialogContent, DialogActions,
  FormControl, InputLabel, Select, MenuItem, Divider,
  CircularProgress, TablePagination, Snackbar,
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import DeleteIcon from '@mui/icons-material/Delete';
import VisibilityIcon from '@mui/icons-material/Visibility';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import RefreshIcon from '@mui/icons-material/Refresh';
import StopIcon from '@mui/icons-material/Stop';

// ── types ─────────────────────────────────────────────────────────────────────

interface JobSummary {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  created_at: string;
  completed_at?: string;
  mode: string;
  summary?: {
    good_ranges?: number;
    nranges?: number;
    backend?: string;
    processing_time?: number;
  } | null;
}

interface BackendInfo { id: string; name: string; available: boolean; active: boolean; }

// ── helpers ───────────────────────────────────────────────────────────────────

const STATUS_COLORS: Record<string, 'default' | 'primary' | 'success' | 'error' | 'warning'> = {
  queued:    'default',
  running:   'primary',
  completed: 'success',
  failed:    'error',
  cancelled: 'warning',
};

function fmtTime(iso?: string) {
  if (!iso) return '—';
  return new Date(iso).toLocaleTimeString();
}

// ── Batch upload dialog ───────────────────────────────────────────────────────

function BatchUploadDialog({
  open, onClose, backends, onJobsStarted,
}: {
  open: boolean;
  onClose: () => void;
  backends: BackendInfo[];
  onJobsStarted: () => void;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [backend, setBackend] = useState('');
  const [stages, setStages] = useState(['acf', 'fitacf', 'grid']);
  const [submitting, setSubmitting] = useState(false);
  const [progress, setProgress] = useState(0);

  const onDrop = useCallback((accepted: File[]) => {
    setFiles(prev => [...prev, ...accepted]);
  }, []);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/octet-stream': ['.rawacf', '.fitacf', '.grid'] },
  });

  const handleSubmit = async () => {
    setSubmitting(true);
    setProgress(0);
    let done = 0;
    for (const file of files) {
      // Upload
      const fd = new FormData();
      fd.append('file', file);
      const up = await fetch('/api/upload/', { method: 'POST', body: fd });
      const { file_id } = await up.json();
      // Start job
      await fetch('/api/processing/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_id,
          mode: 'auto',
          backend: backend || null,
          stages,
          parameters: { min_power: 3.0, xcf_enabled: true, elevation_enabled: true,
                         phase_tolerance: 25.0, batch_size: 64 },
        }),
      });
      done++;
      setProgress(Math.round((done / files.length) * 100));
    }
    setSubmitting(false);
    setFiles([]);
    onJobsStarted();
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Batch Processing</DialogTitle>
      <DialogContent>
        <Box
          {...getRootProps()}
          sx={{
            border: '2px dashed',
            borderColor: isDragActive ? 'primary.main' : 'grey.600',
            borderRadius: 2, p: 3, textAlign: 'center', cursor: 'pointer', mb: 2,
            bgcolor: isDragActive ? 'action.hover' : 'transparent',
          }}
        >
          <input {...getInputProps()} />
          <CloudUploadIcon sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
          <Typography variant="body2">
            {isDragActive ? 'Drop files here…' : 'Drag & drop files, or click to select'}
          </Typography>
        </Box>

        {files.length > 0 && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              {files.length} file(s) selected:
            </Typography>
            {files.map((f, i) => (
              <Chip key={i} label={f.name} size="small" sx={{ m: 0.25 }}
                onDelete={() => setFiles(prev => prev.filter((_, j) => j !== i))} />
            ))}
          </Box>
        )}

        <Divider sx={{ my: 2 }} />

        <FormControl fullWidth sx={{ mb: 2 }}>
          <InputLabel>Backend</InputLabel>
          <Select value={backend} onChange={e => setBackend(e.target.value)} label="Backend">
            <MenuItem value="">Server default</MenuItem>
            {backends.filter(b => b.available).map(b => (
              <MenuItem key={b.id} value={b.id}>
                {b.name} {b.active ? '(default)' : ''}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <FormControl fullWidth>
          <InputLabel>Stages</InputLabel>
          <Select
            multiple value={stages}
            onChange={e => setStages(typeof e.target.value === 'string' ? [e.target.value] : e.target.value)}
            label="Stages"
            renderValue={v => (v as string[]).join(', ')}
          >
            {['acf', 'fitacf', 'lmfit', 'grid', 'cnvmap'].map(s => (
              <MenuItem key={s} value={s}>{s}</MenuItem>
            ))}
          </Select>
        </FormControl>

        {submitting && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="caption">Submitting jobs… {progress}%</Typography>
            <LinearProgress variant="determinate" value={progress} />
          </Box>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} disabled={submitting}>Cancel</Button>
        <Button onClick={handleSubmit} variant="contained"
          disabled={files.length === 0 || submitting}
          startIcon={submitting ? <CircularProgress size={16} /> : <PlayArrowIcon />}>
          Submit {files.length > 0 ? `${files.length} jobs` : ''}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function JobsPage() {
  const navigate = useNavigate();
  const [jobs, setJobs]           = useState<JobSummary[]>([]);
  const [selected, setSelected]   = useState<Set<string>>(new Set());
  const [backends, setBackends]   = useState<BackendInfo[]>([]);
  const [loading, setLoading]     = useState(true);
  const [batchOpen, setBatchOpen] = useState(false);
  const [page, setPage]           = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);
  const [notification, setNotification] = useState<{msg: string; severity: 'success'|'error'} | null>(null);

  const fetchJobs = async () => {
    try {
      const r = await fetch('/api/processing/list');
      const data = await r.json();
      setJobs(data.jobs || []);
    } finally {
      setLoading(false);
    }
  };

  const fetchBackends = async () => {
    try {
      const r = await fetch('/api/processing/backends');
      const d = await r.json();
      setBackends(d.backends || []);
    } catch {}
  };

  useEffect(() => {
    fetchJobs();
    fetchBackends();

    // Global WebSocket for live notifications
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/progress`;
    let ws: WebSocket;
    try {
      ws = new WebSocket(wsUrl);
      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          if (msg.type === 'job_update') {
            if (msg.status === 'completed')
              setNotification({ msg: `Job ${msg.job_id.slice(0,8)} completed`, severity: 'success' });
            else if (msg.status === 'failed')
              setNotification({ msg: `Job ${msg.job_id.slice(0,8)} failed`, severity: 'error' });
            fetchJobs();
          }
        } catch {}
      };
    } catch {}
    return () => ws?.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // WS subscription above handles live updates — no REST polling needed

  const toggleSelect = (id: string) => {
    setSelected(s => {
      const n = new Set(s);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });
  };
  const selectAll = () => {
    if (selected.size === jobs.length) setSelected(new Set());
    else setSelected(new Set(jobs.map(j => j.job_id)));
  };

  const deleteSelected = async () => {
    await Promise.all([...selected].map(id =>
      fetch(`/api/results/${id}`, { method: 'DELETE' }).catch(() => {})
    ));
    setSelected(new Set());
    fetchJobs();
  };

  const counts = {
    running:   jobs.filter(j => j.status === 'running').length,
    completed: jobs.filter(j => j.status === 'completed').length,
    failed:    jobs.filter(j => j.status === 'failed').length,
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3, gap: 2, flexWrap: 'wrap' }}>
        <Typography variant="h4" sx={{ flexGrow: 1 }}>Jobs</Typography>

        {/* Summary chips */}
        {counts.running   > 0 && <Chip label={`${counts.running} running`}   color="primary" />}
        {counts.completed > 0 && <Chip label={`${counts.completed} done`}    color="success" />}
        {counts.failed    > 0 && <Chip label={`${counts.failed} failed`}     color="error"   />}

        <Tooltip title="Refresh">
          <IconButton onClick={fetchJobs}><RefreshIcon /></IconButton>
        </Tooltip>
        <Button variant="contained" startIcon={<CloudUploadIcon />}
          onClick={() => setBatchOpen(true)}>
          Batch Upload
        </Button>
      </Box>

      {/* Bulk action bar */}
      {selected.size > 0 && (
        <Alert severity="info" sx={{ mb: 2 }}
          action={
            <Button color="error" size="small" startIcon={<DeleteIcon />}
              onClick={deleteSelected}>
              Delete {selected.size} selected
            </Button>
          }>
          {selected.size} job{selected.size > 1 ? 's' : ''} selected
        </Alert>
      )}

      <TableContainer component={Paper}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell padding="checkbox">
                <Checkbox
                  indeterminate={selected.size > 0 && selected.size < jobs.length}
                  checked={jobs.length > 0 && selected.size === jobs.length}
                  onChange={selectAll}
                />
              </TableCell>
              <TableCell>Status</TableCell>
              <TableCell>Job ID</TableCell>
              <TableCell>Backend</TableCell>
              <TableCell align="right">Good ranges</TableCell>
              <TableCell align="right">Time (s)</TableCell>
              <TableCell>Created</TableCell>
              <TableCell>Completed</TableCell>
              <TableCell align="center">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {loading && (
              <TableRow>
                <TableCell colSpan={9} align="center">
                  <CircularProgress size={24} />
                </TableCell>
              </TableRow>
            )}
            {!loading && jobs.length === 0 && (
              <TableRow>
                <TableCell colSpan={9} align="center">
                  <Typography color="text.secondary" sx={{ py: 4 }}>
                    No jobs yet. Upload a file on the Processing page or use Batch Upload.
                  </Typography>
                </TableCell>
              </TableRow>
            )}
            {jobs.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage).map(job => (
              <TableRow key={job.job_id}
                hover selected={selected.has(job.job_id)}
                sx={{ opacity: job.status === 'cancelled' ? 0.5 : 1 }}>
                <TableCell padding="checkbox">
                  <Checkbox checked={selected.has(job.job_id)}
                    onChange={() => toggleSelect(job.job_id)} />
                </TableCell>
                <TableCell>
                  <Box>
                    <Chip label={job.status} color={STATUS_COLORS[job.status]}
                      size="small" sx={{ mb: job.status === 'running' ? 0.5 : 0 }} />
                    {job.status === 'running' && (
                      <LinearProgress variant="determinate" value={job.progress}
                        sx={{ height: 3, borderRadius: 1 }} />
                    )}
                  </Box>
                </TableCell>
                <TableCell>
                  <Tooltip title="Click to copy full job ID">
                    <Typography variant="caption" fontFamily="monospace"
                      sx={{ cursor: 'pointer', '&:hover': { color: 'primary.main' } }}
                      onClick={() => navigator.clipboard?.writeText(job.job_id)}>
                      {job.job_id.slice(0, 8)}…
                    </Typography>
                  </Tooltip>
                </TableCell>
                <TableCell>
                  <Chip label={job.summary?.backend || job.mode || '—'}
                    size="small" variant="outlined" />
                </TableCell>
                <TableCell align="right">
                  {job.summary?.good_ranges !== undefined
                    ? `${job.summary.good_ranges} / ${job.summary.nranges ?? '?'}`
                    : '—'}
                </TableCell>
                <TableCell align="right">
                  {job.summary?.processing_time !== undefined
                    ? job.summary.processing_time.toFixed(2)
                    : '—'}
                </TableCell>
                <TableCell><Typography variant="caption">{fmtTime(job.created_at)}</Typography></TableCell>
                <TableCell><Typography variant="caption">{fmtTime(job.completed_at)}</Typography></TableCell>
                <TableCell align="center">
                  <Tooltip title="View results">
                    <span>
                      <IconButton size="small"
                        disabled={job.status !== 'completed'}
                        onClick={() => navigate(`/visualization/${job.job_id}`)}>
                        <VisibilityIcon fontSize="small" />
                      </IconButton>
                    </span>
                  </Tooltip>
                  {job.status === 'running' && (
                    <Tooltip title="Cancel job">
                      <IconButton size="small" color="warning"
                        onClick={() => {
                          fetch(`/api/processing/${job.job_id}`, { method: 'DELETE' }).catch(() => {});
                          setTimeout(fetchJobs, 400);
                        }}>
                        <StopIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  )}
                  <Tooltip title="Delete">
                    <IconButton size="small" color="error"
                      onClick={() => {
                        fetch(`/api/results/${job.job_id}`, { method: 'DELETE' }).catch(() => {});
                        setTimeout(fetchJobs, 200);
                      }}>
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <TablePagination
        component="div"
        count={jobs.length}
        page={page}
        onPageChange={(_, p) => setPage(p)}
        rowsPerPage={rowsPerPage}
        onRowsPerPageChange={e => { setRowsPerPage(+e.target.value); setPage(0); }}
        rowsPerPageOptions={[10, 25, 50, 100]}
      />

      <BatchUploadDialog
        open={batchOpen}
        onClose={() => setBatchOpen(false)}
        backends={backends}
        onJobsStarted={fetchJobs}
      />

      <Snackbar open={!!notification} autoHideDuration={4000}
        onClose={() => setNotification(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}>
        <Alert severity={notification?.severity ?? 'info'} onClose={() => setNotification(null)}>
          {notification?.msg}
        </Alert>
      </Snackbar>
    </Container>
  );
}
