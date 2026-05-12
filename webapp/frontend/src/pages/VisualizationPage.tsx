import { useParams, useNavigate } from 'react-router-dom';
import {
  Container, Typography, Grid, Paper, Box, Tabs, Tab,
  Chip, CircularProgress, Alert, Button, FormGroup,
  FormControlLabel, Checkbox, Divider, Table, TableBody,
  TableCell, TableRow,
} from '@mui/material';
import { useState, useEffect, useRef, useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ResponsiveContainer, ReferenceLine, ScatterChart,
  Scatter, ZAxis, BarChart, Bar,
} from 'recharts';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import DownloadIcon from '@mui/icons-material/Download';

// ── types ─────────────────────────────────────────────────────────────────────

interface RangePoint {
  range: number;
  velocity: number | null;
  power: number | null;
  spectral_width: number | null;
  spectral_width_sigma: number | null;
  elevation: number | null;
}

interface VizData {
  job_id: string;
  stages: Record<string, Record<string, unknown>>;
  performance: Record<string, unknown>;
  plots: {
    rti?: { records: RTIRecord[]; nranges: number; nrecords: number };
    range_profile?: {
      velocity: (number | null)[];
      power: (number | null)[];
      spectral_width: (number | null)[];
      spectral_width_sigma: (number | null)[];
      elevation: (number | null)[];
      quality_flag: (number | null)[];
      nranges: number;
      good_ranges: number;
      backend: string;
      records?: RTIRecord[];   // per-beam arrays when multi-record
      nrecords?: number;
    };
    grid?: { lat: number[]; lon: number[]; velocity: (number | null)[] };
  };
}

// ── helpers ───────────────────────────────────────────────────────────────────

function buildRangeData(
  plots: VizData['plots'],
  record?: RTIRecord | null
): RangePoint[] {
  const rp = plots.range_profile;
  if (!rp) return [];
  // Use per-beam record if provided, otherwise fall back to top-level arrays
  const src = record ?? rp;
  const n = rp.nranges || rp.velocity?.length || 0;
  return Array.from({ length: n }, (_, i) => ({
    range: i,
    velocity:             (src as any).velocity?.[i]             ?? null,
    power:                (src as any).power?.[i]                ?? null,
    spectral_width:       (src as any).spectral_width?.[i]       ?? null,
    spectral_width_sigma: (rp as any).spectral_width_sigma?.[i]  ?? null,
    elevation:            (rp as any).elevation?.[i]             ?? null,
  }));
}

function TabPanel({ children, value, index }: {
  children: React.ReactNode; value: number; index: number;
}) {
  return (
    <div hidden={value !== index}>
      {value === index && <Box sx={{ p: 2 }}>{children}</Box>}
    </div>
  );
}

const FIELD_COLORS: Record<string, string> = {
  velocity:             '#2196f3',
  power:                '#ff9800',
  spectral_width:       '#4caf50',
  spectral_width_sigma: '#9c27b0',
  elevation:            '#f44336',
};

const FIELD_UNITS: Record<string, string> = {
  velocity:             'm/s',
  power:                'arb',
  spectral_width:       'm/s',
  spectral_width_sigma: 'm/s',
  elevation:            '°',
};

// ── Range Profile tab ─────────────────────────────────────────────────────────

function RangeProfileTab({ data, goodRanges, records, selectedBeam, onBeamSelect }: {
  data: RangePoint[];
  goodRanges: number;
  records?: RTIRecord[];
  selectedBeam?: number;
  onBeamSelect?: (beam: number) => void;
}) {
  const available = ['velocity', 'power', 'spectral_width', 'spectral_width_sigma', 'elevation']
    .filter(f => data.some(d => d[f as keyof RangePoint] !== null));

  const [shown, setShown] = useState<Record<string, boolean>>(() =>
    Object.fromEntries(available.map(f => [f, ['velocity','power','spectral_width'].includes(f)]))
  );

  const toggle = (f: string) => setShown(s => ({ ...s, [f]: !s[f] }));

  if (data.length === 0) return <Alert severity="info">No range profile data available.</Alert>;

  return (
    <Box>
      {/* Beam selector when multi-record data is available */}
      {records && records.length > 1 && onBeamSelect && (
        <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
          <Typography variant="body2" color="text.secondary">Beam:</Typography>
          {records.map(rec => (
            <Chip
              key={rec.beam}
              label={`Beam ${rec.beam}`}
              size="small"
              clickable
              color={selectedBeam === rec.beam ? 'primary' : 'default'}
              onClick={() => onBeamSelect(rec.beam)}
            />
          ))}
        </Box>
      )}
      <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
        <Typography variant="body2" color="text.secondary">Show:</Typography>
        {available.map(f => (
          <FormControlLabel
            key={f}
            control={<Checkbox checked={!!shown[f]} onChange={() => toggle(f)}
                       sx={{ color: FIELD_COLORS[f], '&.Mui-checked': { color: FIELD_COLORS[f] } }} />}
            label={`${f.replace(/_/g, ' ')} (${FIELD_UNITS[f]})`}
          />
        ))}
        <Chip size="small" label={`${goodRanges} good ranges`} color="primary" variant="outlined" />
      </Box>

      {/* Velocity chart */}
      {shown.velocity && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>Velocity (m/s) vs Range Gate</Typography>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={data} margin={{ left: 10, right: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="range" label={{ value: 'Range gate', position: 'insideBottom', offset: -5 }} />
              <YAxis />
              <Tooltip formatter={(v: number) => v !== null ? `${v.toFixed(1)} m/s` : '—'} />
              <ReferenceLine y={0} stroke="#666" />
              <Line type="monotone" dataKey="velocity" stroke={FIELD_COLORS.velocity}
                dot={false} connectNulls={false} strokeWidth={1.5} name="Velocity" />
            </LineChart>
          </ResponsiveContainer>
        </Box>
      )}

      {/* Power chart */}
      {shown.power && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>Power vs Range Gate</Typography>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={data} margin={{ left: 10, right: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="range" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="power" stroke={FIELD_COLORS.power}
                dot={false} connectNulls={false} strokeWidth={1.5} name="Power" />
            </LineChart>
          </ResponsiveContainer>
        </Box>
      )}

      {/* Width charts */}
      {(shown.spectral_width || shown.spectral_width_sigma) && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>Spectral Width (m/s) vs Range Gate</Typography>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={data} margin={{ left: 10, right: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="range" />
              <YAxis />
              <Tooltip formatter={(v: number) => v !== null ? `${v.toFixed(1)} m/s` : '—'} />
              <Legend />
              {shown.spectral_width && (
                <Line type="monotone" dataKey="spectral_width" stroke={FIELD_COLORS.spectral_width}
                  dot={false} connectNulls={false} strokeWidth={1.5} name="λ width" />
              )}
              {shown.spectral_width_sigma && (
                <Line type="monotone" dataKey="spectral_width_sigma" stroke={FIELD_COLORS.spectral_width_sigma}
                  dot={false} connectNulls={false} strokeWidth={1.5} strokeDasharray="5 3" name="σ width" />
              )}
            </LineChart>
          </ResponsiveContainer>
        </Box>
      )}

      {/* Elevation */}
      {shown.elevation && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>Elevation (°) vs Range Gate</Typography>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={data} margin={{ left: 10, right: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis dataKey="range" />
              <YAxis domain={[0, 90]} />
              <Tooltip formatter={(v: number) => v !== null ? `${v.toFixed(1)}°` : '—'} />
              <Line type="monotone" dataKey="elevation" stroke={FIELD_COLORS.elevation}
                dot={false} connectNulls={false} strokeWidth={1.5} name="Elevation" />
            </LineChart>
          </ResponsiveContainer>
        </Box>
      )}
    </Box>
  );
}

// ── RTI heatmap (canvas) ──────────────────────────────────────────────────────

type RTIField = 'velocity' | 'power' | 'spectral_width';

interface RTIRecord {
  beam: number;
  velocity?: (number | null)[];
  power?: (number | null)[];
  spectral_width?: (number | null)[];
  quality_flag?: (number | null)[];
  ground_scatter_flag?: (number | null)[];
}

function velocityColor(v: number, vmax = 2000): string {
  const t = Math.max(-1, Math.min(1, v / vmax));
  if (t < 0) {
    const b = Math.round(255 * (-t));
    return `rgb(${255 - b},${255 - b},255)`;
  }
  const r = Math.round(255 * t);
  return `rgb(255,${255 - r},${255 - r})`;
}

function powerColor(p: number, pmax = 60): string {
  const t = Math.max(0, Math.min(1, p / pmax));
  const g = Math.round(255 * t);
  return `rgb(0,${g},${Math.round(128 * (1 - t))})`;
}

function widthColor(w: number, wmax = 600): string {
  const t = Math.max(0, Math.min(1, w / wmax));
  const r = Math.round(255 * t), g = Math.round(128 * (1 - t));
  return `rgb(${r},${g},128)`;
}

function RTICanvas({ records, nranges, field, selectedBeam, onBeamClick }: {
  records: RTIRecord[];
  nranges: number;
  field: RTIField;
  selectedBeam?: number;
  onBeamClick?: (beam: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || records.length === 0 || nranges === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const LABEL_W = 44;   // px reserved for beam number labels on left
    const W = canvas.width, H = canvas.height;
    const plotW = W - LABEL_W;
    const cellW = plotW / nranges, cellH = H / records.length;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, W, H);

    records.forEach((rec, yi) => {
      const data = rec[field] ?? [];
      const qflg = rec.quality_flag ?? [];
      const y0 = Math.round(yi * cellH);
      const yH = Math.max(1, Math.ceil(cellH));

      // Highlight selected beam
      if (selectedBeam !== undefined && rec.beam === selectedBeam) {
        ctx.fillStyle = 'rgba(255,255,255,0.08)';
        ctx.fillRect(LABEL_W, y0, plotW, yH);
      }

      data.forEach((v, xi) => {
        if (v === null || v === undefined) return;
        if (qflg[xi] === 0) return;
        const color = field === 'velocity' ? velocityColor(v)
                    : field === 'power'    ? powerColor(v)
                    : widthColor(v);
        ctx.fillStyle = color;
        ctx.fillRect(LABEL_W + Math.round(xi * cellW), y0,
                     Math.max(1, Math.ceil(cellW)), yH);
      });
      // Ground scatter overlay
      (rec.ground_scatter_flag ?? []).forEach((gs, xi) => {
        if (!gs) return;
        ctx.fillStyle = 'rgba(255,255,255,0.18)';
        ctx.fillRect(LABEL_W + Math.round(xi * cellW), y0,
                     Math.max(1, Math.ceil(cellW)), yH);
      });

      // Beam label
      ctx.fillStyle = rec.beam === selectedBeam ? '#fff' : '#888';
      ctx.font = `${Math.min(11, Math.max(8, cellH - 2))}px monospace`;
      ctx.textAlign = 'right';
      ctx.fillText(`B${rec.beam}`, LABEL_W - 3, y0 + cellH * 0.7);
    });

    // Range gate axis
    ctx.fillStyle = '#666';
    ctx.font = '9px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('0', LABEL_W, H - 2);
    ctx.textAlign = 'right';
    ctx.fillText(`${nranges}`, W - 2, H - 2);
  }, [records, nranges, field, selectedBeam]);

  useEffect(() => { draw(); }, [draw]);

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!onBeamClick || records.length === 0) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect  = canvas.getBoundingClientRect();
    const scaleY = canvas.height / rect.height;
    const y      = (e.clientY - rect.top) * scaleY;
    const yi     = Math.floor(y / (canvas.height / records.length));
    if (yi >= 0 && yi < records.length) onBeamClick(records[yi].beam);
  }, [onBeamClick, records]);

  if (records.length === 0) return <Alert severity="info">No records — process with fitacf stage.</Alert>;
  return (
    <canvas ref={canvasRef} width={900} height={Math.max(200, records.length * 30)}
            style={{ width: '100%', imageRendering: 'pixelated', border: '1px solid #333',
                     cursor: onBeamClick ? 'pointer' : 'default' }}
            onClick={handleCanvasClick}
            title={onBeamClick ? 'Click a beam row to view its range profile' : undefined}
    />
  );
}

function RTITab({ rti, selectedBeam, onBeamClick }: {
  rti?: { records: RTIRecord[]; nranges: number; nrecords: number };
  selectedBeam?: number;
  onBeamClick?: (beam: number) => void;
}) {
  const [field, setField] = useState<RTIField>('velocity');
  if (!rti?.records?.length) {
    return <Alert severity="info">RTI data not available. Re-run with fitacf stage.</Alert>;
  }
  const legendItems = field === 'velocity'
    ? [['Blue', 'Towards'], ['Red', 'Away'], ['White overlay', 'Ground scatter']]
    : field === 'power'
      ? [['Green', 'High power'], ['Dark', 'Low/no signal']]
      : [['Red', 'Broad'], ['Blue', 'Narrow']];

  return (
    <Box>
      <Box sx={{ mb: 2, display: 'flex', gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
        <Typography variant="body2" color="text.secondary">Display:</Typography>
        {(['velocity', 'power', 'spectral_width'] as RTIField[]).map(f => (
          <Chip key={f} label={f.replace('_', ' ')} clickable
            color={field === f ? 'primary' : 'default'}
            onClick={() => setField(f)} />
        ))}
        <Typography variant="caption" color="text.secondary" sx={{ ml: 2 }}>
          {rti.nrecords} beams × {rti.nranges} ranges
        </Typography>
      </Box>
      <RTICanvas records={rti.records} nranges={rti.nranges} field={field}
                 selectedBeam={selectedBeam} onBeamClick={onBeamClick} />
      {onBeamClick && (
        <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
          Click a beam row to view its range profile
        </Typography>
      )}
      <Box sx={{ mt: 1, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
        {legendItems.map(([color, label]) => (
          <Typography key={label} variant="caption" color="text.secondary">
            ■ {color} = {label}
          </Typography>
        ))}
      </Box>
    </Box>
  );
}

// ── Grid tab ──────────────────────────────────────────────────────────────────

function GridTab({ stages }: { stages: Record<string, Record<string, unknown>> }) {
  const grid = stages.grid as { velocity?: (number | null)[]; nlat?: number; nlon?: number } | undefined;
  if (!grid?.velocity) return <Alert severity="info">No grid data — run with the grid stage enabled.</Alert>;

  const vals = grid.velocity
    .map((v, i) => ({ i, v }))
    .filter(x => x.v !== null && x.v !== undefined) as { i: number; v: number }[];

  const buckets = 40;
  const min = Math.min(...vals.map(x => x.v));
  const max = Math.max(...vals.map(x => x.v));
  const step = (max - min) / buckets || 1;
  const hist = Array.from({ length: buckets }, (_, b) => ({
    bin: Math.round(min + b * step),
    count: vals.filter(x => x.v >= min + b * step && x.v < min + (b + 1) * step).length,
  }));

  return (
    <Box>
      <Typography variant="subtitle2" gutterBottom>
        Grid velocity distribution ({vals.length} / {grid.velocity.length} cells filled)
      </Typography>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={hist} margin={{ left: 10, right: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis dataKey="bin" label={{ value: 'Velocity (m/s)', position: 'insideBottom', offset: -5 }} />
          <YAxis label={{ value: 'Cells', angle: -90, position: 'insideLeft' }} />
          <Tooltip />
          <ReferenceLine x={0} stroke="#666" />
          <Bar dataKey="count" fill="#2196f3" name="Grid cells" />
        </BarChart>
      </ResponsiveContainer>
    </Box>
  );
}

// ── Performance tab ───────────────────────────────────────────────────────────

function PerformanceTab({ perf, stages }: {
  perf: Record<string, unknown>;
  stages: Record<string, Record<string, unknown>>;
}) {
  const timing = (perf.stage_timing || {}) as Record<string, number>;
  const totalTime = (perf.total_time || 0) as number;
  const backend = (perf.backend || '?') as string;
  const mode = (perf.mode || '?') as string;

  const timingData = Object.entries(timing).map(([stage, t]) => ({
    stage, time: +(t as number).toFixed(3),
  }));

  return (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>Timing</Typography>
          <Table size="small">
            <TableBody>
              {timingData.map(({ stage, time }) => (
                <TableRow key={stage}>
                  <TableCell>{stage}</TableCell>
                  <TableCell align="right">{time}s</TableCell>
                  <TableCell align="right" sx={{ width: 120 }}>
                    <Box sx={{ bgcolor: 'primary.main', height: 8, borderRadius: 1,
                      width: `${Math.min(100, (time / totalTime) * 100)}%` }} />
                  </TableCell>
                </TableRow>
              ))}
              <TableRow>
                <TableCell><strong>Total</strong></TableCell>
                <TableCell align="right"><strong>{(totalTime as number).toFixed(3)}s</strong></TableCell>
                <TableCell />
              </TableRow>
            </TableBody>
          </Table>
        </Paper>
      </Grid>
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>Run Info</Typography>
          <Table size="small">
            <TableBody>
              <TableRow><TableCell>Backend</TableCell><TableCell>{backend}</TableCell></TableRow>
              <TableRow><TableCell>Mode</TableCell><TableCell>{mode}</TableCell></TableRow>
              {Object.entries(stages).map(([stage, data]) => {
                const d = data as Record<string, unknown>;
                return (
                  <TableRow key={stage}>
                    <TableCell>{stage}</TableCell>
                    <TableCell>
                      {d.good_ranges !== undefined
                        ? `${d.good_ranges} / ${d.nranges} good`
                        : d.nranges !== undefined
                          ? `${d.nranges} ranges`
                          : '✓'}
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </Paper>
      </Grid>
    </Grid>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function VisualizationPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const navigate = useNavigate();
  const [tab, setTab] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [vizData, setVizData] = useState<VizData | null>(null);
  const [selectedBeam, setSelectedBeam] = useState<number | undefined>(undefined);

  useEffect(() => {
    if (!jobId || jobId === 'demo') {
      setLoading(false);
      setError('No job selected. Start a processing job first.');
      return;
    }
    fetch(`/api/results/${jobId}/visualization`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d: VizData) => { setVizData(d); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [jobId]);

  const rp = vizData?.plots.range_profile;
  const activeRecord = rp?.records?.find(r => r.beam === selectedBeam) ?? null;
  const rangeData  = vizData ? buildRangeData(vizData.plots, activeRecord) : [];
  const goodRanges = (activeRecord as any)?.good_ranges ?? rp?.good_ranges ?? 0;

  const handleBeamClick = useCallback((beam: number) => {
    setSelectedBeam(beam);
    setTab(0);   // switch to range profile tab
  }, []);

  if (loading) return (
    <Container>
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 10 }}>
        <CircularProgress />
      </Box>
    </Container>
  );

  return (
    <Container maxWidth="xl">
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
        <Button startIcon={<ArrowBackIcon />} onClick={() => navigate('/jobs')}>Jobs</Button>
        <Typography variant="h4" sx={{ flexGrow: 1 }}>Results</Typography>
        <Button startIcon={<CompareArrowsIcon />} variant="outlined" size="small"
          onClick={() => navigate('/compare')}>
          Compare backends
        </Button>
        {jobId && jobId !== 'demo' && (
          <Button startIcon={<DownloadIcon />} variant="outlined" size="small"
            href={`/api/results/${jobId}/export/csv`} download>
            Download CSV
          </Button>
        )}
      </Box>

      {vizData?.plots.range_profile?.backend && (
        <Box sx={{ mb: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Chip label={`Backend: ${vizData.plots.range_profile.backend}`} color="primary" size="small" />
          <Chip label={`Job: ${jobId}`} variant="outlined" size="small" />
          <Chip label={`${goodRanges} good ranges`} color="success" size="small" />
        </Box>
      )}

      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

      {vizData && (
        <Paper sx={{ mt: 1 }}>
          <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tab label="Range Profile" />
            <Tab label="RTI" />
            <Tab label="Grid" />
            <Tab label="Performance" />
          </Tabs>

          <TabPanel value={tab} index={0}>
            <RangeProfileTab
              data={rangeData}
              goodRanges={goodRanges}
              records={rp?.records}
              selectedBeam={selectedBeam}
              onBeamSelect={setSelectedBeam}
            />
          </TabPanel>
          <TabPanel value={tab} index={1}>
            <RTITab
              rti={vizData.plots.rti as any}
              selectedBeam={selectedBeam}
              onBeamClick={handleBeamClick}
            />
          </TabPanel>
          <TabPanel value={tab} index={2}>
            <GridTab stages={vizData.stages} />
          </TabPanel>
          <TabPanel value={tab} index={3}>
            <PerformanceTab perf={vizData.performance} stages={vizData.stages} />
          </TabPanel>
        </Paper>
      )}
    </Container>
  );
}
