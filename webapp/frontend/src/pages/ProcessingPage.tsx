import { useState, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Typography,
  Grid,
  Paper,
  Box,
  Button,
  Slider,
  FormControlLabel,
  FormGroup,
  Switch,
  LinearProgress,
  Alert,
  Chip,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  CircularProgress,
  Menu,
  MenuItem,
} from '@mui/material';
import TuneIcon from '@mui/icons-material/Tune';
import { useDropzone } from 'react-dropzone';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import MemoryIcon from '@mui/icons-material/Memory';
import StorageIcon from '@mui/icons-material/Storage';
import CodeIcon from '@mui/icons-material/Code';

interface ProcessingParameters {
  minPower: number;
  phaseTolerance: number;
  elevationEnabled: boolean;
  elevationModel: string;
  batchSize: number;
  xcfEnabled: boolean;
}

interface BackendInfo {
  id: string;
  name: string;
  available: boolean;
  gpu: boolean;
  active: boolean;
  error?: string;
}

const BACKEND_ICONS: Record<string, React.ReactNode> = {
  pythonv2: <CodeIcon fontSize="small" />,
  cuda:     <MemoryIcon fontSize="small" />,
  rst:      <StorageIcon fontSize="small" />,
};

const BACKEND_DESCRIPTIONS: Record<string, string> = {
  pythonv2: 'Python / CuPy — two-pass Bendat-Piersol LS fit, sigma width, vectorised lag validation',
  cuda:     'CUDArst — CUDA kernels with CPU fallback, same RST-accurate algorithms',
  rst:      'RST reference — original C library subprocess or numpy fallback',
};

export default function ProcessingPage() {
  const navigate = useNavigate();
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [fileId, setFileId] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentJobId, setCurrentJobId] = useState<string | null>(null);
  const [selectedBackend, setSelectedBackend] = useState<string | null>(null);
  const [backends, setBackends] = useState<BackendInfo[]>([]);
  const [backendsLoading, setBackendsLoading] = useState(true);
  const [selectedStages, setSelectedStages] = useState<string[]>(['acf', 'fitacf', 'grid']);
  const [presets, setPresets] = useState<Record<string, any>>({});
  const [presetAnchor, setPresetAnchor] = useState<HTMLElement | null>(null);
  const [parameters, setParameters] = useState<ProcessingParameters>({
    minPower: 3.0,
    phaseTolerance: 25.0,
    elevationEnabled: true,
    elevationModel: 'GSM',
    batchSize: 64,
    xcfEnabled: true,
  });

  // Fetch presets on mount
  useEffect(() => {
    fetch('/api/settings/presets')
      .then(r => r.json())
      .then(d => setPresets(d.presets || {}))
      .catch(() => {});
  }, []);

  // Fetch available backends on mount
  useEffect(() => {
    fetch('/api/processing/backends')
      .then(r => r.json())
      .then(data => {
        const list: BackendInfo[] = data.backends || [];
        setBackends(list);
        // Pre-select the active backend
        const active = list.find(b => b.active);
        if (active) setSelectedBackend(active.id);
      })
      .catch(() => {
        // Fallback: show all three as available if probe fails
        setBackends([
          { id: 'pythonv2', name: 'pythonv2', available: true, gpu: false, active: true },
          { id: 'cuda',     name: 'CUDArst',  available: true, gpu: false, active: false },
          { id: 'rst',      name: 'RST',       available: true, gpu: false, active: false },
        ]);
        setSelectedBackend('pythonv2');
      })
      .finally(() => setBackendsLoading(false));
  }, []);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setUploadedFile(file);
      
      // Upload file to backend
      const formData = new FormData();
      formData.append('file', file);
      
      try {
        const response = await fetch('/api/upload/', {
          method: 'POST',
          body: formData,
        });
        
        if (response.ok) {
          const data = await response.json();
          setFileId(data.file_id);
        }
      } catch (error) {
        console.error('Upload failed:', error);
      }
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/octet-stream': ['.rawacf', '.fitacf', '.grid'],
    },
    multiple: false,
  });

  const handleStartProcessing = async () => {
    if (!fileId) return;
    
    setIsProcessing(true);
    setProgress(0);
    
    try {
      const response = await fetch('/api/processing/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          file_id: fileId,
          mode: 'auto',
          backend: selectedBackend,
          parameters: {
            min_power: parameters.minPower,
            phase_tolerance: parameters.phaseTolerance,
            elevation_enabled: parameters.elevationEnabled,
            elevation_model: parameters.elevationModel,
            batch_size: parameters.batchSize,
            xcf_enabled: parameters.xcfEnabled,
          },
          stages: selectedStages,
        }),
      });
      
      if (response.ok) {
        const data = await response.json();
        setCurrentJobId(data.job_id);
        
        // Poll for progress
        pollJobStatus(data.job_id);
      }
    } catch (error) {
      console.error('Processing failed:', error);
      setIsProcessing(false);
    }
  };

  const pollJobStatus = (targetJobId: string) => {
    // Use WebSocket for live progress; fall back to REST polling if WS unavailable
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/progress`;
    let ws: WebSocket | null = null;
    let fallbackInterval: ReturnType<typeof setInterval> | null = null;

    const cleanup = () => {
      ws?.close();
      if (fallbackInterval) clearInterval(fallbackInterval);
    };

    try {
      ws = new WebSocket(wsUrl);
      ws.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          if (msg.job_id !== targetJobId) return;
          if (msg.progress !== undefined) setProgress(msg.progress);
          if (msg.status === 'completed') {
            cleanup();
            setIsProcessing(false);
            navigate(`/visualization/${targetJobId}`);
          } else if (msg.status === 'failed') {
            cleanup();
            setIsProcessing(false);
          }
        } catch {}
      };
      ws.onerror = () => {
        ws = null;
        startFallbackPoll();
      };
    } catch {
      startFallbackPoll();
    }

    function startFallbackPoll() {
      fallbackInterval = setInterval(async () => {
        try {
          const r = await fetch(`/api/processing/status/${targetJobId}`);
          if (!r.ok) return;
          const d = await r.json();
          setProgress(d.progress);
          if (d.status === 'completed' || d.status === 'failed') {
            cleanup();
            setIsProcessing(false);
            if (d.status === 'completed') navigate(`/visualization/${targetJobId}`);
          }
        } catch {}
      }, 1500);
    }
  };

  return (
    <Container maxWidth="lg">
      <Typography variant="h3" component="h1" gutterBottom>
        Data Processing
      </Typography>
      
      <Grid container spacing={3}>
        {/* File Upload Section */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Typography variant="h5" gutterBottom>
              1. Upload Data File
            </Typography>
            
            <Box
              {...getRootProps()}
              sx={{
                border: '2px dashed',
                borderColor: isDragActive ? 'primary.main' : 'grey.500',
                borderRadius: 2,
                p: 4,
                textAlign: 'center',
                cursor: 'pointer',
                bgcolor: isDragActive ? 'action.hover' : 'transparent',
                transition: 'all 0.2s',
                mt: 2,
              }}
            >
              <input {...getInputProps()} />
              <CloudUploadIcon sx={{ fontSize: 64, color: 'primary.main', mb: 2 }} />
              <Typography variant="body1" gutterBottom>
                {isDragActive
                  ? 'Drop file here...'
                  : 'Drag & drop SuperDARN file here, or click to select'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Supported formats: RAWACF, FITACF, GRID
              </Typography>
            </Box>
            
            {uploadedFile && (
              <Alert severity="success" sx={{ mt: 2 }}>
                File uploaded: {uploadedFile.name} ({(uploadedFile.size / 1024).toFixed(1)} KB)
              </Alert>
            )}
          </Paper>
        </Grid>
        
        {/* Parameters Section */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3, height: '100%' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              <Typography variant="h5" sx={{ flexGrow: 1 }}>
                2. Configure Parameters
              </Typography>
              {Object.keys(presets).length > 0 && (
                <>
                  <Button size="small" startIcon={<TuneIcon />}
                    onClick={e => setPresetAnchor(e.currentTarget)}>
                    Load preset
                  </Button>
                  <Menu anchorEl={presetAnchor} open={!!presetAnchor}
                    onClose={() => setPresetAnchor(null)}>
                    {Object.entries(presets).map(([id, p]: [string, any]) => (
                      <MenuItem key={id} onClick={() => {
                        const pp = p.parameters ?? {};
                        setParameters({
                          minPower:          pp.min_power         ?? parameters.minPower,
                          phaseTolerance:    pp.phase_tolerance   ?? parameters.phaseTolerance,
                          elevationEnabled:  pp.elevation_enabled ?? parameters.elevationEnabled,
                          elevationModel:    pp.elevation_model   ?? parameters.elevationModel,
                          batchSize:         pp.batch_size        ?? parameters.batchSize,
                          xcfEnabled:        pp.xcf_enabled       ?? parameters.xcfEnabled,
                        });
                        setPresetAnchor(null);
                      }}>
                        <Box>
                          <Typography variant="body2">{p.name}</Typography>
                          <Typography variant="caption" color="text.secondary">{p.description}</Typography>
                        </Box>
                      </MenuItem>
                    ))}
                  </Menu>
                </>
              )}
            </Box>
            
            <Box sx={{ mt: 3 }}>
              <Typography gutterBottom>Minimum Power: {parameters.minPower} dB</Typography>
              <Slider
                value={parameters.minPower}
                onChange={(_, value) => setParameters({ ...parameters, minPower: value as number })}
                min={0}
                max={10}
                step={0.5}
                marks
                valueLabelDisplay="auto"
              />
            </Box>
            
            <Box sx={{ mt: 3 }}>
              <Typography gutterBottom>Phase Tolerance: {parameters.phaseTolerance}°</Typography>
              <Slider
                value={parameters.phaseTolerance}
                onChange={(_, value) => setParameters({ ...parameters, phaseTolerance: value as number })}
                min={5}
                max={45}
                step={5}
                marks
                valueLabelDisplay="auto"
              />
            </Box>
            
            <Box sx={{ mt: 3 }}>
              <Typography gutterBottom>Batch Size: {parameters.batchSize} ranges</Typography>
              <Slider
                value={parameters.batchSize}
                onChange={(_, value) => setParameters({ ...parameters, batchSize: value as number })}
                min={16}
                max={256}
                step={16}
                marks
                valueLabelDisplay="auto"
              />
            </Box>
            
            <Box sx={{ mt: 3 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={parameters.elevationEnabled}
                    onChange={(e) => setParameters({ ...parameters, elevationEnabled: e.target.checked })}
                  />
                }
                label="Enable Elevation Correction"
              />
            </Box>
            
            <Box sx={{ mt: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={parameters.xcfEnabled}
                    onChange={(e) => setParameters({ ...parameters, xcfEnabled: e.target.checked })}
                  />
                }
                label="Enable XCF Processing"
              />
            </Box>
          </Paper>
        </Grid>
        
        {/* Stage selection with dependency enforcement */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>3. Pipeline Stages</Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              Stages run in order. Enabling a later stage automatically enables its prerequisites.
            </Typography>
            <FormGroup row>
              {(['acf', 'fitacf', 'lmfit', 'grid', 'cnvmap'] as const).map((s, idx, arr) => {
                // Dependencies: each stage requires all preceding stages
                const deps = arr.slice(0, idx);
                const isOn = selectedStages.includes(s);
                // A stage is required if any later stage is selected
                const isRequired = arr.slice(idx + 1).some(later => selectedStages.includes(later));
                return (
                  <Tooltip key={s} title={isRequired ? `Required by ${arr.slice(idx+1).filter(l => selectedStages.includes(l)).join(', ')}` : ''}>
                    <FormControlLabel
                      control={
                        <Switch
                          checked={isOn}
                          disabled={isRequired}
                          onChange={e => {
                            if (e.target.checked) {
                              // Enable this stage and all its prerequisites
                              setSelectedStages(prev => {
                                const toAdd = [...deps.filter(d => !prev.includes(d)), s];
                                return [...prev, ...toAdd];
                              });
                            } else {
                              // Disable this stage (downstream already checked via isRequired)
                              setSelectedStages(prev => prev.filter(x => x !== s));
                            }
                          }}
                        />
                      }
                      label={s.toUpperCase()}
                    />
                  </Tooltip>
                );
              })}
            </FormGroup>
          </Paper>
        </Grid>

        {/* Backend Selection */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              4. Select Processing Backend
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              All three backends implement the same RST v3.0-compatible algorithms.
              Switch to compare results.
            </Typography>

            {backendsLoading ? (
              <CircularProgress size={24} />
            ) : (
              <ToggleButtonGroup
                value={selectedBackend}
                exclusive
                onChange={(_, v) => { if (v !== null) setSelectedBackend(v); }}
                aria-label="processing backend"
              >
                {backends.map(b => (
                  <Tooltip
                    key={b.id}
                    title={
                      !b.available
                        ? `Unavailable: ${b.error || 'import failed'}`
                        : BACKEND_DESCRIPTIONS[b.id] || b.name
                    }
                    arrow
                  >
                    <span>   {/* span needed so Tooltip works on disabled buttons */}
                      <ToggleButton
                        value={b.id}
                        disabled={!b.available}
                        sx={{ px: 3, py: 1.5, gap: 1 }}
                      >
                        {BACKEND_ICONS[b.id]}
                        <Box sx={{ textAlign: 'left' }}>
                          <Typography variant="body2" fontWeight="bold">
                            {b.id}
                          </Typography>
                          {b.gpu && (
                            <Chip label="GPU" size="small" color="success" sx={{ height: 16, fontSize: 10 }} />
                          )}
                          {b.active && (
                            <Chip label="default" size="small" variant="outlined" sx={{ height: 16, fontSize: 10, ml: 0.5 }} />
                          )}
                        </Box>
                      </ToggleButton>
                    </span>
                  </Tooltip>
                ))}
              </ToggleButtonGroup>
            )}

            {selectedBackend && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                {BACKEND_DESCRIPTIONS[selectedBackend] || selectedBackend}
              </Typography>
            )}
          </Paper>
        </Grid>

        {/* Processing Control Section */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              5. Start Processing
            </Typography>

            <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
              <Button
                variant="contained"
                size="large"
                onClick={handleStartProcessing}
                disabled={!fileId || isProcessing || !selectedBackend}
              >
                {isProcessing ? 'Processing...' : 'Start Processing'}
              </Button>

              {selectedBackend && (
                <Chip
                  icon={BACKEND_ICONS[selectedBackend] as React.ReactElement}
                  label={selectedBackend}
                  color="primary"
                  variant="outlined"
                />
              )}
              {backends.find(b => b.id === selectedBackend)?.gpu && (
                <Chip label="GPU" color="success" size="small" />
              )}
            </Box>

            {isProcessing && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="body2" gutterBottom>
                  Processing ({selectedBackend}): {progress}%
                  {currentJobId && (
                    <Typography component="span" variant="caption"
                      color="text.secondary" sx={{ ml: 1 }}>
                      [{currentJobId.slice(0, 8)}]
                    </Typography>
                  )}
                </Typography>
                <LinearProgress variant="determinate" value={progress} />
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
}
