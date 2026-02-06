import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Grid,
  Button,
  CircularProgress,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  LinearProgress,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tooltip,
  IconButton,
} from '@mui/material';
import {
  PlayArrow,
  Refresh,
  ExpandMore,
  CheckCircle,
  Error,
  Speed,
  Memory,
  Compare,
  Science,
  Timeline,
} from '@mui/icons-material';

interface ModuleInfo {
  id: string;
  name: string;
  full_name: string;
  version: string;
  description: string;
}

interface ModuleResult {
  module_name: string;
  module_version: string;
  status: string;
  total_tests: number;
  passed_tests: number;
  failed_tests: number;
  avg_speedup: number | null;
  min_speedup: number | null;
  max_speedup: number | null;
  avg_error: number | null;
  max_error: number | null;
  numpy_time_ms: number | null;
  cupy_time_ms: number | null;
  comparisons: any[];
  error_message: string | null;
}

interface TestRun {
  run_id: string;
  status: string;
  progress: number;
  total_modules: number;
  completed_modules: number;
  created_at: string;
  completed_at: string | null;
  results: Record<string, ModuleResult>;
}

interface TestSummary {
  total_runs: number;
  last_run_id: string | null;
  last_run_time: string | null;
  modules_available: string[];
  backends_available: string[];
  overall_pass_rate: number | null;
}

const API_BASE = 'http://localhost:8000/api/tests';

const TestDashboard: React.FC = () => {
  const [summary, setSummary] = useState<TestSummary | null>(null);
  const [modules, setModules] = useState<ModuleInfo[]>([]);
  const [currentRun, setCurrentRun] = useState<TestRun | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Test configuration
  const [selectedModules, setSelectedModules] = useState<string[]>(['all']);
  const [dataSize, setDataSize] = useState('medium');

  // Fetch summary on mount
  useEffect(() => {
    fetchSummary();
    fetchModules();
  }, []);

  // Poll for current run status
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (currentRun && currentRun.status === 'running') {
      interval = setInterval(() => {
        fetchRunStatus(currentRun.run_id);
      }, 1000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [currentRun?.run_id, currentRun?.status]);

  const fetchSummary = async () => {
    try {
      const res = await fetch(`${API_BASE}/summary`);
      const data = await res.json();
      setSummary(data);
    } catch (err) {
      console.error('Failed to fetch summary:', err);
    }
  };

  const fetchModules = async () => {
    try {
      const res = await fetch(`${API_BASE}/modules`);
      const data = await res.json();
      setModules(data.modules);
    } catch (err) {
      console.error('Failed to fetch modules:', err);
    }
  };

  const fetchRunStatus = async (runId: string) => {
    try {
      const res = await fetch(`${API_BASE}/run/${runId}`);
      const data = await res.json();
      setCurrentRun(data);
      
      if (data.status === 'completed' || data.status === 'failed') {
        fetchSummary();
      }
    } catch (err) {
      console.error('Failed to fetch run status:', err);
    }
  };

  const startTestRun = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const res = await fetch(`${API_BASE}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modules: selectedModules,
          backends: ['python_numpy', 'python_cupy'],
          data_size: dataSize,
          include_performance: true,
          include_accuracy: true,
        }),
      });
      
      if (!res.ok) throw new Error('Failed to start test run');
      
      const data = await res.json();
      setCurrentRun(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (ms: number | null) => {
    if (ms === null) return '-';
    if (ms < 1) return `${(ms * 1000).toFixed(2)}μs`;
    if (ms < 1000) return `${ms.toFixed(2)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const formatSpeedup = (speedup: number | null) => {
    if (speedup === null) return '-';
    return `${speedup.toFixed(2)}x`;
  };

  const formatError = (error: number | null) => {
    if (error === null) return '-';
    return error.toExponential(2);
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Science /> Module Comparison Tests
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Side-by-side testing of C/CUDA vs Python implementations
        </Typography>
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>Total Runs</Typography>
              <Typography variant="h4">{summary?.total_runs ?? 0}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>Modules Available</Typography>
              <Typography variant="h4">{summary?.modules_available?.length ?? 0}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>Backends</Typography>
              <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                {summary?.backends_available?.map((b) => (
                  <Chip key={b} label={b} size="small" color="primary" />
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="text.secondary" gutterBottom>Pass Rate</Typography>
              <Typography variant="h4" color={summary?.overall_pass_rate && summary.overall_pass_rate > 0.9 ? 'success.main' : 'warning.main'}>
                {summary?.overall_pass_rate !== null
                  ? `${(summary.overall_pass_rate * 100).toFixed(0)}%`
                  : '-'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Test Configuration */}
      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>Run New Test</Typography>
          
          <Grid container spacing={3} alignItems="center">
            <Grid item xs={12} md={4}>
              <FormControl fullWidth size="small">
                <InputLabel>Modules</InputLabel>
                <Select
                  multiple
                  value={selectedModules}
                  onChange={(e) => setSelectedModules(e.target.value as string[])}
                  label="Modules"
                >
                  <MenuItem value="all">All Modules</MenuItem>
                  {modules.map((m) => (
                    <MenuItem key={m.id} value={m.id}>
                      {m.name} ({m.version})
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Data Size</InputLabel>
                <Select
                  value={dataSize}
                  onChange={(e) => setDataSize(e.target.value)}
                  label="Data Size"
                >
                  <MenuItem value="small">Small (Quick)</MenuItem>
                  <MenuItem value="medium">Medium</MenuItem>
                  <MenuItem value="large">Large (Thorough)</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={3}>
              <Button
                variant="contained"
                startIcon={loading ? <CircularProgress size={20} /> : <PlayArrow />}
                onClick={startTestRun}
                disabled={loading || (currentRun?.status === 'running')}
                fullWidth
              >
                {loading ? 'Starting...' : 'Run Tests'}
              </Button>
            </Grid>
            
            <Grid item xs={12} md={2}>
              <Button
                variant="outlined"
                startIcon={<Refresh />}
                onClick={fetchSummary}
                fullWidth
              >
                Refresh
              </Button>
            </Grid>
          </Grid>
          
          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>
          )}
        </CardContent>
      </Card>

      {/* Current/Latest Run Results */}
      {currentRun && (
        <Card sx={{ mb: 4 }}>
          <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">
                Test Run: {currentRun.run_id.substring(0, 8)}...
              </Typography>
              <Chip
                label={currentRun.status.toUpperCase()}
                color={
                  currentRun.status === 'completed' ? 'success' :
                  currentRun.status === 'running' ? 'primary' :
                  currentRun.status === 'failed' ? 'error' : 'default'
                }
              />
            </Box>
            
            {currentRun.status === 'running' && (
              <Box sx={{ mb: 3 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">Progress</Typography>
                  <Typography variant="body2">{currentRun.progress}%</Typography>
                </Box>
                <LinearProgress variant="determinate" value={currentRun.progress} />
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                  {currentRun.completed_modules} / {currentRun.total_modules} modules completed
                </Typography>
              </Box>
            )}
            
            {/* Results Table */}
            {Object.keys(currentRun.results).length > 0 && (
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Module</TableCell>
                      <TableCell>Status</TableCell>
                      <TableCell align="right">NumPy Time</TableCell>
                      <TableCell align="right">CuPy Time</TableCell>
                      <TableCell align="right">Speedup</TableCell>
                      <TableCell align="right">Mean Error</TableCell>
                      <TableCell align="right">Max Error</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.entries(currentRun.results).map(([name, result]) => (
                      <TableRow key={name}>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="body2" fontWeight="bold">
                              {name.toUpperCase()}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              v{result.module_version}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell>
                          {result.status === 'completed' && result.passed_tests === result.total_tests ? (
                            <Chip
                              icon={<CheckCircle />}
                              label="PASSED"
                              size="small"
                              color="success"
                            />
                          ) : result.status === 'failed' || result.failed_tests > 0 ? (
                            <Chip
                              icon={<Error />}
                              label="FAILED"
                              size="small"
                              color="error"
                            />
                          ) : (
                            <Chip label={result.status.toUpperCase()} size="small" />
                          )}
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2" fontFamily="monospace">
                            {formatTime(result.numpy_time_ms)}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2" fontFamily="monospace">
                            {formatTime(result.cupy_time_ms)}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography
                            variant="body2"
                            fontFamily="monospace"
                            color={result.avg_speedup && result.avg_speedup > 1 ? 'success.main' : 'inherit'}
                            fontWeight="bold"
                          >
                            {formatSpeedup(result.avg_speedup)}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2" fontFamily="monospace">
                            {formatError(result.avg_error)}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2" fontFamily="monospace">
                            {formatError(result.max_error)}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          </CardContent>
        </Card>
      )}

      {/* Module Information */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Compare /> Available Modules
          </Typography>
          
          {modules.map((module) => (
            <Accordion key={module.id}>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                  <Typography fontWeight="bold">{module.name}</Typography>
                  <Chip label={`v${module.version}`} size="small" variant="outlined" />
                  <Typography variant="body2" color="text.secondary" sx={{ ml: 'auto', mr: 2 }}>
                    {module.full_name}
                  </Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Typography variant="body2">{module.description}</Typography>
                <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                  <Chip icon={<Speed />} label="Performance Tested" size="small" />
                  <Chip icon={<Compare />} label="Accuracy Compared" size="small" />
                </Box>
              </AccordionDetails>
            </Accordion>
          ))}
        </CardContent>
      </Card>
    </Box>
  );
};

export default TestDashboard;
