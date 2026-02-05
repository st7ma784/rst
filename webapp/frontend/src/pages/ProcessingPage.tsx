import { useState, useCallback } from 'react';
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
  Switch,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  LinearProgress,
  Alert,
  Chip,
} from '@mui/material';
import { useDropzone } from 'react-dropzone';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

interface ProcessingParameters {
  minPower: number;
  phaseTolerance: number;
  elevationEnabled: boolean;
  elevationModel: string;
  batchSize: number;
  xcfEnabled: boolean;
}

export default function ProcessingPage() {
  const navigate = useNavigate();
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [fileId, setFileId] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [jobId, setJobId] = useState<string | null>(null);
  const [parameters, setParameters] = useState<ProcessingParameters>({
    minPower: 3.0,
    phaseTolerance: 25.0,
    elevationEnabled: true,
    elevationModel: 'GSM',
    batchSize: 64,
    xcfEnabled: true,
  });

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
          parameters: {
            min_power: parameters.minPower,
            phase_tolerance: parameters.phaseTolerance,
            elevation_enabled: parameters.elevationEnabled,
            elevation_model: parameters.elevationModel,
            batch_size: parameters.batchSize,
            xcf_enabled: parameters.xcfEnabled,
          },
          stages: ['acf', 'fitacf', 'grid'],
        }),
      });
      
      if (response.ok) {
        const data = await response.json();
        setJobId(data.job_id);
        
        // Poll for progress
        pollJobStatus(data.job_id);
      }
    } catch (error) {
      console.error('Processing failed:', error);
      setIsProcessing(false);
    }
  };

  const pollJobStatus = async (jobId: string) => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`/api/processing/status/${jobId}`);
        if (response.ok) {
          const data = await response.json();
          setProgress(data.progress);
          
          if (data.status === 'completed' || data.status === 'failed') {
            clearInterval(interval);
            setIsProcessing(false);
            
            if (data.status === 'completed') {
              // Navigate to visualization using React Router
              navigate(`/visualization/${jobId}`);
            }
          }
        }
      } catch (error) {
        console.error('Status check failed:', error);
      }
    }, 1000);
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
            <Typography variant="h5" gutterBottom>
              2. Configure Parameters
            </Typography>
            
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
        
        {/* Processing Control Section */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              3. Start Processing
            </Typography>
            
            <Box sx={{ mt: 2, display: 'flex', alignItems: 'center', gap: 2 }}>
              <Button
                variant="contained"
                size="large"
                onClick={handleStartProcessing}
                disabled={!fileId || isProcessing}
              >
                {isProcessing ? 'Processing...' : 'Start Processing'}
              </Button>
              
              <Chip label="GPU Mode" color="primary" />
              <Chip label="CUDArst v2.0.0" variant="outlined" />
            </Box>
            
            {isProcessing && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="body2" gutterBottom>
                  Processing: {progress}%
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
