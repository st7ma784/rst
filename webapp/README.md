# SuperDARN Interactive Workbench (SIW)

A modern web application for interactive SuperDARN data processing with CUDA acceleration.

## Features

- 🚀 **Real-time Processing**: Live data processing with GPU acceleration
- 📊 **Interactive Visualizations**: 3D convection maps, range-time plots, and more
- 🎛️ **Parameter Tuning**: Adjust processing parameters and see effects immediately
- 💻 **Remote Compute**: Submit jobs to Slurm clusters or remote servers via SSH
- 📈 **Performance Monitoring**: Compare CPU vs GPU processing times
- 🔄 **Pipeline Visualization**: See data flow through processing stages

## Architecture

```
webapp/
├── backend/           # FastAPI Python backend
│   ├── api/          # REST API endpoints
│   ├── core/         # Core business logic
│   ├── models/       # Data models
│   ├── services/     # Processing services
│   └── main.py       # Application entry point
└── frontend/         # React TypeScript frontend
    ├── src/
    │   ├── components/  # React components
    │   ├── pages/       # Page components
    │   ├── services/    # API client services
    │   └── App.tsx      # Main application
    └── package.json
```

## Quick Start

### Backend

```bash
cd webapp/backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd webapp/frontend
npm install
npm start
```

Then open http://localhost:3000 in your browser.

## Development

### Backend API

The backend provides:
- `/api/upload` - Upload FITACF/RAWACF files
- `/api/process` - Start data processing
- `/api/status/{job_id}` - Check processing status
- `/api/results/{job_id}` - Get processing results
- `/ws/progress` - WebSocket for real-time updates

### Frontend

Built with:
- React 18 with TypeScript
- Material-UI for components
- Three.js for 3D visualizations
- Recharts for 2D plots
- WebSocket for real-time updates

## Remote Compute

### Slurm Integration

Configure Slurm connection in `backend/config.yaml`:

```yaml
slurm:
  host: cluster.university.edu
  username: your_username
  partition: gpu
  account: your_account
```

### SSH Direct Connection

For direct SSH execution without Slurm:

```yaml
ssh:
  host: gpu-server.university.edu
  username: your_username
  cuda_path: /usr/local/cuda
  cudarst_path: /opt/cudarst
```

## Docker Deployment

```bash
docker-compose up -d
```

The application will be available at http://localhost:3000

## License

GPL v3.0 - Compatible with RST SuperDARN toolkit
