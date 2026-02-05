"""
Data models for API requests and responses
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class ProcessingMode(str, Enum):
    """Processing mode selection"""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"

class ProcessingStage(str, Enum):
    """Processing pipeline stages"""
    UPLOAD = "upload"
    ACF = "acf"
    FITACF = "fitacf"
    LMFIT = "lmfit"
    GRID = "grid"
    CNVMAP = "cnvmap"
    COMPLETE = "complete"

class JobStatus(str, Enum):
    """Job processing status"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class FitACFParameters(BaseModel):
    """FITACF processing parameters"""
    min_power: float = Field(default=3.0, description="Minimum power threshold (dB)")
    phase_tolerance: float = Field(default=25.0, description="Phase tolerance (degrees)")
    elevation_enabled: bool = Field(default=True, description="Enable elevation correction")
    elevation_model: str = Field(default="GSM", description="Elevation model")
    batch_size: int = Field(default=64, description="CUDA batch size (range gates)")
    xcf_enabled: bool = Field(default=True, description="Enable XCF processing")

class ProcessingRequest(BaseModel):
    """Request to start data processing"""
    file_id: str = Field(..., description="Uploaded file identifier")
    mode: ProcessingMode = Field(default=ProcessingMode.AUTO, description="Processing mode")
    parameters: FitACFParameters = Field(default_factory=FitACFParameters)
    stages: List[ProcessingStage] = Field(
        default=[ProcessingStage.ACF, ProcessingStage.FITACF, ProcessingStage.GRID],
        description="Processing stages to execute"
    )

class JobInfo(BaseModel):
    """Job information and status"""
    job_id: str
    status: JobStatus
    progress: int = Field(default=0, ge=0, le=100)
    current_stage: Optional[ProcessingStage] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    mode: ProcessingMode
    parameters: FitACFParameters

class ProcessingResult(BaseModel):
    """Processing results"""
    job_id: str
    status: JobStatus
    processing_time: float = Field(..., description="Total processing time (seconds)")
    stages: Dict[str, Dict[str, Any]] = Field(..., description="Results from each stage")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance statistics")
    output_files: List[str] = Field(default_factory=list, description="Generated output files")

class RemoteComputeConfig(BaseModel):
    """Remote compute configuration"""
    compute_type: str = Field(..., description="Type: 'slurm' or 'ssh'")
    host: str = Field(..., description="Remote host address")
    username: str = Field(..., description="SSH username")
    port: int = Field(default=22, description="SSH port")
    # Slurm-specific
    partition: Optional[str] = Field(None, description="Slurm partition")
    account: Optional[str] = Field(None, description="Slurm account")
    nodes: int = Field(default=1, description="Number of nodes")
    gpus: int = Field(default=1, description="GPUs per node")
    time_limit: str = Field(default="01:00:00", description="Time limit (HH:MM:SS)")
    # SSH-specific
    cuda_path: Optional[str] = Field(None, description="CUDA installation path")
    cudarst_path: Optional[str] = Field(None, description="CUDArst library path")

class RemoteJobSubmission(BaseModel):
    """Remote job submission request"""
    config: RemoteComputeConfig
    file_id: str
    parameters: FitACFParameters
    stages: List[ProcessingStage]

class UploadResponse(BaseModel):
    """File upload response"""
    file_id: str
    filename: str
    size: int
    upload_time: datetime
    format: str = Field(..., description="Detected file format (rawacf/fitacf/etc)")
    metadata: Dict[str, Any] = Field(default_factory=dict)
