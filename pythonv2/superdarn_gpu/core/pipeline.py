"""
Processing pipeline framework for SuperDARN GPU processing
"""

from typing import List, Dict, Any, Optional, Callable, Union
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass
from enum import Enum

from .backends import get_backend, Backend, get_array_module, synchronize
from .memory import MemoryMonitor, memory_pool
from .datatypes import RadarData

class StageStatus(Enum):
    """Processing stage status"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class StageResult:
    """Result from a processing stage"""
    status: StageStatus
    output_data: Optional[Any] = None
    processing_time: float = 0.0
    memory_used: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

class Stage(ABC):
    """
    Abstract base class for processing pipeline stages
    """
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs
        self.status = StageStatus.PENDING
        self.result = None
        
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Process input data and return output
        
        Parameters
        ----------
        input_data : Any
            Input data for processing
            
        Returns
        -------
        Any
            Processed output data
        """
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data before processing
        
        Parameters
        ----------
        input_data : Any
            Input data to validate
            
        Returns
        -------
        bool
            True if input is valid
        """
        return True
    
    def get_memory_estimate(self, input_data: Any) -> int:
        """
        Estimate memory requirements for this stage
        
        Parameters
        ----------
        input_data : Any
            Input data
            
        Returns
        -------
        int
            Estimated memory usage in bytes
        """
        return 0
    
    def run(self, input_data: Any) -> StageResult:
        """
        Execute the processing stage with timing and error handling
        
        Parameters
        ----------
        input_data : Any
            Input data for processing
            
        Returns
        -------
        StageResult
            Processing result with metadata
        """
        self.status = StageStatus.RUNNING
        start_time = time.time()
        
        try:
            # Validate input
            if not self.validate_input(input_data):
                raise ValueError(f"Invalid input for stage {self.name}")
            
            # Memory monitoring
            with MemoryMonitor(f"Stage {self.name}") as monitor:
                # Process data
                output_data = self.process(input_data)
                
                # Synchronize GPU operations
                synchronize()
                
                # Create successful result
                processing_time = time.time() - start_time
                self.status = StageStatus.COMPLETED
                
                self.result = StageResult(
                    status=StageStatus.COMPLETED,
                    output_data=output_data,
                    processing_time=processing_time,
                    memory_used=getattr(monitor, 'memory_used', 0),
                    metadata={'stage_name': self.name, 'config': self.config}
                )
                
        except Exception as e:
            # Handle errors
            processing_time = time.time() - start_time
            self.status = StageStatus.FAILED
            
            self.result = StageResult(
                status=StageStatus.FAILED,
                processing_time=processing_time,
                error_message=str(e),
                metadata={'stage_name': self.name, 'config': self.config}
            )
            
        return self.result

class ProcessingPipeline:
    """
    GPU-optimized processing pipeline for SuperDARN data
    """
    
    def __init__(self, 
                 name: str = "SuperDARN Processing Pipeline",
                 memory_limit_fraction: float = 0.8):
        self.name = name
        self.stages: List[Stage] = []
        self.results: List[StageResult] = []
        self.memory_limit_fraction = memory_limit_fraction
        self.status = StageStatus.PENDING
        
    def add_stage(self, stage: Stage) -> 'ProcessingPipeline':
        """
        Add a processing stage to the pipeline
        
        Parameters
        ----------
        stage : Stage
            Processing stage to add
            
        Returns
        -------
        ProcessingPipeline
            Self for method chaining
        """
        self.stages.append(stage)
        return self
    
    def remove_stage(self, stage_name: str) -> bool:
        """
        Remove a stage by name
        
        Parameters
        ----------
        stage_name : str
            Name of stage to remove
            
        Returns
        -------
        bool
            True if stage was found and removed
        """
        for i, stage in enumerate(self.stages):
            if stage.name == stage_name:
                del self.stages[i]
                return True
        return False
    
    def get_stage(self, stage_name: str) -> Optional[Stage]:
        """Get stage by name"""
        for stage in self.stages:
            if stage.name == stage_name:
                return stage
        return None
    
    def estimate_memory_requirements(self, input_data: Any) -> int:
        """
        Estimate total memory requirements for pipeline
        
        Parameters
        ----------
        input_data : Any
            Initial input data
            
        Returns
        -------
        int
            Estimated memory requirement in bytes
        """
        total_memory = 0
        current_data = input_data
        
        for stage in self.stages:
            stage_memory = stage.get_memory_estimate(current_data)
            total_memory = max(total_memory, stage_memory)  # Peak memory usage
            
        return total_memory
    
    def run(self, input_data: Any, 
            stop_on_error: bool = True,
            verbose: bool = True) -> List[StageResult]:
        """
        Execute the complete processing pipeline
        
        Parameters
        ----------
        input_data : Any
            Initial input data
        stop_on_error : bool
            Whether to stop pipeline on first error
        verbose : bool
            Whether to print progress information
            
        Returns
        -------
        List[StageResult]
            Results from all executed stages
        """
        if verbose:
            print(f"Starting pipeline: {self.name}")
            print(f"Number of stages: {len(self.stages)}")
            
            if get_backend() == Backend.CUPY:
                memory_info = memory_pool.get_memory_info()
                print(f"GPU memory: {memory_info['used_pool'] / (1024**3):.1f}GB used, "
                      f"{memory_info['available'] / (1024**3):.1f}GB available")
        
        self.status = StageStatus.RUNNING
        self.results = []
        current_data = input_data
        
        start_time = time.time()
        
        try:
            for i, stage in enumerate(self.stages):
                if verbose:
                    print(f"Running stage {i+1}/{len(self.stages)}: {stage.name}")
                
                # Execute stage
                result = stage.run(current_data)
                self.results.append(result)
                
                if verbose:
                    print(f"  Status: {result.status.value}")
                    print(f"  Time: {result.processing_time:.2f}s")
                    if result.memory_used > 0:
                        print(f"  Memory: {result.memory_used / (1024**2):.1f}MB")
                
                # Check for errors
                if result.status == StageStatus.FAILED:
                    if stop_on_error:
                        if verbose:
                            print(f"Pipeline stopped due to error: {result.error_message}")
                        self.status = StageStatus.FAILED
                        return self.results
                    else:
                        if verbose:
                            print(f"Stage failed but continuing: {result.error_message}")
                        continue
                
                # Update current data for next stage
                current_data = result.output_data
            
            # Pipeline completed successfully
            total_time = time.time() - start_time
            self.status = StageStatus.COMPLETED
            
            if verbose:
                print(f"Pipeline completed in {total_time:.2f}s")
                successful_stages = sum(1 for r in self.results if r.status == StageStatus.COMPLETED)
                print(f"Successful stages: {successful_stages}/{len(self.results)}")
        
        except Exception as e:
            self.status = StageStatus.FAILED
            if verbose:
                print(f"Pipeline failed with exception: {e}")
            raise
        
        return self.results
    
    def run_batch(self, 
                  input_batch: List[Any],
                  batch_size: Optional[int] = None,
                  **kwargs) -> List[List[StageResult]]:
        """
        Run pipeline on batch of inputs
        
        Parameters
        ----------
        input_batch : List[Any]
            List of input data items
        batch_size : int, optional
            Process in sub-batches of this size
        **kwargs
            Arguments passed to run()
            
        Returns
        -------
        List[List[StageResult]]
            Results for each input item
        """
        if batch_size is None:
            # Process all at once
            return [self.run(input_data, **kwargs) for input_data in input_batch]
        
        # Process in sub-batches
        all_results = []
        for i in range(0, len(input_batch), batch_size):
            batch = input_batch[i:i+batch_size]
            batch_results = [self.run(input_data, **kwargs) for input_data in batch]
            all_results.extend(batch_results)
            
            # Memory cleanup between batches
            memory_pool.optimize_memory()
        
        return all_results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for completed pipeline
        
        Returns
        -------
        Dict[str, Any]
            Performance statistics
        """
        if not self.results:
            return {}
        
        total_time = sum(r.processing_time for r in self.results)
        total_memory = sum(r.memory_used for r in self.results if r.memory_used)
        successful_stages = sum(1 for r in self.results if r.status == StageStatus.COMPLETED)
        
        stage_times = {r.metadata['stage_name']: r.processing_time 
                      for r in self.results if r.metadata}
        
        return {
            'total_time': total_time,
            'total_memory_used': total_memory,
            'successful_stages': successful_stages,
            'total_stages': len(self.results),
            'success_rate': successful_stages / len(self.results),
            'stage_times': stage_times,
            'pipeline_status': self.status.value
        }