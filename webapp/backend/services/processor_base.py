"""
Abstract base for all processing backends.

Every backend (pythonv2, cuda, rst) must implement this interface so the
dispatcher and test harness can treat them identically.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

from models.schemas import FitACFParameters


class BackendProcessor(ABC):

    @abstractmethod
    async def process_acf(self, file_path: Path, params: FitACFParameters, use_gpu: bool) -> Dict[str, Any]:
        """Return dict with keys: nranges, nlags, acf_power (list), backend"""

    @abstractmethod
    async def process_fitacf(self, file_path: Path, params: FitACFParameters, use_gpu: bool) -> Dict[str, Any]:
        """Return dict with keys: nranges, good_ranges, velocity (list),
           power (list), spectral_width (list), backend"""

    @abstractmethod
    async def process_lmfit(self, previous: Dict, params: FitACFParameters, use_gpu: bool) -> Dict[str, Any]:
        """Return dict with keys: iterations, converged, chi_squared, backend"""

    @abstractmethod
    async def process_grid(self, previous: Dict, params: FitACFParameters, use_gpu: bool) -> Dict[str, Any]:
        """Return dict with keys: nlat, nlon, velocity (list), backend"""

    @abstractmethod
    async def process_cnvmap(self, previous: Dict, params: FitACFParameters, use_gpu: bool) -> Dict[str, Any]:
        """Return dict with keys: order, chi_squared, potential_max, backend"""
