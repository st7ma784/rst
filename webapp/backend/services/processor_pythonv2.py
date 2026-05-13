"""
pythonv2 backend — wires superdarn_gpu processors to the API.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np

from models.schemas import FitACFParameters
from services.processor_base import BackendProcessor

logger = logging.getLogger(__name__)

# superdarn_gpu is either on PYTHONPATH (Docker) or found relative to repo root.
_PYTHONV2 = Path(__file__).parent.parent.parent.parent / "pythonv2"
if _PYTHONV2.exists() and str(_PYTHONV2) not in sys.path:
    sys.path.insert(0, str(_PYTHONV2))


def _to_list(arr) -> list:
    """Convert numpy/cupy array to plain Python list, handling NaN."""
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            arr = cp.asnumpy(arr)
    except ImportError:
        pass
    if hasattr(arr, "tolist"):
        return [x if x == x else None for x in arr.tolist()]  # NaN → None
    return list(arr)


class Pythonv2Processor(BackendProcessor):

    def _load_rawacf(self, file_path: Path):
        """Return first RawACF record (for single-record stages like grid/cnvmap)."""
        return self._load_all_rawacf(file_path)[0]

    def _load_all_rawacf(self, file_path: Path):
        """Return list of RawACF objects — one per record in the file."""
        from utils.rawacf_parser import parse_rawacf_all_records, detect_file_type
        ftype = detect_file_type(file_path)
        if ftype != "rawacf":
            raise ValueError(
                f"File appears to be '{ftype}', not rawacf. "
                "Upload a .rawacf file to run FITACF processing."
            )
        raw_records = parse_rawacf_all_records(file_path)
        if not raw_records:
            raise ValueError("No records found in file")
        return [self._record_to_rawacf(r) for r in raw_records]

    def _load_fitacf_objects(self, file_path: Path):
        """
        Load pre-processed FitACF data from our binary rawacf format.
        Used when user uploads a .fitacf that was previously processed by this system
        and stored as a rawacf-format file (for grid/cnvmap starting point).
        """
        from utils.rawacf_parser import parse_rawacf_all_records
        from superdarn_gpu.core.datatypes import FitACF
        import numpy as np
        raw_records = parse_rawacf_all_records(file_path)
        fitacf_objs = []
        for rec in raw_records:
            hdr    = rec["header"]
            nrang  = int(hdr["nrang"])
            acf_r  = rec["acf_real"]
            fitacf = FitACF(nrang=nrang)
            fitacf.velocity       = acf_r[:, 0].astype(np.float32)   # col 0 = vel
            fitacf.power          = acf_r[:, 1].astype(np.float32) if acf_r.shape[1] > 1 else np.zeros(nrang, dtype=np.float32)
            fitacf.spectral_width = acf_r[:, 2].astype(np.float32) if acf_r.shape[1] > 2 else np.zeros(nrang, dtype=np.float32)
            fitacf.qflg           = (fitacf.power > 0).astype(np.int8)
            fitacf_objs.append(fitacf)
        return fitacf_objs

    def _record_to_rawacf(self, rec: dict):
        """Convert a parsed record dict to a RawACF object."""
        return self._synthetic_rawacf_from_record(
            rec["acf_real"], rec["acf_imag"],
            rec["xcf_real"], rec["xcf_imag"],
            rec["header"]
        )

    def _synthetic_rawacf(self, file_path: Path):
        from utils.rawacf_parser import parse_rawacf_file
        acf_r, acf_i, xcf_r, xcf_i, hdr = parse_rawacf_file(file_path)
        return self._synthetic_rawacf_from_record(acf_r, acf_i, xcf_r, xcf_i, hdr)

    def _synthetic_rawacf_from_record(self, acf_r, acf_i, xcf_r, xcf_i, hdr):
        from superdarn_gpu.core.datatypes import RawACF, RadarParameters
        from superdarn_gpu.core.backends import get_backend, Backend
        from datetime import datetime
        nrang = int(hdr["nrang"])
        mplgs = int(hdr["mplgs"])
        tfreq = int(hdr["tfreq"])   # kHz
        mpinc = int(hdr["mpinc"])   # µs
        nave  = int(hdr["nave"])

        acf = (acf_r + 1j * acf_i).astype(np.complex64)
        xcf = (xcf_r + 1j * xcf_i).astype(np.complex64)

        # Move to GPU when the CUPY backend is active so CUDA kernels receive cupy arrays
        if get_backend() == Backend.CUPY:
            try:
                import cupy as cp
                acf = cp.asarray(acf)
                xcf = cp.asarray(xcf)
            except Exception:
                pass

        prm = RadarParameters(
            station_id=int(hdr["radar_id"]),
            beam_number=int(hdr["bmnum"]),
            scan_flag=int(hdr["scan"]),
            channel=int(hdr["channel"]),
            cp_id=int(hdr["cp"]),
            nave=nave,
            lagfr=int(hdr["frang"]) * 1000 // 300,  # approx
            smsep=300,
            txpow=0,
            atten=0,
            noise_search=float(hdr["noise_lev"]),
            noise_mean=float(hdr["noise_lev"]),
            tfreq=tfreq,
            nrang=nrang,
            frang=int(hdr["frang"]),
            rsep=int(hdr["rsep"]),
            xcf=1,
            mppul=8,
            mpinc=mpinc,
            mplgs=mplgs,
            txpl=300,
            intt_sc=3,
            intt_us=0,
            timestamp=datetime.utcnow()
        )

        # Use the same array module as acf (cupy or numpy)
        try:
            import cupy as cp
            xp = cp if isinstance(acf, cp.ndarray) else np
        except ImportError:
            xp = np

        rawacf = RawACF(nrang=nrang, mplgs=mplgs, nave=nave)
        rawacf.prm   = prm
        rawacf.acf   = acf
        rawacf.xcf   = xcf
        rawacf.power = xp.abs(acf[:, 0]).astype(np.float32 if xp is np else xp.float32)
        rawacf.noise = xp.full(nrang, float(hdr["noise_lev"]), dtype=xp.float32)
        rawacf.slist = xp.arange(nrang, dtype=xp.int16)
        rawacf.qflg  = xp.ones(nrang,  dtype=xp.int8)
        rawacf.gflg  = xp.zeros(nrang, dtype=xp.int8)
        return rawacf

    async def process_acf(self, file_path: Path, params: FitACFParameters, use_gpu: bool) -> Dict[str, Any]:
        from superdarn_gpu.processing.acf import ACFProcessor, ACFConfig
        rawacf = self._load_rawacf(file_path)
        config = ACFConfig()
        proc   = ACFProcessor(config=config)
        # ACFProcessor.process expects a dict of IQ samples; skip re-processing
        # and return the lag-0 power directly from the loaded record.
        pwr = _to_list(rawacf.power)
        return {
            "nranges":  rawacf.nrang,
            "nlags":    rawacf.mplgs,
            "acf_power": pwr,
            "backend": "pythonv2-gpu" if use_gpu else "pythonv2-cpu",
        }

    async def process_fitacf(self, file_path: Path, params: FitACFParameters, use_gpu: bool) -> Dict[str, Any]:
        from superdarn_gpu.processing.fitacf import FitACFProcessor, FitACFConfig, FitACFAlgorithm
        config = FitACFConfig(
            algorithm=FitACFAlgorithm.V3_0,
            min_power_threshold=params.min_power,
            enable_xcf=params.xcf_enabled,
            elevation_correction=params.elevation_enabled,
        )
        proc = FitACFProcessor(config=config)

        all_rawacf = self._load_all_rawacf(file_path)

        def _opt(fitacf, attr):
            return _to_list(getattr(fitacf, attr)) if hasattr(fitacf, attr) else []

        records = []
        all_fitacf_objs = []
        total_good = 0

        for rawacf in all_rawacf:
            fitacf = proc.process(rawacf)
            qmask  = fitacf.qflg > 0
            good   = int(np.sum(qmask))
            total_good += good
            all_fitacf_objs.append(fitacf)
            records.append({
                "beam":                       int(rawacf.prm.beam_number) if rawacf.prm else 0,
                "good_ranges":                good,
                "velocity":                   _opt(fitacf, "velocity"),
                "velocity_error":             _opt(fitacf, "velocity_error"),
                "power":                      _opt(fitacf, "power"),
                "power_error":                _opt(fitacf, "power_error"),
                "spectral_width":             _opt(fitacf, "spectral_width"),
                "spectral_width_error":       _opt(fitacf, "spectral_width_error"),
                "spectral_width_sigma":       _opt(fitacf, "spectral_width_sigma"),
                "spectral_width_sigma_error": _opt(fitacf, "spectral_width_sigma_error"),
                "elevation":                  _opt(fitacf, "elevation"),
                "quality_flag":               _opt(fitacf, "qflg"),
                "ground_scatter_flag":        _opt(fitacf, "gflg"),
                "nlag_fit":                   _opt(fitacf, "nlag_fit"),
            })

        backend_str = "pythonv2-gpu" if use_gpu else "pythonv2-cpu"
        # Store top-level sigma/elevation (not per-beam) + records for per-beam data.
        # Viz endpoint reconstructs range-profile scalars from records[0] at read time.
        first = records[0] if records else {}
        return {
            "nranges":                    all_rawacf[0].nrang,
            "nrecords":                   len(records),
            "good_ranges":                total_good // max(len(records), 1),
            # Fields only available at scan level (not per-beam in records)
            "spectral_width_sigma":       first.get("spectral_width_sigma", []),
            "spectral_width_sigma_error": first.get("spectral_width_sigma_error", []),
            "elevation":                  first.get("elevation", []),
            "elevation_error":            first.get("elevation_error", []),
            # Per-beam arrays (RTI + beam selector)
            "records":                    records,
            "backend":                    backend_str,
            "_fitacf_objs": all_fitacf_objs,
            "_fitacf_obj":  all_fitacf_objs[0] if all_fitacf_objs else None,
        }

    async def process_lmfit(self, previous: Dict, params: FitACFParameters, use_gpu: bool) -> Dict[str, Any]:
        """
        Extract real fit quality metrics from the FitACF objects.
        RST LMFIT is a separate module for non-linear fitting; here we expose
        the chi² and convergence diagnostics from the linear LS fits already done.
        """
        import numpy as np
        fa = previous.get("fitacf", {})
        fitacf_objs = fa.get("_fitacf_objs") or (
            [fa["_fitacf_obj"]] if fa.get("_fitacf_obj") else []
        )

        def _to_np(arr):
            if arr is None:
                return None
            if hasattr(arr, "get"):          # cupy array
                return arr.get()
            return np.asarray(arr)

        all_chi2  = []
        all_nlags = []
        total_fit = 0

        for fitacf_obj in fitacf_objs:
            if not hasattr(fitacf_obj, "qflg"):
                continue
            qmask = _to_np(fitacf_obj.qflg) > 0
            total_fit += int(np.sum(qmask))

            if hasattr(fitacf_obj, "chi2"):
                chi2_vals = _to_np(fitacf_obj.chi2)[qmask]
                valid_chi2 = chi2_vals[np.isfinite(chi2_vals) & (chi2_vals > 0)]
                if len(valid_chi2):
                    all_chi2.extend(valid_chi2.tolist())

            if hasattr(fitacf_obj, "nlag_fit"):
                nlags = _to_np(fitacf_obj.nlag_fit)[qmask]
                all_nlags.extend(nlags[nlags > 0].tolist())

        mean_chi2  = float(np.mean(all_chi2))  if all_chi2  else None
        mean_nlags = float(np.mean(all_nlags)) if all_nlags else None
        converged  = total_fit > 0 and (mean_chi2 is None or mean_chi2 < 1000)

        return {
            "converged":    converged,
            "fitted_ranges": total_fit,
            "mean_chi_squared": mean_chi2,
            "mean_nlag_fit":    mean_nlags,
            "nrecords":    len(fitacf_objs),
            "backend":     "pythonv2-cpu",
        }

    async def process_grid(self, previous: Dict, params: FitACFParameters, use_gpu: bool) -> Dict[str, Any]:
        from superdarn_gpu.processing.grid import GridProcessor, GridConfig
        fa = previous.get("fitacf", {})
        # Use all beam records when available; fall back to single object
        fitacf_objs = fa.get("_fitacf_objs") or (
            [fa["_fitacf_obj"]] if fa.get("_fitacf_obj") else None
        )
        # If fitacf stage was skipped (e.g. user uploaded pre-processed data)
        # try to load from the uploaded file path stored in context
        if not fitacf_objs:
            file_path = previous.get("_file_path")
            if file_path:
                try:
                    fitacf_objs = self._load_fitacf_objects(Path(file_path))
                except Exception:
                    pass
        if not fitacf_objs:
            return {"nlat": 0, "nlon": 0, "velocity": [], "backend": "pythonv2-cpu"}

        config  = GridConfig()
        proc    = GridProcessor(config=config)
        grid    = proc.process(fitacf_objs)
        vel_arr = grid.velocity
        nlat    = len(grid.lat) if hasattr(grid, "lat") else 0
        nlon    = len(grid.lon) if hasattr(grid, "lon") else 0
        return {
            "nlat":      nlat,
            "nlon":      nlon,
            "velocity":  _to_list(vel_arr.ravel()),
            "backend":   "pythonv2-gpu" if use_gpu else "pythonv2-cpu",
            "_grid_obj": grid,   # passed to cnvmap stage
        }

    async def process_cnvmap(self, previous: Dict, params: FitACFParameters, use_gpu: bool) -> Dict[str, Any]:
        from superdarn_gpu.processing.cnvmap import CNVMAPProcessor

        # Prefer GridData objects; fall back to building from fitacf if grid wasn't run
        grid_obj = previous.get("grid", {}).get("_grid_obj")
        if grid_obj is None:
            # Build a minimal GridData from the fitacf objects if available
            fitacf_objs = previous.get("fitacf", {}).get("_fitacf_objs") or []
            if not fitacf_objs:
                return {"order": 0, "chi_squared": None, "potential_max": None,
                        "backend": "pythonv2-cpu", "note": "no input data for cnvmap"}
            from superdarn_gpu.processing.grid import GridProcessor, GridConfig
            grid_obj = GridProcessor(GridConfig()).process(fitacf_objs)

        # Ensure grid_obj arrays are numpy (cnvmap uses pure numpy; grid may hold cupy arrays)
        try:
            import cupy as cp
            def _cpu(arr):
                return arr.get() if isinstance(arr, cp.ndarray) else arr
            if hasattr(grid_obj, "velocity"):
                grid_obj.velocity       = _cpu(grid_obj.velocity)
            if hasattr(grid_obj, "velocity_error"):
                grid_obj.velocity_error = _cpu(grid_obj.velocity_error)
            if hasattr(grid_obj, "lat"):
                grid_obj.lat = _cpu(grid_obj.lat)
            if hasattr(grid_obj, "lon"):
                grid_obj.lon = _cpu(grid_obj.lon)
        except ImportError:
            pass

        try:
            proc = CNVMAPProcessor(lmax=8)
            cmap = proc.process([grid_obj])
            import numpy as np

            def _to_np(arr):
                if hasattr(arr, "get"):   # cupy
                    return arr.get().astype(np.float32)
                return np.asarray(arr, dtype=np.float32)
            pot  = _to_np(cmap.potential)
            vmag = _to_np(cmap.velocity_magnitude)

            # Downsample to ~46×91 (step=4) for storage/API — ~4 KB vs 255 KB full
            s = 4
            def _compact(arr):
                return [[None if np.isnan(v) else round(float(v), 2)
                         for v in row] for row in arr[::s, ::s].tolist()]

            return {
                "order":         proc.lmax,
                "chi_squared":   float(cmap.chi2) if cmap.chi2 else None,
                "potential_max": float(np.nanmax(np.abs(pot))) if pot.size else None,
                "num_vectors":   cmap.num_vectors,
                "has_map":       True,
                "downsample":    s,
                "nlat":          pot.shape[0] // s,
                "nlon":          pot.shape[1] // s,
                "potential":     _compact(pot),
                "velocity_mag":  _compact(vmag),
                "backend":       "pythonv2-cpu",
            }
        except Exception as e:
            logger.warning(f"CNVMAPProcessor failed: {e}")
            return {"order": 8, "chi_squared": None, "potential_max": None,
                    "has_map": False, "backend": "pythonv2-cpu", "note": str(e)}
