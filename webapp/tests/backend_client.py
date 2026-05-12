"""
Thin synchronous client around the webapp backend API.
Used by the test harness to upload files, trigger jobs, and retrieve results.
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import httpx


class BackendClient:
    """Synchronous HTTP client for one backend instance."""

    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self._http    = httpx.Client(timeout=30)

    # ── health ────────────────────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        r = self._http.get(f"{self.base_url}/api/health")
        r.raise_for_status()
        return r.json()

    def is_reachable(self) -> bool:
        try:
            self.health()
            return True
        except Exception:
            return False

    # ── upload ────────────────────────────────────────────────────────────────

    def upload(self, file_path: Path) -> str:
        """Upload a rawacf file and return the file_id."""
        with open(file_path, "rb") as f:
            r = self._http.post(
                f"{self.base_url}/api/upload/",
                files={"file": (file_path.name, f, "application/octet-stream")}
            )
        r.raise_for_status()
        return r.json()["file_id"]

    # ── processing ────────────────────────────────────────────────────────────

    def start_job(
        self,
        file_id: str,
        stages: Optional[List[str]] = None,
        mode: str = "auto",
        parameters: Optional[Dict] = None,
        backend: Optional[str] = None,
    ) -> str:
        """Start a processing job and return the job_id."""
        payload = {
            "file_id":    file_id,
            "mode":       mode,
            "stages":     stages or ["acf", "fitacf", "grid"],
            "parameters": parameters or {},
        }
        if backend is not None:
            payload["backend"] = backend
        r = self._http.post(f"{self.base_url}/api/processing/start", json=payload)
        r.raise_for_status()
        return r.json()["job_id"]

    def wait_for_job(self, job_id: str) -> Dict[str, Any]:
        """Poll until the job is completed/failed; return final status dict."""
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            r = self._http.get(f"{self.base_url}/api/processing/status/{job_id}")
            r.raise_for_status()
            info = r.json()
            if info["status"] in ("completed", "failed", "cancelled"):
                return info
            time.sleep(1)
        raise TimeoutError(f"Job {job_id} did not finish within {self.timeout}s")

    def get_results(self, job_id: str) -> Dict[str, Any]:
        r = self._http.get(f"{self.base_url}/api/results/{job_id}")
        r.raise_for_status()
        return r.json()

    # ── convenience ───────────────────────────────────────────────────────────

    def run_pipeline(
        self,
        file_path: Path,
        stages: Optional[List[str]] = None,
        mode: str = "auto",
        parameters: Optional[Dict] = None,
        backend: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload → start → wait → results in one call."""
        file_id = self.upload(file_path)
        job_id  = self.start_job(file_id, stages=stages, mode=mode,
                                 parameters=parameters, backend=backend)
        status  = self.wait_for_job(job_id)
        if status["status"] != "completed":
            raise RuntimeError(f"Job failed: {status.get('error')}")
        return self.get_results(job_id)

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
