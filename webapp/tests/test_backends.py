"""
Backend comparison test suite.

Runs the same pipeline against every reachable backend and:
  1. Validates that each backend returns the correct schema.
  2. Compares numerical outputs between all pairs of backends.

Usage (all backends running locally):
  cd webapp/tests
  pytest test_backends.py -v

Against specific URLs:
  pytest test_backends.py --backend-urls http://host1:8001,http://host2:8002 -v

Single backend validation (no comparison):
  pytest test_backends.py --backend-urls http://localhost:8001 -v

Mark slow tests to skip with: pytest -m "not slow"
"""

import itertools
import sys
from pathlib import Path
from typing import Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from backend_client import BackendClient
from compare import compare_pipelines, summarise

# ── shared fixtures / helpers ─────────────────────────────────────────────────

@pytest.fixture(scope="session")
def reachable_backends(backend_urls, job_timeout) -> List[Dict]:
    """Return dicts with url + label for every backend that responds to /health."""
    alive = []
    for url in backend_urls:
        client = BackendClient(url, timeout=job_timeout)
        try:
            info  = client.health()
            label = f"{info.get('version','?')} @ {url}"
            alive.append({"url": url, "label": label, "client": client})
        except Exception as e:
            pytest.skip(f"Backend {url} not reachable: {e}")
    if not alive:
        pytest.skip("No backends reachable")
    return alive


# ── individual backend validation ─────────────────────────────────────────────

class TestBackendContract:
    """
    Every reachable backend must implement the full API contract.
    Parametrised over URLs; run against each backend independently.
    """

    @pytest.mark.parametrize("stage_list", [
        ["acf"],
        ["fitacf"],
        ["acf", "fitacf"],
        ["acf", "fitacf", "grid"],
    ])
    def test_pipeline_returns_valid_schema(
        self, reachable_backends, small_rawacf, job_timeout, stage_list
    ):
        for backend in reachable_backends:
            client = backend["client"]
            result = client.run_pipeline(small_rawacf, stages=stage_list)

            assert "stages" in result, f"{backend['url']}: missing 'stages' key"
            assert "performance_metrics" in result

            for stage in stage_list:
                assert stage in result["stages"], \
                    f"{backend['url']}: stage '{stage}' missing from results"

    def test_health_endpoint(self, reachable_backends):
        for backend in reachable_backends:
            info = backend["client"].health()
            assert info.get("status") == "healthy", \
                f"{backend['url']}: unhealthy response {info}"

    def test_upload_and_delete(self, reachable_backends, small_rawacf, http_client):
        for backend in reachable_backends:
            fid = backend["client"].upload(small_rawacf)
            assert fid, f"{backend['url']}: upload returned empty file_id"

            r = http_client.delete(f"{backend['url']}/api/upload/{fid}")
            assert r.status_code == 200

    def test_acf_fields(self, reachable_backends, small_rawacf, job_timeout):
        for backend in reachable_backends:
            result = backend["client"].run_pipeline(small_rawacf, stages=["acf"])
            acf = result["stages"]["acf"]
            assert "nranges"   in acf, f"{backend['url']}: acf missing nranges"
            assert "nlags"     in acf, f"{backend['url']}: acf missing nlags"
            assert "acf_power" in acf, f"{backend['url']}: acf missing acf_power"
            assert acf["nranges"] > 0

    def test_fitacf_fields(self, reachable_backends, small_rawacf, job_timeout):
        for backend in reachable_backends:
            result = backend["client"].run_pipeline(small_rawacf, stages=["fitacf"])
            fa = result["stages"]["fitacf"]
            for key in ("nranges", "good_ranges", "velocity", "power", "spectral_width"):
                assert key in fa, f"{backend['url']}: fitacf missing '{key}'"
            assert fa["nranges"] > 0
            assert len(fa["velocity"]) == fa["nranges"], \
                f"{backend['url']}: velocity length != nranges"

    def test_velocity_range(self, reachable_backends, small_rawacf):
        """Velocities should be within ±Nyquist (~4165 m/s for 12 MHz / 1500 µs)."""
        MAX_VEL = 5000.0   # conservative upper bound; Nyquist ≈ vel_factor*π ≈ 4165 m/s
        for backend in reachable_backends:
            result = backend["client"].run_pipeline(small_rawacf, stages=["fitacf"])
            vel = [v for v in result["stages"]["fitacf"]["velocity"] if v is not None]
            for v in vel:
                assert abs(v) <= MAX_VEL, \
                    f"{backend['url']}: velocity {v:.1f} m/s out of range"

    def test_spectral_width_non_negative(self, reachable_backends, small_rawacf):
        for backend in reachable_backends:
            result = backend["client"].run_pipeline(small_rawacf, stages=["fitacf"])
            widths = [w for w in result["stages"]["fitacf"]["spectral_width"] if w is not None]
            bad = [w for w in widths if w < 0]
            assert not bad, f"{backend['url']}: negative spectral widths: {bad[:5]}"

    def test_power_non_negative(self, reachable_backends, small_rawacf):
        for backend in reachable_backends:
            result = backend["client"].run_pipeline(small_rawacf, stages=["fitacf"])
            powers = [p for p in result["stages"]["fitacf"]["power"] if p is not None]
            bad = [p for p in powers if p < 0]
            assert not bad, f"{backend['url']}: negative power values: {bad[:5]}"

    @pytest.mark.slow
    def test_medium_file_pipeline(self, reachable_backends, medium_rawacf, job_timeout):
        """Full pipeline with the medium test file."""
        for backend in reachable_backends:
            result = backend["client"].run_pipeline(
                medium_rawacf,
                stages=["acf", "fitacf", "lmfit", "grid"],
            )
            assert result["stages"]["fitacf"]["nranges"] > 0


# ── pairwise backend comparison ───────────────────────────────────────────────

class TestBackendConsistency:
    """
    Compare backends numerically.

    Strategy
    --------
    Same-algorithm pairs (cuda + rst both use _np_fitacf):  strict tolerance.
    Cross-algorithm pairs (pythonv2 vs others):              only nranges + ACF power,
      because velocity and spectral-width algorithms intentionally differ.
    """

    @pytest.fixture(scope="class")
    def pipeline_results(self, reachable_backends, small_rawacf):
        """Run the pipeline on every backend once; cache results by URL."""
        cache = {}
        for b in reachable_backends:
            cache[b["url"]] = b["client"].run_pipeline(
                small_rawacf, stages=["acf", "fitacf", "grid"]
            )
        return cache

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _backend_algo(result: dict) -> str:
        """Return a short algorithm-family tag from the backend field."""
        b = result.get("stages", {}).get("fitacf", {}).get("backend", "")
        if "pythonv2" in b:
            return "pythonv2"
        return "numpy"   # cudarst-cpu and rst-numpy both use _np_fitacf

    def _pairwise_compare(self, backends, results, stage, strict=True):
        if len(backends) < 2:
            pytest.skip("Need at least 2 backends to compare")

        for a, b in itertools.combinations(backends, 2):
            res_a = results[a["url"]]["stages"].get(stage, {})
            res_b = results[b["url"]]["stages"].get(stage, {})

            comparison = {stage: __import__("compare").compare_stage_results(
                stage, res_a, res_b
            )}
            summary = summarise(comparison)
            failures = [c for checks in comparison.values() for c in checks if not c.passed]

            if strict:
                assert not failures, (
                    f"Backend mismatch for '{stage}' between "
                    f"{a['url']} and {b['url']}:\n{summary}"
                )
            else:
                # Loose mode: only fail on nranges mismatch
                hard = [c for c in failures if "nranges" in c.field]
                assert not hard, (
                    f"nranges mismatch for '{stage}' between "
                    f"{a['url']} and {b['url']}:\n{summary}"
                )

    # ── nranges ───────────────────────────────────────────────────────────────

    def test_nranges_consistent(self, reachable_backends, pipeline_results):
        """All backends must report the same number of range gates."""
        nranges_set = {
            pipeline_results[b["url"]]["stages"]["fitacf"]["nranges"]
            for b in reachable_backends
            if "fitacf" in pipeline_results[b["url"]]["stages"]
        }
        assert len(nranges_set) == 1, \
            f"nranges differs between backends: {nranges_set}"

    # ── ACF power: deterministic for same data → all backends must agree ──────

    def test_pairwise_acf_power(self, reachable_backends, pipeline_results):
        """ACF lag-0 power depends only on the raw data — all backends must agree."""
        self._pairwise_compare(reachable_backends, pipeline_results, "acf", strict=True)

    # ── fitacf: strict only for same-algorithm pairs ──────────────────────────

    def test_pairwise_fitacf_velocity(self, reachable_backends, pipeline_results):
        """
        cuda and rst use identical numpy code → strict agreement.
        pythonv2 uses a different 2-pass algorithm → only nranges checked.
        """
        if len(reachable_backends) < 2:
            pytest.skip("Need at least 2 backends to compare")

        for a, b in itertools.combinations(reachable_backends, 2):
            same_algo = (self._backend_algo(pipeline_results[a["url"]]) ==
                         self._backend_algo(pipeline_results[b["url"]]))
            self._pairwise_compare([a, b], pipeline_results, "fitacf",
                                   strict=same_algo)

    def test_pairwise_grid_velocity(self, reachable_backends, pipeline_results):
        self._pairwise_compare(reachable_backends, pipeline_results, "grid", strict=False)

    @pytest.mark.slow
    def test_full_comparison_report(self, reachable_backends, pipeline_results, capsys):
        """Print a full cross-backend comparison report (always passes; informational)."""
        if len(reachable_backends) < 2:
            pytest.skip("Need at least 2 backends")

        for a, b in itertools.combinations(reachable_backends, 2):
            comparison = compare_pipelines(
                pipeline_results[a["url"]],
                pipeline_results[b["url"]],
                label_a=a["label"],
                label_b=b["label"],
            )
            print(f"\n{'='*60}")
            print(f"Comparison: {a['label']}  vs  {b['label']}")
            print(summarise(comparison))


# ── parameter sensitivity tests ───────────────────────────────────────────────

class TestParameterConsistency:
    """
    Verify that parameter changes affect output in the expected direction
    on every backend (not a cross-backend comparison, but per-backend sanity).
    """

    def test_higher_min_power_fewer_good_ranges(
        self, reachable_backends, small_rawacf
    ):
        for backend in reachable_backends:
            client = backend["client"]
            r_low  = client.run_pipeline(small_rawacf, stages=["fitacf"],
                                         parameters={"min_power": 1.0})
            r_high = client.run_pipeline(small_rawacf, stages=["fitacf"],
                                         parameters={"min_power": 10.0})
            good_low  = r_low["stages"]["fitacf"]["good_ranges"]
            good_high = r_high["stages"]["fitacf"]["good_ranges"]
            assert good_high <= good_low, (
                f"{backend['url']}: higher min_power should give ≤ good_ranges "
                f"(got {good_high} > {good_low})"
            )

    def test_xcf_flag_does_not_crash(self, reachable_backends, small_rawacf):
        for backend in reachable_backends:
            for xcf in (True, False):
                result = backend["client"].run_pipeline(
                    small_rawacf, stages=["fitacf"],
                    parameters={"xcf_enabled": xcf}
                )
                assert "fitacf" in result["stages"]


# ── Cross-backend output sanity (each backend independently, medium dataset) ──

class TestOutputSanity:
    """
    Per-backend physical sanity checks on the medium dataset (75 ranges, 18 lags).
    Not a cross-backend comparison — tests that each backend produces
    physically credible output on realistic synthetic data.
    """

    @pytest.fixture(scope="class")
    def medium_results(self, reachable_backends, medium_rawacf):
        cache = {}
        for b in reachable_backends:
            cache[b["url"]] = b["client"].run_pipeline(
                medium_rawacf, stages=["acf", "fitacf"]
            )
        return cache

    def test_nranges_matches_file(self, reachable_backends, medium_results):
        """All backends must report nranges=75 for the medium test file."""
        for b in reachable_backends:
            fa = medium_results[b["url"]]["stages"]["fitacf"]
            assert fa["nranges"] == 75, \
                f"{b['url']}: expected nranges=75, got {fa['nranges']}"

    def test_good_ranges_nonzero(self, reachable_backends, medium_results):
        """Medium file has strong signal; all backends should find >10 good ranges."""
        for b in reachable_backends:
            fa = medium_results[b["url"]]["stages"]["fitacf"]
            assert fa["good_ranges"] > 10, \
                f"{b['url']}: expected >10 good ranges, got {fa['good_ranges']}"

    def test_acf_power_decreases_with_range(self, reachable_backends, medium_results):
        """
        Test data uses exponential decay (power = 50 * exp(-0.02*r)).
        The first range gate should have higher power than the last.
        """
        for b in reachable_backends:
            acf = medium_results[b["url"]]["stages"]["acf"]
            pwr = [p for p in acf.get("acf_power", []) if p is not None and p >= 0]
            assert len(pwr) >= 10, f"{b['url']}: too few valid power values"
            # First 5 ranges should be stronger than last 5 on average
            early = sum(pwr[:5]) / 5
            late  = sum(pwr[-5:]) / 5
            assert early > late, \
                f"{b['url']}: ACF power should decay with range (early={early:.2f}, late={late:.2f})"

    def test_spectral_width_non_negative_medium(self, reachable_backends, medium_results):
        """Spectral widths must be ≥ 0 everywhere."""
        for b in reachable_backends:
            fa = medium_results[b["url"]]["stages"]["fitacf"]
            for w in fa.get("spectral_width", []):
                if w is not None:
                    assert w >= 0, f"{b['url']}: negative spectral width {w}"

    def test_power_positive_for_good_ranges(self, reachable_backends, medium_results):
        """Every range with a valid velocity should have positive power."""
        for b in reachable_backends:
            fa   = medium_results[b["url"]]["stages"]["fitacf"]
            vels = fa.get("velocity", [])
            pwrs = fa.get("power", [])
            for v, p in zip(vels, pwrs):
                if v is not None:
                    assert p is not None and p >= 0, \
                        f"{b['url']}: range with valid velocity has non-positive power (p={p})"

    def test_elevation_field_present(self, reachable_backends, medium_results):
        """Elevation field must be returned (may be empty if XCF absent)."""
        for b in reachable_backends:
            fa = medium_results[b["url"]]["stages"]["fitacf"]
            assert "elevation" in fa, f"{b['url']}: 'elevation' field missing from fitacf"

    def test_cuda_rst_exact_agreement(self, reachable_backends, medium_results):
        """
        cuda and rst backends both use _np_fitacf — they must agree to within
        floating-point precision on every field.
        """
        cuda_url = next((b["url"] for b in reachable_backends
                         if "8002" in b["url"]), None)
        rst_url  = next((b["url"] for b in reachable_backends
                         if "8003" in b["url"]), None)
        if not cuda_url or not rst_url:
            pytest.skip("Need both cuda (8002) and rst (8003) backends")

        import compare as cmp
        res_c = medium_results[cuda_url]["stages"]["fitacf"]
        res_r = medium_results[rst_url]["stages"]["fitacf"]
        checks = cmp.compare_fitacf_results(res_c, res_r)
        failures = [c for c in checks if not c.passed]
        assert not failures, (
            "cuda and rst should agree (same numpy code):\n" +
            "\n".join(str(c) for c in failures)
        )


class TestCNVMAPCoverage:
    """Verify cnvmap stage runs and returns structurally valid output."""

    def test_cnvmap_runs_without_error(self, reachable_backends, medium_rawacf):
        """cnvmap stage must complete successfully on the medium dataset."""
        for b in reachable_backends:
            result = b["client"].run_pipeline(
                medium_rawacf,
                stages=["fitacf", "grid", "cnvmap"],
            )
            assert "cnvmap" in result["stages"], \
                f"{b['url']}: cnvmap stage missing from result"

    def test_cnvmap_returns_scalar_fields(self, reachable_backends, medium_rawacf):
        """cnvmap must return at minimum chi_squared and order (or potential_max)."""
        for b in reachable_backends:
            result = b["client"].run_pipeline(
                medium_rawacf, stages=["fitacf", "grid", "cnvmap"]
            )
            cm = result["stages"]["cnvmap"]
            # At least one scalar summary field must be present
            has_field = any(k in cm for k in ("chi_squared", "order", "potential_max",
                                               "num_vectors", "chi2"))
            assert has_field, \
                f"{b['url']}: cnvmap missing expected summary fields. Got: {list(cm.keys())}"


class TestFullPipelineCoverage:
    """End-to-end pipeline tests with the medium and large datasets."""

    def test_all_five_stages_medium(self, reachable_backends, medium_rawacf, job_timeout):
        """All five stages must complete and return non-empty stage results."""
        for b in reachable_backends:
            result = b["client"].run_pipeline(
                medium_rawacf,
                stages=["acf", "fitacf", "lmfit", "grid", "cnvmap"],
            )
            stages = result["stages"]
            for stage in ("acf", "fitacf", "grid", "cnvmap"):
                assert stage in stages, \
                    f"{b['url']}: stage '{stage}' missing from full-pipeline result"
            # lmfit may be a no-op for some backends but must not cause failure
            assert result.get("performance_metrics") is not None

    @pytest.mark.slow
    def test_large_file_pipeline(self, reachable_backends, test_data_dir, job_timeout):
        """Large dataset (150 ranges) must process without error."""
        large = test_data_dir / "test_large.rawacf"
        if not large.exists():
            pytest.skip(f"Large test file not found: {large}")
        for b in reachable_backends:
            result = b["client"].run_pipeline(large, stages=["acf", "fitacf"])
            fa = result["stages"]["fitacf"]
            assert fa["nranges"] == 150, \
                f"{b['url']}: expected nranges=150 for large file, got {fa['nranges']}"
            assert fa["good_ranges"] > 20, \
                f"{b['url']}: expected >20 good ranges in large file, got {fa['good_ranges']}"

    def test_processing_time_reasonable(self, reachable_backends, medium_rawacf):
        """Medium file pipeline should complete within 30 s per backend."""
        import time
        LIMIT_S = 30.0
        for b in reachable_backends:
            t0 = time.time()
            b["client"].run_pipeline(medium_rawacf, stages=["acf", "fitacf"])
            elapsed = time.time() - t0
            assert elapsed < LIMIT_S, \
                f"{b['url']}: pipeline took {elapsed:.1f}s, limit is {LIMIT_S}s"


class TestIQProcessorHTTP:
    """IQ processor tested via a synthetic local call (no HTTP needed)."""

    @pytest.fixture(autouse=True)
    def import_guard(self):
        pytest.importorskip("superdarn_gpu", reason="pythonv2 not installed")

    def test_iq_to_acf_shape(self):
        """IQProcessor.process returns a RawACF with the correct array shapes."""
        from superdarn_gpu.processing.iq import IQProcessor
        import numpy as np

        nrang, mplgs, nave = 10, 15, 5
        rng = np.random.default_rng(7)
        iq  = (rng.standard_normal((nrang, nave, mplgs)) +
               1j * rng.standard_normal((nrang, nave, mplgs))).astype(np.complex64)

        proc   = IQProcessor()
        rawacf = proc.process(iq, nrang=nrang, mplgs=mplgs, nave=nave)

        assert rawacf.acf.shape == (nrang, mplgs), \
            f"Expected acf shape ({nrang}, {mplgs}), got {rawacf.acf.shape}"
        assert rawacf.power.shape == (nrang,)

    def test_encode_decode_roundtrip(self):
        """
        encode_iq normalises by max(|acf|) before int16 packing.
        decode_iq recovers the normalised values, so the roundtrip is:
            restored ≈ acf / max(|acf|)
        The relative error should be < 0.5% (int16 quantisation noise = 1/32767).
        """
        from superdarn_gpu.processing.iq import IQProcessor
        import numpy as np

        nrang, mplgs = 8, 12
        rng = np.random.default_rng(13)
        acf = (rng.standard_normal((nrang, mplgs)) +
               1j * rng.standard_normal((nrang, mplgs))).astype(np.complex64)

        raw      = IQProcessor.encode_iq(acf)
        restored = IQProcessor.decode_iq(raw)

        # Compare against normalised reference (what encode/decode actually preserves)
        max_val  = float(np.max(np.abs(acf))) + 1e-30
        expected = (acf / max_val).astype(np.complex64)
        err = np.abs(restored - expected) / (np.abs(expected) + 1e-10)
        assert float(np.max(err)) < 5e-3, \
            f"encode/decode roundtrip error too large: {float(np.max(err)):.5f}"

    def test_power_spectrum_shape(self):
        """power_spectrum should return (nrang, n_fft//2+1) float32 array."""
        from superdarn_gpu.processing.iq import IQProcessor
        import numpy as np

        nrang, nave, mplgs = 5, 16, 10
        rng = np.random.default_rng(21)
        iq  = (rng.standard_normal((nrang, nave, mplgs)) +
               1j * rng.standard_normal((nrang, nave, mplgs))).astype(np.complex64)
        spec = IQProcessor().power_spectrum(iq, n_fft=nave)
        assert spec.shape == (nrang, nave // 2 + 1)


# ── Algorithm-level unit tests (Python backend only, no HTTP) ─────────────────
# These import the pythonv2 library directly and test the new algorithm modules.

class TestLagValidation:
    """Verify LagValidator produces physically sensible masks."""

    @pytest.fixture(autouse=True)
    def import_guard(self):
        pytest.importorskip("superdarn_gpu", reason="pythonv2 not installed")

    def _make_acf(self, nrang=20, mplgs=10, snr_db=20.0, noise_amp=1.0):
        """Generate synthetic exponential-decay ACF with noise."""
        import numpy as np
        snr = 10 ** (snr_db / 20.0)
        t = np.arange(mplgs)
        sig = snr * noise_amp * np.exp(-0.1 * t)
        # Add complex noise
        rng = np.random.default_rng(42)
        noise = rng.standard_normal((nrang, mplgs)) + 1j * rng.standard_normal((nrang, mplgs))
        acf = (sig[None, :] + noise * noise_amp).astype(np.complex64)
        pwr0 = np.abs(acf[:, 0]).astype(np.float32)
        return acf, pwr0

    def test_high_snr_all_valid(self):
        """At 30 dB SNR virtually every lag should survive the validity check."""
        from superdarn_gpu.algorithms.lag_validation import LagValidator
        import numpy as np
        validator = LagValidator()
        acf, pwr0 = self._make_acf(snr_db=30.0)
        mask = validator.compute_lag_validity_mask(acf, pwr0, None, nave=20, noise_pwr=0.0)
        frac_valid = float(np.mean(mask))
        assert frac_valid > 0.7, f"Expected >70% valid lags at 30 dB SNR, got {frac_valid:.2%}"

    def test_low_snr_prunes_lags(self):
        """At -10 dB SNR later lags should be cut off."""
        from superdarn_gpu.algorithms.lag_validation import LagValidator
        import numpy as np
        validator = LagValidator()
        acf, pwr0 = self._make_acf(snr_db=-10.0)
        mask = validator.compute_lag_validity_mask(acf, pwr0, None, nave=5, noise_pwr=0.5)
        n_valid = np.sum(mask, axis=1)
        # At low SNR at least some ranges should be entirely masked
        assert np.any(n_valid == 0) or float(np.mean(mask[:, -1])) < 0.5, (
            "Low-SNR ACF should prune at least some later lags"
        )

    def test_cumsum_propagation(self):
        """Once a lag is flagged bad, all subsequent lags in that range must be masked."""
        from superdarn_gpu.algorithms.lag_validation import LagValidator
        import numpy as np
        nrang, mplgs = 10, 15
        validator = LagValidator()
        # Craft ACF where lag 5 is deliberately zero (below noise)
        acf = np.ones((nrang, mplgs), dtype=np.complex64) * 100.0
        acf[:, 5:] = 0.0 + 0j   # all lags from 5 onward are zero
        pwr0 = np.abs(acf[:, 0]).astype(np.float32)
        mask = validator.compute_lag_validity_mask(acf, pwr0, None, nave=20, noise_pwr=0.5)
        # Lags 5+ must not be valid for any range
        assert np.all(~mask[:, 5:].astype(bool)), \
            "Lags after a zero-magnitude lag must all be invalid (cumsum propagation)"

    def test_min_lags_gate(self):
        """Ranges with fewer than MIN_LAGS valid lags must be entirely masked."""
        from superdarn_gpu.algorithms.lag_validation import LagValidator
        import numpy as np
        validator = LagValidator()
        nrang, mplgs = 5, 4   # mplgs < MIN_LAGS(=3) in some ranges
        acf = np.ones((nrang, mplgs), dtype=np.complex64)
        # Make lag 1+ zero for all ranges → only lag 0 survives → < MIN_LAGS
        acf[:, 1:] = 0.0 + 0j
        pwr0 = np.abs(acf[:, 0]).astype(np.float32)
        mask = validator.compute_lag_validity_mask(acf, pwr0, None, nave=20, noise_pwr=0.5)
        # With only 1 valid lag, every range should be entirely masked
        assert not np.any(mask), "Ranges with <MIN_LAGS valid lags must be fully masked"


class TestBPSigma:
    """Verify Bendat-Piersol sigma has correct limiting behaviour."""

    @pytest.fixture(autouse=True)
    def import_guard(self):
        pytest.importorskip("superdarn_gpu", reason="pythonv2 not installed")

    def test_sigma_positive(self):
        """BP sigma must always be positive (never zero or NaN)."""
        from superdarn_gpu.algorithms.lag_validation import LagValidator
        import numpy as np
        validator = LagValidator()
        rng = np.random.default_rng(0)
        acf  = (rng.standard_normal((30, 12)) + 1j * rng.standard_normal((30, 12))).astype(np.complex64)
        pwr0 = np.abs(acf[:, 0]).astype(np.float32) + 0.1
        sigma = validator.compute_bendat_piersol_sigma(acf, pwr0, None, nave=20)
        assert np.all(sigma > 0), "BP sigma must be positive everywhere"
        assert not np.any(np.isnan(sigma)), "BP sigma must not contain NaN"

    def test_sigma_formula_direct(self):
        """
        Verify B-P sigma against the closed-form:
            sigma[r,l] = pwr0 * sqrt((2*R_norm^2 + 1/nave) / (2*nave))
        (derived by substituting alpha_2 = nave/(1+nave*R_norm^2) into the
        Bendat-Piersol expression).
        """
        from superdarn_gpu.algorithms.lag_validation import LagValidator
        import numpy as np

        validator = LagValidator()
        nrang, mplgs, nave = 2, 5, 100
        A = 500.0
        # Vary R_norm across lags: 1.0, 0.5, 0.1, 0.01, 0.001
        r_norms = np.array([1.0, 0.5, 0.1, 0.01, 0.001])
        acf = np.tile((A * r_norms + 0j).astype(np.complex64), (nrang, 1))
        pwr0 = np.full(nrang, A, dtype=np.float32)

        sigma = np.asarray(validator.compute_bendat_piersol_sigma(acf, pwr0, None, nave))

        expected = A * np.sqrt((2 * r_norms**2 + 1.0 / nave) / (2 * nave))
        np.testing.assert_allclose(
            sigma[0], expected.astype(np.float32),
            rtol=0.02, err_msg="B-P sigma does not match analytic formula"
        )

    def test_relative_precision_worse_at_low_snr(self):
        """
        The weight 1/sigma should be larger (better precision) at high SNR (large R_norm)
        than at low SNR (small R_norm), so the fitter trusts those lags more.
        """
        from superdarn_gpu.algorithms.lag_validation import LagValidator
        import numpy as np

        validator = LagValidator()
        nrang, mplgs, nave = 5, 5, 20
        A = 1000.0
        r_high = np.array([1.0, 0.9, 0.8, 0.7, 0.6])   # slow decay, high SNR
        r_low  = np.array([1.0, 0.1, 0.01, 0.001, 0.0001])  # fast decay, low SNR

        acf_h = np.tile((A * r_high + 0j).astype(np.complex64), (nrang, 1))
        acf_l = np.tile((A * r_low  + 0j).astype(np.complex64), (nrang, 1))
        pwr0  = np.full(nrang, A, dtype=np.float32)

        s_h = np.asarray(validator.compute_bendat_piersol_sigma(acf_h, pwr0, None, nave))
        s_l = np.asarray(validator.compute_bendat_piersol_sigma(acf_l, pwr0, None, nave))

        # sigma/|R| (relative measurement uncertainty) should be larger at low SNR
        rel_h = np.mean(s_h[:, 1:] / (A * r_high[1:]))  # skip lag-0
        rel_l = np.mean(s_l[:, 1:] / (A * r_low[1:] + 1e-10))
        assert rel_l > rel_h, (
            f"Relative B-P sigma should be larger at low SNR ({rel_l:.4f}) "
            f"than at high SNR ({rel_h:.4f})"
        )


class TestTwoPassFitting:
    """Check that two-pass log-linear fit gives physically correct velocity/width."""

    @pytest.fixture(autouse=True)
    def import_guard(self):
        pytest.importorskip("superdarn_gpu", reason="pythonv2 not installed")

    def _synthetic_acf(self, nrang=10, mplgs=15, vel_rads_per_lag=0.5,
                       decay=0.08, noise_frac=0.01):
        """
        Generate clean synthetic ACF with known velocity and exponential decay.
        R(l) = A * exp(decay*l) * exp(i * vel_rads_per_lag * l)
        """
        import numpy as np
        A = 1000.0
        l = np.arange(mplgs)
        mag   = A * np.exp(-decay * l)
        phase = vel_rads_per_lag * l
        acf_1d = (mag * np.cos(phase) + 1j * mag * np.sin(phase)).astype(np.complex64)
        noise = noise_frac * A * (np.random.randn(nrang, mplgs) +
                                   1j * np.random.randn(nrang, mplgs))
        acf = acf_1d[None, :] + noise.astype(np.complex64)
        return acf

    def test_velocity_factor_correct(self):
        """
        vel_factor = c/(4π·f·Δτ). With tfreq=12 MHz and mpinc=1500 µs
        this should give ≈1326 m/s/rad — validate the formula, not just the number.
        """
        import numpy as np
        c      = 3e8
        tfreq  = 12e6   # Hz
        mpinc  = 1500e-6  # s
        vel_factor = c / (4 * np.pi * tfreq * mpinc)
        assert 1300 < vel_factor < 1360, \
            f"vel_factor for 12 MHz / 1500 µs should be ~1326, got {vel_factor:.1f}"

    def test_fit_returns_required_keys(self):
        """Fitter must return velocity, power, spectral widths and their errors."""
        from superdarn_gpu.algorithms.fitting import LeastSquaresFitter
        from superdarn_gpu.algorithms.lag_validation import LagValidator
        import numpy as np
        fitter    = LeastSquaresFitter()
        validator = LagValidator()

        acf  = self._synthetic_acf(nrang=5, mplgs=15)
        pwr0 = np.abs(acf[:, 0]).astype(np.float32)
        mask = validator.compute_lag_validity_mask(acf, pwr0, None, nave=20)
        sigma = validator.compute_bendat_piersol_sigma(acf, pwr0, None, nave=20)

        result = fitter.fit_lorentzian_batch(
            acf, pwr0,
            valid_mask=mask, bp_sigma=sigma,
            noise_pwr=0.0, tfreq_hz=12e6,
            lag_time_step_sec=1500e-6,   # mpinc = 1500 µs
        )
        required = {"velocity", "power", "spectral_width", "spectral_width_sigma",
                    "velocity_error", "power_error", "spectral_width_error",
                    "spectral_width_sigma_error"}
        missing = required - set(result.keys())
        assert not missing, f"Fitter result missing keys: {missing}"

    def test_velocity_sign_and_magnitude(self):
        """
        A synthetic ACF with positive phase slope should give positive velocity.
        The magnitude should be within 50% of the expected physics value.
        """
        from superdarn_gpu.algorithms.fitting import LeastSquaresFitter
        from superdarn_gpu.algorithms.lag_validation import LagValidator
        import numpy as np

        # RST one-param LS fits individual per-lag phases, so the cumulative phase
        # must stay within (-π, π] across all lags: vel_rads * (mplgs-1) < π.
        # With mplgs=15: vel_rads < π/14 ≈ 0.224. Use 0.15 for safety.
        vel_rads = 0.15   # rad per lag — well within unambiguous range
        mpinc_us = 1500.0
        tfreq_hz = 12e6
        vel_factor = 3e8 / (4 * np.pi * tfreq_hz * mpinc_us * 1e-6)
        expected_vel = vel_rads * vel_factor

        fitter    = LeastSquaresFitter()
        validator = LagValidator()
        acf  = self._synthetic_acf(nrang=8, mplgs=15, vel_rads_per_lag=vel_rads, noise_frac=0.001)
        pwr0 = np.abs(acf[:, 0]).astype(np.float32)
        mask = validator.compute_lag_validity_mask(acf, pwr0, None, nave=50)
        sigma = validator.compute_bendat_piersol_sigma(acf, pwr0, None, nave=50)
        result = fitter.fit_lorentzian_batch(
            acf, pwr0, valid_mask=mask, bp_sigma=sigma,
            noise_pwr=0.0, tfreq_hz=tfreq_hz,
            lag_time_step_sec=mpinc_us * 1e-6,
        )
        vels = np.array(result["velocity"])
        good = np.isfinite(vels)
        assert good.any(), "At least some ranges should have valid velocities"
        mean_vel = float(np.nanmean(vels[good]))
        assert mean_vel > 0, f"Positive phase slope should give positive velocity; got {mean_vel:.1f}"
        assert abs(mean_vel - expected_vel) / expected_vel < 0.5, (
            f"Velocity {mean_vel:.1f} deviates >50% from expected {expected_vel:.1f} m/s"
        )


class TestSigmaWidth:
    """Verify sigma (quadratic) spectral width is distinct from lambda (linear) width."""

    @pytest.fixture(autouse=True)
    def import_guard(self):
        pytest.importorskip("superdarn_gpu", reason="pythonv2 not installed")

    def test_sigma_width_positive_for_gaussian_acf(self):
        """Gaussian-envelope ACF → positive sigma width."""
        from superdarn_gpu.algorithms.fitting import LeastSquaresFitter
        from superdarn_gpu.algorithms.lag_validation import LagValidator
        import numpy as np

        nrang, mplgs = 5, 20
        A   = 1000.0
        l   = np.arange(mplgs, dtype=np.float32)
        # Gaussian envelope: log|R| = a - sigma_coeff * t^2
        sigma_coeff = 0.01
        mag = A * np.exp(-sigma_coeff * l ** 2)
        acf = (mag + 0j).astype(np.complex64)
        acf = np.tile(acf, (nrang, 1))

        fitter    = LeastSquaresFitter()
        validator = LagValidator()
        pwr0 = np.abs(acf[:, 0]).astype(np.float32)
        mask = validator.compute_lag_validity_mask(acf, pwr0, None, nave=20)
        sigma_bp = validator.compute_bendat_piersol_sigma(acf, pwr0, None, nave=20)
        result = fitter.fit_lorentzian_batch(
            acf, pwr0, valid_mask=mask, bp_sigma=sigma_bp,
            noise_pwr=0.0, tfreq_hz=12e6,
            lag_time_step_sec=1500e-6,
        )
        w_s = np.array(result["spectral_width_sigma"])
        good = np.isfinite(w_s) & (w_s > 0)
        assert good.any(), "Gaussian ACF should yield positive sigma spectral width"

    def test_lambda_ne_sigma_width_for_exponential(self):
        """Exponential ACF: lambda width ≠ sigma width (they measure different shapes)."""
        from superdarn_gpu.algorithms.fitting import LeastSquaresFitter
        from superdarn_gpu.algorithms.lag_validation import LagValidator
        import numpy as np

        nrang, mplgs = 5, 20
        A, decay = 1000.0, 0.1
        l   = np.arange(mplgs, dtype=np.float32)
        mag = A * np.exp(-decay * l)
        acf = np.tile((mag + 0j).astype(np.complex64), (nrang, 1))

        fitter    = LeastSquaresFitter()
        validator = LagValidator()
        pwr0 = np.abs(acf[:, 0]).astype(np.float32)
        mask = validator.compute_lag_validity_mask(acf, pwr0, None, nave=20)
        sigma_bp = validator.compute_bendat_piersol_sigma(acf, pwr0, None, nave=20)
        result = fitter.fit_lorentzian_batch(
            acf, pwr0, valid_mask=mask, bp_sigma=sigma_bp,
            noise_pwr=0.0, tfreq_hz=12e6,
            lag_time_step_sec=1500e-6,
        )
        w_l = np.nanmean(result["spectral_width"])
        w_s = np.nanmean(result["spectral_width_sigma"])
        assert abs(w_l - w_s) / (abs(w_l) + 1.0) > 0.05, (
            f"Lambda ({w_l:.1f}) and sigma ({w_s:.1f}) widths should differ for exponential ACF"
        )
