"""
Test result reporters for generating comparison reports

Provides JSON, HTML, and console output formats for test results.
"""

import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from abc import ABC, abstractmethod

from .framework import ModuleComparison, ComparisonResult, TestStatus, Backend


class ComparisonReporter(ABC):
    """Base class for comparison reporters"""
    
    @abstractmethod
    def generate_report(self, 
                       results: Dict[str, ModuleComparison],
                       output_path: Optional[Path] = None) -> str:
        """Generate report from comparison results"""
        pass


class JSONReporter(ComparisonReporter):
    """Generate JSON format reports"""
    
    def generate_report(self,
                       results: Dict[str, ModuleComparison],
                       output_path: Optional[Path] = None) -> str:
        """
        Generate JSON report
        
        Parameters
        ----------
        results : dict
            Comparison results by module name
        output_path : Path, optional
            Output file path (prints to stdout if None)
            
        Returns
        -------
        str
            JSON report string
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self._generate_summary(results),
            'modules': {}
        }
        
        for name, comparison in results.items():
            report['modules'][name] = self._serialize_comparison(comparison)
            
        json_str = json.dumps(report, indent=2, default=str)
        
        if output_path:
            output_path.write_text(json_str)
            
        return json_str
    
    def _generate_summary(self, results: Dict[str, ModuleComparison]) -> Dict[str, Any]:
        """Generate summary statistics"""
        total_tests = sum(m.total_tests for m in results.values())
        passed = sum(m.passed_tests for m in results.values())
        failed = sum(m.failed_tests for m in results.values())
        
        speedups = []
        errors = []
        
        for comparison in results.values():
            for result in comparison.results:
                if result.speedup > 0:
                    speedups.append(result.speedup)
                errors.append(result.mean_absolute_error)
        
        return {
            'modules_tested': len(results),
            'total_tests': total_tests,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total_tests if total_tests > 0 else 0,
            'avg_speedup': sum(speedups) / len(speedups) if speedups else 0,
            'min_speedup': min(speedups) if speedups else 0,
            'max_speedup': max(speedups) if speedups else 0,
            'avg_error': sum(errors) / len(errors) if errors else 0
        }
    
    def _serialize_comparison(self, comparison: ModuleComparison) -> Dict[str, Any]:
        """Serialize a module comparison"""
        return {
            'module_name': comparison.module_name,
            'module_version': comparison.module_version,
            'description': comparison.description,
            'total_tests': comparison.total_tests,
            'passed_tests': comparison.passed_tests,
            'failed_tests': comparison.failed_tests,
            'skipped_tests': comparison.skipped_tests,
            'avg_speedup': comparison.avg_speedup,
            'avg_error': comparison.avg_error,
            'timestamp': comparison.timestamp,
            'results': [self._serialize_result(r) for r in comparison.results]
        }
    
    def _serialize_result(self, result: ComparisonResult) -> Dict[str, Any]:
        """Serialize a comparison result"""
        return {
            'test_name': result.test_name,
            'reference_backend': result.reference_backend.value,
            'comparison_backend': result.comparison_backend.value,
            'status': result.status.value,
            'values_match': result.values_match,
            'max_absolute_error': result.max_absolute_error,
            'max_relative_error': result.max_relative_error,
            'mean_absolute_error': result.mean_absolute_error,
            'correlation': result.correlation,
            'reference_time': result.reference_time,
            'comparison_time': result.comparison_time,
            'speedup': result.speedup,
            'reference_memory': result.reference_memory,
            'comparison_memory': result.comparison_memory,
            'memory_ratio': result.memory_ratio,
            'field_comparisons': result.field_comparisons,
            'error_message': result.error_message
        }


class HTMLReporter(ComparisonReporter):
    """Generate HTML format reports with interactive visualization"""
    
    def generate_report(self,
                       results: Dict[str, ModuleComparison],
                       output_path: Optional[Path] = None) -> str:
        """
        Generate interactive HTML report
        
        Parameters
        ----------
        results : dict
            Comparison results by module name
        output_path : Path, optional
            Output file path
            
        Returns
        -------
        str
            HTML report string
        """
        summary = self._generate_summary(results)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RST Module Comparison Report</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .status-passed {{ background-color: #10b981; color: white; }}
        .status-failed {{ background-color: #ef4444; color: white; }}
        .status-skipped {{ background-color: #6b7280; color: white; }}
        .metric-card {{ transition: transform 0.2s; }}
        .metric-card:hover {{ transform: translateY(-2px); }}
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <!-- Header -->
        <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
            <h1 class="text-3xl font-bold text-gray-800 mb-2">RST Module Comparison Report</h1>
            <p class="text-gray-600">C/CUDA vs Python Implementation Testing</p>
            <p class="text-sm text-gray-500 mt-2">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <!-- Summary Cards -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            {self._render_summary_cards(summary)}
        </div>
        
        <!-- Speedup Chart -->
        <div class="bg-white rounded-lg shadow-lg p-6 mb-8">
            <h2 class="text-xl font-semibold mb-4">Performance Comparison</h2>
            <canvas id="speedupChart" height="100"></canvas>
        </div>
        
        <!-- Module Results -->
        <div class="space-y-6">
            {self._render_module_sections(results)}
        </div>
    </div>
    
    <script>
        {self._generate_chart_js(results)}
    </script>
</body>
</html>"""
        
        if output_path:
            output_path.write_text(html)
            
        return html
    
    def _generate_summary(self, results: Dict[str, ModuleComparison]) -> Dict[str, Any]:
        """Generate summary statistics"""
        total_tests = sum(m.total_tests for m in results.values())
        passed = sum(m.passed_tests for m in results.values())
        failed = sum(m.failed_tests for m in results.values())
        
        speedups = []
        for comparison in results.values():
            for result in comparison.results:
                if result.speedup > 0:
                    speedups.append(result.speedup)
        
        return {
            'modules_tested': len(results),
            'total_tests': total_tests,
            'passed': passed,
            'failed': failed,
            'pass_rate': (passed / total_tests * 100) if total_tests > 0 else 0,
            'avg_speedup': sum(speedups) / len(speedups) if speedups else 0
        }
    
    def _render_summary_cards(self, summary: Dict[str, Any]) -> str:
        """Render summary metric cards"""
        return f"""
            <div class="metric-card bg-white rounded-lg shadow p-6">
                <div class="text-sm text-gray-500 uppercase">Modules Tested</div>
                <div class="text-3xl font-bold text-gray-800">{summary['modules_tested']}</div>
            </div>
            <div class="metric-card bg-white rounded-lg shadow p-6">
                <div class="text-sm text-gray-500 uppercase">Tests Passed</div>
                <div class="text-3xl font-bold text-green-600">{summary['passed']}/{summary['total_tests']}</div>
            </div>
            <div class="metric-card bg-white rounded-lg shadow p-6">
                <div class="text-sm text-gray-500 uppercase">Pass Rate</div>
                <div class="text-3xl font-bold text-blue-600">{summary['pass_rate']:.1f}%</div>
            </div>
            <div class="metric-card bg-white rounded-lg shadow p-6">
                <div class="text-sm text-gray-500 uppercase">Avg Speedup</div>
                <div class="text-3xl font-bold text-purple-600">{summary['avg_speedup']:.2f}x</div>
            </div>
        """
    
    def _render_module_sections(self, results: Dict[str, ModuleComparison]) -> str:
        """Render detailed sections for each module"""
        sections = []
        
        for name, comparison in results.items():
            status_class = 'status-passed' if comparison.failed_tests == 0 else 'status-failed'
            status_text = 'PASSED' if comparison.failed_tests == 0 else 'FAILED'
            
            # Build results table
            rows = []
            for result in comparison.results:
                result_status_class = f'status-{result.status.value}'
                rows.append(f"""
                    <tr class="border-b">
                        <td class="py-3 px-4">{result.reference_backend.value}</td>
                        <td class="py-3 px-4">{result.comparison_backend.value}</td>
                        <td class="py-3 px-4">
                            <span class="px-2 py-1 rounded text-xs {result_status_class}">{result.status.value.upper()}</span>
                        </td>
                        <td class="py-3 px-4 text-right">{result.reference_time*1000:.2f} ms</td>
                        <td class="py-3 px-4 text-right">{result.comparison_time*1000:.2f} ms</td>
                        <td class="py-3 px-4 text-right font-semibold">{result.speedup:.2f}x</td>
                        <td class="py-3 px-4 text-right">{result.mean_absolute_error:.2e}</td>
                        <td class="py-3 px-4 text-right">{result.correlation:.4f}</td>
                    </tr>
                """)
            
            section = f"""
            <div class="bg-white rounded-lg shadow-lg overflow-hidden">
                <div class="bg-gray-800 text-white px-6 py-4 flex justify-between items-center">
                    <div>
                        <h3 class="text-xl font-semibold">{name.upper()}</h3>
                        <p class="text-gray-300 text-sm">{comparison.description}</p>
                    </div>
                    <span class="px-3 py-1 rounded text-sm {status_class}">{status_text}</span>
                </div>
                <div class="p-6">
                    <div class="grid grid-cols-4 gap-4 mb-6">
                        <div class="text-center p-4 bg-gray-50 rounded">
                            <div class="text-2xl font-bold text-gray-800">{comparison.total_tests}</div>
                            <div class="text-sm text-gray-500">Total Tests</div>
                        </div>
                        <div class="text-center p-4 bg-green-50 rounded">
                            <div class="text-2xl font-bold text-green-600">{comparison.passed_tests}</div>
                            <div class="text-sm text-gray-500">Passed</div>
                        </div>
                        <div class="text-center p-4 bg-purple-50 rounded">
                            <div class="text-2xl font-bold text-purple-600">{comparison.avg_speedup:.2f}x</div>
                            <div class="text-sm text-gray-500">Avg Speedup</div>
                        </div>
                        <div class="text-center p-4 bg-blue-50 rounded">
                            <div class="text-2xl font-bold text-blue-600">{comparison.avg_error:.2e}</div>
                            <div class="text-sm text-gray-500">Avg Error</div>
                        </div>
                    </div>
                    <table class="w-full">
                        <thead class="bg-gray-100">
                            <tr>
                                <th class="py-3 px-4 text-left">Reference</th>
                                <th class="py-3 px-4 text-left">Comparison</th>
                                <th class="py-3 px-4 text-left">Status</th>
                                <th class="py-3 px-4 text-right">Ref Time</th>
                                <th class="py-3 px-4 text-right">Cmp Time</th>
                                <th class="py-3 px-4 text-right">Speedup</th>
                                <th class="py-3 px-4 text-right">MAE</th>
                                <th class="py-3 px-4 text-right">Correlation</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(rows)}
                        </tbody>
                    </table>
                </div>
            </div>
            """
            sections.append(section)
            
        return '\n'.join(sections)
    
    def _generate_chart_js(self, results: Dict[str, ModuleComparison]) -> str:
        """Generate Chart.js configuration"""
        labels = []
        cpu_times = []
        gpu_times = []
        python_times = []
        
        for name, comparison in results.items():
            labels.append(name)
            
            # Find times for different backends
            cpu_time = 0
            gpu_time = 0
            python_time = 0
            
            for result in comparison.results:
                if result.reference_backend == Backend.C_CPU:
                    cpu_time = result.reference_time * 1000
                if result.comparison_backend == Backend.PYTHON_NUMPY:
                    python_time = result.comparison_time * 1000
                if result.comparison_backend == Backend.PYTHON_CUPY:
                    gpu_time = result.comparison_time * 1000
                    
            cpu_times.append(cpu_time)
            gpu_times.append(gpu_time)
            python_times.append(python_time)
        
        return f"""
        const ctx = document.getElementById('speedupChart').getContext('2d');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [
                    {{
                        label: 'C CPU',
                        data: {json.dumps(cpu_times)},
                        backgroundColor: 'rgba(59, 130, 246, 0.8)',
                    }},
                    {{
                        label: 'Python NumPy',
                        data: {json.dumps(python_times)},
                        backgroundColor: 'rgba(16, 185, 129, 0.8)',
                    }},
                    {{
                        label: 'Python CuPy',
                        data: {json.dumps(gpu_times)},
                        backgroundColor: 'rgba(139, 92, 246, 0.8)',
                    }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        position: 'top',
                    }},
                    title: {{
                        display: true,
                        text: 'Processing Time by Backend (ms)'
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Time (ms)'
                        }}
                    }}
                }}
            }}
        }});
        """


class ConsoleReporter(ComparisonReporter):
    """Generate console-friendly text reports"""
    
    def generate_report(self,
                       results: Dict[str, ModuleComparison],
                       output_path: Optional[Path] = None) -> str:
        """Generate console report"""
        lines = []
        lines.append("=" * 80)
        lines.append("RST MODULE COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Summary
        total_tests = sum(m.total_tests for m in results.values())
        passed = sum(m.passed_tests for m in results.values())
        
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Modules tested: {len(results)}")
        lines.append(f"  Total tests:    {total_tests}")
        lines.append(f"  Passed:         {passed}")
        lines.append(f"  Failed:         {total_tests - passed}")
        lines.append("")
        
        # Module details
        for name, comparison in results.items():
            status = "✓ PASSED" if comparison.failed_tests == 0 else "✗ FAILED"
            lines.append(f"\n{name.upper()} {status}")
            lines.append("-" * 40)
            lines.append(f"  Version: {comparison.module_version}")
            lines.append(f"  Tests: {comparison.passed_tests}/{comparison.total_tests}")
            lines.append(f"  Avg Speedup: {comparison.avg_speedup:.2f}x")
            lines.append(f"  Avg Error: {comparison.avg_error:.2e}")
            
            for result in comparison.results:
                lines.append(f"\n  {result.reference_backend.value} → {result.comparison_backend.value}")
                lines.append(f"    Status: {result.status.value}")
                lines.append(f"    Time: {result.reference_time*1000:.2f}ms → {result.comparison_time*1000:.2f}ms ({result.speedup:.2f}x)")
                lines.append(f"    Error: MAE={result.mean_absolute_error:.2e}, Corr={result.correlation:.4f}")
        
        lines.append("")
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        if output_path:
            output_path.write_text(report)
        else:
            print(report)
            
        return report
