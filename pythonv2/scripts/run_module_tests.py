#!/usr/bin/env python3
"""
Run module comparison tests with report generation

Usage:
    python run_module_tests.py                    # Run all tests
    python run_module_tests.py --module acf       # Run specific module
    python run_module_tests.py --report html      # Generate HTML report
    python run_module_tests.py --size large       # Use large test data
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add pythonv2 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.comparison.framework import ComparisonTestFramework, Backend
from tests.comparison.fixtures import generate_test_data
from tests.comparison.reporters import ConsoleReporter, JSONReporter, HTMLReporter


def main():
    parser = argparse.ArgumentParser(description='Run RST module comparison tests')
    parser.add_argument('--module', '-m', nargs='+', 
                        choices=['acf', 'fitacf', 'grid', 'convmap', 'all'],
                        default=['all'],
                        help='Modules to test')
    parser.add_argument('--size', '-s',
                        choices=['small', 'medium', 'large'],
                        default='medium',
                        help='Test data size')
    parser.add_argument('--report', '-r',
                        choices=['console', 'json', 'html', 'all'],
                        default='console',
                        help='Report format')
    parser.add_argument('--output', '-o',
                        type=Path,
                        default=Path('test_reports'),
                        help='Output directory for reports')
    parser.add_argument('--tolerance', '-t',
                        type=float,
                        default=1e-4,
                        help='Relative tolerance for comparisons')
    
    args = parser.parse_args()
    
    # Import processors
    from superdarn_gpu.processing.acf import ACFProcessor
    from superdarn_gpu.processing.fitacf import FitACFProcessor
    from superdarn_gpu.processing.grid import GridProcessor
    from superdarn_gpu.processing.convmap import ConvMapProcessor
    
    # Initialize framework
    framework = ComparisonTestFramework(tolerance_rtol=args.tolerance)
    
    # Module definitions
    module_config = {
        'acf': {
            'processor': ACFProcessor,
            'description': 'Auto-correlation function calculation',
            'version': '1.16'
        },
        'fitacf': {
            'processor': FitACFProcessor,
            'description': 'ACF curve fitting for velocity/width extraction',
            'version': '3.0'
        },
        'grid': {
            'processor': GridProcessor,
            'description': 'Spatial gridding of radar measurements',
            'version': '1.24'
        },
        'convmap': {
            'processor': ConvMapProcessor,
            'description': 'Convection map generation via spherical harmonic fitting',
            'version': '1.17'
        }
    }
    
    # Determine modules to test
    if 'all' in args.module:
        modules_to_test = list(module_config.keys())
    else:
        modules_to_test = args.module
    
    # Register modules
    for name in modules_to_test:
        config = module_config[name]
        framework.register_module(
            name=name,
            python_processor=config['processor'],
            description=config['description'],
            version=config['version']
        )
    
    print(f"\n{'='*60}")
    print(f"RST Module Comparison Tests")
    print(f"{'='*60}")
    print(f"Modules: {', '.join(modules_to_test)}")
    print(f"Data size: {args.size}")
    print(f"Tolerance: {args.tolerance}")
    print(f"{'='*60}\n")
    
    # Run tests
    for name in modules_to_test:
        print(f"Testing {name}...")
        test_data = generate_test_data(name, size=args.size)
        framework.run_module_comparison(name, test_data)
    
    # Generate reports
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output.mkdir(parents=True, exist_ok=True)
    
    if args.report in ['console', 'all']:
        ConsoleReporter().generate_report(framework.results)
    
    if args.report in ['json', 'all']:
        json_path = args.output / f'comparison_report_{timestamp}.json'
        JSONReporter().generate_report(framework.results, json_path)
        print(f"\nJSON report: {json_path}")
    
    if args.report in ['html', 'all']:
        html_path = args.output / f'comparison_report_{timestamp}.html'
        HTMLReporter().generate_report(framework.results, html_path)
        print(f"HTML report: {html_path}")
    
    # Summary
    summary = framework.get_summary()
    print(f"\n{'='*60}")
    print(f"Summary: {summary['passed']}/{summary['total_tests']} tests passed")
    print(f"{'='*60}\n")
    
    return 0 if summary['failed'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
