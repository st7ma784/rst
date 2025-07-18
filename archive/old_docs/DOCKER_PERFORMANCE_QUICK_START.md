# SuperDARN RST Docker Performance Testing - Quick Start Guide
# ==========================================================

## Overview

This guide helps you quickly set up and run the SuperDARN RST performance testing workflow using Docker containers and GitHub Actions automation.

## Quick Start Checklist

### ✅ Prerequisites
- [ ] Docker and Docker Compose installed
- [ ] GitHub repository with Actions enabled
- [ ] Test data repository or generation capability
- [ ] GitHub Pages enabled (for dashboard hosting)

### ✅ Initial Setup
- [ ] Copy Dockerfile.optimized to your repository
- [ ] Add GitHub Actions workflow (.github/workflows/performance-testing.yml)
- [ ] Generate or prepare test data
- [ ] Configure GitHub secrets (if using external test data)
- [ ] Test local Docker builds

### ✅ First Run
- [ ] Push changes to trigger workflow
- [ ] Monitor GitHub Actions execution
- [ ] Check generated performance dashboard
- [ ] Verify PR comment integration

## Local Testing Commands

### Build and Test Containers Locally
```bash
# Build both standard and optimized containers
docker build -f dockerfile.optimized --target rst_standard -t rst:standard .
docker build -f dockerfile.optimized --target rst_optimized -t rst:optimized .

# Generate test data
python scripts/generate_test_data.py --output test-data/

# Run quick performance test
docker run --rm \
  -v $(pwd)/test-data:/data:ro \
  -v $(pwd)/results:/results \
  rst:standard /app/run_performance_tests.sh

docker run --rm \
  -v $(pwd)/test-data:/data:ro \
  -v $(pwd)/results:/results \
  rst:optimized /app/run_performance_tests.sh

# Generate dashboard
python scripts/generate_github_dashboard.py \
  --results-dir results/ \
  --output-dir dashboard/
```

### Using Docker Compose
```bash
# Run complete performance comparison
docker-compose -f docker-compose.optimized.yml up superdarn-performance

# Run development environment
docker-compose -f docker-compose.optimized.yml up superdarn-dev

# Run benchmark tests
docker-compose -f docker-compose.optimized.yml up superdarn-benchmark
```

## GitHub Actions Configuration

### Required Secrets (Optional)
```
TEST_DATA_SMALL_URL    # URL to small dataset archive
TEST_DATA_MEDIUM_URL   # URL to medium dataset archive
TEST_DATA_LARGE_URL    # URL to large dataset archive
```

### Workflow Triggers
```yaml
# Automatic triggers
push:                  # Standard tests on main/develop
pull_request:          # Quick tests on PRs
schedule:              # Nightly comprehensive tests

# Manual trigger
workflow_dispatch:     # Custom test selection
```

### Test Suite Options
- **Quick**: Small dataset, 5-10 minutes
- **Standard**: Small + medium datasets, 15-20 minutes  
- **Comprehensive**: All datasets, 45-60 minutes
- **Benchmark**: Full intensive testing, 1-2 hours

## Dashboard Features

### Interactive Performance Plots
- Processing time comparison
- Memory usage analysis
- CPU load monitoring
- File processing rates
- Performance improvement metrics
- Resource efficiency analysis

### Automated Reporting
- PR comments with performance summaries
- Regression detection and alerts
- Historical performance tracking
- GitHub Pages dashboard deployment

### Performance Metrics
- **Time Improvement**: Percentage speedup
- **Memory Efficiency**: Peak memory usage
- **Processing Rate**: Files processed per second
- **Speedup Factor**: Performance multiplier
- **Resource Utilization**: CPU and memory efficiency

## Example Performance Report

```
## 🚀 SuperDARN RST Performance Test Results

**Test Configuration:**
- Datasets tested: small, medium
- Test suite: standard
- Commit: abc12345

**Performance Results:**
| Metric | Standard | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Total Processing Time | 45.2s | 32.1s | **29.0%** |
| Peak Memory Usage | 512MB | 384MB | **25.0%** |
| Processing Rate | 2.1 files/s | 3.0 files/s | **42.9%** |
| Speedup Factor | 1.0x | **1.41x** | - |

✅ **Significant performance improvement detected!**

📊 [View Detailed Dashboard](https://your-org.github.io/rst/performance/latest/)
```

## Troubleshooting Common Issues

### Docker Build Failures
```bash
# Check Docker daemon
docker version

# Clear build cache
docker builder prune

# Build with verbose output
docker build --progress=plain -f dockerfile.optimized .
```

### Test Data Issues
```bash
# Regenerate test data
python scripts/generate_test_data.py --output test-data/

# Check test data structure
tree test-data/

# Validate test data
ls -la test-data/*/
```

### GitHub Actions Failures
```bash
# Check workflow syntax
act --dry-run

# Test workflow locally (if act is installed)
act -j performance-tests

# Check action logs in GitHub interface
```

### Dashboard Generation Issues
```bash
# Test dashboard generation locally
python scripts/generate_github_dashboard.py \
  --results-dir results/ \
  --output-dir dashboard/ \
  --verbose

# Check Python dependencies
pip install plotly pandas numpy scipy jinja2

# Validate results data
find results/ -name "*.csv" -exec head -5 {} \;
```

## Performance Optimization Tips

### Container Optimization
- Use multi-stage builds to reduce image size
- Enable BuildKit for faster builds
- Use Docker layer caching
- Optimize base image selection

### Test Data Management
- Cache test data between runs
- Use appropriate dataset sizes for test types
- Implement data validation
- Version control test datasets

### Workflow Optimization
- Use matrix builds for parallel execution
- Implement artifact caching
- Set appropriate timeouts
- Use dependency caching

### Dashboard Performance
- Limit historical data retention
- Optimize plot complexity
- Use CDN for Plotly.js
- Implement lazy loading for large datasets

## Advanced Configuration

### Custom Test Datasets
```json
{
  "custom_dataset": {
    "name": "custom",
    "description": "Custom test dataset",
    "file_count": 10,
    "target_size_mb": 15,
    "complexity_factor": 1.2,
    "complexity_level": "high"
  }
}
```

### Performance Thresholds
```json
{
  "performance_thresholds": {
    "regression_percent": 5.0,
    "warning_percent": 2.0,
    "max_memory_increase_percent": 10.0,
    "max_time_increase_percent": 5.0
  }
}
```

### Notification Configuration
```yaml
# Slack notifications
- name: Notify Slack
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    channel: '#superdarn-dev'
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## Monitoring and Maintenance

### Regular Tasks
- [ ] Review performance trends weekly
- [ ] Update test datasets monthly
- [ ] Check dashboard functionality
- [ ] Monitor resource usage
- [ ] Clean up old artifacts

### Performance Baseline Updates
- [ ] Update baselines after major optimizations
- [ ] Document significant performance changes
- [ ] Adjust regression thresholds as needed
- [ ] Archive historical performance data

### Security Considerations
- [ ] Secure test data repositories
- [ ] Use GitHub secrets for sensitive data
- [ ] Limit workflow permissions
- [ ] Regular security updates for dependencies

## Getting Help

### Documentation
- [DOCKER_PERFORMANCE_WORKFLOW.md](DOCKER_PERFORMANCE_WORKFLOW.md) - Complete workflow documentation
- [ENHANCED_BUILD_SYSTEM_GUIDE.md](ENHANCED_BUILD_SYSTEM_GUIDE.md) - Build system documentation
- [DOCKER_OPTIMIZATION_GUIDE.md](DOCKER_OPTIMIZATION_GUIDE.md) - Docker optimization guide

### Support Resources
- GitHub Issues: Report bugs and request features
- GitHub Discussions: Community questions and tips
- Action Logs: Detailed execution information
- Dashboard: Performance trends and analysis

### Common Commands Reference
```bash
# Generate test data
python scripts/generate_test_data.py --output test-data/

# Build containers
docker build -f dockerfile.optimized --target rst_optimized -t rst:optimized .

# Run tests
docker-compose -f docker-compose.optimized.yml up superdarn-performance

# Generate dashboard
python scripts/generate_github_dashboard.py -r results/ -o dashboard/

# Deploy locally
docker-compose -f docker-compose.optimized.yml up superdarn-dev
```

---

**Next Steps:**
1. Follow the quick start checklist above
2. Run your first performance test
3. Review the generated dashboard
4. Configure notifications and thresholds
5. Set up regular monitoring schedule

For detailed implementation instructions, see [DOCKER_PERFORMANCE_WORKFLOW.md](DOCKER_PERFORMANCE_WORKFLOW.md).
