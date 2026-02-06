# Monitoring Guide

Monitor RST deployments for performance and health.

## Quick Status Check

```bash
# Overall health
./scripts/ecosystem_validation.sh

# GPU status
nvidia-smi

# Process status
systemctl status rst-processor
```

---

## System Monitoring

### Resource Usage Script

```bash
#!/bin/bash
# monitor_rst.sh - Monitor RST resource usage

while true; do
    echo "=== $(date) ==="
    
    # CPU and memory
    echo "CPU/Memory:"
    ps aux | grep -E "make_fit|make_grid" | grep -v grep | \
        awk '{printf "  %s: CPU=%s%% MEM=%s%%\n", $11, $3, $4}'
    
    # GPU (if available)
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
            --format=csv,noheader | \
            awk -F',' '{printf "  Util=%s Used=%s/%s\n", $1, $2, $3}'
    fi
    
    # Disk usage
    echo "Disk:"
    df -h /data | tail -1 | awk '{printf "  Used=%s/%s (%s)\n", $3, $2, $5}'
    
    echo ""
    sleep 60
done
```

### Prometheus Metrics

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'rst'
    static_configs:
      - targets: ['localhost:9100']  # node_exporter
      - targets: ['localhost:9400']  # nvidia_exporter
```

Node exporter systemd unit:

```ini
# /etc/systemd/system/node_exporter.service
[Unit]
Description=Node Exporter

[Service]
ExecStart=/usr/local/bin/node_exporter

[Install]
WantedBy=multi-user.target
```

### Custom RST Metrics Exporter

```python
#!/usr/bin/env python3
# rst_exporter.py - Export RST metrics for Prometheus

from prometheus_client import start_http_server, Gauge
import subprocess
import time

# Define metrics
processing_time = Gauge('rst_processing_seconds', 
                        'Time to process file', ['operation'])
files_processed = Gauge('rst_files_processed_total',
                        'Total files processed')
cuda_available = Gauge('rst_cuda_available',
                       'CUDA availability (1=yes, 0=no)')

def check_cuda():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True)
        return 1 if result.returncode == 0 else 0
    except:
        return 0

def collect_metrics():
    cuda_available.set(check_cuda())
    # Add more metric collection here

if __name__ == '__main__':
    start_http_server(9101)
    while True:
        collect_metrics()
        time.sleep(15)
```

---

## Log Management

### Structured Logging

```bash
# Configure log output
export RST_LOG_FILE=/var/log/rst/processing.log
export RST_LOG_LEVEL=INFO  # DEBUG, INFO, WARN, ERROR

# Log format: JSON for easy parsing
# 2024-01-15T10:30:00Z INFO make_fit started file=input.rawacf
```

### Logrotate Configuration

```
# /etc/logrotate.d/rst
/var/log/rst/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0640 rst rst
}
```

### Log Analysis

```bash
# Count processing operations
grep "completed" /var/log/rst/processing.log | wc -l

# Find errors
grep -i error /var/log/rst/processing.log | tail -20

# Processing time statistics
grep "processing_time" /var/log/rst/processing.log | \
    awk '{sum+=$NF; count++} END {print "Avg:", sum/count, "ms"}'
```

---

## Alerting

### Simple Alert Script

```bash
#!/bin/bash
# alert.sh - Send alerts for RST issues

ALERT_EMAIL="admin@example.com"
ALERT_THRESHOLD_DISK=90  # percent
ALERT_THRESHOLD_GPU_TEMP=85  # celsius

# Check disk space
disk_usage=$(df /data | tail -1 | awk '{print $5}' | tr -d '%')
if [ $disk_usage -gt $ALERT_THRESHOLD_DISK ]; then
    echo "ALERT: Disk usage at ${disk_usage}%" | \
        mail -s "RST Disk Alert" $ALERT_EMAIL
fi

# Check GPU temperature
if command -v nvidia-smi &> /dev/null; then
    gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu \
               --format=csv,noheader | head -1)
    if [ $gpu_temp -gt $ALERT_THRESHOLD_GPU_TEMP ]; then
        echo "ALERT: GPU temperature at ${gpu_temp}C" | \
            mail -s "RST GPU Alert" $ALERT_EMAIL
    fi
fi

# Check if service is running
if ! systemctl is-active --quiet rst-processor; then
    echo "ALERT: RST processor service is down" | \
        mail -s "RST Service Alert" $ALERT_EMAIL
fi
```

Add to crontab:

```
*/5 * * * * /opt/rst/scripts/alert.sh
```

### Alertmanager Rules (Prometheus)

```yaml
# alertmanager_rules.yml
groups:
  - name: rst
    rules:
      - alert: RSTProcessorDown
        expr: up{job="rst"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "RST processor is down"

      - alert: HighGPUTemperature
        expr: nvidia_gpu_temperature_celsius > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU temperature is high: {{ $value }}°C"

      - alert: LowDiskSpace
        expr: node_filesystem_avail_bytes{mountpoint="/data"} / 
              node_filesystem_size_bytes{mountpoint="/data"} < 0.1
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Data disk space low"
```

---

## Dashboard

### Grafana Dashboard JSON

```json
{
  "dashboard": {
    "title": "RST Processing Monitor",
    "panels": [
      {
        "title": "Processing Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(rst_files_processed_total[5m])",
            "legendFormat": "Files/sec"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "gauge",
        "targets": [
          {
            "expr": "nvidia_gpu_utilization_gpu",
            "legendFormat": "GPU %"
          }
        ]
      },
      {
        "title": "Processing Time",
        "type": "graph",
        "targets": [
          {
            "expr": "rst_processing_seconds",
            "legendFormat": "{{ operation }}"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes",
            "legendFormat": "Available %"
          }
        ]
      }
    ]
  }
}
```

---

## Performance Tracking

### Benchmark Script

```bash
#!/bin/bash
# benchmark_daily.sh - Track performance over time

OUTPUT_DIR=/var/log/rst/benchmarks
mkdir -p $OUTPUT_DIR

DATE=$(date +%Y%m%d)
OUTPUT_FILE=$OUTPUT_DIR/benchmark_$DATE.json

# Run benchmark
cd /opt/rst
python scripts/compare_performance.py --output $OUTPUT_FILE

# Update history
cat $OUTPUT_FILE >> $OUTPUT_DIR/history.jsonl
```

### Trend Analysis

```python
#!/usr/bin/env python3
# analyze_trends.py

import json
import pandas as pd
import matplotlib.pyplot as plt

# Load benchmark history
data = []
with open('/var/log/rst/benchmarks/history.jsonl') as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])

# Plot trends
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

df.plot(x='date', y='fitacf_time', ax=axes[0,0], title='FITACF Time')
df.plot(x='date', y='grid_time', ax=axes[0,1], title='Grid Time')
df.plot(x='date', y='cuda_speedup', ax=axes[1,0], title='CUDA Speedup')
df.plot(x='date', y='memory_usage', ax=axes[1,1], title='Memory Usage')

plt.tight_layout()
plt.savefig('/var/log/rst/benchmarks/trends.png')
```

---

## Health Checks

### Docker Health Check

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD make_fit --help > /dev/null || exit 1
```

### Kubernetes Probes

```yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: rst
    livenessProbe:
      exec:
        command:
        - make_fit
        - --help
      initialDelaySeconds: 10
      periodSeconds: 30
    readinessProbe:
      exec:
        command:
        - /opt/rst/scripts/ecosystem_validation.sh
      initialDelaySeconds: 5
      periodSeconds: 10
```

### Systemd Watchdog

```ini
# /etc/systemd/system/rst-processor.service
[Service]
WatchdogSec=60
Restart=on-failure
RestartSec=10
```
