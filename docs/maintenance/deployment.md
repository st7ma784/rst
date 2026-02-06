# Deployment Guide

Guide for deploying RST in production environments.

## Deployment Options

| Method | Best For | Complexity |
|--------|----------|------------|
| [Docker](#docker-deployment) | Most deployments | Low |
| [Native](#native-deployment) | Maximum performance | Medium |
| [HPC Cluster](#hpc-deployment) | Large-scale processing | High |

---

## Docker Deployment

### Single Server

```bash
# Pull or build image
docker build -t superdarn-rst:latest .

# Create data volume
docker volume create rst-data

# Run container
docker run -d \
    --name rst-processor \
    --restart unless-stopped \
    --gpus all \
    -v rst-data:/data \
    -v /path/to/input:/input:ro \
    -v /path/to/output:/output \
    superdarn-rst:latest

# Check status
docker logs rst-processor
```

### Docker Compose Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  rst:
    image: superdarn-rst:latest
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - /data/input:/input:ro
      - /data/output:/output
      - rst-logs:/var/log/rst
    environment:
      - RST_VERBOSE=0
      - RST_CUDA_DEVICE=0
    healthcheck:
      test: ["CMD", "make_fit", "--help"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  rst-logs:
```

Deploy:

```bash
docker-compose -f docker-compose.prod.yml up -d
```

---

## Native Deployment

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 32 GB |
| Storage | 50 GB | 500 GB SSD |
| GPU | GTX 1060 | RTX 3080+ |

### Installation Steps

```bash
# 1. Install dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential gcc make \
    libnetcdf-dev libhdf5-dev libpng-dev

# 2. Clone repository
git clone https://github.com/SuperDARN/rst.git /opt/rst
cd /opt/rst

# 3. Build
source .profile.bash
cd build && make

# 4. Build CUDA modules (if GPU available)
./scripts/build_all_cuda_modules.sh

# 5. Create system user
sudo useradd -r -s /bin/false rst

# 6. Set permissions
sudo chown -R rst:rst /opt/rst
sudo chmod 755 /opt/rst/build/bin/*
```

### Systemd Service

```ini
# /etc/systemd/system/rst-processor.service
[Unit]
Description=RST Data Processor
After=network.target

[Service]
Type=simple
User=rst
Group=rst
WorkingDirectory=/opt/rst
Environment="RSTPATH=/opt/rst"
Environment="PATH=/opt/rst/build/bin:/usr/local/bin:/usr/bin"
ExecStart=/opt/rst/scripts/process_daemon.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable rst-processor
sudo systemctl start rst-processor
```

### Processing Daemon Script

```bash
#!/bin/bash
# /opt/rst/scripts/process_daemon.sh

INPUT_DIR=/data/input
OUTPUT_DIR=/data/output
PROCESSED_DIR=/data/processed

source /opt/rst/.profile.bash

inotifywait -m -e close_write "$INPUT_DIR" |
while read dir action file; do
    if [[ "$file" == *.rawacf ]]; then
        echo "Processing: $file"
        
        input="$INPUT_DIR/$file"
        base="${file%.rawacf}"
        
        # Process
        make_fit "$input" > "$OUTPUT_DIR/$base.fitacf" 2>/dev/null
        
        if [ $? -eq 0 ]; then
            mv "$input" "$PROCESSED_DIR/"
            echo "Completed: $file"
        else
            echo "Failed: $file"
        fi
    fi
done
```

---

## HPC Deployment

### SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=rst-process
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=rst_%j.log

# Load modules
module load cuda/12.0
module load gcc/11.0

# Set environment
export RSTPATH=/path/to/rst
source $RSTPATH/.profile.bash

# Process data
INPUT_FILE=$1
OUTPUT_DIR=$2

make_fit "$INPUT_FILE" > "$OUTPUT_DIR/$(basename ${INPUT_FILE%.rawacf}).fitacf"
```

Submit job:

```bash
sbatch process_job.sh /data/input.rawacf /data/output/
```

### Array Job (Batch Processing)

```bash
#!/bin/bash
#SBATCH --job-name=rst-batch
#SBATCH --array=1-100
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=1:00:00

export RSTPATH=/path/to/rst
source $RSTPATH/.profile.bash

# Get file for this array task
INPUT_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" file_list.txt)

make_fit "$INPUT_FILE" > "output/$(basename ${INPUT_FILE%.rawacf}).fitacf"
```

---

## Configuration

### Environment Variables

```bash
# /etc/environment or /etc/profile.d/rst.sh

# RST paths
export RSTPATH=/opt/rst
export PATH=$RSTPATH/build/bin:$PATH
export LD_LIBRARY_PATH=$RSTPATH/lib:$LD_LIBRARY_PATH

# CUDA settings
export RST_CUDA_DEVICE=0
export RST_DISABLE_CUDA=0     # Set to 1 to disable

# Performance tuning
export RST_VERBOSE=0
export RST_MEMORY_POOL=1      # Enable memory pooling
export RST_BATCH_SIZE=1000    # Batch processing size
```

### Resource Limits

```bash
# /etc/security/limits.d/rst.conf

# Max open files
rst    soft    nofile    65536
rst    hard    nofile    65536

# Max processes
rst    soft    nproc     4096
rst    hard    nproc     4096

# Max memory (KB)
rst    soft    as        unlimited
rst    hard    as        unlimited
```

---

## Network Deployment

### Nginx Reverse Proxy (for web interface)

```nginx
# /etc/nginx/sites-available/rst

upstream rst_backend {
    server 127.0.0.1:8080;
}

server {
    listen 80;
    server_name rst.example.com;
    
    location / {
        proxy_pass http://rst_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /data {
        alias /data/output;
        autoindex on;
    }
}
```

### Firewall Rules

```bash
# Allow SSH and HTTP
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

---

## Security

### File Permissions

```bash
# RST binaries: executable by owner and group
chmod 750 /opt/rst/build/bin/*

# Configuration: readable by owner only
chmod 600 /opt/rst/config/*

# Data directories
chmod 755 /data/output  # World readable
chmod 700 /data/input   # Owner only
```

### Running as Non-Root

```bash
# Create dedicated user
sudo useradd -r -m -s /bin/bash rst

# Set ownership
sudo chown -R rst:rst /opt/rst

# Run processes as rst user
sudo -u rst make_fit input.rawacf > output.fitacf
```

---

## High Availability

### Load Balancing

```yaml
# docker-compose.ha.yml
version: '3.8'

services:
  rst-1:
    image: superdarn-rst:latest
    volumes:
      - shared-data:/data

  rst-2:
    image: superdarn-rst:latest
    volumes:
      - shared-data:/data

  haproxy:
    image: haproxy:latest
    ports:
      - "80:80"
    volumes:
      - ./haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg

volumes:
  shared-data:
    driver: nfs
    driver_opts:
      share: nfs-server:/data
```

### Failover

```bash
#!/bin/bash
# Health check and failover script

PRIMARY="rst-server-1"
BACKUP="rst-server-2"

check_health() {
    ssh $1 "make_fit --help > /dev/null 2>&1"
}

if ! check_health $PRIMARY; then
    echo "Primary failed, switching to backup"
    # Update DNS or load balancer
    update_dns $BACKUP
fi
```

---

## Deployment Checklist

### Pre-Deployment

- [ ] System requirements verified
- [ ] Dependencies installed
- [ ] Storage provisioned
- [ ] Network configured
- [ ] Firewall rules set

### Deployment

- [ ] RST installed and built
- [ ] CUDA modules built (if using GPU)
- [ ] Environment variables configured
- [ ] Services started
- [ ] Health checks passing

### Post-Deployment

- [ ] Monitoring configured
- [ ] Backups scheduled
- [ ] Documentation updated
- [ ] Team notified
