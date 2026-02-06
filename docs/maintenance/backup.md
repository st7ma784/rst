# Backup & Recovery

Backup strategies and disaster recovery for RST deployments.

## Backup Strategy

### What to Backup

| Component | Priority | Frequency |
|-----------|----------|-----------|
| Processed data | High | Daily |
| Configuration | High | On change |
| Input data | Medium | Weekly |
| RST installation | Low | On update |

### Backup Schedule

```
Daily:      Processed FITACF, Grid, Map files
Weekly:     Full data archive
Monthly:    Complete system backup
On-change:  Configuration files
```

---

## Data Backup

### Rsync Backup Script

```bash
#!/bin/bash
# backup_data.sh - Backup RST data

BACKUP_SERVER="backup.example.com"
BACKUP_PATH="/backup/rst"
DATA_DIR="/data"
LOG_FILE="/var/log/rst/backup.log"

# Daily backup of processed data
rsync -avz --delete \
    $DATA_DIR/output/ \
    $BACKUP_SERVER:$BACKUP_PATH/output/ \
    >> $LOG_FILE 2>&1

# Weekly full backup
if [ $(date +%u) -eq 7 ]; then
    rsync -avz --delete \
        $DATA_DIR/ \
        $BACKUP_SERVER:$BACKUP_PATH/full/ \
        >> $LOG_FILE 2>&1
fi

# Verify backup
ssh $BACKUP_SERVER "ls -la $BACKUP_PATH/output | wc -l" >> $LOG_FILE
```

### Cron Configuration

```
# /etc/cron.d/rst-backup

# Daily backup at 3 AM
0 3 * * * root /opt/rst/scripts/backup_data.sh

# Monthly archive
0 4 1 * * root /opt/rst/scripts/backup_archive.sh
```

### Cloud Backup (AWS S3)

```bash
#!/bin/bash
# backup_s3.sh - Backup to S3

BUCKET="s3://superdarn-backup"
DATA_DIR="/data"
DATE=$(date +%Y%m%d)

# Install AWS CLI if needed
# pip install awscli

# Sync processed data
aws s3 sync $DATA_DIR/output/ $BUCKET/output/

# Create dated archive
tar -czf - $DATA_DIR/output | \
    aws s3 cp - $BUCKET/archives/output_$DATE.tar.gz
```

---

## Configuration Backup

### Backup Script

```bash
#!/bin/bash
# backup_config.sh

BACKUP_DIR="/backup/rst/config"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# RST configuration
tar -czf $BACKUP_DIR/rst_config_$DATE.tar.gz \
    /opt/rst/.profile.bash \
    /opt/rst/config/ \
    /etc/systemd/system/rst-*.service \
    /etc/profile.d/rst.sh

# Docker configuration
if [ -f /opt/rst/docker-compose.yml ]; then
    cp /opt/rst/docker-compose.yml \
       $BACKUP_DIR/docker-compose_$DATE.yml
fi

# Keep last 30 backups
ls -t $BACKUP_DIR/rst_config_*.tar.gz | \
    tail -n +31 | xargs rm -f 2>/dev/null
```

### Git-based Config Management

```bash
# Initialize config repo
cd /opt/rst/config
git init
git add .
git commit -m "Initial configuration"

# Push to remote
git remote add origin git@github.com:org/rst-config.git
git push -u origin main

# On configuration change
git add -A
git commit -m "Updated configuration: $(date)"
git push
```

---

## Database Backup (if applicable)

### PostgreSQL

```bash
#!/bin/bash
# backup_db.sh

DB_NAME="rst_metadata"
BACKUP_DIR="/backup/rst/database"
DATE=$(date +%Y%m%d)

pg_dump $DB_NAME | gzip > $BACKUP_DIR/$DB_NAME_$DATE.sql.gz

# Keep 30 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete
```

---

## Disaster Recovery

### Recovery Plan

1. **Assess damage**
   - Identify what was lost
   - Check backup availability

2. **Restore infrastructure**
   - Reinstall OS if needed
   - Restore from backup

3. **Verify functionality**
   - Run validation tests
   - Compare with known-good results

4. **Resume operations**
   - Reprocess failed jobs
   - Monitor for issues

### System Recovery Script

```bash
#!/bin/bash
# restore_system.sh - Full system recovery

BACKUP_SERVER="backup.example.com"
BACKUP_PATH="/backup/rst"

echo "=== RST System Recovery ==="

# 1. Restore RST installation
echo "Restoring RST installation..."
rsync -avz $BACKUP_SERVER:$BACKUP_PATH/install/ /opt/rst/

# 2. Restore configuration
echo "Restoring configuration..."
tar -xzf /backup/latest/rst_config.tar.gz -C /

# 3. Restore data
echo "Restoring data..."
rsync -avz $BACKUP_SERVER:$BACKUP_PATH/output/ /data/output/

# 4. Rebuild if needed
echo "Rebuilding RST..."
cd /opt/rst
source .profile.bash
cd build && make

# 5. Restart services
echo "Restarting services..."
systemctl daemon-reload
systemctl restart rst-processor

# 6. Verify
echo "Verifying installation..."
./scripts/ecosystem_validation.sh

echo "=== Recovery Complete ==="
```

### Data Recovery

```bash
#!/bin/bash
# restore_data.sh - Restore data from backup

BACKUP_SOURCE=$1
RESTORE_TARGET=${2:-/data}

if [ -z "$BACKUP_SOURCE" ]; then
    echo "Usage: restore_data.sh <backup_source> [target_dir]"
    exit 1
fi

echo "Restoring from: $BACKUP_SOURCE"
echo "Restoring to: $RESTORE_TARGET"

# Check backup exists
if ! [ -d "$BACKUP_SOURCE" ] && ! ssh backup.example.com "test -d $BACKUP_SOURCE"; then
    echo "Backup not found!"
    exit 1
fi

# Restore
rsync -avz --progress "$BACKUP_SOURCE/" "$RESTORE_TARGET/"

# Verify
echo "Restored $(find $RESTORE_TARGET -type f | wc -l) files"
```

---

## Backup Verification

### Verification Script

```bash
#!/bin/bash
# verify_backup.sh - Verify backup integrity

BACKUP_DIR="/backup/rst"
LOG_FILE="/var/log/rst/backup_verify.log"

echo "=== Backup Verification $(date) ===" >> $LOG_FILE

# Check file counts
original_count=$(find /data/output -type f | wc -l)
backup_count=$(find $BACKUP_DIR/output -type f | wc -l)

if [ "$original_count" -eq "$backup_count" ]; then
    echo "File count OK: $original_count" >> $LOG_FILE
else
    echo "WARNING: File count mismatch! Original=$original_count Backup=$backup_count" >> $LOG_FILE
    exit 1
fi

# Check random sample
for i in 1 2 3 4 5; do
    file=$(find /data/output -type f | shuf -n1)
    backup_file="$BACKUP_DIR/output/${file#/data/output/}"
    
    if cmp -s "$file" "$backup_file"; then
        echo "Sample $i: OK" >> $LOG_FILE
    else
        echo "Sample $i: MISMATCH - $file" >> $LOG_FILE
        exit 1
    fi
done

echo "Verification PASSED" >> $LOG_FILE
```

### Test Restore

```bash
#!/bin/bash
# test_restore.sh - Test backup can be restored

TEST_DIR="/tmp/rst_restore_test"
rm -rf $TEST_DIR
mkdir -p $TEST_DIR

# Restore to test location
rsync -avz /backup/rst/output/ $TEST_DIR/

# Verify processing works
source /opt/rst/.profile.bash
sample=$(find $TEST_DIR -name "*.fitacf" | head -1)

if [ -n "$sample" ]; then
    make_grid "$sample" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "Restore test PASSED"
    else
        echo "Restore test FAILED - processing error"
        exit 1
    fi
else
    echo "Restore test FAILED - no files found"
    exit 1
fi

rm -rf $TEST_DIR
```

---

## Retention Policy

### Data Retention

| Data Type | Retention | Storage |
|-----------|-----------|---------|
| Raw input | 1 year | Archive |
| FITACF | 5 years | Active |
| Grid | 5 years | Active |
| Maps | 10 years | Archive |
| Logs | 90 days | Active |

### Cleanup Script

```bash
#!/bin/bash
# cleanup.sh - Apply retention policy

# Remove old logs
find /var/log/rst -name "*.log" -mtime +90 -delete

# Archive old data
find /data/output -name "*.fitacf" -mtime +365 -exec gzip {} \;

# Move to cold storage
find /data/output -name "*.gz" -mtime +365 \
    -exec mv {} /archive/rst/ \;
```
