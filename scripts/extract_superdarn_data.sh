#!/bin/bash
# SuperDARN Data Extraction Script
# Extracts data from MinIO erasure-coded storage to regular files
#
# Usage: ./extract_superdarn_data.sh [output_dir] [limit]

set -e

OUTPUT_DIR="${1:-/home/user/rst/extracted_data}"
LIMIT="${2:-0}"  # 0 = no limit
MC="/tmp/mc"
ALIAS="local"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_prereqs() {
    log_info "Checking prerequisites..."
    
    # Check mc client
    if ! command -v $MC &> /dev/null; then
        log_info "Downloading MinIO client..."
        curl -Lo $MC https://dl.min.io/client/mc/release/linux-amd64/mc
        chmod +x $MC
    fi
    
    # Check MinIO is running
    if ! docker ps | grep -q superdarn-minio; then
        log_error "MinIO container not running. Start it with:"
        echo "  cd /home/user/rst/scripts && docker compose -f minio-compose.yml up -d"
        exit 1
    fi
    
    # Configure mc alias
    $MC alias set $ALIAS http://localhost:9000 minioadmin minioadmin 2>/dev/null || true
}

# List available buckets
list_buckets() {
    log_info "Available buckets:"
    $MC ls $ALIAS/
}

# Extract convmap data
extract_convmap() {
    local bucket="convmap"
    local out_dir="$OUTPUT_DIR/$bucket"
    
    mkdir -p "$out_dir"
    log_info "Extracting $bucket to $out_dir..."
    
    # Count total objects
    local count=$($MC ls --recursive $ALIAS/$bucket/ 2>/dev/null | wc -l)
    log_info "Found $count objects in $bucket"
    
    if [ "$LIMIT" -gt 0 ]; then
        log_info "Limiting extraction to $LIMIT files"
        $MC cp --recursive $ALIAS/$bucket/ "$out_dir/" 2>&1 | head -$LIMIT
    else
        $MC cp --recursive $ALIAS/$bucket/ "$out_dir/"
    fi
    
    log_info "Extracted to $out_dir"
}

# Extract specific file types
extract_by_type() {
    local file_type="${1:-cnvmap}"
    local out_dir="$OUTPUT_DIR/$file_type"
    
    mkdir -p "$out_dir"
    log_info "Extracting *.$file_type files..."
    
    # Find and extract matching objects
    $MC find $ALIAS/ --name "*.$file_type" --exec "$MC cp {} $out_dir/"
}

# Show progress of rsync (if still running)
check_rsync_progress() {
    if pgrep -x rsync > /dev/null; then
        log_warn "rsync processes still running. Data may be incomplete."
        ps aux | grep rsync | grep -v grep
        
        echo ""
        log_info "Current file counts in minio_data:"
        for i in 1 2 3 4; do
            count=$(ls /home/user/rst/minio_data/data$i/convmap/ 2>/dev/null | wc -l)
            echo "  data$i: $count files"
        done
        return 1
    fi
    return 0
}

# Main
main() {
    log_info "SuperDARN Data Extraction"
    log_info "========================="
    
    check_prereqs
    
    if ! check_rsync_progress; then
        log_warn "Wait for rsync to complete before extracting."
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    list_buckets
    
    echo ""
    log_info "Starting extraction..."
    extract_convmap
    
    log_info "Done! Files extracted to: $OUTPUT_DIR"
    echo ""
    log_info "Summary:"
    find "$OUTPUT_DIR" -type f | wc -l | xargs -I{} echo "  Total files: {}"
    du -sh "$OUTPUT_DIR" | awk '{print "  Total size: " $1}'
}

# Handle args
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [output_dir] [limit]"
        echo ""
        echo "Options:"
        echo "  output_dir  Directory for extracted files (default: /home/user/rst/extracted_data)"
        echo "  limit       Max files to extract (default: 0 = unlimited)"
        echo ""
        echo "Examples:"
        echo "  $0                           # Extract all to default location"
        echo "  $0 /tmp/data 100             # Extract 100 files to /tmp/data"
        exit 0
        ;;
    --status)
        check_rsync_progress
        exit $?
        ;;
    *)
        main
        ;;
esac
