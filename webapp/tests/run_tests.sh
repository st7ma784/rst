#!/usr/bin/env bash
# Run the backend comparison test suite.
#
# Usage:
#   ./run_tests.sh                          # against default localhost ports
#   ./run_tests.sh http://host:8001         # single backend validation
#   ./run_tests.sh http://a:8001,http://b:8002   # pairwise comparison
#   INCLUDE_SLOW=1 ./run_tests.sh           # include @pytest.mark.slow tests

set -euo pipefail
cd "$(dirname "$0")"

URLS="${1:-${BACKEND_URLS:-http://localhost:8001,http://localhost:8002,http://localhost:8003}}"
SLOW_FLAG=""
[[ "${INCLUDE_SLOW:-0}" == "1" ]] && SLOW_FLAG="--run-slow -m slow"

pytest test_backends.py \
    --backend-urls "$URLS" \
    --test-data-dir "../../test_data/rawacf_samples" \
    --timeout "${TEST_TIMEOUT:-120}" \
    -v \
    $SLOW_FLAG \
    "$@"
