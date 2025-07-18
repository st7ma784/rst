# Archive Directory

This directory contains legacy files that were moved during repository cleanup to maintain a clean root structure.

## Contents

### `old_docs/`
Legacy documentation files that have been replaced by the new Sphinx-based documentation:
- Docker guides and optimization documentation
- Build system guides
- Implementation summaries
- Testing infrastructure guides

### `old_scripts/`
Legacy shell scripts and Python test files:
- Docker deployment scripts
- Performance testing scripts
- Validation scripts
- Comprehensive test utilities

### `old_docker/`
Legacy Docker configuration files:
- Multiple docker-compose configurations
- Various Dockerfile variants
- Docker-specific build files

## Migration Notes

**New Documentation Location:** `/docs/` (Sphinx-based, deployed to GitHub Pages)
**New Docker Setup:** Single unified `Dockerfile` in root directory
**New Testing:** Integrated test suite in `codebase/superdarn/src.lib/tk/fitacf_v3.0/tests/`

## Accessing Legacy Content

If you need to reference any of these legacy files:

1. **Documentation**: Check the new Sphinx documentation first at `/docs/`
2. **Docker**: Use the unified `Dockerfile` in the root directory
3. **Testing**: Use the integrated test suite with `./run_tests.sh`

These archived files are kept for reference but are no longer maintained.
