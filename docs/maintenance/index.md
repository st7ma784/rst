# Maintenance & Deployment

Documentation for system administrators and maintainers.

```{toctree}
:maxdepth: 2

deployment
troubleshooting
monitoring
backup
updating
```

## Quick Reference

### Health Check

```bash
# Quick system validation
./scripts/ecosystem_validation.sh

# Check CUDA availability
nvidia-smi
./CUDArst/test_integration
```

### Common Operations

| Task | Command |
|------|---------|
| Check status | `./scripts/ecosystem_validation.sh` |
| Update RST | `git pull && make -C build` |
| Rebuild CUDA | `./scripts/build_all_cuda_modules.sh` |
| View logs | `journalctl -u rst-service` |
| Test processing | `make_fit --help` |
