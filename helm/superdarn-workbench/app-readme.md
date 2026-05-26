# SuperDARN Interactive Workbench

REST API + React/MUI web UI over the optimized RST C library stack
(libgrdopt + libfitacf.3.0).

## What this deploys

| Component | Description |
|-----------|-------------|
| **Backend** | FastAPI service that subprocesses to RST's `make_fit` / `make_grid` / `map_grd` for FITACF, grid, and convection-map processing |
| **Frontend** | React + MUI web app served by nginx-unprivileged (WebSocket live progress, RTI heatmap, range profiles, convection maps) |
| **Storage** | PersistentVolumeClaim for uploaded files, results, and the SQLite job database |

## Backend

v2.0 of the chart deploys a single C-backed backend. The pythonv2 (CuPy)
and CUDArst (CUDA kernels) alternatives were retired once `libgrdopt`'s
Phase B + C closeout brought the reference RST stack to the speed
ceiling we'd been chasing with the alternatives (3.5x sort, 27x locate,
all per-op equivalence). `map_grd` is now linked against
`-lgrdopt.1 -lgrd.1` so the optimized library wins symbol resolution.

## Storage

Use **Longhorn** (Rancher's default distributed storage) or
**local-path** for single-node clusters. At least **10 Gi** recommended
for production use with multi-beam scan files.

## Ingress

- **nginx**: set `ingress.className=nginx` (ingress-nginx controller)
- **Traefik**: set `ingress.className=traefik` (Rancher's built-in Traefik v2)

WebSocket connections (`/ws/`) are automatically proxied.

## Image registry

The chart pulls `siw-backend` and `siw-frontend` images. Set
`image.registry=ghcr.io/your-org/` (with trailing slash) and
`image.tag=<version>` to point at your registry. If your Rancher
cluster has a cluster-level registry mirror configured,
`global.cattle.systemDefaultRegistry` is honored automatically.

## After install

Access the UI via the **Ingress hostname** or port-forward:

```bash
kubectl port-forward -n <namespace> svc/<release>-frontend 8080:8080
```

Then open **http://localhost:8080**.

## Troubleshooting

**Frontend pod stuck `NotReady` with `Progress deadline exceeded`:**

Most often the `siw-frontend:<tag>` image isn't reachable from the
cluster (no registry, wrong tag, or missing pull secret). Check:

```bash
kubectl describe pod -n <namespace> -l app.kubernetes.io/component=frontend
kubectl logs    -n <namespace> -l app.kubernetes.io/component=frontend
```

Look for `ImagePullBackOff` / `ErrImagePull` in the events. If the
image is present but nginx is emerg-aborting, the logs will show the
exact reason — common ones:

- `host not found in upstream "backend"` — `BACKEND_URL` env not
  reaching the container (chart v2.0+ sets this automatically; upgrade
  the release).
- `permission denied ... /etc/nginx/conf.d/default.conf` — emptyDir
  permissions; ensure `frontend.podSecurityContext.fsGroup: 101` is
  set in your values overrides.
