# SuperDARN Interactive Workbench

GPU-accelerated processing and interactive visualisation for SuperDARN (Super Dual Auroral Radar Network) data.

## What this deploys

| Component | Description |
|-----------|-------------|
| **Backend** | FastAPI service that runs FITACF, grid, and convection-map processing |
| **Frontend** | React + MUI web app served by nginx (WebSocket live progress, RTI heatmap, range profiles, convection maps) |
| **Storage** | PersistentVolumeClaim for uploaded files, results, and the SQLite job database |

## Backends

All three backends implement the same RST v3.0-compatible algorithms:

| Backend | Description |
|---------|-------------|
| `pythonv2` | Python / CuPy — two-pass Bendat-Piersol fitting, sigma width, vectorised lag validation |
| `cuda` | CUDArst — CUDA kernels with CPU fallback |
| `rst` | RST reference — original C library subprocess |

## GPU requirements

Select **cuda** backend and enable **GPU** to schedule on a GPU node.  
Requires the [NVIDIA device plugin](https://github.com/NVIDIA/k8s-device-plugin) to be installed on the cluster.

## Storage

Use **Longhorn** (Rancher's default distributed storage) or **local-path** for single-node clusters.  
At least **10 Gi** recommended for production use with multi-beam scan files.

## Ingress

- **nginx**: set `ingress.className=nginx` (ingress-nginx controller)
- **Traefik**: set `ingress.className=traefik` (Rancher's built-in Traefik v2)

WebSocket connections (`/ws/`) are automatically proxied.

## After install

Access the UI via the **Ingress hostname** or port-forward:

```bash
kubectl port-forward -n <namespace> svc/<release>-superdarn-workbench-frontend 8080:8080
```

Then open **http://localhost:8080**.
