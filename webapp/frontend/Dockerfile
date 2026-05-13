# ── SuperDARN Workbench — Frontend ────────────────────────────────────────────
# Multi-stage: Node builder → nginxinc/nginx-unprivileged
#
# nginx-unprivileged (uid 101) listens on port 8080 — no root required.
# This makes the image compatible with Rancher PSS "restricted" namespaces.
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci --prefer-offline

COPY . .
RUN npm run build

# ─────────────────────────────────────────────────────────────────────────────
FROM nginxinc/nginx-unprivileged:1.27-alpine AS production

# nginx template entrypoint substitutes ${BACKEND_URL} at startup
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf.template /etc/nginx/templates/default.conf.template

# Default backend URL — override with docker-compose env or Helm ConfigMap
ENV BACKEND_URL=http://backend:8000

# nginx-unprivileged listens on 8080
EXPOSE 8080

HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost:8080/health || exit 1

CMD ["nginx", "-g", "daemon off;"]
