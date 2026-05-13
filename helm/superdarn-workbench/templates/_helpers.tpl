{{/*
Expand the name of the chart.
*/}}
{{- define "superdarn-workbench.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "superdarn-workbench.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{/*
Chart label
*/}}
{{- define "superdarn-workbench.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "superdarn-workbench.labels" -}}
helm.sh/chart: {{ include "superdarn-workbench.chart" . }}
{{ include "superdarn-workbench.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "superdarn-workbench.selectorLabels" -}}
app.kubernetes.io/name: {{ include "superdarn-workbench.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Effective image registry.
Priority: values.image.registry > global.cattle.systemDefaultRegistry > ""
Rancher injects global.cattle.systemDefaultRegistry when a registry mirror is set
at the cluster or project level.
*/}}
{{- define "superdarn-workbench.registry" -}}
{{- $reg := .Values.image.registry -}}
{{- if not $reg }}
  {{- $reg = dig "cattle" "systemDefaultRegistry" "" .Values.global }}
{{- end }}
{{- if and $reg (not (hasSuffix "/" $reg)) }}
  {{- $reg = printf "%s/" $reg }}
{{- end }}
{{- $reg }}
{{- end }}

{{/*
Backend image reference  (<registry>/<repository>:<tag>)
*/}}
{{- define "superdarn-workbench.backendImage" -}}
{{- printf "%s%s:%s" (include "superdarn-workbench.registry" .) .Values.backend.image.repository .Values.image.tag }}
{{- end }}

{{/*
Frontend image reference
*/}}
{{- define "superdarn-workbench.frontendImage" -}}
{{- printf "%s%s:%s" (include "superdarn-workbench.registry" .) .Values.frontend.image.repository .Values.image.tag }}
{{- end }}

{{/*
Service account name
*/}}
{{- define "superdarn-workbench.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "superdarn-workbench.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Backend service URL used in the nginx ConfigMap.
Uses Kubernetes in-cluster DNS: http://<fullname>-backend:<port>
*/}}
{{- define "superdarn-workbench.backendServiceURL" -}}
{{- printf "http://%s-backend:%d" (include "superdarn-workbench.fullname" .) (.Values.backend.service.port | int) }}
{{- end }}

{{/*
Frontend container port (nginx-unprivileged listens on 8080).
*/}}
{{- define "superdarn-workbench.frontendPort" -}}
{{- .Values.frontend.containerPort | default 8080 | int }}
{{- end }}
