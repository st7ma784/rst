{{/*
Expand the chart name.
*/}}
{{- define "superdarn.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "superdarn.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Chart label (name + version).
*/}}
{{- define "superdarn.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels.
*/}}
{{- define "superdarn.labels" -}}
helm.sh/chart: {{ include "superdarn.chart" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Backend selector labels.
*/}}
{{- define "superdarn.backendSelectorLabels" -}}
app.kubernetes.io/name: {{ include "superdarn.name" . }}-backend
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Frontend selector labels.
*/}}
{{- define "superdarn.frontendSelectorLabels" -}}
app.kubernetes.io/name: {{ include "superdarn.name" . }}-frontend
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Backend service name.
*/}}
{{- define "superdarn.backendServiceName" -}}
{{- printf "%s-backend" (include "superdarn.fullname" .) }}
{{- end }}
