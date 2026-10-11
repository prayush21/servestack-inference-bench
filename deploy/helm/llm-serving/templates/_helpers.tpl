{{- define "llm-serving.fullname" -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "llm-serving.labels" -}}
app.kubernetes.io/name: llm-serving
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version }}
llm-serving/backend: {{ .Values.backend }}
{{- end -}}

{{- define "llm-serving.selectorLabels" -}}
app.kubernetes.io/name: llm-serving
app.kubernetes.io/instance: {{ .Release.Name }}
llm-serving/backend: {{ .Values.backend }}
{{- end -}}

{{/* repo:tag@digest — the digest makes the pull immutable; the tag keeps it readable. */}}
{{- define "llm-serving.image" -}}
{{- $img := . -}}
{{- if $img.digest -}}
{{- printf "%s:%s@%s" $img.repository $img.tag $img.digest -}}
{{- else -}}
{{- printf "%s:%s" $img.repository $img.tag -}}
{{- end -}}
{{- end -}}

{{- define "llm-serving.cacheClaim" -}}
{{- .Values.modelCache.claimName -}}
{{- end -}}

{{/* Env shared by every container that touches the model cache. */}}
{{- define "llm-serving.cacheEnv" -}}
- name: HF_HOME
  value: {{ printf "%s/hf" .Values.modelCache.mountPath | quote }}
{{- if .Values.hfToken.secretName }}
- name: HF_TOKEN
  valueFrom:
    secretKeyRef:
      name: {{ .Values.hfToken.secretName }}
      key: {{ .Values.hfToken.secretKey }}
      optional: true
{{- end }}
{{- end -}}
