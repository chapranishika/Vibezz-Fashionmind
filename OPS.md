# Operations

## Endpoints that matter for monitoring

| | means | use |
|---|---|---|
| `GET /health` | process is up | liveness only — don't alert on this |
| `GET /ready` | models loaded (≥ `MIN_MODELS`) **and** DB reachable | **the one to monitor**; Dockerfile `HEALTHCHECK` points here |
| `GET /health/db` | RLS lockdown + event trigger + 3 cron jobs still present | security-posture drift; asserted by `smoke_prod.py` |
| `GET /metrics` | Prometheus text | scrape for the dashboard + alert rules |

## Alerting

**Built-in (nothing to sign up for):** `.github/workflows/uptime.yml` pings
`/ready` every ~10 min; on 3 consecutive failures it opens a labelled `uptime`
issue (GitHub emails the repo owner on new issues) and comments on recovery.
GitHub throttles `schedule`, so treat it as a safety net, not a 1-minute SLA.

**Tighter interval + phone push (recommended for anything real):** point a free
external monitor at the same URL.

- **healthchecks.io** (or Better Stack / UptimeRobot): HTTP check on
  `https://nishika1202-vibezz-fashionmind-api.hf.space/ready`, expect `200`,
  interval 1–5 min, grace 2 min. Notification channel → Slack webhook or the
  mobile app.
- **Prometheus alert rules** (if you run the scrape): 

  ```yaml
  - alert: ApiNotReady
    expr: app_ready == 0
    for: 3m
  - alert: HighErrorRate
    expr: sum(rate(http_requests_total{status=~"5.."}[5m])) > 0.2
    for: 5m
  - alert: SecurityPostureDrift        # scrape /health/db as a probe, or run smoke_prod.py
    expr: probe_success{instance=~".*/health/db"} == 0
    for: 10m
  ```

## Dashboard

Import `deploy/grafana_dashboard.json` (Grafana → Dashboards → New → Import),
pick your Prometheus datasource. Panels: ready, model count, request rate by
path, p50/95/99 latency, 5xx rate, recs/s, tool-calls/s, LLM tokens/s.

Scrape config (Grafana Cloud free tier / self-hosted):

```yaml
scrape_configs:
  - job_name: vibezz-api
    metrics_path: /metrics
    scheme: https
    static_configs:
      - targets: ["nishika1202-vibezz-fashionmind-api.hf.space"]
```

## Deploy

```
set HF_TOKEN=hf_xxx     # write token; delete after, see SECURITY.md
python deploy/hf_space_sync.py deploy    # code -> restart -> wait /ready -> smoke
```

`deploy` fails loudly if `/ready` doesn't come back or the smoke test fails —
a broken deploy can't be mistaken for a slow one.

## Secret rotation

`python scripts/rotate_secret.py <NAME> --from-env` — see SECURITY.md. Never
paste a secret value into a chat or a command argument.
