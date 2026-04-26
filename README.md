# Al-Mg-Si Precipitation Simulation Microservice

## Git

We forked kawin from repo below:
```
upstream  git@github.com:materialsgenomefoundation/kawin.git
```

To merge new commits from kawin main repo:
```bash
git fetch upstream
git rebase upstream/main
```

## Docker Commands

### Build
```bash
docker build -t kawin-api:latest .
```

### Run
```bash
docker run -d -p 8000:8000 --name kawin-api kawin-api:latest
```

### Health Check
```bash
curl http://localhost:8000/health
```

### Stop and Remove
```bash
docker stop kawin-api && docker rm kawin-api
```

## API Testing

### Test Simulation Endpoint

Run a precipitation simulation for Al-0.3wt%Mg-1.0wt%Si at 200°C for 1 hour:

```bash
curl -X POST http://localhost:8000/api/v1/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "composition": {"MG": 0.00333, "SI": 0.00961},
    "temperature_profile": {
      "time_points_hours": [0, 1],
      "temperatures_celsius": [200, 200]
    },
    "simulation_time_hours": 1
  }'
```

### Composition Conversion (wt% to mole fraction)

| Element | Atomic Weight | Formula |
|---------|---------------|---------|
| Al | 26.98 | x_Al = (wt%_Al / 26.98) / Σ(wt%_i / AW_i) |
| Mg | 24.31 | x_Mg = (wt%_Mg / 24.31) / Σ(wt%_i / AW_i) |
| Si | 28.09 | x_Si = (wt%_Si / 28.09) / Σ(wt%_i / AW_i) |

Example: Al-0.3wt%Mg-1.0wt%Si → MG: 0.00333, SI: 0.00961

### Response Structure

The API returns:
- `status`: Simulation status
- `summary`: Final time results (CRSS, yield strength per phase)
- `time_series`: Complete time evolution data
  - `time_hours`: Array of time points
  - `phases`: Per-phase data (diameter, aspect_ratio, volume_fraction, crss_mpa, yield_strength_mpa)
  - `total_crss_mpa`: Total CRSS vs time
  - `total_yield_strength_mpa`: Total yield strength vs time

## API Documentation

Once the container is running, access:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Deploying / updating the service

The production compose file lives at `/opt/deepalloy/docker-compose.yml` on the droplet and is tracked in this repo as `docker-compose.yml`. Compose is the source of truth — never start the container manually with `docker run`.

To update to a newer image and apply config changes:

```bash
ssh deepalloy-precipitate
cd /opt/deepalloy
docker compose pull
docker compose up -d
docker ps                 # should show (healthy) within ~20s
```

The compose config sets:
- `mem_limit: 2g`, `cpus: 0.8` — resource caps
- `healthcheck` against `/health` every 30s
- Log rotation: 10 MB × 3 files per container
- `restart: unless-stopped` — survives reboots