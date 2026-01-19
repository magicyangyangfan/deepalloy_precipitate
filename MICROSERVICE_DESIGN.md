# Al-Mg-Si Precipitation Simulation Microservice Design

## Overview

Convert the `kawin/run_simulation.py` precipitation simulation into a production-ready microservice deployed on DigitalOcean with:
- **FastAPI** REST API
- **Job Queue** for handling concurrent requests
- **Docker** containerization with CI/CD
- **DigitalOcean Spaces** for result storage
- **PostgreSQL** for job metadata

---

## 1. Proposed Project Structure

```
deepalloy_precipitate/
├── app/                          # FastAPI application (NEW)
│   ├── __init__.py
│   ├── main.py                   # FastAPI app entry point
│   ├── config.py                 # Environment configuration
│   ├── models/                   # Pydantic models
│   │   ├── __init__.py
│   │   ├── simulation.py         # Request/Response schemas
│   │   └── job.py                # Job status schemas
│   ├── routers/                  # API routes
│   │   ├── __init__.py
│   │   ├── simulation.py         # /simulate endpoints
│   │   ├── jobs.py               # /jobs endpoints
│   │   └── health.py             # /health endpoint
│   ├── services/                 # Business logic
│   │   ├── __init__.py
│   │   ├── simulation_service.py # Wraps kawin simulation
│   │   ├── job_service.py        # Job queue management
│   │   └── storage_service.py    # DigitalOcean Spaces integration
│   ├── db/                       # Database layer
│   │   ├── __init__.py
│   │   ├── database.py           # PostgreSQL connection
│   │   └── models.py             # SQLAlchemy models
│   └── workers/                  # Background workers
│       ├── __init__.py
│       └── simulation_worker.py  # Celery/ARQ worker
├── kawin/                        # Existing package (unchanged)
│   ├── run_simulation.py         # Gateway - will be wrapped by service
│   └── ...
├── data/                         # Thermodynamic databases
│   └── AlMgSi.tdb
├── tests/                        # API tests (NEW)
│   ├── __init__.py
│   ├── test_api.py
│   └── test_simulation_service.py
├── Dockerfile                    # Container definition (NEW)
├── docker-compose.yml            # Local dev environment (NEW)
├── .dockerignore                 # Docker ignore file (NEW)
├── .github/
│   └── workflows/
│       ├── pytest.yaml           # Existing tests
│       └── deploy.yaml           # DigitalOcean deployment (NEW)
├── pyproject.toml                # Updated with new dependencies
├── requirements.txt              # Production dependencies
└── README.md
```

---

## 2. REST API Design

### Base URL
```
https://your-app.ondigitalocean.app/api/v1
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/simulate` | Submit simulation (sync for quick, returns job_id for long) |
| `POST` | `/simulate/async` | Submit simulation job (always async) |
| `GET` | `/jobs/{job_id}` | Get job status and results |
| `GET` | `/jobs/{job_id}/results` | Download result files (plots, data) |
| `DELETE` | `/jobs/{job_id}` | Cancel/delete a job |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Auto-generated OpenAPI docs |

### Request Schema: `POST /simulate`

```json
{
  "composition": {
    "MG": 0.0072,
    "SI": 0.0057
  },
  "temperature_profile": {
    "time_points_hours": [0, 16, 17],
    "temperatures_celsius": [199, 199, 200]
  },
  "simulation_time_hours": 25,
  "config": {
    "aspect_ratio": {
      "prefactor": 5.55,
      "exponent": 0.24
    },
    "dislocation": {
      "shear_modulus_gpa": 25.4,
      "burgers_vector_nm": 0.286,
      "poisson_ratio": 0.34
    },
    "interfacial_energy": {
      "MGSI_B_P": 0.18,
      "MG5SI6_B_DP": 0.084
    },
    "taylor_factor": 2.24
  }
}
```

### Response Schema: Job Created

```json
{
  "job_id": "uuid-string",
  "status": "pending",
  "created_at": "2026-01-17T10:30:00Z",
  "estimated_duration_seconds": 120
}
```

### Response Schema: Job Complete

```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "created_at": "2026-01-17T10:30:00Z",
  "completed_at": "2026-01-17T10:32:00Z",
  "results": {
    "summary": {
      "final_time_hours": 25.0,
      "phases": [
        {
          "name": "MGSI_B_P",
          "diameter_nm": 11.09,
          "aspect_ratio": 9.89,
          "crss_mpa": 79.3,
          "yield_strength_mpa": 177.6
        }
      ],
      "total_crss_mpa": 79.3,
      "total_yield_strength_mpa": 177.6
    },
    "files": {
      "precipitation_plot": "https://spaces.digitalocean.com/.../precipitation.png",
      "yield_strength_plot": "https://spaces.digitalocean.com/.../yield_strength.png",
      "raw_data": "https://spaces.digitalocean.com/.../results.json"
    }
  }
}
```

---

## 3. Architecture Components

### 3.1 FastAPI Application (`app/main.py`)

```python
from fastapi import FastAPI
from app.routers import simulation, jobs, health

app = FastAPI(
    title="Al-Mg-Si Precipitation Simulation API",
    version="1.0.0",
    description="Microservice for precipitation simulation using kawin"
)

app.include_router(health.router, tags=["Health"])
app.include_router(simulation.router, prefix="/api/v1", tags=["Simulation"])
app.include_router(jobs.router, prefix="/api/v1", tags=["Jobs"])
```

### 3.2 Job Queue Strategy

**Option A: ARQ (Recommended for simplicity)**
- Lightweight async job queue using Redis
- Native async/await support
- Perfect for FastAPI

**Option B: Celery**
- More mature, feature-rich
- Better for complex workflows
- Heavier setup

**Recommendation**: Start with **ARQ** for simplicity, migrate to Celery if needed.

### 3.3 Database Schema (PostgreSQL)

```sql
CREATE TABLE simulation_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    input_params JSONB NOT NULL,
    results JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    result_urls JSONB
);

CREATE INDEX idx_jobs_status ON simulation_jobs(status);
CREATE INDEX idx_jobs_created ON simulation_jobs(created_at DESC);
```

### 3.4 Storage Service (DigitalOcean Spaces)

```python
# app/services/storage_service.py
import boto3
from app.config import settings

class StorageService:
    def __init__(self):
        self.client = boto3.client(
            's3',
            endpoint_url=settings.DO_SPACES_ENDPOINT,
            aws_access_key_id=settings.DO_SPACES_KEY,
            aws_secret_access_key=settings.DO_SPACES_SECRET
        )

    def upload_result(self, job_id: str, filename: str, data: bytes) -> str:
        key = f"simulations/{job_id}/{filename}"
        self.client.put_object(
            Bucket=settings.DO_SPACES_BUCKET,
            Key=key,
            Body=data,
            ACL='public-read'
        )
        return f"{settings.DO_SPACES_CDN_URL}/{key}"
```

---

## 4. Docker Configuration

### Why Dockerfile Over Buildpacks?

DigitalOcean App Platform supports two deployment methods:
1. **Cloud Native Buildpacks** - Auto-detects Python from `requirements.txt` and builds automatically
2. **Dockerfile** - You provide explicit build instructions

**Decision: Use Dockerfile**

| Factor | Buildpacks | Dockerfile (Chosen) |
|--------|------------|---------------------|
| Setup complexity | Simpler, zero-config | Requires writing Dockerfile |
| System dependencies | Limited (Aptfile) | Full control (`apt-get install`) |
| Scientific packages | May fail on numpy/scipy/pycalphad | Guaranteed to work |
| Debugging | Black box | Transparent, reproducible |
| Portability | App Platform only | Works on Droplet, any Docker host |

**Reasoning:**

1. **Scientific Python dependencies** - The kawin package relies on numpy, scipy, and pycalphad which require compiled C/Fortran extensions. These need system libraries (`gfortran`, `libopenblas-dev`) that buildpacks may not provide reliably.

2. **Reproducibility** - A Dockerfile guarantees the same environment locally, on a Droplet, and on App Platform. Buildpacks can change behavior between versions.

3. **Phase 1 compatibility** - Our implementation plan deploys to a basic Droplet first (before App Platform). Having a Dockerfile means we use the same containerization approach throughout all phases.

4. **Debugging** - When scientific package installations fail, having explicit control over the build process makes troubleshooting straightforward.

---

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for scientific packages
RUN apt-get update && apt-get install -y \
    gcc \
    gfortran \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Install kawin package
RUN pip install -e .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml (Local Development)

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/kawin
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./data:/app/data

  worker:
    build: .
    command: arq app.workers.simulation_worker.WorkerSettings
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/kawin
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=kawin
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

volumes:
  postgres_data:
```

---

## 5. CI/CD Pipeline (GitHub Actions → GitHub Container Registry → Droplet)

### Why This Approach?

| Approach | Description | Trade-off |
|----------|-------------|-----------|
| Build on Droplet | Clone repo, build image on server | Slow, needs large Droplet, downtime during build |
| **Build in CI → Registry → Pull** | Build in GitHub Actions, store in registry, Droplet pulls | Fast deploys, smaller Droplet, no build downtime |
| DigitalOcean App Platform | Fully managed PaaS | Higher cost, less control |

**Decision: Build in CI, push to GitHub Container Registry (ghcr.io), pull on Droplet**

### Workflow Overview

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   GitHub    │────▶│   GitHub    │────▶│   ghcr.io   │────▶│   Droplet   │
│   Push      │     │   Actions   │     │  (registry) │     │   (pull)    │
└─────────────┘     │  (build)    │     └─────────────┘     └─────────────┘
                    └─────────────┘
```

### Workflow Files

**`.github/workflows/build-and-push.yaml`** - Builds and pushes Docker image on every push to main.

**`.github/workflows/deploy.yaml`** - Deploys to Droplet (auto-triggered after build, or manual).

### GitHub Repository Secrets Required

Configure these in: **Settings → Secrets and variables → Actions**

| Secret | Description | Example |
|--------|-------------|---------|
| `DROPLET_HOST` | Droplet IP address | `164.90.xxx.xxx` |
| `DROPLET_USER` | SSH username | `root` or `deploy` |
| `DROPLET_SSH_KEY` | Private SSH key for Droplet | Contents of `~/.ssh/id_rsa` |

> **Note:** `GITHUB_TOKEN` is automatically provided by GitHub Actions for ghcr.io authentication.

### Droplet Initial Setup

Run these commands once on a fresh Droplet:

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh

# (Optional) Create deploy user
useradd -m -s /bin/bash deploy
usermod -aG docker deploy

# (Optional) Set up SSH key for deploy user
mkdir -p /home/deploy/.ssh
echo "your-public-key" >> /home/deploy/.ssh/authorized_keys
chmod 700 /home/deploy/.ssh
chmod 600 /home/deploy/.ssh/authorized_keys
chown -R deploy:deploy /home/deploy/.ssh

# Login to GitHub Container Registry (first time)
echo "YOUR_GITHUB_PAT" | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

### Manual Deployment (without CI/CD)

```bash
# SSH to Droplet
ssh root@your-droplet-ip

# Pull and run
docker pull ghcr.io/your-username/deepalloy_precipitate:latest
docker stop kawin-api 2>/dev/null || true
docker rm kawin-api 2>/dev/null || true
docker run -d --name kawin-api --restart unless-stopped -p 8000:8000 \
  ghcr.io/your-username/deepalloy_precipitate:latest

# Verify
curl http://localhost:8000/health
```

---

## 6. Environment Configuration

### `app/config.py`

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str

    # Redis
    REDIS_URL: str

    # DigitalOcean Spaces
    DO_SPACES_ENDPOINT: str
    DO_SPACES_KEY: str
    DO_SPACES_SECRET: str
    DO_SPACES_BUCKET: str
    DO_SPACES_CDN_URL: str

    # App
    DEBUG: bool = False
    MAX_SIMULATION_TIME_HOURS: int = 100

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 7. Implementation Plan

### Phase 1: Core API & Deployment (Droplet)
1. Create `app/` directory structure
2. Implement FastAPI application with health endpoint
3. Create Pydantic models for request/response
4. Wrap `run_simulation.py` in a service class (sync mode, in-memory results)
5. Create Dockerfile
6. Deploy to DigitalOcean Droplet (basic deployment, no external dependencies)
7. Verify API is accessible and simulation runs end-to-end

### Phase 2: Integration with Other Apps
8. Configure CORS for frontend/client applications
9. Add API authentication (API keys or JWT)
10. Set up reverse proxy (nginx) if needed
11. Connect to external services (e.g., frontend app, monitoring)

### Phase 3: Database & Storage
12. Set up PostgreSQL database
13. Create SQLAlchemy models and job table
14. Create job CRUD operations
15. Implement DigitalOcean Spaces integration
16. Add result upload/download functionality
17. Migrate from in-memory to persistent job tracking

### Phase 4: Job Queue & Background Processing
18. Set up Redis
19. Install and configure ARQ
20. Create simulation worker for background processing
21. Implement async job submission endpoint
22. Add job status polling endpoint
23. Update docker-compose.yml with worker service

### Phase 5: CI/CD & Production Hardening
24. Set up GitHub Actions workflow
25. Configure DigitalOcean App Platform (optional migration from Droplet)
26. Add rate limiting
27. Write API tests
28. Add OpenAPI documentation
29. Create deployment README

---

## 8. Files to Create/Modify

### New Files:
- `app/__init__.py`
- `app/main.py`
- `app/config.py`
- `app/models/__init__.py`
- `app/models/simulation.py`
- `app/models/job.py`
- `app/routers/__init__.py`
- `app/routers/simulation.py`
- `app/routers/jobs.py`
- `app/routers/health.py`
- `app/services/__init__.py`
- `app/services/simulation_service.py`
- `app/services/job_service.py`
- `app/services/storage_service.py`
- `app/db/__init__.py`
- `app/db/database.py`
- `app/db/models.py`
- `app/workers/__init__.py`
- `app/workers/simulation_worker.py`
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `.do/app.yaml`
- `.github/workflows/deploy.yaml`
- `tests/test_api.py`

### Modified Files:
- `pyproject.toml` - Add FastAPI, SQLAlchemy, boto3, arq dependencies
- `requirements.txt` - Production dependencies list
- `.gitignore` - Add .env, __pycache__, etc.

---

## 9. Verification Plan

1. **Local Testing**:
   ```bash
   docker-compose up
   curl http://localhost:8000/health
   curl -X POST http://localhost:8000/api/v1/simulate -d '...'
   ```

2. **API Tests**:
   ```bash
   pytest tests/test_api.py
   ```

3. **Deployment Verification**:
   - Push to main branch
   - Verify GitHub Actions passes
   - Check DigitalOcean App Platform deployment
   - Test production endpoints

---

## 10. Security Considerations

- Environment variables for all secrets
- Input validation via Pydantic
- Rate limiting on API endpoints
- CORS configuration for your frontend app
- Signed URLs for Spaces (optional)
