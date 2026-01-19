"""
Al-Mg-Si Precipitation Simulation API

FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import health, simulation

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="""
Al-Mg-Si Precipitation Simulation Microservice

This API provides access to precipitation simulations for the Al-Mg-Si alloy system,
modeling multiple metastable phases:
- MGSI_B_P (β')
- MG5SI6_B_DP (β")
- B_PRIME_L (B')
- U1_PHASE
- U2_PHASE

The simulation calculates:
- Precipitate size evolution (diameter, aspect ratio)
- Volume fractions
- Orowan strengthening (CRSS)
- Yield strength predictions
""",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware (configure for your frontend in Phase 2)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(simulation.router, prefix="/api/v1", tags=["Simulation"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "api": "/api/v1",
    }
