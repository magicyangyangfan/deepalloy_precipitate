"""
Simulation endpoints.

Phase 1: Synchronous simulation endpoint.
"""

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.simulation import SimulationRequest, SimulationResponse
from app.services.simulation_service import SimulationService

router = APIRouter()

# Initialize simulation service
simulation_service = SimulationService()


@router.post("/simulate", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest) -> SimulationResponse:
    """
    Run a precipitation simulation synchronously.

    This endpoint runs the Al-Mg-Si precipitation simulation and returns
    the results directly. For long-running simulations, consider using
    the async endpoint (Phase 4).

    Parameters
    ----------
    request : SimulationRequest
        The simulation parameters including:
        - composition: Mg and Si content in mole fraction
        - temperature_profile: Time points and temperatures
        - simulation_time_hours: Total simulation duration
        - config: Optional configuration overrides

    Returns
    -------
    SimulationResponse
        The simulation results including phase-specific data and totals.
    """
    # Validate simulation time
    if request.simulation_time_hours > settings.MAX_SIMULATION_TIME_HOURS:
        raise HTTPException(
            status_code=400,
            detail=f"Simulation time exceeds maximum allowed ({settings.MAX_SIMULATION_TIME_HOURS} hours)",
        )

    # Validate temperature profile lengths match
    tp = request.temperature_profile
    if len(tp.time_points_hours) != len(tp.temperatures_celsius):
        raise HTTPException(
            status_code=400,
            detail="Temperature profile time points and temperatures must have the same length",
        )

    try:
        result = simulation_service.run_simulation(request)
        return result
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Thermodynamic database not found: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Simulation failed: {str(e)}",
        )
