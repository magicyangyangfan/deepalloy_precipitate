"""
Simulation endpoints.
"""

from fastapi import APIRouter, HTTPException
from app.models.simulation import SimulationRequest, SimulationResponse
from app.services.simulation_service import SimulationService

router = APIRouter()
simulation_service = SimulationService()


@router.post("/simulate", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest) -> SimulationResponse:
    """
    Run Al-Mg-Si precipitation simulation.

    Parameters
    ----------
    request : SimulationRequest
        - aging_temperature: Aging temperature in Celsius
        - aging_time: Aging time in hours
        - mg_content: Mg content in mole fraction (default: 0.0072)
        - si_content: Si content in mole fraction (default: 0.0057)
        - temperature_shift: Temperature correction in Celsius (default: 15)

    Returns
    -------
    SimulationResponse
        Simulation results including yield strength and microstructure evolution.
    """
    try:
        return simulation_service.run_simulation(request)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Database not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")
