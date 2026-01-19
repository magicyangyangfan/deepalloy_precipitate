"""
Pydantic models for simulation request and response schemas.
"""

from typing import Optional
from pydantic import BaseModel, Field


# =============================================================================
# Request Models
# =============================================================================


class CompositionInput(BaseModel):
    """Alloy composition in mole fraction."""

    MG: float = Field(
        ..., ge=0, le=0.1, description="Magnesium content (mole fraction)", examples=[0.0072]
    )
    SI: float = Field(
        ..., ge=0, le=0.1, description="Silicon content (mole fraction)", examples=[0.0057]
    )


class TemperatureProfileInput(BaseModel):
    """Temperature profile for the simulation."""

    time_points_hours: list[float] = Field(
        ...,
        min_length=1,
        description="Time points in hours",
        examples=[[0, 16, 17]],
    )
    temperatures_celsius: list[float] = Field(
        ...,
        min_length=1,
        description="Temperatures in Celsius at each time point",
        examples=[[199, 199, 200]],
    )


class AspectRatioConfigInput(BaseModel):
    """Configuration for aspect ratio calculation: AR = prefactor * (2r/nm)^exponent"""

    prefactor: float = Field(default=5.55, gt=0, description="Aspect ratio prefactor")
    exponent: float = Field(default=0.24, description="Aspect ratio exponent")


class DislocationConfigInput(BaseModel):
    """Dislocation parameters for Orowan strength calculation."""

    shear_modulus_gpa: float = Field(
        default=25.4, gt=0, description="Shear modulus in GPa"
    )
    burgers_vector_nm: float = Field(
        default=0.286, gt=0, description="Burgers vector in nm"
    )
    poisson_ratio: float = Field(
        default=0.34, ge=0, le=0.5, description="Poisson ratio"
    )


class InterfacialEnergyConfigInput(BaseModel):
    """Interfacial energies for each precipitate phase (J/m^2)."""

    MGSI_B_P: float = Field(default=0.18, gt=0, description="Beta-prime interfacial energy")
    MG5SI6_B_DP: float = Field(default=0.084, gt=0, description="Beta-double-prime interfacial energy")
    B_PRIME_L: float = Field(default=0.18, gt=0, description="B-prime interfacial energy")
    U1_PHASE: float = Field(default=0.18, gt=0, description="U1 phase interfacial energy")
    U2_PHASE: float = Field(default=0.18, gt=0, description="U2 phase interfacial energy")


class SimulationConfigInput(BaseModel):
    """Optional configuration parameters for the simulation."""

    aspect_ratio: Optional[AspectRatioConfigInput] = None
    dislocation: Optional[DislocationConfigInput] = None
    interfacial_energy: Optional[InterfacialEnergyConfigInput] = None
    taylor_factor: float = Field(
        default=2.24, gt=0, description="Taylor factor for CRSS to yield strength conversion"
    )


class SimulationRequest(BaseModel):
    """Request schema for running a precipitation simulation."""

    composition: CompositionInput
    temperature_profile: TemperatureProfileInput
    simulation_time_hours: float = Field(
        ..., gt=0, le=100, description="Total simulation time in hours", examples=[25]
    )
    config: Optional[SimulationConfigInput] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "composition": {"MG": 0.0072, "SI": 0.0057},
                    "temperature_profile": {
                        "time_points_hours": [0, 16, 17],
                        "temperatures_celsius": [199, 199, 200],
                    },
                    "simulation_time_hours": 25,
                    "config": {
                        "aspect_ratio": {"prefactor": 5.55, "exponent": 0.24},
                        "taylor_factor": 2.24,
                    },
                }
            ]
        }
    }


# =============================================================================
# Response Models
# =============================================================================


class PhaseResult(BaseModel):
    """Results for a single precipitate phase."""

    name: str = Field(..., description="Phase name")
    diameter_nm: float = Field(..., description="Average diameter in nm")
    aspect_ratio: float = Field(..., description="Aspect ratio")
    major_axis_nm: float = Field(..., description="Major axis length in nm")
    volume_fraction: float = Field(..., description="Volume fraction")
    crss_mpa: float = Field(..., description="Critical resolved shear stress in MPa")
    yield_strength_mpa: float = Field(..., description="Yield strength contribution in MPa")


class SimulationSummary(BaseModel):
    """Summary of simulation results at final time."""

    final_time_hours: float = Field(..., description="Final simulation time in hours")
    phases: list[PhaseResult] = Field(..., description="Results for each phase")
    total_crss_mpa: float = Field(..., description="Total CRSS in MPa")
    total_yield_strength_mpa: float = Field(..., description="Total yield strength in MPa")


class SimulationResponse(BaseModel):
    """Response schema for a completed simulation."""

    status: str = Field(default="completed", description="Simulation status")
    summary: SimulationSummary = Field(..., description="Summary of results")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "completed",
                    "summary": {
                        "final_time_hours": 25.0,
                        "phases": [
                            {
                                "name": "MGSI_B_P",
                                "diameter_nm": 11.09,
                                "aspect_ratio": 9.89,
                                "major_axis_nm": 109.7,
                                "volume_fraction": 0.0012,
                                "crss_mpa": 79.3,
                                "yield_strength_mpa": 177.6,
                            }
                        ],
                        "total_crss_mpa": 79.3,
                        "total_yield_strength_mpa": 177.6,
                    },
                }
            ]
        }
    }
