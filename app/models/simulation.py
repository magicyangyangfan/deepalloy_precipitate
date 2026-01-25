"""
Pydantic models for simulation request and response schemas.
"""

from typing import Optional
from pydantic import BaseModel, Field


class SimulationRequest(BaseModel):
    """Request schema for running a precipitation simulation."""

    aging_temperature: float = Field(
        ..., ge=100, le=300,
        description="Aging temperature in Celsius",
        examples=[175]
    )
    aging_time: float = Field(
        ..., gt=0, le=100,
        description="Aging time in hours",
        examples=[24]
    )
    mg_content: float = Field(
        default=0.0072, ge=0, le=0.1,
        description="Mg content in mole fraction (0.0072 = 0.72 at.%)",
        examples=[0.0072]
    )
    si_content: float = Field(
        default=0.0057, ge=0, le=0.1,
        description="Si content in mole fraction (0.0057 = 0.57 at.%)",
        examples=[0.0057]
    )
    temperature_shift: float = Field(
        default=15.0, ge=-50, le=50,
        description="Temperature correction in Celsius (added to aging_temperature)",
        examples=[15.0]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "aging_temperature": 175,
                    "aging_time": 24,
                    "mg_content": 0.0072,
                    "si_content": 0.0057,
                    "temperature_shift": 15.0
                }
            ]
        }
    }


class PhaseResult(BaseModel):
    """Results for a single precipitate phase at final time."""

    name: str
    diameter_nm: float
    major_axis_nm: float
    volume_fraction_pct: float


class SimulationSummary(BaseModel):
    """Summary of simulation results at final time."""

    final_time_hours: float
    user_temperature_c: float
    effective_temperature_c: float
    phases: list[PhaseResult]
    matrix_mg_wt_pct: float
    matrix_si_wt_pct: float
    orowan_strength_mpa: float
    solid_solution_strength_mpa: float
    total_yield_strength_mpa: float


class PhaseTimeSeries(BaseModel):
    """Time series data for a single precipitate phase."""

    name: str
    diameter_nm: list[float]
    major_axis_nm: list[float]
    volume_fraction_pct: list[float]


class TimeSeriesData(BaseModel):
    """Complete time series data from the simulation."""

    time_hours: list[float]
    phases: list[PhaseTimeSeries]
    matrix_mg_wt_pct: list[float]
    matrix_si_wt_pct: list[float]
    orowan_strength_mpa: list[float]
    solid_solution_strength_mpa: list[float]
    total_yield_strength_mpa: list[float]


class SimulationResponse(BaseModel):
    """Response schema for a completed simulation."""

    status: str = "completed"
    summary: SimulationSummary
    time_series: TimeSeriesData
