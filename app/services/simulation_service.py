"""
Service layer that wraps the kawin precipitation simulation.

Phase 1: Synchronous execution with in-memory results.
"""

import numpy as np

from app.config import settings
from app.models.simulation import (
    SimulationRequest,
    SimulationResponse,
    SimulationSummary,
    PhaseResult,
)
from kawin.run_simulation import (
    AlMgSiPrecipitationSimulator,
    SimulatorConfig,
    AspectRatioConfig,
    DislocationConfig,
    InterfacialEnergyConfig,
)
from kawin.precipitation.PrecipitationParameters import TemperatureParameters


class SimulationService:
    """Service for running Al-Mg-Si precipitation simulations."""

    def __init__(self, tdb_file: str = None):
        """
        Initialize the simulation service.

        Parameters
        ----------
        tdb_file : str, optional
            Path to thermodynamic database file.
            Defaults to settings.TDB_FILE.
        """
        self.tdb_file = tdb_file or settings.TDB_FILE

    def _build_config(self, request: SimulationRequest) -> SimulatorConfig:
        """Build SimulatorConfig from request."""
        config = SimulatorConfig()

        if request.config is None:
            return config

        req_config = request.config

        # Aspect ratio
        if req_config.aspect_ratio:
            config.aspect_ratio = AspectRatioConfig(
                prefactor=req_config.aspect_ratio.prefactor,
                exponent=req_config.aspect_ratio.exponent,
            )

        # Dislocation parameters
        if req_config.dislocation:
            config.dislocation = DislocationConfig(
                shear_modulus=req_config.dislocation.shear_modulus_gpa * 1e9,
                burgers_vector=req_config.dislocation.burgers_vector_nm * 1e-9,
                poisson_ratio=req_config.dislocation.poisson_ratio,
            )

        # Interfacial energies
        if req_config.interfacial_energy:
            ie = req_config.interfacial_energy
            config.interfacial_energy = InterfacialEnergyConfig(
                MGSI_B_P=ie.MGSI_B_P,
                MG5SI6_B_DP=ie.MG5SI6_B_DP,
                B_PRIME_L=ie.B_PRIME_L,
                U1_PHASE=ie.U1_PHASE,
                U2_PHASE=ie.U2_PHASE,
            )

        # Taylor factor
        config.taylor_factor = req_config.taylor_factor

        return config

    def _build_temperature_profile(
        self, request: SimulationRequest
    ) -> TemperatureParameters:
        """Build TemperatureParameters from request."""
        tp = request.temperature_profile
        # Convert Celsius to Kelvin
        temps_kelvin = [t + 273.15 for t in tp.temperatures_celsius]
        return TemperatureParameters(tp.time_points_hours, temps_kelvin)

    def run_simulation(self, request: SimulationRequest) -> SimulationResponse:
        """
        Run a precipitation simulation synchronously.

        Parameters
        ----------
        request : SimulationRequest
            The simulation request containing composition, temperature profile,
            and optional configuration.

        Returns
        -------
        SimulationResponse
            The simulation results.
        """
        # Build configuration
        config = self._build_config(request)
        temperature = self._build_temperature_profile(request)

        # Create simulator
        simulator = AlMgSiPrecipitationSimulator(
            tdb_file=self.tdb_file,
            init_composition=[request.composition.MG, request.composition.SI],
            temperature=temperature,
            config=config,
        )

        # Run simulation (convert hours to seconds)
        simulation_time_seconds = request.simulation_time_hours * 3600
        results = simulator.solve(
            simulation_time=simulation_time_seconds,
            verbose=False,
        )

        # Build response
        return self._build_response(results, config)

    def _build_response(self, results, config: SimulatorConfig) -> SimulationResponse:
        """Build SimulationResponse from simulation results."""
        phase_results = []

        for i, phase in enumerate(results.phases):
            # Get final values
            diameter = float(results.diameter[-1, i])
            aspect_ratio = float(results.aspect_ratio[-1, i])
            major_axis = float(results.major_axis_length[-1, i])
            volume_fraction = float(results.volume_fraction[-1, i])
            crss = results.orowan_crss[-1, i]
            yield_strength = results.yield_strength[-1, i]

            # Handle non-finite values
            crss_mpa = float(crss / 1e6) if np.isfinite(crss) else 0.0
            ys_mpa = float(yield_strength / 1e6) if np.isfinite(yield_strength) else 0.0

            phase_results.append(
                PhaseResult(
                    name=phase,
                    diameter_nm=round(diameter, 4),
                    aspect_ratio=round(aspect_ratio, 4),
                    major_axis_nm=round(major_axis, 4),
                    volume_fraction=round(volume_fraction, 6),
                    crss_mpa=round(crss_mpa, 2),
                    yield_strength_mpa=round(ys_mpa, 2),
                )
            )

        summary = SimulationSummary(
            final_time_hours=round(float(results.time_hours[-1]), 2),
            phases=phase_results,
            total_crss_mpa=round(float(results.orowan_crss_total[-1] / 1e6), 2),
            total_yield_strength_mpa=round(
                float(results.yield_strength_total[-1] / 1e6), 2
            ),
        )

        return SimulationResponse(status="completed", summary=summary)
