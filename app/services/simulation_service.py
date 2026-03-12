"""
Service layer that wraps the kawin precipitation simulation.
"""

import gc
import numpy as np
from app.config import settings
from app.models.simulation import (
    SimulationRequest,
    SimulationResponse,
    SimulationSummary,
    PhaseResult,
    PhaseTimeSeries,
    TimeSeriesData,
)
from kawin.run_simulation import (
    AlMgSiPrecipitationSimulator,
    SimulatorConfig,
)


class SimulationService:
    """Service for running Al-Mg-Si precipitation simulations."""

    def __init__(self, tdb_file: str = None):
        self.tdb_file = tdb_file or settings.TDB_FILE

    def run_simulation(self, request: SimulationRequest) -> SimulationResponse:
        """Run a precipitation simulation."""
        config = SimulatorConfig(temperature_shift=request.temperature_shift)

        simulator = AlMgSiPrecipitationSimulator(
            tdb_file=self.tdb_file,
            init_composition=[request.mg_content, request.si_content],
            aging_temperature=request.aging_temperature,
            aging_time=request.aging_time,
            config=config,
        )

        # Delete temp calculations after simulation
        try:
            results = simulator.solve(verbose=False)
            return self._build_response(results, simulator)
        finally:
            del simulator
            del results
            gc.collect()

    def _build_response(self, results, simulator) -> SimulationResponse:
        """Build SimulationResponse from simulation results."""
        # Build phase results
        phase_results = []
        for i, phase in enumerate(results.phases):
            phase_results.append(
                PhaseResult(
                    name=phase,
                    diameter_nm=round(float(results.diameter[-1, i]), 2),
                    major_axis_nm=round(float(results.major_axis_length[-1, i]), 2),
                    volume_fraction_pct=round(float(results.volume_fraction[-1, i]) * 100, 4),
                )
            )

        summary = SimulationSummary(
            final_time_hours=round(float(results.time_hours[-1]), 2),
            user_temperature_c=simulator.user_temperature,
            effective_temperature_c=simulator.effective_temperature,
            phases=phase_results,
            matrix_mg_wt_pct=round(float(results.matrix_Mg_wt_pct[-1]), 4),
            matrix_si_wt_pct=round(float(results.matrix_Si_wt_pct[-1]), 4),
            orowan_strength_mpa=round(float(results.orowan_strength[-1] / 1e6), 1),
            solid_solution_strength_mpa=round(float(results.solid_solution_strength[-1] / 1e6), 1),
            total_yield_strength_mpa=round(float(results.total_strength[-1] / 1e6), 1),
        )

        time_series = self._build_time_series(results)

        return SimulationResponse(
            status="completed",
            summary=summary,
            time_series=time_series,
        )

    def _build_time_series(self, results) -> TimeSeriesData:
        """Build TimeSeriesData from simulation results."""
        time_hours = [round(float(t), 4) for t in results.time_hours]

        phase_time_series = []
        for i, phase in enumerate(results.phases):
            phase_time_series.append(
                PhaseTimeSeries(
                    name=phase,
                    diameter_nm=self._to_list(results.diameter[:, i]),
                    major_axis_nm=self._to_list(results.major_axis_length[:, i]),
                    volume_fraction_pct=self._to_list(results.volume_fraction[:, i] * 100, 4),
                )
            )

        return TimeSeriesData(
            time_hours=time_hours,
            phases=phase_time_series,
            matrix_mg_wt_pct=self._to_list(results.matrix_Mg_wt_pct, 4),
            matrix_si_wt_pct=self._to_list(results.matrix_Si_wt_pct, 4),
            orowan_strength_mpa=self._to_list(results.orowan_strength / 1e6, 1),
            solid_solution_strength_mpa=self._to_list(results.solid_solution_strength / 1e6, 1),
            total_yield_strength_mpa=self._to_list(results.total_strength / 1e6, 1),
        )

    @staticmethod
    def _to_list(arr: np.ndarray, decimals: int = 2) -> list[float]:
        """Convert numpy array to list."""
        return [round(float(v), decimals) if np.isfinite(v) else 0.0 for v in arr]
