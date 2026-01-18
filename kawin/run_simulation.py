"""
Multiphase Precipitation Simulation for Al-Mg-Si System

This module provides a class-based interface for modeling precipitation in the
Al-Mg-Si system with multiple phases:
- MGSI_B_P (β')
- MG5SI6_B_DP (β")
- B_PRIME_L (B')
- U1_PHASE
- U2_PHASE

References:
1. E. Povoden-Karadeniz et al, "Calphad modeling of metastable phases in the
   Al-Mg-Si system" Calphad 43 (2013) p. 94
2. Q. Du et al, "Modeling over-ageing in Al-Mg-Si alloys by a multi-phase
   Calphad-coupled Kampmann-Wagner Numerical model" Acta Materialia 122 (2017) p. 178
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np
import matplotlib.pyplot as plt
from kawin.thermo import MulticomponentThermodynamics
from kawin.precipitation.PrecipitationParameters import TemperatureParameters
from kawin.precipitation import MatrixParameters, PrecipitateParameters, PrecipitateModel
from kawin.precipitation.coupling.Strength import OrowanContribution, DislocationParameters
from kawin.solver import explicitEulerIterator
from kawin.precipitation.Plot import (
    plotPrecipitateDensity,
    plotVolumeFraction,
    plotAverageRadius
)


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class AspectRatioConfig:
    """
    Configuration for aspect ratio calculation.

    AR = prefactor * (2*r / length_scale)^exponent

    Default values from: Anass Assadiki et al, 2021
    """
    prefactor: float = 5.55
    exponent: float = 0.24
    length_scale: float = 1e-9  # 1 nm in meters

    def calculate(self, radius: np.ndarray) -> np.ndarray:
        """Calculate aspect ratio from radius (m)."""
        with np.errstate(divide='ignore', invalid='ignore'):
            ar = self.prefactor * (2 * radius / self.length_scale) ** self.exponent
            ar = np.where(np.isfinite(ar), ar, 0)
        return ar


@dataclass
class DislocationConfig:
    """
    Dislocation parameters for Orowan strength calculation.

    Default values are typical for aluminum.
    """
    shear_modulus: float = 25.4e9   # G in Pa (25.4 GPa for Al)
    burgers_vector: float = 0.286e-9  # b in m (0.286 nm for Al)
    poisson_ratio: float = 0.34     # nu (dimensionless)

    def to_kawin_params(self) -> DislocationParameters:
        """Convert to kawin DislocationParameters object."""
        return DislocationParameters(
            G=self.shear_modulus,
            b=self.burgers_vector,
            nu=self.poisson_ratio
        )


@dataclass
class InterfacialEnergyConfig:
    """
    Interfacial energies for each precipitate phase (J/m²).
    """
    MGSI_B_P: float = 0.18
    MG5SI6_B_DP: float = 0.084
    B_PRIME_L: float = 0.18
    U1_PHASE: float = 0.18
    U2_PHASE: float = 0.18

    def to_dict(self) -> dict:
        """Convert to dictionary for kawin."""
        return {
            'MGSI_B_P': self.MGSI_B_P,
            'MG5SI6_B_DP': self.MG5SI6_B_DP,
            'B_PRIME_L': self.B_PRIME_L,
            'U1_PHASE': self.U1_PHASE,
            'U2_PHASE': self.U2_PHASE
        }


@dataclass
class MatrixConfig:
    """Matrix phase configuration."""
    molar_volume: float = 1e-5      # m³/mol
    volume_type: str = 'VM'
    atomic_number: int = 4


@dataclass
class PrecipitateConfig:
    """Precipitate phase configuration."""
    shape: str = 'plate'            # 'sphere', 'plate', 'needle'
    molar_volume: float = 1e-5      # m³/mol
    volume_type: str = 'VM'
    atomic_number: int = 4
    nucleation_type: str = 'bulk'   # 'bulk', 'grain_boundary', 'dislocation'


@dataclass
class SimulatorConfig:
    """
    Complete configuration for Al-Mg-Si precipitation simulator.

    Groups all configurable parameters in one place for easy management.
    """
    # Phase definitions
    phases: list = field(default_factory=lambda: [
        'FCC_A1', 'MGSI_B_P', 'MG5SI6_B_DP', 'B_PRIME_L', 'U1_PHASE', 'U2_PHASE'
    ])
    elements: list = field(default_factory=lambda: ['AL', 'MG', 'SI'])

    # Sub-configurations
    aspect_ratio: AspectRatioConfig = field(default_factory=AspectRatioConfig)
    dislocation: DislocationConfig = field(default_factory=DislocationConfig)
    interfacial_energy: InterfacialEnergyConfig = field(default_factory=InterfacialEnergyConfig)
    matrix: MatrixConfig = field(default_factory=MatrixConfig)
    precipitate: PrecipitateConfig = field(default_factory=PrecipitateConfig)

    # Taylor factor for converting CRSS to yield strength
    taylor_factor: float = 2.24


# =============================================================================
# Results Container
# =============================================================================

@dataclass
class SimulationResults:
    """Container for simulation results."""
    time: np.ndarray                    # Time in seconds
    phases: list                        # List of phase names
    radius: np.ndarray                  # Mean radius (m) - shape: (n_times, n_phases)
    diameter: np.ndarray                # Diameter (nm) - shape: (n_times, n_phases)
    volume_fraction: np.ndarray         # Volume fraction - shape: (n_times, n_phases)
    precipitate_density: np.ndarray     # Precipitate density (#/m³) - shape: (n_times, n_phases)
    aspect_ratio: np.ndarray            # Aspect ratio - shape: (n_times, n_phases)
    major_axis_length: np.ndarray       # Major axis length (nm) - shape: (n_times, n_phases)
    orowan_crss: np.ndarray             # Orowan CRSS per phase (Pa) - shape: (n_times, n_phases)
    orowan_crss_total: np.ndarray       # Total Orowan CRSS (Pa) - shape: (n_times,)
    yield_strength: np.ndarray          # Yield strength per phase (Pa) - shape: (n_times, n_phases)
    yield_strength_total: np.ndarray    # Total yield strength (Pa) - shape: (n_times,)

    @property
    def time_hours(self) -> np.ndarray:
        """Time in hours."""
        return self.time / 3600


# =============================================================================
# Main Simulator Class
# =============================================================================

class AlMgSiPrecipitationSimulator:
    """
    Precipitation simulator for Al-Mg-Si alloy system.

    Parameters
    ----------
    tdb_file : str
        Path to the thermodynamic database file (.tdb)
    init_composition : list[float]
        Initial composition [Mg, Si] in mole fraction
    temperature : TemperatureParameters or callable
        Temperature profile. Can be:
        - TemperatureParameters object
        - A callable that takes time (s) and returns temperature (K)
    config : SimulatorConfig, optional
        Configuration object containing all parameters.
        If None, uses default configuration.

    Examples
    --------
    >>> # Define temperature profile
    >>> temp = TemperatureParameters([0, 16, 17], [448, 448, 523])
    >>>
    >>> # Create simulator with default config
    >>> sim = AlMgSiPrecipitationSimulator(
    ...     tdb_file='AlMgSi.tdb',
    ...     init_composition=[0.0072, 0.0057],
    ...     temperature=temp
    ... )
    >>>
    >>> # Or with custom config
    >>> config = SimulatorConfig()
    >>> config.dislocation.shear_modulus = 26e9  # Custom shear modulus
    >>> sim = AlMgSiPrecipitationSimulator(..., config=config)
    >>>
    >>> # Run simulation
    >>> results = sim.solve(simulation_time=25*3600)
    >>>
    >>> # Plot results
    >>> sim.plot_orowan_strength(results)
    >>> sim.plot_yield_strength(results)
    """

    def __init__(
        self,
        tdb_file: str,
        init_composition: list[float],
        temperature: TemperatureParameters | Callable,
        config: Optional[SimulatorConfig] = None
    ):
        self.tdb_file = tdb_file
        self.init_composition = init_composition
        self.config = config or SimulatorConfig()

        # Handle temperature input
        if callable(temperature) and not isinstance(temperature, TemperatureParameters):
            self._temperature = TemperatureParameters(temperature)
        else:
            self._temperature = temperature

        # Initialize thermodynamics
        self._therm = MulticomponentThermodynamics(
            tdb_file, self.config.elements, self.config.phases
        )

        # Model will be created during solve
        self._model: Optional[PrecipitateModel] = None

    def _setup_matrix(self) -> MatrixParameters:
        """Setup matrix parameters."""
        matrix = MatrixParameters(['MG', 'SI'])
        matrix.initComposition = self.init_composition
        matrix.volume.setVolume(
            self.config.matrix.molar_volume,
            self.config.matrix.volume_type,
            self.config.matrix.atomic_number
        )
        return matrix

    def _setup_precipitates(self) -> list[PrecipitateParameters]:
        """Setup precipitate parameters for all phases."""
        gamma_dict = self.config.interfacial_energy.to_dict()
        precipitates = []

        for phase in self.config.phases[1:]:  # Skip FCC_A1 (matrix)
            params = PrecipitateParameters(phase)
            params.shapeFactor.setPrecipitateShape(self.config.precipitate.shape)
            params.gamma = gamma_dict.get(phase, 0.18)
            params.volume.setVolume(
                self.config.precipitate.molar_volume,
                self.config.precipitate.volume_type,
                self.config.precipitate.atomic_number
            )
            params.nucleation.setNucleationType(self.config.precipitate.nucleation_type)
            precipitates.append(params)
        return precipitates

    @staticmethod
    def _volume_fraction_to_Ls(r: np.ndarray, volume_fraction: np.ndarray) -> np.ndarray:
        """
        Convert radius and volume fraction to surface-to-surface particle spacing.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            Ls = r * (np.sqrt(2 * np.pi / (3 * volume_fraction)) - 2)
            Ls = np.where(np.isfinite(Ls) & (Ls > 0), Ls, 0)
        return Ls

    def solve(
        self,
        simulation_time: float,
        verbose: bool = True,
        verbose_interval: int = 10000
    ) -> SimulationResults:
        """
        Run the precipitation simulation.

        Parameters
        ----------
        simulation_time : float
            Total simulation time in seconds
        verbose : bool, optional
            Print progress during simulation. Default: True
        verbose_interval : int, optional
            Print interval for verbose output. Default: 10000

        Returns
        -------
        SimulationResults
            Dataclass containing all simulation results
        """
        # Setup model
        matrix = self._setup_matrix()
        precipitates = self._setup_precipitates()
        self._model = PrecipitateModel(matrix, precipitates, self._therm, self._temperature)

        # Solve
        if verbose:
            print("Solving precipitation model...")
            print("=" * 80)

        self._model.solve(
            simulation_time,
            iterator=explicitEulerIterator,
            verbose=verbose,
            vIt=verbose_interval
        )

        if verbose:
            print("=" * 80)
            print("Simulation complete!")

        # Extract results
        return self._extract_results()

    def _extract_results(self) -> SimulationResults:
        """Extract and compute all results from the solved model."""
        data = self._model.data

        # Basic data
        time = data.time
        phases = list(self._model.phases)
        radius = data.Ravg
        volume_fraction = data.volFrac
        precipitate_density = data.precipitateDensity

        # Derived quantities
        diameter = 2 * radius * 1e9  # nm
        aspect_ratio = self.config.aspect_ratio.calculate(radius)
        major_axis_length = aspect_ratio * diameter  # nm

        # Orowan strength calculation
        Ls = self._volume_fraction_to_Ls(radius, volume_fraction)
        dislocation_params = self.config.dislocation.to_kawin_params()
        orowan = OrowanContribution()
        orowan_crss = orowan.computeCRSS(radius, Ls, dislocation_params)

        # Clean inf values and compute totals
        orowan_clean = np.where(np.isfinite(orowan_crss), orowan_crss, 0)
        orowan_crss_total = np.sum(orowan_clean, axis=1)

        # Convert CRSS to yield strength using Taylor factor
        M = self.config.taylor_factor
        yield_strength = orowan_clean * M
        yield_strength_total = orowan_crss_total * M

        return SimulationResults(
            time=time,
            phases=phases,
            radius=radius,
            diameter=diameter,
            volume_fraction=volume_fraction,
            precipitate_density=precipitate_density,
            aspect_ratio=aspect_ratio,
            major_axis_length=major_axis_length,
            orowan_crss=orowan_crss,
            orowan_crss_total=orowan_crss_total,
            yield_strength=yield_strength,
            yield_strength_total=yield_strength_total
        )

    # =========================================================================
    # Individual Plot Methods
    # =========================================================================

    def plot_precipitation_overview(
        self,
        results: SimulationResults,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot precipitation overview: density, volume fraction, radius.

        Parameters
        ----------
        results : SimulationResults
        save_path : str, optional
            Path to save the figure
        show : bool
            Show plot interactively

        Returns
        -------
        plt.Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        phases_plus_total = results.phases + ['Total']

        plotPrecipitateDensity(
            self._model, ax=axes[0, 0], timeUnits='h',
            phases=phases_plus_total,
            label={'Total': 'Total'}, color={'Total': 'k'},
            linestyle={'Total': (0, (5, 5))}
        )
        axes[0, 0].set_ylim([1e5, 1e25])
        axes[0, 0].set_xscale('linear')
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_title('Precipitate Density')

        plotVolumeFraction(
            self._model, ax=axes[0, 1], timeUnits='h',
            phases=phases_plus_total,
            label={'Total': 'Total'}, color={'Total': 'k'},
            linestyle={'Total': (0, (5, 5))}
        )
        axes[0, 1].set_xscale('linear')
        axes[0, 1].set_title('Volume Fraction')

        plotAverageRadius(self._model, ax=axes[1, 0], timeUnits='h')
        axes[1, 0].set_xscale('linear')
        axes[1, 0].set_title('Average Radius')

        plotVolumeFraction(self._model, ax=axes[1, 1], timeUnits='h')
        axes[1, 1].set_xscale('linear')
        axes[1, 1].set_title('Volume Fraction by Phase')

        fig.suptitle('Al-Mg-Si Precipitation Results', fontsize=14, fontweight='bold')
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved: {save_path}")
        if show:
            plt.show()

        return fig

    def plot_orowan_strength(
        self,
        results: SimulationResults,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot Orowan CRSS vs. aging time for each phase.

        Parameters
        ----------
        results : SimulationResults
        save_path : str, optional
            Path to save the figure
        show : bool
            Show plot interactively

        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        time_hours = results.time_hours

        for i, phase in enumerate(results.phases):
            max_val = np.nanmax(results.orowan_crss[:, i])
            if max_val > 0 and np.isfinite(max_val):
                ax.plot(time_hours, results.orowan_crss[:, i] / 1e6,
                       label=phase, linewidth=1.5)

        ax.plot(time_hours, results.orowan_crss_total / 1e6,
               label='Total', color='k', linestyle='--', linewidth=2)

        ax.set_xlabel('Aging Time (hr)')
        ax.set_ylabel('CRSS (MPa)')
        ax.set_title('Orowan Strengthening vs. Aging Time')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, time_hours[-1]])
        ax.set_ylim(bottom=0)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved: {save_path}")
        if show:
            plt.show()

        return fig

    def plot_yield_strength(
        self,
        results: SimulationResults,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot yield strength vs. aging time for each phase.

        Yield strength = Taylor factor x CRSS

        Parameters
        ----------
        results : SimulationResults
        save_path : str, optional
            Path to save the figure
        show : bool
            Show plot interactively

        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        time_hours = results.time_hours

        for i, phase in enumerate(results.phases):
            max_val = np.nanmax(results.yield_strength[:, i])
            if max_val > 0 and np.isfinite(max_val):
                ax.plot(time_hours, results.yield_strength[:, i] / 1e6,
                       label=phase, linewidth=1.5)

        ax.plot(time_hours, results.yield_strength_total / 1e6,
               label='Total', color='k', linestyle='--', linewidth=2)

        ax.set_xlabel('Aging Time (hr)')
        ax.set_ylabel('Yield Strength (MPa)')
        ax.set_title(f'Yield Strength vs. Aging Time (M = {self.config.taylor_factor})')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, time_hours[-1]])
        ax.set_ylim(bottom=0)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved: {save_path}")
        if show:
            plt.show()

        return fig

    def plot_precipitate_size(
        self,
        results: SimulationResults,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot precipitate diameter and major axis length vs. aging time.

        Parameters
        ----------
        results : SimulationResults
        save_path : str, optional
            Path to save the figure
        show : bool
            Show plot interactively

        Returns
        -------
        plt.Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        time_hours = results.time_hours

        for i, phase in enumerate(results.phases):
            axes[0].plot(time_hours, results.diameter[:, i], label=phase, linewidth=1.5)
            axes[1].plot(time_hours, results.major_axis_length[:, i], label=phase, linewidth=1.5)

        axes[0].set_xlabel('Aging Time (hr)')
        axes[0].set_ylabel('Diameter (nm)')
        axes[0].set_title('Precipitate Diameter')
        axes[0].legend(loc='best', fontsize=8)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([0, time_hours[-1]])

        ar_cfg = self.config.aspect_ratio
        axes[1].set_xlabel('Aging Time (hr)')
        axes[1].set_ylabel('Major Axis Length (nm)')
        axes[1].set_title(f'Major Axis Length\n(AR = {ar_cfg.prefactor} x (d/nm)^{ar_cfg.exponent})')
        axes[1].legend(loc='best', fontsize=8)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([0, time_hours[-1]])

        fig.suptitle('Al-Mg-Si Precipitate Size Evolution', fontsize=14, fontweight='bold')
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved: {save_path}")
        if show:
            plt.show()

        return fig

    def plot_volume_fraction(
        self,
        results: SimulationResults,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot volume fraction vs. aging time for each phase.

        Parameters
        ----------
        results : SimulationResults
        save_path : str, optional
            Path to save the figure
        show : bool
            Show plot interactively

        Returns
        -------
        plt.Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        time_hours = results.time_hours

        for i, phase in enumerate(results.phases):
            ax.plot(time_hours, results.volume_fraction[:, i], label=phase, linewidth=1.5)

        # Total volume fraction
        total_vf = np.sum(results.volume_fraction, axis=1)
        ax.plot(time_hours, total_vf, label='Total', color='k', linestyle='--', linewidth=2)

        ax.set_xlabel('Aging Time (hr)')
        ax.set_ylabel('Volume Fraction')
        ax.set_title('Volume Fraction vs. Aging Time')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, time_hours[-1]])
        ax.set_ylim(bottom=0)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"Saved: {save_path}")
        if show:
            plt.show()

        return fig

    def plot_all(
        self,
        results: SimulationResults,
        save_dir: Optional[str] = None,
        show: bool = True
    ) -> dict:
        """
        Generate all plots.

        Parameters
        ----------
        results : SimulationResults
        save_dir : str, optional
            Directory to save all plots
        show : bool
            Show plots interactively

        Returns
        -------
        dict
            Dictionary of figure objects
        """
        figures = {}

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        def get_path(name):
            return os.path.join(save_dir, name) if save_dir else None

        figures['precipitation'] = self.plot_precipitation_overview(
            results, save_path=get_path('precipitation_results.png'), show=False
        )
        figures['orowan'] = self.plot_orowan_strength(
            results, save_path=get_path('orowan_strengthening.png'), show=False
        )
        figures['yield_strength'] = self.plot_yield_strength(
            results, save_path=get_path('yield_strength.png'), show=False
        )
        figures['size'] = self.plot_precipitate_size(
            results, save_path=get_path('precipitate_size.png'), show=False
        )
        figures['volume_fraction'] = self.plot_volume_fraction(
            results, save_path=get_path('volume_fraction.png'), show=False
        )

        if save_dir:
            print(f"All plots saved to {save_dir}")

        if show:
            plt.show()

        return figures

    def print_summary(self, results: SimulationResults):
        """Print summary of results at final time point."""
        print("\n" + "=" * 100)
        print(f"Summary at Final Time (t = {results.time_hours[-1]:.1f} hr):")
        print("=" * 100)
        print(f"{'Phase':<20} {'Diameter (nm)':<15} {'Aspect Ratio':<15} "
              f"{'Major Axis (nm)':<18} {'CRSS (MPa)':<15} {'YS (MPa)':<15}")
        print("-" * 100)

        for i, phase in enumerate(results.phases):
            d = results.diameter[-1, i]
            ar = results.aspect_ratio[-1, i]
            L = results.major_axis_length[-1, i]
            crss = results.orowan_crss[-1, i] / 1e6
            ys = results.yield_strength[-1, i] / 1e6
            if np.isfinite(crss):
                print(f"{phase:<20} {d:<15.4f} {ar:<15.4f} {L:<18.4f} {crss:<15.4f} {ys:<15.4f}")
            else:
                print(f"{phase:<20} {d:<15.4f} {ar:<15.4f} {L:<18.4f} {'N/A':<15} {'N/A':<15}")

        print("-" * 100)
        print(f"{'Total:':<68} {results.orowan_crss_total[-1]/1e6:<15.4f} "
              f"{results.yield_strength_total[-1]/1e6:<15.4f}")
        print("=" * 100)

    def print_config(self):
        """Print current configuration."""
        print("\n" + "=" * 60)
        print("Current Configuration:")
        print("=" * 60)

        print("\nAspect Ratio:")
        ar = self.config.aspect_ratio
        print(f"  AR = {ar.prefactor} x (2r / {ar.length_scale*1e9:.0f} nm)^{ar.exponent}")

        print("\nDislocation Parameters:")
        d = self.config.dislocation
        print(f"  Shear modulus (G): {d.shear_modulus/1e9:.1f} GPa")
        print(f"  Burgers vector (b): {d.burgers_vector*1e9:.3f} nm")
        print(f"  Poisson ratio (nu): {d.poisson_ratio}")

        print("\nInterfacial Energies (J/m^2):")
        for phase, gamma in self.config.interfacial_energy.to_dict().items():
            print(f"  {phase}: {gamma}")

        print(f"\nTaylor Factor: {self.config.taylor_factor}")
        print("=" * 60)


# =============================================================================
# Main execution
# =============================================================================
if __name__ == '__main__':
    # Get project root directory (one level up from kawin/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    tdb_file = os.path.join(data_dir, 'AlMgSi.tdb')

    # Define temperature profile: hold at 199C, then ramp to 200C
    temperature = TemperatureParameters(
        [0, 16, 17],  # Time points in hours
        [199 + 273.15, 199 + 273.15, 200 + 273.15]  # Temperatures in Kelvin
    )

    # Create configuration (can customize here)
    config = SimulatorConfig()
    # Example customizations:
    # config.dislocation.shear_modulus = 26e9
    # config.aspect_ratio.prefactor = 6.0
    # config.interfacial_energy.MGSI_B_P = 0.20

    # Create simulator
    simulator = AlMgSiPrecipitationSimulator(
        tdb_file=tdb_file,
        init_composition=[0.0072, 0.0057],  # Al-0.72Mg-0.57Si (mole fraction)
        temperature=temperature,
        config=config
    )

    # Print configuration
    simulator.print_config()

    # Run simulation (25 hours)
    results = simulator.solve(simulation_time=25 * 3600, verbose=True)

    # Print summary
    simulator.print_summary(results)

    # Generate and save all plots to results directory
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    simulator.plot_all(results, save_dir=results_dir, show=True)
