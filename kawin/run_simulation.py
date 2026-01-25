"""
Al-Mg-Si Precipitation Simulation with Yield Strength Prediction

Simulates precipitation kinetics and calculates yield strength from:
- Orowan precipitation hardening (Paper1 formula)
- Solid solution strengthening
"""

import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from kawin.thermo import MulticomponentThermodynamics
from kawin.precipitation.PrecipitationParameters import TemperatureParameters
from kawin.precipitation import MatrixParameters, PrecipitateParameters, PrecipitateModel
from kawin.precipitation.coupling.Strength import DislocationParameters
from kawin.solver import explicitEulerIterator


@dataclass
class AspectRatioConfig:
    """Aspect ratio: AR = prefactor * (2r / 1nm)^exponent"""
    prefactor: float = 5.55
    exponent: float = 0.24

    def calculate(self, radius: np.ndarray) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            ar = self.prefactor * (2 * radius * 1e9) ** self.exponent
            ar = np.where(np.isfinite(ar), ar, 0)
        return ar


@dataclass
class DislocationConfig:
    """Dislocation parameters for aluminum."""
    shear_modulus: float = 25.4e9    # G (Pa)
    burgers_vector: float = 0.286e-9  # b (m)
    poisson_ratio: float = 0.34

    def to_params(self) -> DislocationParameters:
        return DislocationParameters(G=self.shear_modulus, b=self.burgers_vector, nu=self.poisson_ratio)


@dataclass
class SolidSolutionConfig:
    """Solid solution strengthening: σ_ss = k_Mg * wt%_Mg + k_Si * wt%_Si"""
    enabled: bool = True
    k_Mg: float = 18.6   # MPa/wt.%
    k_Si: float = 9.2    # MPa/wt.%
    M_Al: float = 26.98154
    M_Mg: float = 24.305
    M_Si: float = 28.0855


@dataclass
class PhaseStoichiometry:
    """Phase stoichiometry and molar volume."""
    n_Al: float = 0.0
    n_Mg: float = 0.0
    n_Si: float = 0.0
    molar_volume: float = 5e-6  # m³/mol

    @property
    def total_atoms(self) -> float:
        return self.n_Al + self.n_Mg + self.n_Si

    @property
    def x_Mg(self) -> float:
        return self.n_Mg / self.total_atoms if self.total_atoms > 0 else 0.0

    @property
    def x_Si(self) -> float:
        return self.n_Si / self.total_atoms if self.total_atoms > 0 else 0.0


PHASE_STOICHIOMETRY = {
    'MGSI_B_P': PhaseStoichiometry(n_Al=0, n_Mg=1.8, n_Si=1, molar_volume=5e-6),
    'MG5SI6_B_DP': PhaseStoichiometry(n_Al=0, n_Mg=5, n_Si=6, molar_volume=5e-6),
    'B_PRIME_L': PhaseStoichiometry(n_Al=3, n_Mg=9, n_Si=7, molar_volume=2e-6),
    'U1_PHASE': PhaseStoichiometry(n_Al=2, n_Mg=1, n_Si=2, molar_volume=3e-6),
    'U2_PHASE': PhaseStoichiometry(n_Al=1, n_Mg=1, n_Si=1, molar_volume=3e-6),
}


@dataclass
class InterfacialEnergyConfig:
    """Interfacial energies (J/m²)."""
    MGSI_B_P: float = 0.16
    MG5SI6_B_DP: float = 0.108
    B_PRIME_L: float = 0.18
    U1_PHASE: float = 0.18
    U2_PHASE: float = 0.18

    def to_dict(self) -> dict:
        return {
            'MGSI_B_P': self.MGSI_B_P,
            'MG5SI6_B_DP': self.MG5SI6_B_DP,
            'B_PRIME_L': self.B_PRIME_L,
            'U1_PHASE': self.U1_PHASE,
            'U2_PHASE': self.U2_PHASE
        }


@dataclass
class SimulatorConfig:
    """Complete simulator configuration."""
    phases: list = field(default_factory=lambda: ['FCC_A1', 'MGSI_B_P', 'MG5SI6_B_DP'])
    elements: list = field(default_factory=lambda: ['AL', 'MG', 'SI'])
    aspect_ratio: AspectRatioConfig = field(default_factory=AspectRatioConfig)
    dislocation: DislocationConfig = field(default_factory=DislocationConfig)
    solid_solution: SolidSolutionConfig = field(default_factory=SolidSolutionConfig)
    interfacial_energy: InterfacialEnergyConfig = field(default_factory=InterfacialEnergyConfig)
    matrix_molar_volume: float = 6e-6  # m³/mol
    taylor_factor: float = 2.24
    orowan_scaling: float = 0.28
    temperature_shift: float = 15.0  # °C, added to user-specified aging temperature


@dataclass
class SimulationResults:
    """Simulation results container."""
    time: np.ndarray
    phases: list
    radius: np.ndarray
    diameter: np.ndarray
    volume_fraction: np.ndarray
    precipitate_density: np.ndarray
    aspect_ratio: np.ndarray
    major_axis_length: np.ndarray
    matrix_Mg_wt_pct: np.ndarray
    matrix_Si_wt_pct: np.ndarray
    solid_solution_strength: np.ndarray
    orowan_strength: np.ndarray
    total_strength: np.ndarray

    @property
    def time_hours(self) -> np.ndarray:
        return self.time / 3600


class AlMgSiPrecipitationSimulator:
    """
    Al-Mg-Si precipitation simulator.

    Parameters
    ----------
    tdb_file : str
        Path to thermodynamic database (.tdb)
    init_composition : list[float]
        Initial [Mg, Si] in mole fraction
    aging_temperature : float
        Aging temperature in Celsius (temperature_shift will be added)
    aging_time : float
        Aging time in hours
    config : SimulatorConfig, optional
        Configuration parameters
    """

    def __init__(
        self,
        tdb_file: str,
        init_composition: list[float],
        aging_temperature: float,
        aging_time: float,
        config: Optional[SimulatorConfig] = None
    ):
        self.tdb_file = tdb_file
        self.init_composition = init_composition
        self.aging_time_hours = aging_time
        self.config = config or SimulatorConfig()

        # Apply temperature shift
        self.user_temperature = aging_temperature
        self.effective_temperature = aging_temperature + self.config.temperature_shift
        temp_K = self.effective_temperature + 273.15
        self._temperature = TemperatureParameters([0, aging_time], [temp_K, temp_K])

        self._therm = MulticomponentThermodynamics(
            tdb_file, self.config.elements, self.config.phases
        )
        self._model = None

    def solve(self, verbose: bool = True) -> SimulationResults:
        """Run simulation and return results."""
        matrix = MatrixParameters(['MG', 'SI'])
        matrix.initComposition = self.init_composition
        matrix.volume.setVolume(1e-5, 'VM', 4)

        gamma_dict = self.config.interfacial_energy.to_dict()
        ar_func = lambda r: self.config.aspect_ratio.calculate(r)
        precipitates = []

        for phase in self.config.phases[1:]:
            p = PrecipitateParameters(phase)
            p.shapeFactor.setPrecipitateShape('needle', ar_func)
            p.gamma = gamma_dict.get(phase, 0.18)
            p.volume.setVolume(1e-5, 'VM', 4)
            p.nucleation.setNucleationType('bulk')
            p.nucleation.useNeedleNucleation = True
            precipitates.append(p)

        self._model = PrecipitateModel(matrix, precipitates, self._therm, self._temperature)

        if verbose:
            print(f"Running simulation: {self.aging_time_hours}h at {self.user_temperature}°C "
                  f"(effective: {self.effective_temperature}°C)")

        self._model.solve(
            self.aging_time_hours * 3600,
            iterator=explicitEulerIterator,
            verbose=verbose,
            vIt=10000
        )

        return self._extract_results()

    def _extract_results(self) -> SimulationResults:
        """Extract results from solved model."""
        data = self._model.data
        time = data.time
        phases = list(self._model.phases)
        radius = data.Ravg
        volume_fraction = data.volFrac
        precipitate_density = data.precipitateDensity

        diameter = 2 * radius * 1e9
        aspect_ratio = self.config.aspect_ratio.calculate(radius)
        major_axis_length = aspect_ratio * diameter
        major_axis_length_m = major_axis_length * 1e-9

        # Matrix composition
        x_Mg_matrix, x_Si_matrix = self._calc_matrix_composition(volume_fraction, phases)
        matrix_Mg_wt_pct, matrix_Si_wt_pct = self._at_to_wt(x_Mg_matrix, x_Si_matrix)

        # Solid solution strength
        ss = self.config.solid_solution
        if ss.enabled:
            solid_solution_strength = (ss.k_Mg * matrix_Mg_wt_pct + ss.k_Si * matrix_Si_wt_pct) * 1e6
        else:
            solid_solution_strength = np.zeros_like(time)

        # Orowan strength (Ref 1)
        # Ref 1: Coupled precipitation and yield strength modelling for non-isothermal treatments of a 6061 aluminium alloy
        M = self.config.taylor_factor
        alpha = self.config.orowan_scaling
        G = self.config.dislocation.shear_modulus
        b = self.config.dislocation.burgers_vector
        orowan_factor = np.sqrt(np.sum(major_axis_length_m * precipitate_density, axis=1))
        orowan_strength = np.sqrt(2) * M * alpha * G * b * orowan_factor

        total_strength = orowan_strength + solid_solution_strength

        return SimulationResults(
            time=time,
            phases=phases,
            radius=radius,
            diameter=diameter,
            volume_fraction=volume_fraction,
            precipitate_density=precipitate_density,
            aspect_ratio=aspect_ratio,
            major_axis_length=major_axis_length,
            matrix_Mg_wt_pct=matrix_Mg_wt_pct,
            matrix_Si_wt_pct=matrix_Si_wt_pct,
            solid_solution_strength=solid_solution_strength,
            orowan_strength=orowan_strength,
            total_strength=total_strength,
        )

    def _calc_matrix_composition(self, volume_fraction, phases):
        """Calculate remaining Mg/Si in matrix."""
        x_Mg_0, x_Si_0 = self.init_composition
        Vm_alpha = self.config.matrix_molar_volume
        n_times = volume_fraction.shape[0]
        Mg_consumed, Si_consumed, f_total = np.zeros(n_times), np.zeros(n_times), np.zeros(n_times)

        for i, phase in enumerate(phases):
            if phase in PHASE_STOICHIOMETRY:
                stoich = PHASE_STOICHIOMETRY[phase]
                f_p = volume_fraction[:, i]
                factor = f_p * (Vm_alpha / stoich.molar_volume)
                Mg_consumed += stoich.x_Mg * factor
                Si_consumed += stoich.x_Si * factor
                f_total += f_p

        with np.errstate(divide='ignore', invalid='ignore'):
            x_Mg = np.maximum((x_Mg_0 - Mg_consumed) / (1 - f_total), 0)
            x_Si = np.maximum((x_Si_0 - Si_consumed) / (1 - f_total), 0)
            x_Mg = np.where(np.isfinite(x_Mg), x_Mg, x_Mg_0)
            x_Si = np.where(np.isfinite(x_Si), x_Si, x_Si_0)

        return x_Mg, x_Si

    def _at_to_wt(self, x_Mg, x_Si):
        """Convert atomic fraction to weight percent."""
        ss = self.config.solid_solution
        x_Al = np.maximum(1 - x_Mg - x_Si, 0)
        total = x_Al * ss.M_Al + x_Mg * ss.M_Mg + x_Si * ss.M_Si
        with np.errstate(divide='ignore', invalid='ignore'):
            wt_Mg = np.where(total > 0, (x_Mg * ss.M_Mg / total) * 100, 0)
            wt_Si = np.where(total > 0, (x_Si * ss.M_Si / total) * 100, 0)
        return wt_Mg, wt_Si

    def plot_results(self, results: SimulationResults, save_dir: Optional[str] = None, show: bool = True):
        """Generate result plots."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        t = results.time_hours

        # Volume fraction
        ax = axes[0, 0]
        for i, phase in enumerate(results.phases):
            ax.plot(t, results.volume_fraction[:, i] * 100, label=phase)
        ax.plot(t, np.sum(results.volume_fraction, axis=1) * 100, 'k--', label='Total')
        ax.set_xlabel('Time (hr)')
        ax.set_ylabel('Volume Fraction (%)')
        ax.set_title('Precipitate Volume Fraction')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Precipitate size
        ax = axes[0, 1]
        for i, phase in enumerate(results.phases):
            ax.plot(t, results.major_axis_length[:, i], label=phase)
        ax.set_xlabel('Time (hr)')
        ax.set_ylabel('Major Axis Length (nm)')
        ax.set_title('Precipitate Size')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Strength contributions
        ax = axes[1, 0]
        ax.plot(t, results.orowan_strength / 1e6, label='Orowan')
        ax.plot(t, results.solid_solution_strength / 1e6, label='Solid Solution')
        ax.plot(t, results.total_strength / 1e6, 'k-', linewidth=2, label='Total')
        ax.set_xlabel('Time (hr)')
        ax.set_ylabel('Yield Strength (MPa)')
        ax.set_title('Strength Contributions')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Matrix composition
        ax = axes[1, 1]
        ax.plot(t, results.matrix_Mg_wt_pct, label='Mg')
        ax.plot(t, results.matrix_Si_wt_pct, label='Si')
        ax.set_xlabel('Time (hr)')
        ax.set_ylabel('Weight %')
        ax.set_title('Matrix Composition')
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle(f'Al-Mg-Si Aging at {self.user_temperature}°C (eff: {self.effective_temperature}°C)',
                     fontsize=12, fontweight='bold')
        fig.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fig.savefig(os.path.join(save_dir, 'results.png'), dpi=150)

        if show:
            plt.show()

        return fig

    def print_summary(self, results: SimulationResults):
        """Print results summary."""
        print("\n" + "=" * 60)
        print(f"Simulation Summary: {self.aging_time_hours}h at {self.user_temperature}°C")
        print(f"(Effective temperature: {self.effective_temperature}°C)")
        print("=" * 60)

        print(f"\nInitial composition: Mg={self.init_composition[0]*100:.3f}at.%, Si={self.init_composition[1]*100:.3f}at.%")

        print(f"\nFinal Results (t = {results.time_hours[-1]:.1f} hr):")
        print("-" * 40)
        for i, phase in enumerate(results.phases):
            vf = results.volume_fraction[-1, i] * 100
            d = results.diameter[-1, i]
            L = results.major_axis_length[-1, i]
            print(f"  {phase}: VF={vf:.4f}%, D={d:.2f}nm, L={L:.2f}nm")

        print(f"\nMatrix: Mg={results.matrix_Mg_wt_pct[-1]:.3f}wt.%, Si={results.matrix_Si_wt_pct[-1]:.3f}wt.%")

        print(f"\nYield Strength:")
        print(f"  Orowan:         {results.orowan_strength[-1]/1e6:.1f} MPa")
        print(f"  Solid Solution: {results.solid_solution_strength[-1]/1e6:.1f} MPa")
        print(f"  TOTAL:          {results.total_strength[-1]/1e6:.1f} MPa")
        print("=" * 60)


def run_simulation(
    aging_temperature: float,
    aging_time: float,
    mg_content: float = 0.0072,
    si_content: float = 0.0057,
    temperature_shift: float = 15.0,
    save_dir: Optional[str] = None,
    show_plots: bool = True,
    verbose: bool = True
) -> SimulationResults:
    """
    Run precipitation simulation.

    Parameters
    ----------
    aging_temperature : float
        Aging temperature in Celsius
    aging_time : float
        Aging time in hours
    mg_content : float
        Mg content in mole fraction (default: 0.0072 = 0.72 at.%)
    si_content : float
        Si content in mole fraction (default: 0.0057 = 0.57 at.%)
    temperature_shift : float
        Temperature correction in Celsius (default: 15)
    save_dir : str, optional
        Directory to save plots
    show_plots : bool
        Show plots interactively
    verbose : bool
        Print progress

    Returns
    -------
    SimulationResults
        Simulation results
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tdb_file = os.path.join(project_root, 'data', 'AlMgSi.tdb')

    config = SimulatorConfig(temperature_shift=temperature_shift)

    simulator = AlMgSiPrecipitationSimulator(
        tdb_file=tdb_file,
        init_composition=[mg_content, si_content],
        aging_temperature=aging_temperature,
        aging_time=aging_time,
        config=config
    )

    results = simulator.solve(verbose=verbose)
    simulator.print_summary(results)

    if save_dir or show_plots:
        simulator.plot_results(results, save_dir=save_dir, show=show_plots)

    return results


if __name__ == '__main__':
    results = run_simulation(
        aging_temperature=175,
        aging_time=24,
        mg_content=0.0072, # in mole fraction
        si_content=0.0057, # in mole fraction
        temperature_shift=15,
        save_dir='results',
        show_plots=True
    )
