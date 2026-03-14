"""
Al-Mg-Si Precipitation Simulation with Yield Strength Prediction

Uses kawin's built-in StrengthModel (Orowan + solid solution)
for yield strength calculation.

Precipitate phases: MGSI_B_P (β'), MG5SI6_B_DP (β'')
No temperature shift applied — aging temperature used directly.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from kawin.thermo import MulticomponentThermodynamics
from kawin.precipitation.PrecipitationParameters import TemperatureParameters
from kawin.precipitation import MatrixParameters, PrecipitateParameters, PrecipitateModel
from kawin.precipitation.coupling import StrengthModel
from kawin.precipitation.coupling import (
    DislocationParameters, OrowanContribution,
    CoherencyContribution, ModulusContribution,
)
from kawin.precipitation.coupling.Strength import SolidSolutionStrength
from kawin.solver import explicitEulerIterator


# ---------------------------------------------------------------------------
# Phase stoichiometry (for matrix mass balance)
# ---------------------------------------------------------------------------

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
    'MGSI_B_P':    PhaseStoichiometry(n_Al=0, n_Mg=1.8, n_Si=1,   molar_volume=5e-6),
    'MG5SI6_B_DP': PhaseStoichiometry(n_Al=0, n_Mg=5,   n_Si=6,   molar_volume=5e-6),
    'B_PRIME_L':   PhaseStoichiometry(n_Al=3, n_Mg=9,   n_Si=7,   molar_volume=2e-6),
    'U1_PHASE':    PhaseStoichiometry(n_Al=2, n_Mg=1,   n_Si=2,   molar_volume=3e-6),
    'U2_PHASE':    PhaseStoichiometry(n_Al=1, n_Mg=1,   n_Si=1,   molar_volume=3e-6),
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimulatorConfig:
    """Complete simulator configuration."""
    phases: list = field(default_factory=lambda: ['FCC_A1', 'MGSI_B_P', 'MG5SI6_B_DP'])
    elements: list = field(default_factory=lambda: ['AL', 'MG', 'SI'])
    matrix_molar_volume: float = 6e-6   # m³/mol

    # Interfacial energies (J/m²)
    gamma: dict = field(default_factory=lambda: {
        'MGSI_B_P':    0.18,
        'MG5SI6_B_DP': 0.084,
        'B_PRIME_L':   0.18,
        'U1_PHASE':    0.18,
        'U2_PHASE':    0.18,
    })

    # Dislocation parameters for Al matrix
    shear_modulus: float = 25.4e9    # G (Pa)
    burgers_vector: float = 0.286e-9 # b (m)
    poisson_ratio: float = 0.34

    # Shearing contributions for β'' / β' precipitates in Al-Mg-Si
    # Coherency misfit strain — β'' is semi-coherent with Al, ε ~ 0.02-0.05
    coherency_eps: float = 0.025
    # Precipitate shear modulus — between Al (25 GPa) and Si (160 GPa)
    precipitate_shear_modulus: float = 50e9  # Gp (Pa)

    # Solid solution strengthening weights (Pa · mol/mol)
    # σ_ss = w_Mg * x_Mg + w_Si * x_Si  (x in mole fraction)
    # Derived from k_Mg=18.6 MPa/wt.%, k_Si=9.2 MPa/wt.%
    ss_weight_Mg: float = 1.676e9
    ss_weight_Si: float = 0.958e9


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class SimulationResults:
    """Simulation results container."""
    time: np.ndarray
    phases: list
    radius: np.ndarray
    diameter: np.ndarray
    volume_fraction: np.ndarray
    precipitate_density: np.ndarray
    matrix_Mg_wt_pct: np.ndarray
    matrix_Si_wt_pct: np.ndarray
    solid_solution_strength: np.ndarray
    orowan_strength: np.ndarray
    total_strength: np.ndarray

    @property
    def time_hours(self) -> np.ndarray:
        return self.time / 3600


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class AlMgSiPrecipitationSimulator:
    """
    Al-Mg-Si precipitation simulator using kawin's built-in strength model.

    Parameters
    ----------
    tdb_file : str
        Path to thermodynamic database (.tdb)
    init_composition : list[float]
        Initial [Mg, Si] in mole fraction
        Default: A356-type (Mg=0.3 wt.%, Si=1.0 wt.%) → [0.003330, 0.009607]
    aging_temperature : float
        Aging temperature in Celsius (used directly, no shift applied)
    aging_time : float
        Aging time in hours
    config : SimulatorConfig, optional
    """

    def __init__(
        self,
        tdb_file: str,
        init_composition: list[float],
        aging_temperature: float,
        aging_time: float,
        config: Optional[SimulatorConfig] = None,
    ):
        self.tdb_file = tdb_file
        self.init_composition = init_composition
        self.aging_time_hours = aging_time
        self.aging_temperature = aging_temperature
        self.config = config or SimulatorConfig()

        temp_K = aging_temperature + 273.15
        self._temperature = TemperatureParameters(
            [0, aging_time], [temp_K, temp_K]
        )

        self._therm = MulticomponentThermodynamics(
            tdb_file, self.config.elements, self.config.phases
        )
        self._model = None
        self._strength_model = None

    def solve(self, verbose: bool = True) -> SimulationResults:
        """Run simulation and return results."""
        # Matrix
        matrix = MatrixParameters(['MG', 'SI'])
        matrix.initComposition = self.init_composition
        matrix.volume.setVolume(1e-5, 'VM', 4)

        # Precipitates — no shape factor, bulk nucleation
        precipitates = []
        for p in self.config.phases[1:]:
            params = PrecipitateParameters(p)
            params.gamma = self.config.gamma[p]
            params.volume.setVolume(1e-5, 'VM', 4)
            params.nucleation.setNucleationType('bulk')
            precipitates.append(params)

        self._model = PrecipitateModel(
            matrix, precipitates, self._therm, self._temperature
        )

        # Kawin strength model: shearing (coherency + modulus) vs Orowan
        # combineCRSS takes min(shearing, Orowan) at each time step, capturing
        # the transition from particle shearing (small r) to Orowan bypass (large r).
        dislocations = DislocationParameters(
            G=self.config.shear_modulus,
            b=self.config.burgers_vector,
            nu=self.config.poisson_ratio,
        )
        contributions = [
            CoherencyContribution(eps=self.config.coherency_eps),
            ModulusContribution(Gp=self.config.precipitate_shear_modulus),
        ]
        ss_model = SolidSolutionStrength({
            'MG': self.config.ss_weight_Mg,
            'SI': self.config.ss_weight_Si,
        })
        self._strength_model = StrengthModel(
            phases=precipitates,
            contributions=contributions,
            dislocations=dislocations,
            ssModel=ss_model,
        )
        self._model.addCouplingModel(self._strength_model)

        if verbose:
            print(f"Running simulation: {self.aging_time_hours}h at {self.aging_temperature}°C")

        self._model.solve(
            self.aging_time_hours * 3600,
            iterator=explicitEulerIterator,
            verbose=verbose,
            vIt=10000,
        )

        return self._extract_results()

    def _extract_results(self) -> SimulationResults:
        data = self._model.data
        time = data.time
        phases = list(self._model.phases)
        radius = data.Ravg
        volume_fraction = data.volFrac
        precipitate_density = data.precipitateDensity
        diameter = 2 * radius * 1e9

        # Matrix composition (mass balance)
        x_Mg, x_Si = self._calc_matrix_composition(volume_fraction, phases)
        matrix_Mg_wt_pct, matrix_Si_wt_pct = self._at_to_wt(x_Mg, x_Si)

        # Kawin strength model output
        total, prec_strength, ss_strength, _ = self._strength_model.totalStrength(
            self._model, returnContributions=True
        )

        return SimulationResults(
            time=time,
            phases=phases,
            radius=radius,
            diameter=diameter,
            volume_fraction=volume_fraction,
            precipitate_density=precipitate_density,
            matrix_Mg_wt_pct=matrix_Mg_wt_pct,
            matrix_Si_wt_pct=matrix_Si_wt_pct,
            solid_solution_strength=ss_strength,
            orowan_strength=prec_strength,
            total_strength=total,
        )

    def _calc_matrix_composition(self, volume_fraction, phases):
        x_Mg_0, x_Si_0 = self.init_composition
        Vm_alpha = self.config.matrix_molar_volume
        n_times = volume_fraction.shape[0]
        Mg_consumed = np.zeros(n_times)
        Si_consumed = np.zeros(n_times)
        f_total = np.zeros(n_times)

        for i, phase in enumerate(phases):
            if phase in PHASE_STOICHIOMETRY:
                s = PHASE_STOICHIOMETRY[phase]
                f_p = volume_fraction[:, i]
                factor = f_p * (Vm_alpha / s.molar_volume)
                Mg_consumed += s.x_Mg * factor
                Si_consumed += s.x_Si * factor
                f_total += f_p

        with np.errstate(divide='ignore', invalid='ignore'):
            x_Mg = np.maximum((x_Mg_0 - Mg_consumed) / (1 - f_total), 0)
            x_Si = np.maximum((x_Si_0 - Si_consumed) / (1 - f_total), 0)
            x_Mg = np.where(np.isfinite(x_Mg), x_Mg, x_Mg_0)
            x_Si = np.where(np.isfinite(x_Si), x_Si, x_Si_0)
        return x_Mg, x_Si

    def _at_to_wt(self, x_Mg, x_Si):
        M_Al, M_Mg, M_Si = 26.98154, 24.305, 28.0855
        x_Al = np.maximum(1 - x_Mg - x_Si, 0)
        total = x_Al * M_Al + x_Mg * M_Mg + x_Si * M_Si
        with np.errstate(divide='ignore', invalid='ignore'):
            wt_Mg = np.where(total > 0, x_Mg * M_Mg / total * 100, 0)
            wt_Si = np.where(total > 0, x_Si * M_Si / total * 100, 0)
        return wt_Mg, wt_Si


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def run_simulation(
    aging_temperature: float,
    aging_time: float,
    # A356-type: Mg=0.3 wt.% → 0.003330 mol frac, Si=1.0 wt.% → 0.009607 mol frac
    mg_content: float = 0.003330,
    si_content: float = 0.009607,
    save_dir: Optional[str] = None,
    verbose: bool = True,
) -> SimulationResults:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tdb_file = os.path.join(project_root, 'data', 'AlMgSi.tdb')

    sim = AlMgSiPrecipitationSimulator(
        tdb_file=tdb_file,
        init_composition=[mg_content, si_content],
        aging_temperature=aging_temperature,
        aging_time=aging_time,
    )
    return sim.solve(verbose=verbose)


# ---------------------------------------------------------------------------
# Main: run 175°C and 200°C, plot σ_y vs aging time up to 12h
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tdb_file = os.path.join(project_root, 'data', 'AlMgSi.tdb')
    out_dir = os.path.join(project_root, 'results', 'kawin_strength')
    os.makedirs(out_dir, exist_ok=True)

    # A356-type: Mg=0.3 wt.%, Si=1.0 wt.%
    MG = 0.003330
    SI = 0.009607

    temps = [175, 200]
    colors = ['steelblue', 'tomato']
    results_all = {}

    for T_C in temps:
        print(f"\n{'='*60}")
        sim = AlMgSiPrecipitationSimulator(
            tdb_file=tdb_file,
            init_composition=[MG, SI],
            aging_temperature=T_C,
            aging_time=12.0,
        )
        res = sim.solve(verbose=True)
        results_all[T_C] = res
        print(f"\nFinal at t=12h, T={T_C}°C:")
        print(f"  Orowan:         {res.orowan_strength[-1]/1e6:.1f} MPa")
        print(f"  Solid solution: {res.solid_solution_strength[-1]/1e6:.1f} MPa")
        print(f"  Total σ_y:      {res.total_strength[-1]/1e6:.1f} MPa")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for color, T_C in zip(colors, temps):
        res = results_all[T_C]
        t_h = res.time_hours

        axes[0].plot(t_h, res.total_strength / 1e6,
                     color=color, lw=2, label=f"{T_C}°C total")
        axes[0].plot(t_h, res.orowan_strength / 1e6,
                     color=color, lw=1.5, ls='--', label=f"{T_C}°C Orowan")
        axes[0].plot(t_h, res.solid_solution_strength / 1e6,
                     color=color, lw=1.5, ls=':', label=f"{T_C}°C SS")

        for i, phase in enumerate(res.phases):
            axes[1].plot(t_h, res.volume_fraction[:, i] * 100,
                         color=color, lw=2 if i == 0 else 1.5,
                         ls='-' if i == 0 else '--',
                         label=f"{T_C}°C {phase}")

        for i, phase in enumerate(res.phases):
            axes[2].plot(t_h, res.radius[:, i] * 1e9,
                         color=color, lw=2 if i == 0 else 1.5,
                         ls='-' if i == 0 else '--',
                         label=f"{T_C}°C {phase}")

    axes[0].set_xlabel('Aging time (h)')
    axes[0].set_ylabel('Yield strength (MPa)')
    axes[0].set_title('Yield strength vs aging time\n(solid=total, dashed=Orowan, dot=SS)')
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Aging time (h)')
    axes[1].set_ylabel('Volume fraction (%)')
    axes[1].set_title('Precipitate volume fraction')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Aging time (h)')
    axes[2].set_ylabel('Mean radius (nm)')
    axes[2].set_title('Mean precipitate radius')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('A356 aging: 175°C vs 200°C  (kawin strength model, no T-shift)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    out_path = os.path.join(out_dir, 'sigma_y_vs_time.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {out_path}")
    plt.close(fig)
