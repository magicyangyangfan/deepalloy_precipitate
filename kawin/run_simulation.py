"""
Al-Mg-Si and Al-Cu Precipitation Simulation with Yield Strength Prediction

Uses kawin's built-in StrengthModel (Orowan + solid solution)
for yield strength calculation.

Al-Mg-Si precipitate phases: MGSI_B_P (β'), MG5SI6_B_DP (β'')
Al-Cu precipitate phases: THETA_PRIME (θ') — metastable only

For a combined Al-Mg-Si-Cu alloy the two simulators are run independently
and their precipitate strengths are summed; solid solution is summed once
across all solutes (no double-counting).

No temperature shift applied — aging temperature used directly.

Class hierarchy
---------------
Subsystem classes (use these directly for single-system work):
  AlMgSiConfig, AlMgSiResults, AlMgSiSimulator
  AlCuConfig,   AlCuResults,   AlCuSimulator

Combined classes (use for full Al-Mg-Si-Cu alloy):
  AlloyConfig, AlloySimulationResults, PrecipitationSimulator

Unified convenience function (routes automatically based on composition):
  run_simulation(T, t, mg_content=0, si_content=0, cu_content=0)
    -> AlloySimulationResults
  Subsystem-only helpers (return subsystem result types):
    run_almgsi_simulation(T, t, mg, si) -> AlMgSiResults
    run_alcu_simulation(T, t, cu)       -> AlCuResults

Backward-compatible aliases (old names still work):
  SimulatorConfig              → AlMgSiConfig
  AlCuSimulatorConfig          → AlCuConfig
  SimulationResults            → AlMgSiResults
  AlCuSimulationResults        → AlCuResults
  CombinedSimulationResults    → AlloySimulationResults
  AlMgSiPrecipitationSimulator → AlMgSiSimulator
  AlCuPrecipitationSimulator   → AlCuSimulator

NOTE: run_simulation() now returns AlloySimulationResults (was AlMgSiResults).
Old code accessing result.matrix_Mg_wt_pct should use result.almgsi.matrix_Mg_wt_pct.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from kawin.thermo import MulticomponentThermodynamics, BinaryThermodynamics
from kawin.precipitation.PrecipitationParameters import TemperatureParameters
from kawin.precipitation import MatrixParameters, PrecipitateParameters, PrecipitateModel
from kawin.precipitation.coupling import StrengthModel
from kawin.precipitation.coupling import (
    DislocationParameters,
    CoherencyContribution, ModulusContribution,
)
from kawin.precipitation.coupling.Strength import SolidSolutionStrength
from kawin.solver import explicitEulerIterator


# ---------------------------------------------------------------------------
# Phase stoichiometry (for matrix mass balance)
# ---------------------------------------------------------------------------

@dataclass
class AlMgSiPhaseStoichiometry:
    """Phase stoichiometry and molar volume — Al-Mg-Si system."""
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


# Backward-compatible alias
PhaseStoichiometry = AlMgSiPhaseStoichiometry

PHASE_STOICHIOMETRY = {
    'MGSI_B_P':    AlMgSiPhaseStoichiometry(n_Al=0, n_Mg=1.8, n_Si=1,   molar_volume=5e-6),
    'MG5SI6_B_DP': AlMgSiPhaseStoichiometry(n_Al=0, n_Mg=5,   n_Si=6,   molar_volume=5e-6),
    'B_PRIME_L':   AlMgSiPhaseStoichiometry(n_Al=3, n_Mg=9,   n_Si=7,   molar_volume=2e-6),
    'U1_PHASE':    AlMgSiPhaseStoichiometry(n_Al=2, n_Mg=1,   n_Si=2,   molar_volume=3e-6),
    'U2_PHASE':    AlMgSiPhaseStoichiometry(n_Al=1, n_Mg=1,   n_Si=1,   molar_volume=3e-6),
}


@dataclass
class AlCuPhaseStoichiometry:
    """Phase stoichiometry and molar volume — Al-Cu system."""
    n_Al: float = 0.0
    n_Cu: float = 0.0
    molar_volume: float = 5e-6  # m³/mol

    @property
    def total_atoms(self) -> float:
        return self.n_Al + self.n_Cu

    @property
    def x_Cu(self) -> float:
        return self.n_Cu / self.total_atoms if self.total_atoms > 0 else 0.0


ALCU_PHASE_STOICHIOMETRY = {
    'AL2CU':        AlCuPhaseStoichiometry(n_Al=2, n_Cu=1, molar_volume=5e-6),
    'THETA_PRIME':  AlCuPhaseStoichiometry(n_Al=2, n_Cu=1, molar_volume=5e-6),
    'THETA_DPRIME': AlCuPhaseStoichiometry(n_Al=3, n_Cu=1, molar_volume=5e-6),
}


# ---------------------------------------------------------------------------
# Configuration — subsystem classes
# ---------------------------------------------------------------------------

@dataclass
class AlMgSiConfig:
    """Complete Al-Mg-Si simulator configuration."""
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

    # Coherency misfit strain — β'' is semi-coherent with Al, ε ~ 0.02-0.05
    coherency_eps: float = 0.025
    # Precipitate shear modulus — between Al (25 GPa) and Si (160 GPa)
    precipitate_shear_modulus: float = 50e9  # Gp (Pa)

    # Solid solution strengthening weights (Pa · mol/mol)
    # σ_ss = w_Mg * x_Mg + w_Si * x_Si  (x in mole fraction)
    # Derived from k_Mg=18.6 MPa/wt.%, k_Si=9.2 MPa/wt.%
    ss_weight_Mg: float = 1.676e9
    ss_weight_Si: float = 0.958e9


@dataclass
class AlCuConfig:
    """Complete Al-Cu simulator configuration.

    Only the metastable θ' (THETA_PRIME) phase is simulated; the stable θ
    (AL2CU) is excluded per user requirement.  The TDB formation enthalpy for
    THETA_PRIME has been adjusted to −40 500 J/mol so that a proper two-phase
    FCC_A1 + THETA_PRIME restricted equilibrium exists at aging temperatures
    (θ' solvus ~0.5 wt%Cu at 175 °C, ~3 wt%Cu at 300 °C).

    Interfacial energy γ = 0.050 J/m² is the effective spherical-equivalent
    value for the broad-face-dominated θ' platelet; this is calibrated so that
    nucleation occurs in a reasonable time at 150–200 °C.

    PBM bounds 0.5–200 nm must be set after creating PrecipitateModel
    (see solve()) to ensure all PSD bins lie above the Gibbs-Thomson critical
    radius (~3 nm at 175 °C).
    """
    phases: list = field(default_factory=lambda: ['FCC_A1', 'THETA_PRIME'])
    elements: list = field(default_factory=lambda: ['AL', 'CU'])
    matrix_molar_volume: float = 6e-6   # m³/mol

    # Interfacial energy (J/m²)
    # First-principles DFT (LDA/GGA) T = 0 K reference values (Wolverton 2004):
    #   γ_coherent (broad face)  ≈ 0.170–0.190 J/m²
    #   γ_semi-coherent (edge)   ≈ 0.520–0.600 J/m²
    #
    # Why we cannot use γ_coherent = 0.19 J/m² directly in this KWN model:
    # The CALPHAD restricted-equilibrium driving force for θ' is ≈ 2050 J/mol
    # at 175 °C (limited by metastability: θ' < θ in stability by definition).
    # Classical CNT gives ΔG*/kT ≈ 440 with γ = 0.19 J/m² → zero nucleation.
    # Even γ = 0.10 J/m² (ΔG*/kT ≈ 64) yields no measurable precipitation in
    # 100 h.  In practice θ' nucleates heterogeneously on GP zones / dislocations
    # with an effective barrier far below the homogeneous CNT value.
    # KWN models account for this by fitting an effective γ to experimental
    # nucleation onset, typically 0.05–0.10 J/m².  We use 0.05 J/m² to obtain
    # physically reasonable nucleation kinetics at these aging temperatures.
    gamma: dict = field(default_factory=lambda: {
        'THETA_PRIME': 0.05,
    })

    # Dislocation parameters for Al matrix
    shear_modulus: float = 25.4e9    # G (Pa)
    burgers_vector: float = 0.286e-9 # b (m)
    poisson_ratio: float = 0.34

    # θ' is semi-coherent: misfit strain is small (~0.005) due to partial
    # stress relaxation at the coherent broad faces
    coherency_eps: float = 0.005
    # θ' shear modulus — similar to Al (Al₂Cu ≈ 40 GPa)
    precipitate_shear_modulus: float = 40e9  # Gp (Pa)

    # PBM size range for θ' particles
    pbm_min: float = 5e-10   # 0.5 nm (below critical radius)
    pbm_max: float = 2e-7    # 200 nm (covers long-time coarsening)
    pbm_bins: int  = 150

    # Solid solution strengthening weight for Cu in Al
    # k_Cu ≈ 13.3 MPa/wt.%  →  w_Cu = k_Cu × (M_Cu/M_Al × 100) × 1e6
    #   = 13.3 × (63.546/26.982 × 100) × 1e6 ≈ 3.13e9 Pa/(mol frac)
    ss_weight_Cu: float = 3.132e9


@dataclass
class AlloyConfig:
    """Combined Al-Mg-Si-Cu simulator configuration.

    Wraps per-subsystem configs.  Pass to PrecipitationSimulator for a
    fully combined Al-Mg-Si-Cu simulation.
    """
    almgsi: AlMgSiConfig = field(default_factory=AlMgSiConfig)
    alcu: AlCuConfig     = field(default_factory=AlCuConfig)


# ---------------------------------------------------------------------------
# Results containers — subsystem classes
# ---------------------------------------------------------------------------

@dataclass
class AlMgSiResults:
    """Al-Mg-Si simulation results container."""
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


@dataclass
class AlCuResults:
    """Al-Cu simulation results container."""
    time: np.ndarray
    phases: list
    radius: np.ndarray
    diameter: np.ndarray
    volume_fraction: np.ndarray
    precipitate_density: np.ndarray
    matrix_Cu_wt_pct: np.ndarray
    solid_solution_strength: np.ndarray
    orowan_strength: np.ndarray
    total_strength: np.ndarray

    @property
    def time_hours(self) -> np.ndarray:
        return self.time / 3600


# ---------------------------------------------------------------------------
# Combined results
# ---------------------------------------------------------------------------

@dataclass
class AlloySimulationResults:
    """
    Combined results for an Al-Mg-Si-Cu alloy (one or both subsystems).

    almgsi and alcu are None when the corresponding subsystem was not run
    (e.g. alloy has no Cu → alcu is None).

    Precipitate strengths are additive (independent precipitation).
    Solid solution is summed once across all solutes — NOT added twice.
    """
    time: np.ndarray
    almgsi: Optional[AlMgSiResults]     # None if Al-Mg-Si not simulated
    alcu: Optional[AlCuResults]          # None if Al-Cu not simulated
    precipitate_strength: np.ndarray
    solid_solution_strength: np.ndarray
    total_strength: np.ndarray

    @property
    def time_hours(self) -> np.ndarray:
        return self.time / 3600


def combine_strength_results(
    almgsi: Optional[AlMgSiResults],
    alcu: Optional[AlCuResults],
) -> AlloySimulationResults:
    """
    Combine Al-Mg-Si and/or Al-Cu results into AlloySimulationResults.

    Either argument may be None (single-subsystem alloy).  When both are
    present, AlMgSi contributions are interpolated onto the AlCu time grid.

    Total combined strength:
        σ_total = σ_prec_AlMgSi + σ_prec_AlCu + σ_SS_AlMgSi + σ_SS_AlCu

    Each SS term covers only its own solutes (Mg+Si or Cu), so there is no
    double-counting even when both subsystems are active.
    """
    if almgsi is None and alcu is None:
        raise ValueError("At least one subsystem result must be provided.")

    if almgsi is not None and alcu is not None:
        # Use AlCu time grid as reference; interpolate AlMgSi onto it
        t_ref = alcu.time
        prec = np.interp(t_ref, almgsi.time, almgsi.orowan_strength) + alcu.orowan_strength
        ss   = np.interp(t_ref, almgsi.time, almgsi.solid_solution_strength) + alcu.solid_solution_strength
    elif almgsi is not None:
        t_ref = almgsi.time
        prec  = almgsi.orowan_strength.copy()
        ss    = almgsi.solid_solution_strength.copy()
    else:  # only alcu
        t_ref = alcu.time
        prec  = alcu.orowan_strength.copy()
        ss    = alcu.solid_solution_strength.copy()

    return AlloySimulationResults(
        time=t_ref,
        almgsi=almgsi,
        alcu=alcu,
        precipitate_strength=prec,
        solid_solution_strength=ss,
        total_strength=prec + ss,
    )


# ---------------------------------------------------------------------------
# Al-Mg-Si Simulator
# ---------------------------------------------------------------------------

class AlMgSiSimulator:
    """
    Al-Mg-Si precipitation simulator using kawin's built-in strength model.

    Parameters
    ----------
    tdb_file : str
        Path to thermodynamic database (.tdb)
    init_composition : list[float]
        Initial [Mg, Si] in mole fraction
    aging_temperature : float
        Aging temperature in Celsius (used directly, no shift applied)
    aging_time : float
        Aging time in hours
    config : AlMgSiConfig, optional
    """

    def __init__(
        self,
        tdb_file: str,
        init_composition: list,
        aging_temperature: float,
        aging_time: float,
        config: Optional[AlMgSiConfig] = None,
    ):
        self.tdb_file = tdb_file
        self.init_composition = init_composition
        self.aging_time_hours = aging_time
        self.aging_temperature = aging_temperature
        self.config = config or AlMgSiConfig()

        temp_K = aging_temperature + 273.15
        self._temperature = TemperatureParameters(
            [0, aging_time], [temp_K, temp_K]
        )

        self._therm = MulticomponentThermodynamics(
            tdb_file, self.config.elements, self.config.phases
        )
        self._model = None
        self._strength_model = None

    def solve(self, verbose: bool = True) -> AlMgSiResults:
        """Run simulation and return results."""
        matrix = MatrixParameters(['MG', 'SI'])
        matrix.initComposition = self.init_composition
        matrix.volume.setVolume(1e-5, 'VM', 4)

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
            print(f"Running Al-Mg-Si simulation: {self.aging_time_hours}h at {self.aging_temperature}°C")

        self._model.solve(
            self.aging_time_hours * 3600,
            iterator=explicitEulerIterator,
            verbose=verbose,
            vIt=10000,
        )

        return self._extract_results()

    def _extract_results(self) -> AlMgSiResults:
        data = self._model.data
        phases = list(self._model.phases)
        radius = data.Ravg
        volume_fraction = data.volFrac
        diameter = 2 * radius * 1e9

        x_Mg, x_Si = self._calc_matrix_composition(volume_fraction, phases)
        matrix_Mg_wt_pct, matrix_Si_wt_pct = self._at_to_wt(x_Mg, x_Si)

        total, prec_strength, ss_strength, _ = self._strength_model.totalStrength(
            self._model, returnContributions=True
        )

        return AlMgSiResults(
            time=data.time,
            phases=phases,
            radius=radius,
            diameter=diameter,
            volume_fraction=volume_fraction,
            precipitate_density=data.precipitateDensity,
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
# Al-Cu Simulator
# ---------------------------------------------------------------------------

class AlCuSimulator:
    """
    Al-Cu precipitation simulator using kawin's built-in strength model.

    Simulates the metastable θ' (THETA_PRIME) phase during artificial aging.

    Parameters
    ----------
    tdb_file : str
        Path to AlCu thermodynamic database (.tdb)
    init_composition : list[float]
        Initial [Cu] in mole fraction.
        Default: Al-4.5 wt.% Cu → [0.01961]
    aging_temperature : float
        Aging temperature in Celsius (used directly, no shift applied)
    aging_time : float
        Aging time in hours
    config : AlCuConfig, optional
    """

    DEFAULT_CU_MOLFRAC: float = 0.01961   # Al-4.5 wt.% Cu

    def __init__(
        self,
        tdb_file: str,
        init_composition: Optional[list] = None,
        aging_temperature: float = 175.0,
        aging_time: float = 100.0,
        config: Optional[AlCuConfig] = None,
    ):
        self.tdb_file = tdb_file
        self.init_composition = (
            init_composition if init_composition is not None
            else [self.DEFAULT_CU_MOLFRAC]
        )
        self.aging_time_hours = aging_time
        self.aging_temperature = aging_temperature
        self.config = config or AlCuConfig()

        temp_K = aging_temperature + 273.15
        self._temperature = TemperatureParameters(
            [0, aging_time], [temp_K, temp_K]
        )

        # Al-Cu is a binary system: use BinaryThermodynamics so that
        # KWNEuler's getInterfacialComposition(T, gExtra) calls are matched correctly.
        self._therm = BinaryThermodynamics(
            tdb_file, self.config.elements, self.config.phases
        )
        self._model = None
        self._strength_model = None

    def solve(self, verbose: bool = True) -> AlCuResults:
        """Run simulation and return results."""
        matrix = MatrixParameters(['CU'])
        matrix.initComposition = self.init_composition
        matrix.volume.setVolume(1e-5, 'VM', 4)

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

        # Extend PBM bounds so that all PSD bins lie above the Gibbs-Thomson
        # critical radius.  Default kawin bounds (0.1–1 nm) are too small for
        # θ' (R* ≈ 3 nm at 175 °C); without this, all bins return -1 and the
        # nucleation / growth loop stalls.
        for p in precipitates:
            self._model.setPBMParameters(
                cMin=self.config.pbm_min,
                cMax=self.config.pbm_max,
                bins=self.config.pbm_bins,
                phase=p.phase,
            )

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
            'CU': self.config.ss_weight_Cu,
        })
        self._strength_model = StrengthModel(
            phases=precipitates,
            contributions=contributions,
            dislocations=dislocations,
            ssModel=ss_model,
        )
        self._model.addCouplingModel(self._strength_model)

        if verbose:
            x_Cu = self.init_composition[0]
            wt_Cu = x_Cu * 63.546 / ((1 - x_Cu) * 26.982 + x_Cu * 63.546) * 100
            print(f"Running Al-Cu simulation: {self.aging_time_hours}h at "
                  f"{self.aging_temperature}°C  (Al-{wt_Cu:.1f}wt.%Cu)")

        self._model.solve(
            self.aging_time_hours * 3600,
            iterator=explicitEulerIterator,
            verbose=verbose,
            vIt=10000,
        )

        return self._extract_results()

    def _extract_results(self) -> AlCuResults:
        data = self._model.data
        phases = list(self._model.phases)
        radius = data.Ravg
        volume_fraction = data.volFrac
        diameter = 2 * radius * 1e9

        x_Cu = self._calc_matrix_composition(volume_fraction, phases)
        matrix_Cu_wt_pct = self._at_to_wt_cu(x_Cu)

        total, prec_strength, ss_strength, _ = self._strength_model.totalStrength(
            self._model, returnContributions=True
        )

        return AlCuResults(
            time=data.time,
            phases=phases,
            radius=radius,
            diameter=diameter,
            volume_fraction=volume_fraction,
            precipitate_density=data.precipitateDensity,
            matrix_Cu_wt_pct=matrix_Cu_wt_pct,
            solid_solution_strength=ss_strength,
            orowan_strength=prec_strength,
            total_strength=total,
        )

    def _calc_matrix_composition(self, volume_fraction, phases) -> np.ndarray:
        x_Cu_0 = self.init_composition[0]
        Vm_alpha = self.config.matrix_molar_volume
        n_times = volume_fraction.shape[0]
        Cu_consumed = np.zeros(n_times)
        f_total = np.zeros(n_times)

        for i, phase in enumerate(phases):
            if phase in ALCU_PHASE_STOICHIOMETRY:
                s = ALCU_PHASE_STOICHIOMETRY[phase]
                f_p = volume_fraction[:, i]
                factor = f_p * (Vm_alpha / s.molar_volume)
                Cu_consumed += s.x_Cu * factor
                f_total += f_p

        with np.errstate(divide='ignore', invalid='ignore'):
            x_Cu = np.maximum((x_Cu_0 - Cu_consumed) / (1 - f_total), 0)
            x_Cu = np.where(np.isfinite(x_Cu), x_Cu, x_Cu_0)
        return x_Cu

    def _at_to_wt_cu(self, x_Cu: np.ndarray) -> np.ndarray:
        M_Al, M_Cu = 26.98154, 63.546
        x_Al = np.maximum(1 - x_Cu, 0)
        total = x_Al * M_Al + x_Cu * M_Cu
        with np.errstate(divide='ignore', invalid='ignore'):
            wt_Cu = np.where(total > 0, x_Cu * M_Cu / total * 100, 0)
        return wt_Cu


# ---------------------------------------------------------------------------
# Combined Al-Mg-Si-Cu Simulator
# ---------------------------------------------------------------------------

class PrecipitationSimulator:
    """
    Combined Al-Mg-Si-Cu precipitation simulator.

    Automatically activates each subsystem based on which compositions are
    provided.  Results are combined into AlloySimulationResults.

    Parameters
    ----------
    almgsi_tdb : str
        Path to Al-Mg-Si thermodynamic database (.tdb).
        Required when almgsi_composition is given.
    alcu_tdb : str
        Path to Al-Cu thermodynamic database (.tdb).
        Required when cu_composition is given.
    almgsi_composition : list[float] or None
        Initial [Mg, Si] mole fractions.  Pass None to skip Al-Mg-Si.
    cu_composition : list[float] or None
        Initial [Cu] mole fraction.  Pass None to skip Al-Cu.
    aging_temperature : float
        Aging temperature in Celsius
    aging_time : float
        Aging time in hours
    config : AlloyConfig, optional
    """

    def __init__(
        self,
        aging_temperature: float,
        aging_time: float,
        almgsi_tdb: Optional[str] = None,
        alcu_tdb: Optional[str] = None,
        almgsi_composition: Optional[list] = None,
        cu_composition: Optional[list] = None,
        config: Optional[AlloyConfig] = None,
    ):
        if almgsi_composition is None and cu_composition is None:
            raise ValueError("Provide at least one of almgsi_composition or cu_composition.")

        cfg = config or AlloyConfig()
        self._almgsi_sim: Optional[AlMgSiSimulator] = None
        self._alcu_sim: Optional[AlCuSimulator] = None

        if almgsi_composition is not None:
            if almgsi_tdb is None:
                raise ValueError("almgsi_tdb is required when almgsi_composition is given.")
            self._almgsi_sim = AlMgSiSimulator(
                tdb_file=almgsi_tdb,
                init_composition=almgsi_composition,
                aging_temperature=aging_temperature,
                aging_time=aging_time,
                config=cfg.almgsi,
            )

        if cu_composition is not None:
            if alcu_tdb is None:
                raise ValueError("alcu_tdb is required when cu_composition is given.")
            self._alcu_sim = AlCuSimulator(
                tdb_file=alcu_tdb,
                init_composition=cu_composition,
                aging_temperature=aging_temperature,
                aging_time=aging_time,
                config=cfg.alcu,
            )

    def solve(self, verbose: bool = True) -> AlloySimulationResults:
        """Run active subsystem simulations and return combined results."""
        almgsi_res = self._almgsi_sim.solve(verbose=verbose) if self._almgsi_sim else None
        alcu_res   = self._alcu_sim.solve(verbose=verbose)   if self._alcu_sim   else None
        return combine_strength_results(almgsi_res, alcu_res)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def _resolve_tdb(name: str) -> str:
    """Return absolute path to a TDB file in the project data/ directory."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, 'data', name)


# Atomic masses (g/mol)
_M_Al = 26.982
_M_Mg = 24.305
_M_Si = 28.086
_M_Cu = 63.546


def _convert_from_wt_to_at(
    Si: float, Mg: float, Cu: float
) -> tuple[float, float, float]:
    """Convert wt% to mole fractions for Al-Mg-Si-Cu alloy (Al balance).

    Returns
    -------
    (x_Mg, x_Si, x_Cu) : mole fractions
    """
    w_Al = 100.0 - Si - Mg - Cu
    n_Al = w_Al / _M_Al
    n_Mg = Mg / _M_Mg
    n_Si = Si / _M_Si
    n_Cu = Cu / _M_Cu
    total = n_Al + n_Mg + n_Si + n_Cu
    return n_Mg / total, n_Si / total, n_Cu / total


def get_composition_in_FCC(
    Si: float,
    Mg: float,
    Cu: float,
    eff_factor: float = 0.8,
) -> tuple[list[float] | None, list[float] | None]:
    """Convert alloy composition (wt%) to the mole fractions that participate in FCC precipitation.
    Ignore other alloying elements (e.g. Fe, Mn) that do not contribute to precipitation.

    Applies an efficiency factor to account for the fraction of solute that
    actually participates in precipitation. High Si (>= 0.8 wt%) is also
    scaled by eff_factor before conversion.

    Parameters
    ----------
    Si, Mg, Cu : float
        Alloy element contents in wt%.
    eff_factor : float
        Fraction of solute assumed to participate in precipitation (default 0.8).

    Returns
    -------
    almgsi_composition : list[float] | None
        [Mg, Si] mole fractions for Al-Mg-Si subsystem, or None if both are zero.
    cu_composition : list[float] | None
        [Cu] mole fraction for Al-Cu subsystem, or None if Cu is zero.

    Raises
    ------
    ValueError
        If Cu == 0 or both Si and Mg are zero (no precipitation-capable solute).
    """
    if Cu == 0 or (Si == 0 and Mg == 0):
        raise ValueError("missing critical solution elements")

    if Si >= 0.8:
        Si = Si * eff_factor
    Mg = Mg * eff_factor
    Cu = Cu * eff_factor

    x_Mg, x_Si, x_Cu = _convert_from_wt_to_at(Si, Mg, Cu)

    almgsi_composition = [x_Mg, x_Si] if (x_Mg > 0 or x_Si > 0) else None
    cu_composition = [x_Cu] if x_Cu > 0 else None

    return almgsi_composition, cu_composition


def run_simulation(
    aging_temperature: float,
    aging_time: float,
    mg_content: float = 0.0,
    si_content: float = 0.0,
    cu_content: float = 0.0,
    verbose: bool = True,
) -> AlloySimulationResults:
    """
    Unified simulation entry point — routes to active subsystems automatically.

    Detects which subsystems to run based on nonzero solute content and
    always returns AlloySimulationResults.  Subsystem results are accessible
    via result.almgsi and result.alcu (None when not simulated).

    Parameters
    ----------
    aging_temperature : float
        Aging temperature in Celsius
    aging_time : float
        Aging time in hours
    mg_content : float
        Mg in mole fraction (skip Al-Mg-Si subsystem if 0)
    si_content : float
        Si in mole fraction (skip Al-Mg-Si subsystem if 0)
    cu_content : float
        Cu in mole fraction (skip Al-Cu subsystem if 0)

    Examples
    --------
    Al-4.5Cu-0.3Mg-1Si at 175°C for 100h:
        run_simulation(175, 100, mg_content=0.003330, si_content=0.009607,
                       cu_content=0.01961)

    Al-4.5Cu only:
        run_simulation(175, 100, cu_content=0.01961)

    Al-Mg-Si only:
        run_simulation(175, 100, mg_content=0.003330, si_content=0.009607)
    """
    run_almgsi = mg_content > 0 or si_content > 0
    run_alcu   = cu_content > 0

    if not run_almgsi and not run_alcu:
        raise ValueError("At least one of mg_content, si_content, or cu_content must be > 0.")

    sim = PrecipitationSimulator(
        aging_temperature=aging_temperature,
        aging_time=aging_time,
        almgsi_tdb=_resolve_tdb('AlMgSi.tdb') if run_almgsi else None,
        alcu_tdb=_resolve_tdb('AlCu.tdb')     if run_alcu   else None,
        almgsi_composition=[mg_content, si_content] if run_almgsi else None,
        cu_composition=[cu_content]                 if run_alcu   else None,
    )
    return sim.solve(verbose=verbose)


def run_almgsi_simulation(
    aging_temperature: float,
    aging_time: float,
    mg_content: float = 0.003330,   # A356-type: 0.3 wt.% Mg
    si_content: float = 0.009607,   # A356-type: 1.0 wt.% Si
    verbose: bool = True,
) -> AlMgSiResults:
    """Run Al-Mg-Si subsystem only; returns AlMgSiResults directly."""
    sim = AlMgSiSimulator(
        tdb_file=_resolve_tdb('AlMgSi.tdb'),
        init_composition=[mg_content, si_content],
        aging_temperature=aging_temperature,
        aging_time=aging_time,
    )
    return sim.solve(verbose=verbose)


def run_alcu_simulation(
    aging_temperature: float,
    aging_time: float,
    cu_content: float = AlCuSimulator.DEFAULT_CU_MOLFRAC,  # Al-4.5 wt.% Cu
    verbose: bool = True,
) -> AlCuResults:
    """Run Al-Cu subsystem only; returns AlCuResults directly."""
    sim = AlCuSimulator(
        tdb_file=_resolve_tdb('AlCu.tdb'),
        init_composition=[cu_content],
        aging_temperature=aging_temperature,
        aging_time=aging_time,
    )
    return sim.solve(verbose=verbose)


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# Keep old names working so existing code does not break.
# NOTE: run_simulation() return type changed from AlMgSiResults to
#       AlloySimulationResults.  Use run_almgsi_simulation() to get the old
#       return type, or access result.almgsi for the Al-Mg-Si sub-results.
# ---------------------------------------------------------------------------

# Config aliases
SimulatorConfig     = AlMgSiConfig
AlCuSimulatorConfig = AlCuConfig

# Results aliases
SimulationResults         = AlMgSiResults
AlCuSimulationResults     = AlCuResults
CombinedSimulationResults = AlloySimulationResults

# Simulator aliases
AlMgSiPrecipitationSimulator = AlMgSiSimulator
AlCuPrecipitationSimulator   = AlCuSimulator


# ---------------------------------------------------------------------------
# Main: run Al-Cu at 175°C and 200°C and plot results
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'kawin_examples',
    )
    os.makedirs(out_dir, exist_ok=True)

    T_C          = 175.0   # aging temperature (°C)
    AGING_TIME_H = 100.0   # aging time (h)

    # Compositions (mole fractions)
    # Al-0.3Mg-1Si:      Mg=0.3 wt.% → 0.003330, Si=1.0 wt.% → 0.009607
    # Al-4.5Cu:          Cu=4.5 wt.% → 0.01961
    # Al-4.5Cu-0.5Mg-1Si: Cu=4.5 wt.% → 0.01961,
    #                      Mg=0.5 wt.% → 0.005551, Si=1.0 wt.% → 0.009604
    examples = [
        dict(label='Al-0.3Mg-1Si',       mg=0.003330, si=0.009607, cu=0.0),
        dict(label='Al-4.5Cu',            mg=0.0,      si=0.0,      cu=0.01961),
        dict(label='Al-4.5Cu-0.5Mg-1Si', mg=0.005551, si=0.009604, cu=0.01961),
    ]
    colors = ['steelblue', 'tomato', 'seagreen']

    print(f"\nRunning 3 examples at {T_C}°C for {AGING_TIME_H}h")
    results = {}
    for ex in examples:
        print(f"\n{'='*60}")
        print(f"Alloy: {ex['label']}")
        res = run_simulation(
            T_C, AGING_TIME_H,
            mg_content=ex['mg'],
            si_content=ex['si'],
            cu_content=ex['cu'],
        )
        results[ex['label']] = res
        print(f"  Precipitate σ: {res.precipitate_strength[-1]/1e6:.1f} MPa")
        print(f"  Solid solution σ: {res.solid_solution_strength[-1]/1e6:.1f} MPa")
        print(f"  Total σ_y:     {res.total_strength[-1]/1e6:.1f} MPa")

    # ---- Plot: 3 rows × 2 cols — one row per alloy --------------------------
    # Col 0: yield strength breakdown (total / precipitate / SS)
    # Col 1: precipitate volume fraction (each active phase)
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    for row, (ex, color) in enumerate(zip(examples, colors)):
        label = ex['label']
        res   = results[label]
        t_h   = res.time_hours
        ax_s  = axes[row, 0]   # strength
        ax_v  = axes[row, 1]   # volume fraction

        # Yield strength
        ax_s.plot(t_h, res.total_strength / 1e6,
                  color=color, lw=2, label='Total')
        ax_s.plot(t_h, res.precipitate_strength / 1e6,
                  color=color, lw=1.5, ls='--', label='Precipitate')
        ax_s.plot(t_h, res.solid_solution_strength / 1e6,
                  color=color, lw=1.5, ls=':', label='Solid solution')
        ax_s.set_title(f'{label}\nYield strength')
        ax_s.set_xlabel('Aging time (h)')
        ax_s.set_ylabel('Strength (MPa)')
        ax_s.legend(fontsize=8)
        ax_s.grid(True, alpha=0.3)

        # Volume fraction — plot each active subsystem's phases
        if res.almgsi is not None:
            t_ms = res.almgsi.time_hours
            for i, phase in enumerate(res.almgsi.phases):
                ls = ['-', '--', ':'][i % 3]
                ax_v.plot(t_ms, res.almgsi.volume_fraction[:, i] * 100,
                          color='cornflowerblue', lw=1.5, ls=ls, label=phase)
        if res.alcu is not None:
            t_cu = res.alcu.time_hours
            for i, phase in enumerate(res.alcu.phases):
                ls = ['-', '--', ':'][i % 3]
                ax_v.plot(t_cu, res.alcu.volume_fraction[:, i] * 100,
                          color='salmon', lw=1.5, ls=ls, label=phase)
        ax_v.set_title(f'{label}\nPrecipitate volume fraction')
        ax_v.set_xlabel('Aging time (h)')
        ax_v.set_ylabel('Volume fraction (%)')
        ax_v.legend(fontsize=8)
        ax_v.grid(True, alpha=0.3)

    fig.suptitle(
        f'Precipitation strengthening — 3 alloy examples at {T_C}°C / {AGING_TIME_H}h\n'
        '(blue shades = Al-Mg-Si phases, red shades = Al-Cu phases)',
        fontsize=11, fontweight='bold',
    )
    fig.tight_layout()
    out_path = os.path.join(out_dir, 'alloy_examples_sigma_y.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved → {out_path}")
    plt.close(fig)

    # ---- Comparison plot: total strength for all 3 alloys -------------------
    fig2, ax = plt.subplots(figsize=(8, 5))
    for ex, color in zip(examples, colors):
        res = results[ex['label']]
        ax.plot(res.time_hours, res.total_strength / 1e6,
                color=color, lw=2, label=ex['label'])
    ax.set_xlabel('Aging time (h)')
    ax.set_ylabel('Total yield strength (MPa)')
    ax.set_title(f'Yield strength comparison — {T_C}°C aging')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    out_path2 = os.path.join(out_dir, 'alloy_examples_comparison.png')
    fig2.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f"Plot saved → {out_path2}")
    plt.close(fig2)
