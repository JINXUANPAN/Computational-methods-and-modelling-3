#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solar Still Model - Based on Vasava et al. (2023)
==============================================
Mathematical modelling matching paper equations with area optimization

This module implements a dynamic solar still model based on Vasava et al. (2023).
It performs the following tasks:
1. Load hourly solar irradiance data from Excel.
2. Fit a 3rd-order polynomial regression.
3. Solve coupled ODEs for Tw, Tg, and M_col.
4. Perform area optimization for target yield.
5. Plot temperature profiles and water production.

Note: Safeguards are included to prevent division by zero errors in calculations.
Although these cases are unlikely in practical scenarios, they ensure numerical stability.
"""

# Importing libraries used 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# PHYSICAL CONSTANTS AND GLOBAL PARAMETERS
# =============================================================================

# Physical constants
WATER_DEPTH = 0.02              # m (20 mm) - from paper
GLASS_THICKNESS = 0.004         # m (4 mm) - from paper
RHO_WATER = 1000.0              # kg/m³
RHO_GLASS = 2500.0              # kg/m³
C_P_WATER = 4200.0              # J/(kg·K) - paper value
C_P_GLASS = 750.0               # J/(kg·K)
EPSILON_W = 0.95                # Water emissivity
EPSILON_G = 0.90                # Glass emissivity
SIGMA = 5.67e-8                 # Stefan-Boltzmann (W/m²·K⁴)
L_V = 2.26e6                    # Latent heat (J/kg) - paper value
ALPHA_WATER = 0.85              # Basin absorptivity
ALPHA_GLASS = 0.05              # Glass absorptivity
TAU_GLASS = 0.90                # Glass transmittance
H_GA = 5.0                      # Convective glass-air (W/m²·K)
ETA_COLL = 0.80                 # Collection efficiency

DT_SEC = 60                     # Timestep (s)
POLY_ORDER = 3                  # Polynomial regression order

# Environmental conditions - CONSTANT TEMPERATURE VERSION
T_AMB_C = 30.0                  # Ambient temperature (°C) - CONSTANT
T_SKY_C = 5.0                   # Sky temperature (°C) - CONSTANT (25K below ambient)

# Configuration - File paths relative to script location
OUTPUT_DIR = os.path.join(script_dir, "CSV Files and Plots Generated")
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create directory if it doesn't exist
EXCEL_FILE = os.path.join(OUTPUT_DIR, "solar_irradiance_data.xlsx")
PERCENTAGE_FILE = os.path.join(script_dir, "hourly_percentage_irradiance.csv")
TOTAL_IRRADIANCE = 3852         # Total daily irradiance (W/m²·day)

# Globals to hold time and irradiance arrays
t_seconds = None
t_hours = None
G_time = None
t_start_hr = None
t_end_hr = None
poly_func = None

# =============================================================================
# LOADING AND PREPROCESSING SOLAR IRRADIANCE DATA
# =============================================================================

def load_percentage_data(filepath):
    """
    Load hourly percentage irradiance data from Excel file.
    
    Parameters
    ----------
    filepath : str
        Path to Excel file containing percentage distribution data.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'Hour' and 'Percentage of Daily Total' columns.
    
    Raises
    ------
    FileNotFoundError
        If the specified Excel file does not exist.
    ValueError
        If required columns are missing from the Excel sheet.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Excel file not found: {filepath}")
    
    # Read from the "Percentage Distribution" sheet
    df = pd.read_excel(filepath, sheet_name='Percentage Distribution')
    if not {'Hour', 'Percentage of Daily Total'}.issubset(df.columns):
        raise ValueError("Excel sheet must contain columns: 'Hour' and 'Percentage of Daily Total'")
    return df

def compute_hourly_irradiance(df, total_irradiance):
    """
    Apply total daily irradiance to percentage distribution.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with hourly percentage data.
    total_irradiance : float
        Total daily irradiance (W/m²·day).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with computed hourly irradiance values.
    """
    df = df.copy()
    df["Estimated Irradiance (W/m2)"] = (df["Percentage of Daily Total"] / 100) * total_irradiance
    df["Hour_float"] = df["Hour"].apply(lambda x: int(x.split(":")[0]) + int(x.split(":")[1]) / 60)
    df = df.sort_values("Hour_float")
    return df

# Load percentage data
print("\n" + "="*60)
print("Loading preprocessed irradiance data...")
print("="*60)

# Check if output directory exists
if not os.path.exists(OUTPUT_DIR):
    raise FileNotFoundError(f"Output directory not found: {OUTPUT_DIR}. Please run the previous script first.")

df_pct = load_percentage_data(EXCEL_FILE)
df_irr = compute_hourly_irradiance(df_pct, TOTAL_IRRADIANCE)

print(f"\nHourly Irradiance Estimated from Total = {TOTAL_IRRADIANCE:.0f} W/m²·day")
print("="*60)
print(df_irr[["Hour", "Percentage of Daily Total", "Estimated Irradiance (W/m2)"]].to_string(index=False, float_format="%.2f"))
print("="*60)

# =============================================================================
# REGRESSION MODEL (INTERPOLATING DATA TO FORM CONTINUOUS FUNCTION)
# =============================================================================

# Loading data from Excel file
x = df_irr["Hour_float"].values
y = df_irr["Estimated Irradiance (W/m2)"].values

# 3rd-order polynomial regression
coeffs = np.polyfit(x, y, 3)
poly_func = np.poly1d(coeffs)

x_smooth = np.linspace(x.min(), x.max(), 300)
y_smooth = poly_func(x_smooth)

print("Polynomial regression:")
print(f"y = {coeffs[0]:.3f}x³ + {coeffs[1]:.3f}x² + {coeffs[2]:.3f}x + {coeffs[3]:.3f}")
print("Average solar irradiance =", np.mean(np.clip(y_smooth, 0, None)))
print("Maximum solar irradiance =", np.max(y_smooth))

# Plotting regression curve
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label="Original data")
plt.plot(x_smooth, y_smooth, color='blue', label="Polynomial regression curve")
equation_text = f"y = {coeffs[0]:.2f}x³ + {coeffs[1]:.2f}x² + {coeffs[2]:.2f}x + {coeffs[3]:.2f}"
plt.text((x.min() + x.max())/2, y.min() + (y.max() - y.min())*0.2,
         equation_text, color="blue", fontsize=12, ha='center',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="blue"))
plt.xlabel("Hour")
plt.ylabel(r"Solar Irradiance (W/m$^2$)")
plt.title("Continuous irradiance function using regression model")
plt.legend()
plt.grid(True)
# Save plot
regression_plot_path = os.path.join(OUTPUT_DIR, "01_Solar_Irradiance_Regression.png")
plt.savefig(regression_plot_path, dpi=300, bbox_inches='tight')
print(f"Saved: {regression_plot_path}")
plt.show(block=False)  # Non-blocking: allows script to continue

# Initialize time arrays for simulation
t_start_hr = x.min()
t_end_hr = x.max()
t_hours = np.arange(t_start_hr, t_end_hr + DT_SEC/3600, DT_SEC/3600)
t_seconds = t_hours * 3600.0
G_time = np.clip(poly_func(t_hours), 0.0, None)

print(f"Simulation window: {t_start_hr:.1f}h - {t_end_hr:.1f}h")

print("\n" + "="*60)
print("Temperature: CONSTANT")
print("="*60)
print(f"Ambient Temperature: {T_AMB_C:.2f}°C")
print(f"Sky Temperature:     {T_SKY_C:.2f}°C")
print("="*60)

# =============================================================================
# ODE MODEL: SOLAR STILL DYNAMICS (BASED ON PAPER EQUATIONS)
# =============================================================================

def p_sat(T_celsius):
    """
    Compute saturation vapour pressure for water (paper's Eq. 9 & 10).
    
    Uses the correlation: P_sat = exp(25.317 - 5144/(T + 273))
    where T is the water temperature in Celsius.
    
    Parameters
    ----------
    T_celsius : float or array_like
        Water temperature in degrees Celsius. Values are clipped
        to the range [0, 100] °C for numerical stability.
    
    Returns
    -------
    float or ndarray
        Saturation vapour pressure (Pa) corresponding to the input temperature.
    """
    T_celsius = np.clip(T_celsius, 0.0, 100.0)
    return np.exp(25.317 - 5144.0 / (T_celsius + 273.0))

# Effective emissivity (Paper's Equation 5)
def epsilon_eff():
    """
    Compute effective emissivity between water and glass (paper's Eq. 5).
    
    Formula: ε_eff = 1 / (1/ε_w + 1/ε_g - 1)
    where ε_w and ε_g are water and glass emissivities.
    
    Returns
    -------
    float
        Effective emissivity (dimensionless) between water and glass.
    """
    return 1.0 / (1.0/EPSILON_W + 1.0/EPSILON_G - 1.0)

# Radiative heat transfer coefficient (Paper's Equation 4 - equivalent Kelvin values 
# introduced for simplicity when using Stefan-Boltzmann)
def h_r_wg(Tw_C, Tg_C):
    """
    Compute radiative heat transfer coefficient between water and glass (paper's Eq. 4).
    
    Formula: h_r,w-g = ε_eff × σ × [(T_w+273)² + (T_g+273)²] × (T_w + T_g + 546)
    where temperatures are in Celsius before converting to Kelvin.
    
    Parameters
    ----------
    Tw_C : float
        Water temperature in degrees Celsius.
    Tg_C : float
        Glass temperature in degrees Celsius.
    
    Returns
    -------
    float
        Radiative heat transfer coefficient (W/m²·K).
    """
    Tw_K = Tw_C + 273
    Tg_K = Tg_C + 273
    eps_eff_val = epsilon_eff()
    h_r = eps_eff_val * SIGMA * (Tw_K**2 + Tg_K**2) * (Tw_K + Tg_K)
    return h_r

# Helper function for signed cube root (to handle negative values)
def signed_cuberoot(x):
    """
    Compute the signed cube root of ``x``.

    This helper function is used to safely evaluate correlations that
    require a cube root, while preserving the sign of the input for
    negative values.

    Parameters
    ----------
    x : float or array_like
        Input value(s) for which the signed cube root is required.

    Returns
    -------
    float or ndarray
        Signed cube root of ``x``, i.e. ``sign(x) * |x|**(1/3)``.
    """
    return np.sign(x) * (np.abs(x) ** (1.0/3.0))

# Convective heat transfer coefficient (Paper's Equation 8)
def h_c_wg(Tw_C, Tg_C):
    """
    Compute convective heat transfer coefficient between water and glass (paper's Eq. 8).
    
    Formula: h_c,w-g = 0.884 × [(T_w - T_g) + (P_w - P_g) × (T_w+273) / (27230 - P_w)]^(1/3)
    where P_w and P_g are saturation pressures at water and glass temperatures.
    
    If T_w ≤ T_g or the combined term is non-positive, returns minimum value
    for numerical stability.
    
    Parameters
    ----------
    Tw_C : float
        Water temperature in degrees Celsius.
    Tg_C : float
        Glass temperature in degrees Celsius.
    
    Returns
    -------
    h_c : float
        Convective heat transfer coefficient (W/m²·K).
    P_w : float
        Saturation pressure at water temperature (Pa).
    P_g : float
        Saturation pressure at glass temperature (Pa).
    """
    if Tw_C <= Tg_C:  # No convection if water temp <= glass temp
        return 0.1, p_sat(Tw_C), p_sat(Tg_C)
    
    P_w = p_sat(Tw_C)
    P_g = p_sat(Tg_C)
    
    term1 = Tw_C - Tg_C
    term2 = (P_w - P_g) * (Tw_C + 273) / (2.7230e4 - P_w + 1e-9)  # Avoid division by zero
    combined = term1 + term2
    
    if combined <= 0:
        return 0.1, P_w, P_g
    
    h_c = 0.884 * signed_cuberoot(combined)
    return max(h_c, 0.1), P_w, P_g

# Evaporative heat transfer coefficient (Paper's Equation 12)
def h_e_wg(Tw_C, Tg_C, h_c, P_w, P_g):
    """
    Compute evaporative heat transfer coefficient between water and glass (paper's Eq. 12).
    
    Formula: h_e,w-g = 16.273×10⁻³ × h_c × (P_w - P_g) / (T_w - T_g)
    
    When temperature difference is extremely small, returns 0.0 to avoid
    division-by-zero issues.
    
    Parameters
    ----------
    Tw_C : float
        Water temperature in degrees Celsius.
    Tg_C : float
        Glass temperature in degrees Celsius.
    h_c : float
        Convective heat transfer coefficient (W/m²·K).
    P_w : float
        Saturation pressure at water temperature (Pa).
    P_g : float
        Saturation pressure at glass temperature (Pa).
    
    Returns
    -------
    float
        Evaporative heat transfer coefficient (W/m²·K).
    """
    if abs(Tw_C - Tg_C) < 1e-6:  # Avoid division by zero
        return 0.0
    h_e = 16.273e-3 * h_c * (P_w - P_g) / (Tw_C - Tg_C)
    return max(h_e, 0.0)  # Ensure non-negative

# System of ODEs (Paper's Equation 1 and 2)
def solar_still_odes(t_sec, y, A):
    """
    Right-hand side of the solar still ODE system (paper's Eq. 1 & 2).
    
    Implements coupled energy balance equations for water and glass,
    plus cumulative mass of collected water.
    
    Energy balances:
    - Water basin:  M_w × C_w × dT_w/dt = Q_solar - Q_rad - Q_conv - Q_evap
    - Glass cover:  M_g × C_g × dT_g/dt = Q_solar + Q_rad + Q_conv + Q_evap - Q_losses
    - Collection:   dM_col/dt = η_coll × m_evap
    
    State vector y contains:
    - y[0] = T_w : water temperature (°C)
    - y[1] = T_g : glass temperature (°C)
    - y[2] = M_col : cumulative collected mass (kg)
    
    Parameters
    ----------
    t_sec : float
        Current simulation time in seconds.
    y : array_like, shape (3,)
        Current state vector [T_w, T_g, M_col].
    A : float
        Basin surface area (m²) used to scale mass and heat fluxes.
    
    Returns
    -------
    list of float
        Time derivatives [dT_w/dt, dT_g/dt, dM_col/dt].
    """
    Tw_C, Tg_C, M_col = y  # Temperatures in Celsius
    
    # Clip temperatures to valid range
    Tw_C = float(np.clip(Tw_C, -40.0, 100.0))
    Tg_C = float(np.clip(Tg_C, -40.0, 100.0))
    
    # Get current irradiance from global array
    idx = int(round((t_sec - t_seconds[0]) / DT_SEC))
    idx = np.clip(idx, 0, len(G_time) - 1)
    G_current = G_time[idx]
    
    # Use constant temperatures
    T_amb_current = T_AMB_C
    T_sky_current = T_SKY_C
    
    # Calculate masses based on area (generalizable for optimization)
    # Note: area is considered equal for water and glass layers, as difference is negligible
    m_water = RHO_WATER * WATER_DEPTH * A
    m_glass = RHO_GLASS * GLASS_THICKNESS * A
    
    # Thermal capacitances
    Cw = m_water * C_P_WATER
    Cg = m_glass * C_P_GLASS
    
    # Calculate heat transfer coefficients
    h_r = h_r_wg(Tw_C, Tg_C)
    h_c, P_w, P_g = h_c_wg(Tw_C, Tg_C)
    h_e = h_e_wg(Tw_C, Tg_C, h_c, P_w, P_g)
    
    # Temperature differences
    delta_Tw_Tg = Tw_C - Tg_C  # Water to glass
    delta_Tg_Ta = Tg_C - T_amb_current  # Glass to ambient
    
    # Solar absorption (W)
    Q_solar_water = TAU_GLASS * ALPHA_WATER * G_current * A  # Solar reaching water through glass
    Q_solar_glass = ALPHA_GLASS * G_current * A  # Solar absorbed by glass itself
    
    # Heat transfers between water and glass (W) - using paper's sign convention
    Q_rad_wg = h_r * A * delta_Tw_Tg
    Q_conv_wg = h_c * A * delta_Tw_Tg
    Q_evap_wg = h_e * A * delta_Tw_Tg
    
    # Glass to environment losses (W) - using full Stefan-Boltzmann as in paper
    Tg_K = Tg_C + 273
    Tsky_K = T_sky_current + 273
    Q_rad_sky = EPSILON_G * SIGMA * A * (Tg_K**4 - Tsky_K**4)  # Radiative loss to sky
    Q_conv_amb = H_GA * A * delta_Tg_Ta  # Convective loss to ambient air
    
    # Energy balances (Paper's Eq. 1 and 2)
    # Water: gains solar, loses to glass via radiation, convection, evaporation
    dTw_dt = (Q_solar_water - Q_rad_wg - Q_conv_wg - Q_evap_wg) / Cw
    
    # Glass: gains solar + heat from water, loses to ambient via radiation and convection
    dTg_dt = (Q_solar_glass + Q_rad_wg + Q_conv_wg + Q_evap_wg - Q_rad_sky - Q_conv_amb) / Cg
    
    # Water collection rate (Equation 11)
    m_dot = Q_evap_wg / L_V
    dMcol_dt = ETA_COLL * max(m_dot, 0.0)
    
    return [dTw_dt, dTg_dt, dMcol_dt]

# Run simulation using RK45 solver
def run_simulation(basin_area=1.0):
    """
    Run a time-domain simulation of the solar still for a given basin area.
    
    Integrates the ODE system using RK45 method over the pre-defined
    simulation window. Irradiance time series (G_time) and time grid
    (t_seconds) must already be defined globally.
    
    Parameters
    ----------
    basin_area : float, optional
        Basin surface area in square metres (m²). Default is 1.0 m².
    
    Returns
    -------
    scipy.integrate.OdeResult
        Solution object containing:
        - t : time points in seconds
        - y : state trajectories [T_w, T_g, M_col]
        - Solver diagnostics and status flags
    """
    if G_time is None:
        raise RuntimeError("Irradiance profile not loaded")
    
    # Initial conditions: start at ambient temperature
    T_initial = T_AMB_C
    y0 = [T_initial, T_initial, 0.0]  # [T_water, T_glass, M_collected]
    
    sol = solve_ivp(
        solar_still_odes,
        (t_seconds[0], t_seconds[-1]),
        y0,
        args=(basin_area,),
        method="RK45",
        t_eval=t_seconds,
        max_step=DT_SEC,
        rtol=1e-6,
        atol=1e-8
    )
    return sol

# =============================================================================
# AREA OPTIMIZATION
# =============================================================================

def production_for_area(basin_area):
    """
    Compute total daily water production for a given basin area.

    This is a convenience wrapper that runs a full simulation and returns
    the final value of the cumulative collected mass component.

    Parameters
    ----------
    basin_area : float
        Basin area in square metres (m²).

    Returns
    -------
    float
        Total mass of water collected over the simulated day (kg).
    """
    sol = run_simulation(basin_area)
    return sol.y[2][-1]  # Final cumulative mass

def find_area_for_target(target_mass=20.0, A_min=0.1, A_max=30.0):
    """
    Find basin area required to achieve a target daily production.
    
    Samples the production curve over a range of areas for visualization,
    then uses Brent's method (scipy.optimize.brentq) to find the area
    that yields the target mass per day.
    
    Parameters
    ----------
    target_mass : float, optional
        Target daily production in kilograms. Default is 20.0 kg/day.
    A_min : float, optional
        Minimum basin area (m²) as lower bound. Default is 0.1 m².
    A_max : float, optional
        Maximum basin area (m²) as upper bound. Default is 30.0 m².
    
    Returns
    -------
    A_optimal : float or None
        Optimal basin area (m²) that achieves target, or None if optimization fails.
    mass_actual : float or None
        Actual mass (kg/day) produced at A_optimal, or None on failure.
    """
    def objective(A):
        return production_for_area(A) - target_mass
    
    print(f"\nSearching for area producing {target_mass} kg/day...")
    
    # Sample production curve for plotting
    trial_As = np.linspace(A_min, A_max, 10)
    trial_Ms = [production_for_area(A) for A in trial_As]
    
    # Find optimal area using Brent's method
    try:
        A_optimal = brentq(objective, A_min, A_max, xtol=1e-3, rtol=1e-3)
        mass_actual = production_for_area(A_optimal)
    except Exception as e:
        print("Optimization failed:", e)
        return None, None
    
    # Plot optimization curve
    plt.figure(figsize=(10, 6))
    plt.plot(trial_As, trial_Ms, "o-", linewidth=2, label="Production curve")
    plt.axhline(target_mass, color="r", linestyle="--", linewidth=2, label=f"Target: {target_mass} kg/day")
    plt.axvline(A_optimal, color="g", linestyle=":", linewidth=2, label=f"Optimal area: {A_optimal:.2f} m²")
    plt.plot(A_optimal, mass_actual, 'go', markersize=12, markeredgewidth=2)
    plt.xlabel("Basin Area (m²)", fontsize=12)
    plt.ylabel("Daily Water Production (kg)", fontsize=12)
    plt.title("Area Optimization: Production vs. Basin Area", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    # Save plot
    optimization_plot_path = os.path.join(OUTPUT_DIR, "02_Area_Optimization_Curve.png")
    plt.savefig(optimization_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {optimization_plot_path}")
    plt.show(block=False)  # Non-blocking: allows script to continue
    
    print(f"Found optimal area: {A_optimal:.3f} m² producing {mass_actual:.3f} kg/day")
    return A_optimal, mass_actual

# =============================================================================
# VISUALIZATION
# =============================================================================

def print_results(sol, basin_area):
    """
    Print simulation results summary.
    
    Parameters
    ----------
    sol : scipy.integrate.OdeResult
        Solution object returned by :func:`run_simulation`.
    basin_area : float
        Basin surface area in square metres (m²).
    """
    Tw_C, Tg_C, M_col = sol.y
    avg_Tw = np.mean(Tw_C)
    avg_Tg = np.mean(Tg_C)
    total_kg = M_col[-1]
    per_m2_day = total_kg / basin_area
    
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    print(f"Basin area: {basin_area:.2f} m²")
    print(f"Average water temperature (°C): {avg_Tw:.2f}")
    print(f"Average glass temperature (°C): {avg_Tg:.2f}")
    print(f"Total collected water (kg/day): {total_kg:.3f}")
    print(f"Specific yield (kg/m²·day): {per_m2_day:.3f}")
    print("="*60 + "\n")

def plot_results(sol, basin_area):
    """
    Plot temperature profiles and cumulative water production.
    
    Generates two subplots:
    1. Water and glass temperatures vs. time (with ambient and sky temperatures)
    2. Cumulative collected water mass vs. time
    
    The plot is saved to the output directory and displayed.
    
    Parameters
    ----------
    sol : scipy.integrate.OdeResult
        Solution object from run_simulation() containing state trajectories.
    basin_area : float
        Basin surface area (m²) for this simulation run.
    
    Returns
    -------
    None
        Function produces and saves plots but returns nothing.
    """
    Tw_C, Tg_C, M_col = sol.y
    t_hours_plot = sol.t / 3600
    
    # Figure: Temperature profiles and cumulative production
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(t_hours_plot, Tw_C, 'r-', label='Water Temperature', linewidth=2)
    ax1.plot(t_hours_plot, Tg_C, 'b-', label='Glass Temperature', linewidth=2)
    ax1.axhline(y=T_AMB_C, color='g', linestyle='--', label='Ambient Temperature', linewidth=1.5)
    ax1.axhline(y=T_SKY_C, color='c', linestyle=':', label='Sky Temperature', linewidth=1.5)
    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax1.set_title(f'Temperature Profiles (A = {basin_area:.2f} m²)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([t_hours_plot[0], t_hours_plot[-1]])
    
    ax2.plot(t_hours_plot, M_col, 'purple', linewidth=2)
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Cumulative Water Collected (kg)', fontsize=12)
    ax2.set_title('Cumulative Water Collection', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([t_hours_plot[0], t_hours_plot[-1]])
    ax2.fill_between(t_hours_plot, M_col, alpha=0.3, color='purple')
    
    plt.tight_layout()
    # Save plot
    results_plot_path = os.path.join(OUTPUT_DIR, f"03_Temperature_and_Water_Cumulation_{basin_area:.2f}m2.png")
    plt.savefig(results_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {results_plot_path}")
    plt.show(block=False)  # Non-blocking: allows script to continue

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run simulation for standard area (1 m²)
    print("\n" + "="*60)
    print("RUNNING SIMULATION FOR STANDARD AREA (1 m²)")
    print("="*60)
    sol_1m2 = run_simulation(basin_area=1.0)
    print_results(sol_1m2, basin_area=1.0)
    
    # Area optimization: find area needed for 20 kg/day
    print("\n" + "="*60)
    print("AREA OPTIMIZATION")
    print("="*60)
    A_opt, M_opt = find_area_for_target(target_mass=20.0, A_min=0.1, A_max=30.0)
    
    # Run simulation with optimal area
    sol_opt = None
    if A_opt is not None:
        print("\n" + "="*60)
        print(f"RUNNING SIMULATION FOR OPTIMAL AREA ({A_opt:.2f} m²)")
        print("="*60)
        sol_opt = run_simulation(basin_area=A_opt)
        print_results(sol_opt, basin_area=A_opt)
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE - GENERATING PLOTS")
    print("="*60)
    
    # Generate temperature + water cumulation plot after loop completes
    if sol_opt is not None:
        plot_results(sol_opt, basin_area=A_opt)
    
    # Keep all plot windows open until user closes them
    plt.show()