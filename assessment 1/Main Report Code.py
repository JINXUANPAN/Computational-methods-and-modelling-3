#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 12:39:07 2025

@author: matiaslander
"""

"""
Solar Still Simulation Framework (Based on Vasava et al., 2023)

This module implements a mathematical model of a solar still derived from the 
equations in Vasava et al. (2023). Users can input local climate data and 
system design targets, and the model computes the expected freshwater output. 
The structure is intentionally general so the model can be applied to various 
climates and output requirements.

Note: safeguards are included to prevent division by zero errors in calculations. 
Although these cases are unlikely in practical scenarios, they ensure numerical stability.
"""

# Importing libraries used 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# Font configuration
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

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

# Environmental conditions - Temperature data (from climate statistics)
T_MIN_C = 17.8                  # Minimum ambient temperature (°C) - early morning
T_MAX_C = 31.6                  # Maximum ambient temperature (°C) - afternoon
T_MEAN_C = 24.67                # Mean ambient temperature (°C)
T_MIN_HOUR = 6.0                # Hour when minimum temperature occurs
T_MAX_HOUR = 14.0               # Hour when maximum temperature occurs

# Configuration
PERCENTAGE_FILE = "hourly_percentage_irradiance.csv"
TOTAL_IRRADIANCE = 3852         # Total daily irradiance (W/m²·day)

# Globals to hold time and irradiance arrays
t_seconds = None
t_hours = None
G_time = None
T_amb_profile = None            # Time-varying ambient temperature array
T_sky_profile = None            # Time-varying sky temperature array
t_start_hr = None
t_end_hr = None
poly_func = None

# =============================================================================
# LOADING AND PREPROCESSING SOLAR IRRADIANCE DATA
# =============================================================================

def load_percentage_data(filepath):
    """Load hourly percentage irradiance data from preprocessed CSV."""
    df = pd.read_csv(filepath)
    if not {'Hour', 'Percentage of Daily Total'}.issubset(df.columns):
        raise ValueError("CSV must contain columns: 'Hour' and 'Percentage of Daily Total'")
    return df

def compute_hourly_irradiance(df, total_irradiance):
    """Apply total daily irradiance to percentage distribution."""
    df = df.copy()
    df["Estimated Irradiance (W/m2)"] = (df["Percentage of Daily Total"] / 100) * total_irradiance
    df["Hour_float"] = df["Hour"].apply(lambda x: int(x.split(":")[0]) + int(x.split(":")[1]) / 60)
    df = df.sort_values("Hour_float")
    return df

# Load percentage data
print("\n" + "="*60)
print("Loading preprocessed irradiance data...")
print("="*60)

df_pct = load_percentage_data(PERCENTAGE_FILE)
df_irr = compute_hourly_irradiance(df_pct, TOTAL_IRRADIANCE)

print(f"\nHourly Irradiance Estimated from Total = {TOTAL_IRRADIANCE:.0f} W/m²·day")
print("="*60)
print(df_irr[["Hour", "Percentage of Daily Total", "Estimated Irradiance (W/m2)"]].to_string(index=False, float_format="%.2f"))
print("="*60)

# =============================================================================
# REGRESSION MODEL (INTERPOLATING DATA TO FORM CONTINUOUS FUNCTION)
# =============================================================================

# loading data from CSV file
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
plt.show()

# Initialize time arrays for simulation
t_start_hr = x.min()
t_end_hr = x.max()
t_hours = np.arange(t_start_hr, t_end_hr + DT_SEC/3600, DT_SEC/3600)
t_seconds = t_hours * 3600.0
G_time = np.clip(poly_func(t_hours), 0.0, None)

print(f"Simulation window: {t_start_hr:.1f}h - {t_end_hr:.1f}h")

# =============================================================================
# TEMPERATURE PROFILE GENERATION (SAME METHOD AS IRRADIANCE FORCER)
# =============================================================================

def load_temperature_percentage_data(filepath):
    """Load hourly percentage temperature distribution (same as irradiance method)."""
    df = pd.read_csv(filepath)
    if not {'Hour', 'Percentage of Daily Variation'}.issubset(df.columns):
        raise ValueError("CSV must contain: 'Hour' and 'Percentage of Daily Variation'")
    return df

def compute_hourly_temperature(df, T_min, T_max):
    """
    Apply temperature range to percentage distribution.
    Same methodology as irradiance forcer.
    
    Args:
        df: DataFrame with percentage distribution
        T_min: Minimum temperature (°C)
        T_max: Maximum temperature (°C)
    
    Returns:
        DataFrame with calculated temperatures
    """
    df = df.copy()
    temp_range = T_max - T_min
    # Apply percentage: T = T_min + (percentage/100) × range
    df["Ambient_Temperature_C"] = T_min + (df["Percentage of Daily Variation"] / 100) * temp_range
    df["Hour_float"] = df["Hour"].apply(lambda x: int(x.split(":")[0]) + int(x.split(":")[1]) / 60)
    df = df.sort_values("Hour_float")
    return df

def calculate_sky_temp(T_amb_C):
    """
    Calculate effective sky temperature for radiative cooling.
    
    The effective sky temperature is taken as 25 K below ambient temperature,
    a value consistent with clear-sky conditions reported in the literature.
    
    Args:
        T_amb_C: Ambient temperature (°C) - scalar or array
    
    Returns:
        Sky temperature (°C) - effective radiative temperature
    """
    # Sky temperature is 25K below ambient (clear sky approximation)
    return T_amb_C - 25.0

# Load and process temperature data (parallel to irradiance)
print("\n" + "="*60)
print("Loading temperature percentage distribution...")
print("="*60)

TEMPERATURE_PCT_FILE = "hourly_percentage_temperature.csv"

df_temp_pct = load_temperature_percentage_data(TEMPERATURE_PCT_FILE)
df_temp = compute_hourly_temperature(df_temp_pct, T_MIN_C, T_MAX_C)

# Add sky temperature
df_temp["Sky_Temperature_C"] = calculate_sky_temp(df_temp["Ambient_Temperature_C"])

print(f"\nTemperature Profile (from percentage distribution):")
print(f"  Range: {T_MIN_C:.2f}°C to {T_MAX_C:.2f}°C")
print("="*60)
print(df_temp[["Hour", "Percentage of Daily Variation", "Ambient_Temperature_C", "Sky_Temperature_C"]].to_string(
    index=False, float_format="%.2f"))
print("="*60)

# Interpolate to match simulation time array (same as irradiance)
from scipy.interpolate import interp1d

# Create interpolation functions
x_temp = df_temp["Hour_float"].values
y_amb = df_temp["Ambient_Temperature_C"].values
y_sky = df_temp["Sky_Temperature_C"].values

# Interpolate to simulation time points
T_amb_profile = np.interp(t_hours, x_temp, y_amb)
T_sky_profile = np.interp(t_hours, x_temp, y_sky)

print(f"\nInterpolated Temperature Profiles (matched to simulation time):")
print(f"  Ambient Temperature:")
print(f"    Minimum: {np.min(T_amb_profile):.2f}°C (at hour {t_hours[np.argmin(T_amb_profile)]:.1f})")
print(f"    Maximum: {np.max(T_amb_profile):.2f}°C (at hour {t_hours[np.argmax(T_amb_profile)]:.1f})")
print(f"    Mean:    {np.mean(T_amb_profile):.2f}°C")
print(f"  Sky Temperature:")
print(f"    Minimum: {np.min(T_sky_profile):.2f}°C")
print(f"    Maximum: {np.max(T_sky_profile):.2f}°C")
print(f"    Mean:    {np.mean(T_sky_profile):.2f}°C")
print(f"  Average Amb-Sky Difference: {np.mean(T_amb_profile - T_sky_profile):.2f}°C")
print("="*60)

# =============================================================================
# ODE MODEL: SOLAR STILL DYNAMICS (BASED ON PAPER EQUATIONS)
# =============================================================================

# Saturation pressure (Paper's Equation 9, 10)
def p_sat(T_celsius):
    """
    Saturation vapor pressure using paper's equation:
    P_sat = exp(25.317 - 5144/(T + 273))
    """
    T_celsius = np.clip(T_celsius, 0.0, 100.0)
    return np.exp(25.317 - 5144.0 / (T_celsius + 273.0))

# Effective emissivity (Paper's Equation 5)
def epsilon_eff():
    """ε_eff = 1 / (1/ε_w + 1/ε_g - 1)"""
    return 1.0 / (1.0/EPSILON_W + 1.0/EPSILON_G - 1.0)

# Radiative heat transfer coefficient (Paper's Equation 4 - equivalent Kelvin values 
# introduced for simplicity when using Stefan-Boltzmann)
def h_r_wg(Tw_C, Tg_C):
    """
    h_r,w-g = ε_eff × σ[(T_w+273)² + (T_g+273)²] × [(T_w + T_g + 546)]
    """
    Tw_K = Tw_C + 273
    Tg_K = Tg_C + 273
    eps_eff_val = epsilon_eff()
    h_r = eps_eff_val * SIGMA * (Tw_K**2 + Tg_K**2) * (Tw_K + Tg_K)
    return h_r

# Helper function for signed cube root (to handle negative values)
def signed_cuberoot(x):
    """Returns signed cube root to handle negative temperature differences"""
    return np.sign(x) * (np.abs(x) ** (1.0/3.0))

# Convective heat transfer coefficient (Paper's Equation 8)
def h_c_wg(Tw_C, Tg_C):
    """
    h_c,w-g = 0.884 × [(T_w - T_g) + (P_w - P_g)×(T_w+273)/(2.723×10⁴ - P_w)]^(1/3)
    """
    if Tw_C <= Tg_C: # No convection if water temp <= glass temp
        return 0.1, p_sat(Tw_C), p_sat(Tg_C)
    
    P_w = p_sat(Tw_C)
    P_g = p_sat(Tg_C)
    
    term1 = Tw_C - Tg_C
    term2 = (P_w - P_g) * (Tw_C + 273) / (2.7230e4 - P_w + 1e-9) # Avoid division by zero
    combined = term1 + term2
    
    h_c = 0.884 * signed_cuberoot(combined)
    return max(h_c, 0.1), P_w, P_g

# Evaporative heat transfer coefficient (Paper's Equation 12)
def h_e_wg(Tw_C, Tg_C, h_c, P_w, P_g):
    """
    h_e,w-g = 16.273×10⁻³ × h_c × (P_w - P_g)/(T_w - T_g)
    """
    if abs(Tw_C - Tg_C) < 1e-6: # Avoid division by zero
        return 0.0
    h_e = 16.273e-3 * h_c * (P_w - P_g) / (Tw_C - Tg_C)
    return max(h_e, 0.0) # Ensure non-negative

# System of ODEs (Paper's Equation 1 and 2)
def solar_still_odes(t_sec, y, A):
    """
    Energy balance equations from paper (Equations 1 and 2):
    
    Water basin:  M_w × C_w × dT_w/dt = α_w×τ_g×I(t) - Q_r,w-g - Q_c,w-g - Q_e,w-g
    
    Glass cover:  M_g × C_g × dT_g/dt = α_g×I(t) + Q_r,w-g + Q_c,w-g + Q_e,w-g 
                                        - Q_r,g-a - Q_c,g-a
    
    Where:
        Q_r,w-g = h_r,w-g × A × (T_w - T_g)     [Radiative heat transfer, water to glass]
        Q_c,w-g = h_c,w-g × A × (T_w - T_g)     [Convective heat transfer, water to glass]
        Q_e,w-g = h_e,w-g × A × (T_w - T_g)     [Evaporative heat transfer, water to glass]
        Q_r,g-a = ε_g×σ×A×(T_g⁴ - T_sky⁴)       [Radiative heat loss, glass to sky]
        Q_c,g-a = h_c,g-a × A × (T_g - T_a)     [Convective heat loss, glass to ambient]
    """
    Tw_C, Tg_C, M_col = y  # Temperatures in Celsius
    
    # Clip temperatures to valid range
    Tw_C = float(np.clip(Tw_C, -40.0, 100.0))
    Tg_C = float(np.clip(Tg_C, -40.0, 100.0))
    
    # Get current irradiance and temperatures from global arrays
    idx = int(round((t_sec - t_seconds[0]) / DT_SEC))
    idx = np.clip(idx, 0, len(G_time) - 1)
    G_current = G_time[idx]
    T_amb_current = T_amb_profile[idx]  # Time-varying ambient temperature
    T_sky_current = T_sky_profile[idx]  # Time-varying sky temperature
    
    # Calculate masses based on area (generalizable for optimization)
    # Note: area is considered equal for water and glass layers, as difference is considered negligible
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
    delta_Tg_Ta = Tg_C - T_amb_current  # Glass to ambient (time-varying)
    
    # Solar absorption (W)
    Q_solar_water = TAU_GLASS * ALPHA_WATER * G_current * A  # Solar reaching water through glass
    Q_solar_glass = ALPHA_GLASS * G_current * A  # Solar absorbed by glass itself
    
    # Heat transfers between water and glass (W) - using paper's sign convention
    Q_rad_wg = h_r * A * delta_Tw_Tg
    Q_conv_wg = h_c * A * delta_Tw_Tg
    Q_evap_wg = h_e * A * delta_Tw_Tg
    
    # Glass to environment losses (W) - using full Stefan-Boltzmann as in paper
    Tg_K = Tg_C + 273.15
    Tsky_K = T_sky_current + 273.15  # Use time-varying sky temperature
    Q_rad_sky = EPSILON_G * SIGMA * A * (Tg_K**4 - Tsky_K**4)  # Radiative loss to sky
    Q_conv_amb = H_GA * A * delta_Tg_Ta  # Convective loss to ambient air
    
    # Energy balances (Paper's Eq. 1 and 2)
    # Water: gains solar, loses to glass via radiation, convection, evaporation
    dTw_dt = (Q_solar_water - Q_rad_wg - Q_conv_wg - Q_evap_wg) / Cw
    
    # Glass: gains solar + heat from water, loses to ambient via radiation and convection
    dTg_dt = (Q_solar_glass + Q_rad_wg + Q_conv_wg + Q_evap_wg - Q_rad_sky - Q_conv_amb) / Cg
    
    # Water collection rate (kg/s)
    m_dot = Q_evap_wg / L_V
    dMcol_dt = ETA_COLL * max(m_dot, 0.0)
    
    return [dTw_dt, dTg_dt, dMcol_dt]

# Run simulation using RK45 solver
def run_simulation(basin_area=1.0):
    """Execute simulation for given basin area"""
    if G_time is None or T_amb_profile is None:
        raise RuntimeError("Irradiance and temperature profiles not loaded")
    
    # Initial conditions: start at initial ambient temperature
    T_initial = T_amb_profile[0]
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
    """Calculate total daily water production for given area"""
    sol = run_simulation(basin_area)
    return sol.y[2][-1]  # Final cumulative mass

def find_area_for_target(target_mass=20.0, A_min=0.1, A_max=30.0):
    """
    Find basin area required to produce target mass using bisection method.
    Also plots optimization curve showing production vs. area.
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
    plt.show()
    
    print(f"Found optimal area: {A_optimal:.3f} m² producing {mass_actual:.3f} kg/day")
    return A_optimal, mass_actual

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(sol, basin_area):
    """Generate temperature and production plots"""
    Tw_C, Tg_C, M_col = sol.y
    t_hours_plot = sol.t / 3600
    
    # Figure 1: Temperature profiles and cumulative production
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(t_hours_plot, Tw_C, 'r-', label='Water Temperature', linewidth=2)
    ax1.plot(t_hours_plot, Tg_C, 'b-', label='Glass Temperature', linewidth=2)
    ax1.plot(t_hours, T_amb_profile, 'g--', label='Ambient Temperature', linewidth=1.5)
    ax1.plot(t_hours, T_sky_profile, 'c:', label='Sky Temperature', linewidth=1.5)
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
    plt.show()
    
    # Print results
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