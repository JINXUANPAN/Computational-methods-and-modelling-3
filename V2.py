# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 13:02:15 2025

Solar Still Model - Based on Vasava et al. (2023)
Mathematical modelling matching paper equations with area optimization

@author: Administrator
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
# LOADING AND PREPROCESSING SOLAR IRRADIANCE DATA
# =============================================================================

file_path = 'assessment 1/solar.csv' 
df = pd.read_csv(file_path)

data = df.iloc[1:, [0, 1]].copy()
data.columns = ["Hour", "Solar_Irradiance"]

# Cleaning Data
data = data.dropna()
data = data[data["Hour"].str.contains(":")]
data["Solar_Irradiance"] = pd.to_numeric(data["Solar_Irradiance"], errors="coerce")
data = data.dropna()
data["Hour"] = data["Hour"].str.replace(":00", "").astype(float)
data = data.groupby("Hour", as_index=False).mean().sort_values("Hour")

# =============================================================================
# REGRESSION MODEL (INTERPOLATING DATA TO FORM CONTINUOUS FUNCTION)
# =============================================================================

x = data["Hour"].values
y = data["Solar_Irradiance"].values

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

# =============================================================================
# ODE MODEL: SOLAR STILL DYNAMICS (BASED ON PAPER EQUATIONS)
# =============================================================================

# Solar irradiance function from regression
def G_t_from_regression(t, scale_factor=2.0):
    """Returns solar irradiance at time t (seconds)"""
    hour = (t / 3600.0) % 24
    G = poly_func(hour)
    G = max(G, 0.0)
    return G * scale_factor

# System parameters
def default_parameters(basin_area=1.0):
    """Returns dictionary of system parameters matching paper values"""
    params = {
        "A": basin_area,       # Basin surface area (m²)
        "m_w": 21.72,          # Mass of water (kg) - 20mm depth from paper
        "c_p_w": 4200,         # Specific heat capacity of water (J/kg·K) - paper value
        "m_g": 1.0,            # Mass of glass (kg)
        "c_p_g": 800,          # Specific heat capacity of glass (J/kg·K)
        "h_ga": 20.0,          # Heat transfer coefficient glass-ambient (W/m²·K)
        "epsilon_w": 0.9,      # Emissivity of water
        "epsilon_g": 0.9,      # Emissivity of glass
        "sigma": 5.67e-8,      # Stefan-Boltzmann constant (W/m²·K⁴)
        "h_fg": 2.26e6,        # Latent heat of vaporization (J/kg) - paper value
        "eta_coll": 0.8,       # Collection efficiency
        "T_amb": 298.0,        # Ambient temperature (K)
        "tau_g": 0.9,          # Glass transmittance
        "alpha_water": 0.9,    # Solar absorptivity of water
        "alpha_glass": 0.05,   # Solar absorptivity of glass
    }
    return params

# Saturation pressure (Paper's Equation 9, 10)
def p_sat(T_celsius):
    """
    Saturation vapor pressure using paper's equation:
    P_sat = exp(25.317 - 5144/(T + 273))
    """
    T_celsius = np.clip(T_celsius, 0.0, 100.0)
    return np.exp(25.317 - 5144.0 / (T_celsius + 273.0))

# Effective emissivity (Paper's Equation 5)
def epsilon_eff(params):
    """ε_eff = 1 / (1/ε_w + 1/ε_g - 1)"""
    eps_w = params["epsilon_w"]
    eps_g = params["epsilon_g"]
    return 1.0 / (1.0/eps_w + 1.0/eps_g - 1.0)

# Radiative heat transfer coefficient (Paper's Equation 4)
def h_r_wg(Tw_C, Tg_C, params):
    """
    h_r,w-g = ε_eff × σ × [(T_w+273)² + (T_g+273)²] × [(T_w + T_g + 546)]
    """
    Tw_K = Tw_C + 273.15
    Tg_K = Tg_C + 273.15
    eps_eff_val = epsilon_eff(params)
    sigma = params["sigma"]
    h_r = eps_eff_val * sigma * (Tw_K**2 + Tg_K**2) * (Tw_K + Tg_K)
    return h_r

# Helper function for signed cube root
def signed_cuberoot(x):
    """Returns signed cube root to handle negative temperature differences"""
    return np.sign(x) * (np.abs(x) ** (1.0/3.0))

# Convective heat transfer coefficient (Paper's Equation 8)
def h_c_wg(Tw_C, Tg_C, params):
    """
    h_c,w-g = 0.884 × [(T_w - T_g) + (P_w - P_g)×(T_w+273)/(2.723×10⁴ - P_w)]^(1/3)
    """
    if Tw_C <= Tg_C:
        return 0.1, p_sat(Tw_C), p_sat(Tg_C)
    
    P_w = p_sat(Tw_C)
    P_g = p_sat(Tg_C)
    
    term1 = Tw_C - Tg_C
    term2 = (P_w - P_g) * (Tw_C + 273.15) / (268900.0 - P_w + 1e-9)
    combined = term1 + term2
    
    if combined <= 0:
        return 0.1, P_w, P_g
    
    h_c = 0.884 * signed_cuberoot(combined)
    return max(h_c, 0.1), P_w, P_g

# Evaporative heat transfer coefficient (Paper's Equation 12)
def h_e_wg(Tw_C, Tg_C, h_c, P_w, P_g):
    """
    h_e,w-g = 16.273×10⁻³ × h_c × (P_w - P_g)/(T_w - T_g)
    """
    if abs(Tw_C - Tg_C) < 1e-6:
        return 0.0
    h_e = 16.273e-3 * h_c * (P_w - P_g) / (Tw_C - Tg_C)
    return max(h_e, 0.0)

# Heat flux calculations
def Q_solar_water(params, t):
    """Solar energy absorbed by water"""
    G_t = G_t_from_regression(t)
    return params["A"] * params["tau_g"] * params["alpha_water"] * G_t

def Q_solar_glass(params, t):
    """Solar energy absorbed by glass"""
    G_t = G_t_from_regression(t)
    return params["A"] * params["alpha_glass"] * G_t

def Q_rad_g_sky(Tg_C, params):
    """Radiative heat transfer glass to sky (Equation 6)"""
    Tg_K = Tg_C + 273.15
    T_sky_K = params["T_amb"]  # Simplified
    return params["epsilon_g"] * params["sigma"] * params["A"] * (Tg_K**4 - T_sky_K**4)

def Q_conv_g_amb(Tg_C, params):
    """Convective heat transfer glass to ambient"""
    T_amb_C = params["T_amb"] - 273.15
    return params["h_ga"] * params["A"] * (Tg_C - T_amb_C)

# System of ODEs (Paper's Equation 1 and 2)
def solar_still_odes(t, y, params):
    """
    Energy balance equations from paper:
    
    Equation 2 (Water): M_w × C_w × dT_w/dt = I(t)×τ×α - Q_evap - Q_rad - Q_conv
    Equation 1 (Glass): M_g × C_g × dT_g/dt = I(t)×α_g + η×Q_evap + Q_rad + Q_conv 
                                               - Q_rad_sky - Q_conv_amb
    """
    Tw_C, Tg_C, M_col = y  # Temperatures in Celsius
    
    # Clip temperatures to valid range
    Tw_C = np.clip(Tw_C, -40.0, 100.0)
    Tg_C = np.clip(Tg_C, -40.0, 100.0)
    
    # Calculate heat transfer coefficients
    h_r = h_r_wg(Tw_C, Tg_C, params)
    h_c, P_w, P_g = h_c_wg(Tw_C, Tg_C, params)
    h_e = h_e_wg(Tw_C, Tg_C, h_c, P_w, P_g)
    
    # Heat fluxes (W)
    Q_solar_w = Q_solar_water(params, t)
    Q_solar_g = Q_solar_glass(params, t)
    Q_evap = h_e * params["A"] * (Tw_C - Tg_C)
    Q_conv = h_c * params["A"] * (Tw_C - Tg_C)
    Q_rad = h_r * params["A"] * (Tw_C - Tg_C)
    Q_rad_sky = Q_rad_g_sky(Tg_C, params)
    Q_conv_amb = Q_conv_g_amb(Tg_C, params)
    
    # Thermal capacitances
    Cw = params["m_w"] * params["c_p_w"]
    Cg = params["m_g"] * params["c_p_g"]
    
    # Energy balances (Paper's Eq. 1 and 2)
    dTw_dt = (Q_solar_w - Q_evap - Q_rad - Q_conv) / Cw
    dTg_dt = (Q_solar_g + params["eta_coll"] * Q_evap + Q_rad + Q_conv 
              - Q_rad_sky - Q_conv_amb) / Cg
    
    # Water collection rate (Equation 11)
    m_dot = Q_evap / params["h_fg"]
    dMcol_dt = params["eta_coll"] * max(m_dot, 0.0)
    
    return [dTw_dt, dTg_dt, dMcol_dt]

# Run simulation using RK45 solver
def run_simulation(basin_area=1.0, t_span=(0, 3600 * 24), y0=None):
    """Execute simulation for given basin area"""
    params = default_parameters(basin_area)
    
    if y0 is None:
        y0 = [30.0, 25.0, 0.0]  # Initial: Tw=30°C, Tg=25°C, M_col=0
    
    sol = solve_ivp(
        solar_still_odes,
        t_span=t_span,
        y0=y0,
        args=(params,),
        method="RK45",
        max_step=60.0,
        dense_output=True
    )
    return sol

# =============================================================================
# AREA OPTIMIZATION
# =============================================================================

def production_for_area(basin_area):
    """Calculate total daily water production for given area"""
    sol = run_simulation(basin_area, t_span=(0, 3600 * 24))
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
    t_hours = sol.t / 3600
    
    # Figure 1: Temperature profiles and cumulative production
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(t_hours, Tw_C, 'r-', label='Water Temperature', linewidth=2)
    ax1.plot(t_hours, Tg_C, 'b-', label='Glass Temperature', linewidth=2)
    params = default_parameters(basin_area)
    ax1.axhline(y=params["T_amb"] - 273.15, color='green', linestyle='--', 
                label='Ambient Temperature', linewidth=1.5)
    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax1.set_title(f'Temperature Profiles (A = {basin_area:.2f} m²)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 24])
    
    ax2.plot(t_hours, M_col, 'purple', linewidth=2)
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Cumulative Water Collected (kg)', fontsize=12)
    ax2.set_title('Cumulative Water Collection', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 24])
    ax2.fill_between(t_hours, M_col, alpha=0.3, color='purple')
    
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

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    # Run baseline simulation (1 m² area)
    print("\nRunning baseline simulation (A = 1.0 m²)...")
    sol_baseline = run_simulation(basin_area=1.0)
    plot_results(sol_baseline, basin_area=1.0)
    
    # Find optimal area for target production
    target_production = 5.0  # kg/day
    A_optimal, actual_prod = find_area_for_target(target_production, A_min=0.1, A_max=20.0)
    
    # Run simulation with optimal area
    if A_optimal:
        print(f"\nRunning simulation with optimal area (A = {A_optimal:.2f} m²)...")
        sol_optimal = run_simulation(basin_area=A_optimal)
        plot_results(sol_optimal, basin_area=A_optimal)