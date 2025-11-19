"""
Solar Still Simulation Framework (Based on Vasava et al., 2023)

This module implements a mathematical model of a solar still derived from the 
equations in Vasava et al. (2023). Users can input local climate data and 
system design targets, and the model computes the expected freshwater output. 
The structure is intentionally general so the model can be applied to various 
climates and output requirements.
"""

# Importing libraries used 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# =============================================================================
# PHYSICAL CONSTANTS AND GLOBAL PARAMETERS
# =============================================================================

# Physical constants - mostly lifted from basis paper in within zip
WATER_DEPTH = 0.02              # m  
GLASS_THICKNESS = 0.004         # m 
RHO_WATER = 1000.0              # kg/m³
RHO_GLASS = 2500.0              # kg/m³
C_P_WATER = 4200.0              # J/(kg·K) 
C_P_GLASS = 750.0               # J/(kg·K)
EPSILON_W = 0.95                # Water emissivity
EPSILON_G = 0.90                # Glass emissivity
SIGMA = 5.67e-8                 # Stefan-Boltzmann (W/m²·K⁴)
L_V = 2.26e6                    # Latent heat (J/kg) 
ALPHA_WATER = 0.85              # Basin absorptivity
ALPHA_GLASS = 0.05              # Glass absorptivity
TAU_GLASS = 0.90                # Glass transmittance
H_GA = 5.0                      # Convective glass-air (W/m²·K)
ETA_COLL = 0.80                 # Collection efficiency

DT_SEC = 60                     # Timestep (s)

# Environmental conditions
T_AMB_C = 30.0                  # Ambient temperature (°C)
T_SKY_C = 20.0                  # Sky temperature (°C)

# Globals to hold time and irradiance arrays
t_seconds = None
t_hours = None
G_time = None
t_start_hr = None
t_end_hr = None

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

# Initialize time arrays for simulation
t_start_hr = x.min()
t_end_hr = x.max()
t_hours = np.arange(t_start_hr, t_end_hr + DT_SEC/3600, DT_SEC/3600)
t_seconds = t_hours * 3600.0
G_time = np.clip(poly_func(t_hours), 0.0, None)

print(f"Simulation window: {t_start_hr:.1f}h - {t_end_hr:.1f}h")

# =============================================================================
# ODE MODEL: SOLAR STILL DYNAMICS 
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

# Radiative heat transfer coefficient (Paper's Equation 4)
def h_r_wg(Tw_C, Tg_C):
    """
    h_r,w-g = ε_eff × σ × [(T_w+273)² + (T_g+273)²] × [(T_w + T_g + 546)]
    """
    Tw_K = Tw_C + 273.15
    Tg_K = Tg_C + 273.15
    eps_eff_val = epsilon_eff()
    h_r = eps_eff_val * SIGMA * (Tw_K**2 + Tg_K**2) * (Tw_K + Tg_K)
    return h_r

# Helper function for signed cube root
def signed_cuberoot(x):
    """Returns signed cube root to handle negative temperature differences"""
    return np.sign(x) * (np.abs(x) ** (1.0/3.0))

# Convective heat transfer coefficient (Paper's Equation 8)
def h_c_wg(Tw_C, Tg_C):
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

# System of ODEs (Paper's Equation 1 and 2)
def solar_still_odes(t_sec, y, A):
    """
    Energy balance equations from paper:
    
    Equation 2 (Water): M_w × C_w × dT_w/dt = I(t)×τ×α - Q_evap - Q_rad - Q_conv
    Equation 1 (Glass): M_g × C_g × dT_g/dt = I(t)×α_g + η×Q_evap + Q_rad + Q_conv 
                                               - Q_rad_sky - Q_conv_amb
    """
    Tw_C, Tg_C, M_col = y  # Temperatures in Celsius
    
    # Clip temperatures to valid range
    Tw_C = float(np.clip(Tw_C, -40.0, 100.0))
    Tg_C = float(np.clip(Tg_C, -40.0, 100.0))
    
    # Get current irradiance from global array
    idx = int(round((t_sec - t_seconds[0]) / DT_SEC))
    idx = np.clip(idx, 0, len(G_time) - 1)
    G_current = G_time[idx]
    
    # Calculate masses based on area (generalizable for optimization)
    m_water = RHO_WATER * WATER_DEPTH * A
    m_glass = RHO_GLASS * GLASS_THICKNESS * A
    
    # Thermal capacitances
    Cw = m_water * C_P_WATER
    Cg = m_glass * C_P_GLASS
    
    # Calculate heat transfer coefficients
    h_r = h_r_wg(Tw_C, Tg_C)
    h_c, P_w, P_g = h_c_wg(Tw_C, Tg_C)
    h_e = h_e_wg(Tw_C, Tg_C, h_c, P_w, P_g)
    
    # Heat fluxes (W)
    Q_solar_water = TAU_GLASS * ALPHA_WATER * G_current * A
    Q_solar_glass = ALPHA_GLASS * G_current * A
    Q_evap = h_e * A * (Tw_C - Tg_C)
    Q_conv = h_c * A * (Tw_C - Tg_C)
    Q_rad = h_r * A * (Tw_C - Tg_C)
    
    # Glass to environment heat losses
    Tw_K = Tw_C + 273.15
    Tg_K = Tg_C + 273.15
    T_amb_K = T_AMB_C + 273.15
    T_sky_K = T_SKY_C + 273.15
    
    Q_rad_sky = EPSILON_G * SIGMA * A * (Tg_K**4 - T_sky_K**4)
    Q_conv_amb = H_GA * A * (Tg_C - T_AMB_C)
    
    # Energy balances (Paper's Eq. 1 and 2)
    dTw_dt = (Q_solar_water - Q_evap - Q_rad - Q_conv) / Cw
    dTg_dt = (Q_solar_glass + ETA_COLL * Q_evap + Q_rad + Q_conv 
              - Q_rad_sky - Q_conv_amb) / Cg
    
    # Water collection rate (Equation 11)
    m_dot = Q_evap / L_V
    dMcol_dt = ETA_COLL * max(m_dot, 0.0)
    
    return [dTw_dt, dTg_dt, dMcol_dt]

# Run simulation using RK45 solver
def run_simulation(basin_area=1.0):
    """Execute simulation for given basin area"""
    if G_time is None:
        raise RuntimeError("Irradiance not loaded")
    
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
    ax1.axhline(y=T_AMB_C, color='green', linestyle='--', 
                label='Ambient Temperature', linewidth=1.5)
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

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    
    # Run baseline simulation (1 m² area)
    print("\nRunning baseline simulation (A = 1.0 m²)...")
    sol_baseline = run_simulation(basin_area=1.0)
    plot_results(sol_baseline, basin_area=1.0)
    
    # Find optimal area for target production
    target_production = 20.0  # kg/day
    A_optimal, actual_prod = find_area_for_target(target_production, A_min=0.1, A_max=20.0)
    
    # Run simulation with optimal area
    if A_optimal:
        print(f"\nRunning simulation with optimal area (A = {A_optimal:.2f} m²)...")
        sol_optimal = run_simulation(basin_area=A_optimal)
        plot_results(sol_optimal, basin_area=A_optimal)