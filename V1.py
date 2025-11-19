#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 02:27:42 2025

@author: matiaslander
"""

# -*- coding: utf-8 -*-
"""
Solar Still Model - Matching Vasava et al. Paper
Key changes from original code marked with # CHANGED or # NEW
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# Loading solar irradiance data
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

# ================================================================================================================
# REGRESSION MODEL (Same as before)
# ================================================================================================================

x = data["Hour"].values
y = data["Solar_Irradiance"].values

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
plt.plot(x_smooth, y_smooth, color='blue', label="Polynomial regression")
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

# ================================================================================================================
# ODE MODEL: MATCHING PAPER EQUATIONS
# ================================================================================================================

def G_t_from_regression(t, scale_factor=2.0):
    """Solar irradiance function"""
    hour = (t / 3600.0) % 24
    G = poly_func(hour)
    G = max(G, 0.0)
    return G * scale_factor

# CHANGED: Parameters adjusted to match paper's experimental setup
def default_parameters():
    params = {
        "A_w": 1.0,         # CHANGED: Basin water surface area (m²) - paper uses 1m²
        "A_g": 1.7376,      # NEW: Glass cover area (m²) - from paper
        "m_w": 21.72,       # CHANGED: Mass of water (kg) - paper uses 20mm depth → 21.72kg
        "c_p_w": 4200,      # CHANGED: Specific heat of water (J/kg·K) - paper uses 4200
        "m_g": 1.0,         # Mass of glass (kg) - estimated
        "c_p_g": 800,       # Specific heat of glass (J/kg·K)
        
        # CHANGED: Emissivities for effective emissivity calculation (Eq. 5 in paper)
        "epsilon_w": 0.9,   # NEW: Emissivity of water surface
        "epsilon_g": 0.9,   # Emissivity of glass
        
        "sigma": 5.67e-8,   # Stefan-Boltzmann constant (W/m²·K⁴)
        "h_ga": 20.0,       # Heat transfer coefficient glass-ambient (W/m²·K)
        
        # NEW: Paper-specific parameters
        "tau_g": 0.9,       # NEW: Transmittance of glass cover (from paper assumption)
        "alpha_g": 0.05,    # NEW: Absorptivity of glass (small value, ~5%)
        
        "h_fg": 2260000,    # CHANGED: Latent heat (J/kg) - paper uses 2260 kJ/kg
        "eta_coll": 0.8,    # Collection efficiency
        "T_amb": 298.0,     # Ambient temperature (K)
        "RH": 0.6,          # Relative humidity
        "alpha": 0.9,       # Solar absorptivity of water
    }
    return params

# CHANGED: Saturation pressure using paper's equation (Eq. 9, 10)
def p_sat(T_celsius):
    """
    Saturation pressure using Antoine-like equation from paper
    P_sat = exp(25.317 - 5144/(T + 273))
    Input: Temperature in Celsius
    Output: Pressure in Pa
    """
    T_celsius = np.clip(T_celsius, 0, 100)
    p = np.exp(25.317 - 5144 / (T_celsius + 273))
    return p

# NEW: Effective emissivity (Eq. 5 from paper)
def epsilon_eff(params):
    """Effective emissivity between water and glass"""
    eps_w = params["epsilon_w"]
    eps_g = params["epsilon_g"]
    return 1.0 / (1.0/eps_w + 1.0/eps_g - 1.0)

# CHANGED: Radiative heat transfer using paper's formulation (Eq. 4)
def h_r_wg(Tw, Tg, params):
    """
    Radiative heat transfer coefficient water to glass (Eq. 4 from paper)
    h_r,w-g = ε_eff * σ * [(T_w + 273)² + (T_g + 273)²] * [(T_w + T_g + 546)]
    Input temperatures in Celsius
    """
    Tw_K = Tw + 273.15
    Tg_K = Tg + 273.15
    eps_eff_val = epsilon_eff(params)
    sigma = params["sigma"]
    
    h_r = eps_eff_val * sigma * ((Tw_K)**2 + (Tg_K)**2) * (Tw_K + Tg_K)
    return h_r

# CHANGED: Convective heat transfer using paper's correlation (Eq. 8)
def h_c_wg(Tw, Tg, params):
    """
    Convective heat transfer coefficient water to glass (Eq. 8 from paper)
    h_c,w-g = 0.884 * [(T_w - T_g) + (P_w - P_g)*(T_w + 273)/(2.723e4 - P_w)]^(1/3)
    Input temperatures in Celsius
    """
    if Tw <= Tg:
        return 0.1  # Small positive value to avoid division issues
    
    P_w = p_sat(Tw)
    P_g = p_sat(Tg)
    
    # Paper's equation
    term1 = (Tw - Tg)
    term2 = (P_w - P_g) * (Tw + 273) / (2.723e4 - P_w)
    
    if (term1 + term2) <= 0:
        return 0.1
    
    h_c = 0.884 * ((term1 + term2) ** (1.0/3.0))
    return max(h_c, 0.1)

# CHANGED: Evaporative heat transfer coefficient (Eq. 12 from paper)
def h_e_wg(Tw, Tg, h_c, params):
    """
    Evaporative heat transfer coefficient (Eq. 12 from paper)
    h_e,w-g = 16.273e-3 * h_c * (P_w - P_g)/(T_w - T_g)
    """
    if abs(Tw - Tg) < 0.01:
        return 0.0
    
    P_w = p_sat(Tw)
    P_g = p_sat(Tg)
    
    h_e = 16.273e-3 * h_c * (P_w - P_g) / (Tw - Tg)
    return max(h_e, 0.0)

# CHANGED: Heat transfer calculations matching paper's Eq. 1, 2
def Q_solar_absorbed(params, t):
    """Solar energy absorbed by water (with transmittance)"""
    G_t = G_t_from_regression(t)
    return params["A_w"] * params["tau_g"] * params["alpha"] * G_t

def Q_solar_glass(params, t):
    """Solar energy absorbed by glass"""
    G_t = G_t_from_regression(t)
    return params["A_g"] * params["alpha_g"] * G_t

def Q_evap(Tw, Tg, h_e, params):
    """Evaporative heat transfer (W)"""
    return h_e * params["A_w"] * (Tw - Tg)

def Q_conv_wg(Tw, Tg, h_c, params):
    """Convective heat transfer water to glass (W)"""
    return h_c * params["A_w"] * (Tw - Tg)

def Q_rad_wg(Tw, Tg, h_r, params):
    """Radiative heat transfer water to glass (W)"""
    return h_r * params["A_w"] * (Tw - Tg)

def Q_rad_g_sky(Tg, params):
    """Radiative heat transfer glass to sky (Eq. 6 from paper)"""
    Tg_K = Tg + 273.15
    T_sky_K = params["T_amb"]  # Simplified: T_sky ≈ T_amb
    return params["epsilon_g"] * params["sigma"] * params["A_g"] * (Tg_K**4 - T_sky_K**4)

def Q_conv_g_amb(Tg, params):
    """Convective heat transfer glass to ambient"""
    return params["h_ga"] * params["A_g"] * (Tg - (params["T_amb"] - 273.15))

# CHANGED: Yield calculation using paper's Eq. 11
def yield_rate(Tw, Tg, h_e, params):
    """
    Instantaneous water production rate (kg/s) - Eq. 11 from paper
    m_dot = Q_e,w-g / h_fg = h_e * A_w * (T_w - T_g) / h_fg
    """
    Q_e = Q_evap(Tw, Tg, h_e, params)
    m_dot = Q_e / params["h_fg"]
    return m_dot

# CHANGED: System of ODEs matching paper's energy balance (Eq. 1, 2)
def solar_still_odes(t, y, params):
    """
    ODEs matching Vasava et al. paper
    Eq. 1: M_g * C_g * dT_g/dt = I(t)*α_g + Q_evap + Q_r,w-g - Q_r,g-sky + Q_c,w-g
    Eq. 2: M_w * C_w * dT_w/dt = I(t)*τ_g*α - Q_evap - Q_r,w-g - Q_c,w-g
    """
    Tw_C, Tg_C, M_col = y  # Temperatures in Celsius
    
    # Heat transfer coefficients
    h_r = h_r_wg(Tw_C, Tg_C, params)
    h_c = h_c_wg(Tw_C, Tg_C, params)
    h_e = h_e_wg(Tw_C, Tg_C, h_c, params)
    
    # Heat fluxes
    Q_solar_w = Q_solar_absorbed(params, t)
    Q_solar_g = Q_solar_glass(params, t)
    Q_evap_val = Q_evap(Tw_C, Tg_C, h_e, params)
    Q_conv = Q_conv_wg(Tw_C, Tg_C, h_c, params)
    Q_rad = Q_rad_wg(Tw_C, Tg_C, h_r, params)
    Q_rad_sky = Q_rad_g_sky(Tg_C, params)
    Q_conv_amb = Q_conv_g_amb(Tg_C, params)
    
    # Thermal capacitances
    C_w = params["m_w"] * params["c_p_w"]
    C_g = params["m_g"] * params["c_p_g"]
    
    # CHANGED: Energy balance for water (Eq. 2 from paper)
    # Note: Paper includes collection efficiency in glass equation, not water
    dTw_dt = (Q_solar_w - Q_evap_val - Q_rad - Q_conv) / C_w
    
    # CHANGED: Energy balance for glass (Eq. 1 from paper)
    # Glass receives: solar absorption, heat from water (evap + radiation + convection)
    # Glass loses: radiation to sky, convection to ambient
    # Paper includes η_coll * Q_evap term
    dTg_dt = (Q_solar_g + Q_evap_val * params["eta_coll"] + Q_rad + Q_conv 
              - Q_rad_sky - Q_conv_amb) / C_g
    
    # Water collection rate
    m_dot = yield_rate(Tw_C, Tg_C, h_e, params)
    dMcol_dt = params["eta_coll"] * m_dot
    
    return [dTw_dt, dTg_dt, dMcol_dt]

# Run simulation
def run_simulation(params=None, t_span=(0, 3600 * 24), y0=None):
    if params is None:
        params = default_parameters()
    if y0 is None:
        # CHANGED: Initial conditions in Celsius (paper starts at ~30°C water, 25°C glass)
        y0 = [30.0, 25.0, 0.0]  # Tw=30°C, Tg=25°C, M_col=0

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

# ================================================================================================================
# SIMULATION & VISUALIZATION
# ================================================================================================================

if __name__ == "__main__":
    params = default_parameters()
    result = run_simulation(params, t_span=(0, 3600 * 24))

    Tw_C, Tg_C, M_col = result.y  # Temperatures now in Celsius
    t_hours = result.t / 3600
    A = params["A_w"]

    avg_Tw = np.mean(Tw_C)
    avg_Tg = np.mean(Tg_C)
    total_kg = M_col[-1]
    per_m2_day = total_kg / A

    print("\n" + "="*60)
    print("SIMULATION RESULTS (Paper Model)")
    print("="*60)
    print(f"Average water temperature (°C): {avg_Tw:.2f}")
    print(f"Average glass temperature (°C): {avg_Tg:.2f}")
    print(f"Collected water total (kg/day): {total_kg:.3f}")
    print(f"Collected water (kg/m²·day): {per_m2_day:.3f}")
    print(f"Efficiency (approx): {(total_kg * params['h_fg']) / (np.trapz([G_t_from_regression(t) for t in result.t], result.t) * A * params['tau_g'] * params['alpha']) * 100:.2f}%")
    print("="*60 + "\n")

    # Visualizations (same structure as before)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    ax1.plot(t_hours, Tw_C, 'r-', label='Water Temperature', linewidth=2)
    ax1.plot(t_hours, Tg_C, 'b-', label='Glass Temperature', linewidth=2)
    ax1.axhline(y=params["T_amb"] - 273.15, color='green', linestyle='--', 
                label='Ambient Temperature', linewidth=1.5)
    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax1.set_title('Temperature Profiles Over 24 Hours (Paper Model)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 24])
    
    ax2.plot(t_hours, M_col, 'purple', linewidth=2)
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Cumulative Water Collected (kg)', fontsize=12)
    ax2.set_title('Cumulative Water Collection Over 24 Hours', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 24])
    ax2.fill_between(t_hours, M_col, alpha=0.3, color='purple')
    
    plt.tight_layout()
    plt.show()
    
    # Additional plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    G_values = [G_t_from_regression(t * 3600) for t in t_hours]
    
    ax1.plot(t_hours, G_values, 'orange', linewidth=2)
    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('Solar Irradiance (W/m²)', fontsize=12)
    ax1.set_title('Solar Irradiance Over 24 Hours', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 24])
    ax1.fill_between(t_hours, G_values, alpha=0.3, color='orange')
    
    Q_solar_values = [Q_solar_absorbed(params, t) for t in result.t]
    Q_evap_values = []
    Q_conv_values = []
    
    for i in range(len(result.t)):
        h_c = h_c_wg(Tw_C[i], Tg_C[i], params)
        h_e = h_e_wg(Tw_C[i], Tg_C[i], h_c, params)
        Q_evap_values.append(Q_evap(Tw_C[i], Tg_C[i], h_e, params))
        Q_conv_values.append(Q_conv_wg(Tw_C[i], Tg_C[i], h_c, params))
    
    ax2.plot(t_hours, Q_solar_values, 'orange', label='Solar Absorbed', linewidth=2)
    ax2.plot(t_hours, Q_evap_values, 'cyan', label='Evaporative', linewidth=2)
    ax2.plot(t_hours, Q_conv_values, 'red', label='Convective', linewidth=2)
    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Heat Flux (W)', fontsize=12)
    ax2.set_title('Heat Fluxes Over 24 Hours', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 24])
    
    plt.tight_layout()
    plt.show()