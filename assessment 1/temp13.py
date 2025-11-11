# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 13:02:15 2025

@author: Administrator
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

file_path = 'solar.csv' # Testing
df = pd.read_csv(file_path)


data = df.iloc[1:, [0, 1]].copy()
data.columns = ["Hour", "Solar_Irradiance"]

data = data.dropna()
data = data[data["Hour"].str.contains(":")]

data["Solar_Irradiance"] = pd.to_numeric(data["Solar_Irradiance"], errors="coerce")
data = data.dropna()

data["Hour"] = data["Hour"].str.replace(":00", "").astype(float)
data = data.groupby("Hour", as_index=False).mean().sort_values("Hour")

x = data["Hour"].values
y = data["Solar_Irradiance"].values
coeffs = np.polyfit(x, y, 3)
poly_func = np.poly1d(coeffs)


x_smooth = np.linspace(x.min(), x.max(), 300)
y_smooth = poly_func(x_smooth)

print("polynomial regression：")
print(f"y = {coeffs[0]:.3f}x³ + {coeffs[1]:.3f}x² + {coeffs[2]:.3f}x + {coeffs[3]:.3f}")
print("average solar irradiance =", np.mean(np.clip(y_smooth, 0, None)))
print("Maximum solar irradiance =", np.max(y_smooth))

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label="original data")
plt.plot(x_smooth, y_smooth, color='blue', label="polynomial regression curve")
equation_text = f"y = {coeffs[0]:.2f}x³ + {coeffs[1]:.2f}x² + {coeffs[2]:.2f}x + {coeffs[3]:.2f}"
plt.text(x.min()+0.3, y.max()*0.8, equation_text, color="blue", fontsize=12)
plt.xlabel("Hour")
plt.ylabel(r"Solar Irradiance (W/m$^2$)")
plt.title("smooth function")
plt.legend()
plt.grid(True)
plt.show()


def G_t_from_regression(t, scale_factor=2.0):
    hour = (t / 3600.0) % 24
    G = poly_func(hour)
    G = max(G, 0.0)
    return G * scale_factor


def default_parameters():
    params = {
        # Geometry
        "A": 1.0,
        # Water properties
        "m_w": 20.0,
        "c_p_w": 4184,
        # Glass properties
        "m_g": 1.0,
        "c_p_g": 800,
        # Heat transfer coefficients
        "h_wg": 18.0,
        "h_ga": 20.0,
        # Radiation parameters
        "epsilon_g": 0.9,
        "sigma": 5.67e-8,
        # Evaporation/condensation
        "k_m": 2e-4,
        "L_v": 2.45e6,
        "eta_coll": 0.8,
        # Environmental parameters
        "T_amb": 298.0,
        "RH": 0.6,
        # Solar parameters
        "alpha": 0.9,
        "G_t": 800.0,
    }
    return params


def p_sat(T):
    T = np.clip(T, 273.0, 373.0)
    return 610.78 * np.exp((17.27 * (T - 273.15)) / (T - 35.85))

def evaporation_flux(Tw, Tg, A, k_m, RH):
    Mw = 0.01801528
    R = 8.314
    T_avg = max((Tw + Tg) / 2.0, 273.0)
    delta_p = max(p_sat(Tw) - RH * p_sat(Tg), 0.0)
    m_dot_per_m2 = k_m * Mw * delta_p / (R * T_avg)
    flux = A * m_dot_per_m2
    return flux

def solar_absorbed(params, t):
    G_t = G_t_from_regression(t)
    return params["A"] * params["alpha"] * G_t

def Q_wg(Tw, Tg, params):
    return params["h_wg"] * params["A"] * (Tw - Tg)

def Q_gamb(Tg, params):
    conv = params["h_ga"] * params["A"] * (Tg - params["T_amb"])
    rad = params["epsilon_g"] * params["sigma"] * params["A"] * (Tg**4 - params["T_amb"]**4)
    return conv + rad


def solar_still_odes(t, y, params):
    Tw, Tg, M_col = y
    Cw = params["m_w"] * params["c_p_w"]
    Cg = params["m_g"] * params["c_p_g"]
    Lv = params["L_v"]
    eta_coll = params["eta_coll"]

    Q_solar = solar_absorbed(params, t)
    Q_wg_val = Q_wg(Tw, Tg, params)
    Q_gamb_val = Q_gamb(Tg, params)
    m_evap = evaporation_flux(Tw, Tg, params['A'], params['k_m'], params['RH'])

    dTw_dt = (Q_solar - Q_wg_val - m_evap * Lv) / Cw
    dTg_dt = (Q_wg_val - Q_gamb_val + eta_coll * m_evap * Lv) / Cg
    dMcol_dt = eta_coll * m_evap

    return [dTw_dt, dTg_dt, dMcol_dt]

def run_simulation(params=None, t_span=(0, 3600 * 24), y0=None):
    if params is None:
        params = default_parameters()
    if y0 is None:
        y0 = [303.0, 298.0, 0.0]

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



if __name__ == "__main__":
    params = default_parameters()
    result = run_simulation(params, t_span=(0, 3600 * 24))

    Tw, Tg, M_col = result.y
    A = params["A"]


    avg_Tw = np.mean(Tw - 273.15)
    avg_Tg = np.mean(Tg - 273.15)
    total_kg = M_col[-1]
    per_m2_day = total_kg / A

    print("Average water temperature (°C):", avg_Tw)
    print("Average glass temperature (°C):", avg_Tg)
    print("Collected water total (kg/day):", total_kg)
    print("Collected water (kg/m²·day):", per_m2_day)
