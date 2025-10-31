import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

file_path = 'assessment 1/solar.csv' #Testing
df = pd.read_csv(file_path)

#Extract valid data columns
data = df.iloc[1:, [0, 1]].copy()
data.columns = ["Hour", "Solar_Irradiance"]

#Remove rows with null or non-numeric values
data = data.dropna()

#Keep rows with valid time format
data = data[data["Hour"].str.contains(":")]

#make solar irradiance numeric
data["Solar_Irradiance"] = pd.to_numeric(data["Solar_Irradiance"], errors="coerce")
data = data.dropna()

#Convert "08:00" to the numeric value 8.0
data["Hour"] = data["Hour"].str.replace(":00", "").astype(float)

#If the same hour repeats, take the average.
data = data.groupby("Hour", as_index=False).mean().sort_values("Hour")

#Polynomial
x = data["Hour"].values
y = data["Solar_Irradiance"].values
coeffs = np.polyfit(x, y, 3)
poly_func = np.poly1d(coeffs)

#smooth
x_smooth = np.linspace(x.min(), x.max(), 300)
y_smooth = poly_func(x_smooth)

print("polynomial regression：")
print(f"y = {coeffs[0]:.3f}x³ + {coeffs[1]:.3f}x² + {coeffs[2]:.3f}x + {coeffs[3]:.3f}")

#plot
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

from scipy.integrate import solve_ivp

def default_parameters():
    
    params = {
        # Geometry
        "A": 1.0,                  
        # Water properties
        "m_w": 2.0,                
        "c_p_w": 4184,             
        # Glass properties
        "m_g": 1.0,                
        "c_p_g": 800,             
        # Heat transfer coefficients
        "h_wg": 10.0,              
        "h_ga": 8.0,               
        # Radiation parameters
        "epsilon_g": 0.9,          
        "sigma": 5.67e-8,          
        # Evaporation/condensation
        "k_m": 0.00025,            
        "L_v": 2.45e6,             
        "eta_coll": 0.95,          
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
    flux = A * k_m * max(p_sat(Tw) - RH * p_sat(Tg), 0)
    return min(flux, 0.05) 

def solar_absorbed(params):
    return params["A"] * params["alpha"] * params["G_t"]

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

    Q_solar = solar_absorbed(params)
    Q_wg_val = Q_wg(Tw, Tg, params)
    Q_gamb_val = Q_gamb(Tg, params)
    m_evap = evaporation_flux(Tw, Tg, params['A'], params['k_m'], params['RH'])

    dTw_dt = (Q_solar - Q_wg_val - m_evap * Lv) / Cw
    dTg_dt = (Q_wg_val - Q_gamb_val + eta_coll * m_evap * Lv) / Cg
    dMcol_dt = eta_coll * m_evap

    return [dTw_dt, dTg_dt, dMcol_dt]


def run_simulation(params=None, t_span=(0, 3600), y0=None):
    """Run the solar still ODE simulation."""
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
        max_step=5.0,
        dense_output=True
    )
    return sol

if __name__ == "__main__":
    params = default_parameters()
    result = run_simulation(params, t_span=(0, 3600 * 24)) 

    Tw, Tg, M_col = result.y
    print("Final water temperature (C):", Tw[-1] - 273.15)
    print("Final glass temperature (C):", Tg[-1] - 273.15)
    print("Collected water (kg):", M_col[-1])
