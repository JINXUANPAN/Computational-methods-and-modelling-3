import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

file_path = 'solar.csv'
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