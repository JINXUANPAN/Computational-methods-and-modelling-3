#!/usr/bin/env python3
"""
Quick preview of temperature profiles - shows what your integrated model will use
"""
import numpy as np
import matplotlib.pyplot as plt

# Your climate data
T_MIN = 17.8
T_MAX = 31.6
T_MEAN = 24.67

# Generate time array (24 hours)
t_hours = np.linspace(0, 24, 1000)

# Create ambient temperature profile
T_mean_calc = (T_MAX + T_MIN) / 2
T_amplitude = (T_MAX - T_MIN) / 2
phase_shift = 6.0  # Minimum at 6:00 AM

angle = 2 * np.pi * (t_hours - phase_shift) / 24.0
T_amb = T_mean_calc - T_amplitude * np.cos(angle)

# Calculate sky temperature
T_amb_K = T_amb + 273.15
T_sky_K = 0.0552 * (T_amb_K ** 1.5)
T_sky = T_sky_K - 273.15

# Create plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Top plot: Temperature profiles
ax1.plot(t_hours, T_amb, 'r-', linewidth=2.5, label='Ambient Temperature')
ax1.plot(t_hours, T_sky, 'b-', linewidth=2.5, label='Sky Temperature')
ax1.axhline(y=T_MIN, color='gray', linestyle='--', alpha=0.5)
ax1.axhline(y=T_MAX, color='gray', linestyle='--', alpha=0.5)
ax1.axhline(y=T_MEAN, color='gray', linestyle=':', alpha=0.5)
ax1.text(1, T_MIN+0.5, f'Min: {T_MIN}°C', fontsize=10)
ax1.text(1, T_MAX-0.5, f'Max: {T_MAX}°C', fontsize=10, va='top')
ax1.text(1, T_MEAN+0.5, f'Mean: {T_MEAN}°C', fontsize=10)
ax1.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
ax1.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
ax1.set_title('Integrated Temperature Profiles (Now Used in Solar Still Model)', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 24)
ax1.set_xticks(range(0, 25, 2))

# Bottom plot: Temperature difference
temp_diff = T_amb - T_sky
ax2.fill_between(t_hours, 0, temp_diff, alpha=0.3, color='orange')
ax2.plot(t_hours, temp_diff, 'orange', linewidth=2)
ax2.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
ax2.set_ylabel('ΔT = T_amb - T_sky (°C)', fontsize=12, fontweight='bold')
ax2.set_title('Radiative Cooling Potential (Drives Glass Heat Loss)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 24)
ax2.set_xticks(range(0, 25, 2))
ax2.text(12, np.max(temp_diff)*0.85, 
         f'Average ΔT = {np.mean(temp_diff):.1f}°C\n(10-20°C is typical for clear sky)',
         ha='center', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.savefig('temperature_preview.png', dpi=150)
print("✓ Preview saved: temperature_preview.png")
print(f"\nTemperature Statistics:")
print(f"  Ambient: {T_amb.min():.2f}°C to {T_amb.max():.2f}°C (mean: {T_amb.mean():.2f}°C)")
print(f"  Sky:     {T_sky.min():.2f}°C to {T_sky.max():.2f}°C (mean: {T_sky.mean():.2f}°C)")
print(f"  ΔT:      {temp_diff.min():.2f}°C to {temp_diff.max():.2f}°C (mean: {temp_diff.mean():.2f}°C)")
plt.show()
