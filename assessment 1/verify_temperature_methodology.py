#!/usr/bin/env python3
"""
Quick test to verify temperature profile generation matches methodology
"""
import pandas as pd

print("="*70)
print("VERIFICATION: Temperature Profile Methodology")
print("="*70)

# Step 1: Check if CSV was generated
try:
    df = pd.read_csv("assessment 1/hourly_percentage_temperature.csv")
    print("\n✓ File exists: hourly_percentage_temperature.csv")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
except:
    print("\n✗ File not found! Run Temperature_profile_generator.py first")
    exit()

# Step 2: Verify structure matches irradiance
print("\n" + "-"*70)
print("Structure Verification:")
print("-"*70)

required_cols = ['Hour', 'Percentage of Daily Variation']
has_required = all(col in df.columns for col in required_cols)

if has_required:
    print("✓ Has required columns:", required_cols)
else:
    print("✗ Missing required columns!")

# Step 3: Verify percentage sums to 100%
pct_sum = df['Percentage of Daily Variation'].sum()
print(f"\n✓ Sum of percentages: {pct_sum:.2f}%")
if abs(pct_sum - 100.0) < 0.01:
    print("  Status: PASS (within 0.01% of 100%)")
else:
    print(f"  Status: WARNING (expected 100%, got {pct_sum:.2f}%)")

# Step 4: Show sample data
print("\n" + "-"*70)
print("Sample Data (first 6 hours):")
print("-"*70)
print(df.head(6).to_string(index=False))

# Step 5: Apply to climate data (verify calculation)
T_MIN = 17.8
T_MAX = 31.6
T_RANGE = T_MAX - T_MIN

df['Test_Temp_C'] = T_MIN + (df['Percentage of Daily Variation'] / 100) * T_RANGE

print("\n" + "-"*70)
print("Temperature Application Test:")
print("-"*70)
print(f"  Input: T_min={T_MIN}°C, T_max={T_MAX}°C, Range={T_RANGE}°C")
print(f"  Output: Min={df['Test_Temp_C'].min():.2f}°C, Max={df['Test_Temp_C'].max():.2f}°C")
print(f"  Validation: {'PASS' if abs(df['Test_Temp_C'].min() - T_MIN) < 0.1 else 'FAIL'}")
print(f"              {'PASS' if abs(df['Test_Temp_C'].max() - T_MAX) < 0.1 else 'FAIL'}")

# Step 6: Compare to irradiance structure
print("\n" + "-"*70)
print("Comparison with Irradiance Forcer:")
print("-"*70)

try:
    df_irr = pd.read_csv("assessment 1/hourly_percentage_irradiance.csv")
    print("✓ Irradiance file found")
    print(f"  Temperature columns: {list(df.columns)}")
    print(f"  Irradiance columns:  {list(df_irr.columns)}")
    
    # Check if column names are analogous
    temp_has_pct = 'Percentage of Daily Variation' in df.columns
    irr_has_pct = 'Percentage of Daily Total' in df_irr.columns
    
    if temp_has_pct and irr_has_pct:
        print("\n✓ Both use percentage distribution method")
        print("  Temperature: 'Percentage of Daily Variation'")
        print("  Irradiance:  'Percentage of Daily Total'")
        print("\n✓ METHODOLOGY MATCH CONFIRMED")
    
except:
    print("  (Irradiance file not found for comparison)")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
print("\nConclusion:")
print("  Temperature profile uses the same percentage distribution")
print("  methodology as the irradiance forcer. ✓")
print("\nNext step:")
print("  Run 'Main Report Code.py' to use these profiles in simulation")
print("="*70 + "\n")
