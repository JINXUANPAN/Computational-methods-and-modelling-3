import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Set the file path relative to the script location
file_path = os.path.join(script_dir, 'solar.csv')
DUMMY_YEAR = 2024 # Used to help Python parse dates that are missing a year #

# Create output directory in the same location as the script
output_dir = os.path.join(script_dir, "CSV Files and Plots Generated")
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Read the raw data without a header for processing #
df_raw = pd.read_csv(file_path, header=None)

data_list = []
current_date_str = None
month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Clean and extract historical solar data #
print("Starting data extraction and cleaning...")

for index, row in df_raw.iterrows():
    col_a = row.iloc[0]
    col_b = row.iloc[1]

    # Date Detection #
    if pd.notna(col_a) and isinstance(col_a, str):
        if any(month in col_a for month in month_names):
            current_date_str = col_a.split(',')[0].strip()
            continue

    # Data Row Detection #
    if current_date_str and pd.notna(col_a) and isinstance(col_a, str) and ':' in col_a:
        try:
            irradiance = pd.to_numeric(col_b, errors='coerce')
            if pd.notna(irradiance):
                datetime_str = f"{current_date_str} {DUMMY_YEAR} {col_a}"
                data_list.append({
                    'DateTime': datetime_str,
                    'Solar_Irradiance_W_m2': irradiance
                })
        except:
            continue

# Create the final cleaned database #
df_cleaned = pd.DataFrame(data_list)

if df_cleaned.empty:
    print("Error: No solar irradiance data was extracted. Check the CSV format.")
    # Exit or raise error, as subsequent steps will fail without data
    exit()

# Final database preparations #
date_format = '%A %B %d %Y %H:%M'
df_cleaned['DateTime'] = pd.to_datetime(df_cleaned['DateTime'], format=date_format, errors='coerce')
df_forecast = df_cleaned.set_index('DateTime')
df_forecast = df_forecast[df_forecast.index.notna()]
df_forecast = df_forecast.sort_index()

# Sort hours by time #
def time_to_seconds(time_str):
    h, m = map(int, time_str.split(':'))
    return h * 3600 + m * 60

# Calculate Hourly Average and Percentage Distribution #
df_forecast['Hour_of_Day'] = df_forecast.index.strftime('%H:%M')
df_hourly_avg = df_forecast.groupby('Hour_of_Day')['Solar_Irradiance_W_m2'].mean().reset_index()
df_hourly_avg.columns = ['Hour', 'Average Irradiance (W/m2)']

# Sort table #
df_hourly_avg = df_hourly_avg.assign(Time_Sort=df_hourly_avg['Hour'].apply(time_to_seconds))
df_hourly_avg = df_hourly_avg.sort_values('Time_Sort').drop(columns=['Time_Sort'])

# Calculate Percentage of Daily Total #
total_daily_avg = df_hourly_avg['Average Irradiance (W/m2)'].sum()
df_hourly_avg['Percentage of Daily Total'] = (df_hourly_avg['Average Irradiance (W/m2)'] / total_daily_avg) * 100

# 🛑 CHANGE THIS VALUE 🛑 to the input total daily irradiance 
DAILY_TOTAL = 4000 # Example: 4000 W/m2 total for the day

# Create a new column applying the percentage distribution #
df_hourly_avg['Estimated Irradiance (W/m2)'] = (df_hourly_avg['Percentage of Daily Total'] / 100) * DAILY_TOTAL

# Display the estimated results
print("\n" + "="*60)
print(f"--- Hourly Irradiance Estimated from a Daily Total of {DAILY_TOTAL:.2f} W/m2 ---")
print("="*60)
print(df_hourly_avg[['Hour', 'Percentage of Daily Total', 'Estimated Irradiance (W/m2)']].to_string(index=False, float_format='%.2f'))
print("="*60)

# Create combined plot with subplots
def create_combined_plot(df, filename):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Average Irradiance
    bars1 = ax1.bar(df['Hour'], df['Average Irradiance (W/m2)'], color='#3498db', alpha=0.7)
    ax1.set_ylabel('Average Solar Irradiance ($W/m^2$)')
    ax1.set_title('Historical Average Hourly Solar Irradiance Forecast')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot 2: Percentage of Daily Total
    bars2 = ax2.bar(df['Hour'], df['Percentage of Daily Total'], color='#2ecc71', alpha=0.7)
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Percentage of Daily Average Irradiance (%)')
    ax2.set_title('Percentage of Daily Average Irradiance by Hour')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set x-ticks for both subplots
    hours = df['Hour'].tolist()
    tick_indices = range(0, len(hours), 2)
    tick_labels = [hours[i] for i in tick_indices]
    
    ax1.set_xticks(tick_indices)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    # Set y-ticks for percentage plot
    max_percentage = df['Percentage of Daily Total'].max()
    ax2.set_yticks(np.arange(0, max_percentage + 5, 5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Combined plot saved as: {os.path.join(output_dir, filename)}")

# Function to auto-fit column widths in Excel
def auto_fit_columns(worksheet):
    for column in worksheet.columns:
        max_length = 0
        column_letter = column[0].column_letter
        for cell in column:
            try:
                if len(str(cell.value)) > max_length:
                    max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = (max_length + 2) * 1.2
        worksheet.column_dimensions[column_letter].width = adjusted_width

# Generate combined Excel file with multiple sheets and auto-fitted columns
def create_excel_file(df, filename):
    excel_path = os.path.join(output_dir, filename)
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Sheet 1: Complete Data
        df.to_excel(writer, sheet_name='Complete Data', index=False)
        
        # Sheet 2: Average Irradiance Only
        df[['Hour', 'Average Irradiance (W/m2)']].to_excel(writer, sheet_name='Average Irradiance', index=False)
        
        # Sheet 3: Percentage Distribution Only
        df[['Hour', 'Percentage of Daily Total']].to_excel(writer, sheet_name='Percentage Distribution', index=False)
        
        # Sheet 4: Estimated Irradiance
        df[['Hour', 'Estimated Irradiance (W/m2)']].to_excel(writer, sheet_name='Estimated Irradiance', index=False)
        
        # Auto-fit columns for all sheets
        workbook = writer.book
        for sheet_name in workbook.sheetnames:
            worksheet = workbook[sheet_name]
            auto_fit_columns(worksheet)
    
    print(f"Excel file with auto-fitted columns saved as: {excel_path}")

# Generate files in the output directory

# Create combined plot
create_combined_plot(df_hourly_avg, 'combined_solar_irradiance_plots.png')

# Create Excel file with multiple sheets and auto-fitted columns
create_excel_file(df_hourly_avg, 'solar_irradiance_data.xlsx')

print(f"\nScript completed. All files saved in: {output_dir}/")
print("Files created:")
print("- combined_solar_irradiance_plots.png (both plots combined)")
print("- solar_irradiance_data.xlsx (with 4 sheets and auto-fitted columns)")