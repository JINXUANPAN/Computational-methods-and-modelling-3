# Computational-Methods-and-Modelling-3
Computational Methods and Modelling 3 | Group 6 | September - November 2025 | Optimised Solar Still

## Project Overview
Over 97% of the Earth’s water is saltwater, and with 2.1 billion people across the planet still without access
to safely managed drinking water, desalination via solar stills could provide a solution to households faced
with this issue. This project models the transient thermal behaviour and freshwater production of a single-basin
passive solar still. Its goal is to optimisie the basin's area to meet the daily freshwater demand for an
average household in Algeria.

## Key features and Numerical methods
### Solar Irradiance forcer
Since historical hourly solar irradiance data was not available, an "hourly irradiance forcer" was created to
estimate the solar irradiance per hour, given a daily total. 
This feature uses cubic polynomial regression to provide a continuous solar irradiance function, which will then
be utilised in the ODE solver.

### System Modelling
The solar still was described as a system with two lumped thermal nodes, representing the basin water and its
glass cover. Their heat exchange was modelled by a pair of coupled ordinary differential equations (ODE's), 
which were solved using the Runge-Kutta 4/5 method, via 'scipy.integrate.solve_ivp'. This method solved for the
temperatures and evaporation rates of the solar still over a 10 hour period (chosen to represent daylight hours, 
during which solar irradiance, and therefore evaporation, is significant).

### Design Optimisation
To find the minimum basin area required to meet the target water yield, a root finding algorithm was used. Brent's
method, via 'scipy.optimise.brentq', was incorporated, which incrementally adjusted the area of the basin until the
predicted water yield matched the demands of the scenario. 

## Project Structure
* `src/`: Contains the source code for the models.
    * `regression.py`: Handles the solar irradiance data fitting.
    * `ode_solver.py`: Defines the energy balance equations and Runge-Kutta 4/5 solver.
    * `optimization.py`: Implements Brent's method for area sizing.
* `docs/`: Project documentation.
* `data/`: Input datasets (e.g., Algeria solar forecast).

## How to use
To run the project, please follow these steps:

```Get the code:
bash
git clone [https://github.com/JINXUANPAN/Computational-methods-and-modelling-3.git](https://github.com/JINXUANPAN/Computational-methods-and-modelling-3.git)
  
Install the required libraries:
pip install numpy scipy matplotlib

Run the scripts in the following order:

python "src/Irradiance Forcer Calculations.py"
python "src/Main Report.py"
```

After running these scripts, you will find the generated results and plots in the folder "src/CSV Files and Plots Generated."


