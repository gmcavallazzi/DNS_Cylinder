# Cylinder Flow Simulation with Adams-Bashforth Integration

## Overview
This project implements a high-performance computational fluid dynamics (CFD) simulation of incompressible flow past a cylindrical obstacle using the fractional step method with Adams-Bashforth time integration.

## Key Features
- 2D incompressible flow solver
- Adams-Bashforth 2nd order time integration
- Flexible boundary conditions (periodic/inlet-outlet)
- Adaptive time-stepping 
- Vortex shedding analysis
- Detailed force and flow field visualization

## Physical Model
- Solves Navier-Stokes equations for incompressible flow
- Uses staggered grid discretization
- Supports various Reynolds numbers
- Optional cylindrical obstacle simulation

## Performance Optimizations
- JIT compilation with Numba
- Vectorized and parallelized computational kernels
- Efficient memory management
- Grid stretching for enhanced near-wall resolution

## Dependencies
- NumPy
- Numba
- Matplotlib
- SciPy (optional, for advanced analysis)
- PyYAML

## Simulation Configuration
Configuration is managed through a `config.yaml` file allowing customization of:
- Grid parameters
- Flow conditions
- Simulation settings
- Boundary conditions
- Perturbation injection
- Cylinder obstacle parameters

## Key Modules
- `main.py`: Primary simulation driver
- `solver.py`: Core numerical methods
- `adams_bashforth.py`: Time integration scheme
- `cylinder_visualization.py`: Flow field analysis
- `forces_analysis.py`: Force coefficient calculations
- `turbulent_helpers.py`: Initial condition and perturbation generation

## Typical Workflow
1. Configure simulation parameters in `config.yaml`
2. Run `main.py`
3. Analyze generated results in `results/` directory

## Visualization Outputs
- Velocity fields
- Vorticity contours
- Force coefficient histories
- Pressure distributions
- Space-time diagrams

## Example Configuration Scenarios
- Channel flow simulation
- Flow past a cylinder at different Reynolds numbers
- Turbulence transition studies

## Performance Metrics Tracked
- Bulk velocity
- Pressure gradients
- Wall shear stresses
- Computational efficiency
- Numerical stability

## Numerical Methods
- Fractional step method
- Adams-Bashforth time integration
- Finite difference discretization
- Pressure Poisson solver with SOR relaxation

## Analysis Capabilities
- Vortex shedding frequency
- Drag and lift coefficient calculation
- Flow field statistical analysis
- Strouhal number computation

## Usage

### Basic Simulation Execution
```bash
# Run simulation with default configuration
python main.py

# Specify a custom configuration file
python main.py --config custom_config.yaml

# Override output directory
python main.py --output custom_results
```

### Configuration Options
The `config.yaml` file allows detailed customization of the simulation:

#### Grid Configuration
```yaml
grid:
  nx: 320               # Number of cells in x-direction
  ny: 256               # Number of cells in y-direction
  lx: 12.0              # Domain length in x-direction
  ly: 3.2               # Domain height in y-direction
  stretch_factor: 1.0   # Grid stretching factor for y-direction
  plot_grid: true       # Visualize grid distribution
```

#### Flow Parameters
```yaml
flow:
  reynolds: 150         # Reynolds number
  u_bulk: 1.0           # Target bulk velocity
```

#### Simulation Settings
```yaml
simulation:
  cfl: 0.4              # CFL number for timestep calculation
  t_end: 60.0           # Total simulation time
  save_interval: 5.0    # Interval for saving results
  output_dir: "results" # Output directory
```

#### Cylinder Obstacle Configuration
```yaml
cylinder:
  enabled: true         # Enable cylinder obstacle
  center_x_ratio: 0.2   # Cylinder x-position (ratio of domain length)
  center_y_ratio: 0.5   # Cylinder y-position (ratio of domain height)
  radius_ratio: 0.06    # Cylinder radius
```

### Performance and Debugging
- Monitor console output for simulation statistics
- Check `results/` directory for:
  - Velocity field visualizations
  - Force coefficient histories
  - Vorticity contours
  - Diagnostic plots

### Typical Workflow
1. Adjust `config.yaml` to match desired simulation parameters
2. Run simulation
3. Analyze output files in `results/` directory
4. Modify configuration for different flow conditions

### Advanced Usage
```bash
# For help and additional options
python main.py --help
```

### Recommended Environments
- Python 3.12+
- Virtual environment recommended
- Install dependencies: `pip install -r requirements.txt`

### Troubleshooting
- Ensure all dependencies are installed
- Check console output for numerical instability warnings
- Adjust CFL number or grid resolution if simulation diverges


## References
- Ferziger & Peric, Computational Methods for Fluid Dynamics
- Patankar, Numerical Heat Transfer and Fluid Flow