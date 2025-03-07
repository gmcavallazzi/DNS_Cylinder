# PyVortexDNS: High-Fidelity Cylinder Flow Simulator

A high-performance CFD solver for simulating channel flows and vortex shedding behind cylinders, using the fractional step method and Adams-Bashforth time integration.

## Features

- Incompressible Navier-Stokes solver on a staggered grid
- Parallel computation with Numba JIT compilation
- Adaptive time-stepping based on CFL condition
- Multiple boundary condition options
- Simulation of flow around a cylinder with vortex shedding analysis
- Force (drag/lift) calculation for immersed objects
- Automatic visualization of results

## Numerical Methods

### Governing Equations

The code solves the incompressible Navier-Stokes equations:

$$\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u}$$

$$\nabla \cdot \mathbf{u} = 0$$

Where:
- $\mathbf{u}$ is the velocity vector
- $p$ is pressure
- $\rho$ is density
- $\nu$ is kinematic viscosity

### Discretization

The solver uses:
- Staggered grid (MAC scheme) with pressure at cell centers and velocities at cell faces
- Finite difference discretization with second-order accuracy in space
- Adams-Bashforth second-order method for time integration
- Upwind scheme for convective terms
- Central differencing for diffusive terms

### Fractional Step Method

The time integration follows the fractional step procedure:

1. **Predictor Step**:
   $ \mathbf{u}^* = \mathbf{u}^n + \Delta t \left( \frac{3}{2}\mathbf{F}^n - \frac{1}{2}\mathbf{F}^{n-1} \right) $

2. **Pressure Poisson Equation**:
   $$ \nabla^2 p^{n+1} = \frac{\rho}{\Delta t} \nabla \cdot \mathbf{u}^* $$

3. **Corrector Step**:
   $$ \mathbf{u}^{n+1} = \mathbf{u}^* - \frac{\Delta t}{\rho} \nabla p^{n+1} $$

4. **Apply Boundary Conditions**: Update ghost cells to enforce desired boundary conditions

### Grid System

The code uses a stretched grid in the y-direction to better resolve near-wall regions:
- Hyperbolic tangent stretching function for y-coordinates
- Uniform grid spacing in the x-direction

## Code Structure

- `main.py`: Main simulation driver
- `solver.py`: Core solver functions (pressure projection, velocity correction)
- `adams_bashforth.py`: Time integration using Adams-Bashforth scheme
- `turbulent_helpers.py`: Functions to add perturbations and vortices
- `utils.py`: Utility functions for data analysis and statistics
- `cylinder_visualization.py`: Flow visualization around cylinders
- `forces_analysis.py`: Calculation of forces and pressure distribution
- `config.yaml`: Configuration file for simulation parameters

## Key Functions

- `run_simulation_ab2()`: Main simulation function
- `fractional_step_ab2()`: Time stepping with Adams-Bashforth
- `solve_pressure_poisson()`: Solves pressure equation
- `velocity_correction()`: Projects velocity field to be divergence-free
- `calculate_forces_cylinder()`: Calculates drag and lift forces
- `analyze_vortex_shedding()`: Analyzes vortex shedding frequency and Strouhal number

## Setup Instructions

### Environment Setup

1. Create a new conda environment:
   ```bash
   conda create -n cfd python=3.10
   conda activate cfd
   ```

2. Install required packages:
   ```bash
   conda install numpy matplotlib pyyaml scipy
   conda install -c conda-forge numba
   ```

### Running a Simulation

1. Configure the simulation by editing `config.yaml`. Key parameters include:
   - Grid resolution (`nx`, `ny`)
   - Domain size (`lx`, `ly`)
   - Reynolds number
   - Boundary conditions
   - Cylinder parameters (if applicable)

2. Run the simulation:
   ```bash
   python main.py --config config.yaml
   ```

3. With custom output directory:
   ```bash
   python main.py --config config.yaml --output custom_results
   ```

## Example Configurations

### Channel Flow

```yaml
grid:
  nx: 128
  ny: 64
  lx: 6.28
  ly: 2.0
  stretch_factor: 1.5

flow:
  reynolds: 180
  u_bulk: 1.0

bc:
  kind: "periodic"

cylinder:
  enabled: false
```

### Cylinder Flow

```yaml
grid:
  nx: 320
  ny: 256
  lx: 12.0
  ly: 3.2

flow:
  reynolds: 150
  u_bulk: 1.0

bc:
  kind: "inletoutlet"

cylinder:
  enabled: true
  center_x_ratio: 0.2
  center_y_ratio: 0.5
  radius_ratio: 0.06
```

## Output and Visualization

Results are saved in the specified output directory:
- PNG images of the flow field
- Force coefficient histories
- Vorticity space-time diagrams
- Metrics stored in NPZ files for further analysis

## Performance Optimization

The code uses several optimization techniques:
- Numba JIT compilation for performance-critical functions
- Parallel execution for key computational loops
- Adaptive time-stepping to maintain stability

## References

1. Chorin, A. J. (1968). Numerical solution of the Navier-Stokes equations.
2. Kim, J., & Moin, P. (1985). Application of a fractional-step method to incompressible Navier-Stokes equations.
3. Williamson, C. H. K. (1996). Vortex dynamics in the cylinder wake.