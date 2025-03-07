# Modified adams_bashforth.py with JIT-compatible functions
import numpy as np
from numba import jit
from solver import apply_bc, compute_fluxes_basic, solve_pressure_poisson, velocity_correction

@jit(nopython=True)
def fractional_step_ab2_jit(u, v, p, nx, ny, dx, dy, dt, nu, rho, bc, target_flow_rate=1.0, prev_rhs=None, 
                          use_cylinder=False, center_x=None, center_y=None, radius=None):
    """JIT-compatible implementation of Adams-Bashforth method"""
    # 1. Compute fluxes for current time step
    grid_data = (nx, ny, dx, dy)
    conv_u, conv_v, diff_u, diff_v = compute_fluxes_basic(u, v, grid_data, nu)
    
    # Calculate current RHS terms
    curr_rhs_u = -conv_u + diff_u
    curr_rhs_v = -conv_v + diff_v
    
    # 2. Calculate intermediate velocity field (u*)
    u_star = u.copy()
    v_star = v.copy()
    
    # If we have previous RHS terms, use Adams-Bashforth 2
    # Otherwise, use Euler for the first step
    if prev_rhs is not None:
        prev_rhs_u, prev_rhs_v = prev_rhs
        
        # Apply AB2 formula: y_{n+1} = y_n + dt * (3/2 * f(y_n) - 1/2 * f(y_{n-1}))
        for i in range(1, nx+2):
            for j in range(1, ny+1):
                u_star[i, j] = u[i, j] + dt * (1.5 * curr_rhs_u[i, j] - 0.5 * prev_rhs_u[i, j])
        
        for i in range(1, nx+1):
            for j in range(1, ny+2):
                v_star[i, j] = v[i, j] + dt * (1.5 * curr_rhs_v[i, j] - 0.5 * prev_rhs_v[i, j])
    else:
        # First step: use regular forward Euler
        for i in range(1, nx+2):
            for j in range(1, ny+1):
                u_star[i, j] = u[i, j] + dt * curr_rhs_u[i, j]
        
        for i in range(1, nx+1):
            for j in range(1, ny+2):
                v_star[i, j] = v[i, j] + dt * curr_rhs_v[i, j]
    
    # Apply boundary conditions to intermediate velocity
    u_star, v_star, _ = apply_bc(u_star, v_star, p.copy(), nx, ny, bc)
    
    # 3. Solve pressure Poisson equation
    p_new, iterations, residual = solve_pressure_poisson(p, u_star, v_star, grid_data, dt, rho)
    
    # 4. Velocity correction
    u_new, v_new, div_after = velocity_correction(u_star, v_star, p_new, nx, ny, dx, dy, dt, rho)

    # Apply cylinder boundary conditions if requested
    if use_cylinder and center_x is not None and center_y is not None and radius is not None:
        u_new, v_new = apply_cylinder_bc(u_new, v_new, nx, ny, dx, dy, center_x, center_y, radius)
    
    # 5. Apply boundary conditions to corrected velocities
    u_new, v_new, p_new = apply_bc(u_new, v_new, p_new, nx, ny)
    
    # 6. Enforce constant mean flow rate
    if target_flow_rate > 0:
        # Calculate current mean velocity
        mean_velocity = 0.0
        total_points = 0
        for j in range(1, ny+1):
            for i in range(1, nx+2):
                mean_velocity += u_new[i, j]
                total_points += 1
        
        mean_velocity /= total_points
        
        # Apply correction
        if abs(mean_velocity) > 1e-10:
            correction = target_flow_rate / mean_velocity
            
            # Limit correction factor for stability
            correction = max(0.5, min(2.0, correction))
            
            for j in range(1, ny+1):
                for i in range(1, nx+2):
                    u_new[i, j] *= correction
    
    # Return updated fields, diagnostics, and current RHS for next step
    return u_new, v_new, p_new, conv_u, conv_v, diff_u, diff_v, iterations, div_after, (curr_rhs_u, curr_rhs_v)

def fractional_step_ab2(u, v, p, grid, dt, nu, rho, bc, target_flow_rate=1.0, prev_rhs=None, cylinder_params=None):
    """Wrapper function that extracts grid components before calling JIT function"""
    # Extract grid parameters
    nx, ny = grid['nx'], grid['ny']
    dx, dy = grid['dx'], grid['dy']
    lx, ly = grid['lx'], grid['ly']
    
    # Process cylinder parameters
    use_cylinder = False
    center_x = None
    center_y = None
    radius = None
    
    if cylinder_params is not None:
        use_cylinder = True
        # Calculate actual positions from ratios
        center_x = cylinder_params.get('center_x_ratio', 0.25) * lx
        center_y = cylinder_params.get('center_y_ratio', 0.5) * ly
        radius = cylinder_params.get('radius_ratio', 0.125) * ly
        
    return fractional_step_ab2_jit(u, v, p, nx, ny, dx, dy, dt, nu, rho, bc, target_flow_rate, prev_rhs,
                                 use_cylinder, center_x, center_y, radius)

@jit(nopython=True)
def apply_cylinder_bc(u, v, nx, ny, dx, dy, center_x, center_y, radius):
    """Apply boundary conditions for a cylinder in the domain"""
    for i in range(1, nx+2):
        for j in range(1, ny+1):
            # Position of u-velocity point
            x = (i-1) * dx
            y = (j-0.5) * dy
            
            # Check if point is inside cylinder
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist <= radius:
                u[i, j] = 0.0  # No-slip
    
    for i in range(1, nx+1):
        for j in range(1, ny+2):
            # Position of v-velocity point
            x = (i-0.5) * dx
            y = (j-1) * dy
            
            # Check if point is inside cylinder
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist <= radius:
                v[i, j] = 0.0  # No-slip
    
    return u, v