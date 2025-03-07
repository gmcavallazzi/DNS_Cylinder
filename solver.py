import numpy as np
from numba import jit
from numba import prange

@jit(nopython=True)
def apply_bc(u, v, p, nx, ny, bc='periodic', u_bulk=1.0):
    # 1. Apply no-slip conditions at walls (y-direction)
    # For u-velocity
    u[:, 0]    = u[:, 1]    # Bottom wall ghost
    u[:, ny+1] = u[:, ny]   # Top wall ghost
    # Uncomment the following for solid walls
    #u[:, 0]    = -u[:, 1]    # Bottom wall ghost
    #u[:, ny+1] = -u[:, ny]   # Top wall ghost
    
    # Anti-symmetric ghost cells for v
    v[:, 0]    = -v[:, 2]    # Bottom ghost
    v[:, ny+2] = -v[:, ny]   # Top ghost
    
    # 2. Apply boundary conditions in x-direction based on type       
    if bc == 'periodic':
        # For u-velocity
        u[0, :]    = u[nx+1, :]   # Left ghost = right interior
        u[nx+2, :] = u[1, :]      # Right ghost = left interior
        
        # For v-velocity
        v[0, :]    = v[nx, :]     # Left ghost = right interior
        v[nx+1, :] = v[1, :]      # Right ghost = left interior
        
        # For pressure
        p[0, :]    = p[nx, :]     # Left ghost = right interior
        p[nx+1, :] = p[1, :]      # Right ghost = left interior
    
    elif bc == 'inletoutlet':
        # Apply inlet (left boundary) with parabolic profile
        for j in range(1, ny+1):
            # Normalized position (0 at bottom wall, 1 at top wall)
            y_norm = float(j) / float(ny)
            
            # Parabolic profile: u = 6*u_bulk*y*(1-y) for unit height
            u[1, j] = 6.0 * u_bulk * y_norm * (1.0 - y_norm)
        
        # Set ghost cell for u at inlet
        u[0, :] = 2.0 * u[1, :] - u[2, :]
        
        # Zero gradient at outlet (right boundary)
        u[nx+2, :] = u[nx+1, :]
        
        # Zero gradient for v at both inlet and outlet
        v[0, :]    = v[1, :]
        v[nx+1, :] = v[nx, :]
        
        # Zero gradient for pressure at inlet
        p[0, :] = p[1, :]
        
        # Fixed pressure at outlet
        p[nx+1, :] = 0.0

    # Neumann BCs for pressure at walls
    p[:, 0] = p[:, 1]         # Bottom wall
    p[:, ny+1] = p[:, ny]     # Top wall
    
    return u, v, p

@jit(nopython=True)
def compute_fluxes_basic(u, v, grid_data, nu):
    nx, ny, dx, dy = grid_data
    
    # Initialize arrays with correct dimensions
    conv_u = np.zeros_like(u)
    conv_v = np.zeros_like(v)
    diff_u = np.zeros_like(u)
    diff_v = np.zeros_like(v)
    
    # For u-momentum equation (interior u-points)
    for i in range(1, nx+2):
        for j in range(1, ny+1):
            # Convective terms - use basic upwind for stability
            # u * du/dx term
            if u[i, j] >= 0:
                du_dx = (u[i, j] - u[i-1, j]) / dx
            else:
                du_dx = (u[i+1, j] - u[i, j]) / dx
            
            # v * du/dy term - interpolate v to u-position
            v_at_u = 0.25 * (v[i-1, j] + v[i, j] + v[i-1, j+1] + v[i, j+1])
            
            if v_at_u >= 0:
                du_dy = (u[i, j] - u[i, j-1]) / dy
            else:
                du_dy = (u[i, j+1] - u[i, j]) / dy
            
            conv_u[i, j] = u[i, j] * du_dx + v_at_u * du_dy
            
            # Diffusive terms - central difference
            d2u_dx2 = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / (dx*dx)
            d2u_dy2 = (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / (dy*dy)
            diff_u[i, j] = nu * (d2u_dx2 + d2u_dy2)
    
    # For v-momentum equation (interior v-points)
    for i in range(1, nx+1):
        for j in range(1, ny+2):
            # Convective terms
            # u * dv/dx term - interpolate u to v-position
            u_at_v = 0.25 * (u[i, j-1] + u[i+1, j-1] + u[i, j] + u[i+1, j])
            
            if u_at_v >= 0:
                dv_dx = (v[i, j] - v[i-1, j]) / dx
            else:
                dv_dx = (v[i+1, j] - v[i, j]) / dx
            
            # v * dv/dy term
            if v[i, j] >= 0:
                dv_dy = (v[i, j] - v[i, j-1]) / dy
            else:
                dv_dy = (v[i, j+1] - v[i, j]) / dy
            
            conv_v[i, j] = u_at_v * dv_dx + v[i, j] * dv_dy
            
            # Diffusive terms - central difference
            d2v_dx2 = (v[i+1, j] - 2*v[i, j] + v[i-1, j]) / (dx*dx)
            d2v_dy2 = (v[i, j+1] - 2*v[i, j] + v[i, j-1]) / (dy*dy)
            diff_v[i, j] = nu * (d2v_dx2 + d2v_dy2)
    
    return conv_u, conv_v, diff_u, diff_v

@jit(nopython=True)
def solve_pressure_poisson(p, u_star, v_star, grid_data, dt, rho, max_iter=100, tol=1e-5):
    nx, ny, dx, dy = grid_data
    
    # Create copy of pressure for iteration
    p_new = p.copy()
    
    # Calculate divergence of intermediate velocity field at cell centers
    div = np.zeros((nx+2, ny+2))
    
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            # Use velocity components on cell faces surrounding pressure point (i,j)
            u_east = u_star[i+1, j]  # u at (i+1/2, j)
            u_west = u_star[i, j]    # u at (i-1/2, j)
            v_north = v_star[i, j+1] # v at (i, j+1/2)
            v_south = v_star[i, j]   # v at (i, j-1/2)
            
            # Calculate divergence
            div[i, j] = ((u_east - u_west) / dx + (v_north - v_south) / dy) / dt
    
    # Use conservative SOR parameter to start
    omega = 1.7  # Gauss-Seidel (no over-relaxation)
    
    # Iterative solution
    for iter in range(max_iter):
        max_residual = 0.0
        
        # Update interior pressure points
        for i in range(1, nx+1):
            for j in range(1, ny+1):
                # Standard 5-point stencil for Laplacian
                p_old = p_new[i, j]
                
                p_new[i, j] = (1.0 - omega) * p_old + omega * (
                    (p_new[i+1, j] + p_new[i-1, j]) / (dx*dx) +
                    (p_new[i, j+1] + p_new[i, j-1]) / (dy*dy) -
                    rho * div[i, j]
                ) / (2.0 / (dx*dx) + 2.0 / (dy*dy))
                
                # Track residual
                residual = abs(p_new[i, j] - p_old)
                max_residual = max(max_residual, residual)
        
        # Apply boundary conditions every iteration
        # Periodic in x
        p_new[0, :] = p_new[nx, :]
        p_new[nx+1, :] = p_new[1, :]
        
        # Neumann at walls (y)
        p_new[:, 0] = p_new[:, 1]
        p_new[:, ny+1] = p_new[:, ny]
        
        # Check convergence
        if max_residual < tol:
            break
        
        # Increase omega slightly after enough iterations if converging well
        if iter == 50 and max_residual < 0.1 * tol:
            omega = min(1.75, omega * 1.1)
    
    # Normalize pressure (remove constant component)
    p_sum = 0.0
    count = 0
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            p_sum += p_new[i, j]
            count += 1
    
    if count > 0:
        p_mean = p_sum / count
        p_new -= p_mean
    
    return p_new, iter+1, max_residual

@jit(nopython=True)
def calculate_stable_dt(u, v, nx, ny, dx, dy, nu, cfl=0.5, safety_factor=1.0):
    """
    Calculate a stable time step based on CFL condition and viscous stability.
    
    Parameters:
    -----------
    u, v : ndarray
        Velocity components with ghost cells
    nx, ny : int
        Grid dimensions
    dx, dy : float
        Grid spacing
    nu : float
        Kinematic viscosity
    cfl : float
        CFL number
    safety_factor : float
        Additional safety factor to reduce timestep
        
    Returns:
    --------
    dt : float
        Stable time step
    """
    # Find maximum velocity magnitudes
    u_max = 0.0
    v_max = 0.0
    
    for i in range(1, nx+2):
        for j in range(1, ny+1):
            u_max = max(u_max, abs(u[i, j]))
    
    for i in range(1, nx+1):
        for j in range(1, ny+2):
            v_max = max(v_max, abs(v[i, j]))
    
    # Avoid division by zero
    u_max = max(u_max, 1e-10)
    v_max = max(v_max, 1e-10)
    
    # CFL condition
    dt_convective = min(cfl * dx / u_max, cfl * dy / v_max)
    
    # Diffusion stability condition
    dt_diffusive = 0.25 * min(dx*dx, dy*dy) / nu
    
    # Return minimum of the two with safety factor
    dt = safety_factor * min(dt_convective, dt_diffusive)
    
    # Cap maximum timestep
    max_dt = 0.01
    dt = min(dt, max_dt)
    
    return dt

@jit(nopython=True)
def velocity_correction(u_star, v_star, p, nx, ny, dx, dy, dt, rho):
    
    # Initialize corrected velocities
    u = u_star.copy()
    v = v_star.copy()
    
    # Correct u-velocity (interior points) - parallelize outer loop
    for i in prange(1, nx+2):
        for j in range(1, ny+1):
            dp_dx = (p[i, j] - p[i-1, j]) / dx
            u[i, j] = u_star[i, j] - dt/rho * dp_dx
    
    # Correct v-velocity (interior points) - parallelize outer loop
    for i in prange(1, nx+1):
        for j in range(1, ny+2):
            dp_dy = (p[i, j] - p[i, j-1]) / dy
            v[i, j] = v_star[i, j] - dt/rho * dp_dy
    
    # Calculate resulting divergence for diagnostics
    max_div = 0.0
    for i in range(1, nx+1):
        for j in range(1, ny+1):
            div = abs((u[i+1, j] - u[i, j]) / dx + (v[i, j+1] - v[i, j]) / dy)
            max_div = max(max_div, div)
    
    return u, v, max_div

def create_stretched_grid(nx, ny, lx, ly, stretch_factor=1.5):
    """
    Create a staggered grid with hyperbolic tangent stretching in y-direction.
    
    Parameters:
    -----------
    nx, ny : int
        Number of cells in x and y directions
    lx, ly : float
        Domain length in x and y directions
    stretch_factor : float, optional
        Controls grid clustering near walls. 
        Larger values (>1) cluster more points near walls.
        
    Returns:
    --------
    Dict containing grid information with stretched y-coordinates
    """
    # Uniform x-direction
    dx = lx / nx
    
    # Create base uniform y-coordinates
    y_uniform = np.linspace(0, ly, ny+1)
    
    # Apply hyperbolic tangent stretching
    def hyperbolic_stretching(y, ly, stretch_factor):
        # Normalize y to [-1, 1]
        y_norm = 2 * y / ly - 1
        
        # Apply tanh stretching
        y_stretched = ly * (np.tanh(stretch_factor * y_norm) / np.tanh(stretch_factor) + 1) / 2
        return y_stretched
    # Skip stretching completely if factor is 1.0
    if stretch_factor == 1.0:
        # Create uniform grid directly
        y_v = np.zeros(ny+3)
        for j in range(ny+3):
            y_v[j] = (j-1) * (ly/ny)
    else:
        # Compute stretched y-coordinates for cell faces
        y_v = np.zeros(ny+3)  # Include ghost cells
    
        # Create stretched coordinates for interior and boundary points
        for j in range(ny+3):
            # Determine actual y coordinate for stretching
            y_base = (j-1) * (ly / ny)
        
            if j == 0:  # Bottom ghost point
                # Use first interior point's spacing as dy
                y_base_first = hyperbolic_stretching(0, ly, stretch_factor)
                y_base_second = hyperbolic_stretching(ly/ny, ly, stretch_factor)
                dy_first = y_base_second - y_base_first
                y_v[j] = y_base_first - dy_first
            elif j == ny+2:  # Top ghost point
                # Use last interior point's spacing as dy
                y_base_last = hyperbolic_stretching(ly, ly, stretch_factor)
                y_base_second_last = hyperbolic_stretching(ly*(ny-1)/ny, ly, stretch_factor)
                dy_last = y_base_last - y_base_second_last
                y_v[j] = y_base_last + dy_last
            else:
                # Actual stretching applied to interior points
                y_v[j] = hyperbolic_stretching(y_base, ly, stretch_factor)
    
    # Compute average dy for Numba compatibility
    dy_avg = np.mean(np.diff(y_v[1:-1]))
    
    # Create pressure and u-velocity y-coordinates (cell centers)
    y_p = np.zeros(ny+2)  # Include ghost cells
    y_u = np.zeros(ny+2)  # Same as pressure points
    
    # Compute cell center coordinates
    for j in range(ny+2):
        if j == 0:  # Bottom ghost
            y_p[j] = y_v[j] + 0.5 * (y_v[j+1] - y_v[j])
            y_u[j] = y_p[j]
        elif j == ny+1:  # Top ghost
            y_p[j] = y_v[j] + 0.5 * (y_v[j+1] - y_v[j])
            y_u[j] = y_p[j]
        else:
            # Average of cell face coordinates for cell centers
            y_p[j] = 0.5 * (y_v[j] + y_v[j+1])
            y_u[j] = y_p[j]
    
    # Create x-coordinates (same as before)
    x_p = np.zeros(nx+2)  # Include ghost cells
    x_u = np.zeros(nx+3)  # Include ghost cells for u-velocity
    x_v = np.zeros(nx+2)  # Same x as pressure
    
    # Fill interior x-coordinates
    for i in range(1, nx+1):
        x_p[i] = (i-0.5) * dx
        x_v[i] = x_p[i]
    
    # Ghost x-coordinates
    x_p[0] = -0.5 * dx        # Left ghost
    x_p[nx+1] = lx + 0.5 * dx  # Right ghost
    
    # x-coordinates for u-velocity
    for i in range(nx+3):
        x_u[i] = (i-1) * dx
    
    print(f"Stretched Grid (tanh): dx = {dx:.6f}, dy_avg = {dy_avg:.6f}, stretch factor = {stretch_factor}")
    #print(f"  Y-coordinates (faces): {y_v[1:-1]}")
    #print(f"  Y-coordinates (centers): {y_p[1:-1]}")
    
    return {
        'x_p': x_p, 'y_p': y_p,  # Pressure points (cell centers)
        'x_u': x_u, 'y_u': y_u,  # U-velocity points
        'x_v': x_v, 'y_v': y_v,  # V-velocity points
        'dx': dx, 'dy': dy_avg,  # Grid spacing
        'nx': nx, 'ny': ny,      # Grid dimensions
        'lx': lx, 'ly': ly,      # Domain size
        'grid_data': (nx, ny, dx, dy_avg),  # For Numba compatibility
        'y_grid_data': y_v,      # Full stretched y-coordinates for reference
    }

def plot_grid_distribution(grid):
    """
    Visualize the grid point distribution after stretching.
    """
    import matplotlib.pyplot as plt
    
    y_v = grid['y_v']
    y_p = grid['y_p']
    
    plt.figure(figsize=(10, 6))
    plt.plot(np.zeros_like(y_v), y_v, 'ro', label='Face points')
    plt.plot(np.ones_like(y_p), y_p, 'bo', label='Center points')
    plt.title('Grid Point Distribution (Stretched Y-coordinates)')
    plt.xlabel('Dummy X')
    plt.ylabel('Y-coordinate')
    plt.grid(True)
    plt.legend()
    plt.show()

def fractional_step(u, v, p, grid, dt, nu, rho, target_flow_rate=1.0):
    """
    Perform one time step using the fractional step method on a staggered grid.
    
    Parameters:
    -----------
    u, v, p : ndarray
        Velocity and pressure fields
    grid : dict
        Grid information
    dt : float
        Time step
    nu : float
        Kinematic viscosity
    rho : float
        Fluid density
    target_flow_rate : float
        Target average flow rate
        
    Returns:
    --------
    Updated velocity and pressure fields with additional diagnostics
    """
    # Extract grid parameters
    nx, ny = grid['nx'], grid['ny']
    dx, dy = grid['dx'], grid['dy']
    grid_data = grid['grid_data']
    ly = grid['ly']
    
    # 1. Compute fluxes
    conv_u, conv_v, diff_u, diff_v = compute_fluxes_basic(u, v, grid_data, nu)
    
    # 2. Calculate intermediate velocity field (u*)
    u_star = u.copy()
    v_star = v.copy()
    
    # Update velocities with explicit Euler
    for i in range(1, nx+2):
        for j in range(1, ny+1):
            u_star[i, j] = u[i, j] + dt * (-conv_u[i, j] + diff_u[i, j])
    
    for i in range(1, nx+1):
        for j in range(1, ny+2):
            v_star[i, j] = v[i, j] + dt * (-conv_v[i, j] + diff_v[i, j])
    
    # Apply boundary conditions to intermediate velocity
    u_star, v_star, _ = apply_bc(u_star, v_star, p.copy(), nx, ny)
    
    # 3. Solve pressure Poisson equation
    p_new, iterations, residual = solve_pressure_poisson(p, u_star, v_star, grid_data, dt, rho)
    
    # 4. Velocity correction
    u_new, v_new, div_after = velocity_correction(u_star, v_star, p_new, nx, ny, dx, dy, dt, rho)
    
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
    
    return u_new, v_new, p_new, conv_u, conv_v, diff_u, diff_v, iterations, div_after

    import numpy as np
