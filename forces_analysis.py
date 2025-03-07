import numpy as np
from numba import jit
import matplotlib.pyplot as plt

@jit(nopython=True)
def calculate_forces_cylinder_jit(u, v, p, nx, ny, dx, dy, x_p, y_p, x_u, y_u, x_v, y_v, 
                               center_x, center_y, radius, nu, rho, u_ref):
    """
    Calculate drag and lift forces on a cylinder by integrating pressure and viscous forces.
    
    Parameters:
    -----------
    u, v : ndarray
        Velocity components on staggered grid
    p : ndarray
        Pressure field
    nx, ny : int
        Grid dimensions
    dx, dy : float
        Grid spacing
    x_p, y_p, x_u, y_u, x_v, y_v : ndarray
        Grid coordinates for pressure and velocity points
    center_x, center_y : float
        Cylinder center coordinates
    radius : float
        Cylinder radius
    nu : float
        Kinematic viscosity
    rho : float
        Fluid density
    u_ref : float
        Reference velocity for force coefficient calculation
        
    Returns:
    --------
    cd, cl : float
        Drag and lift coefficients
    cd_pressure, cd_viscous : float
        Pressure and viscous contributions to drag
    cl_pressure, cl_viscous : float
        Pressure and viscous contributions to lift
    """
    # Number of integration points around cylinder
    n_theta = 360
    d_theta = 2 * np.pi / n_theta
    
    # Reference values for force coefficients
    diameter = 2 * radius
    ref_area = diameter  # For 2D, reference "area" is just the diameter
    dynamic_pressure = 0.5 * rho * u_ref**2
    
    # Initialize force components
    f_drag_pressure = 0.0
    f_lift_pressure = 0.0
    f_drag_viscous = 0.0
    f_lift_viscous = 0.0
    
    # Loop through points around cylinder
    for i_theta in range(n_theta):
        theta = i_theta * d_theta
        
        # Point on cylinder surface
        x_surface = center_x + radius * np.cos(theta)
        y_surface = center_y + radius * np.sin(theta)
        
        # Unit normal vector pointing outward from cylinder
        nx_component = np.cos(theta)
        ny_component = np.sin(theta)
        
        # Tangential unit vector (clockwise rotation of normal)
        tx_component = -ny_component
        ty_component = nx_component
        
        # Find nearest pressure cell for interpolation
        # First convert to indices
        i_p = int((x_surface - x_p[0]) / dx)
        j_p = int((y_surface - y_p[0]) / dy)
        
        # Ensure indices are in bounds
        i_p = max(1, min(i_p, nx))
        j_p = max(1, min(j_p, ny))
        
        # Bilinear interpolation weights for pressure
        x_frac = (x_surface - x_p[i_p]) / dx
        y_frac = (y_surface - y_p[j_p]) / dy
        
        # Clamp interpolation weights to [0,1]
        x_frac = max(0.0, min(1.0, x_frac))
        y_frac = max(0.0, min(1.0, y_frac))
        
        # Interpolate pressure to surface point
        p00 = p[i_p, j_p]
        p10 = p[min(i_p+1, nx), j_p]
        p01 = p[i_p, min(j_p+1, ny)]
        p11 = p[min(i_p+1, nx), min(j_p+1, ny)]
        
        p_surface = (1-x_frac)*(1-y_frac)*p00 + x_frac*(1-y_frac)*p10 + \
                   (1-x_frac)*y_frac*p01 + x_frac*y_frac*p11
        
        # Calculate pressure force components
        f_drag_p = -p_surface * nx_component * radius * d_theta
        f_lift_p = -p_surface * ny_component * radius * d_theta
        
        # Calculate velocity gradients at surface point for viscous forces
        # We need to find du/dy, dv/dx, du/dx, dv/dy at the surface
        
        # For velocity gradient calculations, use neighboring cells around surface point
        # Find indices for u velocity points
        i_u_left = max(1, min(int((x_surface - 0.5*dx - x_u[0]) / dx) + 1, nx+1))
        i_u_right = min(i_u_left + 1, nx+1)
        j_u_bot = max(1, min(int((y_surface - y_u[0]) / dy) + 1, ny))
        j_u_top = min(j_u_bot + 1, ny)
        
        # Find indices for v velocity points
        i_v_left = max(1, min(int((x_surface - x_v[0]) / dx) + 1, nx))
        i_v_right = min(i_v_left + 1, nx)
        j_v_bot = max(1, min(int((y_surface - 0.5*dy - y_v[0]) / dy) + 1, ny+1))
        j_v_top = min(j_v_bot + 1, ny+1)
        
        # Interpolation weights for u and v
        x_frac_u = (x_surface - x_u[i_u_left]) / dx
        y_frac_u = (y_surface - y_u[j_u_bot]) / dy
        x_frac_v = (x_surface - x_v[i_v_left]) / dx
        y_frac_v = (y_surface - y_v[j_v_bot]) / dy
        
        # Clamp interpolation weights to [0,1]
        x_frac_u = max(0.0, min(1.0, x_frac_u))
        y_frac_u = max(0.0, min(1.0, y_frac_u))
        x_frac_v = max(0.0, min(1.0, x_frac_v))
        y_frac_v = max(0.0, min(1.0, y_frac_v))
        
        # Approximate velocity gradients
        # du/dy using neighboring u points
        du_dy = ((1-x_frac_u) * (u[i_u_left, j_u_top] - u[i_u_left, j_u_bot]) + 
                x_frac_u * (u[i_u_right, j_u_top] - u[i_u_right, j_u_bot])) / dy
        
        # dv/dx using neighboring v points
        dv_dx = ((1-y_frac_v) * (v[i_v_right, j_v_bot] - v[i_v_left, j_v_bot]) + 
                y_frac_v * (v[i_v_right, j_v_top] - v[i_v_left, j_v_top])) / dx
        
        # du/dx for near-wall gradient
        du_dx = ((1-y_frac_u) * (u[i_u_right, j_u_bot] - u[i_u_left, j_u_bot]) + 
                y_frac_u * (u[i_u_right, j_u_top] - u[i_u_left, j_u_top])) / dx
        
        # dv/dy for near-wall gradient
        dv_dy = ((1-x_frac_v) * (v[i_v_left, j_v_top] - v[i_v_left, j_v_bot]) + 
                x_frac_v * (v[i_v_right, j_v_top] - v[i_v_right, j_v_bot])) / dy
        
        # Calculate shear stress at surface
        tau_xx = 2 * nu * rho * du_dx
        tau_yy = 2 * nu * rho * dv_dy
        tau_xy = nu * rho * (du_dy + dv_dx)
        
        # Project stress to normal and tangential components
        tau_nn = tau_xx * nx_component * nx_component + tau_yy * ny_component * ny_component + \
                2 * tau_xy * nx_component * ny_component
                
        tau_nt = tau_xx * nx_component * tx_component + tau_yy * ny_component * ty_component + \
                tau_xy * (nx_component * ty_component + ny_component * tx_component)
        
        # Calculate viscous force components (tangential stress contribution to drag/lift)
        f_drag_v = tau_nt * tx_component * radius * d_theta
        f_lift_v = tau_nt * ty_component * radius * d_theta
        
        # Accumulate force components
        f_drag_pressure += f_drag_p
        f_lift_pressure += f_lift_p
        f_drag_viscous += f_drag_v
        f_lift_viscous += f_lift_v
    
    # Calculate force coefficients
    if dynamic_pressure > 0 and ref_area > 0:
        cd_pressure = f_drag_pressure / (dynamic_pressure * ref_area)
        cl_pressure = f_lift_pressure / (dynamic_pressure * ref_area)
        cd_viscous = f_drag_viscous / (dynamic_pressure * ref_area)
        cl_viscous = f_lift_viscous / (dynamic_pressure * ref_area)
        
        cd = cd_pressure + cd_viscous
        cl = cl_pressure + cl_viscous
    else:
        cd = 0.0
        cl = 0.0
        cd_pressure = 0.0
        cd_viscous = 0.0
        cl_pressure = 0.0
        cl_viscous = 0.0
    
    return cd, cl, cd_pressure, cd_viscous, cl_pressure, cl_viscous

def calculate_forces_cylinder(u, v, p, grid, cylinder_params, nu, rho, u_ref):
    """
    Wrapper function for calculating forces on a cylinder.
    
    Parameters:
    -----------
    u, v : ndarray
        Velocity components
    p : ndarray
        Pressure field
    grid : dict
        Grid parameters
    cylinder_params : dict
        Cylinder parameters (center_x_ratio, center_y_ratio, radius_ratio)
    nu : float
        Kinematic viscosity
    rho : float
        Fluid density
    u_ref : float
        Reference velocity
        
    Returns:
    --------
    cd, cl : float
        Drag and lift coefficients
    cd_pressure, cd_viscous : float
        Pressure and viscous contributions to drag
    cl_pressure, cl_viscous : float
        Pressure and viscous contributions to lift
    """
    # Extract grid parameters
    nx, ny = grid['nx'], grid['ny']
    dx, dy = grid['dx'], grid['dy']
    lx, ly = grid['lx'], grid['ly']
    x_p, y_p = grid['x_p'], grid['y_p']
    x_u, y_u = grid['x_u'], grid['y_u']
    x_v, y_v = grid['x_v'], grid['y_v']
    
    # Extract cylinder parameters
    center_x = cylinder_params.get('center_x_ratio', 0.25) * lx
    center_y = cylinder_params.get('center_y_ratio', 0.5) * ly
    radius = cylinder_params.get('radius_ratio', 0.125) * ly
    
    # Calculate forces
    return calculate_forces_cylinder_jit(u, v, p, nx, ny, dx, dy, x_p, y_p, x_u, y_u, x_v, y_v,
                                      center_x, center_y, radius, nu, rho, u_ref)

def calculate_drag_coefficient(u, v, p, grid, cylinder_params, nu, rho, u_bulk):
    """
    Calculate drag coefficient for a cylinder using force integration.
    This replaces the previous empirical formula with a physical calculation.
    
    Parameters:
    -----------
    u, v : ndarray
        Velocity components
    p : ndarray
        Pressure field
    grid : dict
        Grid parameters
    cylinder_params : dict
        Cylinder parameters
    nu : float
        Kinematic viscosity
    rho : float
        Fluid density
    u_bulk : float
        Bulk velocity used as reference velocity
        
    Returns:
    --------
    cd : float
        Drag coefficient
    """
    # Calculate all force coefficients
    cd, cl, cd_p, cd_v, cl_p, cl_v = calculate_forces_cylinder(u, v, p, grid, cylinder_params, nu, rho, u_bulk)
    
    # Return just the drag coefficient
    return cd

def calculate_detailed_forces(u, v, p, grid, cylinder_params, nu, rho, u_bulk, input_reynolds=None):
    """
    Calculate detailed force coefficients including pressure and viscous contributions.
    
    Parameters:
    -----------
    u, v : ndarray
        Velocity components
    p : ndarray
        Pressure field
    grid : dict
        Grid parameters
    cylinder_params : dict
        Cylinder parameters
    nu : float
        Kinematic viscosity
    rho : float
        Fluid density
    u_bulk : float
        Reference velocity
    input_reynolds : float or None
        Reynolds number as provided by the main simulation
        
    Returns:
    --------
    dict : Dictionary containing force coefficients and their components
    """
    # Calculate force coefficients
    cd, cl, cd_p, cd_v, cl_p, cl_v = calculate_forces_cylinder(u, v, p, grid, cylinder_params, nu, rho, u_bulk)
    
    # Return all coefficients as a dictionary
    return {
        'cd': cd,
        'cl': cl,
        'cd_pressure': cd_p,
        'cd_viscous': cd_v,
        'cl_pressure': cl_p,
        'cl_viscous': cl_v,
        'reynolds': input_reynolds if input_reynolds is not None else 0.0
    }

def plot_force_coefficient_history(time_points, force_history, output_dir='results', analysis_start_time=None):
    """
    Plot history of drag and lift coefficients with pressure and viscous contributions.
    
    Parameters:
    -----------
    time_points : list or array
        Time points
    force_history : list of dict
        History of force coefficient dictionaries
    output_dir : str
        Output directory for saving plots
    analysis_start_time : float or None
        If provided, highlight data after this time and use it for statistics
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Extract coefficient histories
    cd = [f['cd'] for f in force_history]
    cl = [f['cl'] for f in force_history]
    cd_p = [f['cd_pressure'] for f in force_history]
    cd_v = [f['cd_viscous'] for f in force_history]
    cl_p = [f['cl_pressure'] for f in force_history]
    cl_v = [f['cl_viscous'] for f in force_history]
    
    # Determine analysis window
    if analysis_start_time is not None and analysis_start_time > time_points[0]:
        # Find the first index where time >= analysis_start_time
        analysis_start_idx = next((i for i, t in enumerate(time_points) if t >= analysis_start_time), 0)
        if analysis_start_idx > 0:
            print(f"Force coefficient analysis using data from t = {time_points[analysis_start_idx]:.2f} onwards")
        else:
            analysis_start_idx = 0
    else:
        # Default: use second half of the simulation for statistics
        analysis_start_idx = len(cd) // 2
    
    # Calculate mean values for the analysis window
    mean_cd = np.mean(cd[analysis_start_idx:])
    mean_cl = np.mean(cl[analysis_start_idx:])
    mean_cd_p = np.mean(cd_p[analysis_start_idx:])
    mean_cd_v = np.mean(cd_v[analysis_start_idx:])
    
    # Create plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Full time history (lighter colors)
    ax1.plot(time_points, cd, 'b-', linewidth=1, alpha=0.5, label='Total (all time)')
    ax1.plot(time_points, cd_p, 'r-', linewidth=0.75, alpha=0.3)
    ax1.plot(time_points, cd_v, 'g-', linewidth=0.75, alpha=0.3)
    
    # Analysis window (darker colors)
    if analysis_start_time is not None:
        ax1.axvline(x=time_points[analysis_start_idx], color='k', linestyle='-', linewidth=1,
                  label=f'Analysis start (t = {time_points[analysis_start_idx]:.2f})')
        
        # Plot analysis window data with bolder lines
        analysis_times = time_points[analysis_start_idx:]
        ax1.plot(analysis_times, cd[analysis_start_idx:], 'b-', linewidth=2, label='Total (analysis window)')
        ax1.plot(analysis_times, cd_p[analysis_start_idx:], 'r--', linewidth=1.5, label='Pressure')
        ax1.plot(analysis_times, cd_v[analysis_start_idx:], 'g--', linewidth=1.5, label='Viscous')
    
    ax1.axhline(y=mean_cd, color='k', linestyle='--', linewidth=1, 
               label=f'Mean Cd = {mean_cd:.3f}')
               
    ax1.set_title('Drag Coefficient History')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cd')
    ax1.grid(True)
    ax1.legend()
    
    # Add annotation with mean values
    ax1.text(0.02, 0.95, 
            f'Mean Cd = {mean_cd:.3f}\nPressure: {mean_cd_p:.3f} ({100*mean_cd_p/mean_cd:.1f}%)\nViscous: {mean_cd_v:.3f} ({100*mean_cd_v/mean_cd:.1f}%)', 
            transform=ax1.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Lift coefficient plot - similar approach
    # Full history (lighter colors)
    ax2.plot(time_points, cl, 'b-', linewidth=1, alpha=0.5, label='Total (all time)')
    ax2.plot(time_points, cl_p, 'r-', linewidth=0.75, alpha=0.3)
    ax2.plot(time_points, cl_v, 'g-', linewidth=0.75, alpha=0.3)
    
    # Analysis window
    if analysis_start_time is not None:
        ax2.axvline(x=time_points[analysis_start_idx], color='k', linestyle='-', linewidth=1)
        
        # Plot analysis window data
        analysis_times = time_points[analysis_start_idx:]
        ax2.plot(analysis_times, cl[analysis_start_idx:], 'b-', linewidth=2, label='Total (analysis window)')
        ax2.plot(analysis_times, cl_p[analysis_start_idx:], 'r--', linewidth=1.5, label='Pressure')
        ax2.plot(analysis_times, cl_v[analysis_start_idx:], 'g--', linewidth=1.5, label='Viscous')
    
    ax2.axhline(y=mean_cl, color='k', linestyle='--', linewidth=1, 
               label=f'Mean Cl = {mean_cl:.3f}')
               
    ax2.set_title('Lift Coefficient History')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Cl')
    ax2.grid(True)
    ax2.legend()
    
    # Add statistics about lift fluctuations in the analysis window
    cl_std = np.std(cl[analysis_start_idx:])
    cl_max = np.max(np.abs(cl[analysis_start_idx:]))
    
    ax2.text(0.02, 0.95, 
            f'Cl RMS = {cl_std:.3f}\nCl Max = {cl_max:.3f}', 
            transform=ax2.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Add Reynolds number
    re = force_history[0]['reynolds']
    plt.figtext(0.5, 0.01, f'Reynolds Number = {re:.1f}', ha='center', 
               bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/force_coefficients.png", dpi=200)
    plt.close()

def plot_pressure_distribution(u, v, p, grid, cylinder_params, time, output_dir='results', analysis_start_time=None):
    """
    Plot pressure distribution around cylinder.
    
    Parameters:
    -----------
    u, v : ndarray
        Velocity components
    p : ndarray
        Pressure field
    grid : dict
        Grid parameters
    cylinder_params : dict
        Cylinder parameters
    time : float
        Current simulation time
    output_dir : str
        Output directory for saving plots
    analysis_start_time : float or None
        Used to add an indicator if we're in the analysis window
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Extract grid parameters
    nx, ny = grid['nx'], grid['ny']
    dx, dy = grid['dx'], grid['dy']
    lx, ly = grid['lx'], grid['ly']
    x_p, y_p = grid['x_p'], grid['y_p']
    
    # Extract cylinder parameters
    center_x = cylinder_params.get('center_x_ratio', 0.25) * lx
    center_y = cylinder_params.get('center_y_ratio', 0.5) * ly
    radius = cylinder_params.get('radius_ratio', 0.125) * ly
    
    # Number of sample points around cylinder
    n_theta = 360
    angles = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    
    # Initialize arrays for pressure and shear stress
    pressure = np.zeros(n_theta)
    
    # Loop through points around cylinder
    for i, theta in enumerate(angles):
        # Point on cylinder surface
        x_surface = center_x + radius * np.cos(theta)
        y_surface = center_y + radius * np.sin(theta)
        
        # Find nearest pressure cell for interpolation
        i_p = int((x_surface - x_p[0]) / dx)
        j_p = int((y_surface - y_p[0]) / dy)
        
        # Ensure indices are in bounds
        i_p = max(1, min(i_p, nx))
        j_p = max(1, min(j_p, ny))
        
        # Bilinear interpolation weights for pressure
        x_frac = (x_surface - x_p[i_p]) / dx
        y_frac = (y_surface - y_p[j_p]) / dy
        
        # Clamp interpolation weights to [0,1]
        x_frac = max(0.0, min(1.0, x_frac))
        y_frac = max(0.0, min(1.0, y_frac))
        
        # Interpolate pressure to surface point
        p00 = p[i_p, j_p]
        p10 = p[min(i_p+1, nx), j_p]
        p01 = p[i_p, min(j_p+1, ny)]
        p11 = p[min(i_p+1, nx), min(j_p+1, ny)]
        
        p_surface = (1-x_frac)*(1-y_frac)*p00 + x_frac*(1-y_frac)*p10 + \
                   (1-x_frac)*y_frac*p01 + x_frac*y_frac*p11
        
        # Store pressure
        pressure[i] = p_surface
    
    # Convert angle to degrees for plotting
    angle_degrees = np.degrees(angles)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot pressure coefficient
    plt.plot(angle_degrees, pressure, 'b-', linewidth=2)
    
    # Add vertical lines for stagnation points
    plt.axvline(x=0, color='r', linestyle='--', label='Front stagnation (0°)')
    plt.axvline(x=180, color='g', linestyle='--', label='Rear stagnation (180°)')
    
    # Add indicator if we're in the analysis window
    title = f'Pressure Distribution Around Cylinder (t = {time:.2f})'
    if analysis_start_time is not None:
        if time >= analysis_start_time:
            title += ' - Analysis Window'
            plt.text(0.02, 0.95, 'Analysis Window', transform=plt.gca().transAxes,
                   bbox=dict(facecolor='green', alpha=0.3))
    
    plt.title(title)
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Pressure')
    plt.grid(True)
    plt.legend()
    
    # Make x-axis cover full 360 degrees with proper ticks
    plt.xlim(0, 360)
    plt.xticks(np.arange(0, 361, 45))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pressure_distribution_{time:.1f}.png", dpi=200)
    plt.close() 