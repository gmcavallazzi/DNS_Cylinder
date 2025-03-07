# Modified utils.py with JIT-compatible functions
import numpy as np
import matplotlib.pyplot as plt
import os
from numba import jit
from numba import prange
import yaml

@jit(nopython=True)
def calculate_wall_shear_stresses_jit(u, y_p, nu, rho, nx, ny, ly):
    
    # Get cell distances near walls
    dy_bottom = y_p[1]
    dy_top = ly - y_p[ny]
    
    # Use reduction for parallel sum
    tau_w_bottom_sum = 0.0
    tau_w_top_sum = 0.0
    
    # Bottom wall - parallelize with prange
    for i in prange(1, nx+2):
        du_dy_bottom = u[i, 1] / dy_bottom
        tau_w_bottom_sum += nu * rho * abs(du_dy_bottom)
    
    # Top wall - parallelize with prange
    for i in prange(1, nx+2):
        du_dy_top = -u[i, ny] / dy_top
        tau_w_top_sum += nu * rho * abs(du_dy_top)
    
    # Average across the wall
    tau_w_bottom = tau_w_bottom_sum / (nx+1)
    tau_w_top = tau_w_top_sum / (nx+1)
    
    return tau_w_bottom, tau_w_top

def calculate_wall_shear_stresses(u, grid, nu, rho):
    """Wrapper for JIT wall shear stress calculation"""
    nx, ny = grid['nx'], grid['ny']
    ly = grid['ly']
    y_p = grid['y_p']
    
    return calculate_wall_shear_stresses_jit(u, y_p, nu, rho, nx, ny, ly)

@jit(nopython=True)
def calculate_pressure_gradient_jit(p, nx, ny, dx):
    """JIT-compatible pressure gradient calculation"""
    # For a periodic channel, we can calculate the pressure drop over the domain length
    dp_dx_sum = 0.0
    
    # Calculate at each y-level
    for j in range(1, ny+1):
        # In a periodic domain, measure from first interior to last interior point
        p_left = p[1, j]
        p_right = p[nx, j]
        
        # Calculate the average local gradient
        dp_dx_local = (p_right - p_left) / ((nx-1) * dx)
        dp_dx_sum += dp_dx_local
    
    # Return average
    return dp_dx_sum / ny

def calculate_pressure_gradient(p, grid):
    """Wrapper for JIT pressure gradient calculation"""
    nx, ny = grid['nx'], grid['ny']
    dx = grid['dx']
    
    return calculate_pressure_gradient_jit(p, nx, ny, dx)

def calculate_enforced_bulk_velocity(u, grid):
    """
    Calculate bulk velocity using the exact same method used in fractional_step_ab2_jit
    to enforce the flow rate. This ensures consistency between enforced and reported values.
    
    Parameters:
    -----------
    u : ndarray
        Velocity field
    grid : dict
        Grid information
        
    Returns:
    --------
    float : Bulk velocity calculated exactly as in the flow rate enforcement
    """
    nx, ny = grid['nx'], grid['ny']
    
    # Compute mean velocity using exactly the same method as in fractional_step_ab2_jit
    mean_velocity = 0.0
    total_points = 0
    
    for j in range(1, ny+1):
        for i in range(1, nx+2):
            mean_velocity += u[i, j]
            total_points += 1
    
    if total_points > 0:
        return mean_velocity / total_points
    else:
        return 0.0

@jit(nopython=True)
def calculate_bulk_velocity_jit(u, nx, ny):
    """
    JIT-compatible bulk velocity calculation.
    
    IMPORTANT: This function MUST match exactly how the velocity is calculated 
    in the fractional_step_ab2_jit function's flow rate enforcement to ensure consistency.
    
    Parameters:
    -----------
    u : ndarray
        Velocity field
    nx, ny : int
        Grid dimensions
        
    Returns:
    --------
    float : Average bulk velocity
    """
    # Sum velocities at interior u-points - EXACTLY matching the algorithm in fractional_step_ab2_jit
    u_sum = 0.0
    total_points = 0
    
    for j in range(1, ny+1):
        for i in range(1, nx+2):
            u_sum += u[i, j]
            total_points += 1
    
    # Return average
    if total_points > 0:
        return u_sum / total_points
    else:
        return 0.0

def calculate_bulk_velocity(u, grid, cylinder_params=None):
    """
    Wrapper for JIT bulk velocity calculation.
    
    Parameters:
    -----------
    u : ndarray
        Velocity field
    grid : dict
        Grid information
    cylinder_params : dict or None
        Cylinder parameters - this is not used but kept for compatibility
        
    Returns:
    --------
    float : Average bulk velocity - this exactly matches how the flow rate is enforced
    """
    nx, ny = grid['nx'], grid['ny']
    
    return calculate_bulk_velocity_jit(u, nx, ny)

def plot_velocity_field(grid, u, v, title='Velocity Field', step=None, output_dir='results', u_bulk=1.0):
    nx, ny = grid['nx'], grid['ny']
    
    # Create meshgrid for plotting
    x = np.linspace(0, grid['lx'], nx)
    y = np.linspace(0, grid['ly'], ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Interpolate u to cell centers for visualization
    u_center = np.zeros((nx, ny))
    v_center = np.zeros((nx, ny))
    
    for i in range(nx):
        for j in range(ny):
            # Map from cell centers to u indices
            i_p = i + 1  # Interior pressure indices start at 1
            j_p = j + 1
            
            # Average u from horizontal faces (i+1/2,j and i-1/2,j)
            u_center[i, j] = 0.5 * (u[i_p+1, j_p] + u[i_p, j_p])
            
            # Average v from vertical faces
            v_center[i, j] = 0.5 * (v[i_p, j_p+1] + v[i_p, j_p])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    # Create filled contour plot of streamwise velocity (u)
    contour1 = ax1.contourf(X, Y, u_center, 
                           levels=50,  # More detailed levels
                           cmap='viridis',
                           extend='both')
    
    plt.colorbar(contour1, ax=ax1, label='Streamwise Velocity (u)')
    ax1.set_title(f"{title} - Streamwise Velocity")
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Create filled contour plot of vertical velocity (v)
    contour2 = ax2.contourf(X, Y, v_center, 
                           levels=50,  # More detailed levels
                           cmap='coolwarm', 
                           extend='both')
    
    plt.colorbar(contour2, ax=ax2, label='Vertical Velocity (v)')
    ax2.set_title(f"{title} - Vertical Velocity")
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    if step is not None:
        plt.savefig(f"{output_dir}/velocity_{step:06d}.png", dpi=200)
    else:
        plt.savefig(f"{output_dir}/velocity_initial.png", dpi=200)
    
    plt.close()

    # Detailed velocity statistics
    print(f"Velocity field statistics:")
    print(f"  Target bulk velocity:  {u_bulk:.6f}")
    print(f"  Min velocity:          {u_center.min():.6f}")
    print(f"  Max velocity:          {u_center.max():.6f}")
    print(f"  Mean velocity:         {u_center.mean():.6f}")
    print(f"  Median velocity:       {np.median(u_center):.6f}")
    print(f"  Standard deviation:    {u_center.std():.6f}")
    
    # Optional: Verify vertical averaging
    u_profile = u_center.mean(axis=0)  # Average across x
    print("\nVertical velocity profile:")
    print(f"  Profile mean:  {u_profile.mean():.6f}")
    print(f"  Profile max:   {u_profile.max():.6f}")
    print(f"  Profile min:   {u_profile.min():.6f}")

    return u_center  # Return for potential further analysis

def plot_flow_metrics(time_points, driving_gradients, wall_shear_bottom, wall_shear_top, 
                     bulk_velocities, dp_dx_laminar, tau_w_laminar, output_dir):
    """
    Plot the history of pressure gradient, wall shear stress, and bulk velocity.
    
    Parameters:
    -----------
    time_points : list
        Simulation time points
    driving_gradients : list
        Driving pressure gradient values
    wall_shear_bottom, wall_shear_top : list
        Wall shear stress values at bottom and top walls
    bulk_velocities : list
        Bulk velocity values
    dp_dx_laminar : float
        Theoretical laminar pressure gradient
    tau_w_laminar : float
        Theoretical laminar wall shear stress
    output_dir : str
        Output directory for saving plots
    """
    plt.figure(figsize=(12, 15))
    
    # Plot pressure gradient history
    plt.subplot(3, 1, 1)
    plt.plot(time_points, driving_gradients, 'b-', linewidth=2, label='Driving dP/dx')
    plt.axhline(y=dp_dx_laminar, color='r', linestyle='--', label='Laminar Theory')
    plt.title('Streamwise Pressure Gradient History')
    plt.xlabel('Time')
    plt.ylabel('dP/dx')
    plt.grid(True)
    plt.legend()
    
    # Plot wall shear stress history
    plt.subplot(3, 1, 2)
    plt.plot(time_points, wall_shear_bottom, 'g-', linewidth=2, label='Bottom Wall')
    plt.plot(time_points, wall_shear_top, 'b-', linewidth=2, label='Top Wall')
    plt.axhline(y=tau_w_laminar, color='r', linestyle='--', label='Laminar Theory')
    plt.title('Wall Shear Stress History')
    plt.xlabel('Time')
    plt.ylabel('Wall Shear Stress')
    plt.grid(True)
    plt.legend()
    
    # Plot bulk velocity history
    plt.subplot(3, 1, 3)
    plt.plot(time_points, bulk_velocities, 'r-', linewidth=2)
    plt.axhline(y=1.0, color='k', linestyle='--', label='Target')
    plt.title('Bulk Velocity History')
    plt.xlabel('Time')
    plt.ylabel('Bulk Velocity / U_bulk')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/flow_metrics_history.png", dpi=200)
    plt.close()

def load_config(config_file):
    """
    Load configuration from YAML file.
    
    Parameters:
    -----------
    config_file : str
        Path to YAML configuration file
        
    Returns:
    --------
    Dict containing configuration parameters
    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        print(f"Loaded configuration from {config_file}")
        
        # Perform validation of required parameters
        required_params = ['grid', 'simulation', 'flow', 'bc']
        for param in required_params:
            if param not in config:
                raise ValueError(f"Missing required configuration section: {param}")
        
        # Print configuration summary
        print("\nConfiguration Summary:")
        print(f"  Grid:       {config['grid']['nx']}x{config['grid']['ny']}")
        print(f"  Domain:     {config['grid']['lx']}x{config['grid']['ly']}")
        print(f"  Re:         {config['flow']['reynolds']}")
        print(f"  BC:         {config['bc']}")
        print(f"  CFL:        {config['simulation']['cfl']}")
        print(f"  End time:   {config['simulation']['t_end']}")
        print(f"  Output dir: {config['simulation']['output_dir']}")
        print()
        
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        raise

def calculate_strouhal_number(drag_history, lift_history, time_points, diameter, u_bulk):
    """
    Calculate Strouhal number from force oscillation frequency
    
    Parameters:
    -----------
    drag_history, lift_history : array
        Time history of drag and lift forces
    time_points : array
        Time points corresponding to force history
    diameter : float
        Cylinder diameter
    u_bulk : float
        Bulk velocity
    
    Returns:
    --------
    strouhal : float
        Strouhal number
    """
    from scipy import signal
    
    # Lift is typically better for frequency detection
    if len(time_points) < 16:
        # Not enough data for reliable frequency detection
        return 0.0, 0.0
    
    # Use last 3/4 of the signal to avoid transients
    start_idx = len(time_points) // 4
    
    try:
        # Find peaks in lift signal
        peaks, _ = signal.find_peaks(lift_history[start_idx:])
        
        if len(peaks) > 3:
            # Calculate average period from peak intervals
            peak_times = [time_points[start_idx + p] for p in peaks]
            periods = np.diff(peak_times)
            avg_period = np.mean(periods)
            
            # Calculate frequency and Strouhal number
            frequency = 1.0 / avg_period if avg_period > 0 else 0.0
            strouhal = frequency * diameter / u_bulk
            
            return strouhal, frequency
        
        # Fallback: Use FFT if peak detection fails
        if len(time_points[start_idx:]) > 30:
            # Sampling rate
            dt = np.mean(np.diff(time_points[start_idx:]))
            fs = 1.0 / dt
            
            # FFT
            signal_data = lift_history[start_idx:]
            signal_data = signal_data - np.mean(signal_data)  # Remove mean
            
            f, Pxx = signal.welch(signal_data, fs, nperseg=min(256, len(signal_data)//2))
            
            # Find peak frequency
            main_freq_idx = np.argmax(Pxx[1:]) + 1  # Skip zero frequency
            frequency = f[main_freq_idx]
            
            # Calculate Strouhal number
            strouhal = frequency * diameter / u_bulk
            
            return strouhal, frequency
    
    except Exception as e:
        print(f"Error calculating Strouhal number: {e}")
    
    return 0.0, 0.0

def plot_cylinder_forces(time_points, drag_history, lift_history, output_dir):
    """
    Plot cylinder drag and lift coefficient history
    
    Parameters:
    -----------
    time_points : array
        Time points
    drag_history, lift_history : array
        Drag and lift coefficient history
    output_dir : str
        Output directory for saving plots
    """
    plt.figure(figsize=(12, 8))
    
    # Plot drag coefficient
    plt.subplot(2, 1, 1)
    plt.plot(time_points, drag_history, 'r-', linewidth=2)
    plt.title('Cylinder Drag Coefficient History')
    plt.xlabel('Time')
    plt.ylabel('Drag Coefficient')
    plt.grid(True)
    
    # Plot lift coefficient
    plt.subplot(2, 1, 2)
    plt.plot(time_points, lift_history, 'b-', linewidth=2)
    plt.title('Cylinder Lift Coefficient History')
    plt.xlabel('Time')
    plt.ylabel('Lift Coefficient')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cylinder_forces.png", dpi=200)
    plt.close()
    
    # If there's enough data, also plot FFT of lift coefficient for frequency analysis
    if len(lift_history) > 100:
        from scipy import signal
        
        # Use second half of data to avoid transients
        lift_data = lift_history[len(lift_history)//2:]
        dt = np.mean(np.diff(time_points[len(time_points)//2:]))
        fs = 1.0 / dt
        
        # Compute power spectral density
        f, psd = signal.welch(lift_data - np.mean(lift_data), fs, nperseg=min(256, len(lift_data)//2))
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(f, psd)
        plt.title('Lift Coefficient Power Spectral Density')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('PSD')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/lift_psd.png", dpi=200)
        plt.close()