import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import os

def calculate_vorticity(u, v, grid):
    """Calculate vorticity from velocity field."""
    nx, ny = grid['nx'], grid['ny']
    dx, dy = grid['dx'], grid['dy']
    
    vorticity = np.zeros((nx, ny))
    
    for i in range(nx):
        for j in range(ny):
            i_p = i + 1
            j_p = j + 1
            
            dv_dx = (v[i_p+1, j_p] - v[i_p-1, j_p]) / (2 * dx)
            du_dy = (u[i_p, j_p+1] - u[i_p, j_p-1]) / (2 * dy)
            
            vorticity[i, j] = dv_dx - du_dy
    
    return vorticity

def interpolate_velocity_to_centers(u, v, grid):
    """Interpolate staggered velocity components to cell centers."""
    nx, ny = grid['nx'], grid['ny']
    
    u_center = np.zeros((nx, ny))
    v_center = np.zeros((nx, ny))
    
    for i in range(nx):
        for j in range(ny):
            i_p = i + 1
            j_p = j + 1
            
            u_center[i, j] = 0.5 * (u[i_p+1, j_p] + u[i_p, j_p])
            v_center[i, j] = 0.5 * (v[i_p, j_p+1] + v[i_p, j_p])
    
    return u_center, v_center

def plot_cylinder_flow(grid, u, v, cylinder_params=None, title=None, step=None, output_dir='results'):
    """Create visualization of flow around a cylinder without streamlines."""
    nx, ny = grid['nx'], grid['ny']
    lx, ly = grid['lx'], grid['ly']
    
    # Create coordinate arrays
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Get velocities at cell centers
    u_center, v_center = interpolate_velocity_to_centers(u, v, grid)
    
    # Calculate vorticity
    vorticity = calculate_vorticity(u, v, grid)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 1. U-velocity component
    ax1 = fig.add_subplot(2, 1, 1)
    
    # Plot u-velocity as colored contours
    contour_vel = ax1.contourf(X, Y, u_center, levels=50, cmap='plasma', extend='both')
    plt.colorbar(contour_vel, ax=ax1, label='Streamwise Velocity (u)')
    
    # Draw cylinder if parameters provided
    if cylinder_params:
        center_x = cylinder_params.get('center_x_ratio', 0.25) * lx
        center_y = cylinder_params.get('center_y_ratio', 0.5) * ly
        radius = cylinder_params.get('radius_ratio', 0.125) * ly
        
        circle = plt.Circle((center_x, center_y), radius, color='black', fill=True)
        ax1.add_artist(circle)
    
    ax1.set_title('Streamwise Velocity')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 2. Vorticity field
    ax2 = fig.add_subplot(2, 1, 2)
    
    # Use diverging colormap for vorticity (blue for negative, red for positive)
    max_vort = max(abs(np.min(vorticity)), abs(np.max(vorticity)))
    norm = Normalize(-max_vort, max_vort)
    
    contour_vort = ax2.contourf(X, Y, vorticity, levels=50, cmap='RdBu_r', 
                              norm=norm, extend='both')
    plt.colorbar(contour_vort, ax=ax2, label='Vorticity')
    
    # Draw cylinder
    if cylinder_params:
        circle = plt.Circle((center_x, center_y), radius, color='black', fill=True)
        ax2.add_artist(circle)
    
    ax2.set_title('Vorticity Field')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle='--', alpha=0.3)
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    if step is not None:
        plt.savefig(f"{output_dir}/cylinder_flow_{step:06d}.png", dpi=200, bbox_inches='tight')
    else:
        plt.savefig(f"{output_dir}/cylinder_flow_initial.png", dpi=200, bbox_inches='tight')
    
    plt.close()
    
    return vorticity  # Return for further analysis if needed

def analyze_vortex_shedding(vorticity_history, time_points, grid, cylinder_params, output_dir='results', u_bulk=1.0, analysis_start_time=None):
   """Analyze vortex shedding frequency and patterns.
   
   Parameters:
   -----------
   vorticity_history : list
       List of vorticity snapshots
   time_points : list
       Corresponding time points
   grid : dict
       Grid information
   cylinder_params : dict
       Cylinder parameters
   output_dir : str
       Output directory
   u_bulk : float
       Bulk velocity
   analysis_start_time : float or None
       If provided, only consider data after this time for frequency analysis
   """
   if len(vorticity_history) < 2:
       print("Not enough time steps for vortex shedding analysis")
       return
   
   lx, ly = grid['lx'], grid['ly']
   center_x = cylinder_params.get('center_x_ratio', 0.25) * lx
   center_y = cylinder_params.get('center_y_ratio', 0.5) * ly
   radius = cylinder_params.get('radius_ratio', 0.125) * ly
   
   # Sample point downstream of cylinder for monitoring
   sample_x_ratio = min(0.75, cylinder_params.get('center_x_ratio', 0.25) + 0.25)
   sample_x = sample_x_ratio * lx
   sample_y = center_y
   
   # Find nearest grid point
   nx, ny = grid['nx'], grid['ny']
   x = np.linspace(0, lx, nx)
   y = np.linspace(0, ly, ny)
   
   i_sample = np.argmin(np.abs(x - sample_x))
   j_sample = np.argmin(np.abs(y - sample_y))
   
   # Make sure we're using just the time points that correspond to vorticity snapshots
   if len(time_points) != len(vorticity_history):
       print(f"Time points ({len(time_points)}) and vorticity history ({len(vorticity_history)}) lengths differ")
       print(f"Using only relevant time points for vorticity analysis")
       # Use the last N time points where N is the length of vorticity_history
       vorticity_times = time_points[-len(vorticity_history):]
   else:
       vorticity_times = time_points
   
   # Extract vorticity at sample point over time
   vorticity_at_point = [v[i_sample, j_sample] for v in vorticity_history]
   
   # Plot vorticity time history with proper time axis
   plt.figure(figsize=(12, 8))
   plt.plot(vorticity_times, vorticity_at_point, 'b-', linewidth=2)
   plt.title(f'Vorticity Time History (x/L={sample_x_ratio:.2f}, y/L={0.5:.2f})')
   plt.xlabel('Time')
   plt.ylabel('Vorticity')
   plt.grid(True)
   
   # Determine indices for analysis based on start time
   if analysis_start_time is not None and analysis_start_time > vorticity_times[0]:
       # Find the first index where time >= analysis_start_time
       analysis_start_idx = next((i for i, t in enumerate(vorticity_times) if t >= analysis_start_time), 0)
       if analysis_start_idx > 0:
           print(f"Starting frequency analysis from t = {vorticity_times[analysis_start_idx]:.2f} (index {analysis_start_idx})")
           # Draw a vertical line at the analysis start time
           plt.axvline(x=vorticity_times[analysis_start_idx], color='g', linestyle='-', linewidth=2, 
                     label=f'Analysis start (t = {vorticity_times[analysis_start_idx]:.2f})')
       else:
           print(f"Analysis start time {analysis_start_time} is before the first data point")
           analysis_start_idx = 0
   else:
       analysis_start_idx = 0
   
   # Attempt to calculate shedding frequency if enough data points after start time
   analysis_times = vorticity_times[analysis_start_idx:]
   analysis_vorticity = vorticity_at_point[analysis_start_idx:]
   
   if len(analysis_times) > 10:  # Need enough points for reliable frequency detection
       try:
           # Simple peak detection on the analysis window
           from scipy.signal import find_peaks
           peaks, _ = find_peaks(analysis_vorticity, height=0, distance=3)
           
           if len(peaks) >= 2:
               # Get actual time values for peaks, not indices
               peak_times = [analysis_times[i] for i in peaks]
               periods = np.diff(peak_times)
               avg_period = np.mean(periods)
               frequency = 1.0 / avg_period
               
               # Calculate Strouhal number: St = f*D/U
               diameter = 2 * radius
               strouhal = frequency * diameter / u_bulk
               
               # Add visualization elements
               plt.axhline(y=0, color='k', linestyle='--')
               
               # Draw peaks on the full plot
               for peak_time in peak_times:
                   plt.axvline(x=peak_time, color='r', linestyle='--', alpha=0.5)
               
               plt.title(f'Vorticity Time History - St={strouhal:.3f}, f={frequency:.3f}')
               
               # Add text box with results
               plt.text(0.05, 0.95, f'Strouhal number: {strouhal:.3f}\nShedding frequency: {frequency:.3f}',
                       transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
               
               print(f"Calculated Strouhal number: {strouhal:.3f}")
               print(f"Vortex shedding frequency: {frequency:.3f}")
               print(f"Analysis used {len(peaks)} peaks over time range {analysis_times[0]:.2f} to {analysis_times[-1]:.2f}")
           else:
               print(f"Not enough peaks detected in the analysis window (found {len(peaks)})")
               if len(peaks) > 0:
                   print(f"Peak times: {[analysis_times[i] for i in peaks]}")
       except ImportError:
           print("scipy not available for peak detection")
       except Exception as e:
           print(f"Error in peak detection: {e}")
   else:
       print(f"Not enough data points after t={analysis_start_time} for frequency analysis")
   
   plt.legend()
   plt.savefig(f"{output_dir}/vortex_shedding_analysis.png", dpi=200)
   plt.close()

def generate_space_time_plot(vorticity_history, time_points, grid, cylinder_params, output_dir='results'):
    """Generate a space-time plot of vorticity along the centerline."""
    if len(vorticity_history) < 2:
        print("Not enough time steps for space-time plot")
        return
    
    nx, ny = grid['nx'], grid['ny']
    lx, ly = grid['lx'], grid['ly']
    
    # Get cylinder center y-coordinate
    center_y = cylinder_params.get('center_y_ratio', 0.5) * ly
    
    # Find nearest grid y-index to cylinder centerline
    y = np.linspace(0, ly, ny)
    j_center = np.argmin(np.abs(y - center_y))
    
    # Extract vorticity along centerline for each time step
    centerline_vorticity = []
    for vort in vorticity_history:
        centerline_vorticity.append(vort[:, j_center])
    
    # Convert to numpy array for easier plotting
    centerline_vorticity = np.array(centerline_vorticity)
    
    # Make sure we're using time points that match the vorticity data
    if len(time_points) != len(vorticity_history):
        print(f"Adjusting time points for space-time diagram")
        # Use the last N time points where N is the length of vorticity_history
        vorticity_times = time_points[-len(vorticity_history):]
    else:
        vorticity_times = time_points
    
    # Create space-time plot
    plt.figure(figsize=(10, 8))
    
    # Get actual time bounds for proper scaling
    time_min = vorticity_times[0]
    time_max = vorticity_times[-1]
    
    # Use imshow with extent for proper axis scaling
    x = np.linspace(0, lx, nx)
    extent = [0, lx, time_min, time_max]
    
    # Set vorticity limits for consistent color scaling
    vort_max = max(abs(np.min(centerline_vorticity)), abs(np.max(centerline_vorticity)))
    
    plt.imshow(centerline_vorticity, aspect='auto', origin='lower', 
              extent=extent, cmap='RdBu_r', vmin=-vort_max, vmax=vort_max)
    
    # Mark cylinder position
    center_x = cylinder_params.get('center_x_ratio', 0.25) * lx
    radius = cylinder_params.get('radius_ratio', 0.125) * ly
    plt.axvline(x=center_x - radius, color='k', linestyle='--')
    plt.axvline(x=center_x + radius, color='k', linestyle='--')
    
    plt.colorbar(label='Vorticity')
    plt.title('Space-Time Diagram of Vorticity along Cylinder Centerline')
    plt.xlabel('x position')
    plt.ylabel('Time')
    plt.grid(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/vorticity_spacetime.png", dpi=200)
    plt.close()