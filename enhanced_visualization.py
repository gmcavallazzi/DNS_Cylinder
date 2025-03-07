import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Circle
import os
from scipy.signal import find_peaks, welch
import pandas as pd

def calculate_vorticity(u, v, grid):
    """Calculate vorticity from velocity field with improved accuracy."""
    nx, ny = grid['nx'], grid['ny']
    dx, dy = grid['dx'], grid['dy']
    
    vorticity = np.zeros((nx, ny))
    
    for i in range(nx):
        for j in range(ny):
            i_p = i + 1
            j_p = j + 1
            
            # Use central difference for better accuracy
            if i > 0 and i < nx-1 and j > 0 and j < ny-1:
                dv_dx = (v[i_p+1, j_p] - v[i_p-1, j_p]) / (2 * dx)
                du_dy = (u[i_p, j_p+1] - u[i_p, j_p-1]) / (2 * dy)
            else:
                # Fall back to forward/backward difference at boundaries
                if i == 0:
                    dv_dx = (v[i_p+1, j_p] - v[i_p, j_p]) / dx
                elif i == nx-1:
                    dv_dx = (v[i_p, j_p] - v[i_p-1, j_p]) / dx
                else:
                    dv_dx = (v[i_p+1, j_p] - v[i_p-1, j_p]) / (2 * dx)
                
                if j == 0:
                    du_dy = (u[i_p, j_p+1] - u[i_p, j_p]) / dy
                elif j == ny-1:
                    du_dy = (u[i_p, j_p] - u[i_p, j_p-1]) / dy
                else:
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

def plot_enhanced_flow(grid, u, v, cylinder_params=None, title=None, step=None, output_dir='results', 
                     include_streamlines=True, include_vortices=True):
    """Enhanced visualization of flow with better streamlines and vorticity identification."""
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
    
    # Calculate velocity magnitude for streamline coloring
    velocity_magnitude = np.sqrt(u_center**2 + v_center**2)
    
    # Create figure with two subplots (vorticity and velocity magnitude)
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Vorticity field with improved visualization
    ax1 = fig.add_subplot(2, 1, 1)
    
    # Use diverging colormap for vorticity with improved normalization
    max_vort = max(abs(np.min(vorticity)), abs(np.max(vorticity)))
    norm = TwoSlopeNorm(vmin=-max_vort, vcenter=0, vmax=max_vort)
    
    contour_vort = ax1.contourf(X, Y, vorticity, levels=50, cmap='RdBu_r', 
                              norm=norm, extend='both')
    plt.colorbar(contour_vort, ax=ax1, label='Vorticity')
    
    # Add vortex detection if requested
    if include_vortices:
        # Simple vortex identification by thresholding vorticity
        vortex_threshold = 0.5 * max_vort
        positive_vortices = (vorticity > vortex_threshold)
        negative_vortices = (vorticity < -vortex_threshold)
        
        # Mark strong vortex regions
        if np.any(positive_vortices):
            ax1.contour(X, Y, positive_vortices, levels=[0.5], colors='r', linewidths=1.0, alpha=0.7)
        if np.any(negative_vortices):
            ax1.contour(X, Y, negative_vortices, levels=[0.5], colors='b', linewidths=1.0, alpha=0.7)
    
    # Draw cylinder
    if cylinder_params:
        center_x = cylinder_params.get('center_x_ratio', 0.25) * lx
        center_y = cylinder_params.get('center_y_ratio', 0.5) * ly
        radius = cylinder_params.get('radius_ratio', 0.125) * ly
        
        circle = Circle((center_x, center_y), radius, color='black', fill=True)
        ax1.add_artist(circle)
    
    ax1.set_title('Vorticity Field', fontsize=14)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 2. Velocity magnitude with streamlines
    ax2 = fig.add_subplot(2, 1, 2)
    
    # Plot velocity magnitude with a better colormap for seeing streamlines
    contour_mag = ax2.contourf(X, Y, velocity_magnitude, levels=50, cmap='plasma', extend='both')
    plt.colorbar(contour_mag, ax=ax2, label='Velocity Magnitude')
    
    # Add streamlines colored by velocity magnitude
    if include_streamlines:
        try:
            # Create new seed points at different positions
            if cylinder_params:
                # Add more points behind the cylinder for wake visualization
                x_wake = np.linspace(center_x + 1.2*radius, 0.95*lx, 5)
                y_wake = np.linspace(center_y - 0.3*ly, center_y + 0.3*ly, 10)
                wake_points = []
                for x_pos in x_wake:
                    for y_pos in y_wake:
                        wake_points.append([x_pos, y_pos])
                
                start_points = np.array(wake_points)
            else:
                # More regular distribution for channel flow
                x_points = np.linspace(0.2*lx, 0.8*lx, 6)
                y_points = np.linspace(0.1*ly, 0.9*ly, 8)
                start_points = []
                for x_pos in x_points:
                    for y_pos in y_points:
                        start_points.append([x_pos, y_pos])
                
                start_points = np.array(start_points)
            
            # Streamlines with white color for better visibility against plasma colormap
            streamlines = ax2.streamplot(x, y, u_center.T, v_center.T,
                                       color='white', linewidth=1.0, arrowsize=1.2,
                                       density=1.0, start_points=start_points)
        except Exception as e:
            print(f"Warning: Could not add custom streamlines in magnitude plot: {e}")
            # Fallback
            ax2.streamplot(x, y, u_center.T, v_center.T, color='white', linewidth=0.8)
    
    # Draw cylinder
    if cylinder_params:
        circle = Circle((center_x, center_y), radius, color='black', fill=True)
        ax2.add_artist(circle)
    
    ax2.set_title('Velocity Magnitude with Streamlines', fontsize=14)
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
        plt.savefig(f"{output_dir}/enhanced_flow_{step:06d}.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"{output_dir}/enhanced_flow_initial.png", dpi=300, bbox_inches='tight')
    
    plt.close()
    
    # Return data for further analysis
    return {
        'vorticity': vorticity,
        'u_center': u_center,
        'v_center': v_center,
        'velocity_magnitude': velocity_magnitude
    }

def calculate_improved_strouhal_number(vorticity_history, time_points, grid, cylinder_params, u_bulk=1.0):
    """
    Calculate Strouhal number with improved signal processing for better accuracy.
    
    Parameters:
    -----------
    vorticity_history : list
        List of vorticity fields at different time steps
    time_points : array
        Time points corresponding to vorticity snapshots
    grid : dict
        Grid information
    cylinder_params : dict
        Parameters for the cylinder
    u_bulk : float
        Bulk velocity
        
    Returns:
    --------
    dict
        Dictionary with Strouhal number and related frequency information
    """
    if len(vorticity_history) < 10:
        print("Not enough time steps for reliable Strouhal number calculation")
        return {"strouhal": 0.0, "frequency": 0.0, "reliable": False}
    
    # Extract cylinder parameters
    lx, ly = grid['lx'], grid['ly']
    center_x = cylinder_params.get('center_x_ratio', 0.25) * lx
    center_y = cylinder_params.get('center_y_ratio', 0.5) * ly
    radius = cylinder_params.get('radius_ratio', 0.125) * ly
    diameter = 2 * radius
    
    # Define sampling points in the wake
    nx, ny = grid['nx'], grid['ny']
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    
    # Get multiple probe points in wake for robust detection
    probe_points = []
    wake_distance = [1.5, 2.0, 2.5, 3.0]  # Multiple distances downstream
    
    for dist in wake_distance:
        probe_x = min(center_x + dist * diameter, 0.9 * lx)
        i_probe = np.argmin(np.abs(x - probe_x))
        j_probe = np.argmin(np.abs(y - center_y))
        
        # Add points slightly above and below centerline
        j_above = min(j_probe + ny//20, ny-1)
        j_below = max(j_probe - ny//20, 0)
        
        probe_points.append((i_probe, j_probe))
        probe_points.append((i_probe, j_above))
        probe_points.append((i_probe, j_below))
    
    # Extract vorticity time series at each probe point
    vorticity_series = []
    for i, j in probe_points:
        series = [v[i, j] for v in vorticity_history]
        vorticity_series.append(series)
    
    # Convert to numpy array for easier processing
    vorticity_series = np.array(vorticity_series)
    
    # Use only the latter part of time series to avoid initial transients
    start_idx = len(time_points) // 3
    
    # Calculate Strouhal numbers from each probe
    strouhal_values = []
    frequencies = []
    
    for series in vorticity_series:
        # Normalize and detrend the signal
        signal = series[start_idx:]
        if len(signal) < 5:
            continue
            
        signal = signal - np.mean(signal)
        
        # Try peak detection first
        try:
            # Find peaks with sufficient prominence
            prominence = 0.2 * np.max(np.abs(signal))
            peaks, _ = find_peaks(signal, prominence=prominence, distance=3)
            
            if len(peaks) >= 3:
                # Calculate average period from peak-to-peak time
                peak_times = [time_points[start_idx + i] for i in peaks]
                periods = np.diff(peak_times)
                avg_period = np.mean(periods)
                
                frequency = 1.0 / avg_period if avg_period > 0 else 0.0
                strouhal = frequency * diameter / u_bulk
                
                strouhal_values.append(strouhal)
                frequencies.append(frequency)
        except Exception as e:
            print(f"Peak detection failed: {e}")
        
        # Always try spectral analysis as well
        try:
            # Calculate power spectral density with Welch method
            if len(signal) >= 20:
                # Get sampling rate from time points
                dt = np.mean(np.diff(time_points[start_idx:]))
                fs = 1.0 / dt
                
                # Use Welch's method for better frequency resolution
                nperseg = min(128, len(signal) // 2)
                if nperseg >= 8:  # Ensure enough points for FFT
                    f, psd = welch(signal, fs, nperseg=nperseg)
                    
                    # Find peak in power spectrum (exclude zero frequency)
                    peak_idx = np.argmax(psd[1:]) + 1
                    peak_freq = f[peak_idx]
                    
                    if peak_freq > 0:
                        strouhal = peak_freq * diameter / u_bulk
                        
                        strouhal_values.append(strouhal)
                        frequencies.append(peak_freq)
        except Exception as e:
            print(f"Spectral analysis failed: {e}")
    
    # If we have multiple estimates, use the median (more robust to outliers)
    if strouhal_values:
        median_strouhal = np.median(strouhal_values)
        median_frequency = np.median(frequencies)
        
        # Check reliability - consistency across probes
        if len(strouhal_values) >= 3:
            std_strouhal = np.std(strouhal_values)
            reliability = "high" if std_strouhal < 0.05 else "medium" if std_strouhal < 0.1 else "low"
        else:
            reliability = "low"
        
        # Print vortex shedding results to terminal
        print("\n=== Vortex Shedding Analysis ===")
        print(f"Strouhal number: {median_strouhal:.4f}")
        print(f"Shedding frequency: {median_frequency:.4f} Hz")
        print(f"Reliability: {reliability} (std: {np.std(strouhal_values):.4f})")
        
        if reliability == "high":
            print("Consistent vortex shedding detected across multiple measurement points")
        elif reliability == "medium":
            print("Vortex shedding detected with moderate variability")
        else:
            print("Vortex shedding measurements show high variability")
        
        # Create and save visualization of vorticity time series
        plt.figure(figsize=(12, 8))
        
        # Plot the most reliable vorticity series
        best_series_idx = np.argmin(np.abs(np.array(strouhal_values) - median_strouhal))
        best_series = vorticity_series[best_series_idx]
        
        plt.plot(time_points, best_series, 'b-', linewidth=1.5)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Mark peaks if available
        signal = best_series[start_idx:] - np.mean(best_series[start_idx:])
        try:
            prominence = 0.2 * np.max(np.abs(signal))
            peaks, _ = find_peaks(signal, prominence=prominence, distance=3)
            
            peak_times = [time_points[start_idx + i] for i in peaks]
            for peak_time in peak_times:
                plt.axvline(x=peak_time, color='r', linestyle='--', alpha=0.5)
        except Exception:
            pass
        
        plt.title(f'Vorticity Time Series - St={median_strouhal:.4f}, f={median_frequency:.4f}')
        plt.xlabel('Time')
        plt.ylabel('Vorticity')
        plt.grid(True)
        
        # Add textbox with Strouhal information
        info_text = (f"Strouhal number: {median_strouhal:.4f}\n"
                     f"Shedding frequency: {median_frequency:.4f}\n"
                     f"Reliability: {reliability}")
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')
        
        plt.savefig(f"{grid.get('output_dir', 'results')}/vorticity_timeseries_analysis.png", dpi=200)
        plt.close()
        
        return {
            "strouhal": median_strouhal,
            "frequency": median_frequency,
            "reliability": reliability,
            "all_strouhal": strouhal_values,
            "all_frequencies": frequencies
        }
    
    print("\nCould not determine vortex shedding frequency/Strouhal number")
    return {"strouhal": 0.0, "frequency": 0.0, "reliable": False} series in vorticity_series:
        # Normalize and detrend the signal
        signal = series[start_idx:]
        if len(signal) < 5:
            continue
            
        signal = signal - np.mean(signal)
        
        # Try peak detection first
        try:
            # Find peaks with sufficient prominence
            prominence = 0.2 * np.max(np.abs(signal))
            peaks, _ = find_peaks(signal, prominence=prominence, distance=3)
            
            if len(peaks) >= 3:
                # Calculate average period from peak-to-peak time
                peak_times = [time_points[start_idx + i] for i in peaks]
                periods = np.diff(peak_times)
                avg_period = np.mean(periods)
                
                frequency = 1.0 / avg_period if avg_period > 0 else 0.0
                strouhal = frequency * diameter / u_bulk
                
                strouhal_values.append(strouhal)
                frequencies.append(frequency)
        except Exception as e:
            print(f"Peak detection failed: {e}")
        
        # Always try spectral analysis as well
        try:
            # Calculate power spectral density with Welch method
            if len(signal) >= 20:
                # Get sampling rate from time points
                dt = np.mean(np.diff(time_points[start_idx:]))
                fs = 1.0 / dt
                
                # Use Welch's method for better frequency resolution
                nperseg = min(128, len(signal) // 2)
                if nperseg >= 8:  # Ensure enough points for FFT
                    f, psd = welch(signal, fs, nperseg=nperseg)
                    
                    # Find peak in power spectrum (exclude zero frequency)
                    peak_idx = np.argmax(psd[1:]) + 1
                    peak_freq = f[peak_idx]
                    
                    if peak_freq > 0:
                        strouhal = peak_freq * diameter / u_bulk
                        
                        strouhal_values.append(strouhal)
                        frequencies.append(peak_freq)
        except Exception as e:
            print(f"Spectral analysis failed: {e}")
    
    # If we have multiple estimates, use the median (more robust to outliers)
    if strouhal_values:
        median_strouhal = np.median(strouhal_values)
        median_frequency = np.median(frequencies)
        
        # Check reliability - consistency across probes
        if len(strouhal_values) >= 3:
            std_strouhal = np.std(strouhal_values)
            reliability = "high" if std_strouhal < 0.05 else "medium" if std_strouhal < 0.1 else "low"
        else:
            reliability = "low"
        
        # Create and save visualization of vorticity time series
        plt.figure(figsize=(12, 8))
        
        # Plot the most reliable vorticity series
        best_series_idx = np.argmin(np.abs(np.array(strouhal_values) - median_strouhal))
        best_series = vorticity_series[best_series_idx]
        
        plt.plot(time_points, best_series, 'b-', linewidth=1.5)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Mark peaks if available
        signal = best_series[start_idx:] - np.mean(best_series[start_idx:])
        try:
            prominence = 0.2 * np.max(np.abs(signal))
            peaks, _ = find_peaks(signal, prominence=prominence, distance=3)
            
            peak_times = [time_points[start_idx + i] for i in peaks]
            for peak_time in peak_times:
                plt.axvline(x=peak_time, color='r', linestyle='--', alpha=0.5)
        except Exception:
            pass
        
        plt.title(f'Vorticity Time Series - St={median_strouhal:.4f}, f={median_frequency:.4f}')
        plt.xlabel('Time')
        plt.ylabel('Vorticity')
        plt.grid(True)
        
        # Add textbox with Strouhal information
        info_text = (f"Strouhal number: {median_strouhal:.4f}\n"
                     f"Shedding frequency: {median_frequency:.4f}\n"
                     f"Reliability: {reliability}")
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')
        
        plt.savefig(f"{grid.get('output_dir', 'results')}/vorticity_timeseries_analysis.png", dpi=200)
        plt.close()
        
        return {
            "strouhal": median_strouhal,
            "frequency": median_frequency,
            "reliability": reliability,
            "all_strouhal": strouhal_values,
            "all_frequencies": frequencies
        }
    
    return {"strouhal": 0.0, "frequency": 0.0, "reliable": False}

def generate_enhanced_spacetime_plot(vorticity_history, time_points, grid, cylinder_params, output_dir='results'):
    """Generate an enhanced space-time plot with better visualization."""
    if len(vorticity_history) < 5:
        print("Not enough time steps for space-time plot")
        return
    
    nx, ny = grid['nx'], grid['ny']
    lx, ly = grid['lx'], grid['ly']
    
    # Get cylinder parameters
    center_x = cylinder_params.get('center_x_ratio', 0.25) * lx
    center_y = cylinder_params.get('center_y_ratio', 0.5) * ly
    radius = cylinder_params.get('radius_ratio', 0.125) * ly
    
    # Find nearest grid y-indices to centerline and other important positions
    y = np.linspace(0, ly, ny)
    j_center = np.argmin(np.abs(y - center_y))
    
    # Also get locations slightly above and below centerline
    offset = max(1, ny // 20)
    j_above = min(j_center + offset, ny-1)
    j_below = max(j_center - offset, 0)
    
    # Create figure for multiple slices
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Process each slice position
    positions = [
        (j_center, "Centerline", axes[0]),
        (j_above, f"Above centerline (y = {y[j_above]:.2f})", axes[1]),
        (j_below, f"Below centerline (y = {y[j_below]:.2f})", axes[2])
    ]
    
    x = np.linspace(0, lx, nx)
    extent = [0, lx, time_points[0], time_points[-1]]
    
    for j, label, ax in positions:
        # Extract vorticity along specified y-position for each time step
        slice_vorticity = []
        for vort in vorticity_history:
            slice_vorticity.append(vort[:, j])
        
        # Convert to numpy array for easier plotting
        slice_vorticity = np.array(slice_vorticity)
        
        # Set vorticity limits for consistent color scaling
        vort_max = max(abs(np.min(slice_vorticity)), abs(np.max(slice_vorticity)))
        
        # Use improved colormap with better normalization
        norm = TwoSlopeNorm(vmin=-vort_max, vcenter=0, vmax=vort_max)
        
        im = ax.imshow(slice_vorticity, aspect='auto', origin='lower', 
                       extent=extent, cmap='RdBu_r', norm=norm)
        
        # Mark cylinder position
        ax.axvline(x=center_x - radius, color='k', linestyle='--')
        ax.axvline(x=center_x + radius, color='k', linestyle='--')
        
        # Draw arrow indicating the wake region
        ax.annotate('Wake region', xy=(center_x + radius*1.5, time_points[0] + (time_points[-1]-time_points[0])*0.1),
                    xytext=(center_x + radius*1.5, time_points[0] + (time_points[-1]-time_points[0])*0.25),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
        
        # Add colorbar for each subplot
        plt.colorbar(im, ax=ax, label='Vorticity')
        
        ax.set_title(f'Space-Time Diagram - {label}')
        ax.set_ylabel('Time')
        ax.grid(False)
    
    # Set common x label
    axes[2].set_xlabel('x position')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/enhanced_vorticity_spacetime.png", dpi=300)
    plt.close()
    
    # Print space-time analysis info
    print("\n=== Space-Time Analysis ===")
    print(f"Generated vorticity space-time plot at {j_center} positions")
    print(f"Results saved to: {output_dir}/enhanced_vorticity_spacetime.png")
    
    # Bonus: Create additional cross-section views at specific x-positions
    # This can help identify vortex patterns in the wake
    
    # Define positions for x cross-sections
    x_positions = []
    if center_x + 2*radius < lx:
        x_positions.append((center_x + 2*radius, "Near wake"))
    if center_x + 5*radius < lx:
        x_positions.append((center_x + 5*radius, "Medium wake"))
    if center_x + 10*radius < lx:
        x_positions.append((center_x + 10*radius, "Far wake"))
    
    if len(x_positions) > 0:
        fig, axes = plt.subplots(len(x_positions), 1, figsize=(10, 4*len(x_positions)), sharex=True)
        if len(x_positions) == 1:
            axes = [axes]  # Make sure axes is a list even with one subplot
        
        y = np.linspace(0, ly, ny)
        extent = [0, ly, time_points[0], time_points[-1]]
        
        for idx, ((x_pos, label), ax) in enumerate(zip(x_positions, axes)):
            # Find nearest grid point
            i_pos = np.argmin(np.abs(x - x_pos))
            
            # Extract vorticity along specified x-position for each time step
            slice_vorticity = []
            for vort in vorticity_history:
                slice_vorticity.append(vort[i_pos, :])
            
            # Convert to numpy array for easier plotting
            slice_vorticity = np.array(slice_vorticity)
            
            # Set vorticity limits for consistent color scaling
            vort_max = max(abs(np.min(slice_vorticity)), abs(np.max(slice_vorticity)))
            norm = TwoSlopeNorm(vmin=-vort_max, vcenter=0, vmax=vort_max)
            
            im = ax.imshow(slice_vorticity, aspect='auto', origin='lower', 
                           extent=extent, cmap='RdBu_r', norm=norm)
            
            # Mark cylinder centerline
            ax.axvline(x=center_y, color='k', linestyle='--')
            
            plt.colorbar(im, ax=ax, label='Vorticity')
            
            ax.set_title(f'Time-Y Diagram at x={x_pos:.2f} ({label})')
            ax.set_ylabel('Time')
            ax.grid(False)
        
        # Set common x label
        axes[-1].set_xlabel('y position')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/vorticity_time_y_slices.png", dpi=300)
        plt.close()
        
        print(f"Generated vortex pattern analysis at {len(x_positions)} downstream positions")
        print(f"Results saved to: {output_dir}/vorticity_time_y_slices.png")

def analyze_wake_structure(vorticity_history, time_points, grid, cylinder_params, u_bulk, output_dir='results'):
    """
    Perform detailed analysis of wake structure behind cylinder.
    
    Parameters:
    -----------
    vorticity_history : list
        List of vorticity fields at different time steps
    time_points : array
        Time points corresponding to vorticity snapshots
    grid : dict
        Grid information
    cylinder_params : dict
        Parameters for the cylinder
    u_bulk : float
        Bulk velocity
    output_dir : str
        Output directory for saving plots
    """
    if len(vorticity_history) < 5:
        print("Not enough time steps for wake structure analysis")
        return
    
    # Extract basic parameters
    nx, ny = grid['nx'], grid['ny']
    lx, ly = grid['lx'], grid['ly']
    center_x = cylinder_params.get('center_x_ratio', 0.25) * lx
    center_y = cylinder_params.get('center_y_ratio', 0.5) * ly
    radius = cylinder_params.get('radius_ratio', 0.125) * ly
    diameter = 2 * radius
    
    # Create coordinate arrays
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    
    # Get indices of wake region
    wake_start_x = center_x + radius
    i_wake_start = np.argmin(np.abs(x - wake_start_x))
    
    # Use only the latter part of time series for steady state analysis
    start_idx = len(time_points) // 3
    if start_idx < 2:
        start_idx = 0
    
    # Extract average wake vorticity
    avg_vorticity = np.zeros_like(vorticity_history[0])
    for i in range(start_idx, len(vorticity_history)):
        avg_vorticity += vorticity_history[i]
    
    if len(vorticity_history) - start_idx > 0:
        avg_vorticity /= (len(vorticity_history) - start_idx)
    
    # Calculate vorticity standard deviation for unsteadiness analysis
    vorticity_std = np.zeros_like(vorticity_history[0])
    if len(vorticity_history) - start_idx > 1:
        for i in range(start_idx, len(vorticity_history)):
            vorticity_std += (vorticity_history[i] - avg_vorticity)**2
        
        vorticity_std = np.sqrt(vorticity_std / (len(vorticity_history) - start_idx - 1))
    
    # Create visualization of wake properties
    plt.figure(figsize=(15, 10))
    
    # 1. Average vorticity
    plt.subplot(2, 2, 1)
    vort_max = max(abs(np.min(avg_vorticity)), abs(np.max(avg_vorticity)))
    norm = TwoSlopeNorm(vmin=-vort_max, vcenter=0, vmax=vort_max)
    
    contour = plt.contourf(x, y, avg_vorticity.T, levels=50, cmap='RdBu_r', norm=norm)
    plt.colorbar(contour, label='Average Vorticity')
    
    # Draw cylinder
    circle = plt.Circle((center_x, center_y), radius, color='black', fill=True)
    plt.gca().add_artist(circle)
    
    # Add wake length measurement line
    plt.axhline(y=center_y, color='k', linestyle='--', alpha=0.7)
    
    plt.title('Time-Averaged Vorticity')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal')
    
    # 2. Vorticity standard deviation (unsteadiness)
    plt.subplot(2, 2, 2)
    contour = plt.contourf(x, y, vorticity_std.T, levels=50, cmap='viridis')
    plt.colorbar(contour, label='Vorticity std dev')
    
    # Draw cylinder
    circle = plt.Circle((center_x, center_y), radius, color='black', fill=True)
    plt.gca().add_artist(circle)
    
    plt.title('Vorticity Unsteadiness (Std Dev)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal')
    
    # 3. Plot centerline wake profile
    plt.subplot(2, 2, 3)
    j_center = np.argmin(np.abs(y - center_y))
    
    wake_x = x[i_wake_start:]
    wake_vorticity = avg_vorticity[i_wake_start:, j_center]
    
    plt.plot(wake_x, wake_vorticity, 'b-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=center_x, color='r', linestyle='--', label='Cylinder center')
    plt.axvspan(center_x-radius, center_x+radius, color='gray', alpha=0.3, label='Cylinder')
    
    plt.title('Centerline Wake Vorticity')
    plt.xlabel('x')
    plt.ylabel('Average Vorticity')
    plt.legend()
    plt.grid(True)
    
    # 4. Plot wake width
    plt.subplot(2, 2, 4)
    
    # Define wake width based on vorticity threshold
    vort_threshold = 0.1 * vort_max
    wake_width = np.zeros(nx - i_wake_start)
    wake_positions = np.zeros(nx - i_wake_start)
    
    for i, i_pos in enumerate(range(i_wake_start, nx)):
        # Find points where vorticity crosses the threshold
        vort_slice = avg_vorticity[i_pos, :]
        crossings = []
        for j in range(1, ny):
            if (vort_slice[j-1] < vort_threshold and vort_slice[j] >= vort_threshold) or \
               (vort_slice[j-1] > -vort_threshold and vort_slice[j] <= -vort_threshold) or \
               (vort_slice[j-1] >= vort_threshold and vort_slice[j] < vort_threshold) or \
               (vort_slice[j-1] <= -vort_threshold and vort_slice[j] > -vort_threshold):
                crossings.append(j)
        
        if len(crossings) >= 2:
            # Use the extreme points to define wake width
            min_j = min(crossings)
            max_j = max(crossings)
            wake_width[i] = y[max_j] - y[min_j]
            wake_positions[i] = x[i_pos]
    
    # Plot wake width where valid
    valid_indices = wake_width > 0
    if np.any(valid_indices):
        plt.plot(wake_positions[valid_indices], wake_width[valid_indices], 'g-', linewidth=2)
        
        # Calculate and display wake growth rate (linear fit to valid width data)
        try:
            valid_pos = wake_positions[valid_indices]
            valid_width = wake_width[valid_indices]
            
            # Normalize positions by diameter for non-dimensional scaling
            normalized_positions = (valid_pos - (center_x + radius)) / diameter
            
            # Linear fit for the wake growth rate
            if len(normalized_positions) > 2:
                polyfit = np.polyfit(normalized_positions, valid_width, 1)
                wake_growth_rate = polyfit[0]
                
                # Plot the linear fit
                fit_line = np.poly1d(polyfit)
                plt.plot(valid_pos, fit_line(normalized_positions), 'r--', 
                        label=f'Growth rate: {wake_growth_rate:.3f}D')
                
                # Expected theoretical growth rate is typically about 0.1-0.2D for vortex shedding regime
                plt.text(0.05, 0.95, f'Wake growth rate: {wake_growth_rate:.3f}D\n'
                        f'(Theory: ~0.1-0.2D in vortex shedding regime)',
                        transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.7),
                        verticalalignment='top')
        except Exception as e:
            print(f"Could not calculate wake growth rate: {e}")
    
    plt.title('Wake Width vs. Distance')
    plt.xlabel('x')
    plt.ylabel('Wake Width')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/wake_structure_analysis.png", dpi=300)
    plt.close()
    
    # Print key wake characteristics
    print("\n=== Wake Structure Analysis ===")
    wake_vorticity_std = vorticity_std[i_wake_start:, j_center]
    if len(wake_vorticity_std) > 0:
        # Formation length is where vorticity std dev peaks
        max_idx = np.argmax(wake_vorticity_std)
        formation_length = (x[i_wake_start + max_idx] - center_x) / diameter
        print(f"Formation length: {formation_length:.2f}D")
        
        # Get maximum wake width
        if np.any(valid_indices):
            max_width = np.max(wake_width[valid_indices]) / diameter
            print(f"Maximum wake width: {max_width:.2f}D")
            
            if len(normalized_positions) > 2:
                print(f"Wake growth rate: {wake_growth_rate:.3f}D")
    
    # Calculate and return key wake characteristics
    results = {
        'vorticity_magnitude': vort_max,
        'wake_instability': np.mean(vorticity_std[i_wake_start:, j_center]) if len(wake_vorticity_std) > 0 else 0.0
    }
    
    # Estimate formation length (distance to vortex formation)
    try:
        if len(wake_vorticity_std) > 0:
            # Formation length is where vorticity std dev peaks
            max_idx = np.argmax(wake_vorticity_std)
            formation_length = (x[i_wake_start + max_idx] - center_x) / diameter
            results['formation_length'] = formation_length
    except Exception:
        results['formation_length'] = None
    
    return results

def create_comprehensive_report(analysis_results, grid, cylinder_params, reynolds, output_dir='results'):
    """
    Generate a comprehensive report comparing analysis results with theoretical expectations.
    
    Parameters:
    -----------
    analysis_results : dict
        Dictionary with analysis results including Strouhal number
    grid : dict
        Grid information
    cylinder_params : dict
        Parameters for the cylinder
    reynolds : float
        Reynolds number
    output_dir : str
        Output directory for saving report
    """
    # Create report content
    report_data = []
    
    # Extract parameters
    lx, ly = grid['lx'], grid['ly'] 
    center_x = cylinder_params.get('center_x_ratio', 0.25) * lx
    center_y = cylinder_params.get('center_y_ratio', 0.5) * ly
    radius = cylinder_params.get('radius_ratio', 0.125) * ly
    diameter = 2 * radius
    strouhal = analysis_results.get('strouhal', {}).get('strouhal', 0.0)
    
    # Theoretical values for comparison
    # Strouhal number vs. Reynolds number relationship (empirical)
    def theoretical_strouhal(re):
        if re < 50:
            return 0.0  # No vortex shedding
        elif re < 200:
            return 0.18 + 0.001 * (re - 100)  # Approximate linear relationship
        elif re < 350:
            return 0.19 + 0.004 * (re - 200)  # Rising slightly
        else:
            return 0.21  # Plateaus at higher Re
    
    # Theoretical drag coefficient vs. Reynolds number
    def theoretical_drag(re):
        if re < 1:
            return 24 / re  # Stokes flow
        elif re < 10:
            return 24 / re * (1 + 0.15 * re**0.687)  # Intermediate
        elif re < 100:
            return 24 / re * (1 + 0.15 * re**0.687) + 0.42 / (1 + 42500 * re**-1.16)
        elif re < 1000:
            if re < 200:
                return 1.17  # Approximate for Re~100-200
            else:
                return 1.0  # Approximate for Re~200-1000
        else:
            return 0.38  # Drag crisis region
    
    # Collect data for the report
    report_data.append(("Reynolds Number", reynolds, f"Based on diameter (2r = {diameter:.4f})"))
    report_data.append(("Theoretical Strouhal Number", theoretical_strouhal(reynolds), "Based on empirical correlation"))
    report_data.append(("Measured Strouhal Number", strouhal, analysis_results.get('strouhal', {}).get('reliability', 'unknown')))
    
    # Calculate theoretical drag
    theor_drag = theoretical_drag(reynolds)
    measured_drag = analysis_results.get('average_drag', 0.0)
    
    report_data.append(("Theoretical Drag Coefficient", theor_drag, "Based on empirical correlation"))
    report_data.append(("Measured Drag Coefficient", measured_drag, "Time-averaged from simulation"))
    
    # Wake formation length
    formation_length = analysis_results.get('wake_analysis', {}).get('formation_length', None)
    if formation_length is not None:
        # Theoretical formation length based on Re
        if reynolds < 100:
            theoretical_formation = 3.0
        elif reynolds < 200:
            theoretical_formation = 2.0
        else:
            theoretical_formation = 1.0
            
        report_data.append(("Formation Length (L/D)", formation_length, f"Distance to peak vorticity fluctuation"))
        report_data.append(("Theoretical Formation Length", theoretical_formation, "Expected from literature"))
    
    # Create pandas DataFrame for nice display
    df = pd.DataFrame(report_data, columns=["Parameter", "Value", "Notes"])
    
    # Create summary plot
    plt.figure(figsize=(10, 8))
    
    # Plot Strouhal number vs Reynolds number with our data point
    plt.subplot(2, 1, 1)
    
    # Plot theoretical curve
    re_range = np.linspace(50, 500, 100)
    st_range = [theoretical_strouhal(re) for re in re_range]
    plt.plot(re_range, st_range, 'b-', label="Theoretical")
    
    # Plot our data point
    plt.plot(reynolds, strouhal, 'ro', markersize=8, label="Simulation")
    
    plt.title("Strouhal Number vs Reynolds Number")
    plt.xlabel("Reynolds Number")
    plt.ylabel("Strouhal Number")
    plt.grid(True)
    plt.legend()
    
    # Plot drag coefficient vs Reynolds for context
    plt.subplot(2, 1, 2)
    
    # Log scale makes more sense for drag coefficient
    re_range_drag = np.logspace(1, 3, 100)
    drag_range = [theoretical_drag(re) for re in re_range_drag]
    plt.semilogx(re_range_drag, drag_range, 'b-', label="Theoretical")
    
    # Plot our data point
    plt.semilogx(reynolds, measured_drag, 'ro', markersize=8, label="Simulation")
    
    plt.title("Drag Coefficient vs Reynolds Number")
    plt.xlabel("Reynolds Number (log scale)")
    plt.ylabel("Drag Coefficient")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cylinder_flow_report.png", dpi=300)
    plt.close()
    
    # Save report as CSV
    df.to_csv(f"{output_dir}/cylinder_analysis_report.csv", index=False)
    
    # Return the DataFrame for potential display
    return df

def measure_recirculation_bubble(u_center, grid, cylinder_params, output_dir='results'):
    """
    Measure recirculation bubble length and width behind cylinder.
    
    Parameters:
    -----------
    u_center : array
        Interpolated u-velocity at cell centers
    grid : dict
        Grid information
    cylinder_params : dict
        Parameters for the cylinder
    output_dir : str
        Output directory for saving plots
    
    Returns:
    --------
    dict
        Dictionary with recirculation bubble measurements
    """
    nx, ny = grid['nx'], grid['ny']
    lx, ly = grid['lx'], grid['ly']
    
    # Extract cylinder parameters
    center_x = cylinder_params.get('center_x_ratio', 0.25) * lx
    center_y = cylinder_params.get('center_y_ratio', 0.5) * ly
    radius = cylinder_params.get('radius_ratio', 0.125) * ly
    diameter = 2 * radius
    
    # Create coordinate arrays
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    
    # Get centerline index
    j_center = np.argmin(np.abs(y - center_y))
    
    # Find recirculation region (where u < 0) in the wake
    recirculation_mask = u_center < 0
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot u velocity contours
    plt.contourf(x, y, u_center.T, levels=50, cmap='RdBu_r', 
                extend='both', vmin=-0.5, vmax=1.5)
    plt.colorbar(label='u velocity')
    
    # Draw recirculation region boundary (u=0 contour)
    plt.contour(x, y, u_center.T, levels=[0], colors='k', linewidths=2)
    
    # Draw cylinder
    circle = plt.Circle((center_x, center_y), radius, color='black', fill=True)
    plt.gca().add_artist(circle)
    
    # Measure recirculation length along centerline
    recirculation_length = 0
    i_start = np.argmin(np.abs(x - (center_x + radius)))
    
    for i in range(i_start, nx):
        if u_center[i, j_center] < 0:
            recirculation_length = x[i] - (center_x + radius)
        else:
            # Found positive velocity - end of recirculation
            if recirculation_length > 0:
                break
    
    # Normalize by diameter
    recirculation_length_D = recirculation_length / diameter
    
    # Measure maximum width of recirculation region
    recirculation_width = 0
    recirculation_width_x = center_x
    
    for i in range(i_start, nx):
        # Find width at this x position by looking for negative u
        neg_u_indices = np.where(u_center[i, :] < 0)[0]
        if len(neg_u_indices) > 0:
            width = y[neg_u_indices[-1]] - y[neg_u_indices[0]]
            if width > recirculation_width:
                recirculation_width = width
                recirculation_width_x = x[i]
    
    # Normalize by diameter
    recirculation_width_D = recirculation_width / diameter
    
    # Add measurements to plot
    if recirculation_length > 0:
        plt.plot([center_x + radius, center_x + radius + recirculation_length],
               [center_y, center_y], 'r-', linewidth=2)
        plt.text(center_x + radius + recirculation_length/2, center_y + 0.05*ly,
                f'L = {recirculation_length_D:.2f}D', 
                color='red', fontsize=12, ha='center',
                bbox=dict(facecolor='white', alpha=0.7))
    
    if recirculation_width > 0:
        plt.plot([recirculation_width_x, recirculation_width_x],
               [center_y - recirculation_width/2, center_y + recirculation_width/2], 
               'g-', linewidth=2)
        plt.text(recirculation_width_x + 0.02*lx, center_y,
                f'W = {recirculation_width_D:.2f}D', 
                color='green', fontsize=12, va='center',
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.title('Recirculation Bubble Analysis')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(f"{output_dir}/recirculation_bubble.png", dpi=300)
    plt.close()
    
    # Print recirculation analysis info
    print("\n=== Recirculation Bubble Analysis ===")
    print(f"Recirculation length: {recirculation_length_D:.2f}D")
    print(f"Maximum width: {recirculation_width_D:.2f}D")
    print(f"Width location: {(recirculation_width_x - center_x)/diameter:.2f}D downstream")
    
    return {
        'length': recirculation_length_D,
        'width': recirculation_width_D,
        'length_pixels': recirculation_length,
        'width_pixels': recirculation_width
    }