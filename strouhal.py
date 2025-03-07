import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import fft
import os
import yaml

def compute_strouhal_number(force_history_path, start_time=None, plot=True):
    """
    Compute Strouhal number from lift coefficient data in a force_history.npz file.
    Automatically extracts cylinder diameter from the simulation_config.yaml file.
    
    Parameters:
    ----------
    force_history_path : str
        Path to the force_history.npz file
    start_time : float or None
        Time to start the analysis from. If None, will use half of the data.
    plot : bool
        Whether to generate plots of the analysis
        
    Returns:
    -------
    dict
        Dictionary containing Strouhal number, frequency, and analysis parameters
    """
    # Load data from the npz file
    try:
        data = np.load(force_history_path)
        time = data['time']
        cl = data['cl']
        reynolds = data.get('reynolds', None)
        
        # Print summary of loaded data
        print(f"Loaded data from {force_history_path}")
        print(f"Time range: {time[0]:.2f} to {time[-1]:.2f}")
        print(f"Reynolds number: {reynolds}")
    except Exception as e:
        print(f"Error loading force history data: {e}")
        return None
    
    # Find and load the simulation_config.yaml file
    results_dir = os.path.dirname(force_history_path)
    config_path = os.path.join(results_dir, "simulation_config.yaml")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract cylinder parameters
        cylinder_config = config.get('cylinder', {})
        if not cylinder_config.get('enabled', False):
            print("Warning: Cylinder not enabled in configuration")
            cylinder_diameter = None
        else:
            # Calculate cylinder diameter from configuration
            radius_ratio = cylinder_config.get('radius_ratio', 0.08)
            ly = config['grid'].get('ly', 2.0)
            cylinder_diameter = 2 * radius_ratio * ly
            u_bulk = config['flow'].get('u_bulk', 1.0)
            
            print(f"Extracted from config: ly = {ly}, radius_ratio = {radius_ratio}")
            print(f"Calculated cylinder diameter = {cylinder_diameter}")
            print(f"Reference velocity (u_bulk) = {u_bulk}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("Proceeding without cylinder diameter information")
        cylinder_diameter = None
        u_bulk = 1.0
    
    # Determine the start index for analysis
    if start_time is not None:
        start_idx = np.searchsorted(time, start_time)
        if start_idx >= len(time):
            print(f"Start time {start_time} is beyond the end of the data")
            return None
    else:
        # Try to get analysis_start_time from config, otherwise use second half
        try:
            analysis_start_time = cylinder_config.get('analysis_start_time')
            if analysis_start_time is not None:
                start_idx = np.searchsorted(time, analysis_start_time)
                start_time = time[start_idx] if start_idx < len(time) else time[len(time)//2]
                print(f"Using analysis_start_time from config: {analysis_start_time}")
            else:
                start_idx = len(time) // 2
                start_time = time[start_idx]
        except:
            start_idx = len(time) // 2
            start_time = time[start_idx]
    
    print(f"Starting analysis from t = {time[start_idx]:.2f} (index {start_idx})")
    
    # Extract the relevant portion of the data
    analysis_time = time[start_idx:]
    analysis_cl = cl[start_idx:]
    
    # Check if we have enough data for analysis
    if len(analysis_time) < 10:
        print("Not enough data points for reliable frequency detection")
        return None
    
    # 1. Method 1: Peak detection
    # Try with different criteria to find peaks
    peaks = None
    peak_methods = [
        lambda x: find_peaks(x, height=np.mean(x)),
        lambda x: find_peaks(x, height=0),
        lambda x: find_peaks(x),
        lambda x: find_peaks(-x),  # Detect troughs
        lambda x: find_peaks(x, distance=10)  # Lower frequency
    ]
    
    for peak_method in peak_methods:
        try:
            peaks, peak_props = peak_method(analysis_cl)
            if len(peaks) >= 2:
                print(f"Found {len(peaks)} peaks with method {peak_methods.index(peak_method) + 1}")
                break
        except:
            continue
    
    peak_strouhal = None
    peak_frequency = None
    peak_times = []
    
    if peaks is not None and len(peaks) >= 2:
        peak_times = [analysis_time[p] for p in peaks]
        periods = np.diff(peak_times)
        avg_period = np.mean(periods)
        peak_frequency = 1.0 / avg_period
        
        if cylinder_diameter is not None:
            peak_strouhal = peak_frequency * cylinder_diameter / u_bulk
        
        print(f"Peak detection found {len(peaks)} peaks")
        print(f"Average period: {avg_period:.4f}")
        print(f"Corresponding frequency: {peak_frequency:.4f} Hz")
        if peak_strouhal:
            print(f"Strouhal number (peak method): {peak_strouhal:.4f}")
    else:
        print("Not enough peaks detected for frequency calculation")
    
    # 2. Method 2: FFT analysis
    # Calculate average sampling rate
    dt = np.mean(np.diff(analysis_time))
    fs = 1.0 / dt
    
    # Normalize the signal by removing mean
    signal = analysis_cl - np.mean(analysis_cl)
    
    # Compute FFT
    n = len(signal)
    yf = fft.rfft(signal)
    xf = fft.rfftfreq(n, dt)
    
    # Find dominant frequency (exclude DC component)
    mask = xf > 0
    if np.any(mask):
        dominant_idx = np.argmax(np.abs(yf[mask])) + 1
        dominant_freq = xf[dominant_idx]
        
        fft_strouhal = None
        if cylinder_diameter is not None:
            fft_strouhal = dominant_freq * cylinder_diameter / u_bulk
        
        print(f"FFT analysis: dominant frequency: {dominant_freq:.4f} Hz")
        if fft_strouhal:
            print(f"Strouhal number (FFT method): {fft_strouhal:.4f}")
    else:
        dominant_freq = None
        fft_strouhal = None
        print("FFT analysis: couldn't determine dominant frequency")
    
    # Create visualizations if requested
    if plot:
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Lift coefficient time history
        plt.subplot(3, 1, 1)
        plt.plot(time, cl, 'b-', alpha=0.5, label='Full data')
        plt.plot(analysis_time, analysis_cl, 'b-', linewidth=1.5, label='Analysis window')
        plt.axvline(x=start_time, color='r', linestyle='--', label=f'Analysis start (t={start_time:.2f})')
        
        # Mark peaks if detected
        if len(peak_times) > 0:
            plt.plot(peak_times, [analysis_cl[peaks[i]] for i in range(len(peaks))], 
                     'ro', markersize=4, label='Detected peaks')
        
        plt.title('Lift Coefficient History')
        plt.xlabel('Time')
        plt.ylabel('Cl')
        plt.grid(True)
        plt.legend()
        
        # Plot 2: Power spectrum
        plt.subplot(3, 1, 2)
        if dominant_freq is not None:
            power = np.abs(yf) ** 2
            plt.semilogy(xf, power, 'g-')
            plt.axvline(x=dominant_freq, color='r', linestyle='--', 
                       label=f'Dominant frequency: {dominant_freq:.4f} Hz')
            plt.title('Power Spectrum')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power')
            plt.grid(True)
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'Could not compute power spectrum', 
                   ha='center', va='center', transform=plt.gca().transAxes)
        
        # Plot 3: Zoomed in lift coefficient with period visualization
        plt.subplot(3, 1, 3)
        
        # Choose a reasonable range to display (a few periods)
        if peak_frequency:
            display_cycles = min(5, len(analysis_time) * dt * peak_frequency / 2)
            display_time = display_cycles / peak_frequency
            display_end = min(analysis_time[-1], analysis_time[0] + display_time)
            display_mask = (analysis_time >= analysis_time[0]) & (analysis_time <= display_end)
            
            plt.plot(analysis_time[display_mask], analysis_cl[display_mask], 'b-', linewidth=1.5)
            
            # Add period markers
            if len(peak_times) > 1:
                for i, t in enumerate(peak_times):
                    if t <= display_end:
                        plt.axvline(x=t, color='r', linestyle=':', alpha=0.7)
                        if i < len(peak_times) - 1:
                            plt.annotate(f'Period: {peak_times[i+1]-t:.3f}', 
                                       xy=(0.5*(t+peak_times[i+1]), max(analysis_cl[display_mask])),
                                       xytext=(0, 10), textcoords='offset points',
                                       ha='center', fontsize=8)
            
            plt.title('Zoomed Lift Coefficient with Period Visualization')
            plt.xlabel('Time')
            plt.ylabel('Cl')
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'Cannot display periods: No frequency detected', 
                   ha='center', va='center', transform=plt.gca().transAxes)
        
        # Add cylinder parameters to the title
        if cylinder_diameter is not None:
            plt.suptitle(f'Strouhal Analysis (Re = {reynolds}, D = {cylinder_diameter:.3f})', fontsize=14)
        else:
            plt.suptitle(f'Strouhal Analysis (Re = {reynolds})', fontsize=14)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        output_path = os.path.join(results_dir, "strouhal_analysis.png")
        plt.savefig(output_path, dpi=200)
        print(f"Saved visualization to {output_path}")
        plt.show()
    
    # Return results
    results = {
        'start_time': start_time,
        'peak_method': {
            'frequency': peak_frequency,
            'strouhal': peak_strouhal,
            'num_peaks': len(peaks) if peaks is not None else 0,
            'periods': np.diff(peak_times).tolist() if len(peak_times) > 1 else []
        },
        'fft_method': {
            'frequency': dominant_freq,
            'strouhal': fft_strouhal,
        },
        'reynolds': reynolds,
        'cylinder_diameter': cylinder_diameter,
        'u_bulk': u_bulk
    }
    
    return results

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute Strouhal number from force history data')
    parser.add_argument('file', type=str, help='Path to force_history.npz file')
    parser.add_argument('--start', type=float, help='Time to start analysis from', default=None)
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    
    args = parser.parse_args()
    
    compute_strouhal_number(args.file, args.start, not args.no_plot)