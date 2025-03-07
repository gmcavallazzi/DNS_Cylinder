import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import os
import time
import argparse
import matplotlib.pyplot as plt
import yaml

# Import from the solver module
from solver import (
    create_stretched_grid, 
    calculate_stable_dt, 
    plot_grid_distribution,
)
from adams_bashforth import fractional_step_ab2
from turbulent_helpers import inject_perturbations, enhanced_initial_condition_with_vortices
from utils import (
    calculate_wall_shear_stresses,
    calculate_pressure_gradient,
    load_config,
    calculate_enforced_bulk_velocity,
    plot_velocity_field
)
# Import the new cylinder visualization functions
from cylinder_visualization import (
    plot_cylinder_flow,
    analyze_vortex_shedding,
    generate_space_time_plot,
    calculate_vorticity
)
from forces_analysis import (
    calculate_detailed_forces,
    plot_force_coefficient_history,
    plot_pressure_distribution
)

def run_simulation_ab2(config):
    """
    Run simulation using configuration parameters.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary loaded from YAML file
    """
    # Extract configuration sections
    grid_config = config['grid']
    sim_config = config['simulation'] 
    flow_config = config['flow']
    bc_config = config['bc']
    
    # Extract parameters from config
    nx = grid_config.get('nx', 64)
    ny = grid_config.get('ny', 64)
    lx = grid_config.get('lx', np.pi)
    ly = grid_config.get('ly', 2.0)
    
    reynolds = flow_config.get('reynolds', 1000)
    u_bulk = flow_config.get('u_bulk', 1.0)
    bc = bc_config.get('kind', 'periodic')
    
    cfl = sim_config.get('cfl', 0.2)
    t_end = sim_config.get('t_end', 20.0)
    save_interval = sim_config.get('save_interval', 0.5)
    print_interval = sim_config.get('print_interval', 10)
    output_dir = sim_config.get('output_dir', 'results_ab2')
    safety_factor = sim_config.get('safety_factor', 0.5)
    
    # Extract perturbation parameters
    perturbation_config = config.get('perturbations', {})
    perturbation_amplitude = perturbation_config.get('amplitude', 0.5)
    injection_frequency = perturbation_config.get('injection_frequency', 1.0)
    injection_amplitude = perturbation_config.get('injection_amplitude', 0.3)
    
    # Optional cylinder parameters
    cylinder_config = config.get('cylinder', {})
    use_cylinder = cylinder_config.get('enabled', False)
    
    print(f"Starting Channel Flow Simulation with Adams-Bashforth 2 Integration")
    print(f"Grid: {nx}x{ny}, Re={reynolds}, CFL={cfl}")
    print(f"Using vortex-enhanced initial conditions with perturbation amplitude: {perturbation_amplitude}")
    print(f"Perturbation injection: Every {injection_frequency} time units with amplitude {injection_amplitude}")
    if use_cylinder:
        print(f"Cylinder obstacle enabled")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the configuration to the output directory for reproducibility
    with open(f"{output_dir}/simulation_config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create the staggered grid
    stretch_factor = grid_config.get('stretch_factor', 1.5)
    grid = create_stretched_grid(nx, ny, lx, ly, stretch_factor=stretch_factor)

    # Optional: visualize grid distribution
    if grid_config.get('plot_grid', False):
        plot_grid_distribution(grid)
    
    # If we have a cylinder, compute viscosity based on cylinder diameter
    if use_cylinder:
        radius_ratio = cylinder_config.get('radius_ratio', 0.08)
        diameter = 2 * radius_ratio * ly
        nu = u_bulk * diameter / reynolds
        print(f"\nCylinder flow configuration:")
        print(f"  Diameter = {diameter:.3f}")
        print(f"  Reynolds number (based on diameter) = {reynolds}")
        print(f"  Kinematic viscosity = {nu:.6f}")
    else:
        # Traditional channel Reynolds number
        h = ly/2  # channel half-height
        nu = u_bulk * h / reynolds
        print(f"\nChannel flow configuration:")
        print(f"  Half-height = {h:.3f}")
        print(f"  Reynolds number (based on half-height) = {reynolds}")
        print(f"  Kinematic viscosity = {nu:.6f}")
    
    rho = 1.0  # density
    
    # Calculate theoretical laminar flow pressure gradient
    h = ly/2  # channel half-height
    dp_dx_laminar = -12.0 * nu * u_bulk / (h * h)
    tau_w_laminar = abs(dp_dx_laminar * h / 2.0)
    
    print(f"Domain Size: {lx}x{ly}")
    print(f"Viscosity: {nu}")
    print(f"Theoretical laminar dP/dx: {dp_dx_laminar:.6f}")
    print(f"Theoretical laminar wall shear stress: {tau_w_laminar:.6f}")
    
    # Setup initial conditions with vortices
    u, v, p = enhanced_initial_condition_with_vortices(grid, u_bulk, perturbation_amplitude)
    
    # Prepare cylinder parameters if enabled
    cylinder_params = None
    if use_cylinder:
        cylinder_params = {
            'center_x_ratio': cylinder_config.get('center_x_ratio', 0.25),
            'center_y_ratio': cylinder_config.get('center_y_ratio', 0.5),
            'radius_ratio': cylinder_config.get('radius_ratio', 0.125)
        }
    
    # Create first visualization with the new cylinder flow visualization
    if use_cylinder:
        plot_cylinder_flow(grid, u, v, cylinder_params, title="Initial Flow Field", step=0, output_dir=output_dir)
    else:
        plot_velocity_field(grid, u, v, title="Initial Velocity Field", step=0, output_dir=output_dir, u_bulk=u_bulk)
    
    # Initialize time tracking variables
    t = 0.0
    step = 0
    next_save = save_interval
    next_print = print_interval
    next_injection = injection_frequency
    vorticity_save_interval = min(0.05, save_interval)
    next_vorticity_save = vorticity_save_interval
    
    # Performance metrics
    step_times = []
    pressure_iterations = []
    divergence_values = []
    
    # Tracking metrics
    time_points = []
    driving_pressure_gradients = []
    computed_pressure_gradients = []
    wall_shear_top = []
    wall_shear_bottom = []
    bulk_velocities = []
    drag_coefficients = []
    
    # For cylinder flow analysis
    vorticity_history = []
    vorticity_time_points = []
    force_history = []

    # For Adams-Bashforth: store previous RHS terms
    prev_rhs = None  # Will be None for first step (uses Euler)
    
    # Timer for performance tracking
    start_time = time.time()
    
    print("\nStarting time integration using Adams-Bashforth 2")
    if use_cylinder:
        print(f"{'Step':>6} {'Time':>10} {'dt':>10} {'CFL':>6} {'CD':>8} {'Bulk U':>8} {'Step Time':>10}")
    else:
        print(f"{'Step':>6} {'Time':>10} {'dt':>10} {'CFL':>6} {'dP/dx':>12} {'τ_wall':>12} {'Bulk U':>8} {'Step Time':>10}")
    print("-" * 80)
    
    # Time integration loop
    while t < t_end:
        # Track iteration time
        iter_start = time.time()
        
        # Calculate adaptive timestep based on CFL
        dt = calculate_stable_dt(u, v, 
                                  nx=grid['nx'], ny=grid['ny'], 
                                  dx=grid['dx'], dy=grid['dy'], 
                                  nu=nu, cfl=cfl, 
                                  safety_factor=safety_factor)
        
        # Check if we should inject perturbations
        max_injection_time = perturbation_config.get('max_injection_time', 5.0)
        if t >= next_injection and t < max_injection_time:
            print(f"Injecting perturbations at t = {t:.2f}")
            u, v = inject_perturbations(u, v, grid, u_bulk, t, 
                                       perturbation_amplitude=injection_amplitude,
                                       injection_frequency=injection_frequency)
            next_injection = t + injection_frequency
        
        # Calculate current bulk velocity before advancing
        curr_bulk_vel = calculate_enforced_bulk_velocity(u, grid)
            
        # Advance one time step using Adams-Bashforth
        u, v, p, conv_u, conv_v, diff_u, diff_v, p_iters, div_after, curr_rhs = fractional_step_ab2(
            u, v, p, grid, dt, nu, rho, bc, target_flow_rate=u_bulk, prev_rhs=prev_rhs,
            cylinder_params=cylinder_params
        )
        
        # Store current RHS for next time step
        prev_rhs = curr_rhs
        
        # Calculate pressure gradient from wall shear stresses
        tau_w_bot, tau_w_top = calculate_wall_shear_stresses(u, grid, nu, rho)
        
        # Calculate the driving pressure gradient from the wall shear stresses
        driving_dp_dx = -(tau_w_bot + tau_w_top) / ly
        
        # Calculate the pressure gradient from the pressure field
        computed_dp_dx = calculate_pressure_gradient(p, grid)
        
        # Calculate drag coefficient if cylinder is present
        cd = 0.0
        forces_data = None
        if use_cylinder:
            # Use the improved force calculation
            forces_data = calculate_detailed_forces(u, v, p, grid, cylinder_params, nu, rho, u_bulk, input_reynolds=reynolds)
            cd = forces_data['cd']
        
        # Track metrics
        pressure_iterations.append(p_iters)
        divergence_values.append(div_after)
        
        # Track pressure gradient and shear stress metrics
        time_points.append(t)
        driving_pressure_gradients.append(driving_dp_dx)
        computed_pressure_gradients.append(computed_dp_dx)
        wall_shear_top.append(tau_w_top)
        wall_shear_bottom.append(tau_w_bot)
        bulk_velocities.append(curr_bulk_vel)
        if use_cylinder:
            drag_coefficients.append(cd)
            force_history.append(forces_data)
        
        # Calculate iteration time
        iter_time = (time.time() - iter_start) * 1000  # Convert to milliseconds
        step_times.append(iter_time)
        
        # Update time
        t += dt
        step += 1
        
        # Print statistics
        if step > next_print:
            if use_cylinder:
                print(f"{step:6d} {t:10.4f} {dt:10.6f} {cfl:6.3f} {cd:8.4f} {forces_data['cl']:8.4f} {curr_bulk_vel:8.4f} {iter_time:10.2f}ms")
            else:
                print(f"{step:6d} {t:10.4f} {dt:10.6f} {cfl:6.3f} {driving_dp_dx:12.6f} {(tau_w_bot+tau_w_top)/2:12.6f} {curr_bulk_vel:8.4f} {iter_time:10.2f}ms")
            next_print = step + print_interval
            
            # If divergence is too high, abort
            max_divergence = sim_config.get('max_divergence', 1.0)
            if div_after > max_divergence or np.isnan(div_after):
                print("ERROR: Simulation diverged due to high divergence.")
                break
        
        # Compute vorticity for vortex shedding analysis
        if t >= next_vorticity_save:
            # Calculate vorticity for this step without visualization
            current_vorticity = calculate_vorticity(u, v, grid)
            vorticity_history.append(current_vorticity)
            vorticity_time_points.append(t)
            next_vorticity_save = t + vorticity_save_interval

        # Save results
        if t >= next_save:
            # Create visualizations based on whether cylinder is present
            if use_cylinder:
                # Calculate vorticity for this step
                vorticity = plot_cylinder_flow(grid, u, v, cylinder_params, 
                                             title=f"Flow Field (t={t:.2f})", 
                                             step=step, output_dir=output_dir)
                
                # Store vorticity for later analysis
                #vorticity_history.append(vorticity)
            else:
                plot_velocity_field(grid, u, v, title=f"Velocity Field (t={t:.2f})", 
                                   step=step, output_dir=output_dir, u_bulk=u_bulk)
                
            # Create force coefficient history plot if cylinder is present
            if use_cylinder and len(force_history) > 1:

                # Get analysis start time from config (if available)
                analysis_start_time = cylinder_config.get('analysis_start_time', None)
        
                # Plot force coefficient history
                plot_force_coefficient_history(time_points[-len(force_history):], force_history, 
                                            output_dir=output_dir, analysis_start_time=analysis_start_time)
        
                # Occasionally plot pressure distribution (every 4th save)
                if step % 4 == 0:
                    plot_pressure_distribution(u, v, p, grid, cylinder_params, t, 
                                            output_dir=output_dir, analysis_start_time=analysis_start_time)

            # Save metrics
            metrics = {
                'steps': np.array(range(len(step_times))),
                'time_points': np.array(time_points),
                'step_times': np.array(step_times),
                'pressure_iterations': np.array(pressure_iterations),
                'divergence_values': np.array(divergence_values),
                'bulk_velocities': np.array(bulk_velocities),
            }
            
            # Add cylinder-specific metrics if available
            if use_cylinder:
                metrics['drag_coefficients'] = np.array(drag_coefficients)
            else:
                # Add channel flow metrics
                metrics['driving_pressure_gradients'] = np.array(driving_pressure_gradients)
                metrics['computed_pressure_gradients'] = np.array(computed_pressure_gradients)
                metrics['wall_shear_top'] = np.array(wall_shear_top)
                metrics['wall_shear_bottom'] = np.array(wall_shear_bottom)
            
            np.savez(f"{output_dir}/metrics.npz", **metrics)
            
            next_save = t + save_interval
    
    # End of time integration loop
    
    # Run vortex shedding analysis if cylinder was simulated and we have enough snapshots
    if use_cylinder and len(vorticity_history) > 5:
        print("\nAnalyzing vortex shedding patterns...")
        
        # Get analysis start time from config (if available)
        analysis_start_time = cylinder_config.get('analysis_start_time', None)
        if analysis_start_time is not None:
            print(f"Using analysis start time t = {analysis_start_time} for frequency calculations")
        
        analyze_vortex_shedding(vorticity_history, vorticity_time_points, 
                               grid, cylinder_params, output_dir, u_bulk,
                               analysis_start_time=analysis_start_time)
        
        # Generate space-time plot
        generate_space_time_plot(vorticity_history, vorticity_time_points, 
                                grid, cylinder_params, output_dir)
        
        if use_cylinder:
            print("\nCylinder flow statistics:")
            if len(force_history) > 0:
                # Use analysis start time from config if available
                analysis_start_time = cylinder_config.get('analysis_start_time', None)
                if analysis_start_time is not None and analysis_start_time > time_points[0]:
                    # Find the index where time >= analysis_start_time
                    analysis_start_idx = next((i for i, t in enumerate(time_points[-len(force_history):]) 
                                            if t >= analysis_start_time), 0)
                    print(f"Using data from t = {time_points[-len(force_history):][analysis_start_idx]:.2f} onwards for statistics")
                else:
                    # Default to using second half of the data
                    analysis_start_idx = len(force_history) // 2
                    print(f"Using second half of data for statistics (from t = {time_points[-len(force_history)//2]:0.2f})")

                # Extract force values for the analysis window
                cd_values = [f['cd'] for f in force_history[analysis_start_idx:]]
                cl_values = [f['cl'] for f in force_history[analysis_start_idx:]]
                cd_p_values = [f['cd_pressure'] for f in force_history[analysis_start_idx:]]
                cd_v_values = [f['cd_viscous'] for f in force_history[analysis_start_idx:]]
            
                # Calculate statistics
                cd_mean = np.mean(cd_values)
                cl_rms = np.std(cl_values)
                cd_p_mean = np.mean(cd_p_values)
                cd_v_mean = np.mean(cd_v_values)
            
                print(f"  Final drag coefficient: {force_history[-1]['cd']:.4f}")
                print(f"  Average drag coefficient: {cd_mean:.4f}")
                print(f"    - Pressure contribution: {cd_p_mean:.4f} ({100*cd_p_mean/cd_mean:.1f}%)")
                print(f"    - Viscous contribution: {cd_v_mean:.4f} ({100*cd_v_mean/cd_mean:.1f}%)")
                print(f"  RMS lift coefficient: {cl_rms:.4f}")
            
                # Save force history for later analysis
                force_data = {
                    'time': np.array(time_points[-len(force_history):]),
                    'cd': np.array([f['cd'] for f in force_history]),
                    'cl': np.array([f['cl'] for f in force_history]),
                    'cd_pressure': np.array([f['cd_pressure'] for f in force_history]),
                    'cd_viscous': np.array([f['cd_viscous'] for f in force_history]),
                    'cl_pressure': np.array([f['cl_pressure'] for f in force_history]),
                    'cl_viscous': np.array([f['cl_viscous'] for f in force_history]),
                    'reynolds': force_history[0]['reynolds'],
                    'analysis_start_time': analysis_start_time
                }
            
                np.savez(f"{output_dir}/force_history.npz", **force_data)
    
    # Print performance statistics
    elapsed = time.time() - start_time
    avg_step_time = np.mean(step_times)
    avg_pressure_iter = np.mean(pressure_iterations)
    avg_divergence = np.mean(divergence_values)
    
    print("\nSimulation completed")
    print(f"Time steps: {step}")
    print(f"Final time: {t:.4f}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Average time per step: {avg_step_time:.2f} milliseconds")
    
    if use_cylinder:
        print("\nCylinder flow statistics:")
        if len(drag_coefficients) > 0:
            print(f"  Final drag coefficient: {drag_coefficients[-1]:.6f}")
            if len(drag_coefficients) > 10:
                print(f"  Average drag coefficient: {np.mean(drag_coefficients[len(drag_coefficients)//2:]):.6f}")
    else:
        # Calculate statistics from second half of simulation (after initial transient)
        avg_driving_dp_dx = np.mean(driving_pressure_gradients[len(driving_pressure_gradients)//2:])
        
        # Calculate theoretical vs. actual comparison
        dp_dx_ratio = avg_driving_dp_dx / dp_dx_laminar
        
        print("\nFlow statistics (second half of simulation):")
        print(f"  Theoretical laminar dP/dx: {dp_dx_laminar:.6f}")
        print(f"  Average driving dP/dx: {avg_driving_dp_dx:.6f}")
        print(f"  Ratio (actual/theoretical): {dp_dx_ratio:.4f}")
        print(f"  Ratio > 1 indicates turbulent flow (more resistance)")
    
    return u, v, p, grid

# Execute the simulation when the script is run directly
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Channel Flow Simulation with Adams-Bashforth')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML configuration file')
    parser.add_argument('--output', type=str, help='Override output directory from config')
    
    args = parser.parse_args()
    
    # Load configuration from YAML file
    config = load_config(args.config)
    
    # Override output directory if specified in command line
    if args.output:
        config['simulation']['output_dir'] = args.output
        print(f"Overriding output directory: {args.output}")
    
    # Run the simulation with parsed arguments
    run_simulation_ab2(config)