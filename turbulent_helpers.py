# turbulent_helpers.py - JIT compatible version
import numpy as np
from numba import jit, vectorize, prange
from solver import apply_bc

@vectorize(['float64(float64)'], target='parallel')
def wall_enhanced_damping(y_norm):
    """Vectorized damping function"""
    return np.sin(np.pi * y_norm) * (1.0 - 0.5*np.sin(np.pi * y_norm) + 0.3*np.sin(4*np.pi*y_norm))

@jit(nopython=True)
def inject_perturbations_jit(u, v, nx, ny, dx, dy, x_p, y_p, x_v, y_v, lx, ly, t, u_bulk, perturbation_amplitude=0.1, injection_frequency=1.0):
    """JIT-compatible version of inject_perturbations"""
    # Skip injection if not at right time
    if abs(t % injection_frequency) > 0.1 * injection_frequency:
        return u, v
    
    # Add structured perturbations
    num_modes_x = 8
    num_modes_y = 6
    
    # 1. Add streamwise vortices
    for i in range(1, nx+2):
        x_pos = (i-1) * dx
        for j in range(1, ny+1):
            y_pos = y_p[j]
            y_norm = y_pos / ly
            
            # Get damping from vectorized function
            damping = np.sin(np.pi * y_norm) * (1.0 - 0.5*np.sin(np.pi * y_norm) + 0.3*np.sin(4*np.pi*y_norm))
            
            # Scale damping to enhance near-wall regions
            if y_norm < 0.2 or y_norm > 0.8:
                damping *= 1.5
            
            # Add structured modes with random variation
            for m in range(1, num_modes_x+1):
                for n in range(1, num_modes_y+1):
                    # Random phase for each injection and each mode
                    phase_x = 2 * np.pi * np.random.random()
                    phase_y = 2 * np.pi * np.random.random()
                    
                    # Random wavenumber variations
                    kx = 2 * np.pi * m / lx * (1.0 + 0.3 * np.random.uniform(-1, 1))
                    ky = np.pi * n / ly * (1.0 + 0.3 * np.random.uniform(-1, 1))
                    
                    # Random amplitude
                    ampl_variation = 1.0 + 0.6 * np.random.uniform(-1, 1)
                    ampl = perturbation_amplitude * u_bulk * damping * ampl_variation / (m*n)**0.5
                    
                    # Add perturbation with random pattern function
                    pattern_type = np.random.randint(0, 3)
                    if pattern_type == 0:
                        u[i, j] += ampl * np.sin(kx * x_pos + phase_x) * np.sin(ky * y_pos + phase_y)
                    elif pattern_type == 1:
                        u[i, j] += ampl * np.cos(kx * x_pos + phase_x) * np.sin(ky * y_pos + phase_y)
                    else:
                        u[i, j] += ampl * np.sin(kx * x_pos + phase_x) * np.cos(ky * y_pos + phase_y)
    
    # 2. Add vertical velocity perturbations
    for i in range(1, nx+1):
        x_pos = (i-0.5) * dx
        for j in range(1, ny+2):
            if j == 1 or j == ny+1:
                continue
                
            y_pos = y_v[j]
            y_norm = y_pos / ly
            
            # Inject perturbations throughout with stronger focus near walls
            damping = np.sin(np.pi * y_norm) * (1.0 - 0.5*np.sin(np.pi * y_norm) + 0.3*np.sin(4*np.pi*y_norm))
            
            # Scale damping to enhance near-wall regions
            if y_norm < 0.25 or y_norm > 0.75:
                damping *= 1.5
            
            for m in range(1, num_modes_x+1):
                for n in range(1, num_modes_y+1):
                    # Random phase
                    phase_x = 2 * np.pi * np.random.random()
                    phase_y = 2 * np.pi * np.random.random()
                    
                    # Random wavenumber variations
                    kx = 2 * np.pi * m / lx * (1.0 + 0.3 * np.random.uniform(-1, 1))
                    ky = np.pi * n / ly * (1.0 + 0.3 * np.random.uniform(-1, 1))
                    
                    # Random amplitude
                    ampl_variation = 1.0 + 0.6 * np.random.uniform(-1, 1)
                    ampl = 0.8 * perturbation_amplitude * u_bulk * damping * ampl_variation / (m*n)**0.5
                    
                    # Add perturbation with random pattern
                    pattern_type = np.random.randint(0, 3)
                    if pattern_type == 0:
                        v[i, j] += ampl * np.cos(kx * x_pos + phase_x) * np.sin(ky * y_pos + phase_y)
                    elif pattern_type == 1:
                        v[i, j] += ampl * np.sin(kx * x_pos + phase_x) * np.sin(ky * y_pos + phase_y)
                    else:
                        v[i, j] += ampl * np.cos(kx * x_pos + phase_x) * np.cos(ky * y_pos + phase_y)
    
    # 3. Add pure random noise for additional irregularity
    for i in range(1, nx+2):
        for j in range(1, ny+1):
            y_norm = y_p[j] / ly
            
            # More noise near walls
            noise_factor = 0.3
            if y_norm < 0.2 or y_norm > 0.8:
                noise_factor = 0.5
                
            wall_damping = np.sin(np.pi * y_norm)
            u[i, j] += noise_factor * perturbation_amplitude * u_bulk * wall_damping * np.random.normal(0, 1)
    
    for i in range(1, nx+1):
        for j in range(1, ny+2):
            if j == 1 or j == ny+1:
                continue
                
            y_norm = y_v[j] / ly
            
            # More noise near walls
            noise_factor = 0.25
            if y_norm < 0.25 or y_norm > 0.75:
                noise_factor = 0.4
                
            wall_damping = np.sin(np.pi * y_norm)
            v[i, j] += noise_factor * perturbation_amplitude * u_bulk * wall_damping * np.random.normal(0, 1)
    
    return u, v

# Non-JIT wrapper function that takes the grid dictionary
def inject_perturbations(u, v, grid, u_bulk, t, perturbation_amplitude=0.1, injection_frequency=1.0):
    """Wrapper function that extracts grid components before calling JIT function"""
    nx, ny = grid['nx'], grid['ny']
    lx, ly = grid['lx'], grid['ly']
    dx, dy = grid['dx'], grid['dy']
    x_p, y_p = grid['x_p'], grid['y_p']
    x_v, y_v = grid['x_v'], grid['y_v']
    
    u, v = inject_perturbations_jit(u, v, nx, ny, dx, dy, x_p, y_p, x_v, y_v, 
                                  lx, ly, t, u_bulk, perturbation_amplitude, injection_frequency)
    
    # Make sure we maintain boundary conditions
    p_dummy = np.zeros((nx+2, ny+2))
    u, v, _ = apply_bc(u, v, p_dummy, nx, ny)
    
    return u, v

@jit(nopython=True)
def add_vortex_jit(u, v, nx, ny, dx, dy, x_p, y_p, x_v, y_v, center_x, center_y, size, strength, clockwise=True):
    """JIT-compatible version of add_vortex"""
    # Set direction
    if not clockwise:
        strength = -strength
    
    # Add vortex component to u-velocity
    for i in range(1, nx+2):
        x = (i-1) * dx  # u-grid x-coordinate
        for j in range(1, ny+1):
            y = y_p[j]  # u-grid y-coordinate
            
            # Distance from vortex center
            dx_vortex = x - center_x
            dy_vortex = y - center_y
            r = np.sqrt(dx_vortex*dx_vortex + dy_vortex*dy_vortex)
            
            # Vortex profile (Gaussian decay)
            if r < 3*size:  # Limit influence to 3x vortex size
                # Velocity magnitude decreases with distance from center
                vel_magnitude = strength * (r/size) * np.exp(-(r/size)**2)
                
                # For u, we need the y-component of the vortex
                # theta is the angle in polar coordinates
                if r > 1e-10:  # Avoid division by zero
                    sin_theta = dy_vortex / r
                    u[i, j] += vel_magnitude * sin_theta
    
    # Add vortex component to v-velocity
    for i in range(1, nx+1):
        x = (i-0.5) * dx  # v-grid x-coordinate
        for j in range(1, ny+2):
            y = y_v[j]  # v-grid y-coordinate
            
            # Distance from vortex center
            dx_vortex = x - center_x
            dy_vortex = y - center_y
            r = np.sqrt(dx_vortex*dx_vortex + dy_vortex*dy_vortex)
            
            # Vortex profile (Gaussian decay)
            if r < 3*size:  # Limit influence to 3x vortex size
                # Velocity magnitude decreases with distance from center
                vel_magnitude = strength * (r/size) * np.exp(-(r/size)**2)
                
                # For v, we need the negative x-component of the vortex
                # theta is the angle in polar coordinates
                if r > 1e-10:  # Avoid division by zero
                    cos_theta = dx_vortex / r
                    v[i, j] -= vel_magnitude * cos_theta

def add_vortex(u, v, grid, center_x, center_y, size, strength, clockwise=True):
    """Wrapper function for add_vortex_jit"""
    nx, ny = grid['nx'], grid['ny']
    dx, dy = grid['dx'], grid['dy']
    x_p, y_p = grid['x_p'], grid['y_p']
    x_v, y_v = grid['x_v'], grid['y_v']
    
    add_vortex_jit(u, v, nx, ny, dx, dy, x_p, y_p, x_v, y_v, center_x, center_y, size, strength, clockwise)

@jit(nopython=True)
def enhanced_initial_condition_with_vortices_jit(nx, ny, dx, dy, x_p, y_p, x_u, y_u, x_v, y_v, lx, ly, u_bulk=1.0, perturbation_amplitude=0.3):
    """JIT-compatible version of enhanced_initial_condition_with_vortices"""
    # Initialize fields with correct dimensions
    u = np.zeros((nx+3, ny+2))    # u-velocity
    v = np.zeros((nx+2, ny+3))    # v-velocity
    p = np.zeros((nx+2, ny+2))    # pressure
    
    # Set up base parabolic profile for u-velocity (Poiseuille flow)
    for i in range(1, nx+2):
        for j in range(1, ny+1):
            # Use y coordinate to compute normalized position
            y_pos = y_p[j]
            y_norm = y_pos / ly
            
            # Parabolic profile: u = 6*u_bulk*y*(1-y) for unit height
            u[i, j] = 6.0 * u_bulk * y_norm * (1.0 - y_norm)
    
    # Add random perturbations first (lower amplitude)
    for i in range(1, nx+2):
        for j in range(1, ny+1):
            y_norm = y_p[j] / ly
            wall_damping = np.sin(np.pi * y_norm)
            u[i, j] += 0.1 * perturbation_amplitude * u_bulk * wall_damping * np.random.normal(0, 1)
    
    for i in range(1, nx+1):
        for j in range(1, ny+2):
            if j == 1 or j == ny+1:  # Keep walls as no-slip
                continue
            y_norm = y_v[j] / ly
            wall_damping = np.sin(np.pi * y_norm)
            v[i, j] += 0.1 * perturbation_amplitude * u_bulk * wall_damping * np.random.normal(0, 1)
    
    return u, v, p

def enhanced_initial_condition_with_vortices(grid, u_bulk=1.0, perturbation_amplitude=0.3):
    """Wrapper function for enhanced_initial_condition_with_vortices_jit"""
    nx, ny = grid['nx'], grid['ny']
    lx, ly = grid['lx'], grid['ly']
    dx, dy = grid['dx'], grid['dy']
    x_p, y_p = grid['x_p'], grid['y_p']
    x_u, y_u = grid['x_u'], grid['y_u']
    x_v, y_v = grid['x_v'], grid['y_v']
    
    # Get base fields from JIT function
    u, v, p = enhanced_initial_condition_with_vortices_jit(
        nx, ny, dx, dy, x_p, y_p, x_u, y_u, x_v, y_v, lx, ly, u_bulk, perturbation_amplitude)
    
    # Now add explicit counter-rotating vortex pairs (this part uses the grid dictionary)
    # Define vortex parameters
    vortex_count = 8  # Number of vortex pairs
    vortex_strength = perturbation_amplitude * u_bulk  # Strength of vortices
    vortex_size = 0.25 * ly  # Size of vortices
    
    # Define vortex centers (staggered arrangement)
    vortex_x = np.linspace(0.1*lx, 0.9*lx, vortex_count)
    vortex_y_top = np.ones(vortex_count) * 0.75 * ly  # Upper row
    vortex_y_bottom = np.ones(vortex_count) * 0.25 * ly  # Lower row
    
    # Add vortex pairs
    for vortex_idx in range(vortex_count):
        # Upper vortex (clockwise)
        add_vortex(u, v, grid, vortex_x[vortex_idx], vortex_y_top[vortex_idx], 
                   vortex_size, vortex_strength, clockwise=True)
        
        # Lower vortex (counter-clockwise)
        add_vortex(u, v, grid, vortex_x[vortex_idx], vortex_y_bottom[vortex_idx], 
                   vortex_size, vortex_strength, clockwise=False)
    
    # Apply boundary conditions to ensure consistency
    u, v, p = apply_bc(u, v, p, nx, ny)
    
    return u, v, p