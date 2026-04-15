"""
Hybrid Vlasov-Maxwell solver implementation in the Darwin approximation with Ohm's law.
Implements the SLICE-3D semi-Lagrangian scheme for V X B rotations and non-spectral spatial advection.
"""
import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.ndimage import map_coordinates
from functools import partial



@partial(jit, static_argnums=(3, 4))
def remap_1d_axis0(f, disp, dx, bc_left='periodic', bc_right='periodic'):
    """
    Applies a 1D semi-Lagrangian shift along the 0th axis. 
    Uses linear interpolation (JAX map_coordinates limit).
    disp is the shift array matching the shape of f.
    """
    n = f.shape[0]
    # Build full coordinate mesh; shift only axis 0 by the semi-Lagrangian displacement.
    coords = jnp.indices(f.shape, dtype=jnp.float32)
    src_coords = coords.at[0].set(coords[0] - disp / dx)
    
    if bc_left == 'periodic':
        mode = 'wrap'
    else: 
        mode = 'nearest'
    
    return map_coordinates(f, src_coords, order=1, mode=mode)


@partial(jit, static_argnums=(2, 3))
def get_gradient(arr, dx, bc_left='periodic', bc_right='periodic'):
    """
    Boundary-aware central difference for 1D arrays or 0th axis of ND arrays.
    """
    if bc_left == 'periodic':
        return (jnp.roll(arr, -1, axis=0) - jnp.roll(arr, 1, axis=0)) / (2 * dx)
    else:
        # jnp.gradient uses 2nd-order central differences for interior 
        # and 1st-order one-sided differences at boundaries.
        return jnp.gradient(arr, dx, axis=0)



def shear_step(f, disp, axis, dx, bc_left='periodic', bc_right='periodic'):
    """
    Executes a 1D shear along 'axis' by shifting memory.
    """
    # Move target axis to front, apply 0-axis remap, then restore axis
    f_front = jnp.moveaxis(f, axis, 0)
    disp_front = jnp.moveaxis(disp, axis, 0)
    
    f_remapped = remap_1d_axis0(f_front, disp_front, dx, bc_left=bc_left, bc_right=bc_right)
    
    return jnp.moveaxis(f_remapped, 0, axis)


class HybridMaxwellSolver:
    def __init__(self, nx, nv, lx, lv, bc_x=('periodic', 'periodic'), bc_v='copy'):
        self.nx = nx
        self.nv = nv
        self.lx = lx
        self.lv = lv
        
        # bc_x should be a tuple (left_bc, right_bc)
        self.bc_x = bc_x
        self.bc_v = bc_v
        
        self.x = jnp.linspace(0, lx, nx, endpoint=False)
        self.v = jnp.linspace(-lv, lv, nv)
        self.dx = self.x[1] - self.x[0]
        self.dv = self.v[1] - self.v[0]
        
        # Slices to store initial boundary values for 'static' BCs
        self.f_left_static = None
        self.f_right_static = None
        self.By_left_static = None
        self.By_right_static = None
        self.Bz_left_static = None
        self.Bz_right_static = None


    def set_static_boundaries(self, f, B_y, B_z):
        """
        Capture initial state for static inflow/outflow enforcement.
        Uses a 2-cell buffer (indices [0:2] / [-2:]) to suppress the
        artificial density notch that forms between a single frozen ghost
        cell and the first free interior cell.
        """
        # Left buffer: cells 0 and 1
        self.f_left_static  = f[0:2]     # shape (2, nv, nv, nv)
        self.By_left_static = B_y[0:2]   # shape (2,)
        self.Bz_left_static = B_z[0:2]   # shape (2,)
        # Right buffer: cells -2 and -1
        self.f_right_static  = f[-2:]    # shape (2, nv, nv, nv)
        self.By_right_static = B_y[-2:]  # shape (2,)
        self.Bz_right_static = B_z[-2:]  # shape (2,)
        


    @partial(jit, static_argnums=(0,))
    def get_moments(self, f):
        """Extracts n, V, and T from the distribution function f."""
        n = jnp.sum(f, axis=(1, 2, 3)) * (self.dv**3)
        n_safe = jnp.maximum(n, 1e-6)
        
        vx_grid = self.v[None, :, None, None]
        vy_grid = self.v[None, None, :, None]
        vz_grid = self.v[None, None, None, :]
        
        Vi_x = jnp.sum(f * vx_grid, axis=(1, 2, 3)) * (self.dv**3) / n_safe
        Vi_y = jnp.sum(f * vy_grid, axis=(1, 2, 3)) * (self.dv**3) / n_safe
        Vi_z = jnp.sum(f * vz_grid, axis=(1, 2, 3)) * (self.dv**3) / n_safe
        
        # Temperature T: P = n*T = m * \int (v - V)^2 f d3v
        # T = (1/n) * \int (vx-Vx)^2 f d3v (simplified to scalar T)
        v_sq_avg = jnp.sum(f * (vx_grid**2 + vy_grid**2 + vz_grid**2), axis=(1, 2, 3)) * (self.dv**3) / n_safe
        V_mag_sq = Vi_x**2 + Vi_y**2 + Vi_z**2
        T = (v_sq_avg - V_mag_sq) / 3.0 # Assuming isotropic T
        
        return n, Vi_x, Vi_y, Vi_z, T



    @partial(jit, static_argnums=(0,))
    def get_fields(self, f, B_x, B_y, B_z):
        """
        Calculates E via Ohm's law and updates J under full 3D B fields.
        Darwin Approximation (Hybrid): Electrons are a massless fluid.
        E + V_i x B = (J x B) / n_e
        => E = -V_i x B + (J x B) / n_e
        """
        # 1. Ion Moments
        n_i, Vi_x, Vi_y, Vi_z, T = self.get_moments(f)
        n_e = jnp.maximum(n_i, 1e-6) # Avoid division by zero
        
        # 2. Current J = curl(B)
        # Because we're in 1D spatially, div(B) = dBx/dx = 0. Therefore B_x is spatially uniform.
        dBy_dx = get_gradient(B_y, self.dx, self.bc_x[0], self.bc_x[1])
        dBz_dx = get_gradient(B_z, self.dx, self.bc_x[0], self.bc_x[1])
        
        J_x = jnp.zeros_like(B_x) # Assuming no external longitudinal current sources
        J_y = -dBz_dx
        J_z = dBy_dx
        
        # 3. Ohm's Law Electric Field
        # V_i x B
        VixB_x = Vi_y * B_z - Vi_z * B_y
        VixB_y = Vi_z * B_x - Vi_x * B_z
        VixB_z = Vi_x * B_y - Vi_y * B_x
        
        # J x B
        JxB_x = J_y * B_z - J_z * B_y
        JxB_y = J_z * B_x - J_x * B_z
        JxB_z = J_x * B_y - J_y * B_x
        
        E_x = -VixB_x + JxB_x / n_e
        E_y = -VixB_y + JxB_y / n_e
        E_z = -VixB_z + JxB_z / n_e
        
        return E_x, E_y, E_z, J_x, J_y, J_z



    @partial(jit, static_argnums=(0,))
    def advance_magnetic_field(self, B_x, B_y, B_z, E_y, E_z, dt):
        """ Faraday's law: partial(B)/partial(t) = -curl(E) """
        dEy_dx = get_gradient(E_y, self.dx, self.bc_x[0], self.bc_x[1])
        dEz_dx = get_gradient(E_z, self.dx, self.bc_x[0], self.bc_x[1])
        
        # B_x is constant in 1D simulations (dEx/dy = dEx/dz = 0)
        B_y_new = B_y + dt * dEz_dx
        B_z_new = B_z - dt * dEy_dx
        
        return B_x, B_y_new, B_z_new



    @partial(jit, static_argnums=(0,))
    def accelerate_v_slice3d(self, f, E_x, E_y, E_z, B_x, B_y, B_z, dt):
        """
        Velocity advance using SLICE-3D shears evaluated through full 3D Lorentz force.
        Applies Strang-splitting dimensional sequences for the 3 velocity bases.
        """
        Ex_4d, Ey_4d, Ez_4d = E_x[:, None, None, None], E_y[:, None, None, None], E_z[:, None, None, None]
        Bx_4d, By_4d, Bz_4d = B_x[:, None, None, None], B_y[:, None, None, None], B_z[:, None, None, None]
        
        vx_grid = self.v[None, :, None, None]
        vy_grid = self.v[None, None, :, None]
        vz_grid = self.v[None, None, None, :]
        
        # q/m = 1 in normalized units (see initialize_maxwell.py normalization header)
        ax = Ex_4d + (vy_grid * Bz_4d - vz_grid * By_4d)
        ay = Ey_4d + (vz_grid * Bx_4d - vx_grid * Bz_4d)
        az = Ez_4d + (vx_grid * By_4d - vy_grid * Bx_4d)
        
        # 1. Sweep X and Y at dt/2
        f = shear_step(f, ax * (dt / 2), axis=1, dx=self.dv, bc_left=self.bc_v, bc_right=self.bc_v)
        f = shear_step(f, ay * (dt / 2), axis=2, dx=self.dv, bc_left=self.bc_v, bc_right=self.bc_v)
        
        # 2. Sweep Z at full dt
        f = shear_step(f, az * dt, axis=3, dx=self.dv, bc_left=self.bc_v, bc_right=self.bc_v)
        
        # 3. Reverse Sweep Y and X at dt/2
        f = shear_step(f, ay * (dt / 2), axis=2, dx=self.dv, bc_left=self.bc_v, bc_right=self.bc_v)
        f = shear_step(f, ax * (dt / 2), axis=1, dx=self.dv, bc_left=self.bc_v, bc_right=self.bc_v)
        
        return f



    @partial(jit, static_argnums=(0,))
    def advect_x_slice3d(self, f, dt):
        """ Spatial Advection (1D-3V) in X. partial(f)/partial(t) + vx partial(f)/partial(x) = 0 """
        vx_grid = self.v[None, :, None, None]
        disp_x = vx_grid * dt
        return shear_step(f, disp_x, axis=0, dx=self.dx, bc_left=self.bc_x[0], bc_right=self.bc_x[1])


    @partial(jit, static_argnums=(0,))
    def apply_bc(self, f, B_y, B_z):
        """
        Explicitly enforces boundary conditions on the fields and distribution function.
        Static BCs use a 2-cell buffer so the CNN (kernel_size=3) never sees an
        artificial step between the frozen ghost zone and the free interior.
        """
        # 1. Left Boundary (-x)
        if self.bc_x[0] == 'copy':
            B_y = B_y.at[0].set(B_y[1])
            B_z = B_z.at[0].set(B_z[1])
            f = f.at[0].set(f[1])
        elif self.bc_x[0] == 'static' and self.f_left_static is not None:
            # 2-cell buffer: clamp cells 0 and 1 to their initial inflow state
            B_y = B_y.at[0:2].set(self.By_left_static)
            B_z = B_z.at[0:2].set(self.Bz_left_static)
            f = f.at[0].set(self.f_left_static[0])
            f = f.at[1].set(self.f_left_static[1])

        # 2. Right Boundary (+x)
        if self.bc_x[1] == 'copy':
            B_y = B_y.at[-1].set(B_y[-2])
            B_z = B_z.at[-1].set(B_z[-2])
            f = f.at[-1].set(f[-2])
        elif self.bc_x[1] == 'static' and self.f_right_static is not None:
            # 2-cell buffer: clamp cells -2 and -1 to their initial downstream state
            B_y = B_y.at[-2:].set(self.By_right_static)
            B_z = B_z.at[-2:].set(self.Bz_right_static)
            f = f.at[-1].set(self.f_right_static[-1])
            f = f.at[-2].set(self.f_right_static[-2])
            
        return f, B_y, B_z



    @partial(jit, static_argnums=(0, 7, 8, 9, 10, 11, 12))
    def strang_step(self, f, B_x, B_y, B_z, dt, 
                    ml_params=None, adv_func=None, acc_func=None, adv_flux=False,
                    adv_clamp=1e-2, acc_clamp=1e-2, boundary_buffer=0):
        """ 
        Full Hybrid Maxwell Strang Split Step with optional Machine Learning corrections. 
        Advc X (dt/2) -> Calc Fields -> Accel V (dt) -> Advc B (dt) -> Advc X (dt/2) 
        """
        # 1. Half Advect X + ML Correction
        f = self.advect_x_slice3d(f, dt=dt/2)
        if adv_func and ml_params:
            if adv_flux:
                # adv_flux: adv_func returns face log-multiplier g_face (shape like f)
                g_face_log = adv_func(f, ml_params['adv'], log_clamp=adv_clamp,
                                      boundary_buffer=boundary_buffer)
                # Upwind face fluxes: use left cell when vx>0 else right cell
                vx_grid = self.v[None, :, None, None]
                f_right = jnp.roll(f, -1, axis=0)
                f_face_upwind = jnp.where(vx_grid > 0, f, f_right)
                flux_face_base = vx_grid * f_face_upwind
                flux_face_corr = flux_face_base * jnp.exp(g_face_log)
                f_after = f - (dt / 2 / self.dx) * (flux_face_corr - jnp.roll(flux_face_corr, 1, axis=0))
                # Conservative non-negativity limiter: project negatives to zero
                # then rescale positives to preserve total mass
                eps = 1e-18
                M0 = jnp.sum(f)
                f_pos = jnp.maximum(f_after, eps)
                M_pos = jnp.sum(f_pos)
                scale = jnp.where(M_pos > 0, M0 / M_pos, 1.0)
                f = f_pos * scale
            else:
                g_log = adv_func(f, ml_params['adv'], log_clamp=adv_clamp, 
                                 boundary_buffer=boundary_buffer)
                f = f * jnp.exp(g_log)
            f, B_y, B_z = self.apply_bc(f, B_y, B_z)
        
        # 2. Calculate Electric Fields via Ohm's Law
        E_x, E_y, E_z, J_x, J_y, J_z = self.get_fields(f, B_x, B_y, B_z)
        
        # 3. Accelerate V (Full Step) + ML Correction
        f = self.accelerate_v_slice3d(f, E_x, E_y, E_z, B_x, B_y, B_z, dt=dt)
        if acc_func and ml_params:
            g_log = acc_func(f, E_x, E_y, E_z, B_x, B_y, B_z, dt, ml_params['acc'],
                             log_clamp=acc_clamp, boundary_buffer=boundary_buffer)
            f = f * jnp.exp(g_log)
            f, B_y, B_z = self.apply_bc(f, B_y, B_z)
            
            # [Physics Consistency] Recalculate fields after ML correction 
            # to ensure Faraday's Law (Adv B) uses corrected ion moments.
            E_x, E_y, E_z, _, _, _ = self.get_fields(f, B_x, B_y, B_z)
        
        # 4. Advance Magnetic Field (Faraday)
        B_x, B_y, B_z = self.advance_magnetic_field(B_x, B_y, B_z, E_y, E_z, dt=dt)
        
        # 5. Half Advect X + ML Correction
        f = self.advect_x_slice3d(f, dt=dt/2)
        if adv_func and ml_params:
            if adv_flux:
                g_face_log = adv_func(f, ml_params['adv'], log_clamp=adv_clamp,
                                      boundary_buffer=boundary_buffer)
                vx_grid = self.v[None, :, None, None]
                f_right = jnp.roll(f, -1, axis=0)
                f_face_upwind = jnp.where(vx_grid > 0, f, f_right)
                flux_face_base = vx_grid * f_face_upwind
                flux_face_corr = flux_face_base * jnp.exp(g_face_log)
                f_after = f - (dt / 2 / self.dx) * (flux_face_corr - jnp.roll(flux_face_corr, 1, axis=0))
                # Conservative non-negativity limiter
                eps = 1e-18
                M0 = jnp.sum(f)
                f_pos = jnp.maximum(f_after, eps)
                M_pos = jnp.sum(f_pos)
                scale = jnp.where(M_pos > 0, M0 / M_pos, 1.0)
                f = f_pos * scale
            else:
                g_log = adv_func(f, ml_params['adv'], log_clamp=adv_clamp, 
                                 boundary_buffer=boundary_buffer)
                f = f * jnp.exp(g_log)
            f, B_y, B_z = self.apply_bc(f, B_y, B_z)
        
        return f, B_x, B_y, B_z, E_x, E_y, E_z
