import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

from .state import SimulationState
from . import field_solver
from . import boundary

@partial(jit, static_argnums=(3, 4))
def remap_1d_axis0(f, disp, dx, bc_left='periodic', bc_right='periodic'):
    """
    Applies a 1D semi-Lagrangian shift along the 0th axis. 
    Uses linear interpolation.
    """
    n = f.shape[0]
    coords = jnp.indices(f.shape, dtype=jnp.float32)
    src_idx_0 = coords[0] - disp / dx
    
    if bc_left == 'periodic':
        src_idx_0 = jnp.mod(src_idx_0, n)
        mode = 'wrap'
    else: 
        mode = 'nearest'
    
    src_coords = coords.at[0].set(src_idx_0)
    from jax.scipy.ndimage import map_coordinates
    return map_coordinates(f, src_coords, order=1, mode=mode)


def shear_step(f, disp, axis, dx, bc_left='periodic', bc_right='periodic'):
    """
    Executes a 1D shear along 'axis' by shifting memory.
    """
    f_front = jnp.moveaxis(f, axis, 0)
    disp_front = jnp.moveaxis(disp, axis, 0)
    f_remapped = remap_1d_axis0(f_front, disp_front, dx, bc_left=bc_left, bc_right=bc_right)
    return jnp.moveaxis(f_remapped, 0, axis)


class HybridMaxwellSolver:
    """
    Orchestrator for the Hybrid Maxwell-Vlasov simulation.
    Coordinates between Vlasov kernels, Field solvers, and Boundary enforcement.
    """
    def __init__(self, nx, nv, lx, lv, bc_x=('periodic', 'periodic'), bc_v='copy', qi=1.0, mi=1.0):
        self.nx = nx
        self.nv = nv
        self.lx = lx
        self.lv = lv
        
        # Ghost Cells for Real Space X
        self.n_ghost = 2
        self.nx_total = nx + 2 * self.n_ghost
        
        # Physical constants
        self.qi = qi
        self.mi = mi
        
        self.bc_x = bc_x
        self.bc_v = bc_v
        
        self.dx = lx / nx
        self.x = jnp.arange(-self.n_ghost, nx + self.n_ghost) * self.dx
        self.v = jnp.linspace(-lv, lv, nv)
        self.dv = self.v[1] - self.v[0]
        
        # Boundary data
        self.static_data = None

    def get_physical(self, arr):
        """ Slices out the ghost cells. """
        return arr[self.n_ghost : -self.n_ghost]

    def set_static_boundaries(self, f, B_y, B_z, E_x, E_y, E_z):
        """ Capture initial boundary layers for static BC enforcement. """
        self.static_data = {
            'f_left': f[0:2],
            'f_right': f[-2:],
            'By_left': B_y[0:2],
            'By_right': B_y[-2:],
            'Bz_left': B_z[0:2],
            'Bz_right': B_z[-2:],
            'Ex_left': E_x[0:2],
            'Ex_right': E_x[-2:],
            'Ey_left': E_y[0:2],
            'Ey_right': E_y[-2:],
            'Ez_left': E_z[0:2],
            'Ez_right': E_z[-2:]
        }

    @partial(jit, static_argnums=(0,))
    def get_moments(self, f):
        """ Delegated to field_solver """
        return field_solver.get_moments(f, self.v, self.dv)

    @partial(jit, static_argnums=(0,))
    def get_fields(self, f, B_x, B_y, B_z):
        """ Delegated to field_solver """
        return field_solver.get_fields(f, B_x, B_y, B_z, self.dx, self.bc_x, self.v, self.dv)

    @partial(jit, static_argnums=(0,))
    def get_fields_state(self, state: SimulationState):
        """ Helper for run loop diagnostics """
        return self.get_fields(state.f, state.B_x, state.B_y, state.B_z)

    @partial(jit, static_argnums=(0,))
    def apply_bc_state(self, state: SimulationState) -> SimulationState:
        """ Delegated to boundary module """
        return boundary.synchronize_state_ghosts(state, self.bc_x, self.static_data)

    @partial(jit, static_argnums=(0,))
    def accelerate_v_slice3d(self, f, E_x, E_y, E_z, B_x, B_y, B_z, dt):
        """ Core Vlasov Velocity Advection Kernel """
        Ex_4d, Ey_4d, Ez_4d = E_x[:, None, None, None], E_y[:, None, None, None], E_z[:, None, None, None]
        Bx_4d, By_4d, Bz_4d = B_x[:, None, None, None], B_y[:, None, None, None], B_z[:, None, None, None]
        
        vx_grid = self.v[None, :, None, None]
        vy_grid = self.v[None, None, :, None]
        vz_grid = self.v[None, None, None, :]
        
        ax = Ex_4d + (vy_grid * Bz_4d - vz_grid * By_4d)
        ay = Ey_4d + (vz_grid * Bx_4d - vx_grid * Bz_4d)
        az = Ez_4d + (vx_grid * By_4d - vy_grid * Bx_4d)
        
        f = shear_step(f, ax * (dt / 2), axis=1, dx=self.dv, bc_left=self.bc_v, bc_right=self.bc_v)
        f = shear_step(f, ay * (dt / 2), axis=2, dx=self.dv, bc_left=self.bc_v, bc_right=self.bc_v)
        f = shear_step(f, az * dt, axis=3, dx=self.dv, bc_left=self.bc_v, bc_right=self.bc_v)
        f = shear_step(f, ay * (dt / 2), axis=2, dx=self.dv, bc_left=self.bc_v, bc_right=self.bc_v)
        f = shear_step(f, ax * (dt / 2), axis=1, dx=self.dv, bc_left=self.bc_v, bc_right=self.bc_v)
        return f

    @partial(jit, static_argnums=(0,))
    def advect_x_slice3d(self, f, dt):
        """ Core Vlasov Spatial Advection Kernel """
        vx_grid = self.v[None, :, None, None]
        disp_x = vx_grid * dt
        return shear_step(f, disp_x, axis=0, dx=self.dx, bc_left=self.bc_x[0], bc_right=self.bc_x[1])

    @partial(jit, static_argnums=(0,))
    def strang_step(self, state: SimulationState, dt: float) -> SimulationState:
        """ 
        Orchestrates the full Split-Step sequence using delegated modules.
        """
        f, B_x, B_y, B_z = state.f, state.B_x, state.B_y, state.B_z
        
        # 1. Half Advect X
        f = self.advect_x_slice3d(f, dt=dt/2)
        
        # 2. Field Calculation (Functional)
        E_x, E_y, E_z, J_x, J_y, J_z = self.get_fields(f, B_x, B_y, B_z)
        
        # 3. Accelerate V (Full Step)
        f = self.accelerate_v_slice3d(f, E_x, E_y, E_z, B_x, B_y, B_z, dt=dt)
        
        # 4. Faraday Advance (Functional Delegation)
        B_x, B_y, B_z = field_solver.advance_magnetic_field(B_x, B_y, B_z, E_y, E_z, self.dx, self.bc_x, dt)
        
        # 5. Half Advect X
        f = self.advect_x_slice3d(f, dt=dt/2)
        
        # Consistency field update
        E_x, E_y, E_z, J_x, J_y, J_z = self.get_fields(f, B_x, B_y, B_z)
        
        return SimulationState(f, B_x, B_y, B_z, E_x, E_y, E_z)
