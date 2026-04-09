"""
Spectral and TVD modelling approaches for solving the 1D-3V Vlasov-Poisson equations.
Modular, OOP-based solver structure.
"""
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial

class VlasovPoissonSolver:
    def __init__(self, nx, nv, lx, lv):
        self.nx = nx
        self.nv = nv
        self.lx = lx
        self.lv = lv
        
        # Mesh setup
        self.x = jnp.linspace(0, lx, nx, endpoint=False)
        self.v = jnp.linspace(-lv, lv, nv)
        self.dx = self.x[1] - self.x[0]
        self.dv = self.v[1] - self.v[0]
        
        # Fourier frequencies for spectral derivatives/Poisson
        self.kx = 2 * jnp.pi * jnp.fft.fftfreq(nx, d=self.dx)
        self.kv = 2 * jnp.pi * jnp.fft.fftfreq(nv, d=self.dv)

    @partial(jit, static_argnums=(0,))
    def get_electric_field(self, f):
        """Poisson Solver: div(E) = rho - 1. Uses spectral method with periodic BCs."""
        # Integrate over 3V to get density rho(x)
        rho = jnp.sum(f, axis=(1, 2, 3)) * (self.dv**3)
        rho_zero_mean = rho - jnp.mean(rho)
        rho_hat = jnp.fft.fft(rho_zero_mean)
        
        # Handle k=0 mode to avoid division by zero
        E_hat = -1j * rho_hat / jnp.where(self.kx == 0, 1.0, self.kx)
        E_hat = E_hat.at[0].set(0.0) 
        
        return jnp.real(jnp.fft.ifft(E_hat))

    def strang_step(self, f, dt):
        """A single Strang-split time step."""
        raise NotImplementedError


class FourierSolver(VlasovPoissonSolver):
    """Original Spectral Fourier-based Solver"""
    
    @partial(jit, static_argnums=(0,))
    def advect_x(self, f, dt):
        """Solves df/dt + vx*df/dx = 0 in Fourier space."""
        # f shape: (nx, nv, nv, nv)
        f_hat = jnp.fft.fft(f, axis=0)
        phase = jnp.exp(-1j * self.kx[:, None, None, None] * self.v[None, :, None, None] * dt)
        return jnp.real(jnp.fft.ifft(f_hat * phase, axis=0))

    @partial(jit, static_argnums=(0,))
    def accelerate_v(self, f, Ex, dt):
        """Solves df/dt + Ex*df/dvx = 0 in Fourier space."""
        # Transforming only the vx dimension (axis 1)
        f_hat = jnp.fft.fft(f, axis=1)
        phase = jnp.exp(-1j * self.kv[None, :, None, None] * Ex[:, None, None, None] * dt)
        return jnp.real(jnp.fft.ifft(f_hat * phase, axis=1))

    @partial(jit, static_argnums=(0,))
    def strang_step(self, f, dt):
        # 1. Half-step Advection X
        f = self.advect_x(f, dt/2)
        # 2. Full-step Acceleration V
        Ex = self.get_electric_field(f)
        f = self.accelerate_v(f, Ex, dt)
        # 3. Half-step Advection X
        f = self.advect_x(f, dt/2)
        return f


@jit
def minmod(r):
    return jnp.maximum(0.0, jnp.minimum(1.0, r))


class TVDSolver(VlasovPoissonSolver):
    """Finite Volume TVD Solver using Minmod Limiter"""

    @partial(jit, static_argnums=(0,))
    def advect_x(self, f, dt):
        """Solve df/dt + vx df/dx = 0 using TVD with Minmod limiter."""
        a = self.v[None, :, None, None]
        nu = jnp.abs(a) * dt / self.dx
        
        f_p1 = jnp.roll(f, -1, axis=0)
        f_p2 = jnp.roll(f, -2, axis=0)
        f_m1 = jnp.roll(f, 1, axis=0)
        
        du_plus = f_p1 - f        # Δu_{i+1/2}
        du_minus = f - f_m1       # Δu_{i-1/2}
        du_plus2 = f_p2 - f_p1    # Δu_{i+3/2}
        
        # r_{i+1/2}
        r = jnp.where(a > 0, du_minus, du_plus2) / (du_plus + 1e-12)
        phi = minmod(r)
        
        # F_{i+1/2}
        F_up = jnp.where(a > 0, a * f, a * f_p1)
        F_high = F_up + 0.5 * jnp.abs(a) * (1.0 - nu) * phi * du_plus
        
        # F_{i-1/2}
        F_high_minus = jnp.roll(F_high, 1, axis=0)
        
        return f - (dt / self.dx) * (F_high - F_high_minus)

    @partial(jit, static_argnums=(0,))
    def accelerate_v(self, f, Ex, dt):
        """Solve df/dt + Ex df/dvx = 0 using TVD with Minmod limiter."""
        a = Ex[:, None, None, None]
        nu = jnp.abs(a) * dt / self.dv
        
        f_p1 = jnp.roll(f, -1, axis=1)
        f_p2 = jnp.roll(f, -2, axis=1)
        f_m1 = jnp.roll(f, 1, axis=1)
        
        du_plus = f_p1 - f
        du_minus = f - f_m1
        du_plus2 = f_p2 - f_p1
        
        r = jnp.where(a > 0, du_minus, du_plus2) / (du_plus + 1e-12)
        phi = minmod(r)
        
        F_up = jnp.where(a > 0, a * f, a * f_p1)
        F_high = F_up + 0.5 * jnp.abs(a) * (1.0 - nu) * phi * du_plus
        
        F_high_minus = jnp.roll(F_high, 1, axis=1)
        
        return f - (dt / self.dv) * (F_high - F_high_minus)

    @partial(jit, static_argnums=(0,))
    def strang_step(self, f, dt):
        # 1. Half-step Advection X
        f = self.advect_x(f, dt/2)
        # 2. Full-step Acceleration V
        Ex = self.get_electric_field(f)
        f = self.accelerate_v(f, Ex, dt)
        # 3. Half-step Advection X
        f = self.advect_x(f, dt/2)
        return f
