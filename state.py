from dataclasses import dataclass
from typing import Tuple
import jax
import jax.numpy as jnp

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SimulationState:
    """
    JAX-native Pytree representing the full state of the Hybrid Maxwell simulation.
    Includes the distribution function f and the electromagnetic fields E and B.
    """
    f: jnp.ndarray
    B_x: jnp.ndarray
    B_y: jnp.ndarray
    B_z: jnp.ndarray
    E_x: jnp.ndarray
    E_y: jnp.ndarray
    E_z: jnp.ndarray

    def tree_flatten(self):
        children = (self.f, self.B_x, self.B_y, self.B_z, self.E_x, self.E_y, self.E_z)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    @property
    def B(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return (self.B_x, self.B_y, self.B_z)

    @property
    def E(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return (self.E_x, self.E_y, self.E_z)
