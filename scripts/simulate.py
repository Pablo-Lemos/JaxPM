
from jax.experimental.ode import odeint
import jax
import jax.numpy as jnp
from functools import partial
from jaxpm.pm import make_ode_fn, make_neural_ode_fn


@partial(jax.jit, static_argnames=["n_mesh"])
def run_pm_simulation(pos, vels, scale_factors, cosmology, n_mesh):
    mesh_shape = [n_mesh, n_mesh, n_mesh]
    return odeint(
        make_ode_fn(mesh_shape),
        [pos, vels],
        jnp.array(scale_factors),
        cosmology,
        rtol=1e-5,
        atol=1e-5,
    )


@partial(jax.jit, static_argnames=["n_mesh", "model"])
def run_pm_simulation_with_correction(
    pos,
    vels,
    scale_factors,
    cosmology,
    n_mesh,
    model,
    params,
):
    mesh_shape = [n_mesh, n_mesh, n_mesh]
    return odeint(
        make_neural_ode_fn(model, mesh_shape),
        [pos, vels],
        jnp.array(scale_factors),
        cosmology,
        params,
        rtol=1e-5,
        atol=1e-5,
    )

