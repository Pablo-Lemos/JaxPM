from functools import partial
from tqdm import tqdm
#import wandb
import numpy as np

import haiku as hk
import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.experimental.ode import odeint
from jaxpm.pm import make_ode_fn, make_neural_ode_fn

from jaxpm.nn import NeuralSplineFourierFilter
import optax

import matplotlib.pyplot as plt
from jaxpm.painting import cic_paint, compensate_cic
from jaxpm.utils import power_spectrum

from camels_utils import read_camels_snapshots

# TODO: Add wandb
@partial(jax.jit, static_argnames=["model", "n_mesh"])
def loss_fn(
    params,
    cosmology,
    target_pos,
    target_vel,
    scales,
    box_size,
    n_mesh,
    model,
    velocity_loss=False,
):
    pos_pm, vel_pm = odeint(
        make_neural_ode_fn(model, [n_mesh, n_mesh, n_mesh]),
        [target_pos[0], target_vel[0]],
        scales,
        cosmology,
        params,
        rtol=1e-5,
        atol=1e-5,
    )
    dx = pos_pm - target_pos
    dx  = jnp.where(dx < 100, dx, 0.)

    # TODO: what's going on here?
    #dx = dx - box_size * jnp.round(dx / box_size)
    mse = jnp.sum(dx**2, axis=-1)
    if velocity_loss:
        mse += jnp.sum((vel_pm - target_vel) ** 2, axis=-1)
    return jnp.mean(mse)


def initialize_model(n_mesh, n_knots: int = 16, latent_size: int = 32):
    model = hk.without_apply_rng(
        hk.transform(
            lambda x, a: NeuralSplineFourierFilter(
                n_knots=n_knots, latent_size=latent_size
            )(x, a)
        ),
    )
    rng_seq = hk.PRNGSequence(1)
    params = model.init(
        next(rng_seq), jnp.zeros([n_mesh, n_mesh, n_mesh]), jnp.ones([1])
    )
    return model, params


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
def run_pm_simulation_with_correction(pos, vels, scale_factors, cosmology, n_mesh, model, params,):
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

if __name__ == "__main__":
    # ------ HYPERPARAMETERS
    n_mesh = 32
    box_size = [25.0, 25.0, 25.0]
    downsampling_factor = 100
    n_knots = 16
    latent_size = 32
    learning_rate = 0.01
    n_steps = 500
    velocity_loss = False
    log_experiment = False
    # ------ LOAD CAMELS DATA
    planck_cosmology = jc.Planck15(
        Omega_c=0.3 - 0.049, Omega_b=0.049, n_s=0.9624, h=0.671, sigma8=0.8
    )
    target_pos, target_vel, z = read_camels_snapshots(
        range(34), downsampling_factor=downsampling_factor
    )
    #target_pos = target_pos / box_size[0] * n_mesh
    #target_vel = target_vel / box_size[0] * n_mesh
    scale_factors = 1 / (1 + jnp.array(z))
    # ------ RUN PM SIMULATION
    pos_pm, vel_pm = run_pm_simulation(
        pos=target_pos[0],
        vels=target_vel[0],
        scale_factors=scale_factors,
        cosmology=planck_cosmology,
        n_mesh=n_mesh,
    )
    print(target_pos[-1].max())
    print(pos_pm[-1].max())
    print(target_pos.shape)
    print(pos_pm.shape)
    k, pk_nbody = power_spectrum(
        compensate_cic(cic_paint(jnp.zeros([n_mesh,n_mesh, n_mesh]), target_pos[-1])),
        boxsize=np.array([25.0] * 3),
        kmin=np.pi / 25.0,
        dk=2 * np.pi / 25.0,
    )
    k, pk_pm = power_spectrum(
        compensate_cic(cic_paint(jnp.zeros([n_mesh, n_mesh, n_mesh]), pos_pm[-1])),
        boxsize=np.array([25.0] * 3),
        kmin=np.pi / 25.0,
        dk=2 * np.pi / 25.0,
    )
    plt.loglog(k, pk_nbody, label="N-body")
    plt.loglog(k, pk_pm, label="JaxPM w/o correction")
    plt.legend()
    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.ylabel(r"$P(k)$")
    plt.savefig("pk_before.png")
    plt.close()
    # ------ INITIALIZE SPLINE
    model, params = initialize_model(
        n_mesh=n_mesh,
        n_knots=n_knots,
        latent_size=latent_size,
    )
    # ------ TRAIN MODEL
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    pbar = tqdm(range(n_steps))
    for step in pbar:
        loss, grads = jax.value_and_grad(loss_fn)(
            params,
            planck_cosmology,
            target_pos,
            target_vel,
            scale_factors,
            box_size[0],
            n_mesh,
            model,
        )
        pbar.set_postfix({'Step': step, 'Loss': loss})
        #wandb.log({'step': step, 'loss': loss})
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

    pos_pm_corr, vel_pm_corr = run_pm_simulation_with_correction(
        pos=target_pos[0],
        vels=target_vel[0],
        scale_factors=scale_factors,
        cosmology=planck_cosmology,
        n_mesh=n_mesh,
        model=model,
        params=params,
    )

    k, pk_pm_corr = power_spectrum(
        compensate_cic(cic_paint(jnp.zeros([n_mesh, n_mesh, n_mesh]), pos_pm_corr[-1])),
        boxsize=np.array([25.0] * 3),
        kmin=np.pi / 25.0,
        dk=2 * np.pi / 25.0,
    )
    plt.loglog(k, pk_nbody, label="N-body")
    plt.loglog(k, pk_pm, label="JaxPM w/o correction")
    plt.loglog(k, pk_pm_corr, label="JaxPM w correction")
    plt.legend()
    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.ylabel(r"$P(k)$")
    plt.savefig("pk_after.png")