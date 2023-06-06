from functools import partial
from tqdm import tqdm
import logging
logging.getLogger().setLevel(logging.INFO)

# import wandb
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

from camels_utils import read_camels_cv_set, normalize_by_mesh


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
    mse = 0
    for pos, vel in zip(target_pos, target_vel):
        pos_pm, vel_pm = odeint(
            make_neural_ode_fn(model, [n_mesh, n_mesh, n_mesh]),
            [pos[0], vel[0]],
            scales,
            cosmology,
            params,
            rtol=1e-5,
            atol=1e-5,
        )
        pos_pm %= n_mesh
        dx = pos_pm - pos 
        dx = dx - n_mesh * jnp.round(dx / n_mesh)
        sim_mse = jnp.sum(dx**2, axis=-1)
        if velocity_loss:
            sim_mse += jnp.sum((vel_pm - vel) ** 2, axis=-1)
        mse += jnp.mean(sim_mse)
    return mse / len(target_pos)


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


if __name__ == "__main__":
    # ------ HYPERPARAMETERS
    n_mesh = 32
    box_size = [25.0, 25.0, 25.0]
    downsampling_factor = 10
    n_knots = 16
    latent_size = 32
    learning_rate = 0.01
    n_steps = 100
    velocity_loss = False
    log_experiment = False
    snapshot_list = range(32)
    train_idx = [0,1,2,3]
    val_idx = [4,]
    test_idx = [5,]
    # ------ LOAD CAMELS DATA
    planck_cosmology = jc.Planck15(
        Omega_c=0.3 - 0.049, Omega_b=0.049, n_s=0.9624, h=0.671, sigma8=0.8
    )
    target_pos, target_vel, z = read_camels_cv_set(
        cv_index_list=train_idx, 
        snapshot_list=snapshot_list,
        downsampling_factor=downsampling_factor
    )
    target_pos, target_vel = normalize_by_mesh(
        target_pos, target_vel, box_size[0], n_mesh,
    )
    scale_factors = 1 / (1 + jnp.array(z))
    val_pos, val_vel, _ = read_camels_cv_set(
        cv_index_list=val_idx, 
        snapshot_list=snapshot_list,
        downsampling_factor=downsampling_factor
    )
    val_pos, val_vel = normalize_by_mesh(
        val_pos, val_vel, box_size[0], n_mesh,
    )
    test_pos, test_vel, _ = read_camels_cv_set(
        cv_index_list=test_idx, 
        snapshot_list=snapshot_list,
        downsampling_factor=downsampling_factor
    )
    test_pos, test_vel = normalize_by_mesh(
        test_pos, test_vel, box_size[0], n_mesh,
    )
    # ------ RUN PM SIMULATION
    plot_test_idx = 0
    pos_pm, vel_pm = run_pm_simulation(
        pos=test_pos[plot_test_idx][0],
        vels=test_vel[plot_test_idx][0],
        scale_factors=scale_factors,
        cosmology=planck_cosmology,
        n_mesh=n_mesh,
    )

    k, pk_nbody = power_spectrum(
        compensate_cic(cic_paint(jnp.zeros([n_mesh, n_mesh, n_mesh]), test_pos[plot_test_idx][-1])),
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
    edges = plt.hist(
        test_vel[plot_test_idx][-1][:, 0],
        bins=100,
        alpha=0.5,
        label="N-body",
        log=True,
    )
    plt.hist(
        vel_pm[-1][:, 0],
        bins=edges[1],
        alpha=0.5,
        log=True,
        label="JaxPM w/o correction",
    )
    plt.legend()
    plt.xlabel("v")
    plt.ylabel("PDF")
    plt.savefig("vel_hist_before.png")
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
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        val_loss = loss_fn(
            params,
            planck_cosmology,
            val_pos,
            val_vel,
            scale_factors,
            box_size[0],
            n_mesh,
            model,
        )
        pbar.set_postfix({"Step": step, "Loss": loss, "Val Loss": val_loss})
        # wandb.log({'step': step, 'loss': loss})

    pos_pm_corr, vel_pm_corr = run_pm_simulation_with_correction(
        pos=test_pos[plot_test_idx][0],
        vels=test_vel[plot_test_idx][0],
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
    plt.close()

    edges = plt.hist(
        test_vel[plot_test_idx][-1][:, 0],
        bins=100,
        alpha=0.5,
        label="N-body",
        log=True,
    )
    plt.hist(
        vel_pm[-1][:, 0],
        bins=edges[1],
        alpha=0.5,
        label="JaxPM w/o correction",
        log=True,
    )
    plt.hist(
        vel_pm_corr[-1][:, 0],
        bins=edges[1],
        alpha=0.5,
        label="JaxPM w correction",
        log=True,
    )
    plt.legend()
    plt.xlabel("v")
    plt.ylabel("PDF")
    plt.savefig("vel_hist_after.png")
    plt.close()
