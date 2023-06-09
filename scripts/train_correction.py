from functools import partial
from tqdm import tqdm
import logging

logging.getLogger().setLevel(logging.INFO)

import wandb
import numpy as np

import haiku as hk
import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax.experimental.ode import odeint

from jaxpm.nn import NeuralSplineFourierFilter
import optax

from jaxpm.pm import make_neural_ode_fn
from camels_utils import read_camels_cv_set, normalize_by_mesh
from evaluate import eval


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


def get_normalized_camels_pos_vel_z(
    cv_index_list,
    snapshot_list,
    downsampling_factor,
    box_size,
    n_mesh,
):
    pos, vel, z = read_camels_cv_set(
        cv_index_list=cv_index_list,
        snapshot_list=snapshot_list,
        downsampling_factor=downsampling_factor,
    )
    pos, vel = normalize_by_mesh(
        pos,
        vel,
        box_size[0],
        n_mesh,
    )
    return pos, vel, z


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
    if log_experiment:
        wandb.init(
            project="pm-nbody",
            config={
                "n_mesh": n_mesh,
                "n_knots": n_knots,
                "latent_size": latent_size,
                "learning_rate": learning_rate,
                "n_steps": n_steps,
                "velocity_loss": velocity_loss,
            },
        )
    snapshot_list = range(34)
    train_idx = range(21)
    val_idx = range(21,23)
    test_idx = range(23,25)
    # ------ LOAD CAMELS DATA
    planck_cosmology = jc.Planck15(
        Omega_c=0.3 - 0.049, Omega_b=0.049, n_s=0.9624, h=0.671, sigma8=0.8
    )
    target_pos, target_vel, z = get_normalized_camels_pos_vel_z(
        cv_index_list=train_idx,
        snapshot_list=snapshot_list,
        downsampling_factor=downsampling_factor,
        box_size=box_size,
        n_mesh=n_mesh,
    )
    val_pos, val_vel, _ = get_normalized_camels_pos_vel_z(
        cv_index_list=val_idx,
        snapshot_list=snapshot_list,
        downsampling_factor=downsampling_factor,
        box_size=box_size,
        n_mesh=n_mesh,
    )
    test_pos, test_vel, z = get_normalized_camels_pos_vel_z(
        cv_index_list=test_idx,
        snapshot_list=snapshot_list,
        downsampling_factor=downsampling_factor,
        box_size=box_size,
        n_mesh=n_mesh,
    )
    scale_factors = 1 / (1 + jnp.array(z))
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
        if log_experiment:
            wandb.log({'step': step, 'loss': loss, "val_loss": val_loss})
    # ------ EVALUATE MODEL
    eval(
        test_pos=test_pos,
        test_vel=test_vel,
        scale_factors=scale_factors,
        cosmology=planck_cosmology,
        n_mesh=n_mesh,
        box_size=box_size[0],
        model=model,
        params=params,
    )
