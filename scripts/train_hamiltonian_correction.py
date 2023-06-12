import tables
from functools import partial
from jaxpm.kernels import (
    fftk,
    gradient_kernel,
    laplace_kernel,
    longrange_kernel,
    PGD_kernel,
)
import h5py
from typing import List
import numpy as np
from pathlib import Path
import jax
import jax.numpy as jnp
import logging
from jaxpm.painting import cic_paint, cic_read
import matplotlib.pyplot as plt
import readgadget
from jax.experimental.ode import odeint
from jaxpm.utils import power_spectrum
from jaxpm.painting import cic_paint, cic_read, compensate_cic
import jax_cosmo as jc
import haiku as hk
from matplotlib.colors import LogNorm
import optax
from tqdm import tqdm


DEFAULT_CAMELS_DATA_DIR = Path(
    "../../projects/rrg-lplevass/data/CAMELS/Sims/IllustrisTNG_DM/"
)


def read_camels(
    snapshot,
    cv_index: int = 0,
    downsampling_factor: int = 100,
    data_dir=DEFAULT_CAMELS_DATA_DIR,
):
    snapshot_filename = str(
        data_dir / f"CV_{cv_index}/snap_{str(snapshot).zfill(3)}.hdf5"
    )
    print(snapshot_filename)
    header = readgadget.header(snapshot_filename)
    BoxSize = header.boxsize / 1e3  # Mpc/h
    Omega_m = header.omega_m  # value of Omega_m
    Omega_l = header.omega_l  # value of Omega_l
    redshift = header.redshift  # redshift of the snapshot

    ptype = [1]  # dark matter is particle type 1
    ids = np.argsort(
        readgadget.read_block(snapshot_filename, "ID  ", ptype) - 1
    )  # IDs starting from 0
    pos = (
        readgadget.read_block(snapshot_filename, "POS ", ptype)[ids] / 1e3
    )  # positions in Mpc/h
    vel = readgadget.read_block(snapshot_filename, "VEL ", ptype)[
        ids
    ]  # peculiar velocities in km/s
    pos = jnp.array(pos)
    vel = jnp.array(vel / 100 * (1.0 / (1 + redshift)))
    if downsampling_factor is not None:
        downsampling_factor = len(pos) // 32**3
        key = jax.random.PRNGKey(0)
        permuted_indices = jax.random.permutation(key, len(pos))
        selected_indices = permuted_indices[: len(pos) // downsampling_factor]
        pos = jnp.take(pos, selected_indices, axis=0)
        vel = jnp.take(vel, selected_indices, axis=0)
    return pos, vel, redshift, Omega_m, Omega_l


def read_camels_snapshots(
    snapshot_list,
    cv_index: int = 0,
    downsampling_factor=500,
    data_dir=DEFAULT_CAMELS_DATA_DIR,
):
    print(f"Reading CAMELS CV {cv_index}")
    pos, vel, redshift = [], [], []
    for s in snapshot_list:
        p, v, z, _, _ = read_camels(
            snapshot=s,
            cv_index=cv_index,
            downsampling_factor=downsampling_factor,
            data_dir=data_dir,
        )
        pos.append(p)
        vel.append(v)
        redshift.append(z)
    return jnp.array(pos), jnp.array(vel), jnp.array(redshift)


def read_camels_cv_set(
    cv_index_list: List[int] = [0, 1],
    snapshot_list=range(34),
    downsampling_factor: int = 500,
    data_dir=DEFAULT_CAMELS_DATA_DIR,
):
    pos, vel = [], []
    for cv_index in cv_index_list:
        p, v, redshift = read_camels_snapshots(
            snapshot_list=snapshot_list,
            cv_index=cv_index,
            downsampling_factor=downsampling_factor,
            data_dir=data_dir,
        )
        pos.append(p)
        vel.append(v)
    return jnp.array(pos), jnp.array(vel), redshift


pos, vel, z = read_camels_cv_set(
    cv_index_list=[
        0,
    ],
    snapshot_list=range(34),
)


class CNN(hk.Module):
    def __init__(
        self,
        n_features: int,
    ):
        super().__init__(name="CNN")
        self.conv1 = hk.Conv3D(
            output_channels=n_features,
            kernel_shape=(3, 3, 3),
            padding="SAME",
        )
        self.conv2 = hk.Conv3D(
            output_channels=n_features,
            kernel_shape=(3, 3, 3),
            padding="SAME",
        )
        self.conv3 = hk.Conv3D(
            output_channels=n_features,
            kernel_shape=(3, 3, 3),
            padding="SAME",
        )
        # Dense layers
        self.flatten = hk.Flatten()
        self.linear1 = hk.Linear(n_features)
        self.linear2 = hk.Linear(n_features)
        self.linear3 = hk.Linear(1)

    def __call__(self, x, positions, global_features):
        #jax.debug.print('scale = {y}, mean x = {x}, std x = {z}', y=global_features,x=jnp.mean(x), z=jnp.std(x))
        x = self.conv1(x)
        x = jax.nn.tanh(x)
        x = self.conv2(x)
        x = jax.nn.tanh(x)
        x = self.conv3(x)
        x = jax.nn.tanh(x)
        features = self.linear1(x)
        #jax.debug.print('scale = {y}, mean features = {x}, std features = {z}', y=global_features,x=jnp.mean(features), z=jnp.std(features))
        vmap_features_cic = jax.vmap(
            cic_read,
            in_axes=(-1, None),
        )
        features_at_pos = vmap_features_cic(features, positions).swapaxes(-2, -1)
        #jax.debug.print('scale = {y}, mean features at pos = {x}, std features at pos = {z}', y=global_features,x=jnp.mean(features_at_pos), z=jnp.std(features_at_pos))
        global_features = jnp.atleast_1d(global_features)
        broadcast_globals = jnp.broadcast_to(
            global_features[:, None],
            (len(positions), len(global_features)),
        )
        features_at_pos = jnp.concatenate([features_at_pos, broadcast_globals], axis=-1)
        #jax.debug.print('scale = {y}, mean features at pos = {x}, std features at pos = {z}', y=global_features,x=jnp.mean(features_at_pos), z=jnp.std(features_at_pos))
        features_at_pos = self.linear2(features_at_pos)
        features_at_pos = jax.nn.tanh(features_at_pos)
        features_at_pos = self.linear3(features_at_pos)
        #jax.debug.print('scale = {y}, mean features at pos after = {x}, std features at pos after = {z}', y=global_features,x=jnp.mean(features_at_pos), z=jnp.std(features_at_pos))
        return features_at_pos


def ConvNet(x, positions, global_features):
    cnn = CNN(n_features=32)
    return cnn(x, positions, global_features)


def get_hamiltonian(mesh_shape, model=None):
    def hamiltonian_from_state_fn(position, momentum, scale_factor, params):
        #jax.debug.print('*********************')
        kvec = fftk(mesh_shape)
        delta = cic_paint(jnp.zeros(mesh_shape), position)
        delta_k = jnp.fft.rfftn(delta)
        pot_k = -delta_k * laplace_kernel(kvec) * longrange_kernel(kvec, r_split=0)
        pot = jnp.fft.irfftn(pot_k)
        grav_potential = 0.5 * (1 + cic_read(pot, position))
        if model is not None:
            corr_potential = model.apply(
                params, 0.5 * (1 + pot[..., None]), position, scale_factor
            )
            #jax.debug.print('pot ratio = {x}', x=corr_potential/grav_potential)
            #jax.debug.print('scale = {y}, mean pot = {x}, mean corr = {z}', y=scale_factor,x=jnp.mean(grav_potential), z=jnp.mean(corr_potential))
            #jax.debug.print('scale = {y}, std pot = {x}, std corr = {z}', y=scale_factor,x=jnp.std(grav_potential), z=jnp.std(corr_potential))
            grav_potential += corr_potential[..., 0]
        # Computes momentum
        momentum_norm = jnp.linalg.norm(momentum, axis=1)
        kinetic_energy = 0.5 * momentum_norm**2
        hamiltonian = grav_potential + kinetic_energy
        return hamiltonian.sum()

    return hamiltonian_from_state_fn


n_mesh = 32
mesh_shape = (n_mesh, n_mesh, n_mesh)


def get_nbody_ode(hamiltonian_gradients_fn):
    @jax.jit
    def gnn_nbody_ode(state, a, cosmo, params):
        """
        state is a tuple (position, velocities)
        """
        pos, vel = state

        # Take the derivatives against position and momentum of the hamiltonian of the system
        dh_dposition, dh_dmomentum = hamiltonian_gradients_fn(pos, vel, a, params)
        # Hamilton equations
        dpos_da = 1.0 / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * dh_dmomentum
        dvel_da = (
            -1.5
            * cosmo.Omega_m
            / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a)))
            * dh_dposition
        )
        return dpos_da, dvel_da

    return gnn_nbody_ode


pos_normed = pos[0] / 25.0 * n_mesh
vel_normed = vel[0] / 25.0 * n_mesh


cosmo = jc.Planck15(Omega_c=0.3 - 0.049, Omega_b=0.049, n_s=0.9624, h=0.671, sigma8=0.8)
scales = 1.0 / (1 + z)

# Run with no correction
h_fn = get_hamiltonian(
    mesh_shape=mesh_shape,
)
hamiltonian_gradients_fn = jax.grad(h_fn, argnums=[0, 1])
gnn_nbody_ode = get_nbody_ode(hamiltonian_gradients_fn)
res = odeint(
    gnn_nbody_ode,
    [pos_normed[0], vel_normed[0]],
    jnp.array(scales),
    cosmo,
    None,
    rtol=1e-4,
    atol=1e-4,
)

k, pk_ref = power_spectrum(
    compensate_cic(cic_paint(jnp.zeros(mesh_shape), pos_normed[-1])),
    boxsize=np.array([25.0] * 3),
    kmin=np.pi / 25.0,
    dk=2 * np.pi / 25.0,
)

k, pk_i = power_spectrum(
    compensate_cic(cic_paint(jnp.zeros(mesh_shape), res[0][-1])),
    boxsize=np.array([25.0] * 3),
    kmin=np.pi / 25.0,
    dk=2 * np.pi / 25.0,
)

plt.loglog(k, pk_ref, label="N-body")
plt.loglog(k, pk_i, label="JaxPM without correction")
plt.legend()
plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
plt.ylabel(r"$P(k)$")
plt.savefig("pk_hamiltonian.png")
plt.close()

# Initialize cnn
delta = cic_paint(jnp.zeros(mesh_shape), pos_normed[-1])
conv_net = hk.without_apply_rng(hk.transform(ConvNet))
rng = jax.random.PRNGKey(42)

params = conv_net.init(
    rng,
    delta[..., None],  # Add channel dimension
    pos_normed[-1],
    jnp.array([1.0]),  # Globals = scale factor
)
preds = conv_net.apply(
    params,
    delta[..., None],  # Add channel dimension
    pos_normed[-1],
    jnp.array([1.0]),
)

h_fn = get_hamiltonian(mesh_shape=mesh_shape, model=conv_net)
hamiltonian_gradients_fn = jax.grad(h_fn, argnums=[0, 1])

gnn_nbody_ode = get_nbody_ode(hamiltonian_gradients_fn)

pos_pm_corr, vel_pm_corr = odeint(
    gnn_nbody_ode,
    [pos_normed[0], vel_normed[0]],
    jnp.array(scales),
    cosmo,
    params,
    rtol=1e-4,
    atol=1e-4,
)

k, pk_corr = power_spectrum(
    compensate_cic(cic_paint(jnp.zeros(mesh_shape), pos_pm_corr[-1])),
    boxsize=np.array([25.0] * 3),
    kmin=np.pi / 25.0,
    dk=2 * np.pi / 25.0,
)

plt.loglog(k, pk_ref, label="N-body")
plt.loglog(k, pk_i, label="JaxPM without correction")
plt.loglog(k, pk_corr, label="JaxPM with correction")
plt.legend()
plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
plt.ylabel(r"$P(k)$")
plt.savefig("pk_hamiltonian_corr_Before_train.png")
plt.close()


@partial(jax.jit, static_argnames=["model", "n_mesh"])
def loss_fn(
    params,
    cosmology,
    pos,
    vel,
    scales,
    n_mesh,
):
    pos_pm, vel_pm = odeint(
        gnn_nbody_ode,
        [pos_normed[0], vel_normed[0]],
        jnp.array(scales),
        cosmology,
        params,
        rtol=1e-4,
        atol=1e-4,
    )
    pos_pm %= n_mesh
    dx = pos_pm - pos
    dx = dx - n_mesh * jnp.round(dx / n_mesh)
    sim_mse = jnp.sum(dx**2, axis=-1)
    return jnp.mean(sim_mse)


learning_rate = 0.01
n_steps = 100
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)
pbar = tqdm(range(n_steps))
for step in pbar:
    loss, grads = jax.value_and_grad(loss_fn)(
        params,
        cosmo,
        pos_normed,
        vel_normed,
        scales,
        n_mesh,
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    pbar.set_postfix(
        {
            "Step": step,
            "Loss": loss,
        }
    )

pos_pm_corr, vel_pm_corr = odeint(
    gnn_nbody_ode,
    [pos_normed[0], vel_normed[0]],
    jnp.array(scales),
    cosmo,
    params,
    rtol=1e-4,
    atol=1e-4,
)

k, pk_corr = power_spectrum(
    compensate_cic(cic_paint(jnp.zeros(mesh_shape), pos_pm_corr[-1])),
    boxsize=np.array([25.0] * 3),
    kmin=np.pi / 25.0,
    dk=2 * np.pi / 25.0,
)

plt.loglog(k, pk_ref, label="N-body")
plt.loglog(k, pk_i, label="JaxPM without correction")
plt.loglog(k, pk_corr, label="JaxPM with correction")
plt.legend()
plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
plt.ylabel(r"$P(k)$")
plt.savefig("pk_hamiltonian_corr.png")
