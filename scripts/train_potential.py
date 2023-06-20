import tables
import pickle
from functools import partial
from jaxpm.kernels import (
    fftk,
    laplace_kernel,
    longrange_kernel,
)
import h5py
import optax
from typing import List
import numpy as np
from pathlib import Path
import jax
import jax.numpy as jnp
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
from jaxpm.nn import CNN, NeuralSplineFourierFilter

DEFAULT_CAMELS_DATA_DIR = Path(
    "../../projects/rrg-lplevass/data/CAMELS/Sims/IllustrisTNG_DM/"
)
OUTPUT_DIR = Path("figs")

# TODO:
# II) Add potential as function of a
# III) Add Pk validation
# IV) Train on multiple sims
# V) Add training bells and wisthles (scheduler, early stopping, ...)


def get_potential(pos, downsampled_pos, n_mesh, boxsize=25.0):
    pos_norm = pos / boxsize * n_mesh
    downsampled_pos_norm = downsampled_pos / boxsize * n_mesh
    mesh_shape = (n_mesh, n_mesh, n_mesh)
    kvec = fftk(mesh_shape)
    delta_k = jnp.fft.rfftn(cic_paint(jnp.zeros(mesh_shape), pos_norm))
    pot_k = -delta_k * laplace_kernel(kvec) * longrange_kernel(kvec, r_split=0)
    pot_grid = 0.5 * jnp.fft.irfftn(pot_k)
    return pot_grid, cic_read(pot_grid, downsampled_pos_norm)


def get_density(pos, downsampled_pos, n_mesh, boxsize=25.0):
    grid_dens = cic_paint(jnp.zeros([n_mesh, n_mesh, n_mesh]), pos / boxsize * n_mesh)
    dens = cic_read(grid_dens, downsampled_pos / boxsize * n_mesh)
    dens = dens / dens.mean() - 1
    grid_dens = grid_dens / grid_dens.mean() - 1
    return grid_dens, dens




def ConvNet(
    x,
    positions,
    scale_factors,
):
    cnn = CNN(n_features=32)
    return cnn(
        x,
        positions,
        scale_factors,
    )


def get_fourier_potential(model, params, n_mesh, a, pos, downsampled_pos, boxsize=25.0):
    mesh_shape = (n_mesh, n_mesh, n_mesh)
    kvec = fftk(mesh_shape)
    delta_k = jnp.fft.rfftn(cic_paint(jnp.zeros(mesh_shape), pos / boxsize * n_mesh))
    pot_k = -delta_k * laplace_kernel(kvec) * longrange_kernel(kvec, r_split=0)
    kk = jnp.sqrt(sum((ki / jnp.pi) ** 2 for ki in kvec))
    pot_k = pot_k * (1.0 + model.apply(params, kk, jnp.atleast_1d(a)))
    pot_grid = 0.5 * jnp.fft.irfftn(pot_k)
    pred_pot = cic_read(pot_grid, downsampled_pos / boxsize * n_mesh)
    pred_pot = (pred_pot - potential_hr_min) / (potential_hr_max - potential_hr_min)
    return pred_pot


def loss_fn(params):
    corrected_potential = neural_net.apply(
        params,
        grid_input,
        downsampled_pos / boxsize * n_mesh_lr,
        scale_factors,
    )
    predicted_potential = corrected_potential.squeeze() + potential_lr
    return jnp.mean((predicted_potential - potential_hr) ** 2)


def loss_fn_fourier(
    params,
):
    predicted_potential = get_fourier_potential(
        neural_net,
        params,
        n_mesh=n_mesh_lr,
        a=scale_factors,
        pos=pos,
        downsampled_pos=downsampled_pos,
    )
    return jnp.mean((predicted_potential - potential_hr) ** 2)


def read_camels(
    snapshot,
    cv_index: int = 0,
    n_mesh: int = 512,
    data_dir=DEFAULT_CAMELS_DATA_DIR,
):
    snapshot_filename = str(
        data_dir / f"CV_{cv_index}/snap_{str(snapshot).zfill(3)}.hdf5"
    )
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
    downsampling_factor = len(pos) // n_mesh**3
    key = jax.random.PRNGKey(0)
    permuted_indices = jax.random.permutation(key, len(pos))
    selected_indices = permuted_indices[: len(pos) // downsampling_factor]
    downsampled_pos = jnp.take(pos, selected_indices, axis=0)
    downsampled_vel = jnp.take(vel, selected_indices, axis=0)
    return pos, vel, downsampled_pos, downsampled_vel, redshift


def read_camels_snapshots(
    snapshots_list: List[int],
    cv_index: int = 0,
    n_mesh: int = 512,
    data_dir=DEFAULT_CAMELS_DATA_DIR,
):
    downsampled_pos, grid_lr, potential_lr, potential_hr, grid_dens, redshift = [], [], [], [], [], []
    for snapshot in snapshots_list:
        print(f'Reading {snapshot}')
        p, _, dp, _, z = read_camels(snapshot, cv_index, n_mesh, data_dir)
        g_lr, p_lr = get_potential(p, dp, n_mesh_lr)
        _, p_hr = get_potential(p, dp, n_mesh_hr)
        dg, _ = get_density(p, dp, n_mesh_lr)
        downsampled_pos.append(dp)
        grid_lr.append(g_lr)
        potential_lr.append(p_lr)
        potential_hr.append(p_hr)
        grid_dens.append(dg)
        redshift.append(z)
    return (
        jnp.stack(downsampled_pos),
        jnp.stack(grid_lr),
        jnp.stack(potential_lr),
        jnp.stack(potential_hr),
        jnp.stack(grid_dens),
        jnp.stack(redshift),
    )


model = "cnn"
n_mesh_lr = 64 #128
n_mesh_hr = 256 
boxsize = 25.0

snapshots_list = range(34)
downsampled_pos, grid_lr, potential_lr, potential_hr, grid_dens, z = read_camels_snapshots(
    snapshots_list=snapshots_list,
    cv_index=0,
    n_mesh=n_mesh_lr,
)
print(f'downsampled pos = {downsampled_pos.shape}')
scale_factors = 1.0 / (1 + z)
print(f"a = {scale_factors}")

# Rescaling the potential for change in resolution
grid_lr /= n_mesh_hr // n_mesh_lr
potential_lr /= n_mesh_hr // n_mesh_lr


add_densities = True
grid_input = jnp.stack([grid_lr, grid_dens], axis=-1)
print('grid input shape = ', grid_input.shape)
grid_input_min = grid_input.min(axis=(0, 1, 2, 3))
grid_input_max = grid_input.max(axis=(0, 1, 2, 3))
grid_input = (grid_input - grid_input_min) / (grid_input_max - grid_input_min)
potential_hr_max = potential_hr.max()
potential_hr_min = potential_hr.min()
potential_hr = (potential_hr - potential_hr_min) / (potential_hr_max - potential_hr_min)
potential_lr = (potential_lr - potential_hr_min) / (potential_hr_max - potential_hr_min)


for i, a in enumerate(scale_factors):
    plt.scatter(potential_hr[i], potential_lr[i], 0.1, )#c=dens[i])
    plt.colorbar(label="density contrast")
    plt.plot([0, 1], [0, 1], color="red")
    plt.xlabel("True Potential")
    plt.ylabel("Low Resolution Potential")
    plt.savefig(OUTPUT_DIR / f"potential_comparison_a{a:.2f}.png")
    plt.close()

rng = jax.random.PRNGKey(42)

if model == "cnn":
    neural_net = hk.without_apply_rng(hk.transform(ConvNet))
    params = neural_net.init(
        rng,
        grid_input,
        downsampled_pos / boxsize * n_mesh_lr,
        scale_factors,
    )
elif model == "fourier":
    neural_net = hk.without_apply_rng(
        hk.transform(
            lambda x, a: NeuralSplineFourierFilter(n_knots=16, latent_size=32)(x, a)
        ),
    )
    params = neural_net.init(
        rng,
        jnp.zeros([len(scale_factors), n_mesh_lr, n_mesh_lr, n_mesh_lr]),
        scale_factors,
    )
else:
    raise ValueError("model not recognized")

print("LR loss = ", jnp.mean((potential_lr - potential_hr) ** 2))

learning_rate = 1.0e-3
optimizer = optax.adam(learning_rate)


@jax.jit
def update(params, opt_state):
    """Single SGD update step."""
    if model == "cnn":
        loss, grads = jax.value_and_grad(loss_fn)(params)
    elif model == "fourier":
        loss, grads = jax.value_and_grad(loss_fn_fourier)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state


losses = []
opt_state = optimizer.init(params)

n_steps = 50_000
pbar = tqdm(range(n_steps))
for step in pbar:
    l, params, opt_state = update(params, opt_state)
    pbar.set_postfix(
        {
            "Step": step,
            "Loss": l,
        }
    )
    losses.append(l)
plt.plot(np.log10(losses))
plt.savefig(OUTPUT_DIR / f"loss_{model}.png")
plt.close()

# store params
with open('model_params/cnn.pkl', 'wb') as f:
    pickle.dump(params, f)
norm_dict = {
    'potential_hr_min': potential_hr_min,
    'potential_hr_max': potential_hr_max,
    'grid_input_min': grid_input_min,
    'grid_input_max': grid_input_max,
}
with open('model_params/norm_dict.pkl', 'wb') as f:
    pickle.dump(norm_dict, f)

if model == "cnn":
    predicted_potential = neural_net.apply(
        params,
        grid_input,
        downsampled_pos / boxsize * n_mesh_lr,
        scale_factors,
    )
    predicted_potential = predicted_potential.squeeze() + potential_lr
elif model == "fourier":
    predicted_potential = get_fourier_potential(
        neural_net,
        params,
        n_mesh=n_mesh_lr,
        a=scale_factors,
        pos=pos,
        downsampled_pos=downsampled_pos,
    )

for i, a in enumerate(scale_factors):
    plt.scatter(potential_hr[i], predicted_potential[i], 0.1,)# c=dens[i])
    plt.colorbar(label="density contrast")
    plt.plot([0, 1], [0, 1], color="red")
    plt.xlabel("True Potential")
    plt.ylabel("Corrected Potential")
    plt.savefig(OUTPUT_DIR / f"potential_correction_{model}_a{a:.2f}.png")
    plt.close()

# Validation
downsampled_pos_val, grid_lr_val, potential_lr_val, potential_hr_val, grid_dens_val, _ = read_camels_snapshots(
    snapshots_list=snapshots_list,
    cv_index=1,
    n_mesh=n_mesh_lr,
)

grid_lr_val /= n_mesh_hr // n_mesh_lr
potential_lr_val /= n_mesh_hr // n_mesh_lr

grid_input_val = jnp.stack([grid_lr_val, grid_dens_val], axis=-1)
grid_input_val = (grid_input_val - grid_input_min) / (grid_input_max - grid_input_min)
potential_hr_val = (potential_hr_val - potential_hr_min) / (
    potential_hr_max - potential_hr_min
)
potential_lr_val = (potential_lr_val - potential_hr_min) / (
    potential_hr_max - potential_hr_min
)

if model == "cnn":
    predicted_potential_val = neural_net.apply(
        params,
        grid_input_val,
        downsampled_pos_val / boxsize * n_mesh_lr,
        scale_factors,
    )
    predicted_potential_val = predicted_potential_val.squeeze() + potential_lr_val
elif model == "fourier":
    predicted_potential_val = get_fourier_potential(
        neural_net,
        params,
        n_mesh=n_mesh_lr,
        a=scale_factors,
        pos=pos_val,
        downsampled_pos=downsampled_pos_val,
    )

for i, a in enumerate(scale_factors):
    plt.scatter(
        potential_hr_val[i],
        predicted_potential_val[i],
        0.1,
        #c=dens_val[i],
    )
    plt.colorbar(label="density contrast")
    plt.plot([0, 1], [0, 1], color="red")
    plt.xlabel("True Potential")
    plt.ylabel("Corrected Potential")
    plt.savefig(OUTPUT_DIR / f"potential_correction_val_{model}_a{a:.2f}.png")
    plt.close()

for i, a in enumerate(scale_factors):
    plt.scatter(potential_hr_val[i], potential_lr_val[i], 0.1,)# c=dens_val[i])
    plt.colorbar(label="density contrast")
    plt.plot([0, 1], [0, 1], color="red")
    plt.xlabel("True Potential")
    plt.ylabel("Low Resolution Potential")
    plt.savefig(OUTPUT_DIR / f"potential_comparison_val_a{a:.2f}.png")
    plt.close()
