import tables
from functools import partial
from jaxpm.kernels import (
    fftk,
    laplace_kernel,
    longrange_kernel,
)
import h5py
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


DEFAULT_CAMELS_DATA_DIR = Path(
    "../../projects/rrg-lplevass/data/CAMELS/Sims/IllustrisTNG_DM/"
)


def read_camels(
    snapshot,
    cv_index: int = 0,
    n_mesh: int = 32,
    downsample: bool =True,
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
    if downsample:
        key = jax.random.PRNGKey(0)
        downsampling_factor = len(pos) // n_mesh**3
        permuted_indices = jax.random.permutation(key, len(pos))
        selected_indices = permuted_indices[: len(pos) // downsampling_factor]
        pos = jnp.take(pos, selected_indices, axis=0)
        vel = jnp.take(vel, selected_indices, axis=0)
    return pos, vel, redshift, Omega_m, Omega_l


def get_potential(dmo_pos_dm, pos, n_mesh):
    dmo_pos_norm = dmo_pos_dm / 25. * n_mesh
    pos_norm = pos / 25. * n_mesh
    mesh_shape = (n_mesh, n_mesh, n_mesh)
    kvec = fftk(mesh_shape)
    delta_k = jnp.fft.rfftn(cic_paint(jnp.zeros(mesh_shape), dmo_pos_norm))
    pot_k = - delta_k * laplace_kernel(kvec) * longrange_kernel(kvec, r_split=0)
    pot_grid = 0.5*jnp.fft.irfftn(pot_k)
    return pot_grid, cic_read(pot_grid, pos_norm) 


n_mesh_lr = 128 
n_mesh_hr = 512 

dmo_pos_dm, _, _, _, _ = read_camels(
    snapshot=20,
    cv_index=0,
    downsample=False,
)
pos, vel, z, _, _ = read_camels(
    snapshot=20,
    cv_index=0,
    n_mesh=n_mesh_lr,
)
# Also getting the density contrast
grid_lr, potential_lr = get_potential(dmo_pos_dm, pos, n_mesh_lr)
grid_hr, potential_hr = get_potential(dmo_pos_dm, pos, n_mesh_hr)

grid_lr /= 4
potential_lr /= 4

grid_dens = cic_paint(jnp.zeros([n_mesh_lr,n_mesh_lr,n_mesh_lr]), dmo_pos_dm / 25. * n_mesh_lr)
dens = cic_read(grid_dens, pos / 25. * n_mesh_lr)
dens = dens/dens.mean() - 1
grid_dens = grid_dens/grid_dens.mean() - 1

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
        self.linear1 = hk.Linear(n_features)
        self.linear2 = hk.Linear(n_features)
        self.linear3 = hk.Linear(1)

    def __call__(self, x, positions,):
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
        #jax.debug.print('scale = {y}, mean features at pos = {x}, std features at pos = {z}', y=global_features,x=jnp.mean(features_at_pos), z=jnp.std(features_at_pos))
        features_at_pos = self.linear2(features_at_pos)
        features_at_pos = jax.nn.tanh(features_at_pos)
        features_at_pos = self.linear3(features_at_pos)
        #jax.debug.print('scale = {y}, mean features at pos after = {x}, std features at pos after = {z}', y=global_features,x=jnp.mean(features_at_pos), z=jnp.std(features_at_pos))
        return features_at_pos


def ConvNet(x, positions,):
    cnn = CNN(n_features=32)
    return cnn(x, positions,)

add_densities = True 
normalize = True 
if add_densities:
    grid_input = jnp.stack([grid_lr, grid_dens], axis=-1)
else:
    grid_input = grid_dens[...,None]
if normalize:
    grid_input_min = grid_input.min(axis=(0,1,2))
    grid_input_max = grid_input.max(axis=(0,1,2))
    grid_input = (grid_input -  grid_input_min) / (grid_input_max - grid_input_min)
    potential_hr_max = potential_hr.max()
    potential_hr_min = potential_hr.min()
    potential_hr = (potential_hr - potential_hr_min) / (potential_hr_max - potential_hr_min)
    potential_lr = (potential_lr - potential_hr_min) / (potential_hr_max - potential_hr_min)

print('grid input shape = ', grid_input.shape)
print(grid_input.min(), grid_input.max())

plt.scatter(potential_hr, potential_lr, 0.1, c=dens)
plt.colorbar(label='density contrast')
if normalize:
    plt.plot([0,1],[0, 1], color='red')
else:
    plt.plot([-5000,500],[-5000, 500], color='red')
plt.xlabel('True Potential')
plt.ylabel('Low Resolution Potential')
plt.savefig('potential_comparison.png')
plt.close()


conv_net = hk.without_apply_rng(hk.transform(ConvNet))
rng = jax.random.PRNGKey(42)
params = conv_net.init(
    rng,
    #grid_input,#
    grid_input,
    pos / 25. * n_mesh_lr,
)
preds = conv_net.apply(
    params,
    #grid_input,#
    grid_input,
    pos / 25. * n_mesh_lr,
)
print('Just LR loss = ', jnp.mean((potential_lr - potential_hr)**2))

def loss_fn(params):
  corrected_potential = conv_net.apply(
      params, 
      grid_input,
      pos/25.*n_mesh_lr,
    )
  predicted_potential = corrected_potential.squeeze() + potential_lr 
  return jnp.mean((predicted_potential - potential_hr)**2)

import optax
learning_rate=1.e-4
optimizer = optax.adam(learning_rate)

@jax.jit
def update(params, opt_state):
    """Single SGD update step."""
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss, new_params, new_opt_state

losses = []
opt_state = optimizer.init(params)

n_steps = 10_000
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
plt.plot(np.log10(losses))#[600:])
if add_densities:
    plt.savefig('loss_densities.png')
elif normalize:
    plt.savefig('loss_normalize.png')
else:
    plt.savefig('loss.png')
plt.close()

predicted_potential = conv_net.apply(
    params, 
    grid_input,
    pos/25.*n_mesh_lr,
)
plt.scatter(potential_hr, (predicted_potential.squeeze()+potential_lr), 0.1, c=dens) 
plt.colorbar(label='density contrast')
if normalize:
    plt.plot([0,1],[0, 1], color='red')
else:
    plt.plot([-5000,500],[-5000, 500], color='red')
plt.xlabel('True Potential')
plt.ylabel('Corrected Potential')
if add_densities:
    plt.savefig('potential_correction_densities.png')
elif normalize:
    plt.savefig('potential_correction_normalize.png')
else:
    plt.savefig('potential_correction.png')
plt.close()

# Validation
dmo_pos_dm_val, _, _, _, _ = read_camels(
    snapshot=20,
    cv_index=1,
    downsample=False,
)
pos_val, vel_val, z, _, _ = read_camels(
    snapshot=20,
    cv_index=1,
    n_mesh=n_mesh_lr,
)
grid_lr_val, potential_lr_val = get_potential(dmo_pos_dm_val, pos_val, n_mesh_lr)
grid_hr_val, potential_hr_val = get_potential(dmo_pos_dm_val, pos_val, n_mesh_hr)

grid_lr_val /= 4
potential_lr_val /= 4

grid_dens_val = cic_paint(jnp.zeros([n_mesh_lr,n_mesh_lr,n_mesh_lr]), dmo_pos_dm_val / 25. * n_mesh_lr)
dens_val = cic_read(grid_dens_val, pos_val / 25. * n_mesh_lr)
dens_val = dens_val/dens_val.mean() - 1
grid_dens_val = grid_dens_val/grid_dens_val.mean() - 1

if add_densities:
    grid_input_val = jnp.stack([grid_lr_val, grid_dens_val], axis=-1)
else:
    grid_input_val = grid_dens_val[...,None]
if normalize:
    grid_input_val = (grid_input_val -  grid_input_min) / (grid_input_max - grid_input_min)
    potential_hr_val = (potential_hr_val - potential_hr_min) / (potential_hr_max - potential_hr_min)
    potential_lr_val = (potential_lr_val - potential_hr_min) / (potential_hr_max - potential_hr_min)


predicted_potential_val = conv_net.apply(
    params, 
    grid_input_val,
    pos_val/25.*n_mesh_lr,
)
plt.scatter(potential_hr_val, (predicted_potential_val.squeeze()+potential_lr_val), 0.1, c=dens_val) 
plt.colorbar(label='density contrast')
if normalize:
    plt.plot([0,1],[0, 1], color='red')
else:
    plt.plot([-5000,500],[-5000, 500], color='red')
plt.xlabel('True Potential')
plt.ylabel('Corrected Potential')
if add_densities:
    plt.savefig('potential_correction_densities_val.png')
elif normalize:
    plt.savefig('potential_correction_normalize_val.png')
else:
    plt.savefig('potential_correction_val.png')
plt.close()

plt.scatter(potential_hr_val, potential_lr_val, 0.1, c=dens_val)
plt.colorbar(label='density contrast')
if normalize:
    plt.plot([0,1],[0, 1], color='red')
else:
    plt.plot([-5000,500],[-5000, 500], color='red')
plt.xlabel('True Potential')
plt.ylabel('Low Resolution Potential')
plt.savefig('potential_comparison_val.png')
plt.close()

