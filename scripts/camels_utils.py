import tables
import h5py
from typing import List
import numpy as np
from pathlib import Path
import jax
import jax.numpy as jnp
import logging
import readgadget

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
    logging.info(f'Reading CAMELS CV {cv_index}')
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
    cv_index_list: List[int] = [0,1],
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

def normalize_by_mesh(positions, velocities, box_size, n_mesh):
    positions = positions / box_size * n_mesh
    velocities = velocities / box_size * n_mesh
    return positions, velocities