import time
import jax
import jax.numpy as jnp
import jax_cosmo as jc

from pathlib import Path
from jax.experimental.ode import odeint

# from jaxpm.painting import cic_paint
import numpy as np
from jaxpm.pm import linear_field, lpt, make_ode_fn, make_hamiltonian_ode_fn
from jaxpm.utils import power_spectrum
from jaxpm.painting import cic_paint, cic_read, compensate_cic



def generate_lpt_ics(omega_c, sigma8,):
    # Create a small function to generate the matter power spectrum
    k = jnp.logspace(-4, 1, 128)
    pk = jc.power.linear_matter_power(jc.Planck15(Omega_c=omega_c, sigma8=sigma8), k)
    pk_fn = lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk).reshape(
        x.shape
    )

    # Create initial conditions
    initial_conditions = linear_field(
        mesh_shape, box_size, pk_fn, seed=jax.random.PRNGKey(0)
    )

    # Create particles
    particles = jnp.stack(
        jnp.meshgrid(*[jnp.arange(s) for s in mesh_shape]), axis=-1
    ).reshape([-1, 3])

    cosmo = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)

    # Initial displacement
    dx, p, f = lpt(cosmo, initial_conditions, particles, snapshots[0])
    return [particles + dx, p]

@jax.jit
def run_simulation(
    omega_c,
    sigma8,
    initial_conditions,
):
    cosmo = jc.Planck15(Omega_c=omega_c, sigma8=sigma8)
    # Evolve the simulation forward
    if use_hamiltonian:
        return odeint(
            make_hamiltonian_ode_fn(mesh_shape),
            initial_conditions,
            snapshots,
            cosmo,
            rtol=1e-5,
            atol=1e-5,
        )
    return odeint(
        make_ode_fn(mesh_shape),
        initial_conditions,
        snapshots,
        cosmo,
        rtol=1e-5,
        atol=1e-5,
    )

def get_pk(pos, boxsize, n_mesh):
    k, pk = power_spectrum(
        compensate_cic(
            cic_paint(np.zeros([n_mesh, n_mesh, n_mesh]), pos)
        ),  # /boxsize * n_mesh)),
        boxsize=np.array([boxsize] * 3),
        kmin=np.pi / boxsize,
        dk=2 * np.pi / boxsize,
    )
    return jnp.vstack([k, pk])

if __name__ == '__main__':
    HR = False 
    downsample_lr = False
    use_hamiltonian = False 
    out_dir = Path("/home/cuestalz/scratch/pm_data/")
    omega_c = 0.25
    sigma8 = 0.8
    box_size = [256.0, 256.0, 256.0]
    snapshots = jnp.linspace(0.01, 1.0, 25)
    if HR:
        n_mesh = 256
        mesh_shape = [n_mesh, n_mesh, n_mesh]
        ics = generate_lpt_ics(omega_c, sigma8)
    else: 
        n_mesh = 64
        mesh_shape = [n_mesh, n_mesh, n_mesh]
        if use_hamiltonian:
            pos = jnp.load(out_dir / f'pos_hamiltonian_256.npy')[0]
            vel = jnp.load(out_dir / f'vel_hamiltonian_256.npy')[0]
        else:
            pos = jnp.load(out_dir / f'pos_256.npy')[0]
            vel = jnp.load(out_dir / f'vel_256.npy')[0]
        pos *= n_mesh / 256
        vel *= n_mesh / 256
        if downsample_lr:
            downsampling_factor = len(pos) // n_mesh **3
            key = jax.random.PRNGKey(0)
            permuted_indices = jax.random.permutation(key, len(pos))
            selected_indices = permuted_indices[: len(pos) // downsampling_factor]
            pos = jnp.take(pos, selected_indices, axis=0)
            vel = jnp.take(vel, selected_indices, axis=0)
        ics = [pos, vel]

    t0 = time.time()
    pos, vel = run_simulation(
        omega_c,
        sigma8,
        initial_conditions=ics,
    )
    print(f"It took {time.time() - t0} seconds to get on sim")
    pk = get_pk(pos[-1], box_size[0], n_mesh)
    if use_hamiltonian:
        jnp.save(out_dir / f"pos_hamiltonian_{n_mesh}.npy", pos)
        jnp.save(out_dir / f"vel_hamiltonian_{n_mesh}.npy", vel)
        jnp.save(out_dir / f"scale_factors_hamiltonian_{n_mesh}.npy", snapshots)
        jnp.save(out_dir / f"pk_hamiltonian_{n_mesh}.npy", pk)
    else:
        jnp.save(out_dir / f"pos_{n_mesh}.npy", pos)
        jnp.save(out_dir / f"vel_{n_mesh}.npy", vel)
        jnp.save(out_dir / f"scale_factors_{n_mesh}.npy", snapshots)
        jnp.save(out_dir / f"pk_{n_mesh}.npy", pk)
