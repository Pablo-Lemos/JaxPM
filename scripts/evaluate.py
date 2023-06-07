import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np
from jaxpm.painting import compensate_cic, cic_paint
from jaxpm.utils import power_spectrum

from simulate import run_pm_simulation, run_pm_simulation_with_correction


def get_power_spectrum(pos, n_mesh, box_size):
    return power_spectrum(
        compensate_cic(cic_paint(jnp.zeros([n_mesh, n_mesh, n_mesh]), pos[-1])),
        boxsize=np.array([box_size] * 3),
        kmin=np.pi / box_size,
        dk=2 * np.pi / box_size,
    )

def plot_power_spectra(
    test_pos,
    test_pos_pm,
    test_pos_pm_corr,
    n_mesh,
    box_size=25.0,
):
    ks, pks_nbody = jax.vmap(get_power_spectrum, in_axes=(0, None, None))(
        test_pos, n_mesh, box_size
    )
    ks, pks_pm = jax.vmap(get_power_spectrum, in_axes=(0, None, None))(
        test_pos_pm, n_mesh, box_size
    )
    ks, pks_pm_corr = jax.vmap(get_power_spectrum, in_axes=(0, None, None))(
        test_pos_pm_corr, n_mesh, box_size
    )
    k = ks[0]

    fig, ax = plt.subplots(2, 1, figsize=(7, 4), gridspec_kw={"height_ratios": [3, 1]})
    for i, (pk_nbody, pk_pm, pk_pm_corr) in enumerate(
        zip(pks_nbody, pks_pm, pks_pm_corr)
    ):
        c = ax[0].loglog(k, pk_nbody, label="N-body")
        ax[0].loglog(
            k,
            pk_pm,
            label="JaxPM w/o correction" if i == 0 else None,
            color=c[0].get_color(),
            linestyle="dotted",
            alpha=0.75,
        )
        ax[0].loglog(
            k,
            pk_pm_corr,
            color=c[0].get_color(),
            linestyle="dashed",
            label="JaxPM w correction" if i == 0 else None,
        )
        ax[1].semilogx(
            k,
            pk_pm / pk_nbody,
            linestyle='dotted',
            color=c[0].get_color(),
            alpha=0.75,
        )
        ax[1].semilogx(
            k,
            pk_pm_corr / pk_nbody,
            linestyle='dashed',
            color=c[0].get_color(),
        )
    ax[1].axhline(y=1, color='gray', alpha=0.5)
    ax[1].plot(k, np.mean(pks_pm[:,-1]/pks_nbody[:,-1]), linestyle='dashed',
               color='black')
    ax[1].plot(k, np.mean(pks_pm_corr[:,-1]/pks_nbody[:,-1]), linestyle='dotted',
               color='black')
    ax[0].legend()
    ax[-1].set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    ax[0].set_ylabel(r"$P(k)$")
    return fig


def eval(
    test_pos,
    test_vel,
    scale_factors,
    cosmology,
    n_mesh,
    box_size,
    model,
    params,
):
    test_pos_pm, test_vel_pm = [], []
    test_pos_pm_corr, test_vel_pm_corr = [], []
    for i in range(len(test_pos)):
        pos_pm, vel_pm = run_pm_simulation(
            pos=test_pos[i][0],
            vels=test_vel[i][0],
            scale_factors=scale_factors,
            cosmology=cosmology,
            n_mesh=n_mesh,
        )
        test_pos_pm.append(pos_pm)
        test_vel_pm.append(vel_pm)
        pos_pm_corr, vel_pm_corr = run_pm_simulation_with_correction(
            pos=test_pos[i][0],
            vels=test_vel[i][0],
            scale_factors=scale_factors,
            cosmology=cosmology,
            n_mesh=n_mesh,
            model=model,
            params=params,
        )
        test_pos_pm_corr.append(pos_pm_corr)
        test_vel_pm_corr.append(vel_pm_corr)
    test_pos_pm = jnp.array(test_pos_pm)
    test_vel_pm = jnp.array(test_vel_pm)
    test_pos_pm_corr = jnp.array(test_pos_pm_corr)
    test_vel_pm_corr = jnp.array(test_vel_pm_corr)
    fig = plot_power_spectra(
        test_pos,
        test_pos_pm,
        test_pos_pm_corr,
        n_mesh,
        box_size=box_size,
    )
    plt.savefig("pk.png",dpi=300)
