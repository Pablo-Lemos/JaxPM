import jax.numpy as jnp
import torch
import jax
import matplotlib.pyplot as plt
import numpy as np
from jaxpm.painting import compensate_cic, cic_paint
from jaxpm.utils import power_spectrum, cross_correlation_coefficients
import jax_cosmo as jc

from pypower import CatalogFFTPower
from jaxpm.painting import cic_paint

from kymatio.torch import HarmonicScattering3D
from kymatio.scattering3d.backend.torch_backend import TorchBackend3D
from mpi4py import MPI
from simulate import run_pm_simulation, run_pm_simulation_with_correction


def get_power_spectrum(pos, n_mesh, box_size):
    return power_spectrum(
        compensate_cic(cic_paint(jnp.zeros([n_mesh, n_mesh, n_mesh]), pos[-1])),
        boxsize=np.array([box_size] * 3),
        kmin=np.pi / box_size,
        dk=2 * np.pi / box_size,
    )


def add_rsd(pos, vel, cosmology, scale_factor, box_size):
    z_rsd = (
        pos[..., -1]
        + 100.*vel[..., -1] / jc.background.H(cosmology, a=scale_factor) / scale_factor
    )
    z_rsd %= box_size
    pos_rsd = pos.copy()
    pos_rsd = pos_rsd.at[..., -1].set(z_rsd)
    return pos_rsd


def get_power_spectrum_multipoles(pos, vel, cosmology, scale_factor, box_size, n_mesh):
    pos_rsd = add_rsd(
        pos=pos,
        vel=vel,
        cosmology=cosmology,
        scale_factor=scale_factor,
        box_size=box_size,
    )
    kedges = np.linspace(0.0, 10.0, 31)
    comm = MPI.COMM_WORLD
    result = CatalogFFTPower(
        data_positions1=np.asarray(pos_rsd),
        edges=kedges,
        ells=(0, 2),
        boxsize=box_size,
        nmesh=n_mesh,
        resampler="tsc",
        los="z",
        position_type="pos",
        mpiroot=0,
        mpicomm=comm,
    )
    return result.poles.k, result.poles.power


def plot_power_spectra_multipoles(
    test_pos,
    test_vel,
    test_pos_pm,
    test_vel_pm,
    test_pos_pm_corr,
    test_vel_pm_corr,
    cosmology,
    n_mesh,
    scale_factor=1.0,
    box_size=25.0,
):
    fig, ax = plt.subplots(
        4,
        1,
        figsize=(7, 8),
        gridspec_kw={"height_ratios": [3, 1, 3, 1]},
        sharex=True,
    )
    pks_nbody, pks_pm, pks_corr = [], [], []
    for idx in range(len(test_pos)):
        k, pk_nbody = get_power_spectrum_multipoles(
            pos=test_pos[idx][-1],
            vel=test_vel[idx][-1],
            cosmology=cosmology,
            box_size=box_size,
            n_mesh=n_mesh,
            scale_factor=scale_factor,
        )
        k, pk_pm = get_power_spectrum_multipoles(
            pos=test_pos_pm[idx][-1],
            vel=test_vel_pm[idx][-1],
            cosmology=cosmology,
            box_size=box_size,
            n_mesh=n_mesh,
            scale_factor=scale_factor,
        )
        k, pk_pm_corr = get_power_spectrum_multipoles(
            pos=test_pos_pm_corr[idx][-1],
            vel=test_vel_pm_corr[idx][-1],
            cosmology=cosmology,
            box_size=box_size,
            n_mesh=n_mesh,
            scale_factor=scale_factor,
        )
        pks_nbody.append(pk_nbody)
        pks_pm.append(pk_pm)
        pks_corr.append(pk_pm_corr)
        for i in range(2):
            c = ax[2 * i].semilogx(k, k*pk_nbody[i], label="N-body" if idx == 0 else None)
            ax[2 * i].semilogx(
                k,
                k*pk_pm[i],
                label="JaxPM w/o correction" if idx == 0 else None,
                color=c[0].get_color(),
                linestyle="dotted",
                alpha=0.75,
            )
            ax[2 * i].semilogx(
                k,
                k*pk_pm_corr[i],
                label="JaxPM w correction" if idx == 0 else None,
                color=c[0].get_color(),
                linestyle="dashed",
            )

            ax[2 * i + 1].semilogx(
                k,
                pk_pm[i] / pk_nbody[i],
                linestyle="dotted",
                color=c[0].get_color(),
                alpha=0.75,
            )
            ax[2 * i + 1].semilogx(
                k,
                pk_pm_corr[i] / pk_nbody[i],
                linestyle="dashed",
                color=c[0].get_color(),
            )
    pks_nbody = np.array(pks_nbody)
    pks_pm = np.array(pks_pm)
    pks_corr = np.array(pks_corr)

    ax[0].legend()
    ax[1].axhline(y=1, color="gray", alpha=0.5)
    ax[3].axhline(y=1, color="gray", alpha=0.5)
    ax[1].plot(
        k,
        np.mean(pks_pm[:, 0] / pks_nbody[:, 0], axis=0),
        linestyle="dashed",
        color="black",
    )
    ax[1].plot(
        k,
        np.mean(
            pks_corr[:, 0] / pks_nbody[:, 0],
            axis=0,
        ),
        linestyle="dotted",
        color="black",
    )
    ax[3].plot(
        k,
        np.mean(pks_pm[:, 1] / pks_nbody[:, 1], axis=0),
        linestyle="dashed",
        color="black",
    )
    ax[3].plot(
        k,
        np.mean(pks_corr[:, 1] / pks_nbody[:, 1], axis=0),
        linestyle="dotted",
        color="black",
    )

    ax[0].legend()
    ax[-1].set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    ax[0].set_ylabel(r"$k P_0(k)$")
    ax[2].set_ylabel(r"$k P_2(k)$")
    return fig


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

    fig, ax = plt.subplots(
        2, 1, figsize=(7, 4), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
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
            linestyle="dotted",
            color=c[0].get_color(),
            alpha=0.75,
        )
        ax[1].semilogx(
            k,
            pk_pm_corr / pk_nbody,
            linestyle="dashed",
            color=c[0].get_color(),
        )
    ax[1].axhline(y=1, color="gray", alpha=0.5)
    ax[1].semilogx(
        k, np.mean(pks_pm / pks_nbody, axis=0), linestyle="dashed", color="black"
    )
    ax[1].semilogx(
        k, np.mean(pks_pm_corr / pks_nbody, axis=0), linestyle="dotted", color="black"
    )
    ax[0].legend()
    ax[-1].set_xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    ax[0].set_ylabel(r"$P(k)$")
    return fig


def get_wst(S, density_field, integral_powers, device):
    full_density_batch = torch.from_numpy(np.asarray(np.real(density_field)))
    full_density_batch = full_density_batch.to(device).float()
    full_density_batch = full_density_batch.contiguous()
    w_orders_12 = S(full_density_batch)
    s_mat_avg = np.real(w_orders_12.cpu().numpy()[:, :, 0])
    s_mat_avg = s_mat_avg.flatten()
    test_shape = np.empty(
        [
            1,
            density_field.shape[0],
            density_field.shape[0],
            density_field.shape[0],
        ]
    )
    test_shape[0, :] = np.asarray(np.absolute(density_field))
    s0_batch = torch.from_numpy(test_shape)
    integr = TorchBackend3D.compute_integrals(s0_batch, integral_powers)
    s0 = integr.cpu().numpy()[0, 0]
    return np.hstack((s0, s_mat_avg)) / len(density_field) ** 3


def compute_wst_coefficients(
    nbody_mesh,
    pm_mesh,
    pm_corr_mesh,
    J_3d: int = 4,
    L_3d: int = 4,
    integral_powers=[
        0.8,
    ],
    sigma: float = 0.8,
    n_mesh: int = 360,
):
    S = HarmonicScattering3D(
        J=J_3d,
        shape=(
            n_mesh,
            n_mesh,
            n_mesh,
        ),
        L=L_3d,
        sigma_0=sigma,
        integral_powers=integral_powers,
        max_order=2,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    S.to(device)
    wst_nbody = np.array(
        [
            get_wst(
                S=S,
                density_field=df,
                integral_powers=integral_powers,
                device=device,
            )
            for df in nbody_mesh
        ]
    )
    wst_pm = np.array(
        [
            get_wst(
                S=S,
                density_field=df,
                integral_powers=integral_powers,
                device=device,
            )
            for df in pm_mesh
        ]
    )
    wst_pm_corr = np.array(
        [
            get_wst(
                S=S,
                density_field=df,
                integral_powers=integral_powers,
                device=device,
            )
            for df in pm_corr_mesh
        ]
    )
    return wst_nbody, wst_pm, wst_pm_corr


def plot_wst(
    nbody_mesh,
    pm_mesh,
    pm_corr_mesh,
    n_mesh,
):

    wst_nbody, wst_pm, wst_pm_corr = compute_wst_coefficients(
        nbody_mesh=nbody_mesh,
        pm_mesh=pm_mesh,
        pm_corr_mesh=pm_corr_mesh,
        n_mesh=n_mesh,
    )
    fig, ax = plt.subplots()
    x_range = list(range(len(wst_nbody[0])))
    for idx in range(len(nbody_mesh)):
        ax.plot(x_range, wst_nbody[idx], label="N-body" if idx == 0 else None)
        ax.plot(
            x_range,
            wst_pm[idx],
            label="JaxPM w/o correction" if idx == 0 else None,
            linestyle="dashed",
        )
        ax.plot(
            x_range,
            wst_pm_corr[idx],
            label="JaxPM w correction" if idx == 0 else None,
            linestyle="dotted",
        )
    ax.set_xlabel('Coefficient index')
    ax.set_ylabel('WST coefficients')
    ax.legend()
    return fig

def plot_cross_corr(nbody_mesh, pm_mesh, pm_corr_mesh, n_mesh, box_size,):
    k, cross_pm = jax.vmap(
        cross_correlation_coefficients,
        in_axes=(0, 0, None, None, None,),
        
    )(
            nbody_mesh,
            pm_mesh,
            boxsize=np.array([box_size] * 3),
            kmin=np.pi / box_size,
            dk=2 * np.pi / box_size,
    )
    k, cross_corr = jax.vmap(
        cross_correlation_coefficients,
        in_axes=(0, 0, None, None, None),
        
    )(
            nbody_mesh,
            pm_corr_mesh,
            boxsize=np.array([box_size] * 3),
            kmin=np.pi / box_size,
            dk=2 * np.pi / box_size,
    )
    fig, ax = plt.subplots()
    for idx in range(len(nbody_mesh)):
        ax.semilogx(
            k,
            cross_pm[idx],
            label="JaxPM w/o correction" if idx == 0 else None,
            linestyle="dashed",
        )
        ax.semilogx(
            k,
            cross_corr[idx],
            label="JaxPM w correction" if idx == 0 else None,
            linestyle="dotted",
        )
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$C(k)$')
    ax.legend()
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
    test_pos_pm = jnp.array(test_pos_pm) / n_mesh * box_size
    test_vel_pm = jnp.array(test_vel_pm) / n_mesh * box_size
    test_pos_pm_corr = jnp.array(test_pos_pm_corr) / n_mesh * box_size
    test_vel_pm_corr = jnp.array(test_vel_pm_corr) / n_mesh * box_size
    test_pos = test_pos / n_mesh * box_size
    test_vel = test_vel / n_mesh * box_size
    test_pos_pm %= box_size
    test_pos_pm_corr %= box_size
    fig = plot_power_spectra(
        test_pos,
        test_pos_pm,
        test_pos_pm_corr,
        n_mesh,
        box_size=box_size,
    )
    plt.savefig("pk.png", dpi=300)
    fig = plot_power_spectra_multipoles(
        test_pos=test_pos,
        test_vel=test_vel,
        test_pos_pm=test_pos_pm,
        test_vel_pm=test_vel_pm,
        test_pos_pm_corr=test_pos_pm_corr,
        test_vel_pm_corr=test_vel_pm_corr,
        cosmology=cosmology,
        n_mesh=n_mesh,
        box_size=box_size,
    )
    plt.savefig("pk_rsd.png", dpi=300)
    nbody_mesh = jax.vmap(
        cic_paint,
        in_axes=(None, 0),
    )(jnp.zeros([n_mesh, n_mesh, n_mesh]), test_pos[:, -1])
    pm_mesh = jax.vmap(
        cic_paint,
        in_axes=(None, 0),
    )(jnp.zeros([n_mesh, n_mesh, n_mesh]), test_pos_pm[:, -1])
    pm_corr_mesh = jax.vmap(
        cic_paint,
        in_axes=(None, 0),
    )(jnp.zeros([n_mesh, n_mesh, n_mesh]), test_pos_pm_corr[:, -1])
    fig = plot_wst(
        nbody_mesh,
        pm_mesh,
        pm_corr_mesh,
        n_mesh=n_mesh,
    )
    plt.savefig("wst.png", dpi=300)
    fig = plot_cross_corr(
        nbody_mesh,
        pm_mesh,
        pm_corr_mesh,
        n_mesh=n_mesh,
        box_size=box_size,
    )
    plt.savefig("cross_corr.png", dpi=300)
