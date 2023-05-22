# idealized.ipynb
# DESCRIPTION
# -----------
# The idea is to compute two models of the ocean surface response (SST) to an atmospheric stochastic forcing (SAT)
# The surface ocean response represents the AMO/AMV (Atlantic Multidecadal Oscillation/Variability)
# The atmsopheric stochastic forcing represents the NAO (North Atlantic Oscillation)
# The first model - Spunge Ocean (_spg) - follows is a simple spunge ocean - where SST is restored to 0
# The second model - Oscillatry Ocean (_ocs) - follows a simple damped ocsillator - where ocean oscillate between SST and DOT (Deep Ocean Temperature)
# Both ocean models are stimulated by random white noise from SAT - where only the SST is affected.
#
# EQUATIONS
# ---------
# => For the Spunge Ocean:
# dSST = - \lambda SST dt + SAT dW,
# where \lambda is the inverse of a damping time scale
# => For the Oscillatory Ocean:
# Damped oscillation are outcome of
# [ d^2 x /dt^2 ] + \lambda [ dt x / dt ]+ \omega_0^2 [ x ] = 0,
# where \omega_0= 2 * pi / per0 is the naturale frequency of the system and per0 the natural period.
# It can be decomposed in two eqaution by introducing  dt x / dt = \omega_0 y, this reads:
# [ d y dt ] =  - \lambda -  \omega_0 x and dt x / dt = \omega_0 y.
# Hence adding the stochastic noise This leads to:
# dSST =  - \lambda SST dt -  \omega_0 DOT dt + SAT dW and dDOT = \omega_0 SST dt.
#
# Coded by Florian S�vellec <florian.sevellec@univ-brest.fr> 12may2023
#
# CODE
# ----
# Import libray
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib import pyplot as plt


def idealized_ocean(time_steps = 1000, 
                    dt = 365.25,
                    ocean_restoring_timescale = 10 * 365.25,
                    ocean_oscillation_timescale = 24 * 365.25,
                    save_path = None):
    # Numerical parameter
    nt = time_steps  # number of timestep
    dt = dt  # 365.25 # (days) time step
    dW = np.sqrt(dt)  # (sqrt (days)) Stochastic time step

    # Physical parameters
    tau0 = ocean_restoring_timescale  # (days) ocean restoring timescale
    per0 = ocean_oscillation_timescale  # (days) ocean oscillation timescale
    df = 1.15e-1  # (K days-1/2) stochastic forcing intensity

    # Precomputation
    l0 = 2 / tau0  # (days-1) inverse restoring timescale
    o0 = 2 * 3.14 / per0  # (days-1) inverse oscillation timescale

    # Initialization
    SAT = np.zeros(nt)
    SST_spg = np.zeros(nt)
    SST_osc = np.zeros(nt)
    DOT_osc = np.zeros(nt)
    time = np.zeros(nt)

    # Time Loop
    for it in np.arange(1, nt):
        fi = random.normalvariate(0, df)
        SAT[it] = fi
        time[it] = time[it - 1] + dt
        SST_spg[it] = SST_spg[it - 1] + (-l0 * SST_spg[it - 1]) * dt + (SAT[it - 1]) * dW
        SST_osc[it] = (
            SST_osc[it - 1]
            + (-l0 * SST_osc[it - 1] - o0 * DOT_osc[it - 1]) * dt
            + (SAT[it - 1]) * dW
        )
        DOT_osc[it] = DOT_osc[it - 1] + (o0 * SST_osc[it - 1]) * dt

    timep = time / 365.25  # (yr) TIME for plot

    ds = integration_to_netcdf(
            time = time, 
            SAT = SAT, 
            SST_spg = SST_spg, 
            SST_osc = SST_osc, 
            DOT_osc = DOT_osc, 
            tau0 = tau0, 
            per0 = per0, 
            df = df,
            )
    
    if save_path is not False or save_path is not None:
        ds.to_netcdf(save_path, mode="w")
    else:
        return ds


# create a xarray to store the data as netcdf file
def integration_to_netcdf(time, SAT, SST_spg, SST_osc, DOT_osc, tau0, per0, df) :

    models = ["sponge", "oscillator"]
    desciption = "Data created by a idealized ocean model.\n"
    desciption += "It includes two model runs."
    desciption += "surface_air_temperature is used as forcing for both models."
    desciption += "Models:"
    desciption += "- sponge: Sponge Ocean without interior oscillation."
    desciption += "- oscillator: Oscillting Ocean of two layer with interior oscillation between surface and deep ocean."

    ds = xr.Dataset(
        coords=dict(
            time=(["time"], time),
            model=(["model"], models),
            time_years=(["time"], timep),
        ),
        data_vars=dict(
            surface_air_temperature=(["time"], SAT),
            sea_surface_temperature=(
                ["time", "model"],
                np.array([SST_spg, SST_osc]).swapaxes(0, 1),
            ),
            sponge_sea_surface_temperature=(["time"], SST_spg),
            oscillator_sea_surface_temperature=(["time"], SST_osc),
            oscillator_deep_ocean_temperature=(["time"], DOT_osc),
        ),
        attrs=dict(
            coder="Florian Sévellec <florian.sevellec@univ-brest.fr>",
            description=desciption,
            ocean_restoring_timescale=f"{tau0} in days",
            ocean_oscillation_timescale=f"{per0} in days",
            stochastic_forcing_intensity=rf"{df} in (K days^(-1/2))",
        ),
    )
    return ds

def plot_idealized_results(timep, SAT, SST_spg, SST_osc, DOT_osc) :
    fig1, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.set_ylabel("SAT (K days-1/2)")
    ax1.set_xlabel("TIME (years)")
    ax1.set_title("SPUNGE OCEAN")
    ax1.plot(timep[0:nt], SAT[0:nt], c="blue")
    ax1.set_xlim(min(timep), max(timep))
    varlim1 = np.max(abs(SAT))
    ax1.set_ylim(-varlim1, varlim1)
    ax1.grid()
    ax2.set_ylabel("SST(K)")
    ax2.set_xlabel("TIME (years)")
    ax2.plot(timep[0:nt], SST_spg[0:nt], c="red")
    ax2.set_xlim(min(timep), max(timep))
    varlim2 = np.max(abs(SST_spg))
    ax2.set_ylim(-varlim2, varlim2)
    ax2.grid()

    fig2, (ax1, ax2, ax3) = plt.subplots(nrows=3)
    ax1.set_ylabel("SAT (K days-1/2)")
    ax1.set_xlabel("TIME (years)")
    ax1.set_title("OSCILLATORY OCEAN")
    ax1.plot(timep[0:nt], SAT[0:nt], c="blue")
    ax1.set_xlim(min(timep), max(timep))
    varlim1 = np.max(abs(SAT))
    ax1.set_ylim(-varlim1, varlim1)
    ax1.grid()
    ax2.set_ylabel("SST(K)")
    ax2.set_xlabel("TIME (years)")
    ax2.plot(timep[0:nt], SST_osc[0:nt], c="red")
    ax2.set_xlim(min(timep), max(timep))
    varlim2 = np.max(abs(SST_osc))
    ax2.set_ylim(-varlim2, varlim2)
    ax2.grid()
    ax3.set_ylabel("DOT(K)")
    ax3.set_xlabel("TIME (years)")
    ax3.plot(timep[0:nt], DOT_osc[0:nt], c="red")
    ax3.set_xlim(min(timep), max(timep))
    varlim3 = np.max(abs(DOT_osc))
    ax3.set_ylim(-varlim3, varlim3)
    ax3.grid()

    plt.show()
