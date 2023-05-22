#%%
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

import numpy as np
import xarray as xr


#%%

def spunge_ocean(nt=1000,
    dt=365.25,
    df=1.15e-1,
    tau0=24 * 365.25,
    save_path=None,
    seed=331381460666):

    dW = np.sqrt(dt)  # (sqrt (days)) Stochastic time step

    # Precomputation
    l0 = 2 / tau0  # (days-1) inverse restoring timescale

    # Initialization
    SAT = np.zeros(nt)
    SST = np.zeros(nt)
    time = np.zeros(nt)

    random_forcing = np.random.default_rng(seed=seed).normal(0, df, nt)

    # Time Loop
    for it in np.arange(1, nt):
        fi = random_forcing[it]
        SAT[it] = fi
        time[it] = time[it - 1] + dt
        SST[it] = (
            SST[it - 1] + (-l0 * SST[it - 1]) * dt + (SAT[it - 1]) * dW
        )
    timep = time / 365.25  # (yr) TIME for plot

    ds = xr.Dataset(
    coords=dict(
        time=(["time"], time),
        time_years=(["time"], timep),
        ocean_restoring_timescale= (["ocean_restoring_timescale"], [tau0]),
        stochastic_forcing_intensity= (["stochastic_forcing_intensity"], [df]),
        ),
        data_vars=dict(
            random_forcing=(["time"], random_forcing),
            surface_air_temperature=(["time", "ocean_restoring_timescale", "stochastic_forcing_intensity"], SAT[:, np.newaxis, np.newaxis]),
            sponge_sea_surface_temperature=(["time", "ocean_restoring_timescale", "stochastic_forcing_intensity"], SST[:, np.newaxis, np.newaxis]),
        ),
        attrs=dict(
            coder="Florian Sévellec <florian.sevellec@univ-brest.fr>",
        ),
    )

    if save_path is not None:
        ds.to_netcdf(save_path, mode="w")
    else:
        return ds


def oscillator_ocean(nt=1000,
    dt=365.25,
    df=1.15e-1,
    per0=10 * 365.25,
    tau0=24 * 365.25,

    save_path=None,
    seed=331381460666):

    dW = np.sqrt(dt)  # (sqrt (days)) Stochastic time step

    # Precomputation
    l0 = 2 / tau0  # (days-1) inverse restoring timescale
    o0 = 2 * 3.14 / per0  # (days-1) inverse oscillation timescale

    # Initialization
    SAT = np.zeros(nt)
    SST = np.zeros(nt)
    DOT = np.zeros(nt)
    time = np.zeros(nt)

    random_forcing = np.random.default_rng(seed=seed).normal(0, df, nt)

    # Time Loop
    for it in np.arange(1, nt):
        SAT[it] = random_forcing[it]
        time[it] = time[it - 1] + dt
        SST[it] = (
            SST[it - 1]
            + (-l0 * SST[it - 1] - o0 * DOT[it - 1]) * dt
            + (SAT[it - 1]) * dW
        )
        DOT[it] = DOT[it - 1] + (o0 * SST[it - 1]) * dt
    timep = time / 365.25  # (yr) TIME for plot

    ds = xr.Dataset(
    coords=dict(
        time=(["time"], time),
        time_years=(["time"], timep),
        ocean_restoring_timescale= (["ocean_restoring_timescale"], [tau0]),
        ocean_oscillation_timescale= (["ocean_oscillation_timescale"], [per0]),
        stochastic_forcing_intensity= (["stochastic_forcing_intensity"], [df]),

        ),
        data_vars=dict(
            random_forcing=(["time"], random_forcing),
            surface_air_temperature=(["time", "ocean_restoring_timescale", "ocean_oscillation_timescale", "stochastic_forcing_intensity"], SAT[:, np.newaxis, np.newaxis, np.newaxis]),
            oscillator_sea_surface_temperature=(["time", "ocean_restoring_timescale", "ocean_oscillation_timescale", "stochastic_forcing_intensity"], SST[:, np.newaxis, np.newaxis, np.newaxis]),
            oscillator_deep_ocean_temperature=(["time", "ocean_restoring_timescale", "ocean_oscillation_timescale", "stochastic_forcing_intensity"], DOT[:, np.newaxis, np.newaxis, np.newaxis]),
        ),
        attrs=dict(
            coder="Florian Sévellec <florian.sevellec@univ-brest.fr>",
        ),
    )

    if save_path is not None:
        ds.to_netcdf(save_path, mode="w")
    else:
        return ds


def integrate_idealized_ocean(
    time_steps=1000,
    dt=365.25,
    stochastic_forcing_intensity=1.15e-1,
    ocean_restoring_timescale=10 * 365.25,
    ocean_oscillation_timescale=24 * 365.25,
    save_path=None,
    seed=331381460666,
):
    spunge = spunge_ocean(nt=time_steps,
                          dt=dt,
                          df=stochastic_forcing_intensity,
                          tau0=ocean_restoring_timescale,
                          seed=seed,
                          save_path=None
                          )
    oscillator = oscillator_ocean(nt=time_steps,
                          dt=dt,
                          df=stochastic_forcing_intensity,
                          tau0=ocean_restoring_timescale,
                          per0=ocean_oscillation_timescale,
                          seed=seed,
                          save_path=None
                          )
    ds = xr.merge([spunge, oscillator])

    if save_path is not None:
        ds.to_netcdf(save_path, mode="w")
    else:
        return ds
