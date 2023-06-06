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

import os

from typing import Dict, Tuple, TypeVar, Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# typing objects
PathLike = TypeVar("PathLike", str, os.PathLike, None)
ModelOutput = TypeVar("ModelOutput", xr.Dataset, Tuple[xr.Dataset, Dict])

# Functions


def __timesteps_as_int__(timesteps):
    # make sure the timesteps are of int type:
    if not isinstance(timesteps, int):
        timesteps = int(timesteps)
        warn(
            f"The given timesteps were not in int format.\nThey were modified and are now: timesteps = {timesteps}"
        )
    return timesteps


def spunge_ocean(
    nt: Union[int, float] = 1000,
    dt: float = 365.25,
    df: float = 1.15e-1,
    tau0: float = 10 * 365.25,
    save_path: PathLike = None,
    return_settings: bool = False,
    seed: int = 331381460666,
) -> ModelOutput:
    """
    Simulates the temperature evolution of the ocean surface using a stochastic
    model with a restoring timescale.

    Parameters:
        nt (int): Number of time steps (default: 1000).
        dt (float): Time step size in days (default: 365.25).
        df (float): Intensity of stochastic forcing (default: 1.15e-1).
        tau0 (float): Restoring timescale in days (default: 10 * 365.25).
        save_path (str): Path to save the resulting data as a NetCDF file (default: None).
        return_settings (bool): If True, returns the simulation settings along with the data (default: False).
        seed (int): Seed for the random number generator (default: 331381460666).

    Returns:
        xr.Dataset: Dataset containing simulated data variables.
    """

    # Verify that timesteps are of int type:
    nt = __timesteps_as_int__(nt)

    # Stochastic time step
    dW = np.sqrt(dt)

    # Precomputation
    l0 = 2 / tau0  # Inverse restoring timescale

    # Generate random forcing
    random_forcing = np.random.default_rng(seed=seed).normal(0, df, nt)

    # Time array
    time = np.arange(nt) * dt

    # Time Loop
    SAT = random_forcing.copy()
    SST = np.zeros_like(SAT)

    # Generate random forcing
    random_forcing = np.random.default_rng(seed=seed).normal(0, df, nt)

    # Time Loop
    for it in range(1, nt):
        time[it] = time[it - 1] + dt
        SST[it] = SST[it - 1] + (-l0 * SST[it - 1]) * dt + (SAT[it - 1]) * dW

    # Time for plot (in years)
    timep = time / 365.25

    # Create the dataset
    ds = xr.Dataset(
        coords=dict(
            time=(["time"], time),
            time_years=(["time"], timep),
        ),
        data_vars=dict(
            random_forcing=(["time"], random_forcing),
            SAT=(
                ["time"],
                SAT,
            ),
            SST=(
                ["time"],
                SST,
            ),
        ),
        attrs=dict(
            coder="Florian Sévellec <florian.sevellec@univ-brest.fr>",
            stochastic_forcing_intensity=df,
            ocean_restoring_timescale=tau0,
        ),
    )

    # Save the dataset if save_path is provided
    if save_path is not None:
        ds.to_netcdf(save_path, mode="w")

    # Return dataset and settings if return_settings is True
    elif return_settings is not False:
        settings = dict(
            nt=nt,
            dt=dt,
            df=df,
            tau0=tau0,
            seed=seed,
        )
        return ds, settings

    # Return the dataset
    else:
        return ds


def oscillatory_ocean(
    nt: Union[int, float] = 1000,
    dt: float = 365.25,
    df: float = 1.15e-1,
    per0: float = 24 * 365.25,
    tau0: float = 10 * 365.25,
    save_path: PathLike = None,
    return_settings: bool = False,
    seed: int = 331381460666,
) -> ModelOutput:
    """
    Simulates the temperature evolution of the ocean surface and deep ocean
    using a stochastic model with restoring and oscillation timescales.

    Parameters
    ----------
    nt : int
        Number of time steps (default: 1000).
    dt : float
        Time step size in days (default: 365.25).
    df : float
        Intensity of stochastic forcing (default: 1.15e-1).
    per0 : float
        Period of oscillation in days (default: 24 * 365.25).
    tau0 : float
        Restoring timescale in days (default: 10 * 365.25).
    save_path : str
        Path to save the resulting data as a NetCDF file (default: None).
    return_settings : bool
        If True, returns the simulation settings along with the data (default: False).
    seed : int
        Seed for the random number generator (default: 331381460666).

    Returns
    -------
    xr.Dataset
        Dataset containing simulated data variables.
    """

    # Verify that timesteps are of int type:
    nt = __timesteps_as_int__(nt)

    # Stochastic time step
    dW = np.sqrt(dt)

    # Precomputation
    l0 = 2 / tau0  # Inverse restoring timescale
    o0 = 2 * np.pi / per0  # Inverse oscillation timescale

    # Generate random forcing
    random_forcing = np.random.default_rng(seed=seed).normal(0, df, nt)

    # Time array
    time = np.arange(nt) * dt

    # Time Loop
    SAT = random_forcing.copy()
    SST = np.zeros_like(SAT)
    DOT = np.zeros_like(SAT)
    for it in range(1, nt):
        SST[it] = (
            SST[it - 1] + (-l0 * SST[it - 1] - o0 * DOT[it - 1]) * dt + SAT[it - 1] * dW
        )
        DOT[it] = DOT[it - 1] + o0 * SST[it - 1] * dt

    # Time for plot (in years)
    timep = time / 365.25

    # Create the dataset
    ds = xr.Dataset(
        coords=dict(
            time=(["time"], time),
            time_years=(["time"], timep),
        ),
        data_vars=dict(
            random_forcing=(["time"], random_forcing),
            SAT=(
                [
                    "time",
                ],
                SAT,
            ),
            SST=(
                [
                    "time",
                ],
                SST,
            ),
            DOT=(
                [
                    "time",
                ],
                DOT,
            ),
        ),
        attrs=dict(
            coder="Florian Sévellec <florian.sevellec@univ-brest.fr>",
            stochastic_forcing_intensity=df,
            ocean_restoring_timescale=tau0,
            ocean_oscillation_timescale=per0,
        ),
    )

    if save_path is not None:
        ds.to_netcdf(save_path, mode="w")
    elif return_settings is not False:
        # create settings dict to store all information
        settings = dict(
            nt=nt,
            dt=dt,
            df=df,
            per0=per0,
            tau0=tau0,
            seed=seed,
        )
        return ds, settings
    else:
        return ds


def AMO_oscillatory_ocean(
    nt: Union[int, float] = 1000,
    dt: float = 365.25,
    per0: float = 24 * 365.25,
    tau0: float = 10 * 365.25,
    dNAO: float = 0.1,  # (K days-1/2) stochastic amplitude of NAO
    dEAP: float = 0.1,  # (K days-1/2) stochastic amplitude of EAP
    cNAOvsEAP: float = 0,  # (K^2 days) Covariance of NAO and EAP
    save_path: PathLike = None,
    return_settings: bool = False,
    seed: int = 331381460666,
) -> ModelOutput:
    """
    Simulates the temperature evolution of the Atlantic Multidecadal
    Oscillation (AMO) and related atmospheric variables using a stochastic
    model with restoring and oscillation timescales.

    The AMO is a natural climate variability pattern characterized by long-lived fluctuations in the sea surface temperature of the North Atlantic Ocean. It plays a significant role in modulating regional and global climate conditions, including precipitation patterns, hurricane activity, and marine ecosystems.

    This function implements a stochastic model that captures the essential dynamics of the AMO. The model simulates the interactions between the AMO and two atmospheric variables, the North Atlantic Oscillation (NAO) and the East Atlantic Pattern (EAP), which influence the ocean temperature through stochastic atmospheric forcing.

    Parameters
    ----------
    nt : int
        Number of time steps (default: 1000).
    dt : float
        Time step size in days (default: 365.25).
    per0 : float
        Period of oscillation in days (default: 24 * 365.25).
    tau0 : float
        Restoring timescale in days (default: 10 * 365.25).
    dNAO : float
        Stochastic amplitude of NAO (default: 0.1).
    dEAP : float
        Stochastic amplitude of EAP (default: 0.1).
    cNAOvsEAP : float
        Covariance of NAO and EAP (default: 0).
    save_path : str
        Path to save the resulting data as a NetCDF file (default: None).
    return_settings : bool
        If True, returns the simulation settings along with the data (default: False).
    seed : int
        Seed for the random number generator (default: 331381460666).

    Returns
    -------
    xr.Dataset
        Dataset containing simulated data variables.
    """

    # Verify that timesteps are of int type:
    nt = __timesteps_as_int__(nt)

    # Stochastic time step
    dW = np.sqrt(dt)

    # Precomputation
    l0 = 2 / tau0  # Inverse restoring timescale
    o0 = 2 * np.pi / per0  # Inverse oscillation timescale
    A = np.array(
        [[dNAO**2, cNAOvsEAP], [cNAOvsEAP, dEAP**2]]
    )  # Covariance Matrix of atmospheric forcing
    L = np.linalg.cholesky(A)  # Cholesky factorization of the Covariance Matrix

    # Time array
    time = np.arange(nt) * dt

    # Initialize arrays
    NAO = np.zeros(nt)
    EAP = np.zeros(nt)
    AMO = np.zeros(nt)
    ZOT = np.zeros(nt)

    # Random number generator
    rng = np.random.default_rng(seed=seed)

    # Time Loop
    for it in range(1, nt):
        time[it] = time[it - 1] + dt

        # Generate random forcing
        ft = rng.standard_normal(2)
        ftt = np.matmul(ft, L.T)
        NAO[it] = ftt[0]
        EAP[it] = ftt[1]

        AMO[it] = (
            AMO[it - 1] + (-l0 * AMO[it - 1] - o0 * ZOT[it - 1]) * dt + EAP[it - 1] * dW
        )
        ZOT[it] = (
            ZOT[it - 1] + (-l0 * ZOT[it - 1] + o0 * AMO[it - 1]) * dt + NAO[it - 1] * dW
        )

    # Convert time to years for plotting
    timep = time / 365.25

    # Create the dataset
    ds = xr.Dataset(
        coords=dict(
            time=(["time"], time),
            time_years=(["time"], timep),
        ),
        data_vars=dict(
            AMO=(
                ["time"],
                AMO,
            ),
            EAP=(
                ["time"],
                EAP,
            ),
            NAO=(
                ["time"],
                NAO,
            ),
            ZOT=(
                ["time"],
                ZOT,
            ),
        ),
        attrs=dict(coder="Florian Sévellec <florian.sevellec@univ-brest.fr>"),
    )

    if save_path is not None:
        ds.to_netcdf(save_path, mode="w")
    elif return_settings:
        settings = dict(
            nt=nt,
            dt=dt,
            per0=per0,
            tau0=tau0,
            dNAO=dNAO,
            dEAP=dEAP,
            cNAOvsEAP=cNAOvsEAP,
            seed=seed,
        )
        return ds, settings
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
    # Verify that timesteps are of int type:
    time_steps = __timesteps_as_int__(time_steps)

    spunge = spunge_ocean(
        nt=time_steps,
        dt=dt,
        df=stochastic_forcing_intensity,
        tau0=ocean_restoring_timescale,
        seed=seed,
        save_path=None,
    )
    oscillator = oscillatory_ocean(
        nt=time_steps,
        dt=dt,
        df=stochastic_forcing_intensity,
        tau0=ocean_restoring_timescale,
        per0=ocean_oscillation_timescale,
        seed=seed,
        save_path=None,
    )
    ds = xr.merge([spunge, oscillator])

    if save_path is not None:
        ds.to_netcdf(save_path, mode="w")
    else:
        return ds


def integrate_all(
    time_steps=1000,
    dt=365.25,
    stochastic_forcing_intensity=1.15e-1,
    ocean_restoring_timescale=10 * 365.25,
    ocean_oscillation_timescale=24 * 365.25,
    dNAO=0.1,  # (K days-1/2) stochastic amplitude of NAO
    dEAP=0.1,  # (K days-1/2) stochastic amplitude of EAP
    cNAOvsEAP=0,  # (K^2 days) Covariance of NAO and EAP
    save_path=None,
    return_settings=False,
    seed=331381460666,
):
    # Verify that timesteps are of int type:
    time_steps = __timesteps_as_int__(time_steps)

    spunge = spunge_ocean(
        nt=time_steps,
        dt=dt,
        df=stochastic_forcing_intensity,
        tau0=ocean_restoring_timescale,
        seed=seed,
        save_path=None,
    )
    oscillator = oscillatory_ocean(
        nt=time_steps,
        dt=dt,
        df=stochastic_forcing_intensity,
        tau0=ocean_restoring_timescale,
        per0=ocean_oscillation_timescale,
        seed=seed,
        save_path=None,
    )
    rossby = AMO_oscillatory_ocean(
        nt=time_steps,
        dt=dt,
        tau0=ocean_restoring_timescale,
        per0=ocean_oscillation_timescale,
        dNAO=dNAO,  # (K days-1/2) stochastic amplitude of NAO
        dEAP=dEAP,  # (K days-1/2) stochastic amplitude of EAP
        cNAOvsEAP=cNAOvsEAP,
        seed=seed,
        save_path=None,
    )
    ds = xr.merge([spunge, oscillator, rossby])

    if save_path is not None:
        ds.to_netcdf(save_path, mode="w")
    elif return_settings is not False:
        # create settings dict to store all information
        settings = dict(
            nt=time_steps,
            dt=dt,
            df=stochastic_forcing_intensity,
            tau0=ocean_restoring_timescale,
            dNAO=dNAO,  # (K days-1/2) stochastic amplitude of NAO
            dEAP=dEAP,  # (K days-1/2) stochastic amplitude of EAP
            cNAOvsEAP=cNAOvsEAP,
            seed=seed,
        )
        return ds, settings

    else:
        return ds
