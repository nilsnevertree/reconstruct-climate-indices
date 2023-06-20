"""
This file runs the ``Kalmnan-SEM`` function on the CiCMOD netCDF files.
The settings will be stored as "settings.yaml" in the ``results_path``.
Also multiple plots will be created and stored in the same directory.


An example would be :
python plot_CiCMOD.py --file_name climate_indices_FOCI.nc --experiment_name FOCI_default

NOTE:
    The kalman settings need to be manually changes in the file and can not be passed as arguments using ``argparse``.
"""

# %%
import argparse

from pathlib import Path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--file_name",
    help="The filename should be is usually ``climate_indices_{model}.nc``.",
)
parser.add_argument(
    "--dir_path",
    default=Path("data") / Path("earth_system_models") / Path("CiCMOD"),
    help="The source directory path where the netCDF file is stored.\n Default to ``.data/earth_system_models/CiCMOD``",
)
parser.add_argument(
    "--results_path",
    default=Path("results") / Path("CiCMOD"),
    help="The name to store the figures. \n Default to ``results/CiCMOD``",
)
parser.add_argument(
    "--experiment_name",
    default=Path("default"),
    help="The name of directory below ``results_path`` to store the figures in. \n Default to ``default``",
)

args = parser.parse_args()
print("Start imports.")


# from scipy import fftpack
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml

from kalman_reconstruction import pipeline
from kalman_reconstruction.custom_plot import (
    adjust_lightness,
    ncols_nrows_from_N,
    plot_state_with_probability,
    set_custom_rcParams,
)
from kalman_reconstruction.statistics import compute_fft_spectrum, crosscorr, normalize
from tqdm import tqdm


set_custom_rcParams()
plt.rcParams["axes.grid"] = True

# %%
# Get arguments
file_name = args.file_name
dir_path = args.dir_path
results_path = args.results_path
experiment_name = args.experiment_name

# extract model name
model = file_name.split("_")[-1]
model = model.split(".")[0]

# Set all paths
REPO_PATH = Path(__file__).parent.parent.parent
data_path = REPO_PATH / Path(dir_path) / Path(file_name)
results_path = REPO_PATH / Path(results_path) / Path(experiment_name)
results_path.mkdir(parents=True, exist_ok=True)

SAVE_FIGURES = True


def save_fig(fig, relative_path, **kwargs):
    store_path = results_path / relative_path
    store_path.parent.mkdir(parents=True, exist_ok=True)
    if SAVE_FIGURES:
        fig.savefig(store_path, **kwargs)
    else:
        pass


# %%
# Set the Kalman and plot settings
time_slice = slice(0, 201)
rolling_window = 10 * 12
# Set Kalman-SEM settings
rng_seed = 834567
random_variance = 1
nb_iter_SEM = 25
observation_variables = ["AMO", "PDO_PC", "NAO_PC"]
state_variables = ["AMO", "PDO_PC", "NAO_PC", "latent"]
smoothed_variables = ["AMO"]

settings = dict(
    rolling_window=rolling_window,
    rng_seed=rng_seed,
    random_variance=random_variance,
    observation_variables=observation_variables,
    state_variables=state_variables,
    smoothed_variables=smoothed_variables,
    nb_iter_SEM=nb_iter_SEM,
    time_slice=dict(
        start=time_slice.start,
        stop=time_slice.stop,
    ),
    data_path=str(data_path),
)
with open(results_path / "settings.yaml", "w") as stream:
    stream.write(
        "#Settings used in the application of the Kalman_SEM on the CiCOD dataset.\n"
    )
    yaml.dump(data=settings, stream=stream, default_flow_style=False)


# %%
data_original = xr.load_dataset(data_path)

# %%
data = data_original.sel(time=time_slice).copy()
try:
    for var in smoothed_variables:
        data[var] = data[var].rolling(time=rolling_window).mean()
    data = data.isel(time=slice(rolling_window, -rolling_window))
except:
    pass


random_vars = ["latent"]
for random_var in random_vars:
    rng = np.random.default_rng(seed=rng_seed)
    pipeline.add_random_variable(
        ds=data,
        var_name=random_var,
        random_generator=rng,
        variance=random_variance,
        dim="time",
    )


# %%
kalman_results = pipeline.xarray_Kalman_SEM(
    ds=data,
    observation_variables=observation_variables,
    state_variables=state_variables,
    nb_iter_SEM=nb_iter_SEM,
)

# %%
kalman_states = pipeline.from_standard_dataset(kalman_results)

# %% Plot loglikelihood
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(kalman_results.kalman_itteration, kalman_results.log_likelihod)
ax.set_xlabel("kalman itteration")
ax.set_ylabel("log likelihood")
fig.suptitle("CiCMOD | Loglikelihood Kalman SEM")
save_fig(fig, "CiCMOD_loglikelihood.png", dpi=400)


# %% Plot scatter truth against reconstruction
fig, axs = plt.subplots(
    nrows=len(state_variables),
    ncols=1,
    layout="constrained",
    figsize=(12, 7),
    sharex=True,
)
axs = axs.flatten()
for idx, var in enumerate(state_variables):
    handle1, handle2 = plot_state_with_probability(
        ax=axs[idx],
        x_value=kalman_results.time,
        state=kalman_results.states.sel(state_name=var),
        prob=kalman_results.covariance.sel(state_name=var, state_name_copy=var),
        line_kwargs=dict(label=f"{var}"),
        output=True,
    )

    # if "latent" not in var:
    # color = adjust_lightness(handle1[0].get_color(), )
    # axs[idx].plot(data.time, data[var], label = f"{var} truth", alpha = 0.7, linestyle = ":")
    axs[idx].set_title(var)
    axs[idx].set_ylabel("value")
    axs[idx].legend()
axs[idx].set_xlabel("time in years")
fig.suptitle("CiCMOD | Reconstruction against truth")
save_fig(fig, "CiCMOD_recons_truth.png", dpi=400)

# %% Plot scatter truth against reconstruction
# n_cols = len(state_variables)
# fig, axs = plt.subplots(layout = "constrained", figsize = (12,4), ncols = n_cols)
# axs = axs.flatten()

# for idx, var in enumerate(state_variables):
#     axs[idx].scatter(
#         kalman_states[var],
#         data[var],
#     )
#     axs[idx].set_xlabel("reconstruction")
#     axs[idx].set_ylabel("truth")
#     axs[idx].set_title(var)

# fig.suptitle("CiCMOD | Reconstruction against truth")
# save_fig(fig, "CiCMOD_recons_truth_scatter.png", dpi = 400)

# %% Plot scatter ``latent`` varibale against reconstruction
data_vars = data.data_vars
row_col = ncols_nrows_from_N(len(data_vars))

fig, axs = plt.subplots(layout="constrained", figsize=(20, 20), **row_col)
axs = axs.flatten()

for idx, var in enumerate(data_vars):
    axs[idx].scatter(
        normalize(kalman_states.latent),
        normalize(data[var]),
        marker=".",
        alpha=0.75,
    )
    axs[idx].set_xlabel(var)
    axs[idx].set_ylabel("variable")
    axs[idx].set_title(var)

fig.suptitle("CiCMOD | Latent variable against climate indeces")
save_fig(fig, "CiCMOD_latent_indices_scatter.png", dpi=400)

# %% [markdown]
### Compute lagged cross correlation and covariance

# %% Calculate lagged cross correlation for all variables in the CiCMOD dataset against ``latent``
lag_years = np.arange(-30, 30, 1)
data_vars = data.data_vars
data_ccor_list = []
kalman_ccor_list = []
for idx, var in tqdm(enumerate(data_vars)):
    for lag in lag_years:
        # because data is stored in monthly form, we need to multiply the shift by 12 to have teh lag in years
        lag_months = lag * 12
        # # No need to calculate covarinace
        # # calculate the covariance
        # ccov = xr.cov(data[var], kalman_states.latent.shift(time=lag*12), dim = "time").values
        # da_ccov = xr.DataArray(
        #     data = ccov[np.newaxis],
        #     dims=["lag_years"],
        #     coords = dict(
        #         lag_years = (["lag_years"], [lag]),
        #     )
        # )
        # da_ccov = da_ccov.rename(var)
        # da_ccov_list.append(da_ccov)

        # calculate the lagged cross correlation
        data_ccor = crosscorr(
            ds1=data[var], ds2=kalman_states.latent, lag=lag_months, dim="time"
        )
        # create the DataArray
        data_ccor = xr.DataArray(
            data=data_ccor.values[np.newaxis],
            dims=["lag_years"],
            coords=dict(
                lag_years=(["lag_years"], [lag]),
            ),
        )
        data_ccor = data_ccor.rename(var)
        data_ccor_list.append(data_ccor)
        # Also for all varibales try to make cross correaltino with the results from the kalman SEM
        try:
            kalman_ccor = crosscorr(
                ds1=kalman_states[var],
                ds2=kalman_states.latent,
                lag=lag_months,
                dim="time",
            )
            # Create the DataArray
            kalman_ccor = xr.DataArray(
                data=kalman_ccor.values[np.newaxis],
                dims=["lag_years"],
                coords=dict(
                    lag_years=(["lag_years"], [lag]),
                ),
            )
            kalman_ccor = kalman_ccor.rename(var)
            kalman_ccor_list.append(kalman_ccor)
        except:
            pass

# Merge all DataArrays to Datasets
data_ccor = xr.merge(data_ccor_list)  # lagged cross correlation with all CICMOD vars
kalman_ccor = xr.merge(
    kalman_ccor_list
)  # lagged cross correlation with all vars avaibales in the kalman results.

# %% STATE VARIABLES | Plot Lagged correlation of latent variable against state varibales
data_vars = state_variables
row_col = ncols_nrows_from_N(len(data_vars))

fig, axs = plt.subplots(
    layout="constrained",
    figsize=(12, 4),
    sharex=True,
    sharey=True,
    ncols=len(data_vars),
)
axs = axs.flatten()

for idx, var in enumerate(data_vars):
    axs[idx].step(data_ccor.lag_years, np.abs(data_ccor[var]), label="turth")
    axs[idx].step(kalman_ccor.lag_years, np.abs(kalman_ccor[var]), label="recons")
    axs[idx].set_xlabel("lag in years")
    axs[idx].set_ylabel("correlation")
    axs[idx].set_title(var)
    axs[idx].legend()

axs[idx].set_ylim((-1, 1))

fig.suptitle(
    "CiCMOD | Absolute value of lagged correlation of latent variable against state varibales"
)
save_fig(fig, "CiCMOD_latent_states_lagged_corr.png", dpi=400)


# %% ALL INDICES | lagged correlation of latent variable against state varibales
data_vars = data.data_vars
row_col = ncols_nrows_from_N(len(data_vars))

fig, axs = plt.subplots(
    layout="constrained", figsize=(20, 20), sharex=True, sharey=True, **row_col
)
axs = axs.flatten()

for idx, var in enumerate(data_vars):
    axs[idx].step(data_ccor.lag_years, data_ccor[var], label="turth")
    try:
        axs[idx].step(kalman_ccor.lag_years, kalman_ccor[var], label="recons")
    except:
        pass
    axs[idx].set_xlabel("lag in years")
    axs[idx].set_ylabel("correlation")
    axs[idx].set_title(var)
    axs[idx].legend()

extend = np.max(np.abs(axs[idx].get_ylim()))
axs[idx].set_ylim((-extend, extend))

fig.suptitle("CiCMOD | Lagged correlation of latent against climate indices.")
save_fig(fig, "CiCMOD_latent_indices_lagged_corr.png", dpi=400)

# %% ALL INDICES | Absolute value of lagged correlation of latent variable against state varibales
data_vars = data.data_vars
row_col = ncols_nrows_from_N(len(data_vars))

fig, axs = plt.subplots(
    layout="constrained", figsize=(20, 20), sharex=True, sharey=True, **row_col
)
axs = axs.flatten()

for idx, var in enumerate(data_vars):
    axs[idx].step(data_ccor.lag_years, np.abs(data_ccor[var]), label="turth")
    try:
        axs[idx].step(kalman_ccor.lag_years, np.abs(kalman_ccor[var]), label="recons")
    except:
        pass
    axs[idx].set_xlabel("lag in years")
    axs[idx].set_ylabel("correlation")
    axs[idx].set_title(var)
    axs[idx].legend()

axs[idx].set_ylim((0, 1))

fig.suptitle(
    "CiCMOD | Absolute value lagged correlation of latent against climate indices."
)
save_fig(fig, "CiCMOD_latent_indices_lagged_corr_abs.png", dpi=400)


# %% [markdown]
# ### Perform frequency analyis on input and ouptut data

# %% Frequency analysis of all state variables.
data_vars = state_variables
fig, axs = plt.subplots(
    layout="constrained",
    figsize=(12, 4),
    sharex=True,
    sharey=True,
    ncols=len(data_vars),
)
axs = axs.flatten()

for idx, var in enumerate(data_vars):
    ax = axs[idx]
    xf, yf, yf_plot, f_min, f_max = compute_fft_spectrum(
        time=data.time, signal=data[var].values
    )
    ax.loglog(xf, yf_plot, label="truth", alpha=0.7)
    (xf, yf, yf_plot, f_min, f_max) = compute_fft_spectrum(
        time=kalman_states.time, signal=kalman_states[var].values
    )
    ax.loglog(xf, yf_plot, label="reconst.", alpha=0.7)

    f_min = 1 / 200  #  years^{-1}
    ax.set_xlim((f_min, f_max))
    xticks = ax.get_xticks().copy()  # 1/years
    new_ticks = np.round(1 / xticks, decimals=2)  # years
    ax.set_xticks(ticks=xticks, labels=new_ticks)
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlim((f_min, f_max))

    ax.set_xlabel(r"Period in years")
    ax.set_ylabel("Power in ????")
    ax.set_title(var)
    ax.legend()

ax.set_ylim(bottom=10 ** (-5.5), top=10 ** (0.5))
fig.suptitle("CiCMOD | Frequency spectrum state varibales")
save_fig(fig, "CiCMOD_fft_states.png", dpi=400)
