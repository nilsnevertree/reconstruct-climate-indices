# %%
import argparse

from pathlib import Path


parser = argparse.ArgumentParser(description="Description see file.")
parser.add_argument(
    "--run_name",
    help="Name of the mlflow run you want to plot",
)
args = parser.parse_args()
print("Start imports.")

import itertools

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml

from kalman_reconstruction import pipeline
from kalman_reconstruction.custom_plot import (
    adjust_lightness,
    handler_map_alpha,
    set_custom_rcParams,
)
from kalman_reconstruction.statistics import normalize
from tqdm import tqdm


set_custom_rcParams()
plt.rcParams["figure.figsize"] = (15, 15)
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# %% [markdown]
#  # Reload the data from filepaths

# %%
run_name = args.run_name
SubdataPath = "simplified_ocean_experiments"
RunPath = Path(__file__).parent.parent / "data" / SubdataPath / run_name
InputPath = RunPath / (run_name + "_input.nc")
KalmanPath = RunPath / (run_name + "_kalman.nc")
ParameterSettingsPath = RunPath / (run_name + "_parameter_setup.yaml")
KalmanSettingsPath = RunPath / (run_name + "_kalman_setup.yaml")
experiments = xr.open_dataset(InputPath)
experiments_kalman = xr.open_dataset(KalmanPath)
experiments_kalman_states = pipeline.from_standard_dataset(experiments_kalman)

# read settings gile
# Read YAML file
with open(ParameterSettingsPath, "r") as stream:
    parameter_settings = yaml.safe_load(stream)
with open(KalmanSettingsPath, "r") as stream:
    kalman_settings = yaml.safe_load(stream)

modified_arguments_dict = dict(
    [
        (key, value)
        for key, value in parameter_settings.items()
        if isinstance(value, list)
    ]
)
modified_arguments = list(modified_arguments_dict.keys())
mod_arg_1 = modified_arguments[0]
mod_arg_2 = modified_arguments[1]
observation_variables = kalman_settings["observation_variables"]


# %%
try:
    experiments = experiments.drop("random_forcing")
except Exception:
    pass


# %%
PATH_FIGURES = Path("../results/") / SubdataPath / run_name
SAVE_FIGURES = True


def save_fig(fig, relative_path, **kwargs):
    store_path = PATH_FIGURES / relative_path
    store_path.parent.mkdir(parents=True, exist_ok=True)
    if SAVE_FIGURES:
        fig.savefig(store_path, **kwargs)
    else:
        pass


# %%
fig, axs = plt.subplots(
    nrows=len(experiments[mod_arg_1]),
    ncols=len(experiments[mod_arg_2]),
    figsize=(15, 15),
)
for i, mod1 in tqdm(enumerate(experiments[mod_arg_1])):
    for j, mod2 in enumerate(experiments[mod_arg_2]):
        select_dict = {
            mod_arg_1: mod1,
            mod_arg_2: mod2,
        }
        axs[i, j].set_title(f"{mod_arg_1}: {mod1:.2f}, {mod_arg_2}: {mod2:.2f}")
        for var in experiments.data_vars:
            axs[i, j].plot(
                experiments.time_years, experiments[var].sel(select_dict), label=var
            )
        axs[i, j].set_ylabel("value")
        axs[i, j].set_xlabel("years")
        axs[i, j].legend()

fig.suptitle(f"All variables | Variations of {mod_arg_1} and {mod_arg_2}")
fig.tight_layout()

# save_fig(fig, "svgs\deterministic-evolution.svg")
save_fig(fig, "all-evolution.png", dpi=400)


# %%
fig, axs = plt.subplots(
    nrows=len(experiments_kalman_states[mod_arg_1]),
    ncols=len(experiments_kalman_states[mod_arg_2]),
    layout="constrained",
)
for i, mod1 in tqdm(enumerate(experiments[mod_arg_1])):
    for j, mod2 in enumerate(experiments[mod_arg_2]):
        select_dict = {
            mod_arg_1: mod1,
            mod_arg_2: mod2,
        }
        reconst = experiments_kalman.sel(select_dict)
        axs[i, j].plot(reconst["kalman_itteration"], reconst["log_likelihod"])
        axs[i, j].set_ylabel("log_likelihod")
        axs[i, j].set_xlabel("kalman itteration")

fig.suptitle(f"Log Likelihood | Variation of {mod_arg_1} and {mod_arg_2}")


save_fig(fig, "svgs\loglikelihood.svg")
save_fig(fig, "loglikelihood.png", dpi=400)


# %%
experiments_kalman_states = normalize(experiments_kalman_states)
fig, axs = plt.subplots(
    nrows=len(experiments_kalman_states[mod_arg_1]),
    ncols=len(experiments_kalman_states[mod_arg_2]),
    layout="constrained",
)
handles = dict()
for i, mod1 in tqdm(enumerate(experiments[mod_arg_1])):
    for j, mod2 in enumerate(experiments[mod_arg_2]):
        select_dict = {
            mod_arg_1: mod1,
            mod_arg_2: mod2,
        }
        truth = experiments.sel(select_dict)
        reconst = experiments_kalman_states.sel(select_dict)
        time_years = reconst.time_years
        # plot SAT
        handles["SAT truth"] = axs[i, j].plot(
            time_years,
            normalize(truth["SAT"], method="mean"),
            label="SAT truth",
            alpha=0.7,
        )
        # set same color as in oscillator_sea_surface_temperature turth but darker
        color = adjust_lightness(handles["SAT truth"][0].get_color())
        handles["SAT"] = axs[i, j].plot(
            time_years,
            normalize(reconst["SAT"], method="mean"),
            label="SAT",
            color=color,
            alpha=0.7,
        )
        # plot latent
        handles["latent"] = axs[i, j].plot(
            time_years,
            normalize(reconst["latent"], method="mean"),
            label="latent",
            alpha=0.7,
        )

        axs[i, j].set_title(f"{mod_arg_1}: {mod1:.2f}, {mod_arg_2}: {mod2:.2f}")
        axs[i, j].set_ylabel("value")
        axs[i, j].set_xlabel("years")
        # axs[i,j].legend()

# create a flat list from the handles dict
handles = list(itertools.chain.from_iterable(handles.values()))

fig.legend(
    handles=handles,
    loc=7,
    markerscale=3,
)

fig.suptitle(
    f"Surface air tempearture and latent variable | Variation of {mod_arg_1} and {mod_arg_2} "
)
fig.tight_layout()

# save_fig(fig, "svgs\deterministic-evolution-kalman.svg")
save_fig(fig, "stochastic-evolution-kalman.png", dpi=400)


# %%
experiments_kalman_states = normalize(experiments_kalman_states)
fig, axs = plt.subplots(
    nrows=len(experiments_kalman_states[mod_arg_1]),
    ncols=len(experiments_kalman_states[mod_arg_2]),
    layout="constrained",
)
handles = dict()
for i, mod1 in tqdm(enumerate(experiments[mod_arg_1])):
    for j, mod2 in enumerate(experiments[mod_arg_2]):
        select_dict = {
            mod_arg_1: mod1,
            mod_arg_2: mod2,
        }
        truth = experiments.sel(select_dict)
        reconst = experiments_kalman_states.sel(select_dict)
        time_years = reconst.time_years
        # plot SST
        handles["SST truth"] = axs[i, j].plot(
            time_years,
            normalize(truth["SST"], method="mean"),
            label="SST truth",
            alpha=0.7,
        )
        # set same color as in oscillator_sea_surface_temperature turth but darker
        color = adjust_lightness(handles["SST truth"][0].get_color())
        handles["SST"] = axs[i, j].plot(
            time_years,
            normalize(reconst["SST"], method="mean"),
            label="SST",
            color=color,
            alpha=0.7,
        )
        try:
            # plot DOT
            handles["DOT truth"] = axs[i, j].plot(
                time_years,
                normalize(truth["DOT"], method="mean"),
                label="DOT truth",
                alpha=0.7,
            )
        except:
            pass
        try:
            # set same color as in oscillator_sea_surface_temperature turth but darker
            color = adjust_lightness(handles["DOT truth"][0].get_color())
            handles["DOT"] = axs[i, j].plot(
                time_years,
                normalize(reconst["DOT"], method="mean"),
                label="DOT",
                color=color,
                alpha=0.7,
            )
        except:
            pass
        # plot latent
        handles["latent"] = axs[i, j].plot(
            time_years,
            normalize(reconst["latent"], method="mean"),
            label="latent",
            alpha=0.7,
        )

        axs[i, j].set_title(f"{mod_arg_1}: {mod1:.2f}, {mod_arg_2}: {mod2:.2f}")
        axs[i, j].set_ylabel("value")
        axs[i, j].set_xlabel("years")
        # axs[i,j].legend()

# create a flat list from the handles dict
handles = list(itertools.chain.from_iterable(handles.values()))

fig.legend(
    handles=handles,
    loc=7,
    markerscale=3,
)

fig.suptitle(
    f"Deterministic Variables KalmanSEM result | Variation of {mod_arg_1} and {mod_arg_2} "
)
fig.tight_layout()

# save_fig(fig, "svgs\deterministic-evolution-kalman.svg")
save_fig(fig, "deterministic-evolution-kalman.png", dpi=400)


# %%
experiments_kalman_states = normalize(experiments_kalman_states)
fig, axs = plt.subplots(
    nrows=len(experiments_kalman_states[mod_arg_1]),
    ncols=len(experiments_kalman_states[mod_arg_2]),
    layout="constrained",
)
handles = dict()
for i, mod1 in tqdm(enumerate(experiments[mod_arg_1])):
    for j, mod2 in enumerate(experiments[mod_arg_2]):
        select_dict = {
            mod_arg_1: mod1,
            mod_arg_2: mod2,
        }
        reconst = normalize(experiments_kalman_states.sel(select_dict))
        truth = normalize(experiments.sel(select_dict))
        for k, state in enumerate(observation_variables):
            handles[k] = axs[i, j].scatter(
                reconst[state],
                truth[state],
                marker=".",
                alpha=0.5,
                label=state,
            )
        axs[i, j].set_title(f"{mod_arg_1}: {mod1:.2f}, {mod_arg_2}: {mod2:.2f}")
        axs[i, j].set_xlabel("truth")
        axs[i, j].set_ylabel("kalman")
        # axs[i,j].legend()

# create a flat list from the handles dict
handles = handles.values()

fig.suptitle(
    f"Truth against KalmanSEM result | Variation of {mod_arg_1} and {mod_arg_2} "
)
fig.legend(
    handles=handles,
    loc=7,
    markerscale=3,
    handler_map=handler_map_alpha(),
)
fig.tight_layout()
# save_fig(fig, "svgs\deterministic-evolution-kalman.svg")
save_fig(fig, "Truth-against-KalmanSEM-result.png", dpi=400)


# %%
experiments_kalman_states = normalize(experiments_kalman_states)
fig, axs = plt.subplots(
    nrows=len(experiments_kalman_states[mod_arg_1]),
    ncols=len(experiments_kalman_states[mod_arg_2]),
    layout="constrained",
)
for i, mod1 in tqdm(enumerate(experiments[mod_arg_1])):
    for j, mod2 in enumerate(experiments[mod_arg_2]):
        select_dict = {
            mod_arg_1: mod1,
            mod_arg_2: mod2,
        }
        reconst = normalize(experiments_kalman_states.sel(select_dict))
        truth = normalize(experiments.sel(select_dict))
        for k, state in enumerate(experiments.data_vars):
            try:
                corr = xr.corr(truth[state], reconst["latent"])
                axs[i, j].scatter(
                    truth[state],
                    reconst["latent"],
                    marker=".",
                    alpha=np.abs(corr.values),
                    label=f"{state} : {corr:.2f}",
                )
            except:
                pass
        axs[i, j].set_title(f"{mod_arg_1}: {mod1:.2f}, {mod_arg_2}: {mod2:.2f}")
        axs[i, j].set_xlabel("truth")
        axs[i, j].set_ylabel("latent | kalman")
        axs[i, j].legend(
            markerscale=3,
            handler_map=handler_map_alpha(),
        )


fig.suptitle(
    f"Truth against Latent Variable | Variation of {mod_arg_1} and {mod_arg_2} "
)
# save_fig(fig, "svgs\deterministic-evolution-kalman.svg")
save_fig(fig, "Truth-against-LatentVariable-result.png", dpi=400)