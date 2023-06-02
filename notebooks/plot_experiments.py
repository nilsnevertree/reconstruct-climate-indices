# %%
import itertools

from pathlib import Path

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
# # Reload the data from filepaths

# %%
run_id = "b3270eb6d6914f3e92665d3216e1dae0"
SubdataPath = "parameter-experiments-storage"
RunPath = Path("../data/") / SubdataPath / run_id
InputPath = RunPath / (run_id + "_input.nc")
KalmanPath = RunPath / (run_id + "_kalman.nc")
ParameterSettingsPath = RunPath / (run_id + "_parameter_settings.yml")
KalmanSettingsPath = RunPath / (run_id + "_kalman_settings.yml")
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
ObservartionVariables = kalman_settings["ObservartionVariables"]

# %%
PATH_FIGURES = Path("../results/") / SubdataPath / run_id
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
        axs[i, j].plot(
            experiments.time_years, experiments["AMO"].sel(select_dict), label="AMO"
        )
        axs[i, j].plot(
            experiments.time_years, experiments["ZOT"].sel(select_dict), label="ZOT"
        )
        axs[i, j].set_ylabel("value")
        axs[i, j].set_xlabel("years")
        axs[i, j].legend()

fig.suptitle(f"Deterministic variables | Variations of {mod_arg_1} and {mod_arg_2}")
fig.tight_layout()

save_fig(fig, "svgs\deterministic-evolution.svg")
save_fig(fig, "deterministic-evolution.png", dpi=400)

# %%
fig, axs = plt.subplots(
    nrows=len(experiments[mod_arg_1]),
    ncols=len(experiments[mod_arg_2]),
    layout="constrained",
)
for i, mod1 in tqdm(enumerate(experiments[mod_arg_1])):
    for j, mod2 in enumerate(experiments[mod_arg_2]):
        select_dict = {
            mod_arg_1: mod1,
            mod_arg_2: mod2,
        }
        axs[i, j].set_title(f"{mod_arg_1}: {mod1:.2f}, {mod_arg_2}: {mod2:.2f}")
        axs[i, j].plot(
            experiments["AMO"].sel(select_dict),
            experiments["ZOT"].sel(select_dict),
            linestyle="-",
            linewidth=0.5,
            marker=".",
            alpha=0.7,
        )
        axs[i, j].set_ylabel("ZOT")
        axs[i, j].set_xlabel("AMO")

fig.suptitle(f"ZOT and AMO relation | Variations of {mod_arg_1} and {mod_arg_2}")
fig.tight_layout()

save_fig(fig, "svgs\deterministic-relation.svg")
save_fig(fig, "deterministic-relation.png", dpi=400)

# %%
fig, axs = plt.subplots(
    nrows=len(experiments[mod_arg_1]),
    ncols=len(experiments[mod_arg_2]),
    layout="constrained",
)
for i, mod1 in tqdm(enumerate(experiments[mod_arg_1])):
    for j, mod2 in enumerate(experiments[mod_arg_2]):
        select_dict = {
            mod_arg_1: mod1,
            mod_arg_2: mod2,
        }
        axs[i, j].plot(
            experiments.time_years,
            experiments["NAO"].sel(select_dict),
            label="NAO",
            alpha=0.7,
        )
        axs[i, j].plot(
            experiments.time_years,
            experiments["EAP"].sel(select_dict),
            label="EAP",
            alpha=0.7,
        )
        axs[i, j].set_title(f"{mod_arg_1}: {mod1:.2f}, {mod_arg_2}: {mod2:.2f}")
        axs[i, j].set_ylabel("value")
        axs[i, j].set_xlabel("years")
        axs[i, j].legend()

fig.suptitle(f"Stochstic variables | Variations of {mod_arg_1} and {mod_arg_2}")
fig.tight_layout()

save_fig(fig, "svgs\stochastic-evolution.svg")
save_fig(fig, "stochastic-evolution.png", dpi=400)

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
        reconst = experiments_kalman.sel(select_dict, method="nearest")
        axs[i, j].plot(reconst["kalman_itteration"], reconst["log_likelihod"])
        axs[i, j].set_ylabel("log_likelihod")
        axs[i, j].set_xlabel("kalman itteration")

fig.suptitle(f"Log Likelihood | ObservaVariation of {mod_arg_1} and {mod_arg_2}")


save_fig(fig, "svgs\loglikelihood.svg")
save_fig(fig, "loglikelihood.png", dpi=400)

# %%
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
        time_years = experiments.time_years
        truth = experiments.sel(select_dict, method="nearest")
        reconst = experiments_kalman_states.sel(select_dict, method="nearest")
        # plot AMO
        handles["AMO truth"] = axs[i, j].plot(
            time_years,
            normalize(truth["AMO"], method="mean"),
            label="AMO truth",
            alpha=0.7,
        )
        try:
            # set same color as in AMO turth but darker
            color = adjust_lightness(handles["AMO truth"][0].get_color())
            handles["AMO"] = axs[i, j].plot(
                time_years,
                normalize(reconst["AMO"], method="mean"),
                label="AMO",
                color=color,
                alpha=0.7,
            )
        except Exception as e:
            pass
        # plot ZOT
        handles["ZOT truth"] = axs[i, j].plot(
            time_years,
            normalize(truth["ZOT"], method="mean"),
            label="ZOT truth",
            alpha=0.7,
        )
        try:
            # set same color as in AMO turth but darker
            color = adjust_lightness(handles["ZOT truth"][0].get_color())
            handles["ZOT"] = axs[i, j].plot(
                time_years,
                normalize(reconst["ZOT"], method="mean"),
                label="ZOT",
                color=color,
                alpha=0.7,
            )
        except Exception as e:
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

# create a flat list from the handles dict
handles = list(itertools.chain.from_iterable(handles.values()))

fig.suptitle(
    f"Deterministic Variables KalmanSEM result | ObservaVariation of {mod_arg_1} and {mod_arg_2} "
)
fig.legend(
    handles=handles,
    loc=7,
    markerscale=3,
)

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
        truth = experiments.sel(select_dict, method="nearest")
        reconst = experiments_kalman_states.sel(select_dict, method="nearest")
        # plot NAO
        handles["NAO truth"] = axs[i, j].plot(
            time_years,
            normalize(truth["NAO"], method="mean"),
            label="NAO truth",
            alpha=0.7,
        )
        try:
            # set same color as in AMO turth but darker
            color = adjust_lightness(handles["AMO truth"][0].get_color())
            handles["ANO"] = axs[i, j].plot(
                time_years,
                normalize(reconst["NAO"], method="mean"),
                label="NAO",
                color=color,
                alpha=0.7,
            )
        except Exception as e:
            pass
        # plot  EAP
        handles["EAP truth"] = axs[i, j].plot(
            time_years,
            normalize(truth["EAP"], method="mean"),
            label="EAP truth",
            alpha=0.7,
        )
        try:
            # set same color as in AMO turth but darker
            color = adjust_lightness(handles["EAP truth"][0].get_color())
            handles["EAP"] = axs[i, j].plot(
                time_years,
                normalize(reconst["EAP"], method="mean"),
                label="EAP",
                color=color,
                alpha=0.7,
            )
        except Exception as e:
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

axs[i, j].legend()

fig.suptitle(
    f"Stochastic Variables KalmanSEM result | ObservaVariation of {mod_arg_1} and {mod_arg_2} "
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
        reconst = normalize(
            experiments_kalman_states.sel(select_dict, method="nearest")
        )
        truth = normalize(experiments.sel(select_dict, method="nearest"))
        for k, state in enumerate(ObservartionVariables):
            handles[f"{k}"] = axs[i, j].scatter(
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
    f"Truth against KalmanSEM result | ObservaVariation of {mod_arg_1} and {mod_arg_2} "
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
        reconst = normalize(
            experiments_kalman_states.sel(select_dict, method="nearest")
        )
        truth = normalize(experiments.sel(select_dict, method="nearest"))
        for k, state in enumerate(experiments.data_vars):
            corr = xr.corr(truth[state], reconst["latent"])
            axs[i, j].scatter(
                truth[state],
                reconst["latent"],
                marker=".",
                alpha=np.abs(corr.values),
                label=f"{state} : {corr:.2f}",
            )
        axs[i, j].set_title(f"{mod_arg_1}: {mod1:.2f}, {mod_arg_2}: {mod2:.2f}")
        axs[i, j].set_xlabel("truth")
        axs[i, j].set_ylabel("latent | kalman")
        axs[i, j].legend(
            markerscale=3,
            handler_map=handler_map_alpha(),
        )


fig.suptitle(
    f"Truth against Latent Variable | ObservaVariation of {mod_arg_1} and {mod_arg_2} "
)
# save_fig(fig, "svgs\deterministic-evolution-kalman.svg")
save_fig(fig, "Truth-against-LatentVariable-result.png", dpi=400)
