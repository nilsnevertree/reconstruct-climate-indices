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
    ncols_nrows_from_N,
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
run_id = "3b52871ac30a48ef9abeb7b413b47d30"
SubdataPath = "amo-oscillator-time-slice-experiment"
RunPath = Path("../data/") / SubdataPath / run_id
InputPath = RunPath / (run_id + "_input.nc")
KalmanPath = RunPath / (run_id + "_kalman.nc")
ParameterSettingsPath = RunPath / (run_id + "_parameter_settings.yml")
KalmanSettingsPath = RunPath / (run_id + "_kalman_settings.yml")
experiments = xr.open_dataset(InputPath)
experiments_kalman = xr.open_dataset(KalmanPath)
experiments_kalman = experiments_kalman.transpose("time", "kalman_itteration", ...)
experiments_kalman_states = pipeline.from_standard_dataset(experiments_kalman)
# read settings gile
# Read YAML file
with open(ParameterSettingsPath, "r") as stream:
    parameter_settings = yaml.safe_load(stream)
with open(KalmanSettingsPath, "r") as stream:
    kalman_settings = yaml.safe_load(stream)

mod_arg_1 = "time_slices"
# cenvert to slices
parameter_settings["time_slices"] = [
    slice(s["start"], s["stop"]) for s in parameter_settings["time_slices"]
]

ObservartionVariables = kalman_settings["ObservartionVariables"]
N = len(parameter_settings[mod_arg_1])
ncols_nrows = ncols_nrows_from_N(N)

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


# %% Deterministic evolution
fig, axs = plt.subplots(figsize=(15, 15), layout="constrained", **ncols_nrows)
axs = axs.flatten()
for i, mod1 in tqdm(enumerate(parameter_settings[mod_arg_1])):
    select_dict = {
        "time": mod1,
    }
    truth = experiments.sel(select_dict)
    axs[i].set_title(f"{mod_arg_1}: {mod1}")
    axs[i].plot(truth.time_years, truth["AMO"], label="AMO")
    axs[i].plot(truth.time_years, truth["ZOT"], label="ZOT")
    axs[i].set_ylabel("value")
    axs[i].set_xlabel("years")
    axs[i].legend()

fig.suptitle(f"Deterministic variables | Variations of {mod_arg_1}")
fig.tight_layout()

save_fig(fig, "svgs\deterministic-evolution.svg")
save_fig(fig, "deterministic-evolution.png", dpi=400)

# %% Deterministic relation
fig, axs = plt.subplots(figsize=(15, 15), layout="constrained", **ncols_nrows)
axs = axs.flatten()
for i, mod1 in tqdm(enumerate(parameter_settings[mod_arg_1])):
    select_dict = {
        "time": mod1,
    }
    truth = experiments.sel(select_dict)
    axs[i].set_title(f"{mod_arg_1}: {mod1}")
    axs[i].plot(
        truth["AMO"],
        truth["ZOT"],
        linestyle="-",
        linewidth=0.5,
        marker=".",
        alpha=0.7,
    )
    axs[i].set_ylabel("ZOT")
    axs[i].set_xlabel("AMO")

fig.suptitle(f"ZOT and AMO relation | Variations of {mod_arg_1} ")
fig.tight_layout()

save_fig(fig, "svgs\deterministic-relation.svg")
save_fig(fig, "deterministic-relation.png", dpi=400)

# %% Stochastic evolution
fig, axs = plt.subplots(figsize=(15, 15), layout="constrained", **ncols_nrows)
axs = axs.flatten()
for i, mod1 in tqdm(enumerate(parameter_settings[mod_arg_1])):
    select_dict = {
        "time": mod1,
    }
    truth = experiments.sel(select_dict)
    axs[i].plot(
        truth.time_years,
        truth["NAO"],
        label="NAO",
        alpha=0.7,
    )
    axs[i].plot(
        truth.time_years,
        truth["EAP"],
        label="EAP",
        alpha=0.7,
    )
    axs[i].set_title(f"{mod_arg_1}: {mod1}")
    axs[i].set_ylabel("value")
    axs[i].set_xlabel("years")
    axs[i].legend()

fig.suptitle(f"Stochstic variables | Variations of {mod_arg_1} ")
fig.tight_layout()

save_fig(fig, "svgs\stochastic-evolution.svg")
save_fig(fig, "stochastic-evolution.png", dpi=400)

# %%
fig, axs = plt.subplots(figsize=(15, 15), layout="constrained", **ncols_nrows)
axs = axs.flatten()
for i, mod1 in tqdm(enumerate(parameter_settings[mod_arg_1])):
    select_dict = {
        "time": mod1,
    }
    kalman_select_dict = {"start_time": mod1, **select_dict}
    reconst = experiments_kalman.sel(kalman_select_dict)
    axs[i].plot(reconst["kalman_itteration"], reconst["log_likelihod"])
    axs[i].set_ylabel("log_likelihod")
    axs[i].set_xlabel("kalman iteration")

fig.suptitle(f"Log Likelihood | ObservaVariation of {mod_arg_1} ")


save_fig(fig, "svgs\loglikelihood.svg")
save_fig(fig, "loglikelihood.png", dpi=400)

# %%
fig, axs = plt.subplots(figsize=(15, 15), layout="constrained", **ncols_nrows)
axs = axs.flatten()
handles = dict()
for i, mod1 in tqdm(enumerate(parameter_settings[mod_arg_1])):
    select_dict = {
        "time": mod1,
    }
    kalman_select_dict = {"start_time": mod1, **select_dict}
    truth = experiments.sel(select_dict)
    reconst = experiments_kalman_states.sel(kalman_select_dict)
    # plot AMO
    handles["AMO truth"] = axs[i].plot(
        truth.time_years,
        normalize(truth["AMO"], method="mean"),
        label="AMO truth",
        alpha=0.7,
    )
    try:
        # set same color as in AMO turth but darker
        color = adjust_lightness(handles["AMO truth"][0].get_color())
        handles["AMO"] = axs[i].plot(
            reconst.time_years,
            normalize(reconst["AMO"], method="mean").T,
            label="AMO",
            color=color,
            alpha=0.7,
        )
    except Exception as e:
        pass
    # plot ZOT
    handles["ZOT truth"] = axs[i].plot(
        truth.time_years,
        normalize(truth["ZOT"], method="mean"),
        label="ZOT truth",
        alpha=0.7,
    )
    try:
        # set same color as in AMO turth but darker
        color = adjust_lightness(handles["ZOT truth"][0].get_color())
        handles["ZOT"] = axs[i].plot(
            reconst.time_years,
            normalize(reconst["ZOT"], method="mean"),
            label="ZOT",
            color=color,
            alpha=0.7,
        )
    except Exception as e:
        pass
    # plot latent
    handles["latent"] = axs[i].plot(
        reconst.time_years,
        normalize(reconst["latent"], method="mean"),
        label="latent",
        alpha=0.7,
    )

    axs[i].set_title(f"{mod_arg_1}: {mod1}")
    axs[i].set_ylabel("value")
    axs[i].set_xlabel("years")

# create a flat list from the handles dict
handles = list(itertools.chain.from_iterable(handles.values()))

fig.suptitle(
    f"Deterministic Variables KalmanSEM result | ObservaVariation of {mod_arg_1}  "
)
fig.legend(
    handles=handles,
    loc=7,
    markerscale=3,
)

# save_fig(fig, "svgs\deterministic-evolution-kalman.svg")
save_fig(fig, "deterministic-evolution-kalman.png", dpi=400)

# %%
fig, axs = plt.subplots(figsize=(15, 15), layout="constrained", **ncols_nrows)
axs = axs.flatten()
handles = dict()
for i, mod1 in tqdm(enumerate(parameter_settings[mod_arg_1])):
    select_dict = {
        "time": mod1,
    }
    kalman_select_dict = {"start_time": mod1, **select_dict}
    truth = experiments.sel(select_dict)
    reconst = experiments_kalman_states.sel(kalman_select_dict)
    time_years = reconst.time_years  # plot NAO
    handles["NAO truth"] = axs[i].plot(
        time_years,
        normalize(truth["NAO"], method="mean"),
        label="NAO truth",
        alpha=0.7,
    )
    try:
        # set same color as in AMO turth but darker
        color = adjust_lightness(handles["AMO truth"][0].get_color())
        handles["ANO"] = axs[i].plot(
            time_years,
            normalize(reconst["NAO"], method="mean"),
            label="NAO",
            color=color,
            alpha=0.7,
        )
    except Exception as e:
        pass
    # plot  EAP
    handles["EAP truth"] = axs[i].plot(
        time_years,
        normalize(truth["EAP"], method="mean"),
        label="EAP truth",
        alpha=0.7,
    )
    try:
        # set same color as in AMO turth but darker
        color = adjust_lightness(handles["EAP truth"][0].get_color())
        handles["EAP"] = axs[i].plot(
            time_years,
            normalize(reconst["EAP"], method="mean"),
            label="EAP",
            color=color,
            alpha=0.7,
        )
    except Exception as e:
        pass
    # plot latent
    handles["latent"] = axs[i].plot(
        time_years,
        normalize(reconst["latent"], method="mean"),
        label="latent",
        alpha=0.7,
    )

    axs[i].set_title(f"{mod_arg_1}: {mod1}")
    axs[i].set_ylabel("value")
    axs[i].set_xlabel("years")
    # axs[i,j].legend()

axs[i].legend()

fig.suptitle(
    f"Stochastic Variables KalmanSEM result | ObservaVariation of {mod_arg_1}  "
)
fig.tight_layout()

# save_fig(fig, "svgs\deterministic-evolution-kalman.svg")
save_fig(fig, "stochastic-evolution-kalman.png", dpi=400)

# %%
fig, axs = plt.subplots(figsize=(15, 15), layout="constrained", **ncols_nrows)
axs = axs.flatten()
handles = dict()
for i, mod1 in tqdm(enumerate(parameter_settings[mod_arg_1])):
    select_dict = {
        "time": mod1,
    }
    kalman_select_dict = {"start_time": mod1, **select_dict}
    truth = normalize(experiments.sel(select_dict))
    reconst = normalize(experiments_kalman_states.sel(kalman_select_dict))
    time_years = reconst.time_years  # plot NAO
    for k, state in enumerate(ObservartionVariables):
        handles[f"{k}"] = axs[i].scatter(
            reconst[state],
            truth[state],
            marker=".",
            alpha=0.5,
            label=state,
        )
    axs[i].set_title(f"{mod_arg_1}: {mod1}")
    axs[i].set_xlabel("truth")
    axs[i].set_ylabel("kalman")
    # axs[i,j].legend()

# create a flat list from the handles dict
handles = handles.values()

fig.suptitle(f"Truth against KalmanSEM result | ObservaVariation of {mod_arg_1}  ")
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
fig, axs = plt.subplots(figsize=(15, 15), layout="constrained", **ncols_nrows)
axs = axs.flatten()
handles = dict()
for i, mod1 in tqdm(enumerate(parameter_settings[mod_arg_1])):
    select_dict = {
        "time": mod1,
    }
    kalman_select_dict = {"start_time": mod1, **select_dict}
    truth = normalize(experiments.sel(select_dict))
    reconst = normalize(experiments_kalman_states.sel(kalman_select_dict))
    for k, state in enumerate(experiments.data_vars):
        corr = xr.corr(truth[state], reconst["latent"])
        axs[i].scatter(
            truth[state],
            reconst["latent"],
            marker=".",
            alpha=np.abs(corr.values),
            label=f"{state} : {corr:.2f}",
        )
    axs[i].set_title(f"{mod_arg_1}: {mod1}")
    axs[i].set_xlabel("truth")
    axs[i].set_ylabel("latent | kalman")
    axs[i].legend(
        markerscale=3,
        handler_map=handler_map_alpha(),
    )


fig.suptitle(f"Truth against Latent Variable | ObservaVariation of {mod_arg_1}  ")
# save_fig(fig, "svgs\deterministic-evolution-kalman.svg")
save_fig(fig, "Truth-against-LatentVariable-result.png", dpi=400)
