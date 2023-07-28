# # Track parameter Experiments
#
# ### Track the experiment using ``mlflow``
# Using mlflow, the following information will be stored:
# - Dataset containing the results from the ``AMO_oscillatory_ocean`` for all experiment settings.
# - Dataset containing the results from the ``xarray_Kalman_SEM`` for all experiment settings.
# - Settings to create the different Model runs using (``AMO_oscillatory_ocean``).
# - Settings used by the ``xarray_Kalman_SEM``.
#
# Therefore multiple setting for mlflow will need to be set by the User:
# - ExperimentID : Corresponds to the experiment_id used by ``mlflow`` to set the ``set_tracking_uri``.
# - SubdataPath : Name of the directory in which to store the results. This will be a child of the ``data`` directory.
# - MlflowPath : Name of the directory in which the mlflow tracking uri shall be used.
# - NOTE:
#     - Make sure that the RepoPath is correct!
#     - Make sure that ExperimentID exists!

# The folder structure will be :
#
# **Folder structure**
#
#     └───RepoPath
#         └───data
#             └───SubdataPath
#                 └───run_name
#                     │    run_name_input.nc
#                     │    run_name_kalman.nc
#                     │    run_name_kalman_setup.yml
#                     │    run_name_parameter_setup.yml
# Where ``run_name`` is e.g. *553cbd3bc6ce44028c8daad12647c306*
#


import argparse

from pathlib import Path


parser = argparse.ArgumentParser(description="Description see file.")
parser.add_argument(
    "--general_path",
    default=Path("data") / Path("setups_default") / "general_setup.yaml",
    help="path relative to REPO_PATH to the general setup stored in a .yaml file",
)
parser.add_argument(
    "--mlflow_path",
    default=Path("data") / Path("setups_default") / "mlflow_setup.yaml",
    help="path relative to REPO_PATH to the mlflow setup stored in a .yaml file",
)

args = parser.parse_args()

# if __name__ == "__main__":
print("Start imports.")

import copy
import itertools
import warnings

import numpy as np
import xarray as xr
import yaml

from kalman_reconstruction.pipeline import (
    add_random_variable,
    expand_and_assign_coords,
    run_function_on_multiple_subdatasets,
)
from mlflow import end_run, log_artifact, log_params, set_tracking_uri, start_run
from tqdm import tqdm

print("Done!")

# Verify the Path
ThisPath = Path(__file__)
RepoPath = ThisPath.parent.parent.parent
print(
    f"Is this the Repository Path correct?:\n{RepoPath}\nNote: The settings files are NOT yet loaded!"
)
try:
    assert "yes" == input("Write 'yes' any press Enter!\n")
except:
    raise UserWarning(
        f"User stopped the code due to incorrect Repository Path\n{RepoPath}"
    )

print("------------\nLoad Settings from files!\n------------\n")
general_path = RepoPath / Path(args.general_path)
mlflow_path = RepoPath / Path(args.mlflow_path)

# LOAD THE MLFLOW SETUP FILES
with open(mlflow_path, "r") as stream:
    try:
        MLFLOW_SETUP = yaml.safe_load(stream)
        mlflow_setup = MLFLOW_SETUP["mlflow_setup"]
        MlflowPath = mlflow_setup["mlflow_path"]
        ExperimentID = mlflow_setup["experiment_id"]
        SubdataPath = mlflow_setup["subdata_path"]
    except yaml.YAMLError as exc:
        raise (exc)

# LOAD THE GENERAL SETUP
with open(general_path, "r") as stream:
    try:
        general_setup = yaml.safe_load(stream)
        model_setup = general_setup["model_setup"]
        kalman_setup = general_setup["kalman_setup"]
        random_setup = general_setup["random_setup"]
        function_setup = general_setup["function_setup"]
    except yaml.YAMLError as exc:
        raise (exc)
# load the corresponding function from the libraries:
#  IMPORT THE MODEL FUNCTION
try:
    model_function = model_setup["model_function"]
    if "AMO_oscillatory_ocean" in model_function:
        from reconstruct_climate_indices.idealized_ocean import (
            AMO_oscillatory_ocean as model_function,
        )
    elif "sponge_ocean" in model_function:
        from reconstruct_climate_indices.idealized_ocean import (
            sponge_ocean as model_function,
        )
    elif "oscillatory_ocean" in model_function:
        from reconstruct_climate_indices.idealized_ocean import (
            oscillatory_ocean as model_function,
        )
    else:
        raise ValueError(
            f"The provided model function {model_function} is not available"
        )
except Exception as e:
    raise e

#  IMPORT THE KALMAN FUNCTION
try:
    # NOTE: this needs to be updated manually available processing functions
    available_processing_functions = [
        "pipeline.xarray_Kalman_SEM",
        "pipeline.xarray_Kalman_SEM_time_dependent",
        "pipeline.xarray_Kalman_SEM_full_output",
    ]

    processing_function = kalman_setup["processing_function"]
    # Import the processing function
    if processing_function in ["pipeline.xarray_Kalman_SEM", "xarray_Kalman_SEM"]:
        from kalman_reconstruction.pipeline import (
            xarray_Kalman_SEM as processing_function,
        )
    elif processing_function in [
        "pipeline.xarray_Kalman_SEM_time_dependent",
        "xarray_Kalman_SEM_time_dependent",
    ]:
        from kalman_reconstruction.pipeline import (
            xarray_Kalman_SEM_time_dependent as processing_function,
        )
    elif processing_function in [
        "pipeline.xarray_Kalman_SEM_full_output",
        "xarray_Kalman_SEM_full_output",
    ]:
        from kalman_reconstruction.pipeline import (
            xarray_Kalman_SEM_full_output as processing_function,
        )
    else:
        raise ValueError(
            f"The provided model function {processing_function} is not an available function of\n{available_processing_functions}"
        )
except Exception as e:
    raise e

print(f"Model_function is:{model_function}")
print(f"processing_function is:{processing_function}")

def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))

# create all the experiment setups
experiment_setups = dict()

# Use factor multiplication for all arguments given in "modified_arguments"
if "modified_arguments" in model_setup:
    if isinstance(model_setup["modified_arguments"], list):
        for key in model_setup["modified_arguments"]:
            try:
                # make sure to not go into too much details
                experiment_setups[key] = np.round(
                    a=model_setup["default_settings"][key]
                    * np.array(model_setup["factors"]),
                    decimals=model_setup["numpy_round_factor"],
                )
            except Exception:
                warnings.warn(
                    f"An Exception for {key} occurred while using it from ``modified_arguments``.\nThis key will be ignored!"
                )
    else:
        warnings.warn(
            f"Type of 'modified_arguments' in 'model_setup' needs to be 'list' but is {type(model_setup['modified_arguments'])}"
        )
else:
    print("No 'modified_arguments' dictionary was provided in the ``model_setup``.")

# Use given list from model_setup for arguments given in "modified_settings".
if "modified_settings" in model_setup:
    if isinstance(model_setup["modified_settings"], dict):
        for key in model_setup["modified_settings"]:
            try:
                experiment_setups[key] = model_setup["modified_settings"][key]
            except Exception:
                warnings.warn(
                    f"An Exception for {key} occurred while using it from ``modified_settings``.\nThis key will be ignored!"
                )
    else:
        warnings.warn(
            f"Type of 'modified_settings' in 'model_setup' needs to be 'dict' but is {type(model_setup['modified_settings'])}"
        )
else:
    print("No 'modified_settings' dictionary was provided in the ``model_setup``.")

# If no change is given : raise error!
if not bool(experiment_setups):  # if dictornary is empty
    raise NotImplementedError(
        f"The experiment_setups seem to be empty.\nThis may be a results of not providing either of:\n- 'modified_arguments' with 'factors'\n- 'modified_settings'"
    )

print(f"Are those the experiment_setups you want to perform?:\n{experiment_setups}")
try:
    assert "yes" == input("Write 'yes' any press Enter!\n")
except:
    raise UserWarning(f"User stopped the code before tracking started.")

# all experiment settings are made up by the all combinations of the experiment setups lists.
experiment_settings = list(product_dict(**experiment_setups))

print("------------\nStart tracking of the experiment!\n------------\n")
set_tracking_uri(RepoPath / MlflowPath)
with start_run(experiment_id=ExperimentID) as run:
    # prepare the parameter_setup to indlude all arrays used in the experiment_setup
    parameter_setup = dict()
    parameter_setup.update(model_setup["default_settings"])
    parameter_setup.update(experiment_setups)
    for key in parameter_setup:
        try:
            parameter_setup[key] = parameter_setup[key].tolist()
        except:
            pass
    # set the tracking_uri
    # retrieve the run_name
    run_name = run.info.run_name
    run_id = run.info.run_id

    # print information about the run
    print("Information:")
    print(f"ExperimentID : {ExperimentID}")
    print(f"RunName : {run_name}")
    print(f"RunID : {run_id}")

    # update run_name and run_name into the mlflow_setups
    mlflow_setup["run_name"] = run_name
    mlflow_setup["run_id"] = run_id

    # Create Paths to the corresponding directories names
    DataPath = RepoPath / "data"
    SubdataPath = DataPath / SubdataPath / f"{run_name}"
    SubdataPath.mkdir(parents=True, exist_ok=True)

    # Create file names to store the  different settings
    SettingsPath = SubdataPath / f"{run_name}_setup.yaml"
    MlflowSettingsPath = SubdataPath / f"{run_name}_mlflow_setup.yaml"
    ParameterSettingsPath = SubdataPath / f"{run_name}_parameter_setup.yaml"
    KalmanSettingsPath = SubdataPath / f"{run_name}_kalman_setup.yaml"
    InputFile = SubdataPath / f"{run_name}_input.nc"
    KalmanFile = SubdataPath / f"{run_name}_kalman.nc"

    # log all settings and file locations.
    log_params(kalman_setup)
    log_params(parameter_setup)
    log_params(dict(Iterative=random_setup["iterative"]))
    log_params(
        dict(
            ModelFunction=model_function.__name__,
            KalmanFunction=processing_function.__name__,
            SettingsFile=SettingsPath.relative_to(RepoPath).as_posix(),
            ParameterSettingsFile=ParameterSettingsPath.relative_to(
                RepoPath
            ).as_posix(),
            KalmanSettingsFile=KalmanSettingsPath.relative_to(RepoPath).as_posix(),
            InputFile=InputFile.relative_to(RepoPath).as_posix(),
            KalmanFile=KalmanFile.relative_to(RepoPath).as_posix(),
        )
    )
    with open(SettingsPath, "w") as yaml_file:
        yaml.dump(general_setup, yaml_file, default_flow_style=False)
    with open(MlflowSettingsPath, "w") as yaml_file:
        yaml.dump(mlflow_setup, yaml_file, default_flow_style=False)
    with open(ParameterSettingsPath, "w") as yaml_file:
        yaml.dump(parameter_setup, yaml_file, default_flow_style=False)
    with open(KalmanSettingsPath, "w") as yaml_file:
        yaml.dump(kalman_setup, yaml_file, default_flow_style=False)
    # log artifact of the settings
    log_artifact(SettingsPath.as_posix())
    log_artifact(ParameterSettingsPath.as_posix())
    log_artifact(KalmanSettingsPath.as_posix())

    # ### Run the experiments:

    # #### Create all datasets by running the model function ``AMO_oscillatory_ocean``.
    #
    # The results will be combined into a single Dataset
    print("Create experiments")
    data_list = []
    setting = model_setup["default_settings"].copy()
    # we will not track each individual model run.
    expand_ds = xr.Dataset(
        coords=experiment_setups,
    )
    for sub_setting in tqdm(experiment_settings):
        # update the settings with the current set from the experiment settings.
        setting.update(**sub_setting)
        # integrate the model and store the output xr.Dataset
        data = model_function(**setting)
        data = expand_and_assign_coords(
            ds1=data, ds2=expand_ds, select_dict=sub_setting
        )
        data_list.append(data)
    # merge all output Dataset into a single Dataset
    experiments = xr.merge(data_list)
    print("Done!")
    print("Save experiments file.")
    experiments.to_netcdf(InputFile)
    print("Done!")

    # Create random variables as needed
    input_kalman = experiments.copy()

    # If user decided  to add random variables all at once (not iteratively)
    rng_seed = random_setup["seed"]
    if random_setup["iterative"] != True:
        for random_var in random_setup["name_random_variables"]:
            rng = np.random.default_rng(seed=rng_seed)
            add_random_variable(
                ds=input_kalman,
                var_name=random_var,
                random_generator=rng,
                variance=random_setup["random_variance"],
            )
            rng_seed += 1

        print(
            f"Run {processing_function.__name__} for : {function_setup['func_kwargs']['nb_iter_SEM']} iterations"
        )
        # #### Run the ``xarray_Kalman_SEM`` function from the ``pipeline`` library on all experiment setups.
        #
        # The ``run_function_on_multiple_subdatasets`` function allows to run the input function on all ``subdatasets`` specified by the ``subdataset_selections``. In this case these selections are given by the ``experiment_settings``.

        experiments_kalman = run_function_on_multiple_subdatasets(
            processing_function=processing_function,
            parent_dataset=input_kalman,
            subdataset_selections=experiment_settings,
            func_args=function_setup["func_args"],
            func_kwargs=function_setup["func_kwargs"],
        )
        print("Done!")
        # ---- Save Files ----
        print("Save kalman file.")
        experiments_kalman.to_netcdf(KalmanFile)
    else:
        # choose the base state variables as the ones not in the random variables
        base_state_variables = [
            var
            for var in kalman_setup["state_variables"]
            if var not in random_setup["name_random_variables"]
        ]
        base_function_setup = copy.deepcopy(function_setup)
        current_state_variables = base_state_variables
        # add the random variables iteratively
        for random_var in random_setup["name_random_variables"]:
            # update the random number generator

            rng = np.random.default_rng(seed=rng_seed)
            add_random_variable(
                ds=input_kalman,
                var_name=random_var,
                random_generator=rng,
                variance=random_setup["random_variance"],
            )
            rng_seed += 1
            # update the state_variables
            current_state_variables.append(random_var)
            current_function_setup = copy.deepcopy(base_function_setup)
            current_function_setup["func_kwargs"][
                "state_variables"
            ] = current_state_variables
            print(
                f"Run {processing_function.__name__} for : {function_setup['func_kwargs']['nb_iter_SEM']} iterations"
            )
            # #### Run the ``xarray_Kalman_SEM`` function from the ``pipeline`` library on all experiment setups.
            #
            # The ``run_function_on_multiple_subdatasets`` function allows to run the input function on all ``subdatasets`` specified by the ``subdataset_selections``. In this case these selections are given by the ``experiment_settings``.

            experiments_kalman = run_function_on_multiple_subdatasets(
                processing_function=processing_function,
                parent_dataset=input_kalman,
                subdataset_selections=experiment_settings,
                func_args=current_function_setup["func_args"],
                func_kwargs=current_function_setup["func_kwargs"],
            )
            print("Done!")
            # ---- Save Files ----
            print("Save kalman file.")
            experiments_kalman.to_netcdf(
                SubdataPath / f"{run_name}_{random_var}.nc"
            )
        experiments_kalman.to_netcdf(KalmanFile)
    print("Done!")

end_run()
print("------------\nTracking Done!------------\n")
print(f"ExperimentID : {ExperimentID}")
print(f"RunName : {run_name}")
print(f"RunID : {run_id}")
