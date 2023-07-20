import argparse

from pathlib import Path


parser = argparse.ArgumentParser(description="Description see file.")
parser.add_argument(
    "--general_path",
    default=Path("data") / Path("setups_default") / "cmip6_setup.yaml",
    help="path relative to REPO_PATH to the general setup stored in a .yaml file",
)
parser.add_argument(
    "--mlflow_path",
    default=Path("data") / Path("setups_default") / "mlflow_setup.yaml",
    help="path relative to REPO_PATH to the mlflow setup stored in a .yaml file",
)

args = parser.parse_args()
print("Start imports.")
import itertools
import warnings

import numpy as np
import xarray as xr
import yaml

from kalman_reconstruction.pipeline import (
    add_random_variable,
    all_dims_as_choords,
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

print(f"processing_function is:{processing_function}")


print("------------\nStart tracking of the experiment!\n------------\n")
set_tracking_uri(RepoPath / MlflowPath)
with start_run(experiment_id=ExperimentID) as run:
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

    ResultDataPath = DataPath / SubdataPath / f"{run_name}"
    ResultDataPath.mkdir(parents=True, exist_ok=True)

    # Create file names to store the  different settings
    SettingsPath = ResultDataPath / f"{run_name}_setup.yaml"
    MlflowSettingsPath = ResultDataPath / f"{run_name}_mlflow_setup.yaml"
    KalmanSettingsPath = ResultDataPath / f"{run_name}_kalman_setup.yaml"
    InputFile = ResultDataPath / f"{run_name}_input.nc"
    KalmanFile = ResultDataPath / f"{run_name}_kalman.nc"

    # log all settings and file locations.
    log_params(kalman_setup)
    log_params(
        dict(
            KalmanFunction=processing_function.__name__,
            SettingsFile=SettingsPath.relative_to(RepoPath).as_posix(),
            KalmanSettingsFile=KalmanSettingsPath.relative_to(RepoPath).as_posix(),
            InputFile=InputFile.relative_to(RepoPath).as_posix(),
            KalmanFile=KalmanFile.relative_to(RepoPath).as_posix(),
        )
    )
    with open(SettingsPath, "w") as yaml_file:
        yaml.dump(general_setup, yaml_file, default_flow_style=False)
    with open(MlflowSettingsPath, "w") as yaml_file:
        yaml.dump(mlflow_setup, yaml_file, default_flow_style=False)
    with open(KalmanSettingsPath, "w") as yaml_file:
        yaml.dump(kalman_setup, yaml_file, default_flow_style=False)
    # log artifact of the settings
    log_artifact(SettingsPath.as_posix())
    log_artifact(KalmanSettingsPath.as_posix())

    #  LOAD THE MODEL DATA
    print("Load input files")
    list_datasets = []
    # for each indice import the corresponding file
    for indice in model_setup["indices"]:
        try:
            indice_setup = model_setup["indices"][indice]
            current_dataset = xr.open_dataset(
                DataPath
                / str(model_setup["parent_directory"])
                / str(model_setup["model_name"])
                / str(indice_setup["file_name"])
            )
            current_dataset = current_dataset.rename(
                {indice_setup["original_var_name"]: indice_setup["new_var_name"]}
            )
            list_datasets.append(current_dataset)
        except Exception as e:
            warnings.warn(
                f"An Exception for {indice} occurred while loading and converting the names of the datasets.\nThis indice will be ignored!"
            )
    model_dataset = xr.merge(list_datasets)
    # make sure all dims are also choords!
    model_dataset = all_dims_as_choords(model_dataset)
    print("Done!")
    print("Add random variables.")

    # Create random variables as needed
    rng_seed = random_setup["seed"]
    for random_var in random_setup["name_random_variables"]:
        rng = np.random.default_rng(seed=rng_seed)
        add_random_variable(
            ds=model_dataset,
            var_name=random_var,
            random_generator=rng,
            variance=random_setup["random_variance"],
        )
        rng_seed += 1

    print("Save input file.")
    model_dataset.to_netcdf(InputFile)
    print("Done!")

    # #### Run the ``xarray_Kalman_SEM`` function from the ``pipeline`` library on all experiment setups.
    #
    # The ``run_function_on_multiple_subdatasets`` function allows to run the input function on all ``subdatasets`` specified by the ``subdataset_selections``. In this case these selections are given by the ``experiment_settings``.
    subdataset_selections = [
        {f"{model_setup['dimension']}": idx}
        for idx in model_dataset[model_setup["dimension"]].values
    ]
    print(f"Are those the subdatasets you want to use?")
    print(subdataset_selections)
    try:
        assert "yes" == input("Write 'yes' any press Enter!\n")
    except:
        raise UserWarning(f"User stopped the code before tracking started.")

    print(
        f"Run {processing_function.__name__} for : {function_setup['func_kwargs']['nb_iter_SEM']} iterations."
    )

    kalman_SEM_result = run_function_on_multiple_subdatasets(
        processing_function=processing_function,
        parent_dataset=model_dataset,
        subdataset_selections=subdataset_selections,
        func_args=function_setup["func_args"],
        func_kwargs=function_setup["func_kwargs"],
    )
    print("Done!")
    # ---- Save Files ----
    print("Save kalman file.")
    kalman_SEM_result.to_netcdf(KalmanFile)
    print("Done!")

end_run()
print("------------\nTracking Done!------------\n")
print(f"ExperimentID : {ExperimentID}")
print(f"RunName : {run_name}")
print(f"RunID : {run_id}")
