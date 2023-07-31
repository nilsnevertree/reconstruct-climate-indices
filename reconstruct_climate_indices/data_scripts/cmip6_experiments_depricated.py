"""
This script allows to run multiple experiments with different settings on the CMIP6 data.
For each experiment the Kalman-SEM is applied using the ``xarray_Kalman_SEM`` functions from the ``kalman_reconstruction`` library.
THe input data, result data and settings are stored in the directory specified by the ``mlflow_setup`` dictionary in the ``mlflow_setup.yaml`` file.

The script is designed to be run from the command line using the ``python parameter_experiments.py`` command.
The script will then ask the user to confirm the experiment setups and then start the tracking of the experiment using ``mlflow``.
The mlflow library is used to track the experiment.
For each mlflow run initialized by this script, the following data will be stored in the directory given by the ``mlflow_setup`` dictionary in the ``mlflow_setup.yaml`` file:
- The input Dataset created by the idealized model function chosen in the general_setup.yaml file.
- The output Dataset created by the Kalman-SEM function chosen in the general_setup.yaml file.
- A copy of the ``general_setup.yaml`` provided to the script.

Example:
run_name = "sassy-zebra-42"             # This name is automatically generated by mlflow for each run.
subdata_path = "your_subdata_path"

Folder structure
'''
    └───RepoPath
        └───data
            └───scripts
                │   parameter_experiments.py
                │   ...
            └───setups_default
                │   mlflow_setup.yaml
                │   general_setup.yaml
                │   ...
            └───your_subdata_path                               # This path is given by the ``mlflow_setup`` dictionary in the ``mlflow_setup.yaml`` file.
                └───sassy-zebra-42                              # This name is automatically generated by mlflow.
                    │    sassy-zebra-42_input.nc                # This file contains the input Dataset created by the model function and used as input by the Kalman-SEM function.
                    │    sassy-zebra-42_kalman.nc               # This file contains the output Dataset created by the Kalman-SEM function.
                    │    sassy-zebra-42_general_setup.yaml      # This file can be used to reproduce the experiment!
                    │    ...
'''

In the ``mlflow_setup.yaml`` file the user can set the following information:
    - ``mlflow_path`` : The path to the directory in which the mlflow tracking uri shall be used.
    - ``experiment_id`` : The experiment_id used by ``mlflow`` to set the ``set_tracking_uri``.
    - ``experiment_name`` : The name of the experiment used by ``mlflow`` to set the ``set_tracking_uri``.
    - ``subdata_path`` : The name of the directory in which to store the results. This will be a child of the ``data`` directory.

To modify the setup of the experiment, the user can modify the ``general_setup.yaml`` file passed to the script.
The ``general_setup.yaml`` file contains the following information:
- ``model_setup`` : Contains the settings for the model function.
    - ``model_name`` : The name of the Earth-System-Model to be used.
    - ``parent_directory`` : The name of the parent directory in which the data of the Earth-System-Model is stored.
    - ``dimension`` : The name of the dimension for which the Kalman-SEM function shall be applied for each value.
    - ``indices`` : A dictionary containing the settings for each indice to be used.
        - ``indice_name`` : The name of the indice to be used.
        - ``file_name`` : The name of the file containing the data for the indice to be used.
        - ``original_var_name`` : The name of the variable in the file.
        - ``new_var_name`` : The name of the variable in the Dataset created by the model function.

- ``kalman_setup`` : Contains the settings for the Kalman-SEM function.
    - ``processing_function`` : The Kalman-SEM function to be used. Currently the following functions are available:
        - ``xarray_Kalman_SEM`` : The Kalman-SEM function from the kalman_reconstruction library.
        - ``xarray_Kalman_SEM_time_dependent`` : The Kalman-SEM function from the kalman_reconstruction library.
        - ``xarray_Kalman_SEM_full_output`` : The Kalman-SEM function from the kalman_reconstruction library.
    - ``state_variables`` : The state variables to be used by the Kalman-SEM function.
    - ``observation_variables`` : The observation variables to be used by the Kalman-SEM function.
- ``random_setup`` : Contains the settings for the random variables.
    - ``name_random_variables`` : The names of the random variables to be added.
    - ``random_variance`` : The variance of the random variables to be added.
    - ``iterative`` : If True, the random variables will be added iteratively. If False, all random variables will be added at once.
        This means after addition of each random variable, the Kalman-SEM function will be applied.
        A new child run will be created for each random variable by mlflow.
    - ``seed`` : The seed to be used for the random number generator.

- ``function_setup`` : Contains the settings for the Kalman-SEM function.
    - ``func_args`` : The arguments of the Kalman-SEM function.
    - ``func_kwargs`` : The settings of the Kalman-SEM function.


NOTE:
    - Make sure that the RepoPath is correct!
    - Make sure that ExperimentID exists!
    - Make sure that the mlflow_setup.yaml file is correct!
    - Make sure that the general_setup.yaml file is correct!
    - If iterative is True, for each random variable a new child run will be created by mlflow.
      The results of each child run will be stored in a own directory in the same subdata_path directory as the parent run.

"""

import argparse

from pathlib import Path

def main():
    """Run the script."""
    def get_first_lines(docstring, line_number):
        """Return the first lines of a docstring."""
        if not docstring:
            return ""

        lines = docstring.strip().splitlines()
        return "\n".join(lines[:line_number])


    parser = argparse.ArgumentParser(
        description=get_first_lines(__doc__, 4),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

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

    from typing import Any, Dict, List, Optional, Tuple, Union

    import numpy as np
    import xarray as xr
    import yaml

    from kalman_reconstruction.pipeline import (
        add_random_variable,
        all_dims_as_choords,
        expand_and_assign_coords,
        from_standard_dataset,
        run_function_on_multiple_subdatasets,
    )
    from mlflow import end_run, log_artifact, log_params, set_tracking_uri, start_run
    from mlflow.tracking import MlflowClient
    from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
    from tqdm import tqdm

    print("Done!")

    # Verify the Path
    ThisPath = Path(__file__)
    RepoPath = ThisPath.parent.parent.parent
    DataPath = RepoPath / "data"
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

    ###########
    # Setup the mlflow tracking
    ###########
    set_tracking_uri(RepoPath / MlflowPath)
    client = MlflowClient()
    ExperimentName = mlflow_setup["experiment_name"]
    try:
        ExperimentID = client.create_experiment(ExperimentName)
    except:
        ExperimentID = client.get_experiment_by_name(ExperimentName).experiment_id
    ###########
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

    def save_run_information(
        general_setup: dict,
        mlflow_setup: dict,
        kalman_setup: dict,
        run_name: str,
        run_id: int,
    ) -> Tuple[Path, Path]:
        """
        Save the run information in the corresponding files.

        Parameters
        ----------
        general_setup : dict
            The general setup dictionary.
        mlflow_setup : dict
            The mlflow setup dictionary.
        kalman_setup : dict
            The kalman setup dictionary.
        run_name : str
            The name of the run.
        run_id : str
            The id of the run.

        Returns
        -------
        InputFile : Path
            The path to the input file.
        KalmanFile : Path
            The path to the kalman file.

        Notes
        -----
        The function saves the following files:
            - ``{run_name}_setup.yaml`` : The general setup dictionary.
            - ``{run_name}_mlflow_setup.yaml`` : The mlflow setup dictionary.
            - ``{run_name}_kalman_setup.yaml`` : The kalman setup dictionary.
            - ``{run_name}_input.nc`` : The input file.
            - ``{run_name}_kalman.nc`` : The kalman file.
        """
        # Create Paths to the corresponding directories names
        ResultDataPath = DataPath / SubdataPath / f"{run_name}"
        ResultDataPath.mkdir(parents=True, exist_ok=True)

        # Create file names to store the  different settings
        SettingsPath = ResultDataPath / f"{run_name}_setup.yaml"
        MlflowSettingsPath = ResultDataPath / f"{run_name}_mlflow_setup.yaml"
        KalmanSettingsPath = ResultDataPath / f"{run_name}_kalman_setup.yaml"
        InputFile = ResultDataPath / f"{run_name}_input.nc"
        KalmanFile = ResultDataPath / f"{run_name}_kalman.nc"

        # log all settings and file locations.
        for key in kalman_setup:
            client.log_param(run_id=run_id, key=key, value=kalman_setup[key])
        # log the model name and the dimension used
        client.log_param(
            run_id=run_id,
            key="ModelName",
            value=model_setup["model_name"],
        )
        client.log_param(
            run_id=run_id,
            key="KalmanFunction",
            value=processing_function.__name__,
        )
        client.log_param(
            run_id=run_id,
            key="SettingsFile",
            value=SettingsPath.relative_to(RepoPath).as_posix(),
        )
        client.log_param(
            run_id=run_id,
            key="KalmanSettingsFile",
            value=KalmanSettingsPath.relative_to(RepoPath).as_posix(),
        )
        client.log_param(
            run_id=run_id,
            key="InputFile",
            value=InputFile.relative_to(RepoPath).as_posix(),
        )
        client.log_param(
            run_id=run_id,
            key="KalmanFile",
            value=KalmanFile.relative_to(RepoPath).as_posix(),
        )

        with open(SettingsPath, "w") as yaml_file:
            yaml.dump(general_setup, yaml_file, default_flow_style=False)
        with open(MlflowSettingsPath, "w") as yaml_file:
            yaml.dump(mlflow_setup, yaml_file, default_flow_style=False)
        with open(KalmanSettingsPath, "w") as yaml_file:
            yaml.dump(kalman_setup, yaml_file, default_flow_style=False)
        # log artifact of the settings
        client.log_artifact(run_id, SettingsPath.as_posix())
        client.log_artifact(run_id, KalmanSettingsPath.as_posix())

        return InputFile, KalmanFile

    ############
    # Create the parent run
    ############
    parent_run = client.create_run(experiment_id=ExperimentID)

    # create parent copies of the dictionaries
    parent_run_name = parent_run.info.run_name
    parent_run_id = parent_run.info.run_id
    parent_general_setup = general_setup.copy()
    parent_mlflow_setup = mlflow_setup.copy()
    parent_kalman_setup = kalman_setup.copy()
    parent_random_setup = random_setup.copy()
    parent_function_setup = function_setup.copy()

    parent_state_variabels = kalman_setup["state_variables"].copy()
    parent_observation_variables = kalman_setup["observation_variables"].copy()
    parent_random_variables = random_setup["name_random_variables"].copy()

    # print information about the run
    print("Information:")
    print(f"ExperimentID : {ExperimentID}")
    print(f"RunName : {parent_run_name}")
    print(f"RunID : {parent_run_id}")

    # save the run information of the parent run:
    InputFile, KalmanFile = save_run_information(
        general_setup=parent_general_setup,
        mlflow_setup=parent_mlflow_setup,
        kalman_setup=parent_kalman_setup,
        run_name=parent_run_name,
        run_id=parent_run_id,
    )

    # load the model data
    #  LOAD THE MODEL DATA
    print("Load model data files")
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

    if random_setup["iterative"] != True:
        print("Non-Iterative Process initialized!\nAdd all latent variables at once.")
        current_run = parent_run
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
        # The ``run_function_on_multiple_subdatasets`` function allows to run the input function on all ``subdatasets`` specified by the ``subdataset_selections``. In this case these selections are given by the ``experiment_settings``.
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
    else:
        print("Iterative Process initialized!\nAdd latent variables one by one.")
        # store parent Input file
        model_dataset.to_netcdf(InputFile)
        # idx of first random variable in the state variables
        idx_first_random_var_state = len(parent_state_variabels) - len(
            parent_random_variables
        )
        current_state_variabels = parent_state_variabels[:idx_first_random_var_state]

        # Iteratively run the Kalman SEM and add random variables one by one
        rng_seed = random_setup["seed"]
        for idx, random_var in enumerate(random_setup["name_random_variables"]):
            # create the child run
            current_run = client.create_run(
                experiment_id=ExperimentID,
                tags={MLFLOW_PARENT_RUN_ID: parent_run.info.run_id},
            )
            # save the run information of the child run:
            run_name = current_run.info.run_name
            run_id = current_run.info.run_id

            # modify the setup dictionaries
            # set the state_variables to the current state variables
            current_state_variabels.append(random_var)
            kalman_setup["state_variables"] = current_state_variabels
            # set the random_variables to the current random variables
            random_setup["name_random_variables"] = parent_random_variables[: idx + 1]
            random_setup["seed"] = rng_seed
            function_setup["func_kwargs"]["state_variables"] = current_state_variabels

            function_setup["func_kwargs"]["ds"] = False
            # update the general setup
            # general_setup['model_setup'] = model_setup
            # general_setup['kalman_setup'] = kalman_setup
            # general_setup['random_setup'] = random_setup
            # general_setup['function_setup'] = function_setup

            InputFile, KalmanFile = save_run_information(
                general_setup=general_setup,
                mlflow_setup=mlflow_setup,
                kalman_setup=kalman_setup,
                run_name=run_name,
                run_id=run_id,
            )

            # in the first iteration use the original input file
            if idx == 0:
                print("Use original input file.")
                current_dataset = model_dataset
            else:
                print("Use previous kalman file as input file.")
                # convert the dataset from the standard dataset to dataset containing all state variables as data variables
                current_dataset = from_standard_dataset(result)
                current_dataset = current_dataset.drop_vars(
                    ["state_name_copy", "kalman_iteration"]
                )
            # add the random variable
            print(f"Add random variable {random_var}.")
            rng = np.random.default_rng(seed=rng_seed + idx)
            add_random_variable(
                ds=current_dataset,
                var_name=random_var,
                random_generator=rng,
                variance=random_setup["random_variance"],
            )
            print("Save input file.")
            current_dataset.to_netcdf(InputFile)
            print("Done!")

            # #### Run the ``xarray_Kalman_SEM`` function from the ``pipeline`` library on all experiment setups.
            # The ``run_function_on_multiple_subdatasets`` function allows to run the input function on all ``subdatasets`` specified by the ``subdataset_selections``. In this case these selections are given by the ``experiment_settings``.
            print(
                f"Run {processing_function.__name__} for : {function_setup['func_kwargs']['nb_iter_SEM']} iterations."
            )

            result = run_function_on_multiple_subdatasets(
                processing_function=processing_function,
                parent_dataset=current_dataset,
                subdataset_selections=subdataset_selections,
                func_args=function_setup["func_args"],
                func_kwargs=function_setup["func_kwargs"],
            )
            print("Done!")
            # ---- Save Files ----
            print("Save kalman file.")
            result.to_netcdf(KalmanFile)
            print("Done!")
            # end the child run
            client.set_terminated(current_run.info.run_id)
    # end the parent run
    client.set_terminated(parent_run.info.run_id)

    print("------------\nTracking Done!------------\n")
    print(f"ExperimentID : {ExperimentID}")
    print(f"RunName : {run_name}")
    print(f"RunID : {run_id}")

if __name__ == "__main__":
    main()