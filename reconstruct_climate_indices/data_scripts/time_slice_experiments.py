# %% [markdown]
"""
# Track parameter Experiments

### Track the experiment using ``mlflow``
Using mlflow, the following information will be stored:
- Dataset containing the results from the ``AMO_oscillatory_ocean`` for all experiment settings.
- Dataset containing the results from the ``xarray_Kalman_SEM`` for all experiment settings.
- Settings to create the different Model runs using (``AMO_oscillatory_ocean``).
- Settings used by the ``xarray_Kalman_SEM``.

Therefore multiple setting for mlflow will need to be set by the User:
- ExperimentID : Corresponds to the experiment_id used by ``mlflow`` to set the ``set_tracking_uri``.
- SubdataPath : Name of the directory in which to store the results. This will be a child of the ``data`` directory.
- MlflowPath : Name of the directory in which the mlflow tracking uri shall be used.
- NOTE:
    - Make sure that the RepoPath is correct!
    - Make sure that ExperimentID exists!

The folder structure will be :

**Folder structure**

    └───RepoPath
        └───data
            └───SubdataPath
                └───run_id
                    │    run_id_input.nc
                    │    run_id_kalman.nc
                    │    run_id_kalman_settings.yml
                    │    run_id_parameter_settings.yml
Where ``run_id`` is e.g. *553cbd3bc6ce44028c8daad12647c306*
"""

# %%


def main():
    """Main function to run the script."""
    print("Start imports.")
    import itertools

    from pathlib import Path

    import numpy as np
    import xarray as xr
    import yaml

    from kalman_reconstruction import pipeline
    from mlflow import end_run, log_artifact, log_params, set_tracking_uri, start_run
    from tqdm import tqdm

    from reconstruct_climate_indices.idealized_ocean import AMO_oscillatory_ocean

    print("Done!")

    # %%
    def product_dict(**kwargs):
        keys = kwargs.keys()
        for instance in itertools.product(*kwargs.values()):
            yield dict(zip(keys, instance))

    # %% [markdown]
    # ## Settings

    # %% [markdown]
    # #### Kalman Settings

    # %%
    # seed for the randomnumber generator
    seed = 39266
    # Varaince of the randomly initialized latent variable
    random_variance = 1
    # iterations of the kalman SEM
    nb_iter_SEM = 30
    # observation variables
    observation_variables = ["AMO", "NAO", "EAP"]
    # state variables
    state_variables = ["AMO", "NAO", "EAP", "latent"]

    # create the dictionary that shall be used to store the kalman_settings in the mlflow tracking
    kalman_settings = dict(
        RandomNumberGeneratorSeed=seed,
        RandomVariance=random_variance,
        NumberKalmanIteration=nb_iter_SEM,
        ObservartionVariables=observation_variables,
        StateVariables=state_variables,
    )
    # positional args for the kalman_SEM algorithm
    func_args = dict()
    # key word args for the kalman_SEM algorithm
    func_kwargs = dict(
        observation_variables=observation_variables,
        state_variables=state_variables,
        nb_iter_SEM=nb_iter_SEM,
    )

    # Random number generators used to create the latent variable.
    rng1 = np.random.default_rng(seed=seed)
    # rng2 = np.random.default_rng(seed=seed + 1)
    # rng3 = np.random.default_rng(seed=seed + 2)
    # rng4 = np.random.default_rng(seed=seed + 3)

    # %% [markdown]
    # #### Experiment settings
    #
    # The model used is the ``AMO_oscillatory_ocean``. The parameters ``dNAO`` and ``dEAP`` will be changed.

    # %% In this different slices from the same model will be used to create kalman analysis.

    duration_single_analyis = int(400 * 365.25)
    number_of_analysis = 3**2
    nt = number_of_analysis * duration_single_analyis
    start_times = np.arange(0, nt, duration_single_analyis)

    time_slices = [
        slice(start_time, start_time + duration_single_analyis - 1)
        for start_time in start_times
    ]

    experiment_setups = dict(time=time_slices)
    # %%
    default_settings = dict(
        nt=nt,  # timesteps
        dt=30,  # days
        per0=24 * 365.25,  # days
        tau0=10 * 365.25,  # days
        dNAO=0.1,
        dEAP=0.1,
        cNAOvsEAP=0,
    )

    # modified_arguments = ["dNAO", "per0"]
    # factors = np.array([0.1, 0.5, 1, 2, 5])

    # # create all the experiment setups
    # experiment_setups = dict()
    # for key in modified_arguments:
    #     # make sure to not go into too much details
    #     experiment_setups[key] = np.round(default_settings[key] * factors, 6)

    # # all experiment settings are made up by the all combinations of the experiment setups
    experiment_settings = list(product_dict(**experiment_setups))

    # %%
    ExperimentID = 221777028065508978
    SubdataPath = "amo-oscillator-time-slice-experiment"
    MlflowPath = "mlruns"
    ThisPath = Path(__file__)
    RepoPath = ThisPath.parent.parent
    print(f"Is this the Repository Path correct?:\n{RepoPath}")
    try:
        assert "yes" == input("Write 'yes' any press Enter!\n")
    except:
        raise UserWarning(
            f"User stopped the code due to incorrect Repository Path\n{RepoPath}"
        )
    print("------------\nStart tracking of the experiment!\n------------\n")
    set_tracking_uri(RepoPath / MlflowPath)
    with start_run(experiment_id=ExperimentID) as run:
        # prepare the parameter_settings to indlude all arrays used in the experiment_setup
        parameter_settings = dict()
        parameter_settings.update(default_settings)
        parameter_settings["duration_single_analyis"] = duration_single_analyis
        parameter_settings["number_of_analysis"] = number_of_analysis

        for key in parameter_settings:
            try:
                parameter_settings[key] = parameter_settings[key].tolist()
            except:
                pass
        # set the tracking_uri
        # retrieve the run_id
        run_id = run.info.run_id

        # Create Paths to the corresponding directories names
        DataPath = RepoPath / "data"
        SubdataPath = DataPath / SubdataPath / f"{run_id}"
        SubdataPath.mkdir(parents=True, exist_ok=True)

        # Create file names to store the  different settings
        ParameterSettingsPath = SubdataPath / f"{run_id}_parameter_settings.yml"
        KalmanSettingsPath = SubdataPath / f"{run_id}_kalman_settings.yml"
        InputFile = SubdataPath / f"{run_id}_input.nc"
        KalmanFile = SubdataPath / f"{run_id}_kalman.nc"

        # log all settings and file locations.
        log_params(kalman_settings)
        log_params(parameter_settings)
        log_params(
            dict(
                ParameterSettingsFile=ParameterSettingsPath.relative_to(
                    RepoPath
                ).as_posix(),
                KalmanSettingsFile=KalmanSettingsPath.relative_to(RepoPath).as_posix(),
                InputFile=InputFile.relative_to(RepoPath).as_posix(),
                KalmanFile=KalmanFile.relative_to(RepoPath).as_posix(),
            )
        )
        # after logging, update time_slices
        parameter_settings["time_slices"] = [
            dict(
                start=float(s.start),
                stop=float(s.stop),
            )
            for s in experiment_setups["time"]
        ]

        with open(ParameterSettingsPath, "w") as yaml_file:
            yaml.dump(parameter_settings, yaml_file, default_flow_style=False)
        with open(KalmanSettingsPath, "w") as yaml_file:
            yaml.dump(kalman_settings, yaml_file, default_flow_style=False)
        # log artifact of the settings
        log_artifact(ParameterSettingsPath.as_posix())
        log_artifact(KalmanSettingsPath.as_posix())

        # ### Run the experiments:

        # #### Create all datasets by running the model function ``AMO_oscillatory_ocean``.
        #
        # The results will be combined into a single Dataset
        print("Create experiments")
        setting = default_settings.copy()
        # we will not track each individual model run.
        experiments = AMO_oscillatory_ocean(**setting)
        print("Done!")
        print("Save experiments file.")
        experiments.to_netcdf(InputFile)
        print("Done!")

        # %% [markdown]
        # #### Run the ``xarray_Kalman_SEM`` function from the ``pipeline`` library.
        #
        # The ``run_function_on_multiple_subdatasets`` function allows to run the input function on all ``subdatasets`` specified by the ``subdataset_selections``. In this case these selections are given by the ``experiment_settings``.

        # %%
        print(f"Run Kalman SEM for : {nb_iter_SEM} iterations")
        input_kalman = experiments.copy()
        pipeline.add_random_variable(
            ds=input_kalman,
            var_name="latent",
            random_generator=rng1,
            variance=random_variance,
        )
        experiments_kalman = pipeline.run_function_on_multiple_time_slices(
            processing_function=pipeline.xarray_Kalman_SEM,
            parent_dataset=input_kalman,
            time_slices=experiment_settings,
            func_args=func_args,
            func_kwargs=func_kwargs,
        )
        print("Done!")
        # ---- Save Files ----
        print("Save kalman file.")
        experiments_kalman.to_netcdf(KalmanFile)
        print("Done!")

    end_run()
    print("------------\nTracking Done!------------\n")
    print(f"ExperimentID : {ExperimentID}")
    print(f"RunID : {run_id}")


if __name__ == "__main__":
    main()
