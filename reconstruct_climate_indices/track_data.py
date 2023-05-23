# %%
from pathlib import Path
from random import randint, random

from mlflow import (
    end_run,
    log_artifacts,
    log_metric,
    log_param,
    log_params,
    set_tracking_uri,
    start_run,
)


def track_model(_func, mlflow_args={}, func_args={}, func_kwargs={}):
    REPO_PATH = Path(__file__).parent.parent.resolve()
    print(REPO_PATH)
    DATA_PATH = REPO_PATH / "data" / "idealized_ocean"
    set_tracking_uri(REPO_PATH / "mlruns")
    with start_run(**mlflow_args) as run:
        run_id = run.info.run_id
        # perform model run
        ds, setting = _func(*func_args, **func_kwargs)
        log_params(setting)
        # log filepath and store file
        FILE_PATH = DATA_PATH / f"{run_id}.nc"
        log_param("FileName", FILE_PATH.relative_to(DATA_PATH).as_posix())
        ds.to_netcdf(FILE_PATH)
    end_run()
