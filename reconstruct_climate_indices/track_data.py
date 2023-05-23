# %%
import os

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


def track_model(
    _func, mlflow_args={}, func_args={}, func_kwargs={}, subdata_path="idealized_ocean"
):
    REPO_PATH = Path(__file__).parent.parent.resolve()
    print(REPO_PATH)
    DATA_PATH = REPO_PATH / "data" / subdata_path
    set_tracking_uri(REPO_PATH / "mlruns")
    with start_run(**mlflow_args) as run:
        # log function name
        run_id = run.info.run_id
        log_param("FunctionName", _func.__name__)

        # log filepath and store file
        FILE_PATH = DATA_PATH / f"{run_id}.nc"
        log_param("FilePath", FILE_PATH.relative_to(DATA_PATH).as_posix())

        # create directory if needed
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)

        # perform model run as save file
        ds, setting = _func(*func_args, **func_kwargs)
        log_params(setting)
        ds.to_netcdf(FILE_PATH)

        log_artifacts(DATA_PATH.as_posix())
    end_run()
    return ds
