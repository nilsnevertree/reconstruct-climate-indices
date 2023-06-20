"""
Converts given CiCMOD csv file into multiple xr.Datasets for each individual
model.

These will be stored under DESTINATION_DIR/climate_indices_``MODEL``.nc
Function ``cicmod_csv_to_netcdf`` can also be imported.
This is the main function of the file.

For options use --help
"""

import argparse

from os import PathLike
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def cicmod_csv_to_netcdf(
    data_url: PathLike, destination_dir: PathLike, file_name: str = "climate_indices"
) -> None:
    """
    Convert climate indices data from a CSV file to NetCDF format and save to
    the destination directory. It is made for converting the CiCMOD dataset csv
    results into netCDF. For each model present in the csv file a seperate file
    will be created: Example of csv file with two models "FOCI" and "CESM":
    Output will be two files with Paths:

        - ``destination_dir`` / ``file_name``_CESM.nc
        - ``destination_dir`` / ``file_name``_FOCI.nc

    This function reads climate indices data from a CSV file, performs necessary transformations, and saves the data
    as NetCDF files for each model in the specified destination directory.

    Parameters:
        data_url (PathLike): The path or URL to the input CSV file.
        destination_dir (PathLike): The destination directory where the NetCDF files will be saved.
        file_name (str, optional): The base name for the resulting NetCDF files (default: "climate_indices").

    Returns:
        None

    NOTE: Using another coordinate with the name "time" should be suppresed!
        # add more time coords to the datasets
        time coord it wanted use always day 15, which is roughly the mean of each month
        time = [
            cftime.DatetimeNoLeap(year, month, 15)
            for year, month in zip(years, months)
        ]
        ds = ds.assign_coords(time=(["year"], time))

        store the data

    Examples:
        # Example 1: Convert climate indices data from CSV to NetCDF assume two models in the ``climate_indices.csv`` file : "CSEM" and "FOCI"
        >>> data_url = "path/to/climate_indices.csv"
        >>> destination_dir = "path/to/destination"
        >>> file_name = "climate_data"
        >>> cicmod_csv_to_netcdf(data_url, destination_dir, file_name)
        # Output will be two files with Paths:
        # - destination_dir / file_name_CESM.nc
        # - destination_dir / file_name_FOCI.nc

        # Example 2: Convert climate indices data from CSV to NetCDF (using default file name)
        >>> data_url = "path/to/climate_indices.csv"
        >>> destination_dir = "path/to/destination"
        >>> cicmod_csv_to_netcdf(data_url, destination_dir)
    """
    destination_data = lambda model: destination_dir / f"{file_name}_{model}.nc"

    climind = pd.read_csv(data_url)
    # Set index:
    climind = climind.set_index(["model", "year", "month", "index"]).unstack(level=-1)[
        "value"
    ]

    model_index = climind.index.get_level_values("model")
    available_models = model_index.unique()
    for model in available_models:
        # select the dataframe for each model
        df = climind.loc[model]
        # extract years and months
        years = np.array(df.index.get_level_values("year"))
        months = np.array(df.index.get_level_values("month"))
        year_index = years + (months - 1) / 12

        # create time index.
        # NOTE: Other dimension names are not yet supported by the kalman_reconstruction.pipeline module
        df = df.set_index(year_index)
        df.index.names = ["time"]

        # create the xr.Dataset for each model
        ds = xr.Dataset(df)
        ds.assign_attrs(
            dict(
                model=model,
                time_unit="Units in years.\nThe decimals correspond to the fraction of the month compared to 12.\n"
                "Example: year 5 month 6 -> 'time' = 5 + 6/12 = 5.5",
            )
        )

        # store the data
        ds.to_netcdf(destination_data(model))


# for further informattion on the data version see __version__.json in the same directory
if __name__ == "__main__":
    cicmod_dir = Path(__file__).parent.parent / "earth_system_models" / "CiCMOD"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_url",
        default=cicmod_dir / "climate_indices.csv",
        help="path or URL to the CiCMOD csv file. \nDefault to ../earth_system_models/CiCMOD/climate_indices.csv",
    )
    parser.add_argument(
        "--destination_dir",
        default=cicmod_dir,
        help="path to directory to store all individual xr.Datasets for each model in the CiCMOD csv file. \nDefault to ../earth_system_models/CiCMOD",
    )

    args = parser.parse_args()
    data_url = args.data_url
    destination_dir = args.destination_dir
    cicmod_csv_to_netcdf(data_url=data_url, destination_dir=destination_dir)
