from typing import Dict, Tuple, TypedDict, Union

import numpy as np
import xarray as xr

from scipy import signal
from sklearn.linear_model import LinearRegression


def my_mean(x):
    """
    Calculate the mean of a numpy array using np.mean This functions only
    purpose is to check the CI workflows of the repository.

    Parameters
    ----------
    x: np.ndarray
        Array for which the mean shall be calculated.

    Returns
    -------
    np.ndarray
        Mean of input array x.
        ndim=1
    """

    return np.mean(x)


WelchKwargs = TypedDict(
    "WelchKwargs",
    {
        "nperseg": int,
        "nfft": int,
        "detrend": str,
        "return_onesided": bool,
        "scaling": str,
        # 'axis': int,      # this should be fixes in the functions
        "average": str,
        "fs": float,
    },
)


def xarray_dataarray_welch(
    da: xr.DataArray,
    dim: str = "time",
    new_dim: str = "frequency",
    welch_kwargs: WelchKwargs = dict(fs=1),
) -> xr.Dataset:
    """
    Perform Welch's method on a single xarray DataArray along a specified
    dimension.

    Parameters:
        da (xr.DataArray): The input xarray DataArray.
        dim (str): The dimension along which to perform the Welch's method. Default is "time".
        new_dim (str): The name of the new dimension representing frequency. Default is "frequency".
        welch_kwargs (Dict): Additional keyword arguments to pass to `scipy.signal.welch`. Default is {'fs': 1}.

    Returns:
        xr.Dataset: A new xarray Dataset containing the resulting power spectral density.

    Example:
        # Perform Welch's method on a temperature DataArray along the "time" dimension
        result = xarray_array_welch(da=temperature, dim="time", new_dim="frequency", welch_kwargs={'fs': 2})
    """
    da = da.transpose(dim, ...)
    dims = da.dims
    coords_keys = [val for val in dims if dim not in val]
    coords_values = dict([(key, da.coords[key].values) for key in coords_keys])
    new_dims = [new_dim] + coords_keys

    f, s = signal.welch(x=da, axis=0, **welch_kwargs)
    result = xr.DataArray(
        s,
        coords={
            new_dim: f,
            **coords_values,
        },
        dims=new_dims,
    )
    return result


def xarray_dataset_welch(
    ds: xr.Dataset,
    dim: str = "time",
    new_dim: str = "frequency",
    welch_kwargs: WelchKwargs = dict(fs=1),
) -> xr.Dataset:
    """
    Perform Welch's method on all variables within an xarray Dataset.

    Parameters:
        ds (xr.Dataset): The input xarray Dataset.
        dim (str): The dimension along which to perform the Welch's method. Default is "time".
        new_dim (str): The name of the new dimension representing frequency. Default is "frequency".
        welch_kwargs (Dict): Additional keyword arguments to pass to `scipy.signal.welch`. Default is {'fs': 1}.

    Returns:
        xr.Dataset: A new xarray Dataset containing the resulting power spectral densities for all variables.

    Example:
        # Perform Welch's method on a dataset along the "time" dimension
        result = xarray_dataset_welch(ds=dataset, dim="time", new_dim="frequency", welch_kwargs={'fs': 2})
    """
    dims = ds.dims
    coords_keys = [val for val in dims if dim not in val]
    coords_values = dict([(key, ds.coords[key].values) for key in coords_keys])
    res = xr.Dataset(coords=coords_values)

    for var in ds.data_vars:
        res[var] = xarray_dataarray_welch(
            ds[var], dim=dim, new_dim=new_dim, welch_kwargs=welch_kwargs
        )
    return res
