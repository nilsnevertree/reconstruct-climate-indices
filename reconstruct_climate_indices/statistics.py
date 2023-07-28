# %%
from typing import Dict, Tuple, TypedDict, Union

import numpy as np
import xarray as xr

from scipy import fft, signal
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


def transform_to_loglog(x: np.ndarray) -> np.ndarray:
    """
    Transform an array of values to a logarithmic scale.

    Parameters:
        x : np.ndarray
            The input array of values.

    Returns:
        np.ndarray
            The transformed array of values.

    Raises:
        TypeError
            If the input array has more than 1 dimension.

    Examples:
        # Example 1: Transform an array to a logarithmic scale
        >>> import numpy as np
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> transformed = transform_to_loglog(x)
        >>> print(transformed)
        array([0.        , 0.30103   , 0.47712125, 0.60205999, 0.69897   ])
    """
    if np.ndim(x) > 1:
        raise TypeError("Only 1D arrays are supported.")

    # only use values where x is not equal to zeros
    x = x[x != 0]
    x_log = np.log10(x)
    x_log = x_log[np.isfinite(x_log)]
    return x_log


def fit_linear_regression_from_loglog(
    frequencies: np.ndarray,
    spectrum: np.ndarray,
    f_low: float = -np.inf,
    f_high: float = np.inf,
    weights: Union[str, None, np.ndarray, list] = None,
) -> LinearRegression:
    """
    Fit a linear regression model to a spectrum for a given frequency range in
    a log10, log10 space.

    Parameters:
        frequencies (np.ndarray): The array of frequency values.
        spectrum (np.ndarray): The array of spectrum values.
        f_low (float): The lower frequency limit for fitting (default: -np.inf).
        f_high (float): The upper frequency limit for fitting (default: np.inf).
        weights (Union[str, None, np.ndarray, list]): Optional weights for the linear regression.
            - If 'f_inverse', the weights are set as the inverse of frequencies.
            - If None, "", "ones", "ONES", or "1", the weights are set as an array of ones.
            - If 'power_log', the weights are set as the log10 of the spectrum itself.
            - If an np.ndarray or a list, the weights are set as the provided array or list.
            (default: None)

    Returns:
        LinearRegression: The fitted linear regression model.

    Raises:
        ValueError: If the given weights of type are not supported.

    Examples:
        # Example 1: Fit linear regression with default parameters
        >>> import numpy as np
        >>> from sklearn.linear_model import LinearRegression
        >>> frequencies = np.array([1, 2, 3, 4, 5])
        >>> spectrum = np.array([10, 20, 30, 40, 50])
        >>> model = fit_linear_regression(frequencies, spectrum)
        >>> intercept = model.intercept_
        >>> slope = model.coef_[0]
        >>> print("Intercept:", intercept)
        >>> print("Slope:", slope)

        # Example 2: Fit linear regression with specified frequency range and weights
        >>> frequencies = np.array([1, 2, 3, 4, 5])
        >>> spectrum = np.array([10, 20, 30, 40, 50])
        >>> weights = [1, 2, 3, 4, 5]
        >>> model = fit_linear_regression(frequencies, spectrum, f_low=2, f_high=4, weights=weights)
        >>> intercept = model.intercept_
        >>> slope = model.coef_[0]
        >>> print("Intercept:", intercept)
        >>> print("Slope:", slope)
    """
    # Filter data within the specified frequency range
    mask = np.logical_and.reduce(
        (frequencies >= f_low, frequencies <= f_high, frequencies != 0)
    )
    frequencies_filtered = frequencies[mask]
    spectrum_filtered = spectrum[mask]

    # Create log10 values of the original data

    frequencies_log = np.log10(frequencies_filtered)
    spectrum_log = np.log10(spectrum_filtered)

    # Remove NaN and infinite values for all arrays
    finite_mask = np.logical_and(
        np.isfinite(frequencies_log), np.isfinite(spectrum_log)
    )

    frequencies_log = frequencies_log[finite_mask]
    spectrum_log = spectrum_log[finite_mask]
    frequencies_filtered = frequencies_filtered[finite_mask]
    spectrum_filtered = spectrum_filtered[finite_mask]

    # Set weights based on the provided option
    if weights == "f_inverse":
        weights = (frequencies_filtered) ** (-1)
    elif weights == "power_log":
        weights = spectrum_log
    elif weights in [None, "", "ones", "ONES", "1"]:
        weights = np.ones_like(frequencies_filtered)
    elif isinstance(weights, (np.ndarray, list)):
        assert len(weights) == len(spectrum_filtered)
        weights = np.array(weights)
    else:
        raise ValueError(
            f"The given weights of type {type(weights)} are not supported."
        )

    # Fit linear regression
    return LinearRegression().fit(
        X=frequencies_log.reshape(-1, 1),
        y=spectrum_log.reshape(-1, 1),
        sample_weight=weights,
    )


def predict_to_loglog(
    x: np.ndarray, regression: LinearRegression
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict values using a logarithmic scale regression model.
    Returned values are in the same scalling as ``x``.

    Parameters:
        x : np.ndarray
            The input array of values.
        regression : LinearRegression
            The trained linear regression model.

    Returns:
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the transformed x values and the predicted y values.

    Raises:
        TypeError
            If the input array has more than 1 dimension.

    Examples:
        # Example 1: Predict using a linear regression model
        >>> import numpy as np
        >>> from sklearn.linear_model import LinearRegression
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> regression = LinearRegression()
        >>> regression.fit(x.reshape(-1, 1), np.log10(x))
        >>> x_new, y_pred = predict_to_loglog(x, regression)
        >>> print("Transformed x values:", x_new)
        >>> print("Predicted y values:", 10**y_pred)

        # Example 2: Predict using a pre-trained regression model
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> regression = LinearRegression()
        >>> regression.coef_ = np.array([0.5])  # Set custom coefficients for demonstration
        >>> regression.intercept_ = np.array([1.0])  # Set custom intercept for demonstration
        >>> x_new, y_pred = predict_to_loglog(x, regression)
        >>> print("Transformed x values:", x_new)
        >>> print("Predicted y values:", 10**y_pred)
    """
    x_transformed = transform_to_loglog(x)
    y_pred = regression.predict(x_transformed.reshape(-1, 1))
    return 10**x_transformed, 10**y_pred


def linear_regression_loglog(
    frequencies: np.ndarray,
    spectrum: np.ndarray,
    f_low: float = -np.inf,
    f_high: float = np.inf,
    weights: Union[str, None, np.ndarray, list] = None,
) -> Tuple[np.ndarray, np.ndarray, LinearRegression]:
    """
    Fit a linear regression model on log-log transformed data and predict using
    the trained model.

    This function fits a linear regression model on log-log transformed data given the input frequencies and spectrum,
    and then predicts the transformed values for the same input frequencies using the trained model.

    Parameters:
        frequencies : np.ndarray
            The input array of frequencies.
        spectrum : np.ndarray
            The input array of spectrum values.
        f_low : float, optional
            The lower frequency limit for fitting the linear regression model (default: -inf).
        f_high : float, optional
            The upper frequency limit for fitting the linear regression model (default: inf).
        weights : Union[str, None, np.ndarray, list], optional
            The weights to be used during model fitting. Supported values are:
            - None: No weights are applied.
            - "f_inverse": Weights are set as the inverse of the frequencies.
            - If 'power_log', the weights are set as the log10 of the spectrum itself.
            - np.ndarray or list: Custom weights provided as an array or list.

    Returns:
        Tuple[np.ndarray, np.ndarray]
            A tuple containing the transformed frequencies and the predicted log-log transformed spectrum values.

    Examples:
        # Example 1: Fit linear regression model and make predictions
        >>> import numpy as np
        >>> frequencies = np.array([1, 2, 3, 4, 5])
        >>> spectrum = np.array([10, 20, 30, 40, 50])
        >>> transformed_frequencies, predicted_spectrum = linear_regression_loglog(frequencies, spectrum)
        >>> print("Transformed frequencies:", transformed_frequencies)
        >>> print("Predicted log-log transformed spectrum:", predicted_spectrum)

        # Example 2: Fit linear regression model with custom weights and make predictions
        >>> frequencies = np.array([1, 2, 3, 4, 5])
        >>> spectrum = np.array([10, 20, 30, 40, 50])
        >>> weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        >>> transformed_frequencies, predicted_spectrum = linear_regression_loglog(frequencies, spectrum, weights=weights)
        >>> print("Transformed frequencies:", transformed_frequencies)
        >>> print("Predicted log-log transformed spectrum:", predicted_spectrum)
    """
    regression = fit_linear_regression_from_loglog(
        frequencies=frequencies,
        spectrum=spectrum,
        f_low=f_low,
        f_high=f_high,
        weights=weights,
    )
    frequencies_linear, spectrum_linear = predict_to_loglog(
        x=frequencies, regression=regression
    )
    # frequencies_res = frequencies.copy()
    f_res = np.zeros_like(frequencies) * np.nan
    f_res[frequencies != 0] = frequencies_linear
    spectrum_res = np.zeros_like(spectrum) * np.nan
    spectrum_res[frequencies != 0] = spectrum_linear.flatten()
    assert len(f_res) == len(frequencies)
    return f_res, spectrum_res, regression


PDS_Info = TypedDict(
    "PDS",
    {
        "frequencies": np.ndarray,
        "spectrum": np.ndarray,
    },
)


PDS_linear = TypedDict(
    "PDS_LogLogRegression",
    {
        "frequencies": np.ndarray,
        "spectrum": np.ndarray,
        "regression": LinearRegression,
        "slope": float,
    },
)


def power_density_spectrum(
    x: np.ndarray,
    fs: float = 1,
    window: str = "hann",
    welch_window_width: int = np.inf,
    linear_reg_kwargs: dict = dict(weights="f_inverse"),
) -> Tuple[PDS_Info, PDS_Info, PDS_linear]:
    """
    Calculate the power density spectrum of a given signal using Welch's method
    and perform linear regression on the log-log transformed spectrum.

    This function calculates the power density spectrum of a given signal using Welch's method, which involves
    applying a window function to the signal and estimating the spectrum using the Fast Fourier Transform (FFT).
    The discrete FFT transform is used.
    See also https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html

    It also performs linear regression on the log-log transformed spectrum to estimate the slope.
    The function ``linear_regression_loglog`` is used.


    All three options are returned:
    a) "frequencies", "spectrum" using infinite welch ``window`` of form "hann" -> NO welch method applied -> pure DFFT
    b) "frequencies", "spectrum" using welch method
    c) "frequencies", "spectrum", "regression" and "slope" using results from a) for ``linear_regression_loglog``.

    Parameters:
        x : np.ndarray
            The input signal.
        fs : float, optional
            The sampling frequency of the input signal `x` (default: 1).
        window : str, optional
            The type of window to use in the Welch method (default: "hann").
        welch_window_width : int, optional
            The width of the Welch window in samples. If set to np.inf, the entire signal length is used (default: np.inf).
        linear_reg_kwargs : dict, optional
            Additional keyword arguments to pass to the linear regression fitting function ``linear_regression_loglog`` (default: {'weights': 'f_inverse'}).

    Returns:
        Tuple[PDS_Info, PDS_Info, PDS_linear]
            A tuple containing the computed power density spectrum (PDS) and the results of linear regression.
                1. "frequencies", "spectrum" using infinite welch ``window`` of form "hann" -> NO welch method applied -> pure DFFT
                2. "frequencies", "spectrum" using welch method
                3. "frequencies", "spectrum", "regression" and "slope" using results from a) for ``linear_regression_loglog``.

    Examples:
        # Example 1: Compute PDS and perform linear regression
        >>> import numpy as np
        >>> from scipy import signal
        >>> from sklearn.linear_model import LinearRegression
        >>> x = np.random.randn(1000)
        >>> pds_pure, pds_welch, pds_linear = power_density_spectrum(x, fs=1000, window='hann')
        >>> print("Pure PDS frequencies:", pds_pure['frequencies'])
        >>> print("Pure PDS spectrum:", pds_pure['spectrum'])
        >>> print("Welch PDS frequencies:", pds_welch['frequencies'])
        >>> print("Welch PDS spectrum:", pds_welch['spectrum'])
        >>> print("Linear PDS frequencies:", pds_linear['frequencies'])
        >>> print("Linear PDS spectrum:", pds_linear['spectrum'])
        >>> print("Linear regression slope:", pds_linear['slope'])

        # Example 2: Compute PDS with custom Welch window width and linear regression parameters
        >>> x = np.random.randn(2000)
        >>> welch_width = 0.5
        >>> linear_reg_params = {'weights': None}
        >>> pds_pure, pds_welch, pds_linear = power_density_spectrum(x, fs=2000, window='hann', welch_window_width=welch_width, linear_reg_kwargs=linear_reg_params)
        >>> print("Pure PDS frequencies:", pds_pure['frequencies'])
        >>> print("Pure PDS spectrum:", pds_pure['spectrum'])
        >>> print("Welch PDS frequencies:", pds_welch['frequencies'])
        >>> print("Welch PDS spectrum:", pds_welch['spectrum'])
        >>> print("Linear PDS frequencies:", pds_linear['frequencies'])
        >>> print("Linear PDS spectrum:", pds_linear['spectrum'])
        >>> print("Linear regression slope:", pds_linear['slope'])
    """
    frequencies, spectrum = signal.welch(x=x, fs=fs, window=window, nperseg=len(x))

    frequencies_welch, spectrum_welch = signal.welch(
        x=x, fs=fs, window=window, nperseg=int(welch_window_width * fs)
    )

    frequencies_linear, spectrum_linear, regression = linear_regression_loglog(
        frequencies=frequencies, spectrum=spectrum, **linear_reg_kwargs
    )

    res_pure = dict(
        frequencies=frequencies,
        spectrum=spectrum,
    )

    res_welch = dict(
        frequencies=frequencies_welch,
        spectrum=spectrum_welch,
    )

    res_linear = dict(
        frequencies=frequencies_linear,
        spectrum=spectrum_linear,
        regression=regression,
        slope=regression.coef_[0, 0],
    )

    return res_pure, res_welch, res_linear


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

    Note:
        See also docstring of welch for the kwargs that can be supplied
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
        ./reconstruct_climate_indices/note-scipy-signal-welch.md
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

    Note:
        See also docstring of welch for the kwargs that can be supplied
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
        ./reconstruct_climate_indices/note-scipy-signal-welch.md
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


import numpy as np

# %%
from scipy import fft


def smooth_in_freq_space(
    x: np.ndarray, fft_dim: int = 1, smooth_dim: int = 0
) -> np.ndarray:
    """
    Smooths an array along one dimension using median filtering in the
    frequency domain.

    The function applies a median filter in the frequency domain to smooth the input array along the specified dimension.

    Parameters:
        x (np.ndarray): The input array to be smoothed.
        fft_dim (int, optional): The dimension along which the FFT will be computed. Default is 1.
        smooth_dim (int, optional): The dimension along which the smoothing will be applied. Default is 0.

    Returns:
        np.ndarray: The smoothed array.

    Example:
        # Smooth an array along the first dimension using median filtering in the frequency domain
        >>> import numpy as np
    """
    if fft_dim >= smooth_dim:
        res = fft.irfft(
            np.median(fft.rfft(x, axis=fft_dim), axis=smooth_dim), axis=fft_dim - 1
        )
    else:
        res = fft.irfft(
            np.median(np.abs(fft.rfft(x, axis=fft_dim)), axis=smooth_dim), axis=fft_dim
        )
    return res


# # %%
# from numpy.testing import assert_allclose
# import matplotlib.pyplot as plt

# # def test_smooth_in_freq_space(x):
# #     # Test case 3: Concatenation of three sinus functions with different frequencies
# #     x3 =  np.array((
# #         np.sin(2 * np.pi * 0.2 * np.linspace(0, 10*np.pi, 1000)),
# #         np.sin(2 * np.pi * 0.2 * np.linspace(0, 10*np.pi, 1000)),
# #         np.sin(2 * np.pi * 0.2 * np.linspace(0, 10*np.pi, 1000))
# #     ))

# #     expected_result3 = np.sin(2 * np.pi * 0.1 * np.linspace(0, 10*np.pi, 1000))
# #     result3 = smooth_in_freq_space(x3, fft_dim=0, smooth_dim=1)
# #     return x3, result3
# #     #assert_allclose(result3, expected_result3, atol=1e-6)
# n = 10000
# p = 30
# x_values = np.ones((p, n))* np.linspace(0, 50*2*np.pi, n)
# random_values = np.random.random_integers(low= -20, high = 100, size = (p, n))
# random_phase = 0.89589* np.array([x_values[0,:]*0 + 1 + rn for rn in random_values[:,0]])
# x3 = x_values*0+ random_values*0.01 + (
#         np.sin(2 * np.pi * 0.03 * (x_values - random_phase))
#         + np.sin(2 * np.pi * 0.07 * (x_values - random_phase))
#         + np.sin(2 * np.pi * 0.01 * x_values)
# )


# fig, axs = plt.subplots(5, 1)
# for i, x3_temp in enumerate(x3[0:4,:]) :
#     axs[i].plot(x3_temp)

# axs[i+1].plot(smooth_in_freq_space(x3, fft_dim=1, smooth_dim=0))
# # %%

# plt.loglog(np.abs(fft.rfft(x3, axis=1)).T, color = [0.5,0.5,0.5], alpha = 0.1)
# plt.loglog(np.mean(np.abs((fft.rfft(x3, axis=1))), axis = 0).T)
# plt.loglog(np.median(np.abs((fft.rfft(x3, axis=1))), axis = 0).T)
# # %%
