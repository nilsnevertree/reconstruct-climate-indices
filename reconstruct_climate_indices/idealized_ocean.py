#%%
# idealized.ipynb
# DESCRIPTION
# -----------
# The idea is to compute two models of the ocean surface response (SST) to an atmospheric stochastic forcing (SAT)
# The surface ocean response represents the AMO/AMV (Atlantic Multidecadal Oscillation/Variability)
# The atmsopheric stochastic forcing represents the NAO (North Atlantic Oscillation)
# The first model - Spunge Ocean (_spg) - follows is a simple spunge ocean - where SST is restored to 0
# The second model - Oscillatry Ocean (_ocs) - follows a simple damped ocsillator - where ocean oscillate between SST and DOT (Deep Ocean Temperature)
# Both ocean models are stimulated by random white noise from SAT - where only the SST is affected.
#
# EQUATIONS
# ---------
# => For the Spunge Ocean:
# dSST = - \lambda SST dt + SAT dW,
# where \lambda is the inverse of a damping time scale
# => For the Oscillatory Ocean:
# Damped oscillation are outcome of
# [ d^2 x /dt^2 ] + \lambda [ dt x / dt ]+ \omega_0^2 [ x ] = 0,
# where \omega_0= 2 * pi / per0 is the naturale frequency of the system and per0 the natural period.
# It can be decomposed in two eqaution by introducing  dt x / dt = \omega_0 y, this reads:
# [ d y dt ] =  - \lambda -  \omega_0 x and dt x / dt = \omega_0 y.
# Hence adding the stochastic noise This leads to:
# dSST =  - \lambda SST dt -  \omega_0 DOT dt + SAT dW and dDOT = \omega_0 SST dt.
#
# Coded by Florian S�vellec <florian.sevellec@univ-brest.fr> 12may2023
#
# CODE
# ----
# Import libray

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from matplotlib import pyplot as plt

#%%
class idealized_ocean_model():
    default_seed=331381460666,

    def __init__(
            self,
            time_steps=1000,
            dt=365.25,
            df=1.15e-1,
            ) -> None:
        
        # Set up the settings of the model
        self.time_steps = time_steps 
        self.dt=dt
        self.df=df
        self.dW = np.sqrt(dt)

        # Set up the variables of the model
        self._create_stochastic_forcing_()
        self._create_variables_()


    def _create_variables_(self):
        pass

    def _create_stochastic_forcing_(self) :
        self.forcing = np.random.default_rng(seed=idealized_ocean_model.default_seed).normal(0, self.df, self.time_steps)

    def _add_setting_(self, name):
        for current_setting in settings:
            self.current_setting = settings[current_setting]

    def integrate(self):
        pass

class spunge_ocean_model(idealized_ocean_model):

    def __init__(
            self,
            time_steps=1000,
            dt=365.25,
            df = 1.15e-1,
            tau0=10 * 365.25,
        ):
        super().__init__(
            time_steps = time_steps, 
            dt=dt,
            df=df,
        )
        self._create_variables_()
        self._set_settings_(dict(tau0=tau0))

    def _create_variables_(self) :
        self.SAT = np.zeros(self.time_steps)
        self.SST = np.zeros(self.time_steps)
    
    def integrate(self) :
        for it in np.arange(1, self.time_steps):
            self.SAT[it] = self.forcing[it]
            self.SST[it] = self.SST[it-1] + (-l0 * SST_spg[it - 1]) * dt + (SAT[it - 1]) * dW
    

#%%


def integrate_idealized_ocean(
    time_steps=1000,
    dt=365.25,
    stochastic_forcing_intensity=1.15e-1,
    ocean_restoring_timescale=10 * 365.25,
    ocean_oscillation_timescale=24 * 365.25,
    save_path=None,
    seed=331381460666,
):
    # Note: the seed was produced with
    # >>> np.random.default_rng(seed=9384657836).integers(0, 1e12, 1)[0]
    # 331381460666

    # Numerical parameter
    nt = time_steps  # number of timestep
    dt = dt  # 365.25 # (days) time step
    dW = np.sqrt(dt)  # (sqrt (days)) Stochastic time step

    # Physical parameters
    tau0 = ocean_restoring_timescale  # (days) ocean restoring timescale
    per0 = ocean_oscillation_timescale  # (days) ocean oscillation timescale
    df = stochastic_forcing_intensity  # (K days-1/2) stochastic forcing intensity

    # Precomputation
    l0 = 2 / tau0  # (days-1) inverse restoring timescale
    o0 = 2 * 3.14 / per0  # (days-1) inverse oscillation timescale

    # Initialization
    SAT = np.zeros(nt)
    SST_spg = np.zeros(nt)
    SST_osc = np.zeros(nt)
    DOT_osc = np.zeros(nt)
    time = np.zeros(nt)

    normal = np.random.default_rng(seed=seed).normal(0, df, nt)

    # Time Loop
    for it in np.arange(1, nt):
        fi = normal[it]
        SAT[it] = fi
        time[it] = time[it - 1] + dt
        SST_spg[it] = (
            SST_spg[it - 1] + (-l0 * SST_spg[it - 1]) * dt + (SAT[it - 1]) * dW
        )
        SST_osc[it] = (
            SST_osc[it - 1]
            + (-l0 * SST_osc[it - 1] - o0 * DOT_osc[it - 1]) * dt
            + (SAT[it - 1]) * dW
        )
        DOT_osc[it] = DOT_osc[it - 1] + (o0 * SST_osc[it - 1]) * dt

    timep = time / 365.25  # (yr) TIME for plot

    ds = integration_to_netcdf(
        time=time,
        timep=timep,
        SAT=SAT,
        SST_spg=SST_spg,
        SST_osc=SST_osc,
        DOT_osc=DOT_osc,
        tau0=tau0,
        per0=per0,
        df=df,
    )

    if save_path is not None:
        ds.to_netcdf(save_path, mode="w")
    else:
        return ds


# create a xarray to store the data as netcdf file
def integration_to_netcdf(time, timep, SAT, SST_spg, SST_osc, DOT_osc, tau0, per0, df):
    models = ["sponge", "oscillator"]
    desciption = "Data created by a idealized ocean model.\n"
    desciption += "It includes two model runs."
    desciption += "surface_air_temperature is used as forcing for both models."
    desciption += "Models:"
    desciption += "- sponge: Sponge Ocean without interior oscillation."
    desciption += "- oscillator: Oscillting Ocean of two layer with interior oscillation between surface and deep ocean."

    ds = xr.Dataset(
        coords=dict(
            time=(["time"], time),
            time_years=(["time"], timep),
        ),
        data_vars=dict(
            surface_air_temperature=(["time"], SAT),
            sponge_sea_surface_temperature=(["time"], SST_spg),
            oscillator_sea_surface_temperature=(["time"], SST_osc),
            oscillator_deep_ocean_temperature=(["time"], DOT_osc),
        ),
        attrs=dict(
            coder="Florian Sévellec <florian.sevellec@univ-brest.fr>",
            description=desciption,
            ocean_restoring_timescale=f"{tau0} in days",
            ocean_oscillation_timescale=f"{per0} in days",
            stochastic_forcing_intensity=rf"{df} in (K days^(-1/2))",
        ),
    )
    return ds
