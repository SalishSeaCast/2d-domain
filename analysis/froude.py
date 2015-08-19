# This is a module with functions that can be used to calculate the Froude
# number in a simple 2D system

# Nancy Soontiens, 2015

import numpy as np
import datetime
from salishsea_tools.nowcast import analyze


def find_mixed_depth_indices(n2, n2_thres=5e-6):
    """Finds the index of the mixed layer depth for each x-position.
    The mixed layer depth is chosen based on the lowest near-surface vertical
    grid cell where n2 >= n2_thres
    A resaonable value for n2_thres is 5e-6.
    If n2_thres = 'None' then the index of the maximum n2 is returned.

    n2 is the masked array of buoyancy frequencies with dimensions (depth, x)
    returns a list of indices of mixed layer depth cell for each x-position
    """
    if n2_thres == 'None':
        dinds = np.argmax(n2, axis=0)
    else:
        dinds = []
        for ii in np.arange(n2.shape[-1]):
            inds = np.where(n2[:, ii] >= n2_thres)
            # exlclude first vertical index less <=1 because the
            # buoyancy frequency is hard to define there
            if inds[0].size:
                inds = filter(lambda x: x > 1, inds[0])
                if inds:
                    dinds.append(min(inds))
                else:
                    dinds.append(0)  # if no mixed layer depth found, set to 0
            else:
                dinds.append(0)  # if no mixed layer depth found, set it to 0

    return dinds


def average_mixed_layer_depth(mixed_depths, xmin, xmax):
    """Averages the mixed layer depths over indices xmin and xmax
    mixed_depths is a 1d array of mixed layer depths

    returns the mean mixed layer depth in the defined region
    """
    mean_md = np.mean(mixed_depths[xmin:xmax+1])

    return mean_md


def mld_time_series(n2, deps, times, time_origin,
                    xmin=300, xmax=700, n2_thres=5e-6):
    """Calculates the mean mixed layer depth in a region defined by
    xmin and xmax over time
    n2 is the buoyancy frequency array with dimensions (time, depth, x)
    deps is the model depth array
    times is the model time_counter array
    time_origin is the model's time_origin as a datetime

    returns a list of mixed layer depths mlds and dates
    """

    mlds = []
    dates = []
    for t in np.arange(n2.shape[0]):
        dinds = find_mixed_depth_indices(n2[t, ...], n2_thres=n2_thres)
        mld = average_mixed_layer_depth(deps[dinds], xmin, xmax,)
        mlds.append(mld)
        dates.append(time_origin + datetime.timedelta(seconds=times[t]))

    return mlds, dates


def calculate_density(t, s):
    """Caluclates the density given temperature in deg C (t)
    and salinity in psu (s).

    returns the density as an array (rho)
    """

    rho = (
        999.842594 + 6.793952e-2 * t
        - 9.095290e-3 * t*t + 1.001685e-4 * t*t*t
        - 1.120083e-6 * t*t*t*t + 6.536332e-9 * t*t*t*t*t
        + 8.24493e-1 * s - 4.0899e-3 * t*s
        + 7.6438e-5 * t*t*s - 8.2467e-7 * t*t*t*s
        + 5.3875e-9 * t*t*t*t*s - 5.72466e-3 * s**1.5
        + 1.0227e-4 * t*s**1.5 - 1.6546e-6 * t*t*s**1.5
        + 4.8314e-4 * s*s
        )

    return rho


def calculate_internal_wave_speed(rho, deps, dinds):
    """Calculates the internal wave speed
    c = sqrt(g*(rho2-rho1)/rho2*h1)
    where g is acceleration due to gravity, rho2 is denisty of lower layer,
    rho1 is density of upper layer and h1 is thickness of upper layer.

    rho is the model density (shape is depth, x), deps is the array of depths
    and dinds is a list of indices that define the mixed layer depth.
    rho must be a masked array

    returns c, an array of internal wave speeds at each x-index in rho
    """
    # acceleration due to gravity (m/s^2)
    g = 9.81

    # calculate average density in upper and lower layers
    rho_1 = np.zeros((rho.shape[-1]))
    rho_2 = np.zeros((rho.shape[-1]))
    for ind, d in enumerate(dinds):
        rho_1[ind] = analyze.depth_average(rho[0:d+1, ind],
                                           deps[0:d+1], depth_axis=0)
        rho_2[ind] = analyze.depth_average(rho[d+1:, ind],
                                           deps[d+1:], depth_axis=0)
    # calculate mixed layer depth
    h_1 = deps[dinds]
    # calcuate wave speed
    c = np.sqrt(g*(rho_2-rho_1)/rho_2*h_1)

    return c


def depth_averaged_current(u, deps):
    """Calculates the depth averaged current

    u is the array with current speeds (shape is depth, x).
    u must be a masked array
    deps is the array of depths

    returns u_avg, the depths averaged current (shape x)
    """
    u_avg = analyze.depth_average(u, deps, depth_axis=0)

    return u_avg


def calculate_froude_number(n2, rho, u, deps, depsU, n2_thres=5e-6):
    """Calculates the Froude number

    n2, rho, u are buoyancy frequency, density and current arrays
    (shape depth, x)
    deps is the depth array
    depsU is the depth array at U poinnts

    returns the Froude number for each x-index
    """

    # calculate mixed layers
    dinds = find_mixed_depth_indices(n2, n2_thres=n2_thres)
    # calculate internal wave speed
    c = calculate_internal_wave_speed(rho, deps, dinds)
    # calculate depth averaged currents
    u_avg = depth_averaged_current(u, depsU)

    # Froude numer
    Fr = np.abs(u_avg)/c

    return Fr


def froude_time_series(n2, rho, u, deps, depsU, times, time_origin,
                       xmin=300, xmax=700, n2_thres=5e-6):
    """Calculates the Froude number time series

    n2, rho, u are buoyancy frequency, density and current arrays
    (shape time, depth, x)
    deps is the model depth array
    depsU is the model deps array at U points
    times is the model time_counter array
    time_origin is the mode's time_origin as a datetime

    xmin,xmax define the averaging area

    returns the Froude number for each time associated with dates

    """

    Frs = []
    dates = []
    for t in np.arange(n2.shape[0]):
        Fr = calculate_froude_number(n2[t, ...], rho[t, ...], u[t, ...],
                                     deps, depsU, n2_thres=n2_thres)
        Fr = np.mean(Fr[xmin:xmax+1])
        Frs.append(Fr)
        dates.append(time_origin + datetime.timedelta(seconds=times[t]))

    return Frs, dates
