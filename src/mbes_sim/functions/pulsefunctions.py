# SPDX-FileCopyrightText: 2022 GEOMAR Helmholtz Centre for Ocean Research Kiel
#
# SPDX-License-Identifier: MPL-2.0

"""
Functions to create pulse responses and computing the effective pulse length for the MBES simulation
 - the functions are accelerated using numba (tested with numba 0.50.1)
"""

# ------------------- Imports -------------------
# python imports
from numba import njit
import math
import numpy as np
import numpy.typing as npt
from scipy import signal

# project imports
import mbes_sim.functions.helperfunctions as hlp


# ------------------- Functions -------------------
@njit
def get_hann_pulse_response(t: float,
                            effective_pulse_length: float) -> float:
    """Returns the response of a hann shaped pulse with the specified effective pulse length at the relative position dt
    The HannPulse is symmetrical around t == 0.
        - at t == 0 it is 1
        - at t <= -effective_pulse_length it is 0
        - at t >=  effective_pulse_length it is 0

    The integral over the generated hann pulse is equal to the specified effective_pulse_length

    Parameters
    ----------
    t : float
        distance from pulse center (in seconds)
    effective_pulse_length : float
        Effective pulse length of the Hann pulse (in seconds)

    Returns
    -------
    float
        Hann pulse response value
    """

    if t > effective_pulse_length:
        return 0

    if t < -effective_pulse_length:
        return 0

    response = 0.5 * (1 - math.cos(math.pi * (t / effective_pulse_length + 1)))

    return response


@njit
def get_rect_pulse_response(t: float,
                            effective_pulse_length: float) -> float:
    """Returns the response of a rectangular shaped pulse with the specified effective pulse length at the relative
    position dt

    The HannPulse is symmetrical around t == 0.
        - at t == 0 it is 1
        - at t <= - effective_pulse_length it is 0
        - at t >=   effective_pulse_length it is 0
    The integral over the generated hann pulse is equal to the specified effective_pulse_length

    Parameters
    ----------
    t : float
        distance from pulse center (in seconds)
    effective_pulse_length : float
        Effective pulse length of the Hann pulse (in seconds)

    Returns
    -------
    float
        Hann pulse response value
    """

    effective_pulse_length_2 = effective_pulse_length * 0.5

    if -effective_pulse_length_2 <= t < effective_pulse_length_2:
        return 1

    return 0


@njit
def get_effective_pulse_length(pulse_response_array: npt.ArrayLike,
                              delta_time: float = None) -> float:
    """Takes a pulse response array and computes the effective pulse length by integrating over the power of the
    normalized response. (Normalized by the maximum response)

    Parameters
    ----------
    pulse_response_array : npt.ArrayLike
        np array including the pulse response
    delta_time : float, optional
        time difference between the pulse responses within the array (in seconds), by default None

    Returns
    -------
    float
        effective pulse length (in seconds)
    """

    pulse_sum = 0
    pulse_max = 0
    for i in range(len(pulse_response_array)):
        if pulse_response_array[i] > pulse_max:
            pulse_max = pulse_response_array[i]

        pulse_sum += pulse_response_array[i]

    return pulse_sum * delta_time / pulse_max


# ------------------- Main (for testing) -------------------
if __name__ == "__main__":
    """
    This is a simple test that plots a hann pulse and the corresponding ideal pulse from the effective pulse length
    """
    from matplotlib import pyplot as plt
    plt.ioff()

    fig = plt.figure("Pulses")
    fig.clear()

    ax = fig.subplots()

    times = np.linspace(-3, 3, 1000)
    pulse_distance = 1.2345
    hann_pulse = np.asarray([get_hann_pulse_response(n, pulse_distance) for n in times]) * 10

    ax.plot(times, hann_pulse, label='hann')

    dtime = times[1] - times[0]

    hann_pulse_signal = signal.hann(2 * hlp.round_int(1 + pulse_distance / dtime))
    times = np.array([t - (len(hann_pulse) - 1) / 2 for t, _ in enumerate(hann_pulse)]) * dtime
    ax.plot(times, hann_pulse, label='new')

    rect_pulse = np.asarray([get_rect_pulse_response(n, pulse_distance) for n in times]) * 10

    ax.plot(times, rect_pulse, label='ideal')

    print(get_effective_pulse_length(hann_pulse,
                                    delta_time=dtime))

    print(get_effective_pulse_length(hann_pulse_signal,
                                    delta_time=dtime))

    print(get_effective_pulse_length(rect_pulse,
                                    delta_time=dtime))

    ax.legend()
    plt.show()
