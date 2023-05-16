# SPDX-FileCopyrightText: 2022 GEOMAR Helmholtz Centre for Ocean Research Kiel
# SPDX-FileCopyrightText: 2012 Dr. Andrew Greensted (http://www.labbookpages.co.uk/audio/beamforming/delaySum.html)
#
# SPDX-License-Identifier: MPL-2.0

"""
Functions to create beampattern for the MBES simulation
 - the functions are accelerated using numba 0.50.1
 - The functions usually return or operate on a vector of angles
   that represents the angles from -90° to 90° in ANGLE_RESOLUTION steps
   Use get_angle_index to transform
 - use the BEAMPATTERN_ANGLES_DEGREES and BEAMPATTERN_ANGLES_RAD structures
   to convert indices to angles
"""

# ------------------- Imports -------------------
import math
import numpy as np
from scipy import signal

from numba import njit, prange
import numba.types as ntypes

import mbes_sim.functions.helperfunctions as hlp

# ------------------- Processing setup -------------------
# This is the angular resolution used for the beampattern array
# 18000*3 = 0.00333333°
ANGLE_RESOLUTION: int = 18000 * 3 + 1


def init(angle_resolution=ANGLE_RESOLUTION):
    """Initialize the beampattern functions on import

    Parameters
    ----------
    angle_resolution : number of angles in the beampattern array used in the simulation
    """

    global ANGLE_RESOLUTION
    global DELTA_ANGLE_DEG
    global DELTA_ANGLE_RAD
    global BEAMPATTERN_ANGLES_DEGREES
    global BEAMPATTERN_ANGLES_RAD
    ANGLE_RESOLUTION = angle_resolution

    # ------------------- Precomputing variables -------------------
    DELTA_ANGLE_DEG = 180 / (ANGLE_RESOLUTION - 1)
    DELTA_ANGLE_RAD = hlp.M_PI / (ANGLE_RESOLUTION - 1)

    # precompute beampattern angle structure
    BEAMPATTERN_ANGLES_DEGREES = np.asarray(
        [
            angle_index * DELTA_ANGLE_DEG - 90.0
            for angle_index in range(ANGLE_RESOLUTION)
        ],
        dtype=np.float64,
    )
    BEAMPATTERN_ANGLES_RAD = np.asarray(
        [
            angle_index * DELTA_ANGLE_RAD - hlp.M_PI_2
            for angle_index in range(ANGLE_RESOLUTION)
        ],
        dtype=np.float64,
    )


init()

# ------------------- Helper functions -------------------
# operate on index/angles
@njit
def beampatternindex_from_angle(angle_degrees: float) -> int:
    """Convert a given angle (in degrees) to a index in the beampattern array (nearest neighbor)

    Index runs from -90° -> 90° in ANGLE_RESOLUTION steps

    Parameters
    ----------
    angle_degrees : float
        Input angle to be converted [°, -90° <= angle_degrees <= 90°]

    Returns
    -------
    int
        angle_index - [0 <= angle_index <= ANGLE_RESOLUTION - 1]
    """

    assert angle_degrees >= -90.0, "Angle (%d) outside +-90° "
    assert angle_degrees <= 90.0, "Angle (%d) outside +-90°"

    return hlp.round_int((angle_degrees + 90.0) / DELTA_ANGLE_DEG)


@njit
def beampatternindex_from_angle_fraction(angle_degrees: float) -> float:
    """Convert a given angle (in degrees) to a index in the beampattern array.
    Preserve the fraction instead of rounding to the nearest integer Index runs
    from -90° -> 90° in ANGLE_RESOLUTION steps

    Parameters
    ----------
    angle_degrees : float
        Input angle to be converted [°, -90° <= angle_degrees <= 90°]

    Returns
    -------
    float
        angle_index_fraction - [float, 0 <= angle_index <= ANGLE_RESOLUTION - 1]
    """

    assert angle_degrees >= -90.0, "Angle (%d) outside +-90° "
    assert angle_degrees <= 90.0, "Angle (%d) outside +-90°"

    return (angle_degrees + 90.0) / DELTA_ANGLE_DEG


@njit
def beampattern_value_linear_interploation(
    angle_degrees: float, beampattern: np.ndarray
) -> float:
    """Takes a given input angle and returns the beampattern response according to the given
    beampattern structure. The response is the linear interpolation between the two closest
    angle indices within the beampattern_structure

    Parameters
    ----------
    angle_degrees : float
        Input angle to be converted [°, -90° <= angle_degrees <= 90°]
    beampattern : np.ndarray
        np array including the beampattern response [float, len == ANGLE_RESOLUTION]

    Returns
    -------
    float
        previous  interpolated value
    """

    assert angle_degrees >= -90.0, "Angle outside +-90°"
    assert angle_degrees <= 90.0, "Angle outside +-90°"
    assert len(beampattern) == ANGLE_RESOLUTION

    # get angle interpolation as fraction
    angle_index_float = beampatternindex_from_angle_fraction(angle_degrees)

    # previous and following index
    ai0 = int(math.floor(angle_index_float))
    ai1 = ai0 + 1

    # fraction between indices
    fraction = angle_index_float - ai0

    # interpolate
    m = beampattern[ai1] - beampattern[ai0]
    c = beampattern[ai0]

    return m * fraction + c


# --------- get beam width from beampattern ---------
@njit
def get_3db_beamwidth_from_beampattern(
    beampattern: np.ndarray, degrees: bool = True
) -> float:
    """Takes a db beampattern array and searches for the 3dB beam width.
    WARNING: Use for info only. This function is neither performant nor precise

    Parameters
    ----------
    beampattern : np.ndarray
        np array including the beampattern response as power value [float, len == ANGLE_RESOLUTION]
    degrees : bool, optional
        if False output will be in radians, by default True

    Returns
    -------
    float
        3dB beamwidth - [degrees or radians]
    """

    x1 = np.nan
    x2 = np.nan

    if degrees:
        angles = BEAMPATTERN_ANGLES_DEGREES
    else:
        angles = BEAMPATTERN_ANGLES_RAD

    db_array = beampattern.copy() / max(beampattern)

    min_3db = math.pow(10, -0.3)

    for t in range(len(db_array)):
        val = db_array[t]
        a = angles[t]
        if not np.isfinite(x1):
            if val >= min_3db:
                x1 = a
        elif not np.isfinite(x2):
            if val <= min_3db:
                x2 = a

    return x2 - x1


@njit
def get_equivalent_beam_angle_from_beampattern(
    beampattern: np.ndarray, delta_angle: float = None, degrees: bool = True
) -> float:
    """Takes a linear beampattern array and computes the equivalent beam angle by
    integrating over the power of the normalized linear response.

    Parameters
    ----------
    beampattern : np.ndarray
        np array including the beampattern response (linear) [float, len == ANGLE_RESOLUTION]
    delta_angle : float, optional
        angle steps between beampattern indices defaults to DELTA_ANGLE_RAD, by default None
    degrees : bool, optional
        if False output will be in radians, by default True

    Returns
    -------
    float
        2D equivalent beam angle - [degrees or radians]
    """

    if delta_angle is None:
        delta_angle = DELTA_ANGLE_RAD

    # integrate beampattern energy
    beampattern_power = 0
    beampattern_max = 0
    for i in range(len(beampattern)):
        if beampattern[i] > beampattern_max:
            beampattern_max = beampattern[i]

        beampattern_power += beampattern[i]

    # normalize by the maximum value of the beampattern (should be one)
    equivalent_beam_angle = beampattern_power * delta_angle / (beampattern_max)

    if degrees:
        return equivalent_beam_angle / hlp.M_PI_180
    return equivalent_beam_angle


# --------- generate Beampattern ----------
@njit(parallel=True)
def generate_delay_and_sum_beampattern(
    steering_angle_degrees: float,
    window: np.ndarray,
    f0: float = 100000,
    freq: float = 80000,
) -> np.ndarray:
    """Create beampattern response for an indivdual beam using 2D delay and sum beamforming.
    Adapted from Dr. Andrew Greensted http://www.labbookpages.co.uk/audio/beamforming/delaySum.html

    Parameters
    ----------
    steering_angle_degrees : float
        steering angle of the beam [°, -90° <= angle_degrees <= 90°]
    window : np.ndarray
        Shading window used for the array as an 1D array. Length of window determines
        also the number of elements! [example1: np.ones(num_elements),
        example2: scipy.signal.hann(num_elements)]
    f0 : float, optional
        Base frequency of the transducer. Is used to compute the element spacing (lambda/2) [Hz], by default 100000
    freq : float, optional
        Frequency of the signal/acoustic pulse. Is used to compute the arrival phase delay [Hz], by default 80000

    Returns
    -------
    np.ndarray
        beampattern - Array with beam response value (Power) normalized to 1 (max)
    """

    assert steering_angle_degrees >= -90.0, "Angle outside +-90°"
    assert steering_angle_degrees <= 90.0, "Angle outside +-90°"
    assert f0 > 0, "Frequency must be larger than 0"
    assert freq > 0, "Frequency must be larger than 0"

    window_power = np.nansum(window)
    window_power *= window_power

    # compute spacing
    sound_speed = 1500.0  # m/s
    lamda = sound_speed / f0  # f0 is the frequency the transducer was designed for
    spacing = lamda / 2  # Element separation in meters

    steering_angle_degrees = hlp.M_PI_180 * steering_angle_degrees

    beampattern = np.zeros(ANGLE_RESOLUTION, dtype=ntypes.float64)

    # Iterate through arrival angles to pointsgenerate response
    for a in prange(ANGLE_RESOLUTION):

        real_sum = 0
        imag_sum = 0

        # Time it takes the wave to move from one element to the next.
        # Is corrected by the artificial delay added by steering the array
        delta_delay = (
            spacing
            * (math.sin(BEAMPATTERN_ANGLES_RAD[a]) - math.sin(steering_angle_degrees))
            / sound_speed
        )

        # Phase change of wave when moving from one element to the next
        delta_phase = hlp.M_2_PI * freq * delta_delay

        # Iterate through array elements
        for element_nr, element_response in enumerate(window):

            # Add Waves according to phase difference
            real_sum += math.cos(element_nr * delta_phase) * element_response
            imag_sum += math.sin(element_nr * delta_phase) * element_response

        response = (real_sum * real_sum + imag_sum * imag_sum) / window_power

        beampattern[a] = response

    return beampattern


@njit(parallel=True)
def generate_idealized_beampattern(
    steering_angle_degrees: float,
    window: np.ndarray,
    f0: float = 100000,
    freq: float = 80000,
    equivalent_beam_angle_degrees: float = np.nan,
    preserve_equivalent_beam_angle=True,
) -> np.ndarray:
    """Create idealized rectangular beampattern response for an indivdual beam with the
    specified equivalent beam angle. If no equivalent beam angle is provided, a 2D delay 
    and sum beamforming is performed to determine the equivalent beam angle the beam 
    would have. The rectangular beam pattern will be 1 inside the equivalent beam angle 
    and 0 outside. If the preserve equivalent beam angle option is turned on, the outer 
    parts of the beam will not be 1 but rounded such that the equivalent beam angle is 
    preserved (avoids the sampling error). Otherwise the beam width is rounded.


    Parameters
    ----------
    steering_angle_degrees : float
        steering angle of the beam [°, -90° <= angle_degrees <= 90°]
    window : np.ndarray
        Shading window used for the array as an 1D array. Length of window determines 
        also the number of elements! [example1: np.ones(num_elements),
                                      example2: scipy.signal.hann(num_elements)]
    f0 : float, optional
        Base frequency of the transducer. Is used to compute the element spacing (lambda/2) [Hz], by default 100000
    freq : float, optional
        Frequency of the signal/acoustic pulse. Is used to compute the arrival phase delay [Hz], by default 80000
    equivalent_beam_angle_degrees : float, optional
        Specify a equivalent beam angle to avoid computing a delay and sum beampattern, by default np.nan
    preserve_equivalent_beam_angle : bool, optional
        Preserve the equivalent beam angle (outer parts will be a fraction between 0 and 1), by default True

    Returns
    -------
    np.ndarray
        - beampattern - Array with beam response as linear value normalized to 1 (max)
    #   - beampattern_db - Array with beam response as db power value normalized to 0 (max)
    """

    assert steering_angle_degrees >= -90.0, "Angle outside +-90°"
    assert steering_angle_degrees <= 90.0, "Angle outside +-90°"
    assert f0 > 0, "Frequency must be larger than 0"
    assert freq > 0, "Frequency must be larger than 0"


    # get equivalent beam angle of true beam
    if not np.isfinite(equivalent_beam_angle_degrees):
        real_beampattern = generate_delay_and_sum_beampattern(
            steering_angle_degrees, window, f0, freq
        )

        equivalent_beam_angle_degrees = get_equivalent_beam_angle_from_beampattern(
            real_beampattern
        )
    else:
        assert equivalent_beam_angle_degrees > 0, "Negative equivalent beam angle"

    beampattern = np.zeros(ANGLE_RESOLUTION, dtype=ntypes.float64)

    # compute first and last position
    if not preserve_equivalent_beam_angle:
        # inner and outer values are rounded
        a0 = beampatternindex_from_angle(
            steering_angle_degrees - equivalent_beam_angle_degrees / 2.0
        )
        a1 = beampatternindex_from_angle(
            steering_angle_degrees + equivalent_beam_angle_degrees / 2.0
        )

        # Iterate through arrival angle points
        for a in prange(a0, a1 + 1):
            beampattern[a] = 1

    else:
        # interpolate the outer parts
        angle_index_float0 = beampatternindex_from_angle_fraction(
            (steering_angle_degrees - equivalent_beam_angle_degrees / 2.0)
        )
        angle_index_float1 = beampatternindex_from_angle_fraction(
            (steering_angle_degrees + equivalent_beam_angle_degrees / 2.0)
        )

        # previous and following index
        ai00 = int(math.floor(angle_index_float0))
        ai01 = ai00 + 1
        ai10 = int(math.floor(angle_index_float1))
        ai11 = ai10 + 1

        # fraction between indices
        fraction0 = angle_index_float0 % 1
        fraction1 = angle_index_float1 % 1

        # Iterate through arrival angle points
        for a in prange(ai01 + 1, ai10):
            beampattern[a] = 1

        # for outer samples interpolation
        if fraction0 < 0.5:
            # beampattern[ai00] = math.sqrt(0.5 - fraction0) # sqrt to preserves power and not linear sum of the beam
            beampattern[ai00] = (
                0.5 - fraction0
            )  # sqrt to preserves power and not linear sum of the beam
            beampattern[ai01] = 1

        else:
            beampattern[ai00] = 0
            # beampattern[ai01] = math.sqrt(1.5 - fraction0)
            beampattern[ai01] = 1.5 - fraction0

        if fraction1 < 0.5:
            # beampattern[ai10] = math.sqrt(0.5 + fraction1)
            beampattern[ai10] = 0.5 + fraction1
            beampattern[ai11] = 0

        else:
            beampattern[ai10] = 1
            # beampattern[ai11] = math.sqrt(fraction1 - 0.5)
            beampattern[ai11] = fraction1 - 0.5

    return beampattern


# Some plotting
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from tqdm.auto import tqdm

    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    ax.set_thetalim(-math.pi, 0)
    # ax.append(fig.add_subplot(projection='polar'))
    # ax.append(fig.add_subplot(projection='rect'))

    elements = 28

    crazy = np.zeros(elements)

    windows = [
        [np.ones(elements), "rect"],
        [signal.windows.hann(elements), "hann"],
        [signal.windows.exponential(elements, tau=elements / 2), "exp elements/2"],
    ]

    angles = [i for i in np.linspace(-60, 60, int(round(120 / 3)) + 1)]

    for angle in angles:
        for win, name in windows:
            bp = generate_delay_and_sum_beampattern(angle, window=win)
            ax.plot(
                BEAMPATTERN_ANGLES_RAD - math.pi / 2,
                hlp.to_db(bp, min_db_value=-25),
                label=name,
                c="grey",
            )
            print(
                name + " equivalent beam angle:",
                get_equivalent_beam_angle_from_beampattern(bp),
            )

    ax.set_title("beam pattern")
    fig.show()

    pause = input("press any key to continue")
