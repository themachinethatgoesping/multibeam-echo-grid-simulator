# SPDX-FileCopyrightText: 2022 GEOMAR Helmholtz Centre for Ocean Research Kiel
#
# SPDX-License-Identifier: MPL-2.0

"""
Simple helper functions used within the simulation
Functions are accelerated using numba 50.1
"""

# ------------------- Imports -------------------
import math
import numpy as np

from numba import njit
from scipy.spatial.transform import Rotation

# ---------- precompute constants ---------
M_PI = math.pi
M_PI_2 = math.pi / 2
M_2_PI = 2 * math.pi
M_PI_180 = math.pi / 180

MIN_DB_VALUE: float = -50.

# ------------------- Functions -------------------
@njit
def round_int(val: float) -> int:
    """Helper function: rounds float to int using decimal rounding instead of pythons bankers rounding

    Parameters
    ----------
    val : float
        input value

    Returns
    -------
    int
        rounded value
    """

    return int(math.copysign(math.floor(math.fabs(val) + 0.5), val))

@njit
def to_db(beampattern: np.ndarray,
          min_db_value = MIN_DB_VALUE):
    """Convert an array beampattern to dB

    Parameters
    ----------
    beampattern : np.ndarray
        Beampattern to convert
    min_db_value : _type_, optional
        min value used when value is 0, by default MIN_DB_VALUE

    Returns
    -------
    _type_
        Beampattern in dB
    """

    beampattern_db = beampattern.copy()

    for i,val in np.ndenumerate(beampattern):
        v = 10 * math.log10(val)

        if v > min_db_value:
            beampattern_db[i] = v
        else:
            beampattern_db[i] = min_db_value

    return beampattern_db

def compute_max_pingrate(depth: float, mbes_rx_swath_angle_degrees:float = 120):
    """Compute the maximum pingrate for a given depth and swath angle assuming a 2-way travel time and a sound speed of 1500 m/s

    Parameters
    ----------
    depth : float
        Depth in meters
    mbes_rx_swath_angle_degrees : float, optional
        Maximum swath angle of the mbes, by default 120

    Returns
    -------
    _type_
        maximum pingrate, assuming the MBES waits for the return signal of the outer beams
    """
    c = 1500
    range = depth / math.cos(math.radians(mbes_rx_swath_angle_degrees / 2))

    return_time = 2 * range / c

    return 1 / return_time

def compute_ping_spacing_from_pingrate(pingrate: float, speed_knots: float):
    """Compute the ping spacing from the pingrate and the speed of the ship

    Parameters
    ----------
    pingrate : float
        pingrate in Hz
    speed_knots : float
        vessel speed in knots

    Returns
    -------
    float
        ping spacing in meters
    """
    
    speed = speed_knots * 0.514444

    return speed / pingrate

def compute_ping_spacing_from_num_pings(num_pings : int, min_x : float, max_x : float):
    """Compute the ping spacing from the number of pings between min_x and max_x

    Parameters
    ----------
    num_pings : _type_
        number of pings to distribute between min_x and max_x
    min_x : _type_
        start x value
    max_x : _type_
        end x value

    Returns
    -------
    float
        number of pings that fit between min_x and max_x
    """
    return abs(max_x - min_x) / (num_pings - 1)

def compute_course(northing : np.ndarray, easting : np.ndarray) -> np.ndarray:
    """compute the course from a list of northing and easting values

    Parameters
    ----------
    northing : np.ndarray,
        northing in meters
    easting : np.ndarray,
        easting in meters

    Returns
    -------
    np.ndarray
        course in degrees (0 is north, 90 is east)
    """

    dx = [northing[1] - northing[0]]
    dy = [easting[1] - easting[0]]
    for i in range(1,len(northing)-1):
        dx.append(northing[i+1] - northing[i-1])
        dy.append(easting[i+1] - easting[i-1])
        
    dx.append(northing[-1] - northing[-2])
    dy.append(easting[-1] - easting[-2])
        
    course = np.degrees(np.arctan2(dx,dy))
    
    return course

# Some testing
if __name__ == "__main__":
    print("Nothing to do")
