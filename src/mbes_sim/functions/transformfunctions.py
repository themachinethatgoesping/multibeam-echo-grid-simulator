# SPDX-FileCopyrightText: 2022 GEOMAR Helmholtz Centre for Ocean Research Kiel
#
# SPDX-License-Identifier: MPL-2.0

"""
Coordinate transformation functions used within the simulation
Functions are accelerated using numba (tested with numba 50.1)
"""

# ------------------- Imports -------------------
# python imports
import math
import numpy as np
import numpy.typing as npt

from scipy.spatial.transform import Rotation

from numba import njit,prange
import numba.types as ntypes

PARALLEL=False


# ------------------- Functions -------------------
@njit
def dxyz_to_range_txrx_angle(dx: float, dy: float, dz: float, degrees: bool = True):
    """Takes relative dx,dy and dz distances of a target towards the transducer and converts it to range, transmit-
    and receive- angle. Note: dx,dy and dz are in ship relative coordinates.

    Parameters
    ----------
    dx : float
        relative distance of target forward   [m, positive forward]
    dy : float
        relative distance of target starboard [m, positive starboard]
    dz : float
        relative distance of target downward  [m, positive downward]
    degrees : bool, optional
        if True return angles in degrees, else return angles in rad, by default True

    Returns
    -------
    (float, float, float)
        - range to target - [m]
        - alpha (transmit angle) - [°, -90° (aft) - 90° (bow)]
        - beta (receive angle) - [°, -90° (port) - 90° (starboard)]
    """

    r = math.sqrt(dx * dx + dy * dy + dz * dz)
    tx_angle =  math.asin(dx / r)
    rx_angle = -math.asin(dy / r)

    if degrees:
        tx_angle = math.degrees(tx_angle)
        rx_angle = math.degrees(rx_angle)

    return r, tx_angle, rx_angle


@njit (parallel=PARALLEL)
def get_target_ranges_txangles_rxangles(targets_dx: npt.ArrayLike,
                                        targets_dy: npt.ArrayLike,
                                        targets_dz: npt.ArrayLike,
                                        degrees: bool = True
                                        ) -> (npt.NDArray, npt.NDArray, npt.NDArray):
    """Takes relative dx,dy and dz distances of multiple targets towards the transducer and converts it to range,
    transmit- and receive- angle. Note: dx,dy and dz are in ship relative coordinates.

    Parameters
    ----------
    targets_dx : npt.ArrayLike
        relative distances of targets forward   [m, positive forward]
    targets_dy : npt.ArrayLike
        relative distances of targets starboard [m, positive starboard]
    targets_dz : npt.ArrayLike
        relative distances of targets downward  [m, positive downward]
    degrees : bool, optional
        if True return angles in degrees, else return angles in rad, by default True

    Returns
    -------
    (npt.NDArray, npt.NDArray, npt.NDArray)
        - range to target - [m]
        - alpha (transmit angle) - [°, -90° (aft) - 90° (bow)]
        - beta (receive angle) - [°, -90° (port) - 90° (starboard)]
    """

    # make sure all arrays have the same length
    assert len(targets_dx) == len(targets_dy) == len(targets_dz)

    # initialize arrays
    n_targets = len(targets_dx)
    ranges    = np.empty(n_targets, dtype=ntypes.float64)
    angles_tx = np.empty(n_targets, dtype=ntypes.float64)
    angles_rx = np.empty(n_targets, dtype=ntypes.float64)

    # compute ranges for each target
    for n in prange(n_targets):
        ranges[n], angles_tx[n], angles_rx[n] = dxyz_to_range_txrx_angle(targets_dx[n],
                                                                         targets_dy[n],
                                                                         targets_dz[n],
                                                                         degrees=degrees)

    return ranges, angles_tx, angles_rx

def rotate_points(X: npt.ArrayLike,
                  Y: npt.ArrayLike,
                  Z: npt.ArrayLike,
                  yaw: float,
                  pitch: float,
                  roll: float,
                  inverse: bool,
                  degrees: bool = True) -> (npt.NDArray, npt.NDArray, npt.NDArray):
    """Rotate points (or vectors of points) according to the yaw pitch roll euler angles. Note rotation is:
    z(yaw) than y(pitch) than x(roll).

    Parameters
    ----------
    X : npt.ArrayLike
        1D-array of X positions
    Y : npt.ArrayLike
        1D-array of Y positions
    Z : npt.ArrayLike
        1D-array of Z positions
    yaw : float
        rotation around z-axis
    pitch : float
        rotation around y-axis
    roll : float
        rotation around x-axis
    inverse : bool, optional
        if True apply inverse rotation
    degrees : bool, optional
        if True input angles must be given in degrees, else input angles must be given in rad, by default True

    Returns
    -------
    (npt.NDArray, npt.NDArray, npt.NDArray)        
        - new rotated X positions of points -
        - new rotated Y positions of points -
        - new rotated Z positions of points -
    """

    # make sure all arrays have the same length
    assert len(X) == len(Y) == len(Z)

    # initialize vectors
    vectors = np.empty((len(X), 3))
    vectors[:, 0] = np.array(X)
    vectors[:, 1] = np.array(Y)
    vectors[:, 2] = np.array(Z)

    # initialize euler rotation (zyx)
    rz = Rotation.from_euler('z', yaw, degrees=degrees)
    ry = Rotation.from_euler('y', pitch, degrees=degrees)
    rx = Rotation.from_euler('x', roll, degrees=degrees)

    r = rz * ry * rx
    if inverse:
        r = r.inv()

    # apply rotation to vector
    vectors = r.apply(vectors)

    return vectors[:, 0], vectors[:, 1], vectors[:, 2]

def rotate_points_2D(X: np.ndarray, Y: np.ndarray ,heading: np.ndarray ,inverse: bool, degrees: bool = True) -> (np.ndarray, np.ndarray): 
    """Rotate a list of X and Y points according to a heading

    Parameters
    ----------
    X : np.ndarray
        List of X coordinates
    Y : np.ndarray
        List of Y coordinates
    heading : np.ndarray
        heading in degrees
    inverse : bool, optional
        if True apply inverse rotation
    degrees : bool, optional
        if True input angles must be given in degrees, else input angles must be given in rad, by default True

    Returns
    -------
    (np.ndarray, np.ndarray)
        rotated points (X, Y)
    """

    vectors = np.empty((len(X),3))
    
    vectors[:,0] = np.array(X)
    vectors[:,1] = np.array(Y)
    
    r = Rotation.from_euler('z', heading,   degrees=degrees)

    if inverse:
        r = r.inv()
        
    vectors = r.apply(vectors)
    
    return vectors[:,1],vectors[:,0]



# ------------------- Main (for testing) -------------------
if __name__ == "__main__":
    """
    This is a simple test of the rotate points function
    """
    targets_x = [0]
    targets_y = [0]
    targets_z = [1]
    yaw = 0
    pitch = 0
    roll = 45
    ind = 0

    x, y, z = rotate_points(targets_x, targets_y, targets_z, yaw, pitch, roll, inverse=True)

    print(x[ind])
    print(y[ind])
    print(z[ind])
