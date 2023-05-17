# SPDX-FileCopyrightText: 2022 GEOMAR Helmholtz Centre for Ocean Research Kiel
#
# SPDX-License-Identifier: MPL-2.0

from numba import njit, prange, jit
import numba.types as ntypes
import numpy as np
import math

from copy import deepcopy
from tqdm.auto import tqdm

# Project imports
import mbes_sim.functions.beampatternfunctions as bf
import mbes_sim.functions.pulsefunctions as pf
import mbes_sim.functions.transformfunctions as tf
import mbes_sim.functions.helperfunctions as hlp

"""
Functions to create, raytrace and process water column images (WCI).
"""

#parallel = True causes crash when using the function a threading work space
RUN_PARALLEL=False

#@jit(parallel=True,forceobj=True)
#@njit(parallel=RUN_PARALLEL)
def get_wci_xyz(beamsteeringangles_radians: np.ndarray,
                sampleranges: np.ndarray,
                swath_angle_radians: np.ndarray,
                pos_x:float,pos_y:float,pos_z:float,
                yaw_radians:float,
                pitch_radians:float,
                roll_radians:float) -> (np.ndarray,np.ndarray,np.ndarray):
    """Generate xyz coordinates for a water column image (WCI). This assumes a homogeneous water column.

    Parameters
    ----------
    beamsteeringangles_radians : np.ndarray
        beamsteering angles of the WCI (in radians)
    sampleranges : np.ndarray
        ranges of the samples (in meters)
    swath_angle_radians : np.ndarray
        forward angle of the swath (in radians)
    pos_x : float
        x position of the transducer (in meters)
    pos_y : float
        y position of the transducer (in meters)
    pos_z : float
        z position of the transducer (in meters)
    yaw_radians : float
        yaw angle of the transducer (in radians)
    pitch_radians : float
        pitch angle of the transducer (in radians)
    roll_radians : float
        roll angle of the transducer (in radians)

    Returns
    -------
    (np.ndarray,np.ndarray,np.ndarray)
        X,Y,Z coordinates of the WCI, dimensions of X,Y,Z are (len(beamsteeringangles_radians), len(sampleranges))
    """

    X = np.zeros((len(beamsteeringangles_radians), len(sampleranges)),dtype = np.float64)
    Y = np.zeros((len(beamsteeringangles_radians), len(sampleranges)),dtype = np.float64)
    Z = np.zeros((len(beamsteeringangles_radians), len(sampleranges)),dtype = np.float64)


    for bnr in prange(len(beamsteeringangles_radians)):
        #generate base vector form beam
        bz = sampleranges
        bx = np.zeros(len(sampleranges))
        by = np.zeros(len(sampleranges))

        # first rotate vector according to swath_angle and beamsteering angle of the respective beam
        X[bnr], Y[bnr], Z[bnr] = tf.rotate_points(bx, by, bz,
                                      0, swath_angle_radians, beamsteeringangles_radians[bnr],
                                      degrees=False, inverse=False)

        # then rotate vector according to transducer orientation
        X[bnr], Y[bnr], Z[bnr] = tf.rotate_points(X[bnr], Y[bnr], Z[bnr],
                                                  yaw_radians, pitch_radians, roll_radians,
                                                  degrees=False, inverse=False)

        X[bnr] += pos_x
        Y[bnr] += pos_y
        Z[bnr] += pos_z


    return X,Y,Z

@njit(parallel=RUN_PARALLEL)
def get_sample_volume(beamsteeringangles_radians: np.ndarray,
                      sampleranges: np.ndarray,
                      equivalent_beam_angle_tx_radians: float,
                      equivalent_beam_angles_rx_radians: np.ndarray,
                      effective_pulse_length: float
               ) -> np.ndarray:
    """Compute sample volumes for a water column image (WCI). The dimensions of the returned array are
    (len(beamsteeringangles_radians), len(sampleranges)).

    Parameters
    ----------
    beamsteeringangles_radians : np.ndarray
        Beamsteering angles of the WCI (in radians)
    sampleranges : np.ndarray
        Sample ranges (in meters)
    equivalent_beam_angle_tx_radians : float
        equivalent beam angle of the transmitter (in radians)
    equivalent_beam_angles_rx_radians : np.ndarray
        equivalent beam angles of the receiver beams(in radians)
    effective_pulse_length : float
        effective pulse length (in seconds)

    Returns
    -------
    np.ndarray
        Array with sample volume size of the WCI, dimensions of the array are (len(beamsteeringangles_radians), len(sampleranges))
    """

    SampleVolume = np.zeros((len(beamsteeringangles_radians), len(sampleranges)),dtype = np.float64)

    for bnr in prange(beamsteeringangles_radians.shape[0]):
        for snr,range in enumerate(sampleranges):
            SampleVolume[bnr][snr] = range * equivalent_beam_angle_tx_radians \
                                     * range * equivalent_beam_angles_rx_radians[bnr] \
                                     * effective_pulse_length

    return SampleVolume

@njit(parallel=RUN_PARALLEL)
def create_wci(target_ranges: np.ndarray, 
               target_angles_tx: np.ndarray, 
               target_angles_rx: np.ndarray, 
               target_values: np.ndarray,
               beamsteeringangles: np.ndarray,
               sampleranges: np.ndarray,
               beampattern_tx: np.ndarray,
               beampattern_rx: np.ndarray,
               pulse_response_function, 
               effective_pulse_length: float
               ) -> np.ndarray:
    """Create a simulated water column image using the given relative target positions and response functions

    Parameters
    ----------
    target_ranges : np.ndarray
        Range of targets from the transducer (in meters)
    target_angles_tx : np.ndarray
        Along track angles of the targets (in radians)
    target_angles_rx : np.ndarray
        Across track angle of the targets (in radians)
    target_values : np.ndarray
        Backscattering values of the targets
    beamsteeringangles : np.ndarray
        Beamsteering angles of the WCI (in radians)
    sampleranges : np.ndarray
        Ranges of the WCI samples (in meters)
    beampattern_tx : np.ndarray
        Beampattern array of the transmitter
    beampattern_rx : np.ndarray
        Beampattern array of the receiver
    pulse_response_function : _type_
        Pulse response function of the transmitted pulse
    effective_pulse_length : float
        Effective pulse length (in seconds)

    Returns
    -------
    np.ndarray
        Simulated water column image (integrated recording backscattering cross-section), 
        dimensions of the array are (len(beamsteeringangles_radians), len(sampleranges)
    """

    assert target_ranges.shape == target_angles_tx.shape == target_angles_rx.shape == target_values.shape

    n_targets = target_ranges.shape[0]
    n_beams = beamsteeringangles.shape[0]
    n_samples = sampleranges.shape[0]

    WCI = np.zeros((n_beams, n_samples),dtype = np.float64)

    # get swath response for each target
    target_responses_tx = np.zeros(n_targets)

    for tnr, angle_tx in enumerate(target_angles_tx):
        target_responses_tx[tnr] = bf.beampattern_value_linear_interploation(angle_tx,beampattern_tx)

    # get beam responses
    target_responses_rx = np.zeros((n_beams,n_targets),dtype = np.float64)

    for bnr in prange(n_beams):
        for tnr, angle_rx in enumerate(target_angles_rx):
            target_responses_rx[bnr][tnr] = bf.beampattern_value_linear_interploation(angle_rx,
                                                                                    beampattern_rx[bnr])

    pulse_response_vec = np.zeros(len(sampleranges))
    for snr in prange(len(sampleranges)):

        sample_range = sampleranges[snr]

        target_min_range = sample_range - effective_pulse_length * 2
        target_max_range = sample_range + effective_pulse_length * 2

        for tnr,target_range, in enumerate(target_ranges):

            if target_min_range < target_range < target_max_range:
                pulse_response = pulse_response_function(target_range-sample_range,effective_pulse_length)

                pulse_response_vec[snr] += pulse_response

                for bnr in range(n_beams):
                    WCI[bnr][snr] += target_responses_tx[tnr] \
                                    * target_responses_rx[bnr][tnr] \
                                    * pulse_response \
                                    * target_values[tnr]

    return WCI


if __name__ == "__main__":

    # the effect of the functions is tested together with the mbes class

    import mbes_sim.functions.create_bubblesfunctions as bubbles
    import mbes_sim.functions.transformfunctions as tf
    import mbes_sim.mbes as mb

    bubbleGenerator = bubbles.BubbleGenerator()

    targets = \
        bubbleGenerator.generate_bubbles_within_cylindrical_section_along_path(
            start_x=-0.1,
            end_x=0.1,
            min_range=10,
            max_range=120,
            min_beamsteeringangle=-60,
            max_beamsteeringangle=45,
            min_z=5,
            # max_z=90,
            min_y=-90,
            max_y=60,
            nbubbles=10,
            uniform_likelyhood_along_x_range_angle=False
        )

    targets_ranges, targets_tx, targets_rx = tf.get_target_ranges_txangles_rxangles(*targets. xyzval_vectors()[:-1])

    multibeam = mb.Multibeam(progress=True)

    TS_new = create_wci(targets_ranges, targets_tx, targets_rx, targets.val,
                           multibeam.beamsteeringangles_radians,
                           multibeam.sampleranges,
                           multibeam.beampattern_tx,
                           multibeam.beampattern_beam_rx,
                           pf.get_hann_pulse_response, multibeam.effective_pulse_length)

    VOL_new = get_sample_volume(multibeam.beamsteeringangles_radians,
                                multibeam.sampleranges,
                                multibeam.equivalent_beam_angle_tx_radians,
                                multibeam.equivalent_beam_angles_beam_rx_radians,
                                multibeam.effective_pulse_length)

    X,Y,Z = get_wci_xyz(multibeam.beamsteeringangles_radians,
                              multibeam.sampleranges,
                              math.radians(multibeam.transmit_steeringangle_degrees),
                              multibeam.pos_x, multibeam.pos_y, multibeam.pos_z,
                              math.radians(multibeam.yaw_degrees),
                              math.radians(multibeam.pitch_degrees),
                              math.radians(multibeam.roll_degrees))


    beamspacing = multibeam.beamsteeringangles_degrees[1] - multibeam.beamsteeringangles_degrees[0]

    extent = [multibeam.beamsteeringangles_degrees[0] - beamspacing * 0.5,
              multibeam.beamsteeringangles_degrees[-1] + beamspacing * 0.5,
              multibeam.sampleranges[-1] + multibeam.effective_pulse_length * 0.5,
              multibeam.sampleranges[0] - multibeam.effective_pulse_length * 0.5]

    from matplotlib import pyplot as plt

    fig = plt.figure('echo')
    fig.clear()
    fig.show()

    ts = 10 * np.log10(TS_new+0.0000000001)
    ts = TS_new

    axes = fig.subplots(ncols = 2)

    ax = axes[0]
    ax.set_title('ts')
    ax.imshow(ts.transpose(),extent=extent)
    ax.scatter(targets_rx,targets_ranges)

    ax = axes[1]
    ax.set_title('volume')
    ax.imshow(VOL_new.transpose(),extent=extent)
    ax.scatter(targets_rx,targets_ranges)

    plt.show()
