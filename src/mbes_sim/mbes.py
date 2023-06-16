# SPDX-FileCopyrightText: 2022 GEOMAR Helmholtz Centre for Ocean Research Kiel
#
# SPDX-License-Identifier: MPL-2.0

"""
Class to initialize a virtual multibeam system for the MBES simulation
"""

# ------------------- Imports -------------------
import math
import numpy as np
from scipy import signal

from numba import njit, prange
import numba.types as ntypes

import matplotlib as mpl
#mpl.rcParams['figure.dpi'] = 200

#from tqdm.auto import tqdm
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

# Project imports
import mbes_sim.functions.helperfunctions as hlp
import mbes_sim.functions.beampatternfunctions as bf
import mbes_sim.functions.pulsefunctions as pf
import mbes_sim.functions.transformfunctions as tf
import mbes_sim.functions.create_echofunctions as ef
import mbes_sim.functions.create_bubblesfunctions as bubbles

def init(angle_resolution):
    print("Initializing Beampattern functions")
    bf.init(angle_resolution)
    print("ANGLE_RESOLUTION:",bf.ANGLE_RESOLUTION)


class Multibeam(object):
    """Class to initialize a virtual multibeam system for the MBES simulation
    """

    def __init__(self,
                 beamsteeringangles_degrees: np.ndarray = np.linspace(-60,60,255),
                 sampleranges: np.ndarray = np.linspace(1, 120, 376),
                 effective_pulse_length: float = 0.375, #0.36845,
                 window: np.ndarray = np.ones(128),
                 f0: float = 100000,
                 freq: float = 80000,
                 idealized_beampattern_preserve_equivalent_beam_angle: bool = True,
                 transmit_steeringangle_degrees = 0,
                 progress: bool = False
                 ):
        """The Multibeam class holds processing parameters for the MBES simulation

        Parameters
        ----------
        beamsteeringangles_degrees : np.ndarray, optional
            Array with the beamsteering angles [°] of all beams. Does also determine the used number of beams, by default np.linspace(-60,60,255)
        sampleranges : np.ndarray, optional
            Array with the sampleranges [m]. Does also determine the used number of samples, by default np.linspace(1, 120, 376)
        effective_pulse_length : float, optional
            Effective pulse length of the pulse used in meter! [m], by default 0.375
        window : np.ndarray, optional
            Shading window used for the array as an 1D array. Length of window determines also the number of elements! 
            - example1: np.ones(num_elements),
            - example2: scipy.signal.hann(num_elements)], 
            by default np.ones(128)
        f0 : float, optional
            Base frequency of the transducer. Is used to compute the element spacing (lambda/2) [Hz], by default 100000
        freq : float, optional
            Frequency of the signal/acoustic pulse. Is used to compute the arrival phase delay [Hz], by default 80000
        idealized_beampattern_preserve_equivalent_beam_angle : bool, optional
            Preserve the equivalent beam angle when computing the idealized beampattern (outer parts will be a fraction between 0 and 1), 
            by default True
        transmit_steeringangle_degrees : int, optional
            Set a steeringangle for the transmit swath [°], by default 0
        progress : bool, optional
            Show progress for creating the beampattern, by default False
        """

        self.transmit_steeringangle_degrees = transmit_steeringangle_degrees

        self.beamsteeringangles_degrees = beamsteeringangles_degrees
        self.beamsteeringangles_radians = np.radians(beamsteeringangles_degrees)
        self.sampleranges = sampleranges

        self.n_beams = beamsteeringangles_degrees.shape[0]
        self.n_samples = sampleranges.shape[0]

        self.effective_pulse_length = effective_pulse_length

        self.window_rx = window
        self.window_tx = window

        self.f0 = f0
        self.freq = freq

        self.idealized_beampattern_preserve_equivalent_beam_angle = idealized_beampattern_preserve_equivalent_beam_angle

        # will be initialized in compute beampattern rx
        self.beampattern_beam_rx = None
        self.beampattern_idealized_beam_rx = None
        self.equivalent_beam_angles_beam_rx_degrees = None
        self.equivalent_beam_angles_beam_rx_radians = None
        self.beampattern_beam_rx = None
        self.center_beam_index_rx = None

        # will be initialized in compute beampattern tx
        self.beampattern_tx = None
        self.beampattern_idealized_tx = None
        self.equivalent_beam_angle_tx_degrees = None
        self.equivalent_beam_angle_tx_radians = None

        self.wci_sample_volume = None

        self.set_navigation(0,0,0,0,0,0)

        self.recalculate(progress = progress)


    def recalculate(self, no_beampattern: bool = False, progress: bool = False):
        """Recalculate the beampattern, sample volume and beamspacing

        Parameters
        ----------
        no_beampattern : bool, optional
            If set true, beampattern will not be recalculated , by default False
        progress : bool, optional
            Show progress, by default False
        """

        self.sample_distance = (self.sampleranges.max() - self.sampleranges.min()) / (len(self.sampleranges) - 1)

        if not no_beampattern:
            self.create_beampattern_rx(progress)
            self.create_beampattern_tx()

        self.wci_sample_volume = ef.get_sample_volume(self.beamsteeringangles_radians,
                                                      self.sampleranges,
                                                      self.equivalent_beam_angle_tx_radians,
                                                      self.equivalent_beam_angles_beam_rx_radians,
                                                      self.effective_pulse_length)

        self.beamspacing_degrees = []
        for bnr in range(len(self.beamsteeringangles_degrees)):
            if bnr == 0:
                self.beamspacing_degrees.append(self.beamsteeringangles_degrees[1] - self.beamsteeringangles_degrees[0])
            elif bnr == len(self.beamsteeringangles_degrees) - 1:
                self.beamspacing_degrees.append(self.beamsteeringangles_degrees[-1] - self.beamsteeringangles_degrees[-2])
            else:
                self.beamspacing_degrees.append(
                    (self.beamsteeringangles_degrees[bnr + 1] - self.beamsteeringangles_degrees[bnr - 1]) / 2)
        self.beamspacing_degrees = np.array(self.beamspacing_degrees)

        if (progress):
            print('recalculated MBES')


    def create_beampattern_rx(self, progress:bool = False):
        """Create the beampattern for the receiver

        Parameters
        ----------
        progress : bool, optional
            Show progress, by default False
        """

        window = self.window_rx

        if progress:
            iterator = tqdm(self.beamsteeringangles_degrees)
        else:
            iterator = self.beamsteeringangles_degrees

        self.beampattern_beam_rx = np.empty((self.n_beams,bf.ANGLE_RESOLUTION),dtype=np.float64)
        self.beampattern_idealized_beam_rx = np.empty((self.n_beams,bf.ANGLE_RESOLUTION),dtype=np.float64)
        self.equivalent_beam_angles_beam_rx_degrees = np.empty(self.n_beams,dtype=np.float64)
        self.equivalent_beam_angles_beam_rx_radians = np.empty(self.n_beams,dtype=np.float64)

        self.center_beam_index_rx = None

        angle_last = np.nan
        for bnr,beam_steering_angle in enumerate(iterator):
            if self.center_beam_index_rx is None:
                if abs(beam_steering_angle) > abs(angle_last):
                    self.center_beam_index_rx = bnr -1
                else:
                    angle_last = beam_steering_angle

            self.beampattern_beam_rx[bnr] = bf.generate_delay_and_sum_beampattern(beam_steering_angle,
                                                                                  window=window,
                                                                                  f0=self.f0,
                                                                                  freq=self.freq)

            self.equivalent_beam_angles_beam_rx_degrees[bnr] = bf.get_equivalent_beam_angle_from_beampattern(
                self.beampattern_beam_rx[bnr]
            )

            self.equivalent_beam_angles_beam_rx_radians[bnr] = self.equivalent_beam_angles_beam_rx_degrees[bnr] * hlp.M_PI_180

            self.beampattern_idealized_beam_rx[bnr] = bf.generate_idealized_beampattern(
                beam_steering_angle,
                window=window,
                f0=self.f0,
                freq=self.freq,
                equivalent_beam_angle_degrees=self.equivalent_beam_angles_beam_rx_degrees[bnr],
                preserve_equivalent_beam_angle=self.idealized_beampattern_preserve_equivalent_beam_angle
            )

    def create_beampattern_tx(self):
        """Create the beampattern for the transmitter
        """

        window = self.window_tx

        self.beampattern_tx = bf.generate_delay_and_sum_beampattern(0,
                                                                    window=window,
                                                                    f0=self.f0,
                                                                    freq=self.freq)

        self.equivalent_beam_angle_tx_degrees = bf.get_equivalent_beam_angle_from_beampattern(self.beampattern_tx)
        self.equivalent_beam_angle_tx_radians = self.equivalent_beam_angle_tx_degrees * hlp.M_PI_180

        self.beampattern_idealized_tx = bf.generate_idealized_beampattern(
            self.transmit_steeringangle_degrees,
            window=window,
            f0=self.f0,
            freq=self.freq,
            equivalent_beam_angle_degrees=self.equivalent_beam_angle_tx_degrees,
            preserve_equivalent_beam_angle=self.idealized_beampattern_preserve_equivalent_beam_angle)


    def plot_beampattern_rx(self,
                            fig: plt.Figure=None,
                            ax: plt.Axes = None,
                            plot_beampattern: bool = True,
                            plot_idealized_beampattern: bool = True,
                            close_plots: bool = False,
                            log: bool = True,
                            windows_names_pltbp_pltideal: (np.ndarray, str, bool, bool) = None,
                            marker='+'):
        """Plot the beampattern for the receiver

        Parameters
        ----------
        fig : plt.Figure, optional
            If set, create axes in this figure, by default None
        ax : plt.Axes, optional
            If set, plot into this axes, by default None
        plot_beampattern : bool, optional
            Plot the beampattern, by default True
        plot_idealized_beampattern : bool, optional
            Plot the idealized (rectangular) beampattern, by default True
        close_plots : bool, optional
            Close plots before creating figure, usefull for plotting in notebooks, by default False
        log : bool, optional
            Convert beampattern to logarithmic values, by default True
        windows_names_pltbp_pltideal : (np.ndarray, str, bool, bool), optional
            tuple with plot parameters: (1D window array, name, plot_beampattern, plot_ideal) by default None
        marker : str, optional
            Marker used for points in the beampattern plots, by default '+'

        Returns
        -------
        (plt.Figure, plt.Axes)
            Figure and axes used for plotting
        """

        add_legend = False
        if fig is None and ax is None:
            if close_plots: plt.close('beampattern rx')
            fig = plt.figure('beampattern rx')
            fig.clf()
        if ax is None:
            add_legend = True
            ax = fig.subplots()
            ax.set_title('beampattern rx')

        beampattern = []
        if windows_names_pltbp_pltideal is None:
            if plot_beampattern:
                beampattern.append([self.beampattern_beam_rx,"beampattern"])
            if plot_idealized_beampattern:
                beampattern.append([self.beampattern_idealized_beam_rx,"idealized"])

        else:
            old_window = self.window_rx
            for window, window_name, plt_bp, plt_ideal in tqdm(windows_names_pltbp_pltideal):
                self.window_rx = window
                self.recalculate(no_beampattern=False)

                if plt_bp:
                    beampattern.append([self.beampattern_beam_rx.copy(), "beampattern-" + window_name])
                if plt_ideal:
                    beampattern.append([self.beampattern_idealized_beam_rx.copy(), "idealized-" + window_name])

            self.window_rx = old_window
            self.recalculate(no_beampattern=False)



        for index in [self.center_beam_index_rx, -1]:
            for bp,bpname in beampattern:

                if log:
                    bp = hlp.to_db(bp[index])

                ax.plot(bf.BEAMPATTERN_ANGLES_DEGREES,
                        bp,
                        marker=marker,
                        label="rx {}: {}/{} °".format(bpname,index,round(self.beamsteeringangles_degrees[index],2)))

        if add_legend:
            ax.legend()

        return ax.figure,ax

    def plot_beampattern_tx(self,
                            fig: plt.Figure=None,
                            ax: plt.Axes = None,
                            plot_beampattern: bool = True,
                            plot_idealized_beampattern: bool = True,
                            close_plots: bool = False,
                            log: bool = True,
                            windows_names_pltbp_pltideal: (np.ndarray, str, bool, bool) = None,
                            marker='+'):
        """Plot the beampattern for the transmitter

        Parameters
        ----------
        fig : plt.Figure, optional
            If set, create axes in this figure, by default None
        ax : plt.Axes, optional
            If set, plot into this axes, by default None
        plot_beampattern : bool, optional
            Plot the beampattern, by default True
        plot_idealized_beampattern : bool, optional
            Plot the idealized (rectangular) beampattern, by default True
        close_plots : bool, optional
            Close plots before creating figure, usefull for plotting in notebooks, by default False
        log : bool, optional
            Convert beampattern to logarithmic values, by default True
        windows_names_pltbp_pltideal : (np.ndarray, str, bool, bool), optional
            tuple with plot parameters: (1D window array, name, plot_beampattern, plot_ideal) by default None
        marker : str, optional
            Marker used for points in the beampattern plots, by default '+'

        Returns
        -------
        (plt.Figure, plt.Axes)
            Figure and axes used for plotting
        """

        add_legend = False
        if fig is None and ax is None:
            if close_plots: plt.close('beampattern tx')
            fig = plt.figure('beampattern tx')
            fig.clf()
        if ax is None:
            add_legend = True
            ax = fig.subplots()
            ax.set_title('beampattern tx')

        beampattern = []
        if windows_names_pltbp_pltideal is None:
            if plot_beampattern:
                beampattern.append([self.beampattern_tx, "beampattern"])
            if plot_idealized_beampattern:
                beampattern.append([self.beampattern_idealized_tx,"idealized"])
        else:
            old_window = self.window_tx
            for window, window_name, plt_bp, plt_ideal in tqdm(windows_names_pltbp_pltideal):
                self.window_tx = window
                self.recalculate(no_beampattern=False)

                if plt_bp:
                    beampattern.append([self.beampattern_tx.copy(), "beampattern-" + window_name])
                if plt_ideal:
                    beampattern.append([self.beampattern_idealized_tx.copy(), "idealized-" + window_name])

            self.window_tx = old_window
            self.recalculate(no_beampattern=False)

        for bp,bpname in beampattern:
            if log:
                bp = hlp.to_db(bp)
            ax.plot(bf.BEAMPATTERN_ANGLES_DEGREES, bp, marker=marker, label = bpname)

        if add_legend:
            ax.legend()

        return ax.figure,ax

    def set_navigation(self,
                       x: float, y: float, z: float,
                       yaw: float, pitch: float, roll: float,
                       degrees: bool = True):
        """Set the position and alititude of the Transceiver. Use this function prior to calling the 'create_wci" and/or the 'raytrace_wci' function

        Parameters
        ----------
        x : float
            x position [m]
        y : float
            y position [m]
        z : float
            z position [m]
        yaw : float
            yaw angle [° or rad]
        pitch : float
            pitch angle [° or rad]
        roll : float
            roll angle [° or rad]
        degrees : bool, optional
            If true, angles are in ° otherwise in rad, by default True
        """
        self.pos_x = x
        self.pos_y = y
        self.pos_z = z
        if degrees:
            self.yaw_degrees   = yaw
            self.pitch_degrees = pitch
            self.roll_degrees  = roll
        else:
            self.yaw_degrees   = math.degrees(yaw)
            self.pitch_degrees = math.degrees(pitch)
            self.roll_degrees  = math.degrees(roll)

    def get_relative_xyz(self,targets_x: np.ndarray, targets_y: np.ndarray, targets_z: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
        """Get the positions (x,y,z) of the targets relative to the transceiver

        Note: this function is called by get_target_range_tx_rx

        Parameters
        ----------
        targets_x : np.ndarray
            Absolute x positions of the targets
        targets_y : np.ndarray
            Absolute y positions of the targets
        targets_z : np.ndarray
            Absolute z positions of the targets

        Returns
        -------
        (np.ndarray, np.ndarray, np.ndarray)
            tuple with relative x,y,z positions of the targets
        """

        dx = np.array(targets_x) - self.pos_x
        dy = np.array(targets_y) - self.pos_y
        dz = np.array(targets_z) - self.pos_z

        tx, ty, tz = tf.rotate_points(dx, dy, dz,
                                      self.yaw_degrees,
                                      self.pitch_degrees,
                                      self.roll_degrees,
                                      inverse=True)

        return tx, ty, tz

    def get_target_range_tx_rx(self, targets_x: np.ndarray, targets_y: np.ndarray, targets_z: np.ndarray, degrees: bool = True) -> (np.ndarray, np.ndarray, np.ndarray):
        """_summary_

        Parameters
        ----------
        targets_x : np.ndarray
            Absolute x positions of the targets [m]
        targets_y : np.ndarray
            Absolute y positions of the targets [m]
        targets_z : np.ndarray
            Absolute z positions of the targets [m]

        Returns
        -------
        (np.ndarray, np.ndarray, np.ndarray)
            Ranges, tx angles, rx angles of the targets
        """
        return tf.get_target_ranges_txangles_rxangles(*self.get_relative_xyz(targets_x, targets_y, targets_z), degrees = degrees)



    def plot_rel_pos(self, targets_x: np.ndarray, targets_y: np.ndarray, targets_z: np.ndarray,
                            fig: plt.Figure=None,
                            ax: plt.Axes = None,
                            close_plots: bool = False):
        """Plot the relative positions of the targets

        Parameters
        ----------
        targets_x : np.ndarray
            Absolute x positions of the targets [m]
        targets_y : np.ndarray
            Absolute y positions of the targets [m]
        targets_z : np.ndarray
            Absolute z positions of the targets [m]
        fig : plt.Figure, optional
            If set: use this figure to create the plotting axis, by default None
        ax : plt.Axes, optional
            If set: use this axis for plotting, by default None
        close_plots : bool, optional
            If set: close figures before creating a new one. This is usefull for plotting in notebooks, by default False
        """

        add_legend = False
        if fig is None and ax is None:
            if close_plots: plt.close('Targets')
            fig = plt.figure('Targets')
            fig.clf()
        if ax is None:
            add_legend = True
            ax = fig.subplots()
            ax.set_title('Targets')

        # rotate points to relative position
        targets_x = np.array(targets_x)
        targets_y = np.array(targets_y)
        targets_z = np.array(targets_z)
        tx,ty,tz = self.get_relative_xyz(targets_x, targets_y, targets_z)

        axes = fig.subplots(nrows=2,ncols=2)
        axit = axes.flat

        ax = next(axit)
        ax.clear()
        ax.set_title('targets y-z')
        ax.scatter(targets_y, -targets_z, label = 'original')
        ax.scatter(ty, -tz, label = 'transformed')
        ax.set_aspect('equal')
        ax.set_xlabel("y")
        ax.set_xlabel("y")
        ax.set_ylabel("z")
        ax.set_ylabel("z")
        ax.legend()

        #mbes = MBES(120, 1, 1)
        #mbes.plot_across_swathoverlap_at_linespacing(120, 0, axes=ax)

        ax = next(axit)
        ax.clear()
        ax.set_title('targets x-z')
        ax.scatter(targets_x, -targets_z, label = 'original')
        ax.scatter(tx, -tz, label = 'transformed')
        ax.set_aspect('equal')
        ax.set_xlabel("x")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_ylabel("z")
        ax.set_xlim(-220, 220)
        ax.legend()

        #mbes.plot_along(120, "blue", True, axes=ax)

        ax = next(axit)
        ax.clear()
        ax.set_title('targets x-y')
        ax.scatter(targets_y, targets_x, label = 'original')
        ax.scatter(ty, tx, label = 'transformed')
        ax.set_aspect('equal')
        ax.set_xlabel("y")
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        ax.set_ylabel("x")
        ax.set_ylim(-220, 220)
        ax.set_xlim(-220, 220)
        ax.legend()

    # Create echo functions
    def get_wci_extent(self):
        """Get the extent of the wci in the coordinate system of the sonar 
        (rx_angle_min, rx_angle_max, range_max, range_min)

        Returns
        -------
        np.ndarray
            [rx_angle_min, rx_angle_max, range_max, range_min]
        """

        samplespacing = self.sampleranges[1] - self.sampleranges[0]
        extent = [self.beamsteeringangles_degrees[0] - self.beamspacing_degrees[0] * 0.5,
                  self.beamsteeringangles_degrees[-1] + self.beamspacing_degrees[-1] * 0.5,
                  self.sampleranges[-1] + samplespacing * 0.5,
                  self.sampleranges[0] - samplespacing * 0.5]
        return extent

    def create_wci(self, targets_x: np.ndarray,
                   targets_y: np.ndarray,
                   targets_z: np.ndarray,
                   targets_val: np.ndarray,
                   return_ts : bool = False,
                   return_sv : bool = True,
                   idealized_beampattern: bool = False)-> np.ndarray:
        """Create a Water Column Image (wci) for the given targets

        Parameters
        ----------
        targets_x : np.ndarray
            Absolute x positions of the targets [m]
        targets_y : np.ndarray
            Absolute y positions of the targets [m]
        targets_z : np.ndarray
            Absolute z positions of the targets [m]
        targets_val : np.ndarray
            Backscattering values for the targets (linear scale)
        return_ts : bool, optional
            If true: return the target strength value, by default False
        return_sv : bool, optional
            If true: return the volume backscattering strength value , by default True
        idealized_beampattern : bool, optional
            Use idealized beampattern instead of the real ones, by default False

        Returns
        -------
        np.ndarray
            ts or sv values for the targets (linear scale)

        Raises
        ------
        RuntimeError
            Error if neither return_ts nor return_sv is set to true
        """

        if return_ts == return_sv == False:
            raise RuntimeError('Error[create_wci]: neither ts nor sv are selected as return value!')

        targets_x = np.array(targets_x)
        targets_y = np.array(targets_y)
        targets_z = np.array(targets_z)
        targets_val = np.array(targets_val)

        # pre filter targets
        def filter_index(x,y,z,v,idx):
            return x[idx],y[idx],z[idx],v[idx]

        #filter targets before computing exact ranges / angles (speed up)
        R = np.max(self.sampleranges)

        idx = np.argwhere(targets_x>(self.pos_x-R)).flatten()
        targets_x_,targets_y_,targets_z_,targets_val_ = filter_index(targets_x,targets_y,targets_z,targets_val,idx)

        idx = np.argwhere(targets_x_<(self.pos_x+R)).flatten()
        targets_x_,targets_y_,targets_z_,targets_val_ = filter_index(targets_x_,targets_y_,targets_z_,targets_val_,idx)

        idx = np.argwhere(targets_y_>(self.pos_y-R)).flatten()
        targets_x_,targets_y_,targets_z_,targets_val_ = filter_index(targets_x_,targets_y_,targets_z_,targets_val_,idx)

        idx = np.argwhere(targets_y_<(self.pos_y+R)).flatten()
        targets_x_,targets_y_,targets_z_,targets_val_ = filter_index(targets_x_,targets_y_,targets_z_,targets_val_,idx)

        print(R,self.pos_x,self.pos_y,len(targets_x_),len(targets_x),"                      ",end='\r')

        # compute position relative to the ship
        targets_range, targets_tx, targets_rx = self.get_target_range_tx_rx(targets_x_,
                                                                            targets_y_,
                                                                            targets_z_)

        if not idealized_beampattern:
            ts = ef.create_wci(targets_range, targets_tx, targets_rx, np.array(targets_val_),
                               self.beamsteeringangles_radians,
                               self.sampleranges,
                               self.beampattern_tx,
                               self.beampattern_beam_rx,
                               pf.get_hann_pulse_response, self.effective_pulse_length)
        else:
            ts = ef.create_wci(targets_range, targets_tx, targets_rx, np.array(targets_val),
                               self.beamsteeringangles_radians,
                               self.sampleranges,
                               self.beampattern_idealized_tx,
                               self.beampattern_idealized_beam_rx,
                               pf.get_rect_pulse_response, self.effective_pulse_length)

        if return_sv:
            sv = ts / self.wci_sample_volume

            if return_ts:
                return ts,sv
            return sv

        if return_ts:
            return ts

    def raytrace_wci(self) -> (np.ndarray,np.ndarray,np.ndarray):
        """Return the x,y,z coordinates of the wci in the absolute coordinate system.
        Note this function uses the current position and orientation of the sonar set by the 'set_navigation' function.

        Returns
        -------
        (np.ndarray,np.ndarray,np.ndarray)
            X,Y,Z coordinates of the WCI, dimensions of X,Y,Z are (len(self.beamsteeringangles_radians), len(self.sampleranges))
        """
        return ef.get_wci_xyz(self.beamsteeringangles_radians,
                              self.sampleranges,
                              math.radians(self.transmit_steeringangle_degrees),
                              self.pos_x, self.pos_y, self.pos_z,
                              math.radians(self.yaw_degrees),
                              math.radians(self.pitch_degrees),
                              math.radians(self.roll_degrees))




if __name__ == '__main__':
    # some plotting examples
    init(1800)

    mbes = Multibeam(
        window=signal.windows.exponential(128,tau=64),
        progress=True
    )

    fig,ax = mbes.plot_beampattern_rx()
    mbes.plot_beampattern_tx(ax = ax)
    ax.legend()

    tx = [0]
    ty = [0]
    tz = [60]
    tv = [1]

    mbes.set_navigation(x = 0,
                        y = 0,
                        z = 0,
                        yaw = 0,
                        pitch = 0,
                        roll = 45)

    mbes.plot_rel_pos(tx,ty,tz)

    tr,ttx,trx = mbes.get_target_range_tx_rx(tx,ty,tz)

    ts,sv = mbes.create_wci(tx,ty,tz,tv,return_ts=True,return_sv=True)

    fig = plt.figure('echo')
    fig.clear()
    fig.show()

    ts_db = hlp.to_db(ts)
    sv_db = hlp.to_db(sv)

    axes = fig.subplots(ncols=2)
    axit = axes.flat

    ax = next(axit)
    ax.set_title('ts')
    ax.imshow(ts.transpose(), extent=mbes.get_wci_extent())
    ax.scatter(trx, tr)
    windows_names_pltbp_pltideal = [
        (signal.windows.boxcar(128), 'boxcar', True, True),
        (signal.windows.exponential(128, tau=64), 'exponential', True, False),
        (signal.windows.hann(128), 'hann', True, False),
    ]

    fig, ax = mbes.plot_beampattern_rx(windows_names_pltbp_pltideal=windows_names_pltbp_pltideal,marker=None)
    fig, ax = mbes.plot_beampattern_tx(windows_names_pltbp_pltideal=windows_names_pltbp_pltideal)

    plt.show()