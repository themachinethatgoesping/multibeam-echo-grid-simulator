# SPDX-FileCopyrightText: 2022 GEOMAR Helmholtz Centre for Ocean Research Kiel
#
# SPDX-License-Identifier: MPL-2.0

"""
Two classes for processing virtual navigation for the MBES simulation
    - Survey: Class and static functions for defining virtual survey positions for the simulation
    - MotionData: Class for loading navigation data from a pandas file
"""

# ------------------- Imports -------------------
# Python
from numba import njit, prange
import numba.types as ntypes

import numpy as np
import numpy.typing as npt
import pandas as pd
import pickle
import random
import math
import time
from copy import deepcopy
import random

from matplotlib import pyplot as plt

# Project imports
import mbes_sim.functions.beampatternfunctions as bf
import mbes_sim.functions.pulsefunctions as pf
import mbes_sim.functions.transformfunctions as tf
import mbes_sim.functions.helperfunctions as hlp

# class type definition
from typing import TypeVar, Type, Tuple


# ------------------- Classes 1 SURVEY -------------------
class Survey(object):
    Survey = TypeVar("Survey", bound=object)
    """
    A class to define/contain virtual navigation data for the simulation. The required information are:
        - pingpositions_x
        - pingpositions_y
        - yaws_degree
        - pitchs_degree
        - rolls_degree
        - heaves
        - times

    These are each 1d arrays of same length. Each element describes a position,yaw,pitch,roll,heave,timepoint
    that will be used as a pingposition in the simulation that uses this object
    To initialize use either:
        - init function ( survey=Survey(parameters)): provide information directly
        - static functions (from_ping_distance, from_num_pings, from_fixed_pingrate, from_max_pingrate)
            - e.g Survey=from_ping_distance(parameters)
            - These generate a survey object from given parameters (e.g. survey time, start position, pingrate

    The survey object implements a list like interface:
        - len(survey) returns the number of pings
        - survey[10] returns the 11th element of the survey as: (time,pingpos_x,pingpos_y,pingpos_z,yaw,pitch,roll)

    Notes on the coordinate system
        - x: northing in meter
        - y: easting in meter
        - z: depth in meter (positive downwards)
        - heave: heave in meter (positive upwards)
        - yaw: heading in degrees (0° north 90° east)
        - pitch: pitch in degrees (0° straight 45° bow up)
        - roll: roll in degrees (0° straight 45° port-side up)
    """

    def __init__(
        self,
        pingpositions_x: npt.ArrayLike,
        pingpositions_y: npt.ArrayLike,
        yaws_degree: npt.ArrayLike,
        pitchs_degree: npt.ArrayLike,
        rolls_degree: npt.ArrayLike,
        heaves: npt.ArrayLike,
        times: npt.ArrayLike,
    ):
        """Survey object that defines ping positions and related navigation parameters for the MBES simulation.
        Each element is 1d arrays of same lengths. Each element describes on position,yaw,pitch,roll,heave,timepoint
        that will be used as a pingposition in the simulation that uses this object

        Parameters
        ----------
        pingpositions_x : npt.ArrayLike
            Pingpositions in m. X represents north.
        pingpositions_y : npt.ArrayLike
            Pingpositions in m. Y represents east.
        yaws_degree : npt.ArrayLike
            Yaw of each ping in degrees in m. 0° is north, 90° is east.
        pitchs_degree : npt.ArrayLike
            Pitch of each ping in degrees in m. 0 is straight, +45° is ship Bow up.
        rolls_degree : npt.ArrayLike
            Pitch of each ping in degrees in m. 0 is straight, +45° is port side up.
        heaves : npt.ArrayLike
            Heave of each ping in m. 1 is one meter up.  Z is positive downwards
        times : npt.ArrayLike
            Time of each ping in seconds. (e.g a unix time stamp)
        """

        self.pingpositions_x = np.array(pingpositions_x)
        self.pingpositions_y = np.array(pingpositions_y)
        self.yaws_degree = np.array(yaws_degree)
        self.pitchs_degree = np.array(pitchs_degree)
        self.rolls_degree = np.array(rolls_degree)
        self.heaves = np.array(heaves)
        self.times = np.array(times)

        self.RandomOffsetX = 0
        self.RandomOffsetY = 0
        self.RandomOffsetZ = 0

    def setRandomOffsetX(self, GridCellSize: float, fixedOffset: float = None):
        """Add a random static offset to the x-positions of the survey.
        Each position well be modified using the exact same offset value.
        This function is meant to avoid grid alignment in repeated simulations by shifting the entire survey
        by a random position relative to the grid cell-size.
        Note that the offset is always applied to the original data. Thus repeatedly calling this function will
        not lead to accumulated shifting.
        Parameters
        ----------
        GridCellSize : float
            The grid cell size. The survey is shifted within +- 0.5 * GridCellSize
        fixedOffset : float, optional
            If this value is set, GridCellSize is ignored. The survey is shifted by fixedOffset, by default None
        """
        self.pingpositions_x -= self.RandomOffsetX

        if fixedOffset is None:
            self.RandomOffsetX = random.uniform(-GridCellSize * 0.5, GridCellSize * 0.5)
        else:
            self.RandomOffsetX = fixedOffset

        self.pingpositions_x += self.RandomOffsetX

    def setRandomOffsetY(self, GridCellSize, fixedOffset=None):
        """Add a random static offset to the y-positions of the survey.
        Each position well be modified using the exact same offset value.
        This function is meant to avoid grid alignment in repeated simulations by shifting the entire survey
        by a random position relative to the grid cell-size.
        Note that the offset is always applied to the original data. Thus repeated calling of this function will
        not lead to accumulated shifting.

        Parameters
        ----------
        GridCellSize : float
            The grid cell size. The survey is shifted within +- 0.5 * GridCellSize
        fixedOffset : float, optional
            If this value is set, GridCellSize is ignored. The survey is shifted by fixedOffset, by default None
        """
        self.pingpositions_y -= self.RandomOffsetY

        if fixedOffset is None:
            self.RandomOffsetY = random.uniform(-GridCellSize * 0.5, GridCellSize * 0.5)
        else:
            self.RandomOffsetY = fixedOffset

        self.pingpositions_y += self.RandomOffsetY

    def setRandomOffsetZ(self, GridCellSize, fixedOffset=None):
        """Add a random static offset to the z-positions of the survey.
        Each position well be modified using the exact same offset value.
        This function is meant to avoid grid alignment in repeated simulations by shifting the entire survey
        by a random position relative to the grid cell-size.
        Note that the offset is always applied to the original data. Thus repeated calling of this function will
        not lead to accumulated shifting.

        Parameters
        ----------
        GridCellSize : float
            The grid cell size. The survey is shifted within +- 0.5 * GridCellSize
        fixedOffset : float, optional
            If this value is set, GridCellSize is ignored. The survey is shifted by fixedOffset, by default None
        """
        self.heaves += self.RandomOffsetZ

        if fixedOffset is None:
            self.RandomOffsetZ = random.uniform(-GridCellSize * 0.5, GridCellSize * 0.5)
        else:
            self.RandomOffsetZ = fixedOffset
        self.heaves -= self.RandomOffsetZ

    def __len__(self) -> int:
        """Return the number of pings in the survey,

        Returns
        -------
        int
            Number of pings in the survey
        """
        return len(self.pingpositions_x)

    def __getitem__(
        self, index: int
    ) -> (float, float, float, float, float, float, float):
        """Return the index+1 element of the survey.

        Notes on the coordinate system
        - x: northing in meter
        - y: easting in meter
        - z: depth in meter (positive downwards)
        - heave: heave in meter (positive upwards)
        - yaw: heading in degrees (0° north 90° east)
        - pitch: pitch in degrees (0° straight 45° bow up)
        - roll: roll in degrees (0° straight 45° port-side up)

        :param index: index of position in survey
        :return: time,x,y,z,yaw,pitch,roll

        Parameters
        ----------
        index : int
            index of position in survey

        Returns
        -------
        tuple(float, float, float, float, float, float, float)
            (time, x, y, z, yaw, pitch, roll)

        Raises
        ------
        IndexError
            _description_
        """
        if index < 0:
            index = len(self) - index

        if index >= len(self):
            raise IndexError(
                "Survey: out of bounds! 0 <= {} < {}".format(index, len(self) - 1)
            )

        return (
            self.times[index],
            self.pingpositions_x[index],
            self.pingpositions_y[index],
            -self.heaves[index],
            self.yaws_degree[index],
            self.pitchs_degree[index],
            self.rolls_degree[index],
        )

    def plot(self, fig: plt.Figure = None, close_plots: bool = False):
        """A function for quick plotting of the internal survey arrays

        Parameters
        ----------
        fig : plt.Figure, optional
            If provided: Add the plot to this figure., by default None
        close_plots : bool, optional
            If true, close figure named 'Survey' before creating a new one.
                            (Useful for repeated plotting in jupyter notebooks), by default False
        """

        if fig is None:
            if close_plots:
                plt.close("Survey")
            fig = plt.figure("Survey")
            fig.clf()

        axes = fig.subplots(ncols=3)
        axit = axes.flat

        ax = next(axit)
        ax.set_title("X/Y Positions")
        ax.set_xlabel("x positions [m]")
        ax.set_ylabel("y positions [m]")
        ax.plot(self.pingpositions_x, self.pingpositions_y, marker="x")

        ax = next(axit)
        ax.set_title("yaw/pitch/roll")
        ax.plot(self.times, self.yaws_degree, marker="x", label="yaw")
        ax.plot(self.times, self.pitchs_degree, marker="x", label="pitch")
        ax.plot(self.times, self.rolls_degree, marker="x", label="roll")
        ax.set_xlabel("time")
        ax.set_ylabel("angle [°]")
        ax.legend()

        ax = next(axit)
        ax.set_title("heave")
        ax.plot(self.times, self.heaves, marker="x", label="heave")
        ax.set_xlabel("time")
        ax.set_ylabel("heave [m]")

    @staticmethod
    def from_ping_distance(
        min_x: float = -20,
        max_x: float = 20,
        ping_spacing: float = 0.9,
        speed_knots: float = 3,
    ) -> Survey:
        """Create a survey object from the mentioned parameters.
        The survey will start at min_x and move to max_x and ping at a regular interval (ping spacing)

        Parameters
        ----------
        min_x : float, optional
            smallest x position (start position) [meter], by default -20
        max_x : float, optional
            largest x position (end position) [meter], by default 20
        ping_spacing : float, optional
            spacing between consecutive pings [meter], by default 0.9
        speed_knots : float, optional
            speed of the vessel in knots [knots], by default 3

        Returns
        -------
        Survey
            Initialized survey object
        """

        # pre compute
        speed = speed_knots * 0.514
        dist = abs(max_x - min_x)
        max_time = dist / speed

        # create vectors
        pingpositions_x = np.arange(
            min_x, max_x + ping_spacing * 0.001, ping_spacing, dtype=float
        )
        num_pings = len(pingpositions_x)

        # initialize arrays
        pingpositions_y = np.zeros(num_pings)
        yaws = np.zeros(num_pings)
        pitchs = np.zeros(num_pings)
        rolls = np.zeros(num_pings)
        heaves = np.zeros(num_pings)
        times = np.linspace(0, max_time, num_pings)

        # create survey object
        return Survey(
            pingpositions_x, pingpositions_y, yaws, pitchs, rolls, heaves, times
        )

    @staticmethod
    def from_num_pings(
        min_x: float = -20,
        max_x: float = 20,
        num_pings: int = 42,
        speed_knots: float = 3,
    ) -> Survey:
        """Create a survey object from the mentioned parameters.
        The survey will start at min_x and move to max_x and ping at a regular interval such that num_pings are
        created over the entire survey.

        Parameters
        ----------
        min_x : float, optional
            smallest x position (start position) [meter], by default -20
        max_x : float, optional
            largest x position (end position) [meter], by default 20
        num_pings : int, optional
            number of pings to create (for the whole survey), by default 42
        speed_knots : float, optional
            speed of the vessel in knots [knots], by default 3

        Returns
        -------
        Survey
            initialized survey object
        """
        swath_distance = hlp.compute_ping_spacing_from_num_pings(num_pings, min_x, max_x)
        return Survey.from_ping_distance(min_x, max_x, swath_distance, speed_knots)

    @staticmethod
    def from_fixed_pingrate(
        min_x: float = -20,
        max_x: float = 20,
        pingrate: int = 1.0,
        speed_knots: float = 3,
    ) -> Survey:
        """Create a survey object from the mentioned parameters.
        The survey will start at min_x and move to max_x and ping at a regular interval. (pingrate)

        Parameters
        ----------
        min_x : float, optional
            smallest x position (start position) [meter], by default -20
        max_x : float, optional
            largest x position (end position) [meter], by default 20
        pingrate : int, optional
            pings per second [1/second], by default 1.0
        speed_knots : float, optional
            speed of the vessel in knots [knots], by default 3

        Returns
        -------
        Survey
            initialized survey object
        """

        ping_distance = hlp.compute_ping_spacing_from_pingrate(pingrate, speed_knots)

        return Survey.from_ping_distance(min_x, max_x, ping_distance, speed_knots)

    @staticmethod
    def from_max_pingrate(
        min_x: float = -20,
        max_x: float = 20,
        depth: float = 120,
        mbes_rx_swath_angle_degrees: float = 120,
        speed_knots: float = 3,
    ) -> Survey:
        """Create a survey object from the mentioned parameters.
        The survey will start at min_x and move to max_x and ping at a regular interval. (max_pingrate)
        The max_pingrate is computed using a soundspeed of 1500 m/s and the two-way range of the outer mbes beams.
        (the time a ping needs to return from the seafloor at beam-steering angle mbes_rx_swath_angle_degrees/2)
        This optimistic as in reality MBES typically wait a bit longer before emitting the next bing.

        Parameters
        ----------
        min_x : float, optional
            smallest x position (start position) [meter], by default -20
        max_x : float, optional
            largest x position (end position) [meter], by default 20
        depth : float, optional
            depth below transducer [m], by default 120
        mbes_rx_swath_angle_degrees : float, optional
            swath opening angle of the MBES [°], by default 120
        speed_knots : float, optional
            speed of the vessel in knots [knots], by default 3

        Returns
        -------
        Survey
            initialized survey object
        """

        pingrate = hlp.compute_max_pingrate(depth, mbes_rx_swath_angle_degrees)

        return Survey.from_fixed_pingrate(min_x, max_x, pingrate, speed_knots)


# ------------------- Classes 2 MotionData -------------------
class MotionData(object):
    """
    A class to combine a Survey object with real motion data.
    The idea is to define a synthetic ideal survey with perfect ping distance but then modify this survey using real
    motion data saved within a pandas object.
    The output will be the synthetic survey + the real motion

    WARNING: the code was only tested with the motion data file than the provided test_data/m143_l0154_motion.csv
    We will possibly find bugs when using other motion data files

    Note:
        - yaw,pitch,roll are interpolated individually. This would not be 100% correct for precise geo-referencing.
            However, it does not influence the validity of the uncertainty estimation
        - The time frame (max - min) of the navigation data must be larger than the timeframe of the simulation survey

    Use: modified_survey = MotionData.get_modified_survey(survey, parameters ...)

    Notes on the dataformat:
        - Create a tab limited csv file with one header line and the following columns:
            - time: unixtime (seconds)
            - northing (meters): note: this does not have to be a valid utm coordinate, it can be any coordinate system
            - easting (meters), note: it does not have to be a valid utm coordinate, it can be any coordinate system
            - yaw: heading (°), 0 means north, 90° means east
            - pitch: pitch (°), +45° means bow up
            - roll: pitch (°), +45° means bow up
            - heave: heave (meter), +1 means one meter up

    """

    def __init__(self, motion_data_path: str):
        """Initialize a Navigation data object with the given motion data frame
        Parameters
        ----------
        motion_data_path : path to a tab limited csv file with one header line and the following columns:
           - time: unixtime (seconds)
           - northing (meters): note: this does not have to be a valid utm coordinate, it can be any coordinate system
           - easting (meters), note: it does not have to be a valid utm coordinate, it can be any coordinate system
           - yaw: heading (°), 0 means north, 90° means east
           - pitch: pitch (°), +45° means bow up
           - roll: pitch (°), +45° means bow up
           - heave: heave (meter), +1 means one meter up
        """

        # read csv files
        data = pd.read_csv(motion_data_path, sep="\t")

        # compute course
        course = hlp.compute_course(data["northing"].values, data["easting"].values)
        # use mean course to rotate the data
        X, Y = tf.rotate_points_2D(
            data["northing"].values,
            data["easting"].values,
            np.mean(course),
            degrees=True,
            inverse=False,
        )

        # create motion data frame
        self.motion_data = pd.DataFrame()
        self.motion_data["time"] = data["time"].values - np.nanmin(data["time"].values)
        self.motion_data["X"] = X - np.nanmin(X)
        self.motion_data["Y"] = Y - np.nanmean(Y)
        self.motion_data["heave"] = data["heave"].values - np.nanmean(
            data["heave"].values
        )
        self.motion_data["yaw"] = data["yaw"].values - np.nanmean(data["yaw"].values)
        self.motion_data["pitch"] = data["pitch"].values - np.nanmean(
            data["pitch"].values
        )
        self.motion_data["roll"] = data["roll"].values - np.nanmean(data["roll"].values)
        self.motion_data["heave"] = data["heave"].values - np.nanmean(
            data["heave"].values)
        self.motion_data["interpolated_point"] = False # the points added here are marked as not interpolated (they are real data points)
        

        # self.motion_data = data

    def get_modified_survey(
        self,
        survey: Survey,
        start_time: float = None,
        random_seed: int = None,
        exaggerate_yaw: float = 1,
        exaggerate_pitch: float = 1,
        exaggerate_roll: float = 1,
        exaggerate_heave: float = 1,
    ) -> Survey:
        """Modify a given Survey object with the given motion. If start_time and random_seed are not set, the start_time
        within the given motion dataset will be random. The motion is therefor random.

        Parameters
        ----------
        survey : Survey
            Survey object to be modified
        start_time : float, optional
            first time point in the navigation data that is to be used, by default None
        random_seed : int, optional
            provide seed for randomization (for testing only! do not use in uncertainty simulation!), by default None
        exaggerate_yaw : float, optional
            exaggerate the yaw value by this value. (1 means no modification), by default 1
        exaggerate_pitch : float, optional
            exaggerate the pitch value by this value. (1 means no modification), by default 1
        exaggerate_roll : float, optional
            exaggerate the roll value by this value. (1 means no modification), by default 1
        exaggerate_heave : float, optional
            exaggerate the heave value by this value. (1 means no modification), by default 1

        Returns
        -------
        Survey
            Survey modified by a random section of the given motion data
        """

        random.seed(random_seed)

        survey_intern = deepcopy(survey)

        # create random start time within the given navigation data for this survey
        # (if a fixed start time is not provided)
        if start_time is None:
            start_time = random.uniform(
                0, max(self.motion_data["time"]) - survey_intern.times[-1] - 1
            )

        # interpolate navigation data onto the ping times of the survey object
        data_points = self.get_interpolated_dataframe(survey_intern.times + start_time)

        # modify ping positions
        survey_intern.pingpositions_x += data_points["delta_forward"].to_numpy()
        survey_intern.pingpositions_y += data_points["delta_sideward"].to_numpy()

        # modify angles
        survey_intern.yaws_degree += data_points["yaw"].to_numpy() * exaggerate_yaw
        survey_intern.pitchs_degree += data_points["pitch"].to_numpy() * exaggerate_pitch
        survey_intern.rolls_degree += data_points["roll"].to_numpy() * exaggerate_roll
        survey_intern.heaves += data_points["heave"].to_numpy() * exaggerate_heave

        # modify survey times
        survey_intern.times += start_time

        return survey_intern

    def get_interpolated_dataframe(self, time_points: npt.ArrayLike) -> pd.DataFrame:
        """This functions provides interpolated motion values extracted from navigation data
        Each column is interpolated individually to the ping times indicated in time_points

        Parameters
        ----------
        time_points : npt.ArrayLike
            Time points for which the motion data is to be interpolated

        Returns
        -------
        pd.DataFrame
            a pandas dataframe with the following columns:
            - time: unixtime (seconds)
            - heave: heave (meter), +1 means one meter up
            - yaw: heading (°), 0 means north, 90° means east
            - pitch: pitch (°), +45° means bow up
            - roll: pitch (°), +45° means bow up
            
        """

        # initialize new data frame
        data_points = pd.DataFrame(time_points, columns=["time"])
        data_points["heave"] = np.nan
        data_points["yaw"] = np.nan
        data_points["pitch"] = np.nan
        data_points["roll"] = np.nan
        data_points["interpolated_point"] = True # the points added here are marked as interpolated 

        # concatinate data and interpolate (pandas default interpolation is linear)
        data_points = pd.concat(
            (data_points, self.motion_data), ignore_index=True
        ).sort_values("time", ignore_index=True)
        data_points.index = data_points["time"]
        data_points = data_points.interpolate(method="index")

        # get the interpolated datapoints (remove the original ones)
        data_points = data_points[data_points["interpolated_point"] == True]
        data_points.index = np.arange(0, len(data_points))

        # compute delta forward and delta sideward (see notes in the documentation above)
        x1 = np.min(data_points["X"])
        x2 = np.max(data_points["X"])
        y1 = np.min(data_points["Y"])
        y2 = np.max(data_points["Y"])

        angle = math.atan2((y2 - y1), (x2 - x1))
        r = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        ideal_forward = np.linspace(0, r, len(data_points["X"]))
        actual_forward = [
            math.sqrt((y - y1) ** 2 + (x - x1) ** 2)
            for x, y in zip(data_points["X"], data_points["Y"])
        ]
        actual_sideward = [0]

        actual_angle = [
            math.atan2((y - y1), (x - x1))
            for x, y in zip(data_points["X"][1:], data_points["Y"][1:])
        ]

        actual_sideward.extend(
            [
                af * math.sin(aa - angle)
                for af, aa in zip(actual_forward[1:], actual_angle)
            ]
        )

        data_points["delta_forward"] = actual_forward - ideal_forward
        data_points["delta_sideward"] = actual_sideward

        data_points["delta_forward"] -= data_points["delta_forward"].mean()
        data_points["delta_sideward"] -= data_points["delta_sideward"].mean()

        return data_points

    def plot(
        self,
        survey: Survey | None = None,
        fig: plt.Figure | None = None,
        close_plots: bool = False,
        show_xy_plot: bool = True,
        show_ypr_plot: bool = True,
        show_heave_plot: bool = True,
        marker: str = None,
    ):
        """A function for quickly plotting the effects of modifying a given survey object.

        Parameters
        ----------
        survey : Survey | None, optional
            Survey object to be modified, by default None
        fig : plt.Figure | None, optional
            If provided: Add the plot to this figure, by default None
        close_plots : bool, optional
            If true, close figure named 'Survey' before creating a new one.
                            (Useful for repeated plotting in jupyter notebooks), by default False
        show_xy_plot : bool, optional
            activate the xy data point plot, by default True
        show_ypr_plot : bool, optional
            activate the yaw pitch roll plot, by default True
        show_heave_plot : bool, optional
            activate the heave plot, by default True
        marker : str, optional
            specify a matplotlib marker for the plot lines (default is +), by default None
        """

        if fig is None:
            if close_plots:
                plt.close("Navigation Data")
            fig = plt.figure("Navigation Data")
            fig.clf()

        sdata = None
        if survey:
            sdata = self.get_interpolated_dataframe(survey.times)

        axes = fig.subplots(ncols=show_xy_plot + show_ypr_plot + show_heave_plot)
        if show_xy_plot + show_ypr_plot + show_heave_plot == 1:
            axes = np.array([axes])
        axit = axes.flat

        if show_xy_plot:
            ax = next(axit)
            ax.set_title("X/Y Positions")
            ax.set_xlabel("x positions [m]")
            ax.set_ylabel("y positions [m]")
            if not survey:
                ax.plot(self.motion_data["X"], self.motion_data["Y"])
            else:
                ax.plot(self.motion_data["X"], self.motion_data["Y"], c="grey")
                ax.plot(sdata["X"], sdata["Y"], marker=marker)

        if show_ypr_plot:
            ax = next(axit)
            ax.set_title("yaw/pitch/roll")
            if not survey:
                ax.plot(self.motion_data["time"], self.motion_data["yaw"], label="yaw")
                ax.plot(
                    self.motion_data["time"], self.motion_data["pitch"], label="pitch"
                )
                ax.plot(self.motion_data["time"], self.motion_data["roll"], label="roll")
            else:
                ax.plot(
                    self.motion_data["time"],
                    self.motion_data["yaw"],
                    label="yaw",
                    c="grey",
                )
                ax.plot(
                    self.motion_data["time"],
                    self.motion_data["pitch"],
                    label="pitch",
                    c="grey",
                )
                ax.plot(
                    self.motion_data["time"],
                    self.motion_data["roll"],
                    label="roll",
                    c="grey",
                )
                ax.plot(sdata["time"], sdata["yaw"], label="yaw", marker=marker)
                ax.plot(sdata["time"], sdata["pitch"], label="pitch", marker=marker)
                ax.plot(sdata["time"], sdata["roll"], label="roll", marker=marker)
            ax.set_xlabel("time")
            ax.set_ylabel("angle [°]")
            ax.legend()

        if show_heave_plot:
            ax = next(axit)
            ax.set_title("heave")
            if survey is None:
                ax.plot(
                    self.motion_data["time"], self.motion_data["heave"], label="heave"
                )
            else:
                ax.plot(
                    self.motion_data["time"],
                    self.motion_data["heave"],
                    label="heave",
                    c="grey",
                )
                ax.plot(sdata["time"], sdata["heave"], label="heave", marker=marker)

            ax.set_xlabel("time")
            ax.set_ylabel("heave [m]")


# ------------------- Main (for testing) -------------------
if __name__ == "__main__":
    """
    This is a simple test that plots a synthetic survey and the modifications on this survey using the navigation
    data object provided in the repository.
    """

    import matplotlib as mpl

    try:
        navdata = MotionData(motion_data_path="../../../test_data/m143_l0154_motion.csv")
    except:
        navdata = MotionData(motion_data_path="../test_data/m143_l0154_motion.csv")

    print(navdata.motion_data)

    survey = Survey.from_num_pings(-20, 20, 42)
    survey.plot()

    survey_nav = navdata.get_modified_survey(survey)
    survey_nav.plot()

    # test if the program crashes in repeated calls
    from tqdm import tqdm
    for _ in tqdm(range(100)):
        survey = Survey.from_num_pings(-20, 20, 42)
        survey_nav = navdata.get_modified_survey(survey)

    navdata.plot(survey_nav, marker="+")

    def test(x, y, z, ya, p, r):
        print(x, y, z, ya, p, r)

    test(*survey_nav[1][1:])

    plt.ioff()
    plt.show()
